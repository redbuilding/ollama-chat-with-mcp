import asyncio
import json
import re
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel, Field, constr
from bson import ObjectId
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
import ollama # Ensure ollama is imported
import threading
import time
import queue

# --- Environment Setup ---
MCP_SERVER_SCRIPT = "server_search.py"
# Determine the absolute path to MCP_SERVER_SCRIPT relative to this file's directory
# This makes it more robust regardless of where main.py is executed from.
_main_py_dir = os.path.dirname(os.path.abspath(__file__))
MCP_SERVER_SCRIPT_ABS_PATH = os.path.join(_main_py_dir, MCP_SERVER_SCRIPT)

if not os.path.exists(MCP_SERVER_SCRIPT_ABS_PATH):
    logging.error(f"MCP server script '{MCP_SERVER_SCRIPT_ABS_PATH}' not found. Searched in directory: '{_main_py_dir}'.")
    # Potentially exit or disable MCP features if critical
    # For now, it will likely fail in mcp_service_loop and log there.
else:
    logging.info(f"MCP server script found at: {MCP_SERVER_SCRIPT_ABS_PATH}")


os.makedirs('logs', exist_ok=True)
# BasicConfig sets the root logger. We'll set the specific logger level later.
logging.basicConfig(
    level=logging.DEBUG, # Set basicConfig to DEBUG to allow specific loggers to use it.
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/mcp_backend_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mcp_backend")
logger.setLevel(logging.DEBUG) # Set our specific logger to DEBUG level

# --- MongoDB Setup ---
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DATABASE_NAME = os.getenv('MONGODB_DATABASE_NAME', 'mcp_chat_db')
MONGODB_COLLECTION_NAME = os.getenv('MONGODB_COLLECTION_NAME', 'conversations')
DEFAULT_OLLAMA_MODEL = os.getenv('DEFAULT_OLLAMA_MODEL', 'qwen2:7b') # Default model

try:
    mongo_client = MongoClient(MONGODB_URI)
    mongo_client.admin.command('ping') # Verify connection
    db = mongo_client[MONGODB_DATABASE_NAME]
    conversations_collection = db[MONGODB_COLLECTION_NAME]
    logger.info(f"Successfully connected to MongoDB: {MONGODB_URI}")
except ConnectionFailure:
    logger.error(f"Failed to connect to MongoDB at {MONGODB_URI}. Please ensure MongoDB is running and accessible.")
    mongo_client = None
    db = None
    conversations_collection = None
except Exception as e:
    logger.error(f"An error occurred during MongoDB setup: {e}")
    mongo_client = None
    db = None
    conversations_collection = None


# --- Communication Queues ---
request_queue = queue.Queue()
response_queue = queue.Queue()

# --- State Management (Primarily for MCP service status) ---
class AppState:
    def __init__(self):
        self.current_date = datetime.now(timezone.utc) # Use timezone-aware datetime
        self.service_ready = False
        # llm_conversation_history is now managed per-request, loaded from/saved to DB
        # ollama_model is also managed per-conversation or via payload

    @property
    def formatted_date(self):
        return f"{self.current_date.strftime('%B')} {self.current_date.strftime('%d').lstrip('0')}, {self.current_date.strftime('%Y')}"

    @property
    def month_year(self):
        return f"{self.current_date.strftime('%B')} {self.current_date.strftime('%Y')}"

    @property
    def year(self):
        return self.current_date.strftime('%Y')

app_state = AppState()

# --- MCP Service (runs in a separate thread) ---
def run_mcp_service():
    asyncio.run(mcp_service_loop())

async def mcp_service_loop():
    logger.info("Starting MCP service loop")
    if not os.path.exists(MCP_SERVER_SCRIPT_ABS_PATH):
        logger.error(f"MCP service cannot start: script '{MCP_SERVER_SCRIPT_ABS_PATH}' not found.")
        app_state.service_ready = False
        return

    try:
        # Pass the directory of server_search.py as CWD for the subprocess
        # This helps server_search.py find its .env file reliably if it's in the same directory
        server_script_dir = os.path.dirname(MCP_SERVER_SCRIPT_ABS_PATH)
        
        server_params = StdioServerParameters(
            command=sys.executable, 
            args=[MCP_SERVER_SCRIPT_ABS_PATH], 
            env=None, # Inherits current environment
            # cwd=server_script_dir # StdioServerParameters doesn't have cwd, set via create_subprocess_exec
        )
        logger.info(f"Attempting to start MCP server with: {sys.executable} {MCP_SERVER_SCRIPT_ABS_PATH} in CWD: {server_script_dir}")
        
        # Customizing subprocess creation to set CWD
        # Based on mcp.client.stdio.stdio_client internal structure
        proc = await asyncio.create_subprocess_exec(
            server_params.command,
            *server_params.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, # Capture stderr for better debugging
            env=server_params.env,
            cwd=server_script_dir # Explicitly set CWD for the subprocess
        )

        # Log stderr from MCP server for debugging
        async def log_stderr():
            while proc.stderr and not proc.stderr.at_eof():
                line = await proc.stderr.readline()
                if line:
                    logger.info(f"[MCP Server STDERR] {line.decode().strip()}")
                else:
                    break
            logger.info("MCP Server STDERR stream ended.")

        stderr_task = asyncio.create_task(log_stderr())

        # Proceed with stdio_client logic, but using the manually created process
        # This is a bit of a workaround as stdio_client itself doesn't expose CWD for the subprocess easily
        # A cleaner way would be if StdioServerParameters supported CWD.
        # For now, we adapt. The key is ClientSession needs read/write streams.
        
        # The original stdio_client context manager handles proc.wait() and cleanup.
        # We need to replicate parts or use it carefully.
        # Let's try to use the existing stdio_client but ensure it uses our proc if possible,
        # or accept that CWD setting needs a custom loop.
        # Given StdioServerParameters doesn't support CWD, we'll stick to its default behavior
        # and rely on server_search.py's load_dotenv searching upwards.
        # The print() fix in server_search.py is the primary one.
        # The MCP_SERVER_SCRIPT_ABS_PATH ensures we run the correct script.

        # Reverting to standard stdio_client call, the absolute path and print fix are main.
        server_params_for_client = StdioServerParameters(
            command=sys.executable,
            args=[MCP_SERVER_SCRIPT_ABS_PATH], # Use absolute path
            env=None 
        )

        async with stdio_client(server_params_for_client, logger=logger.getChild("mcp_client")) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                logger.info("MCP session initialized. Listing tools...")
                tools_response = await session.list_tools()
                
                if tools_response and tools_response.tools is not None: # Check tools is not None
                    tool_names = [tool.name for tool in tools_response.tools]
                    logger.info(f"Available tools from MCP server: {tool_names}")
                    if not any(tool.name == "web_search" for tool in tools_response.tools):
                        logger.error(f"Error: Required 'web_search' tool not found among available tools: {tool_names}.")
                        app_state.service_ready = False
                        # stderr_task.cancel() # If we had the custom proc
                        return
                else:
                    logger.error(f"Error: No tools found or invalid/empty response from session.list_tools(). Response: {tools_response}")
                    app_state.service_ready = False
                    # stderr_task.cancel() # If we had the custom proc
                    return
                
                app_state.service_ready = True
                logger.info("MCP service initialized and ready with 'web_search' tool.")
                
                while True: # Main operational loop
                    try:
                        if not request_queue.empty():
                            request_data = request_queue.get_nowait()
                            if request_data["type"] == "search":
                                query = request_data["query"]
                                request_id = request_data["id"]
                                try:
                                    logger.debug(f"Calling 'web_search' tool with query: {query}")
                                    result = await session.call_tool("web_search", {"query": query})
                                    logger.debug(f"'web_search' tool call successful. Result content type: {type(result.content)}")
                                    response_queue.put({"id": request_id, "type": "search_result", "status": "success", "data": result.content})
                                except Exception as e:
                                    logger.error(f"Error in 'web_search' tool call: {e}", exc_info=True)
                                    response_queue.put({"id": request_id, "type": "search_result", "status": "error", "error": str(e)})
                        # Check if subprocess is still running
                        if proc.returncode is not None: # proc might not be defined here if using original stdio_client
                             logger.warning(f"MCP server subprocess exited with code {proc.returncode}. Stopping MCP service loop.")
                             break # Exit while loop
                    except queue.Empty:
                        pass
                    except Exception as e: # Catch errors within the while loop
                        logger.error(f"Exception in MCP service operational loop: {e}", exc_info=True)
                        # Decide if to break or continue
                    await asyncio.sleep(0.1)
    except ConnectionRefusedError as e:
        logger.error(f"MCP service connection refused. Is the server script ({MCP_SERVER_SCRIPT_ABS_PATH}) runnable and not crashing immediately? Error: {e}", exc_info=True)
    except asyncio.TimeoutError as e:
        logger.error(f"Timeout in MCP service communication (e.g., during initialize or list_tools): {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Generic error in MCP service setup or during connection: {e}", exc_info=True)
    finally:
        app_state.service_ready = False
        logger.info("MCP service stopped or failed to start.")
        # if 'stderr_task' in locals() and not stderr_task.done():
        #    stderr_task.cancel()
        #    try:
        #        await stderr_task
        #    except asyncio.CancelledError:
        #        logger.info("MCP Server STDERR logging task cancelled.")


# --- Utility Functions for MCP Interaction ---
def submit_search_request(query: str) -> str:
    request_id = f"req_{time.time()}"
    request_queue.put({"id": request_id, "type": "search", "query": query})
    return request_id

def wait_for_response(request_id: str, timeout: int = 30) -> Dict:
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            items_in_queue, found_response = [], None
            while not response_queue.empty():
                item = response_queue.get_nowait()
                if item.get("id") == request_id: found_response = item; break
                else: items_in_queue.append(item)
            for item in items_in_queue: response_queue.put(item)
            if found_response: return found_response
        except queue.Empty: pass
        time.sleep(0.5)
    return {"id": request_id, "type": "search_result", "status": "error", "error": "Request timed out"}

# --- Ollama Interaction ---
async def chat_with_ollama(messages: List[Dict[str, str]], model_name: str) -> Optional[str]:
    try:
        logger.info(f"[Ollama] Sending prompt to model '{model_name}'. History length: {len(messages)}")
        if messages: logger.debug(f"[Ollama] Last message: {messages[-1]['content'][:200]}")
        response = await asyncio.to_thread(ollama.chat, model=model_name, messages=messages)
        if response and "message" in response and "content" in response["message"]:
            return response["message"]["content"]
        logger.warning(f"[Ollama] Unexpected response from model '{model_name}': {response}")
        return None
    except Exception as e:
        logger.error(f"[Ollama] Error with model '{model_name}': {e}", exc_info=True)
        return None

# --- Search Result Processing ---
def extract_search_results(response_content):
    # MCP TextContent has a 'text' attribute.
    # The web_search tool in server_search.py returns a dict.
    # If result.content is already a dict, use it directly.
    # If it's TextContent containing JSON string, parse it.
    
    if isinstance(response_content, dict):
        logger.debug("extract_search_results: received dict, using directly.")
        return response_content
    elif hasattr(response_content, 'text') and isinstance(response_content.text, str):
        logger.debug("extract_search_results: received TextContent, attempting to parse JSON from its 'text' attribute.")
        try:
            return json.loads(response_content.text)
        except json.JSONDecodeError as e:
            logger.error(f"extract_search_results: JSONDecodeError parsing TextContent: {e}. Content: {response_content.text[:500]}")
            return {"status": "error", "message": "Failed to parse search result JSON from TextContent."}
    elif isinstance(response_content, str): # Fallback if it's just a string (should ideally be JSON string)
        logger.debug("extract_search_results: received plain string, attempting to parse as JSON.")
        try:
            return json.loads(response_content)
        except json.JSONDecodeError as e:
            logger.error(f"extract_search_results: JSONDecodeError parsing plain string: {e}. Content: {response_content[:500]}")
            return {"status": "error", "message": "Failed to parse search result JSON from string."}
    
    logger.warning(f"extract_search_results: Unhandled response_content type: {type(response_content)}. Content: {str(response_content)[:500]}")
    return {"status": "error", "message": f"Unhandled search result type: {type(response_content)}"}


def format_search_results_for_prompt(results_data, query, max_results=3):
    # results_data is now expected to be the dictionary returned by web_search or parsed by extract_search_results
    if not isinstance(results_data, dict) or results_data.get("status") == "error":
        error_message = results_data.get("message", "Search returned no recognizable results or an error.")
        logger.warning(f"format_search_results_for_prompt: Invalid or error in results_data for query '{query}'. Message: {error_message}")
        return f"Search for '{query}': {error_message}"

    organic_results = results_data.get('organic_results', []) # server_search.py returns 'organic_results'

    if organic_results:
        formatted = "\n".join(
            f"{i+1}. {item.get('title', 'N/A')}\n   {item.get('snippet', 'N/A')}\n   Source: {item.get('link', 'N/A')}"
            for i, item in enumerate(organic_results[:max_results])
        )
        return f"Web search results for '{query}':\n{formatted}"
    
    logger.info(f"Search for '{query}' returned no organic_results. Full results_data: {results_data}")
    return f"Search for '{query}' returned no specific organic results. (May include other categories like top stories or Q&A if available in raw data)."


# --- FastAPI Application ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], # Adjust if your frontend runs elsewhere
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- Pydantic Models for API ---
class ChatMessage(BaseModel):
    role: str
    content: str
    is_html: Optional[bool] = False
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ChatPayload(BaseModel):
    user_message: str
    chat_history: List[ChatMessage] 
    use_search: bool
    conversation_id: Optional[str] = None
    ollama_model_name: Optional[str] = None # New field for model selection

class ChatResponse(BaseModel):
    conversation_id: str
    chat_history: List[ChatMessage] # Full updated UI history for this turn
    ollama_model_name: Optional[str] = None # Return the model used for this conversation turn

class ConversationListItem(BaseModel):
    id: str = Field(alias="_id")
    title: Optional[str] = "New Chat"
    created_at: datetime
    updated_at: datetime
    message_count: int
    ollama_model_name: Optional[str] = None # Model used for this conversation

    class Config:
        populate_by_name = True 
        json_encoders = {ObjectId: str, datetime: lambda dt: dt.isoformat()}

class RenamePayload(BaseModel):
    new_title: constr(strip_whitespace=True, min_length=1, max_length=100)


async def get_default_ollama_model() -> str:
    try:
        logger.debug("Attempting to fetch Ollama models list for default model selection...")
        models_response = await asyncio.to_thread(ollama.list)
        logger.debug(f"Ollama models_response for default model: {models_response}")

        if models_response and hasattr(models_response, 'models') and isinstance(models_response.models, list) and models_response.models:
            actual_models_list = models_response.models
            
            valid_models_with_tags = [
                m for m in actual_models_list
                if hasattr(m, 'model') and m.model 
            ]

            if not valid_models_with_tags:
                logger.warning("No models with valid tags found in Ollama response for default model selection (checked 'model' attribute).")
            else:
                non_embedding_models = []
                for m in valid_models_with_tags:
                    model_family = ""
                    if m.details and hasattr(m.details, 'family') and m.details.family:
                        model_family = m.details.family.lower()
                    
                    model_tag_lower = m.model.lower() 
                    if 'embed' not in model_family and 'embed' not in model_tag_lower:
                        non_embedding_models.append(m.model)

                if non_embedding_models:
                    logger.debug(f"Found non-embedding models for default: {non_embedding_models}, selecting {non_embedding_models[0]}")
                    return non_embedding_models[0]
                
                logger.info(f"No non-embedding models found, falling back to the first available model with a tag: {valid_models_with_tags[0].model}")
                return valid_models_with_tags[0].model 
        
        logger.warning("No Ollama models found or 'models' attribute is not a non-empty list in API response when determining default. Falling back to hardcoded default.")
    except Exception as e:
        logger.warning(f"Could not fetch or parse Ollama models list to determine default: {e}. Falling back to hardcoded default: {DEFAULT_OLLAMA_MODEL}", exc_info=True)
    return DEFAULT_OLLAMA_MODEL


async def process_chat_request(payload: ChatPayload) -> ChatResponse:
    if conversations_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB service not available.")

    user_message_content = payload.user_message
    conversation_id = payload.conversation_id
    
    llm_conversation_history: List[Dict[str, str]] = []
    ui_messages_for_response: List[ChatMessage] = []
    model_for_conversation: Optional[str] = None
    conv_object_id: Optional[ObjectId] = None

    if conversation_id:
        if not ObjectId.is_valid(conversation_id):
             raise HTTPException(status_code=400, detail="Invalid conversation_id format.")
        conv_object_id = ObjectId(conversation_id)
        conversation = conversations_collection.find_one({"_id": conv_object_id})
        if conversation:
            model_for_conversation = conversation.get("ollama_model_name")
            for msg_data in conversation.get("messages", []):
                if 'role' in msg_data and 'content' in msg_data:
                    llm_conversation_history.append({"role": msg_data["role"], "content": msg_data.get("raw_content_for_llm", msg_data["content"])})
                ui_messages_for_response.append(ChatMessage(**msg_data))
        else:
            raise HTTPException(status_code=404, detail=f"Conversation with ID '{conversation_id}' not found.")

    if not model_for_conversation: 
        if payload.ollama_model_name:
            model_for_conversation = payload.ollama_model_name
        else:
            model_for_conversation = await get_default_ollama_model()
            logger.info(f"No model specified in payload or conversation, using determined default: {model_for_conversation}")
    
    if not conversation_id: 
        new_conv_doc = {
            "title": f"Chat started {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "messages": [],
            "ollama_model_name": model_for_conversation 
        }
        result = conversations_collection.insert_one(new_conv_doc)
        conversation_id = str(result.inserted_id)
        conv_object_id = result.inserted_id
    elif conv_object_id and conversation and not conversation.get("ollama_model_name"): 
        conversations_collection.update_one(
            {"_id": conv_object_id},
            {"$set": {"ollama_model_name": model_for_conversation, "updated_at": datetime.now(timezone.utc)}}
        )

    user_chat_message = ChatMessage(role="user", content=user_message_content)
    ui_messages_for_response.append(user_chat_message)

    if user_message_content.lower() == '#clear':
        llm_conversation_history = [] 
        assistant_response_content = "Chat context cleared. Previous messages won't be used for the next response in this session."
        assistant_chat_message = ChatMessage(role="assistant", content=assistant_response_content)
        ui_messages_for_response.append(assistant_chat_message)
        
        if conv_object_id:
            conversations_collection.update_one(
                {"_id": conv_object_id},
                {
                    "$push": {"messages": {"$each": [
                        user_chat_message.model_dump(exclude_none=True),
                        assistant_chat_message.model_dump(exclude_none=True)
                    ]}},
                    "$set": {"updated_at": datetime.now(timezone.utc)}
                }
            )
        return ChatResponse(conversation_id=conversation_id, chat_history=ui_messages_for_response, ollama_model_name=model_for_conversation)

    prompt_for_llm = user_message_content
    search_performed_indicator_html = None

    if payload.use_search:
        if not app_state.service_ready:
            logger.warning("[MCP] Search requested but MCP service is not ready. Skipping search.")
            error_msg_obj = ChatMessage(role="assistant", content="⚠️ Web search is currently unavailable. I'll answer from my knowledge.")
            ui_messages_for_response.append(error_msg_obj)
            # We still need to save the user message before this assistant warning
            if conv_object_id:
                 user_message_to_save_dict_no_search = user_chat_message.model_dump(exclude_none=True)
                 user_message_to_save_dict_no_search["raw_content_for_llm"] = prompt_for_llm # Original prompt
                 conversations_collection.update_one(
                    {"_id": conv_object_id},
                    {"$push": {"messages": {"$each" : [
                        user_message_to_save_dict_no_search,
                        error_msg_obj.model_dump(exclude_none=True)
                    ]}}, 
                    "$set": {"updated_at": datetime.now(timezone.utc)}}
                )
            # Fall through to LLM without search results
        else: # MCP service is ready
            query = user_message_content 
            logger.info(f"[MCP] Search active. Querying for: '{query}' with model {model_for_conversation}")
            try:
                request_id = submit_search_request(query)
                response_data_mcp = wait_for_response(request_id) # This is already a dict
                
                if response_data_mcp.get("status") == "error":
                    raise Exception(response_data_mcp.get("error", "Unknown search error from MCP response"))
                
                # extract_search_results expects the raw content from MCP, which is response_data_mcp["data"]
                # server_search.py already returns a dict, so response_data_mcp["data"] should be that dict.
                extracted_results = extract_search_results(response_data_mcp.get("data")) 
                
                if extracted_results.get("status") == "error": # Check status from extracted_results
                    raise Exception(extracted_results.get("message", "Search result extraction indicated an error."))

                search_results_text = format_search_results_for_prompt(extracted_results, query)
                
                search_performed_indicator_html = f"<div class='search-indicator-custom'><b>🔍 Web Search:</b> Results for \"{query}\" used.</div>"
                
                prompt_for_llm = (
                    f"Today's date is {app_state.formatted_date}. Search results for '{query}':\n{search_results_text}\n\n"
                    f"Using these search results, answer: '{query}'"
                )
            except Exception as e:
                logger.error(f"[MCP] Search processing error: {e}", exc_info=True)
                error_msg_obj = ChatMessage(role="assistant", content=f"⚠️ Search failed for '{query}'. I'll answer from my knowledge. Error: {str(e)[:100]}")
                ui_messages_for_response.append(error_msg_obj)
                # Save user message and this assistant error if search fails mid-way
                if conv_object_id:
                    user_message_to_save_dict_search_fail = user_chat_message.model_dump(exclude_none=True)
                    user_message_to_save_dict_search_fail["raw_content_for_llm"] = query # Original query as LLM prompt
                    conversations_collection.update_one(
                        {"_id": conv_object_id},
                        {"$push": {"messages": {"$each" : [
                            user_message_to_save_dict_search_fail,
                            error_msg_obj.model_dump(exclude_none=True)
                        ]}}, 
                        "$set": {"updated_at": datetime.now(timezone.utc)}}
                    )


    llm_conversation_history.append({"role": "user", "content": prompt_for_llm})
    
    user_message_to_save_dict = user_chat_message.model_dump(exclude_none=True)
    user_message_to_save_dict["raw_content_for_llm"] = prompt_for_llm 

    # Save user message only if it wasn't saved during a search failure/skip path
    # Check if the last message in ui_messages_for_response is the user's current message
    # This logic is getting complex; simplify by saving user message once before LLM call if not already part of a specific error flow.
    # For now, assume it's pushed to DB *before* LLM call if not handled by specific error paths above.
    # The current structure pushes user message to DB *before* this block if search is involved and fails.
    # If search is not used, or succeeds, it's pushed here.
    
    # Let's ensure user message is saved if not already handled by an error path that added an assistant message
    last_ui_message_is_user = ui_messages_for_response and ui_messages_for_response[-1].role == "user"
    
    if conv_object_id and last_ui_message_is_user : # Only save if not already handled by error path that added an assistant msg
        # Check if this exact user message (by content and role) is already the last in DB.
        # This is to prevent double saving if an error path already saved it.
        # This check is imperfect. A more robust way is needed if flows get more complex.
        # For now, we assume if an error message was added to ui_messages_for_response, user message was handled.
        
        # Simpler: only update if the user message is the last one added to ui_messages_for_response
        # (meaning no intermediate assistant error message was added after it)
        if len(ui_messages_for_response) > 0 and ui_messages_for_response[-1].content == user_chat_message.content:
             conversations_collection.update_one(
                {"_id": conv_object_id},
                {"$push": {"messages": user_message_to_save_dict}, "$set": {"updated_at": datetime.now(timezone.utc)}}
            )


    model_response_content = await chat_with_ollama(llm_conversation_history, model_name=model_for_conversation)

    if model_response_content:
        llm_conversation_history.append({"role": "assistant", "content": model_response_content})
        
        assistant_response_for_ui = model_response_content
        is_html = False
        if search_performed_indicator_html: # Only add indicator if search was successful
            assistant_response_for_ui = f"{search_performed_indicator_html}\n\n{model_response_content}"
            is_html = True
        
        assistant_chat_message = ChatMessage(role="assistant", content=assistant_response_for_ui, is_html=is_html)
        ui_messages_for_response.append(assistant_chat_message)

        assistant_message_to_save_dict = assistant_chat_message.model_dump(exclude_none=True)
        assistant_message_to_save_dict["raw_content_for_llm"] = model_response_content # Save raw LLM response

        if conv_object_id:
            conversations_collection.update_one(
                {"_id": conv_object_id},
                {"$push": {"messages": assistant_message_to_save_dict}, "$set": {"updated_at": datetime.now(timezone.utc)}}
            )
    else:
        err_msg_content = f"Sorry, I couldn't generate a response using model {model_for_conversation}."
        error_chat_message = ChatMessage(role="assistant", content=err_msg_content)
        ui_messages_for_response.append(error_chat_message)
        if conv_object_id: # Save this error response
            conversations_collection.update_one(
                {"_id": conv_object_id},
                {"$push": {"messages": error_chat_message.model_dump(exclude_none=True)}, 
                 "$set": {"updated_at": datetime.now(timezone.utc)}}
            )


    return ChatResponse(conversation_id=conversation_id, chat_history=ui_messages_for_response, ollama_model_name=model_for_conversation)


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatPayload):
    try:
        return await process_chat_request(payload)
    except HTTPException:
        raise 
    except Exception as e:
        logger.error(f"Error in /api/chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error processing chat request.")

@app.get("/api/status")
async def get_status():
    ollama_available = False
    try:
        await asyncio.to_thread(ollama.list) 
        ollama_available = True
    except Exception:
        ollama_available = False
    return {
        "service_ready": app_state.service_ready, 
        "db_connected": conversations_collection is not None,
        "ollama_available": ollama_available
    }

@app.get("/api/ollama-models", response_model=List[str])
async def list_ollama_models():
    try:
        logger.info("Attempting to fetch Ollama models list from Ollama server...")
        models_response = await asyncio.to_thread(ollama.list)
        logger.debug(f"Ollama models_response content: {models_response}")

        if models_response and hasattr(models_response, 'models') and isinstance(models_response.models, list):
            actual_models_list = models_response.models
            model_tags = [m.model for m in actual_models_list if hasattr(m, 'model') and m.model]
            
            if not model_tags:
                 logger.warning("No Ollama models with valid tags found in the response (checked 'model' attribute).")
                 return [] 
            logger.info(f"Successfully fetched and parsed model tags: {model_tags}")
            return model_tags
        
        logger.warning(f"Unexpected format from Ollama API or 'models' attribute missing/not a list. Response: {models_response}")
        raise HTTPException(status_code=500, detail="Unexpected format received from Ollama server when listing models.")

    except ollama.ResponseError as e: 
        logger.error(f"Ollama API response error when fetching models: Status {e.status_code} - {e.error}", exc_info=True)
        raise HTTPException(status_code=e.status_code if e.status_code else 500, detail=f"Ollama API error: {e.error}")
    
    except ollama.RequestError as e: 
        ollama_host_env = os.getenv('OLLAMA_HOST')
        actual_host_tried = "http://localhost:11434" 
        if ollama_host_env:
            if not ollama_host_env.startswith(('http://', 'https://')):
                actual_host_tried = f"http://{ollama_host_env}" 
            else:
                actual_host_tried = ollama_host_env
        
        logger.error(f"Failed to connect to Ollama server (tried approximately: {actual_host_tried}, based on OLLAMA_HOST='{ollama_host_env}'). Error: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Could not connect to Ollama server (tried {actual_host_tried}). Please ensure it's running and accessible. Check OLLAMA_HOST environment variable if it's not on default.")

    except Exception as e: 
        logger.error(f"An unexpected error occurred while fetching Ollama models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected internal server error occurred while fetching Ollama models.")


@app.get("/api/conversations", response_model=List[ConversationListItem], response_model_by_alias=False)
async def list_conversations():
    if conversations_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB service not available.")
    try:
        convs_cursor = conversations_collection.find(
            {}, {"messages": 0} 
        ).sort("updated_at", -1).limit(50) 
        
        conversation_list = []
        _default_model_for_list = None 

        for conv_data_from_db in convs_cursor:
            full_conv_doc = conversations_collection.find_one({"_id": conv_data_from_db["_id"]}, {"messages": 1})
            message_count = len(full_conv_doc.get("messages", [])) if full_conv_doc else 0
            
            item_data_for_validation = dict(conv_data_from_db)
            if "_id" in item_data_for_validation and isinstance(item_data_for_validation["_id"], ObjectId):
                item_data_for_validation["_id"] = str(item_data_for_validation["_id"])
            
            item_data_for_validation["message_count"] = message_count
            
            if "ollama_model_name" not in item_data_for_validation or not item_data_for_validation["ollama_model_name"]:
                if _default_model_for_list is None: 
                    _default_model_for_list = await get_default_ollama_model()
                item_data_for_validation["ollama_model_name"] = _default_model_for_list
            
            conversation_list.append(ConversationListItem.model_validate(item_data_for_validation))
        return conversation_list
    except Exception as e:
        logger.error(f"Error listing conversations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error fetching conversations.")

@app.get("/api/conversations/{conversation_id}", response_model=List[ChatMessage])
async def get_conversation_messages(conversation_id: str):
    if conversations_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB service not available.")
    if not ObjectId.is_valid(conversation_id):
        raise HTTPException(status_code=400, detail="Invalid conversation_id format.")
    try:
        conversation = conversations_collection.find_one({"_id": ObjectId(conversation_id)})
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        
        messages = [ChatMessage.model_validate(msg_data) for msg_data in conversation.get("messages", [])]
        return messages
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error fetching conversation details.")

@app.delete("/api/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(conversation_id: str):
    if conversations_collection is None:
        logger.error(f"Attempt to delete conversation {conversation_id} when MongoDB is not available.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="MongoDB service not available.")
    
    if not ObjectId.is_valid(conversation_id):
        logger.warning(f"Invalid ObjectId format for deletion: {conversation_id}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid conversation_id format.")
    
    try:
        obj_id = ObjectId(conversation_id)
        delete_result = conversations_collection.delete_one({"_id": obj_id})
        
        if delete_result.deleted_count == 0:
            logger.warning(f"Conversation with ID {conversation_id} not found for deletion.")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
        
        logger.info(f"Successfully deleted conversation with ID: {conversation_id}")
        return Response(status_code=status.HTTP_204_NO_CONTENT)
        
    except OperationFailure as e:
        logger.error(f"MongoDB operation failure while deleting conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database operation failed during deletion.")
    except Exception as e:
        logger.error(f"Unexpected error while deleting conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred during deletion.")

@app.put("/api/conversations/{conversation_id}/rename", response_model=ConversationListItem, response_model_by_alias=False)
async def rename_conversation_title(conversation_id: str, payload: RenamePayload):
    if conversations_collection is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="MongoDB service not available.")
    
    if not ObjectId.is_valid(conversation_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid conversation_id format.")
    
    obj_id = ObjectId(conversation_id)
    
    try:
        # Check if conversation exists
        conversation = conversations_collection.find_one({"_id": obj_id}, {"messages": 0}) # Exclude messages for this check
        if not conversation:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")

        # Update the title and updated_at timestamp
        new_timestamp = datetime.now(timezone.utc)
        update_result = conversations_collection.update_one(
            {"_id": obj_id},
            {"$set": {"title": payload.new_title, "updated_at": new_timestamp}}
        )
        
        if update_result.matched_count == 0: # Should be caught by find_one above, but as a safeguard
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found during update.")

        # Fetch the updated document to return it
        updated_conv_doc = conversations_collection.find_one({"_id": obj_id})
        if not updated_conv_doc: # Should not happen if update was successful
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve updated conversation.")


        # Recalculate message_count as it's not stored with the main doc if we excluded messages before
        full_conv_doc_for_count = conversations_collection.find_one({"_id": obj_id}, {"messages": 1})
        message_count = len(full_conv_doc_for_count.get("messages", [])) if full_conv_doc_for_count else 0

        item_data_for_validation = dict(updated_conv_doc)
        if "_id" in item_data_for_validation and isinstance(item_data_for_validation["_id"], ObjectId):
            item_data_for_validation["_id"] = str(item_data_for_validation["_id"])
        item_data_for_validation["message_count"] = message_count
        
        # Ensure model name is present
        if "ollama_model_name" not in item_data_for_validation or not item_data_for_validation["ollama_model_name"]:
            item_data_for_validation["ollama_model_name"] = await get_default_ollama_model()
            
        logger.info(f"Successfully renamed conversation ID {conversation_id} to '{payload.new_title}'")
        return ConversationListItem.model_validate(item_data_for_validation)

    except HTTPException: # Re-raise known HTTP exceptions
        raise
    except OperationFailure as e:
        logger.error(f"MongoDB operation failure while renaming conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database operation failed during rename.")
    except Exception as e:
        logger.error(f"Unexpected error while renaming conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred during rename.")


# Serve static files
frontend_dist_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'dist')
if os.path.exists(frontend_dist_path):
    logger.info(f"Serving static files from: {frontend_dist_path}")
    app.mount("/", StaticFiles(directory=frontend_dist_path, html=True), name="static_frontend")
else:
    logger.warning(f"Frontend build directory not found at {frontend_dist_path}.")

# --- Main Application Logic ---
def main_backend():
    if conversations_collection is None:
        logger.warning("MongoDB is not connected. Persistence features will be unavailable.")
    
    service_thread = threading.Thread(target=run_mcp_service, daemon=True)
    service_thread.start()
    logger.info("MCP service thread started")
    # It might take a moment for the service_ready flag to be set.
    # The frontend will poll /api/status.
    # time.sleep(2) # Removed fixed sleep, status polling is better
    logger.info(f"FastAPI backend setup complete. Default Ollama model: {DEFAULT_OLLAMA_MODEL}. Run with: uvicorn backend.main:app --reload --port 8000")

if __name__ == "__main__":
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, initiating shutdown...")
        if mongo_client:
            mongo_client.close()
            logger.info("MongoDB connection closed.")
        # Potentially add cleanup for service_thread if needed, though daemon=True helps.
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main_backend()
