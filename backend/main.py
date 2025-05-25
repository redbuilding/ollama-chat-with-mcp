import asyncio
import json
import re
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
from contextlib import asynccontextmanager # For lifespan events

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel, Field, constr
from bson import ObjectId
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

from mcp import ClientSession, StdioServerParameters, types # Import types for MCPError
from mcp.client.stdio import stdio_client
import ollama # Ensure ollama is imported
# import threading # No longer needed for MCP service thread
import time # Still used for request_id
import queue

# --- Environment Setup ---
MCP_SERVER_SCRIPT = "server_search.py"
_main_py_dir = os.path.dirname(os.path.abspath(__file__))
MCP_SERVER_SCRIPT_ABS_PATH = os.path.join(_main_py_dir, MCP_SERVER_SCRIPT)

if not os.path.exists(MCP_SERVER_SCRIPT_ABS_PATH):
    logging.error(f"CRITICAL: MCP server script '{MCP_SERVER_SCRIPT_ABS_PATH}' not found. Searched in directory: '{_main_py_dir}'. Search functionality will be disabled.")
else:
    logging.info(f"MCP server script found at: {MCP_SERVER_SCRIPT_ABS_PATH}")


os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/mcp_backend_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mcp_backend")
logger.setLevel(logging.DEBUG) 

# --- MongoDB Setup ---
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DATABASE_NAME = os.getenv('MONGODB_DATABASE_NAME', 'mcp_chat_db')
MONGODB_COLLECTION_NAME = os.getenv('MONGODB_COLLECTION_NAME', 'conversations')
DEFAULT_OLLAMA_MODEL = os.getenv('DEFAULT_OLLAMA_MODEL', 'qwen2:7b') 

try:
    mongo_client = MongoClient(MONGODB_URI)
    mongo_client.admin.command('ping') 
    db = mongo_client[MONGODB_DATABASE_NAME]
    conversations_collection = db[MONGODB_COLLECTION_NAME]
    logger.info(f"Successfully connected to MongoDB: {MONGODB_URI}")
except ConnectionFailure:
    logger.error(f"Failed to connect to MongoDB at {MONGODB_URI}.")
    mongo_client = None; db = None; conversations_collection = None
except Exception as e:
    logger.error(f"An error occurred during MongoDB setup: {e}")
    mongo_client = None; db = None; conversations_collection = None

# --- Communication Queues ---
request_queue = queue.Queue()
response_queue = queue.Queue()

# --- State Management ---
class AppState:
    def __init__(self):
        self.current_date = datetime.now(timezone.utc)
        self.service_ready = False
        self.mcp_task: Optional[asyncio.Task] = None # To hold the MCP service task
    @property
    def formatted_date(self): return f"{self.current_date.strftime('%B')} {self.current_date.strftime('%d').lstrip('0')}, {self.current_date.strftime('%Y')}"
    @property
    def month_year(self): return f"{self.current_date.strftime('%B')} {self.current_date.strftime('%Y')}"
    @property
    def year(self): return self.current_date.strftime('%Y')

app_state = AppState() # Global app state instance

# --- MCP Service ---
async def mcp_service_loop():
    logger.info("MCP_SERVICE_LOOP: Starting...")
    if not os.path.exists(MCP_SERVER_SCRIPT_ABS_PATH):
        logger.error(f"MCP_SERVICE_LOOP: MCP service cannot start: script '{MCP_SERVER_SCRIPT_ABS_PATH}' not found.")
        app_state.service_ready = False
        return

    mcp_client_comms_logger = logger.getChild("mcp_client_comms")
    mcp_client_comms_logger.setLevel(logging.DEBUG) 

    while True: # Outer loop to allow for reconnection attempts
        try:
            server_params_for_client = StdioServerParameters(
                command=sys.executable,
                args=[MCP_SERVER_SCRIPT_ABS_PATH], 
                env=None 
            )
            logger.info(f"MCP_SERVICE_LOOP: Attempting to start MCP server using stdio_client with: {sys.executable} {MCP_SERVER_SCRIPT_ABS_PATH}")
            
            async with stdio_client(server_params_for_client, logger=mcp_client_comms_logger) as (read, write):
                logger.info("MCP_SERVICE_LOOP: stdio_client connected. Initializing ClientSession...")
                async with ClientSession(read, write) as session:
                    logger.info("MCP_SERVICE_LOOP: ClientSession created. Calling session.initialize()...")
                    await session.initialize() 
                    logger.info("MCP_SERVICE_LOOP: MCP session initialized successfully. Listing tools...")
                    tools_response = await session.list_tools() 
                    
                    logger.debug(f"MCP_SERVICE_LOOP: Raw tools_response from MCP server: {tools_response!r}")

                    if tools_response and tools_response.tools is not None:
                        tool_names = [tool.name for tool in tools_response.tools]
                        logger.info(f"MCP_SERVICE_LOOP: Available tools from MCP server: {tool_names}")
                        if "web_search" not in tool_names: 
                            logger.error(f"MCP_SERVICE_LOOP: CRITICAL: Required 'web_search' tool NOT FOUND among available tools: {tool_names}.")
                            app_state.service_ready = False
                            # No return here, will let the outer loop retry after a delay
                        else:
                            app_state.service_ready = True
                            logger.info("MCP_SERVICE_LOOP: MCP service fully initialized and 'web_search' tool is available.")
                    else:
                        logger.error(f"MCP_SERVICE_LOOP: CRITICAL: No tools found or invalid/empty response from session.list_tools(). Response object: {tools_response!r}")
                        app_state.service_ready = False
                        # No return here, will let the outer loop retry

                    # If service became ready, enter operational loop
                    if app_state.service_ready:
                        while True: # Inner operational loop
                            if not request_queue.empty():
                                request_data = request_queue.get_nowait()
                                if request_data["type"] == "search":
                                    query = request_data["query"]
                                    request_id = request_data["id"]
                                    try:
                                        logger.debug(f"MCP_SERVICE_LOOP: Calling 'web_search' tool with query: {query}")
                                        result = await session.call_tool("web_search", {"query": query})
                                        logger.debug(f"MCP_SERVICE_LOOP: 'web_search' tool call successful. Result content type: {type(result.content)}")
                                        response_queue.put({"id": request_id, "type": "search_result", "status": "success", "data": result.content})
                                    except Exception as e_tool_call:
                                        logger.error(f"MCP_SERVICE_LOOP: Error in 'web_search' tool call: {e_tool_call}", exc_info=True)
                                        response_queue.put({"id": request_id, "type": "search_result", "status": "error", "error": str(e_tool_call)})
                            await asyncio.sleep(0.1) # Check queue periodically
        
        except ConnectionRefusedError as e_conn_refused:
            logger.error(f"MCP_SERVICE_LOOP: Connection refused. Is the server script ({MCP_SERVER_SCRIPT_ABS_PATH}) runnable? Error: {e_conn_refused}", exc_info=False) # Less verbose exc_info
        except asyncio.TimeoutError as e_timeout: 
            logger.error(f"MCP_SERVICE_LOOP: Timeout in MCP service communication: {e_timeout}", exc_info=False)
        except types.MCPError as e_mcp_protocol: 
            logger.error(f"MCP_SERVICE_LOOP: MCP protocol error: {e_mcp_protocol}", exc_info=False)
        except Exception as e_generic: 
            logger.error(f"MCP_SERVICE_LOOP: Generic error in MCP service: {e_generic}", exc_info=True) # Full exc_info for unexpected
        finally:
            app_state.service_ready = False 
            logger.info("MCP_SERVICE_LOOP: Connection lost or failed. Will attempt to reconnect after a delay.")
            await asyncio.sleep(10) # Wait before retrying connection


# --- FastAPI Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI Lifespan: Startup sequence initiated.")
    # Start the MCP service loop as a background task
    app_state.mcp_task = asyncio.create_task(mcp_service_loop())
    logger.info("FastAPI Lifespan: MCP service task created.")
    yield # Application runs here
    logger.info("FastAPI Lifespan: Shutdown sequence initiated.")
    if app_state.mcp_task:
        app_state.mcp_task.cancel()
        try:
            await app_state.mcp_task
        except asyncio.CancelledError:
            logger.info("FastAPI Lifespan: MCP service task successfully cancelled.")
        except Exception as e:
            logger.error(f"FastAPI Lifespan: Error during MCP service task shutdown: {e}", exc_info=True)
    if mongo_client:
        mongo_client.close()
        logger.info("FastAPI Lifespan: MongoDB connection closed.")
    logger.info("FastAPI Lifespan: Shutdown complete.")


# --- FastAPI Application ---
app = FastAPI(lifespan=lifespan) # Pass the lifespan manager to the app

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], 
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- Utility Functions for MCP Interaction (submit_search_request, wait_for_response) ---
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
    if isinstance(response_content, dict):
        return response_content
    elif hasattr(response_content, 'text') and isinstance(response_content.text, str):
        try:
            return json.loads(response_content.text)
        except json.JSONDecodeError as e:
            logger.error(f"extract_search_results: JSONDecodeError parsing TextContent: {e}. Content: {response_content.text[:500]}")
            return {"status": "error", "message": "Failed to parse search result JSON from TextContent."}
    elif isinstance(response_content, str): 
        try:
            return json.loads(response_content)
        except json.JSONDecodeError as e:
            logger.error(f"extract_search_results: JSONDecodeError parsing plain string: {e}. Content: {response_content[:500]}")
            return {"status": "error", "message": "Failed to parse search result JSON from string."}
    logger.warning(f"extract_search_results: Unhandled response_content type: {type(response_content)}. Content: {str(response_content)[:500]}")
    return {"status": "error", "message": f"Unhandled search result type: {type(response_content)}"}

def format_search_results_for_prompt(results_data, query, max_results=3):
    if not isinstance(results_data, dict) or results_data.get("status") == "error":
        error_message = results_data.get("message", "Search returned no recognizable results or an error.")
        return f"Search for '{query}': {error_message}"
    organic_results = results_data.get('organic_results', []) 
    if organic_results:
        formatted = "\n".join(
            f"{i+1}. {item.get('title', 'N/A')}\n   {item.get('snippet', 'N/A')}\n   Source: {item.get('link', 'N/A')}"
            for i, item in enumerate(organic_results[:max_results])
        )
        return f"Web search results for '{query}':\n{formatted}"
    return f"Search for '{query}' returned no specific organic results."

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
    ollama_model_name: Optional[str] = None 

class ChatResponse(BaseModel):
    conversation_id: str
    chat_history: List[ChatMessage] 
    ollama_model_name: Optional[str] = None 

class ConversationListItem(BaseModel):
    id: str = Field(alias="_id")
    title: Optional[str] = "New Chat"
    created_at: datetime
    updated_at: datetime
    message_count: int
    ollama_model_name: Optional[str] = None 
    class Config:
        populate_by_name = True 
        json_encoders = {ObjectId: str, datetime: lambda dt: dt.isoformat()}

class RenamePayload(BaseModel):
    new_title: constr(strip_whitespace=True, min_length=1, max_length=100)

async def get_default_ollama_model() -> str:
    try:
        models_response = await asyncio.to_thread(ollama.list)
        if models_response and hasattr(models_response, 'models') and isinstance(models_response.models, list) and models_response.models:
            actual_models_list = models_response.models
            valid_models_with_tags = [m for m in actual_models_list if hasattr(m, 'model') and m.model]
            if not valid_models_with_tags:
                logger.warning("No models with valid tags found in Ollama response for default model selection.")
            else:
                non_embedding_models = [m.model for m in valid_models_with_tags if 'embed' not in (m.details.family.lower() if m.details and hasattr(m.details, 'family') and m.details.family else "") and 'embed' not in m.model.lower()]
                if non_embedding_models: return non_embedding_models[0]
                return valid_models_with_tags[0].model 
        logger.warning("No Ollama models found or 'models' attribute is not a non-empty list in API response when determining default. Falling back to hardcoded default.")
    except Exception as e:
        logger.warning(f"Could not fetch or parse Ollama models list to determine default: {e}. Falling back to hardcoded default: {DEFAULT_OLLAMA_MODEL}", exc_info=False)
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
        if payload.ollama_model_name: model_for_conversation = payload.ollama_model_name
        else: model_for_conversation = await get_default_ollama_model()
    
    if not conversation_id: 
        new_conv_doc = {
            "title": f"Chat: {user_message_content[:30]}..." if len(user_message_content) > 30 else f"Chat: {user_message_content}",
            "created_at": datetime.now(timezone.utc), "updated_at": datetime.now(timezone.utc),
            "messages": [], "ollama_model_name": model_for_conversation 
        }
        result = conversations_collection.insert_one(new_conv_doc)
        conversation_id = str(result.inserted_id); conv_object_id = result.inserted_id
    elif conv_object_id and conversation and not conversation.get("ollama_model_name"): 
        conversations_collection.update_one(
            {"_id": conv_object_id},
            {"$set": {"ollama_model_name": model_for_conversation, "updated_at": datetime.now(timezone.utc)}})

    user_chat_message = ChatMessage(role="user", content=user_message_content)
    ui_messages_for_response.append(user_chat_message)
    user_message_to_save_dict = user_chat_message.model_dump(exclude_none=True)
    user_message_to_save_dict["raw_content_for_llm"] = user_message_content
    if conv_object_id:
        conversations_collection.update_one(
            {"_id": conv_object_id},
            {"$push": {"messages": user_message_to_save_dict}, "$set": {"updated_at": datetime.now(timezone.utc)}})

    if user_message_content.lower() == '#clear':
        llm_conversation_history = [] 
        assistant_response_content = "Chat context cleared."
        assistant_chat_message = ChatMessage(role="assistant", content=assistant_response_content)
        ui_messages_for_response.append(assistant_chat_message)
        if conv_object_id:
            conversations_collection.update_one(
                {"_id": conv_object_id},
                {"$push": {"messages": assistant_chat_message.model_dump(exclude_none=True)}, 
                 "$set": {"updated_at": datetime.now(timezone.utc)}})
        return ChatResponse(conversation_id=conversation_id, chat_history=ui_messages_for_response, ollama_model_name=model_for_conversation)

    prompt_for_llm = user_message_content
    search_performed_indicator_html = None
    assistant_error_message_obj = None

    if payload.use_search:
        if not app_state.service_ready:
            logger.warning("[API_CHAT] Search requested but MCP service is not ready.")
            assistant_error_message_obj = ChatMessage(role="assistant", content="⚠️ Web search is currently unavailable.")
        else:
            query = user_message_content 
            logger.info(f"[API_CHAT] Search active for: '{query}'")
            try:
                request_id = submit_search_request(query)
                response_data_mcp = wait_for_response(request_id)
                if response_data_mcp.get("status") == "error": raise Exception(response_data_mcp.get("error", "MCP search error"))
                extracted_results = extract_search_results(response_data_mcp.get("data")) 
                if extracted_results.get("status") == "error": raise Exception(extracted_results.get("message", "Search result extraction error."))
                search_results_text = format_search_results_for_prompt(extracted_results, query)
                search_performed_indicator_html = f"<div class='search-indicator-custom'><b>🔍 Web Search:</b> Results for \"{query}\" used.</div>"
                prompt_for_llm = (f"Search results for '{query}':\n{search_results_text}\n\nUsing these, answer: '{query}'")
                if conv_object_id:
                    conversations_collection.update_one(
                        {"_id": conv_object_id, "messages.timestamp": user_chat_message.timestamp},
                        {"$set": {"messages.$.raw_content_for_llm": prompt_for_llm}})
            except Exception as e:
                logger.error(f"[API_CHAT] Search processing error: {e}", exc_info=True)
                assistant_error_message_obj = ChatMessage(role="assistant", content=f"⚠️ Search failed for '{query}'. Error: {str(e)[:100]}")

    if assistant_error_message_obj:
        ui_messages_for_response.append(assistant_error_message_obj)
        if conv_object_id:
            conversations_collection.update_one(
                {"_id": conv_object_id},
                {"$push": {"messages": assistant_error_message_obj.model_dump(exclude_none=True)},
                 "$set": {"updated_at": datetime.now(timezone.utc)}})

    llm_conversation_history.append({"role": "user", "content": prompt_for_llm})
    model_response_content = await chat_with_ollama(llm_conversation_history, model_name=model_for_conversation)

    if model_response_content:
        assistant_response_for_ui = model_response_content; is_html = False
        if search_performed_indicator_html and not assistant_error_message_obj:
            assistant_response_for_ui = f"{search_performed_indicator_html}\n\n{model_response_content}"; is_html = True
        assistant_chat_message = ChatMessage(role="assistant", content=assistant_response_for_ui, is_html=is_html)
        ui_messages_for_response.append(assistant_chat_message)
        assistant_message_to_save_dict = assistant_chat_message.model_dump(exclude_none=True)
        assistant_message_to_save_dict["raw_content_for_llm"] = model_response_content 
        if conv_object_id:
            conversations_collection.update_one(
                {"_id": conv_object_id},
                {"$push": {"messages": assistant_message_to_save_dict}, "$set": {"updated_at": datetime.now(timezone.utc)}})
    elif not assistant_error_message_obj:
        err_msg_content = f"Sorry, I couldn't generate a response using model {model_for_conversation}."
        llm_error_chat_message = ChatMessage(role="assistant", content=err_msg_content)
        ui_messages_for_response.append(llm_error_chat_message)
        if conv_object_id: 
            conversations_collection.update_one(
                {"_id": conv_object_id},
                {"$push": {"messages": llm_error_chat_message.model_dump(exclude_none=True)}, 
                 "$set": {"updated_at": datetime.now(timezone.utc)}})
    return ChatResponse(conversation_id=conversation_id, chat_history=ui_messages_for_response, ollama_model_name=model_for_conversation)

# --- API Endpoints ---
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatPayload):
    try: return await process_chat_request(payload)
    except HTTPException: raise 
    except Exception as e: logger.error(f"Error in /api/chat: {e}", exc_info=True); raise HTTPException(status_code=500, detail="Internal server error.")

@app.get("/api/status")
async def get_status():
    ollama_available = False
    try: await asyncio.to_thread(ollama.list); ollama_available = True
    except Exception: ollama_available = False
    return {"service_ready": app_state.service_ready, "db_connected": conversations_collection is not None, "ollama_available": ollama_available}

@app.get("/api/ollama-models", response_model=List[str])
async def list_ollama_models():
    try:
        models_response = await asyncio.to_thread(ollama.list)
        if models_response and hasattr(models_response, 'models') and isinstance(models_response.models, list):
            model_tags = [m.model for m in models_response.models if hasattr(m, 'model') and m.model]
            if not model_tags: logger.warning("No Ollama models with valid tags found."); return [] 
            return model_tags
        logger.warning(f"Unexpected format from Ollama API: {models_response}"); raise HTTPException(status_code=500, detail="Unexpected format from Ollama.")
    except ollama.ResponseError as e: logger.error(f"Ollama API error: {e}", exc_info=True); raise HTTPException(status_code=e.status_code or 500, detail=f"Ollama API error: {e.error}")
    except ollama.RequestError as e: 
        host = os.getenv('OLLAMA_HOST', 'localhost:11434'); actual_host = f"http://{host}" if not host.startswith(('http://','https://')) else host
        logger.error(f"Failed to connect to Ollama (tried {actual_host}): {e}", exc_info=True); raise HTTPException(status_code=503, detail=f"Could not connect to Ollama (tried {actual_host}).")
    except Exception as e: logger.error(f"Unexpected error fetching Ollama models: {e}", exc_info=True); raise HTTPException(status_code=500, detail="Unexpected error fetching models.")

@app.get("/api/conversations", response_model=List[ConversationListItem], response_model_by_alias=False)
async def list_conversations():
    if conversations_collection is None: raise HTTPException(status_code=503, detail="MongoDB not available.")
    try:
        convs_cursor = conversations_collection.find({}, {"messages": 0}).sort("updated_at", -1).limit(50)
        conv_list = []
        default_model = None
        for db_conv in convs_cursor:
            msg_count = conversations_collection.count_documents({"_id": db_conv["_id"], "messages": {"$exists": True}}) # More accurate count
            item_data = {**db_conv, "_id": str(db_conv["_id"]), "message_count": msg_count}
            if not item_data.get("ollama_model_name"):
                if default_model is None: default_model = await get_default_ollama_model()
                item_data["ollama_model_name"] = default_model
            conv_list.append(ConversationListItem.model_validate(item_data))
        return conv_list
    except Exception as e: logger.error(f"Error listing conversations: {e}", exc_info=True); raise HTTPException(status_code=500, detail="Error fetching conversations.")

@app.get("/api/conversations/{conversation_id}", response_model=List[ChatMessage])
async def get_conversation_messages(conversation_id: str):
    if conversations_collection is None: raise HTTPException(status_code=503, detail="MongoDB not available.")
    if not ObjectId.is_valid(conversation_id): raise HTTPException(status_code=400, detail="Invalid ID format.")
    try:
        conv = conversations_collection.find_one({"_id": ObjectId(conversation_id)})
        if not conv: raise HTTPException(status_code=404, detail="Conversation not found.")
        return [ChatMessage.model_validate(msg) for msg in conv.get("messages", [])]
    except HTTPException: raise
    except Exception as e: logger.error(f"Error fetching conv {conversation_id}: {e}", exc_info=True); raise HTTPException(status_code=500, detail="Error fetching details.")

@app.delete("/api/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(conversation_id: str):
    if conversations_collection is None: raise HTTPException(status_code=503, detail="MongoDB not available.")
    if not ObjectId.is_valid(conversation_id): raise HTTPException(status_code=400, detail="Invalid ID format.")
    try:
        res = conversations_collection.delete_one({"_id": ObjectId(conversation_id)})
        if res.deleted_count == 0: raise HTTPException(status_code=404, detail="Conversation not found.")
        logger.info(f"Deleted conversation ID: {conversation_id}")
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except Exception as e: logger.error(f"Error deleting conv {conversation_id}: {e}", exc_info=True); raise HTTPException(status_code=500, detail="Error during deletion.")

@app.put("/api/conversations/{conversation_id}/rename", response_model=ConversationListItem, response_model_by_alias=False)
async def rename_conversation_title(conversation_id: str, payload: RenamePayload):
    if conversations_collection is None: raise HTTPException(status_code=503, detail="MongoDB not available.")
    if not ObjectId.is_valid(conversation_id): raise HTTPException(status_code=400, detail="Invalid ID format.")
    obj_id = ObjectId(conversation_id)
    try:
        if not conversations_collection.find_one({"_id": obj_id}, {"_id": 1}): raise HTTPException(status_code=404, detail="Conversation not found.")
        conversations_collection.update_one({"_id": obj_id}, {"$set": {"title": payload.new_title, "updated_at": datetime.now(timezone.utc)}})
        updated_conv = conversations_collection.find_one({"_id": obj_id})
        if not updated_conv: raise HTTPException(status_code=500, detail="Failed to retrieve updated conversation.")
        msg_count = conversations_collection.count_documents({"_id": updated_conv["_id"], "messages": {"$exists": True}})
        item_data = {**updated_conv, "_id": str(updated_conv["_id"]), "message_count": msg_count}
        if not item_data.get("ollama_model_name"): item_data["ollama_model_name"] = await get_default_ollama_model()
        logger.info(f"Renamed conversation ID {conversation_id} to '{payload.new_title}'")
        return ConversationListItem.model_validate(item_data)
    except Exception as e: logger.error(f"Error renaming conv {conversation_id}: {e}", exc_info=True); raise HTTPException(status_code=500, detail="Error during rename.")

# Serve static files
frontend_dist_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'dist')
if os.path.exists(frontend_dist_path):
    logger.info(f"Serving static files from: {frontend_dist_path}")
    app.mount("/", StaticFiles(directory=frontend_dist_path, html=True), name="static_frontend")
else:
    logger.warning(f"Frontend build directory not found at {frontend_dist_path}. Ensure 'npm run build' has been executed in 'frontend' directory.")

# Uvicorn execution (if running this script directly, e.g. python backend/main.py)
if __name__ == "__main__":
    logger.info(f"Starting Uvicorn server directly for {__name__}. Note: Lifespan events (like MCP service startup) are managed by FastAPI app instance.")
    # Setup signal handlers for graceful shutdown when run directly
    def handle_signal(sig, frame):
        logger.info(f"Received signal {sig}, initiating graceful shutdown...")
        # This is tricky because Uvicorn's own signal handling might conflict or take precedence.
        # For Uvicorn run via CLI, CTRL+C is handled by Uvicorn to trigger lifespan shutdown.
        # This custom handler is more for `python main.py` scenarios if Uvicorn programmatic start was used differently.
        if app_state.mcp_task and not app_state.mcp_task.done():
            app_state.mcp_task.cancel()
        if mongo_client:
            mongo_client.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
