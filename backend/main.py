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

# MCP Imports
import subprocess
from mcp import ClientSession
from mcp.client.stdio import StdioClientTransport

import ollama 
import time 
import queue

# --- Environment Setup ---
# Determine the absolute path to the backend directory to run server_search.py from there
_main_py_dir = os.path.dirname(os.path.abspath(__file__)) 

# Configuration for launching the MCP server
MCP_SERVER_SCRIPT = "server_search.py" 
# The command will be run with cwd set to _main_py_dir (backend directory)
MCP_SERVER_COMMAND = ["fastmcp", "run", MCP_SERVER_SCRIPT] 

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

logger.info(f"MCP Client will launch FastMCP server using command: {' '.join(MCP_SERVER_COMMAND)} from directory: {_main_py_dir}")

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
        self.mcp_task: Optional[asyncio.Task] = None 
    @property
    def formatted_date(self): return f"{self.current_date.strftime('%B')} {self.current_date.strftime('%d').lstrip('0')}, {self.current_date.strftime('%Y')}"
    @property
    def month_year(self): return f"{self.current_date.strftime('%B')} {self.current_date.strftime('%Y')}"
    @property
    def year(self): return self.current_date.strftime('%Y')

app_state = AppState() 

# --- MCP Service ---
async def mcp_service_loop():
    logger.info("MCP_SERVICE_LOOP: Starting STDIO client loop...")
    
    mcp_client_comms_logger = logger.getChild("mcp_client_comms")
    mcp_client_comms_logger.setLevel(logging.DEBUG) # Or use parent logger's level

    process = None # Ensure process is defined for finally block

    while True:
        try:
            logger.info(f"MCP_SERVICE_LOOP: Starting FastMCP server subprocess: {' '.join(MCP_SERVER_COMMAND)}")
            
            # Start the FastMCP server as a subprocess
            process = await asyncio.create_subprocess_exec(
                *MCP_SERVER_COMMAND,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, # Capture stderr
                cwd=_main_py_dir  # Run in backend directory
            )
            
            logger.info(f"MCP_SERVICE_LOOP: FastMCP server subprocess started successfully (PID: {process.pid}).")
            
            # Create STDIO transport
            transport = StdioClientTransport(
                process.stdout,
                process.stdin,
                logger=mcp_client_comms_logger
            )
            
            async with transport.connect() as (read, write):
                logger.info("MCP_SERVICE_LOOP: Connected to FastMCP server via STDIO. Initializing ClientSession...")
                
                async with ClientSession(read, write) as session:
                    logger.info("MCP_SERVICE_LOOP: ClientSession created. Calling session.initialize()...")
                    await session.initialize() # Add capabilities if needed, e.g., {"notifications": "optional"}
                    
                    logger.info("MCP_SERVICE_LOOP: MCP session initialized successfully. Listing tools...")
                    tools_response = await session.list_tools()
                    
                    logger.debug(f"MCP_SERVICE_LOOP: Raw tools_response from MCP server: {tools_response!r}")

                    if tools_response and tools_response.tools is not None:
                        tool_names = [tool.name for tool in tools_response.tools]
                        logger.info(f"MCP_SERVICE_LOOP: Available tools from MCP server: {tool_names}")
                        if "web_search" not in tool_names:
                            logger.error(f"MCP_SERVICE_LOOP: CRITICAL: Required 'web_search' tool NOT FOUND among available tools: {tool_names}.")
                            app_state.service_ready = False
                        else:
                            app_state.service_ready = True
                            logger.info("MCP_SERVICE_LOOP: MCP service fully initialized and 'web_search' tool is available.")
                    else:
                        logger.error(f"MCP_SERVICE_LOOP: CRITICAL: No tools found or invalid/empty response from session.list_tools(). Response object: {tools_response!r}")
                        app_state.service_ready = False

                    if app_state.service_ready:
                        # Main service loop
                        while True:
                            if not request_queue.empty():
                                request_data = request_queue.get_nowait()
                                if request_data["type"] == "search":
                                    query = request_data["query"]
                                    request_id = request_data["id"]
                                    try:
                                        logger.debug(f"MCP_SERVICE_LOOP: Calling 'web_search' tool with query: {query}")
                                        result = await session.call_tool("web_search", {"query": query}) # result is CallToolResponse
                                        logger.debug(f"MCP_SERVICE_LOOP: 'web_search' tool call successful. Result content type: {type(result.content)}")
                                        # result.content is expected to be a dict if tool returns JSON
                                        response_queue.put({"id": request_id, "type": "search_result", "status": "success", "data": result.content})
                                    except Exception as e_tool_call: # Catch mcp.error.MCPError or others
                                        logger.error(f"MCP_SERVICE_LOOP: Error in 'web_search' tool call: {e_tool_call}", exc_info=True)
                                        response_queue.put({"id": request_id, "type": "search_result", "status": "error", "error": str(e_tool_call)})
                            
                            # Check if subprocess is still alive
                            if process.returncode is not None:
                                stderr_output = ""
                                if process.stderr:
                                    stderr_bytes = await process.stderr.read()
                                    stderr_output = stderr_bytes.decode(errors='replace')
                                logger.error(f"MCP_SERVICE_LOOP: FastMCP server subprocess has terminated unexpectedly with code {process.returncode}. Stderr: {stderr_output}")
                                break # Exit inner while loop to trigger reconnect
                                
                            await asyncio.sleep(0.1) # Yield control

        except FileNotFoundError:
            logger.error(f"MCP_SERVICE_LOOP: 'fastmcp' command not found. Please ensure FastMCP is installed and in PATH: pip install fastmcp", exc_info=True)
        except asyncio.TimeoutError as e_timeout: # This might be from session calls if they timeout
            logger.error(f"MCP_SERVICE_LOOP: TimeoutError in MCP service communication: {e_timeout}", exc_info=True)
        except ConnectionRefusedError as e_conn_refused: # Should not happen with STDIO, but good to be aware
            logger.error(f"MCP_SERVICE_LOOP: ConnectionRefusedError: {e_conn_refused}", exc_info=True)
        except Exception as e_generic: # Catch-all for other issues during setup or main loop
            logger.error(f"MCP_SERVICE_LOOP: Generic Exception during MCP service operation: {e_generic}", exc_info=True)
        finally:
            app_state.service_ready = False
            # Clean up subprocess
            if process and process.returncode is None: # Check if process was started and is still running
                logger.info(f"MCP_SERVICE_LOOP: Terminating FastMCP server subprocess (PID: {process.pid})...")
                process.terminate() # Send SIGTERM
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                    logger.info(f"MCP_SERVICE_LOOP: FastMCP server subprocess (PID: {process.pid}) terminated gracefully with code {process.returncode}.")
                except asyncio.TimeoutError:
                    logger.warning(f"MCP_SERVICE_LOOP: FastMCP server subprocess (PID: {process.pid}) didn't terminate gracefully after 5s, killing it (SIGKILL)...")
                    process.kill() # Send SIGKILL
                    await process.wait() # Ensure it's reaped
                    logger.info(f"MCP_SERVICE_LOOP: FastMCP server subprocess (PID: {process.pid}) killed with code {process.returncode}.")
            elif process and process.returncode is not None:
                 logger.info(f"MCP_SERVICE_LOOP: FastMCP server subprocess (PID: {process.pid}) had already terminated with code {process.returncode}.")
            
            process = None # Reset process variable for the next iteration
            logger.info("MCP_SERVICE_LOOP: Connection lost or failed. Will attempt to reconnect after a 10s delay.")
            await asyncio.sleep(10)


# --- FastAPI Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI Lifespan: Startup sequence initiated.")
    app_state.mcp_task = asyncio.create_task(mcp_service_loop())
    logger.info("FastAPI Lifespan: MCP service client task created.")
    yield 
    logger.info("FastAPI Lifespan: Shutdown sequence initiated.")
    if app_state.mcp_task:
        app_state.mcp_task.cancel()
        try:
            await app_state.mcp_task
        except asyncio.CancelledError:
            logger.info("FastAPI Lifespan: MCP service client task successfully cancelled.")
        except Exception as e:
            logger.error(f"FastAPI Lifespan: Error during MCP service client task shutdown: {e}", exc_info=True)
    if mongo_client:
        mongo_client.close()
        logger.info("FastAPI Lifespan: MongoDB connection closed.")
    logger.info("FastAPI Lifespan: Shutdown complete.")

app = FastAPI(lifespan=lifespan) 

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], 
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

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
        time.sleep(0.5) # Reduced sleep from 0.5 to 0.1 for faster response, but 0.5 is fine.
    return {"id": request_id, "type": "search_result", "status": "error", "error": "Request timed out"}

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

def extract_search_results(response_content): # response_content is now result.content from MCP
    if isinstance(response_content, dict): 
        return response_content # Expected path: MCP tool returns deserialized JSON (dict)
    # The following branches might be less relevant if MCP tool behaves as expected
    elif hasattr(response_content, 'text') and isinstance(response_content.text, str):
        try: return json.loads(response_content.text)
        except json.JSONDecodeError as e: logger.error(f"extract_search_results: JSONDecodeError from .text attribute: {e}. Content: {response_content.text[:200]}..."); return {"status": "error", "message": "Failed to parse search JSON from .text."}
    elif isinstance(response_content, str): 
        try: return json.loads(response_content)
        except json.JSONDecodeError as e: logger.error(f"extract_search_results: JSONDecodeError from string: {e}. Content: {response_content[:200]}..."); return {"status": "error", "message": "Failed to parse search JSON string."}
    
    # If it's not a dict, or a string/object-with-text that can be parsed as JSON, log a warning.
    # This could happen if the tool returns a simple string or number directly.
    logger.warning(f"extract_search_results: Unhandled type or non-JSON content: {type(response_content)}. Content: {str(response_content)[:200]}..."); 
    # Depending on requirements, you might want to wrap non-dict/non-JSON content or return an error.
    # For now, returning an error if it's not a dict.
    return {"status": "error", "message": f"Search result was not a recognized JSON structure. Type: {type(response_content)}"}


def format_search_results_for_prompt(results_data, query, max_results=3):
    if not isinstance(results_data, dict) or results_data.get("status") == "error":
        # If extract_search_results returned an error, or if the data isn't a dict.
        return f"Search for '{query}': {results_data.get('message', 'Error or no valid results structure.')}"
    
    # Assuming the MCP tool (server_search.py) returns a structure compatible with Serper.dev API
    # e.g., {"organic_results": [...]} or similar, directly as a dictionary.
    organic = results_data.get('organic_results', []) 
    
    # Fallback if data is directly a list of results (less likely with Serper but possible for other tools)
    if not organic and isinstance(results_data, list): 
        organic = results_data

    if organic and isinstance(organic, list): # Ensure organic is a list before iterating
        return f"Web search results for '{query}':\n" + "\n".join(f"{i+1}. {item.get('title', 'N/A')}\n   {item.get('snippet', 'N/A')}\n   Source: {item.get('link', 'N/A')}" for i, item in enumerate(organic[:max_results]) if isinstance(item, dict))
    elif not organic:
        return f"Search for '{query}' returned no specific organic results. Data: {str(results_data)[:200]}"
    else: # Organic is not a list or not present as expected
        return f"Search for '{query}' returned data in an unexpected format. Data: {str(results_data)[:200]}"


class ChatMessage(BaseModel):
    role: str; content: str; is_html: Optional[bool] = False; timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
class ChatPayload(BaseModel):
    user_message: str; chat_history: List[ChatMessage]; use_search: bool; conversation_id: Optional[str] = None; ollama_model_name: Optional[str] = None 
class ChatResponse(BaseModel):
    conversation_id: str; chat_history: List[ChatMessage]; ollama_model_name: Optional[str] = None 
class ConversationListItem(BaseModel):
    id: str = Field(alias="_id"); title: Optional[str] = "New Chat"; created_at: datetime; updated_at: datetime; message_count: int; ollama_model_name: Optional[str] = None 
    class Config: populate_by_name = True; json_encoders = {ObjectId: str, datetime: lambda dt: dt.isoformat()}
class RenamePayload(BaseModel):
    new_title: constr(strip_whitespace=True, min_length=1, max_length=100)

async def get_default_ollama_model() -> str:
    try:
        resp = await asyncio.to_thread(ollama.list)
        # Check if resp and resp.models exist and resp.models is a list
        if resp and hasattr(resp, 'models') and isinstance(resp.models, list) and resp.models:
            # Filter for models that have a 'model' attribute (name string)
            valid_models_info = [m for m in resp.models if hasattr(m, 'model') and isinstance(m.model, str) and m.model]
            if not valid_models_info: 
                logger.warning("No valid Ollama models found after filtering.")
            else:
                # Prefer non-embedding models
                non_embed_models = [
                    m.model for m in valid_models_info 
                    if 'embed' not in (m.details.family.lower() if hasattr(m, 'details') and hasattr(m.details, 'family') and m.details.family else "") 
                    and 'embed' not in m.model.lower()
                ]
                if non_embed_models: return non_embed_models[0]
                return valid_models_info[0].model # Fallback to the first valid model if no non-embed found
        logger.warning("No Ollama models found or parsed correctly from ollama.list(). Falling back to default.")
    except Exception as e: 
        logger.warning(f"Could not get Ollama models due to an error: {e}. Falling back to default.", exc_info=False)
    return DEFAULT_OLLAMA_MODEL


async def process_chat_request(payload: ChatPayload) -> ChatResponse:
    if conversations_collection is None: raise HTTPException(status_code=503, detail="MongoDB unavailable.")
    user_msg_content = payload.user_message; conv_id = payload.conversation_id
    llm_history: List[Dict[str, str]] = []; ui_history: List[ChatMessage] = []
    model_name: Optional[str] = None; obj_id: Optional[ObjectId] = None

    if conv_id:
        if not ObjectId.is_valid(conv_id): raise HTTPException(status_code=400, detail="Invalid conv_id.")
        obj_id = ObjectId(conv_id)
        conv = conversations_collection.find_one({"_id": obj_id})
        if conv:
            model_name = conv.get("ollama_model_name")
            for msg_data in conv.get("messages", []):
                if 'role' in msg_data and 'content' in msg_data: 
                    # Use raw_content_for_llm if available, otherwise use content
                    llm_content = msg_data.get("raw_content_for_llm", msg_data["content"])
                    llm_history.append({"role": msg_data["role"], "content": llm_content})
                ui_history.append(ChatMessage(**msg_data)) # Reconstruct ChatMessage for UI history
        else: raise HTTPException(status_code=404, detail=f"Conv ID '{conv_id}' not found.")

    if not model_name: model_name = payload.ollama_model_name or await get_default_ollama_model()
    
    if not conv_id: # New conversation
        new_title = f"Chat: {user_msg_content[:30]}{'...' if len(user_msg_content) > 30 else ''}"
        new_doc = {"title": new_title, "created_at": datetime.now(timezone.utc), "updated_at": datetime.now(timezone.utc), "messages": [], "ollama_model_name": model_name}
        res = conversations_collection.insert_one(new_doc)
        conv_id = str(res.inserted_id); obj_id = res.inserted_id
    elif obj_id and model_name: # Existing conversation, ensure model_name is set if it wasn't
        current_db_model = conversations_collection.find_one({"_id": obj_id}, {"ollama_model_name": 1})
        if current_db_model and not current_db_model.get("ollama_model_name"):
             conversations_collection.update_one({"_id": obj_id}, {"$set": {"ollama_model_name": model_name, "updated_at": datetime.now(timezone.utc)}})


    user_chat_msg = ChatMessage(role="user", content=user_msg_content)
    ui_history.append(user_chat_msg)
    user_msg_to_save = user_chat_msg.model_dump(exclude_none=True)
    # Store the original user message as raw_content_for_llm before any modification by search
    user_msg_to_save["raw_content_for_llm"] = user_msg_content 
    if obj_id: conversations_collection.update_one({"_id": obj_id}, {"$push": {"messages": user_msg_to_save}, "$set": {"updated_at": datetime.now(timezone.utc)}})

    if user_msg_content.lower() == '#clear':
        llm_history.clear() # Clear LLM history for this turn
        assist_msg = ChatMessage(role="assistant", content="Chat context cleared.")
        ui_history.append(assist_msg)
        if obj_id: conversations_collection.update_one({"_id": obj_id}, {"$push": {"messages": assist_msg.model_dump(exclude_none=True)}, "$set": {"updated_at": datetime.now(timezone.utc)}})
        return ChatResponse(conversation_id=conv_id, chat_history=ui_history, ollama_model_name=model_name)

    prompt_llm = user_msg_content; search_html_indicator = None; assist_err_msg_obj = None

    if payload.use_search:
        if not app_state.service_ready:
            logger.warning("[API_CHAT] Search requested but MCP service unavailable.")
            assist_err_msg_obj = ChatMessage(role="assistant", content="⚠️ Web search unavailable at the moment.")
        else:
            logger.info(f"[API_CHAT] Search active for: '{user_msg_content}'")
            try:
                req_id = submit_search_request(user_msg_content) # User's original query
                mcp_resp = wait_for_response(req_id) # This is a dict from response_queue

                if mcp_resp.get("status") == "error": 
                    raise Exception(mcp_resp.get("error", "MCP search tool returned an error status"))
                
                # mcp_resp["data"] is result.content from MCP, expected to be a dict
                extracted_results_data = extract_search_results(mcp_resp.get("data")) 
                
                if extracted_results_data.get("status") == "error": # Check if extract_search_results itself found an issue
                    raise Exception(extracted_results_data.get("message", "Failed to extract or parse search results."))
                
                search_summary_text = format_search_results_for_prompt(extracted_results_data, user_msg_content)
                search_html_indicator = f"<div class='search-indicator-custom'><b>🔍 Web Search:</b> Results for \"{user_msg_content}\" were used.</div>"
                
                # Construct the prompt for the LLM
                prompt_llm = (f"Based on the following web search results for '{user_msg_content}':\n{search_summary_text}\n\n"
                              f"Please answer the user's original question: '{user_msg_content}'")
                
                # Update the stored user message's raw_content_for_llm to include the search context for future turns if needed
                # This might make the raw_content_for_llm very long. Consider if this is the desired behavior.
                # For now, we'll keep raw_content_for_llm as the original user message and prepend search results only for this turn's LLM call.
                # The llm_history is built fresh each time from raw_content_for_llm, so this is fine.

            except Exception as e:
                logger.error(f"[API_CHAT] Search processing error: {e}", exc_info=True)
                assist_err_msg_obj = ChatMessage(role="assistant", content=f"⚠️ Search failed: {str(e)[:100]}")

    if assist_err_msg_obj: # If search failed and generated an error message
        ui_history.append(assist_err_msg_obj)
        if obj_id: conversations_collection.update_one({"_id": obj_id}, {"$push": {"messages": assist_err_msg_obj.model_dump(exclude_none=True)}, "$set": {"updated_at": datetime.now(timezone.utc)}})
        # Do not proceed to LLM call if search was mandatory and failed critically, or let LLM try to respond?
        # For now, we let it fall through, LLM will get the original prompt_llm (user_message) if search failed to modify it.

    # Add the (potentially modified by search) user prompt to LLM history for this turn
    llm_history.append({"role": "user", "content": prompt_llm})
    
    model_resp_content = await chat_with_ollama(llm_history, model_name=model_name)

    if model_resp_content:
        assist_ui_resp = model_resp_content; is_html_resp = False
        if search_html_indicator and not assist_err_msg_obj: # Add search indicator only if search was successful
            assist_ui_resp = f"{search_html_indicator}\n\n{model_resp_content}"; is_html_resp = True
        
        assist_chat_msg = ChatMessage(role="assistant", content=assist_ui_resp, is_html=is_html_resp)
        ui_history.append(assist_chat_msg)
        
        assist_msg_to_save = assist_chat_msg.model_dump(exclude_none=True)
        assist_msg_to_save["raw_content_for_llm"] = model_resp_content # Store LLM's direct response
        if obj_id: conversations_collection.update_one({"_id": obj_id}, {"$push": {"messages": assist_msg_to_save}, "$set": {"updated_at": datetime.now(timezone.utc)}})
    elif not assist_err_msg_obj: # Only add "no response" if no other error message was generated
        llm_err_chat_msg = ChatMessage(role="assistant", content=f"Sorry, I could not get a response from the model ({model_name}).")
        ui_history.append(llm_err_chat_msg)
        if obj_id: conversations_collection.update_one({"_id": obj_id}, {"$push": {"messages": llm_err_chat_msg.model_dump(exclude_none=True)}, "$set": {"updated_at": datetime.now(timezone.utc)}})
    
    return ChatResponse(conversation_id=conv_id, chat_history=ui_history, ollama_model_name=model_name)


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatPayload):
    try: 
        return await process_chat_request(payload)
    except HTTPException: 
        raise 
    except Exception as e: 
        logger.error(f"Error in /api/chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@app.get("/api/status")
async def get_status():
    ollama_ok = False
    try: 
        await asyncio.to_thread(ollama.list)
        ollama_ok = True
    except Exception: 
        pass # ollama_ok remains False
    return {"service_ready": app_state.service_ready, "db_connected": conversations_collection is not None, "ollama_available": ollama_ok}

@app.get("/api/ollama-models", response_model=List[str])
async def list_ollama_models():
    try:
        resp = await asyncio.to_thread(ollama.list)
        if resp and hasattr(resp, 'models') and isinstance(resp.models, list):
            tags = [m.model for m in resp.models if hasattr(m, 'model') and isinstance(m.model, str) and m.model]
            if not tags: 
                logger.warning("ollama.list() returned models, but no valid model tags found after filtering.")
                return [] 
            return sorted(list(set(tags))) # Return unique, sorted list
        logger.warning(f"Unexpected format from ollama.list(): {resp}. Expected .models list.")
        raise HTTPException(status_code=500, detail="Received unexpected format from Ollama API when listing models.")
    except ollama.ResponseError as e: 
        logger.error(f"Ollama API ResponseError: {e.status_code} - {e.error}", exc_info=True)
        raise HTTPException(status_code=e.status_code or 500, detail=f"Ollama API error: {e.error or 'Unknown Ollama API response error'}")
    except ollama.RequestError as e: 
        host = os.getenv('OLLAMA_HOST','localhost:11434')
        actual_host = f"http://{host}" if not host.startswith(('http://','https://')) else host
        logger.error(f"Ollama API RequestError (could not connect to {actual_host}): {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Could not connect to Ollama service at {actual_host}. Ensure Ollama is running.")
    except Exception as e: 
        logger.error(f"An unexpected error occurred while fetching Ollama models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred while fetching Ollama models.")

@app.get("/api/conversations", response_model=List[ConversationListItem], response_model_by_alias=False)
async def list_conversations():
    if conversations_collection is None: raise HTTPException(status_code=503, detail="MongoDB unavailable.")
    try:
        # Exclude messages field from the initial query for efficiency
        cursor = conversations_collection.find({}, {"messages": 0}).sort("updated_at", -1).limit(50)
        convs_data = list(cursor) # Execute query
        
        conv_list_items = []
        default_model_name = None # Lazy load default model name

        for db_conv_doc in convs_data:
            # Efficiently get message count if messages are stored as an array
            # This was previously doing another query per conversation, which is inefficient.
            # If messages are embedded, we can get count from the (excluded) array size if it were projected.
            # For now, assuming we need to query for it if not directly available.
            # A better approach would be to store message_count in the conversation document itself and update it.
            # For simplicity, let's assume a way to get this count. If messages were projected, it'd be len(db_conv_doc.get("messages",[]))
            # Since messages are excluded, we'll do a count query for now.
            # TODO: Optimize message_count by denormalizing it into the conversation document.
            message_count_query = {"_id": db_conv_doc["_id"], "messages.0": {"$exists": True}} # Check if messages array is not empty
            msg_count = conversations_collection.count_documents(message_count_query)

            item_data = {**db_conv_doc, "_id": str(db_conv_doc["_id"]), "message_count": msg_count}
            
            if not item_data.get("ollama_model_name"):
                if default_model_name is None: 
                    default_model_name = await get_default_ollama_model()
                item_data["ollama_model_name"] = default_model_name
            
            conv_list_items.append(ConversationListItem.model_validate(item_data))
        return conv_list_items
    except Exception as e: 
        logger.error(f"Error listing conversations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error listing conversations.")

@app.get("/api/conversations/{conversation_id}", response_model=List[ChatMessage])
async def get_conversation_messages(conversation_id: str):
    if conversations_collection is None: raise HTTPException(status_code=503, detail="MongoDB unavailable.")
    if not ObjectId.is_valid(conversation_id): raise HTTPException(status_code=400, detail="Invalid conversation ID format.")
    try:
        conv = conversations_collection.find_one({"_id": ObjectId(conversation_id)})
        if not conv: raise HTTPException(status_code=404, detail="Conversation not found.")
        # Validate and return messages
        return [ChatMessage.model_validate(msg) for msg in conv.get("messages", []) if isinstance(msg, dict)]
    except HTTPException: raise
    except Exception as e: 
        logger.error(f"Error getting messages for conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error fetching conversation details.")

@app.delete("/api/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(conversation_id: str):
    if conversations_collection is None: raise HTTPException(status_code=503, detail="MongoDB unavailable.")
    if not ObjectId.is_valid(conversation_id): raise HTTPException(status_code=400, detail="Invalid conversation ID format.")
    try:
        result = conversations_collection.delete_one({"_id": ObjectId(conversation_id)})
        if result.deleted_count == 0: 
            raise HTTPException(status_code=404, detail="Conversation not found for deletion.")
        logger.info(f"Deleted conversation ID: {conversation_id}")
        return Response(status_code=status.HTTP_204_NO_CONTENT) # FastAPI handles this automatically for 204
    except HTTPException: raise
    except Exception as e: 
        logger.error(f"Error deleting conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error deleting conversation.")

@app.put("/api/conversations/{conversation_id}/rename", response_model=ConversationListItem, response_model_by_alias=False)
async def rename_conversation_title(conversation_id: str, payload: RenamePayload):
    if conversations_collection is None: raise HTTPException(status_code=503, detail="MongoDB unavailable.")
    if not ObjectId.is_valid(conversation_id): raise HTTPException(status_code=400, detail="Invalid conversation ID format.")
    
    obj_id = ObjectId(conversation_id)
    try:
        # Check if conversation exists before update
        if conversations_collection.count_documents({"_id": obj_id}) == 0:
            raise HTTPException(status_code=404, detail="Conversation not found for renaming.")

        update_result = conversations_collection.update_one(
            {"_id": obj_id}, 
            {"$set": {"title": payload.new_title, "updated_at": datetime.now(timezone.utc)}}
        )
        
        if update_result.matched_count == 0: # Should be caught by pre-check, but as safeguard
             raise HTTPException(status_code=404, detail="Conversation not found during update operation.")

        updated_conv_doc = conversations_collection.find_one({"_id": obj_id}) # Fetch the updated document
        if not updated_conv_doc: # Should not happen if update was successful
            logger.error(f"Failed to retrieve conversation {conversation_id} after rename.")
            raise HTTPException(status_code=500, detail="Failed to retrieve updated conversation details.")

        # Get message count (consistent with list_conversations)
        msg_count = conversations_collection.count_documents({"_id": updated_conv_doc["_id"], "messages.0": {"$exists": True}})
        
        item_data = {**updated_conv_doc, "_id": str(updated_conv_doc["_id"]), "message_count": msg_count}
        if not item_data.get("ollama_model_name"): 
            item_data["ollama_model_name"] = await get_default_ollama_model() # Ensure model name is present

        logger.info(f"Renamed conversation ID {conversation_id} to '{payload.new_title}'")
        return ConversationListItem.model_validate(item_data)
    except HTTPException: raise
    except Exception as e: 
        logger.error(f"Error renaming conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error renaming conversation.")

frontend_dist_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'dist')
if os.path.exists(frontend_dist_path):
    logger.info(f"Serving static files from: {frontend_dist_path}")
    app.mount("/", StaticFiles(directory=frontend_dist_path, html=True), name="static_frontend")
else:
    logger.warning(f"Frontend build directory not found: {frontend_dist_path}. Run 'npm run build' in 'frontend' directory.")

if __name__ == "__main__":
    logger.info(f"Starting Uvicorn for {__name__}. MCP service startup will be handled by FastAPI's lifespan manager.")
    
    def graceful_shutdown_handler(sig, frame):
        logger.info(f"Signal {signal.Signals(sig).name} received, initiating graceful shutdown...")
        # FastAPI's lifespan manager should handle task cancellation and resource cleanup.
        # For Uvicorn, sending SIGINT/SIGTERM usually triggers its own graceful shutdown.
        # This custom handler is more for ensuring our logging captures it.
        # If Uvicorn doesn't shut down cleanly, further steps might be needed here.
        
        # Request Uvicorn to shut down (if running in a way that this helps)
        # This is complex as Uvicorn runs in its own loop.
        # The primary mechanism is that lifespan's exit handler will be called.
        sys.exit(0) # This will interrupt Uvicorn if it's the main thread

    signal.signal(signal.SIGINT, graceful_shutdown_handler)
    signal.signal(signal.SIGTERM, graceful_shutdown_handler)
    
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
