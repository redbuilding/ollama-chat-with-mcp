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
import time 
import queue

# --- Environment Setup ---
MCP_SERVER_SCRIPT = "server_search.py"
_main_py_dir = os.path.dirname(os.path.abspath(__file__)) # Directory of main.py (backend)
MCP_SERVER_SCRIPT_ABS_PATH = os.path.join(_main_py_dir, MCP_SERVER_SCRIPT)
PROJECT_ROOT_DIR = os.path.dirname(_main_py_dir) # Parent directory of backend (e.g., /Users/qasim/Documents/ai/mcp)

if not os.path.exists(MCP_SERVER_SCRIPT_ABS_PATH):
    logging.error(f"CRITICAL: MCP server script '{MCP_SERVER_SCRIPT_ABS_PATH}' not found. Searched in directory: '{_main_py_dir}'. Search functionality will be disabled.")
else:
    logging.info(f"MCP server script found at: {MCP_SERVER_SCRIPT_ABS_PATH}")
logging.info(f"Project root directory determined as: {PROJECT_ROOT_DIR}")


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
    logger.info("MCP_SERVICE_LOOP: Starting...")
    if not os.path.exists(MCP_SERVER_SCRIPT_ABS_PATH):
        logger.error(f"MCP_SERVICE_LOOP: MCP service cannot start: script '{MCP_SERVER_SCRIPT_ABS_PATH}' not found.")
        app_state.service_ready = False
        return

    mcp_client_comms_logger = logger.getChild("mcp_client_comms")
    mcp_client_comms_logger.setLevel(logging.DEBUG) 

    server_script_dir = os.path.dirname(MCP_SERVER_SCRIPT_ABS_PATH) 

    # Prepare environment for the subprocess
    subproc_env = os.environ.copy()
    # Prepend project root to PYTHONPATH to help find project-local 'mcp' library
    existing_pythonpath = subproc_env.get("PYTHONPATH")
    new_pythonpath = PROJECT_ROOT_DIR
    if existing_pythonpath:
        new_pythonpath = f"{PROJECT_ROOT_DIR}{os.pathsep}{existing_pythonpath}"
    subproc_env["PYTHONPATH"] = new_pythonpath
    logger.info(f"MCP_SERVICE_LOOP: Subprocess PYTHONPATH set to: {new_pythonpath}")


    while True: 
        try:
            logger.info(f"MCP_SERVICE_LOOP: Setting CWD for subprocess to: {server_script_dir}")
            server_params_for_client = StdioServerParameters(
                command=sys.executable, 
                args=[MCP_SERVER_SCRIPT_ABS_PATH], 
                env=subproc_env, # Pass the modified environment
                cwd=server_script_dir 
            )
            logger.info(f"MCP_SERVICE_LOOP: Attempting to start MCP server using stdio_client with: command='{sys.executable}', args=['{MCP_SERVER_SCRIPT_ABS_PATH}'], cwd='{server_script_dir}'")
            
            async with stdio_client(server_params_for_client, logger=mcp_client_comms_logger) as (read, write):
                logger.info("MCP_SERVICE_LOOP: stdio_client context entered (implies subprocess started and pipes connected). Initializing ClientSession...")
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
                        else:
                            app_state.service_ready = True
                            logger.info("MCP_SERVICE_LOOP: MCP service fully initialized and 'web_search' tool is available.")
                    else:
                        logger.error(f"MCP_SERVICE_LOOP: CRITICAL: No tools found or invalid/empty response from session.list_tools(). Response object: {tools_response!r}")
                        app_state.service_ready = False

                    if app_state.service_ready:
                        while True: 
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
                            await asyncio.sleep(0.1) 
        
        except ConnectionRefusedError as e_conn_refused: 
            logger.error(f"MCP_SERVICE_LOOP: ConnectionRefusedError (unexpected for stdio): {e_conn_refused}", exc_info=True) 
        except asyncio.TimeoutError as e_timeout: 
            logger.error(f"MCP_SERVICE_LOOP: TimeoutError in MCP service communication: {e_timeout}", exc_info=True)
        except types.MCPError as e_mcp_protocol: 
            logger.error(f"MCP_SERVICE_LOOP: MCPError (protocol error): {e_mcp_protocol}", exc_info=True)
        except Exception as e_generic: 
            logger.error(f"MCP_SERVICE_LOOP: Generic Exception during MCP service setup/connection: {e_generic}", exc_info=True) 
        finally:
            app_state.service_ready = False 
            logger.info("MCP_SERVICE_LOOP: Connection lost or failed. Will attempt to reconnect after a 10s delay.")
            await asyncio.sleep(10) 


# --- FastAPI Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI Lifespan: Startup sequence initiated.")
    app_state.mcp_task = asyncio.create_task(mcp_service_loop())
    logger.info("FastAPI Lifespan: MCP service task created.")
    yield 
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
        time.sleep(0.5)
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

def extract_search_results(response_content):
    if isinstance(response_content, dict): return response_content
    elif hasattr(response_content, 'text') and isinstance(response_content.text, str):
        try: return json.loads(response_content.text)
        except json.JSONDecodeError as e: logger.error(f"extract_search_results: JSONDecodeError TextContent: {e}. Content: {response_content.text[:200]}..."); return {"status": "error", "message": "Failed to parse search JSON."}
    elif isinstance(response_content, str): 
        try: return json.loads(response_content)
        except json.JSONDecodeError as e: logger.error(f"extract_search_results: JSONDecodeError string: {e}. Content: {response_content[:200]}..."); return {"status": "error", "message": "Failed to parse search JSON string."}
    logger.warning(f"extract_search_results: Unhandled type: {type(response_content)}. Content: {str(response_content)[:200]}..."); return {"status": "error", "message": f"Unhandled search result type: {type(response_content)}"}

def format_search_results_for_prompt(results_data, query, max_results=3):
    if not isinstance(results_data, dict) or results_data.get("status") == "error":
        return f"Search for '{query}': {results_data.get('message', 'Error or no results.')}"
    organic = results_data.get('organic_results', []) 
    if organic:
        return f"Web search results for '{query}':\n" + "\n".join(f"{i+1}. {item.get('title', 'N/A')}\n   {item.get('snippet', 'N/A')}\n   Source: {item.get('link', 'N/A')}" for i, item in enumerate(organic[:max_results]))
    return f"Search for '{query}' returned no specific organic results."

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
        if resp and hasattr(resp, 'models') and isinstance(resp.models, list) and resp.models:
            valid = [m for m in resp.models if hasattr(m, 'model') and m.model]
            if not valid: logger.warning("No valid Ollama models found.")
            else:
                non_embed = [m.model for m in valid if 'embed' not in (m.details.family.lower() if m.details and hasattr(m.details, 'family') and m.details.family else "") and 'embed' not in m.model.lower()]
                if non_embed: return non_embed[0]
                return valid[0].model 
        logger.warning("No Ollama models found/parsed. Falling back to default.")
    except Exception as e: logger.warning(f"Could not get Ollama models: {e}. Falling back to default.", exc_info=False)
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
                if 'role' in msg_data and 'content' in msg_data: llm_history.append({"role": msg_data["role"], "content": msg_data.get("raw_content_for_llm", msg_data["content"])})
                ui_history.append(ChatMessage(**msg_data))
        else: raise HTTPException(status_code=404, detail=f"Conv ID '{conv_id}' not found.")

    if not model_name: model_name = payload.ollama_model_name or await get_default_ollama_model()
    
    if not conv_id: 
        new_title = f"Chat: {user_msg_content[:30]}{'...' if len(user_msg_content) > 30 else ''}"
        new_doc = {"title": new_title, "created_at": datetime.now(timezone.utc), "updated_at": datetime.now(timezone.utc), "messages": [], "ollama_model_name": model_name}
        res = conversations_collection.insert_one(new_doc)
        conv_id = str(res.inserted_id); obj_id = res.inserted_id
    elif obj_id and conv and not conv.get("ollama_model_name"): 
        conversations_collection.update_one({"_id": obj_id}, {"$set": {"ollama_model_name": model_name, "updated_at": datetime.now(timezone.utc)}})

    user_chat_msg = ChatMessage(role="user", content=user_msg_content)
    ui_history.append(user_chat_msg)
    user_msg_to_save = user_chat_msg.model_dump(exclude_none=True)
    user_msg_to_save["raw_content_for_llm"] = user_msg_content
    if obj_id: conversations_collection.update_one({"_id": obj_id}, {"$push": {"messages": user_msg_to_save}, "$set": {"updated_at": datetime.now(timezone.utc)}})

    if user_msg_content.lower() == '#clear':
        llm_history.clear()
        assist_msg = ChatMessage(role="assistant", content="Chat context cleared.")
        ui_history.append(assist_msg)
        if obj_id: conversations_collection.update_one({"_id": obj_id}, {"$push": {"messages": assist_msg.model_dump(exclude_none=True)}, "$set": {"updated_at": datetime.now(timezone.utc)}})
        return ChatResponse(conversation_id=conv_id, chat_history=ui_history, ollama_model_name=model_name)

    prompt_llm = user_msg_content; search_html_indicator = None; assist_err_msg_obj = None

    if payload.use_search:
        if not app_state.service_ready:
            logger.warning("[API_CHAT] Search requested but MCP service unavailable.")
            assist_err_msg_obj = ChatMessage(role="assistant", content="⚠️ Web search unavailable.")
        else:
            logger.info(f"[API_CHAT] Search active for: '{user_msg_content}'")
            try:
                req_id = submit_search_request(user_msg_content)
                mcp_resp = wait_for_response(req_id)
                if mcp_resp.get("status") == "error": raise Exception(mcp_resp.get("error", "MCP search error"))
                extracted = extract_search_results(mcp_resp.get("data")) 
                if extracted.get("status") == "error": raise Exception(extracted.get("message", "Search extraction error."))
                search_txt = format_search_results_for_prompt(extracted, user_msg_content)
                search_html_indicator = f"<div class='search-indicator-custom'><b>🔍 Web Search:</b> Results for \"{user_msg_content}\" used.</div>"
                prompt_llm = (f"Search results for '{user_msg_content}':\n{search_txt}\n\nUsing these, answer: '{user_msg_content}'")
                if obj_id: conversations_collection.update_one({"_id": obj_id, "messages.timestamp": user_chat_msg.timestamp}, {"$set": {"messages.$.raw_content_for_llm": prompt_llm}})
            except Exception as e:
                logger.error(f"[API_CHAT] Search error: {e}", exc_info=True)
                assist_err_msg_obj = ChatMessage(role="assistant", content=f"⚠️ Search failed: {str(e)[:100]}")

    if assist_err_msg_obj:
        ui_history.append(assist_err_msg_obj)
        if obj_id: conversations_collection.update_one({"_id": obj_id}, {"$push": {"messages": assist_err_msg_obj.model_dump(exclude_none=True)}, "$set": {"updated_at": datetime.now(timezone.utc)}})

    llm_history.append({"role": "user", "content": prompt_llm})
    model_resp_content = await chat_with_ollama(llm_history, model_name=model_name)

    if model_resp_content:
        assist_ui_resp = model_resp_content; is_html_resp = False
        if search_html_indicator and not assist_err_msg_obj:
            assist_ui_resp = f"{search_html_indicator}\n\n{model_resp_content}"; is_html_resp = True
        assist_chat_msg = ChatMessage(role="assistant", content=assist_ui_resp, is_html=is_html_resp)
        ui_history.append(assist_chat_msg)
        assist_msg_to_save = assist_chat_msg.model_dump(exclude_none=True)
        assist_msg_to_save["raw_content_for_llm"] = model_resp_content 
        if obj_id: conversations_collection.update_one({"_id": obj_id}, {"$push": {"messages": assist_msg_to_save}, "$set": {"updated_at": datetime.now(timezone.utc)}})
    elif not assist_err_msg_obj:
        llm_err_chat_msg = ChatMessage(role="assistant", content=f"Sorry, no response from model {model_name}.")
        ui_history.append(llm_err_chat_msg)
        if obj_id: conversations_collection.update_one({"_id": obj_id}, {"$push": {"messages": llm_err_chat_msg.model_dump(exclude_none=True)}, "$set": {"updated_at": datetime.now(timezone.utc)}})
    return ChatResponse(conversation_id=conv_id, chat_history=ui_history, ollama_model_name=model_name)

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatPayload):
    try: return await process_chat_request(payload)
    except HTTPException: raise 
    except Exception as e: logger.error(f"Error /api/chat: {e}", exc_info=True); raise HTTPException(status_code=500, detail="Internal error.")

@app.get("/api/status")
async def get_status():
    ollama_ok = False
    try: await asyncio.to_thread(ollama.list); ollama_ok = True
    except Exception: pass
    return {"service_ready": app_state.service_ready, "db_connected": conversations_collection is not None, "ollama_available": ollama_ok}

@app.get("/api/ollama-models", response_model=List[str])
async def list_ollama_models():
    try:
        resp = await asyncio.to_thread(ollama.list)
        if resp and hasattr(resp, 'models') and isinstance(resp.models, list):
            tags = [m.model for m in resp.models if hasattr(m, 'model') and m.model]
            if not tags: logger.warning("No Ollama models found."); return [] 
            return tags
        logger.warning(f"Bad Ollama API format: {resp}"); raise HTTPException(status_code=500, detail="Bad Ollama format.")
    except ollama.ResponseError as e: logger.error(f"Ollama API error: {e}", exc_info=True); raise HTTPException(status_code=e.status_code or 500, detail=f"Ollama API error: {e.error}")
    except ollama.RequestError as e: 
        host = os.getenv('OLLAMA_HOST','localhost:11434'); actual_host = f"http://{host}" if not host.startswith(('http://','https://')) else host
        logger.error(f"Ollama connection error (tried {actual_host}): {e}", exc_info=True); raise HTTPException(status_code=503, detail=f"Ollama connection error (tried {actual_host}).")
    except Exception as e: logger.error(f"Error fetching Ollama models: {e}", exc_info=True); raise HTTPException(status_code=500, detail="Error fetching models.")

@app.get("/api/conversations", response_model=List[ConversationListItem], response_model_by_alias=False)
async def list_conversations():
    if conversations_collection is None: raise HTTPException(status_code=503, detail="MongoDB unavailable.")
    try:
        cursor = conversations_collection.find({}, {"messages": 0}).sort("updated_at", -1).limit(50)
        convs = []
        default_model = None
        for db_c in cursor:
            msg_count = conversations_collection.count_documents({"_id": db_c["_id"], "messages.0": {"$exists": True}})
            item = {**db_c, "_id": str(db_c["_id"]), "message_count": msg_count}
            if not item.get("ollama_model_name"):
                if default_model is None: default_model = await get_default_ollama_model()
                item["ollama_model_name"] = default_model
            convs.append(ConversationListItem.model_validate(item))
        return convs
    except Exception as e: logger.error(f"Error listing convs: {e}", exc_info=True); raise HTTPException(status_code=500, detail="Error listing convs.")

@app.get("/api/conversations/{conversation_id}", response_model=List[ChatMessage])
async def get_conversation_messages(conversation_id: str):
    if conversations_collection is None: raise HTTPException(status_code=503, detail="MongoDB unavailable.")
    if not ObjectId.is_valid(conversation_id): raise HTTPException(status_code=400, detail="Invalid ID.")
    try:
        conv = conversations_collection.find_one({"_id": ObjectId(conversation_id)})
        if not conv: raise HTTPException(status_code=404, detail="Not found.")
        return [ChatMessage.model_validate(msg) for msg in conv.get("messages", [])]
    except Exception as e: logger.error(f"Error get conv {conversation_id}: {e}", exc_info=True); raise HTTPException(status_code=500, detail="Error fetching details.")

@app.delete("/api/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(conversation_id: str):
    if conversations_collection is None: raise HTTPException(status_code=503, detail="MongoDB unavailable.")
    if not ObjectId.is_valid(conversation_id): raise HTTPException(status_code=400, detail="Invalid ID.")
    try:
        if conversations_collection.delete_one({"_id": ObjectId(conversation_id)}).deleted_count == 0: raise HTTPException(status_code=404, detail="Not found.")
        logger.info(f"Deleted conv ID: {conversation_id}"); return Response(status_code=status.HTTP_204_NO_CONTENT)
    except Exception as e: logger.error(f"Error delete conv {conversation_id}: {e}", exc_info=True); raise HTTPException(status_code=500, detail="Error deleting.")

@app.put("/api/conversations/{conversation_id}/rename", response_model=ConversationListItem, response_model_by_alias=False)
async def rename_conversation_title(conversation_id: str, payload: RenamePayload):
    if conversations_collection is None: raise HTTPException(status_code=503, detail="MongoDB unavailable.")
    if not ObjectId.is_valid(conversation_id): raise HTTPException(status_code=400, detail="Invalid ID.")
    obj_id = ObjectId(conversation_id)
    try:
        if not conversations_collection.find_one({"_id": obj_id}, {"_id": 1}): raise HTTPException(status_code=404, detail="Not found.")
        conversations_collection.update_one({"_id": obj_id}, {"$set": {"title": payload.new_title, "updated_at": datetime.now(timezone.utc)}})
        updated_c = conversations_collection.find_one({"_id": obj_id})
        if not updated_c: raise HTTPException(status_code=500, detail="Failed to get updated conv.")
        msg_c = conversations_collection.count_documents({"_id": updated_c["_id"], "messages.0": {"$exists": True}})
        item = {**updated_c, "_id": str(updated_c["_id"]), "message_count": msg_c}
        if not item.get("ollama_model_name"): item["ollama_model_name"] = await get_default_ollama_model()
        logger.info(f"Renamed conv ID {conversation_id} to '{payload.new_title}'")
        return ConversationListItem.model_validate(item)
    except Exception as e: logger.error(f"Error rename conv {conversation_id}: {e}", exc_info=True); raise HTTPException(status_code=500, detail="Error renaming.")

frontend_dist_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'dist')
if os.path.exists(frontend_dist_path):
    logger.info(f"Serving static files from: {frontend_dist_path}")
    app.mount("/", StaticFiles(directory=frontend_dist_path, html=True), name="static_frontend")
else:
    logger.warning(f"Frontend build directory not found: {frontend_dist_path}. Run 'npm run build' in 'frontend'.")

if __name__ == "__main__":
    logger.info(f"Starting Uvicorn for {__name__}. MCP service startup via FastAPI lifespan.")
    def handle_sig(sig, frame):
        logger.info(f"Signal {sig} received, shutting down...")
        if app_state.mcp_task and not app_state.mcp_task.done(): app_state.mcp_task.cancel()
        if mongo_client: mongo_client.close()
        sys.exit(0)
    signal.signal(signal.SIGINT, handle_sig); signal.signal(signal.SIGTERM, handle_sig)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
