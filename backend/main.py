import asyncio
import json
import re # Added for SQL extraction
import logging
import os
import signal
import sys
import time # Keep for time.time() but not time.sleep()
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple, Union
from contextlib import asynccontextmanager # For lifespan events
from dataclasses import dataclass, field

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
from mcp.common.model.uri import Uri # Corrected import for Uri
from fastmcp.client.transports import StdioServerParameters, stdio_client
# from mcp.common.content import TextContent # For type checking if needed

import ollama
# import asyncio # Duplicate import, removed for cleanliness

# --- Environment Setup ---
_main_py_dir = os.path.dirname(os.path.abspath(__file__))

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/mcp_backend_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mcp_backend")
logger.setLevel(logging.INFO)

# --- Constants ---
WEB_SEARCH_SERVICE_NAME = "web_search_service"
MYSQL_DB_SERVICE_NAME = "mysql_db_service"
MAX_DB_RESULT_CHARS = 5000 # Proxy for token limit (approx 1000-1200 tokens)
# For more accurate token counting, consider using a library like tiktoken:
# import tiktoken
# def count_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
#     try:
#         encoding = tiktoken.encoding_for_model(model_name)
#     except KeyError:
#         encoding = tiktoken.get_encoding("cl100k_base")
#     return len(encoding.encode(text))

# --- MongoDB Setup ---
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DATABASE_NAME = os.getenv('MONGODB_DATABASE_NAME', 'mcp_chat_db')
MONGODB_COLLECTION_NAME = os.getenv('MONGODB_COLLECTION_NAME', 'conversations')
DEFAULT_OLLAMA_MODEL = os.getenv('DEFAULT_OLLAMA_MODEL', 'qwen2:7b') # Example, choose your default

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

# --- MCP Service Configuration ---
@dataclass
class MCPServiceConfig:
    name: str
    script_name: str # e.g., "server_search.py"
    command_verb: str = "run" # e.g., "run" for `fastmcp run ...`
    required_tools: List[str] = field(default_factory=list) # Tools are still useful for service readiness
    # Resources are typically identified by URI and don't need explicit listing here for readiness
    enabled: bool = True # Allows disabling a service via config if needed

    @property
    def full_command(self) -> List[str]:
        # Assumes mcp server scripts are in the same directory as main.py or a known subdir
        # For now, let's assume they are in the same directory (_main_py_dir)
        return ["fastmcp", self.command_verb, os.path.join(_main_py_dir, self.script_name)]

# --- State Management ---
class AppState:
    def __init__(self):
        self.current_date = datetime.now(timezone.utc)
        self.mcp_tasks: Dict[str, asyncio.Task] = {}
        self.mcp_service_queues: Dict[str, Tuple[asyncio.Queue, asyncio.Queue]] = {}
        self.mcp_service_ready: Dict[str, bool] = {}
        self.mcp_configs: Dict[str, MCPServiceConfig] = {
            WEB_SEARCH_SERVICE_NAME: MCPServiceConfig(
                name=WEB_SEARCH_SERVICE_NAME,
                script_name="server_search.py", # Ensure this script is in _main_py_dir
                required_tools=["web_search"]
            ),
            MYSQL_DB_SERVICE_NAME: MCPServiceConfig(
                name=MYSQL_DB_SERVICE_NAME,
                script_name="server_mysql.py", # Ensure this script is in _main_py_dir
                required_tools=["execute_sql_query_tool"], # Still list tools for general service health
                # Resources like "resource://tables" will be called directly
                # enabled=False # Example: Can be disabled here if not ready for use
            ),
        }

    @property
    def formatted_date(self): return f"{self.current_date.strftime('%B')} {self.current_date.strftime('%d').lstrip('0')}, {self.current_date.strftime('%Y')}"
    # ... other date properties

app_state = AppState()

# --- MCP Service Loop (Generalized) ---
async def run_mcp_service_instance(config: MCPServiceConfig):
    service_name = config.name
    logger.info(f"MCP_SERVICE ({service_name}): Starting STDIO client loop...")
    app_state.mcp_service_ready[service_name] = False

    # Create queues for this service instance
    request_q = asyncio.Queue()
    response_q = asyncio.Queue()
    app_state.mcp_service_queues[service_name] = (request_q, response_q)

    server_params = StdioServerParameters(
        command=config.full_command[0],
        args=config.full_command[1:],
        cwd=_main_py_dir # Run from backend directory
    )

    while True: # Outer loop for reconnections
        try:
            logger.info(f"MCP_SERVICE ({service_name}): Launching subprocess with params: {server_params}")
            async with stdio_client(server_params) as streams:
                read_stream, write_stream = streams
                logger.info(f"MCP_SERVICE ({service_name}): Connected to subprocess. Initializing ClientSession...")

                async with ClientSession(read_stream, write_stream) as session:
                    logger.info(f"MCP_SERVICE ({service_name}): ClientSession created. Initializing session...")
                    await session.initialize()
                    logger.info(f"MCP_SERVICE ({service_name}): Session initialized. Listing tools...")
                    
                    tools_response = await session.list_tools()
                    logger.debug(f"MCP_SERVICE ({service_name}): Raw tools_response: {tools_response!r}")

                    if tools_response and tools_response.tools is not None:
                        available_tool_names = [tool.name for tool in tools_response.tools]
                        logger.info(f"MCP_SERVICE ({service_name}): Available tools: {available_tool_names}")
                        
                        all_required_found = True
                        if config.required_tools: # Check required tools for basic service health
                            for req_tool in config.required_tools:
                                if req_tool not in available_tool_names:
                                    logger.error(f"MCP_SERVICE ({service_name}): CRITICAL: Required tool '{req_tool}' NOT FOUND. Available: {available_tool_names}")
                                    all_required_found = False
                                    break
                        
                        if all_required_found:
                            app_state.mcp_service_ready[service_name] = True
                            logger.info(f"MCP_SERVICE ({service_name}): Service fully initialized and all required tools are available.")
                        else:
                            app_state.mcp_service_ready[service_name] = False
                    else:
                        logger.error(f"MCP_SERVICE ({service_name}): CRITICAL: No tools found or invalid response from list_tools(). Response: {tools_response!r}")
                        app_state.mcp_service_ready[service_name] = False

                    if app_state.mcp_service_ready.get(service_name, False):
                        while True: # Inner loop for processing requests
                            try:
                                request_data = request_q.get_nowait()
                                request_id = request_data["id"]
                                request_type = request_data.get("type", "tool") # Default to "tool"

                                try:
                                    start_time = time.time()
                                    if request_type == "tool":
                                        tool_to_call = request_data["tool"]
                                        params = request_data["params"]
                                        logger.debug(f"MCP_SERVICE ({service_name}): Calling TOOL '{tool_to_call}' with params: {params} (req_id: {request_id})")
                                        result = await session.call_tool(tool_to_call, params)
                                        duration = time.time() - start_time
                                        logger.info(f"MCP_SERVICE ({service_name}): TOOL '{tool_to_call}' completed in {duration:.2f}s (req_id: {request_id})")
                                        logger.debug(f"MCP_SERVICE ({service_name}): TOOL '{tool_to_call}' result content type: {type(result.content)}. Content preview: {str(result.content)[:200]}")
                                        await response_q.put({"id": request_id, "status": "success", "data": result.content})
                                    
                                    elif request_type == "resource":
                                        uri_to_get = request_data["uri"]
                                        logger.debug(f"MCP_SERVICE ({service_name}): Getting RESOURCE '{uri_to_get}' (req_id: {request_id})")
                                        # Ensure URI is valid if necessary, though MCP client might handle this
                                        # validated_uri = Uri(uri_to_get) # Example validation
                                        resource_content = await session.get_resource_content(Uri(uri_to_get))
                                        duration = time.time() - start_time
                                        logger.info(f"MCP_SERVICE ({service_name}): RESOURCE '{uri_to_get}' completed in {duration:.2f}s (req_id: {request_id})")
                                        logger.debug(f"MCP_SERVICE ({service_name}): RESOURCE '{uri_to_get}' result content type: {type(resource_content)}. Content preview: {str(resource_content)[:200]}")
                                        # get_resource_content directly returns the content (e.g., a string)
                                        await response_q.put({"id": request_id, "status": "success", "data": resource_content})
                                    else:
                                        logger.error(f"MCP_SERVICE ({service_name}): Unknown request type '{request_type}' (req_id: {request_id})")
                                        await response_q.put({"id": request_id, "status": "error", "error": f"Unknown request type: {request_type}"})

                                except Exception as e_mcp_call:
                                    call_target = request_data.get("tool", request_data.get("uri", "N/A"))
                                    logger.error(f"MCP_SERVICE ({service_name}): Error in '{request_type}' call to '{call_target}' (req_id: {request_id}): {e_mcp_call}", exc_info=True)
                                    await response_q.put({"id": request_id, "status": "error", "error": str(e_mcp_call)})
                                request_q.task_done()
                            except asyncio.QueueEmpty:
                                await asyncio.sleep(0.01) # Brief pause if queue is empty
                            except Exception as e_queue:
                                logger.error(f"MCP_SERVICE ({service_name}): Error processing request queue: {e_queue}", exc_info=True)
                                await asyncio.sleep(0.1) # Avoid tight loop on unexpected queue errors

        except FileNotFoundError:
            logger.error(f"MCP_SERVICE ({service_name}): Command for script '{config.script_name}' not found. Ensure FastMCP is installed and script path is correct.", exc_info=True)
        except asyncio.TimeoutError as e_timeout:
            logger.error(f"MCP_SERVICE ({service_name}): TimeoutError in communication: {e_timeout}", exc_info=True)
        except Exception as e_generic:
            logger.error(f"MCP_SERVICE ({service_name}): Generic Exception (subprocess might have failed): {e_generic}", exc_info=True)
        finally:
            app_state.mcp_service_ready[service_name] = False
            logger.info(f"MCP_SERVICE ({service_name}): Connection lost or subprocess ended. Will attempt to reconnect after 10s.")
            await asyncio.sleep(10)


# --- MCP Interaction Helpers ---
async def submit_mcp_tool_request(service_name: str, tool_name: str, params: Dict, request_id_prefix: str = "tool_req") -> str:
    if service_name not in app_state.mcp_service_queues:
        raise HTTPException(status_code=503, detail=f"MCP service '{service_name}' is not available or queues not initialized.")
    
    request_q, _ = app_state.mcp_service_queues[service_name]
    request_id = f"{request_id_prefix}_{service_name}_{tool_name}_{time.time()}"
    await request_q.put({"id": request_id, "type": "tool", "tool": tool_name, "params": params})
    return request_id

async def submit_mcp_resource_request(service_name: str, resource_uri: str, request_id_prefix: str = "res_req") -> str:
    if service_name not in app_state.mcp_service_queues:
        raise HTTPException(status_code=503, detail=f"MCP service '{service_name}' is not available or queues not initialized.")
    
    request_q, _ = app_state.mcp_service_queues[service_name]
    # Sanitize URI for request_id to avoid issues if URI has special chars, though time.time() makes it unique
    safe_uri_part = re.sub(r'[^a-zA-Z0-9_-]', '', resource_uri.split("://")[-1]) # Basic sanitization
    request_id = f"{request_id_prefix}_{service_name}_{safe_uri_part}_{time.time()}"
    await request_q.put({"id": request_id, "type": "resource", "uri": resource_uri})
    return request_id

async def wait_mcp_response(service_name: str, request_id: str, timeout: int = 45) -> Dict:
    if service_name not in app_state.mcp_service_queues:
        return {"id": request_id, "status": "error", "error": f"MCP service '{service_name}' queues not found during response wait."}

    _, response_q = app_state.mcp_service_queues[service_name]
    start_time = time.time()
    try:
        while time.time() - start_time < timeout:
            try:
                item = await asyncio.wait_for(response_q.get(), timeout=0.1)
                if item.get("id") == request_id:
                    response_q.task_done()
                    return item
                else:
                    logger.warning(f"MCP_RESPONSE_WAIT ({service_name}): Received item for unexpected ID {item.get('id')}, expected {request_id}. Re-queuing.")
                    await response_q.put(item) 
            except asyncio.TimeoutError: 
                pass 
            except asyncio.QueueEmpty: 
                pass
    except Exception as e:
        logger.error(f"MCP_RESPONSE_WAIT ({service_name}): Error waiting for response for {request_id}: {e}", exc_info=True)
        return {"id": request_id, "status": "error", "error": f"Exception while waiting for MCP response: {str(e)}"}

    return {"id": request_id, "status": "error", "error": "Request timed out"}


# --- FastAPI Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI Lifespan: Startup sequence initiated.")
    for service_name, config in app_state.mcp_configs.items():
        if config.enabled:
            task = asyncio.create_task(run_mcp_service_instance(config))
            app_state.mcp_tasks[service_name] = task
            logger.info(f"FastAPI Lifespan: MCP service task created for '{service_name}'.")
        else:
            logger.info(f"FastAPI Lifespan: MCP service '{service_name}' is disabled by configuration.")
            app_state.mcp_service_ready[service_name] = False
            if service_name in app_state.mcp_service_queues:
                del app_state.mcp_service_queues[service_name]


    yield # Application runs here

    logger.info("FastAPI Lifespan: Shutdown sequence initiated.")
    for service_name, task in app_state.mcp_tasks.items():
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.info(f"FastAPI Lifespan: MCP service task for '{service_name}' successfully cancelled.")
            except Exception as e:
                logger.error(f"FastAPI Lifespan: Error during MCP service task '{service_name}' shutdown: {e}", exc_info=True)
    
    if mongo_client:
        mongo_client.close()
        logger.info("FastAPI Lifespan: MongoDB connection closed.")
    logger.info("FastAPI Lifespan: Shutdown complete.")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], # Adjust if your frontend runs elsewhere
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


# --- Ollama and Chat Logic ---
async def chat_with_ollama(messages: List[Dict[str, str]], model_name: str) -> Optional[str]:
    try:
        valid_messages = [msg for msg in messages if isinstance(msg, dict) and 'role' in msg and 'content' in msg]
        if not valid_messages:
            logger.error(f"[Ollama] No valid messages provided to model '{model_name}'.")
            return None

        response = await asyncio.to_thread(ollama.chat, model=model_name, messages=valid_messages)
        if response and "message" in response and "content" in response["message"]:
            return response["message"]["content"]
        logger.warning(f"[Ollama] Unexpected response structure from model '{model_name}': {response}")
        return None
    except Exception as e:
        logger.error(f"[Ollama] Error with model '{model_name}': {e}", exc_info=True)
        return None

def extract_search_results(response_content: Any) -> Dict:
    # This function is generally for parsing JSON content that might come from MCP tools/resources
    logger.debug(f"extract_search_results: Input type: {type(response_content)}, content preview: {str(response_content)[:200]}")
    if isinstance(response_content, dict): return response_content # Already a dict
    
    # If response_content is a string (likely JSON from a resource or TextContent)
    if isinstance(response_content, str):
        try: return json.loads(response_content)
        except json.JSONDecodeError:
            logger.error(f"extract_search_results: Failed to decode JSON string: {response_content[:200]}")
            return {"status": "error", "message": "Result was a string but not valid JSON."}

    # Handling for MCP's TextContent-like objects if they are passed directly
    if hasattr(response_content, 'text') and isinstance(response_content.text, str):
        try: return json.loads(response_content.text)
        except json.JSONDecodeError:
            logger.error(f"extract_search_results: Failed to decode JSON from TextContent's text attribute: {response_content.text[:200]}")
            return {"status": "error", "message": "Result had .text attribute but it was not valid JSON."}
            
    # Handling for lists that might contain TextContent (less common for direct resource calls)
    if isinstance(response_content, list) and len(response_content) > 0:
        item = response_content[0]
        if isinstance(item, dict): return item
        if hasattr(item, 'text') and isinstance(item.text, str):
            try: return json.loads(item.text)
            except json.JSONDecodeError:
                logger.error(f"extract_search_results: Failed to decode JSON from list's first item's .text: {item.text[:200]}")
                return {"status": "error", "message": "Result was a list, first item's .text not valid JSON."}
    
    logger.error(f"extract_search_results: Unhandled type or content: {type(response_content)}. Preview: {str(response_content)[:200]}")
    return {"status": "error", "message": f"Search result was not a recognized JSON format. Type: {type(response_content)}"}


def format_search_results_for_prompt(results_data, query, max_results=3):
    # (Same as existing)
    if not isinstance(results_data, dict) or results_data.get("status") == "error":
        return f"Search for '{query}': {results_data.get('message', 'Error or no valid results structure.')}"
    organic = results_data.get('organic_results', [])
    if organic and isinstance(organic, list):
        formatted_results = []
        for i, item in enumerate(organic[:max_results]):
            if isinstance(item, dict):
                title = item.get('title', 'N/A')
                snippet = item.get('snippet', 'N/A')
                link = item.get('link', 'N/A')
                formatted_results.append(f"{i+1}. {title}\n   {snippet}\n   Source: {link}")
        if not formatted_results: return f"Search for '{query}' returned no usable organic result items."
        return f"Web search results for '{query}':\n" + "\n".join(formatted_results)
    answer = results_data.get('answer_box', {}).get('answer')
    if answer: return f"Web search results for '{query}':\n{answer}"
    return f"Search for '{query}' returned no specific organic results or answer box."

def format_db_results_for_prompt(query: str, db_results: Union[Dict, List], max_chars: int) -> str:
    if not db_results:
        return f"Database query '{query}' yielded no results."
    
    try:
        data_to_format = db_results 
        if isinstance(db_results, str): 
            data_to_format = json.loads(db_results)

        if isinstance(data_to_format, dict) and "error" in data_to_format:
            return f"Database query '{query}' failed: {data_to_format['error']}"
        
        if isinstance(data_to_format, dict) and "rows" in data_to_format:
            rows = data_to_format["rows"]
            columns = data_to_format.get("columns", [])
            if not rows:
                return f"Database query '{query}' returned no rows."

            header = "| " + " | ".join(map(str, columns)) + " |" if columns else ""
            lines = [header] if header else []
            if columns:
                 lines.append("|" + "---|" * len(columns))

            for row_idx, row_data in enumerate(rows):
                if isinstance(row_data, dict):
                    values = [str(row_data.get(col, "")) for col in columns] if columns else [str(v) for v in row_data.values()]
                    lines.append("| " + " | ".join(values) + " |")
                elif isinstance(row_data, list): 
                    lines.append("| " + " | ".join(map(str, row_data)) + " |")
                else: 
                    lines.append(str(row_data))
                
                current_output = "\n".join(lines)
                if len(current_output) > max_chars:
                    lines.append(f"... (results truncated due to length limit of {max_chars} chars)")
                    break
            
            formatted_str = f"Database query results for '{query}':\n" + "\n".join(lines)
        else: 
            formatted_str = f"Database query results for '{query}':\n{json.dumps(data_to_format, indent=2)}"

        if len(formatted_str) > max_chars:
            return formatted_str[:max_chars-25] + "... (results truncated)"
        return formatted_str

    except json.JSONDecodeError:
        return f"Failed to parse database results for query '{query}'. Raw: {str(db_results)[:200]}"
    except Exception as e:
        logger.error(f"Error formatting DB results: {e}", exc_info=True)
        return f"Error formatting database results for query '{query}': {str(e)}"


# --- Pydantic Models ---
class ChatMessage(BaseModel):
    role: str; content: str; is_html: Optional[bool] = False; timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
class ChatPayload(BaseModel):
    user_message: str; chat_history: List[ChatMessage]; use_search: bool; use_database: bool = False; conversation_id: Optional[str] = None; ollama_model_name: Optional[str] = None
class ChatResponse(BaseModel):
    conversation_id: str; chat_history: List[ChatMessage]; ollama_model_name: Optional[str] = None
class ConversationListItem(BaseModel):
    id: str = Field(alias="_id"); title: Optional[str] = "New Chat"; created_at: datetime; updated_at: datetime; message_count: int; ollama_model_name: Optional[str] = None
    class Config: populate_by_name = True; json_encoders = {ObjectId: str, datetime: lambda dt: dt.isoformat()}
class RenamePayload(BaseModel):
    new_title: constr(strip_whitespace=True, min_length=1, max_length=100)


async def get_default_ollama_model() -> str:
    # (Same as existing)
    try:
        resp = await asyncio.to_thread(ollama.list)
        if resp and hasattr(resp, 'models') and isinstance(resp.models, list) and resp.models:
            valid_models_info = [m for m in resp.models if hasattr(m, 'model') and isinstance(m.model, str) and m.model]
            if not valid_models_info: logger.warning("No valid Ollama models found after filtering.")
            else:
                non_embed_models = [m.model for m in valid_models_info if 'embed' not in (m.details.family.lower() if hasattr(m, 'details') and hasattr(m.details, 'family') and m.details.family else "") and 'embed' not in m.model.lower()]
                if non_embed_models: return non_embed_models[0]
                return valid_models_info[0].model
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
                    llm_content = msg_data.get("raw_content_for_llm", msg_data["content"])
                    llm_history.append({"role": msg_data["role"], "content": llm_content})
                ui_history.append(ChatMessage(**msg_data))
        else: raise HTTPException(status_code=404, detail=f"Conv ID '{conv_id}' not found.")
    
    if not model_name: model_name = payload.ollama_model_name or await get_default_ollama_model()

    if not conv_id: 
        new_title = f"Chat: {user_msg_content[:30]}{'...' if len(user_msg_content) > 30 else ''}"
        new_doc = {"title": new_title, "created_at": datetime.now(timezone.utc), "updated_at": datetime.now(timezone.utc), "messages": [], "ollama_model_name": model_name}
        res = conversations_collection.insert_one(new_doc)
        conv_id = str(res.inserted_id); obj_id = res.inserted_id
    elif obj_id and model_name and not conversations_collection.find_one({"_id": obj_id, "ollama_model_name": {"$exists": True}}):
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

    prompt_for_llm = user_msg_content
    search_html_indicator = None
    db_html_indicator = None
    assistant_error_message_obj = None

    if payload.use_search:
        if not app_state.mcp_service_ready.get(WEB_SEARCH_SERVICE_NAME, False):
            logger.warning("[API_CHAT] Web search requested but MCP service unavailable.")
            assistant_error_message_obj = ChatMessage(role="assistant", content="⚠️ Web search is currently unavailable.")
        else:
            logger.info(f"[API_CHAT] Web search active for: '{user_msg_content}'")
            try:
                req_id = await submit_mcp_tool_request(WEB_SEARCH_SERVICE_NAME, "web_search", {"query": user_msg_content})
                mcp_resp = await wait_mcp_response(WEB_SEARCH_SERVICE_NAME, req_id, timeout=90) 

                if mcp_resp.get("status") == "error":
                    raise Exception(mcp_resp.get("error", "MCP web_search tool returned an error"))
                
                raw_data = mcp_resp.get("data")
                extracted_results = extract_search_results(raw_data)
                if extracted_results.get("status") == "error":
                    raise Exception(extracted_results.get("message", "Failed to parse search results"))

                search_summary_text = format_search_results_for_prompt(extracted_results, user_msg_content)
                search_html_indicator = f"<div class='search-indicator-custom'><b>🔍 Web Search:</b> Results for \"{user_msg_content}\" were used.</div>"
                prompt_for_llm = (f"Based on the following web search results for '{user_msg_content}':\n{search_summary_text}\n\n"
                                  f"Please answer the user's original question: '{user_msg_content}'")
                logger.info(f"[API_CHAT] Web search successful, enhanced prompt created.")

            except Exception as e:
                logger.error(f"[API_CHAT] Web search processing error: {e}", exc_info=True)
                assistant_error_message_obj = ChatMessage(role="assistant", content=f"⚠️ Web search failed: {str(e)[:100]}")

    if payload.use_database and not assistant_error_message_obj: 
        if not app_state.mcp_service_ready.get(MYSQL_DB_SERVICE_NAME, False):
            logger.warning("[API_CHAT] Database interaction requested but MySQL MCP service unavailable.")
            assistant_error_message_obj = ChatMessage(role="assistant", content="⚠️ Database interaction is currently unavailable.")
        else:
            logger.info(f"[API_CHAT] Database interaction active for: '{user_msg_content}'")
            try:
                tables_req_id = await submit_mcp_resource_request(MYSQL_DB_SERVICE_NAME, "resource://tables")
                tables_resp = await wait_mcp_response(MYSQL_DB_SERVICE_NAME, tables_req_id)
                
                schema_context_parts = []
                if tables_resp.get("status") == "success":
                    # extract_search_results will parse the JSON string from the resource content
                    tables_data = extract_search_results(tables_resp.get("data")) 
                    if isinstance(tables_data, list) and tables_data: 
                        schema_context_parts.append(f"Available tables: {', '.join(tables_data[:5])}{'...' if len(tables_data) > 5 else ''}.")
                        for table_name in tables_data[:2]: 
                            schema_req_id = await submit_mcp_resource_request(MYSQL_DB_SERVICE_NAME, f"resource://tables/{table_name}/schema")
                            schema_resp = await wait_mcp_response(MYSQL_DB_SERVICE_NAME, schema_req_id)
                            if schema_resp.get("status") == "success":
                                schema_data = extract_search_results(schema_resp.get("data"))
                                schema_context_parts.append(f"Schema for '{table_name}': {json.dumps(schema_data, indent=None)[:200]}...") 
                            else:
                                schema_context_parts.append(f"Could not fetch schema for '{table_name}'. Error: {schema_resp.get('error', 'Unknown')}")
                    elif isinstance(tables_data, dict) and "error" in tables_data:
                         schema_context_parts.append(f"Could not list tables: {tables_data['error']}")
                    else: # If tables_data is not a list (e.g. error from extract_search_results) or empty
                        schema_context_parts.append(f"Could not parse table list or no tables found. Raw: {str(tables_data)[:100]}")

                else:
                    schema_context_parts.append(f"Could not retrieve table list from database. Error: {tables_resp.get('error', 'Unknown')}")
                
                full_schema_context = "\n".join(schema_context_parts)
                logger.debug(f"[API_CHAT_DB] Schema context for LLM: {full_schema_context}")

                sql_generation_prompt_messages = [
                    {"role": "system", "content": f"You are a SQL expert. Given the database schema context:\n{full_schema_context}\nGenerate a safe, read-only SQL SELECT query to answer the user's question. Output ONLY the SQL query. If you cannot generate a query, output 'NO_QUERY'."},
                    {"role": "user", "content": user_msg_content}
                ]
                raw_llm_sql_response = await chat_with_ollama(sql_generation_prompt_messages, model_name)

                extracted_sql = None
                if raw_llm_sql_response:
                    # Try to extract SQL from markdown code block first
                    sql_match = re.search(r"```(?:sql)?\s*([\s\S]+?)\s*```", raw_llm_sql_response, re.IGNORECASE)
                    if sql_match:
                        extracted_sql = sql_match.group(1).strip()
                    else:
                        # Fallback: if no markdown, try to clean the response and check if it's a SELECT
                        cleaned_response = raw_llm_sql_response.strip()
                        # A more robust fallback might be needed if LLMs are inconsistent
                        if cleaned_response.lower().startswith("select"):
                            extracted_sql = cleaned_response
                        elif cleaned_response.upper() == "NO_QUERY":
                             extracted_sql = "NO_QUERY"


                if not extracted_sql or extracted_sql.upper() == "NO_QUERY" or not extracted_sql.lower().startswith("select"):
                    logger.error(f"[API_CHAT_DB] Could not extract a valid SQL query. LLM raw response: '{raw_llm_sql_response}'. Extracted: '{extracted_sql}'")
                    raise Exception(f"Could not generate/extract a valid SQL query. LLM response: {str(raw_llm_sql_response)[:200]}...")
                
                logger.info(f"[API_CHAT_DB] LLM generated SQL (extracted): {extracted_sql}")
                
                if not extracted_sql.lower().strip().startswith("select "): # server_mysql.py also checks, but good to have here
                    raise Exception("Extracted query is not a SELECT query.")

                query_req_id = await submit_mcp_tool_request(MYSQL_DB_SERVICE_NAME, "execute_sql_query_tool", {"query": extracted_sql})
                query_resp = await wait_mcp_response(MYSQL_DB_SERVICE_NAME, query_req_id)

                if query_resp.get("status") == "error":
                    raise Exception(f"Database query execution failed: {query_resp.get('error', 'Unknown error')}")

                db_results_data = extract_search_results(query_resp.get("data")) 
                
                formatted_db_results = format_db_results_for_prompt(extracted_sql, db_results_data, MAX_DB_RESULT_CHARS)
                
                db_html_indicator = f"<div class='db-indicator-custom'><b>💾 Database:</b> Info from query \"{extracted_sql[:50].replace('<', '&lt;').replace('>', '&gt;')}...\" was used.</div>"
                
                prompt_for_llm = (f"Using the following database information related to '{user_msg_content}':\n{formatted_db_results}\n\n"
                                  f"{prompt_for_llm}") 
                logger.info(f"[API_CHAT_DB] Database interaction successful, enhanced prompt with DB results.")

            except Exception as e:
                logger.error(f"[API_CHAT_DB] Database interaction processing error: {e}", exc_info=True)
                assistant_error_message_obj = ChatMessage(role="assistant", content=f"⚠️ Database interaction failed: {str(e)[:100]}")


    if assistant_error_message_obj:
        ui_history.append(assistant_error_message_obj)
        if obj_id: conversations_collection.update_one({"_id": obj_id}, {"$push": {"messages": assistant_error_message_obj.model_dump(exclude_none=True)}, "$set": {"updated_at": datetime.now(timezone.utc)}})
        return ChatResponse(conversation_id=conv_id, chat_history=ui_history, ollama_model_name=model_name)

    llm_history.append({"role": "user", "content": prompt_for_llm}) 
    model_response_content = await chat_with_ollama(llm_history, model_name=model_name)

    if model_response_content:
        assistant_ui_response_content = model_response_content
        is_html_response = False
        
        html_prefix = ""
        if search_html_indicator:
            html_prefix += search_html_indicator
            is_html_response = True
        if db_html_indicator:
            html_prefix += db_html_indicator
            is_html_response = True
        
        if is_html_response:
            assistant_ui_response_content = f"{html_prefix}\n\n{model_response_content}"

        assistant_chat_msg = ChatMessage(role="assistant", content=assistant_ui_response_content, is_html=is_html_response)
        ui_history.append(assistant_chat_msg)
        
        assist_msg_to_save = assistant_chat_msg.model_dump(exclude_none=True)
        assist_msg_to_save["raw_content_for_llm"] = model_response_content 
        if obj_id: conversations_collection.update_one({"_id": obj_id}, {"$push": {"messages": assist_msg_to_save}, "$set": {"updated_at": datetime.now(timezone.utc)}})
    else: 
        llm_fail_msg = ChatMessage(role="assistant", content=f"Sorry, I could not get a response from the model ({model_name}).")
        ui_history.append(llm_fail_msg)
        if obj_id: conversations_collection.update_one({"_id": obj_id}, {"$push": {"messages": llm_fail_msg.model_dump(exclude_none=True)}, "$set": {"updated_at": datetime.now(timezone.utc)}})

    return ChatResponse(conversation_id=conv_id, chat_history=ui_history, ollama_model_name=model_name)


# --- FastAPI Endpoints ---
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatPayload):
    try:
        return await process_chat_request(payload)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /api/chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred processing your chat request.")

@app.get("/api/status")
async def get_status():
    ollama_ok = False
    try:
        await asyncio.to_thread(ollama.list) 
        ollama_ok = True
    except Exception:
        pass
    
    status_payload = {
        "db_connected": conversations_collection is not None,
        "ollama_available": ollama_ok,
        "mcp_services": {}
    }
    for service_name in app_state.mcp_configs.keys():
        if app_state.mcp_configs[service_name].enabled:
            status_payload["mcp_services"][service_name] = {
                "ready": app_state.mcp_service_ready.get(service_name, False)
            }
        else:
             status_payload["mcp_services"][service_name] = {
                "ready": False, "status": "disabled"
            }
    return status_payload


@app.get("/api/ollama-models", response_model=List[str])
async def list_ollama_models():
    # (Same as existing)
    try:
        resp = await asyncio.to_thread(ollama.list)
        if resp and hasattr(resp, 'models') and isinstance(resp.models, list):
            tags = [m.model for m in resp.models if hasattr(m, 'model') and isinstance(m.model, str) and m.model]
            if not tags:
                logger.warning("ollama.list() returned models, but no valid model tags found after filtering.")
                return []
            return sorted(list(set(tags)))
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

# --- Conversation History Endpoints (largely same as existing) ---
@app.get("/api/conversations", response_model=List[ConversationListItem], response_model_by_alias=False)
async def list_conversations():
    if conversations_collection is None: raise HTTPException(status_code=503, detail="MongoDB unavailable.")
    try:
        cursor = conversations_collection.find({}, {"messages": 0}).sort("updated_at", -1).limit(50)
        convs_data = list(cursor)
        conv_list_items = []
        default_model_name_cache = None 
        for db_conv_doc in convs_data:
            message_count_query = {"_id": db_conv_doc["_id"], "messages.0": {"$exists": True}} 
            msg_count = conversations_collection.count_documents(message_count_query) 
            
            item_data = {**db_conv_doc, "_id": str(db_conv_doc["_id"]), "message_count": msg_count}
            if not item_data.get("ollama_model_name"):
                if default_model_name_cache is None:
                    default_model_name_cache = await get_default_ollama_model()
                item_data["ollama_model_name"] = default_model_name_cache
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
        return [ChatMessage.model_validate(msg) for msg in conv.get("messages", []) if isinstance(msg, dict)]
    except HTTPException: raise
    except Exception as e:
        logger.error(f"Error getting messages for conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error fetching conversation details.")

@app.delete("/api/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation_endpoint(conversation_id: str): 
    if conversations_collection is None: raise HTTPException(status_code=503, detail="MongoDB unavailable.")
    if not ObjectId.is_valid(conversation_id): raise HTTPException(status_code=400, detail="Invalid conversation ID format.")
    try:
        result = conversations_collection.delete_one({"_id": ObjectId(conversation_id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Conversation not found for deletion.")
        logger.info(f"Deleted conversation ID: {conversation_id}")
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except HTTPException: raise
    except Exception as e:
        logger.error(f"Error deleting conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error deleting conversation.")

@app.put("/api/conversations/{conversation_id}/rename", response_model=ConversationListItem, response_model_by_alias=False)
async def rename_conversation_title_endpoint(conversation_id: str, payload: RenamePayload): 
    if conversations_collection is None: raise HTTPException(status_code=503, detail="MongoDB unavailable.")
    if not ObjectId.is_valid(conversation_id): raise HTTPException(status_code=400, detail="Invalid conversation ID format.")
    obj_id = ObjectId(conversation_id)
    try:
        if conversations_collection.count_documents({"_id": obj_id}) == 0:
            raise HTTPException(status_code=404, detail="Conversation not found for renaming.")
        
        update_result = conversations_collection.update_one(
            {"_id": obj_id},
            {"$set": {"title": payload.new_title, "updated_at": datetime.now(timezone.utc)}}
        )
        if update_result.matched_count == 0: 
             raise HTTPException(status_code=404, detail="Conversation not found during update operation.")

        updated_conv_doc = conversations_collection.find_one({"_id": obj_id})
        if not updated_conv_doc: 
            logger.error(f"Failed to retrieve conversation {conversation_id} after rename.")
            raise HTTPException(status_code=500, detail="Failed to retrieve updated conversation details.")
        
        msg_count = conversations_collection.count_documents({"_id": updated_conv_doc["_id"], "messages.0": {"$exists": True}})
        item_data = {**updated_conv_doc, "_id": str(updated_conv_doc["_id"]), "message_count": msg_count}
        if not item_data.get("ollama_model_name"):
            item_data["ollama_model_name"] = await get_default_ollama_model() 
        
        logger.info(f"Renamed conversation ID {conversation_id} to '{payload.new_title}'")
        return ConversationListItem.model_validate(item_data)
    except HTTPException: raise
    except Exception as e:
        logger.error(f"Error renaming conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error renaming conversation.")


# --- Static Files Hosting ---
frontend_dist_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'dist')
if os.path.exists(frontend_dist_path):
    logger.info(f"Serving static files from: {frontend_dist_path}")
    app.mount("/", StaticFiles(directory=frontend_dist_path, html=True), name="static_frontend")
else:
    logger.warning(f"Frontend build directory not found: {frontend_dist_path}. Run 'npm run build' in 'frontend' directory.")

# --- Main Execution ---
if __name__ == "__main__":
    logger.info(f"Starting Uvicorn for {__name__} when script is run directly. MCP services startup handled by FastAPI's lifespan manager.")
    
    uvicorn.run(
        "main:app",  
        host="127.0.0.1",
        port=8000,
        log_level="info",
        reload=True 
    )
