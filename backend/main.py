import asyncio
import json
import re
import logging
import os
import signal
import sys
import time # Keep for time.time() but not time.sleep()
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

# MCP Imports - Updated for fastmcp v2.x
import subprocess
from mcp import ClientSession
# from mcp.client.stdio import stdio_client # Old (v0.4.1 / MCP 1.6.0 style)
from fastmcp.client.transports import StdioServerParameters, stdio_client # New (v2.x)
# from mcp.common.content import TextContent # Could import for isinstance, or use duck typing


import ollama
# import queue # Replaced with asyncio.Queue
import asyncio # Ensure asyncio is imported for asyncio.Queue

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
request_queue = asyncio.Queue() # Changed to asyncio.Queue
response_queue = asyncio.Queue() # Changed to asyncio.Queue

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
    mcp_client_comms_logger.setLevel(logging.DEBUG)

    server_params = StdioServerParameters(
        command=MCP_SERVER_COMMAND[0],
        args=MCP_SERVER_COMMAND[1:],
        cwd=_main_py_dir
    )

    while True:
        try:
            logger.info(f"MCP_SERVICE_LOOP: Starting FastMCP server subprocess with params: {server_params}")

            async with stdio_client(server_params) as streams:
                read, write = streams
                logger.info("MCP_SERVICE_LOOP: Connected to FastMCP server via STDIO. Initializing ClientSession...")

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
                            try:
                                request_data = request_queue.get_nowait() # Still use get_nowait for non-blocking check
                                if request_data["type"] == "search":
                                    query = request_data["query"]
                                    request_id = request_data["id"]
                                    try:
                                        start_time_tool_call = time.time()
                                        logger.debug(f"MCP_SERVICE_LOOP: Calling 'web_search' tool with query: {query} at {start_time_tool_call}")
                                        result = await session.call_tool("web_search", {"query": query})
                                        end_time_tool_call = time.time()
                                        logger.info(f"MCP_SERVICE_LOOP: 'web_search' completed in {end_time_tool_call - start_time_tool_call:.2f} seconds")
                                        # result.content is List[ContentPart]. For web_search, it's List[TextContent]
                                        logger.debug(f"MCP_SERVICE_LOOP: 'web_search' tool call successful. Result content type: {type(result.content)}. Content: {str(result.content)[:500]}") # Log content preview
                                        await response_queue.put({"id": request_id, "type": "search_result", "status": "success", "data": result.content})
                                    except Exception as e_tool_call:
                                        logger.error(f"MCP_SERVICE_LOOP: Error in 'web_search' tool call: {e_tool_call}", exc_info=True)
                                        await response_queue.put({"id": request_id, "type": "search_result", "status": "error", "error": str(e_tool_call)})
                                request_queue.task_done() # For asyncio.Queue if we were using await get()
                            except asyncio.QueueEmpty: # Specific exception for asyncio.Queue.get_nowait()
                                pass # No item in queue, continue
                            
                            await asyncio.sleep(0.01)

        except FileNotFoundError:
            logger.error(f"MCP_SERVICE_LOOP: '{MCP_SERVER_COMMAND[0]}' command not found. Please ensure FastMCP is installed and in PATH: pip install fastmcp", exc_info=True)
        except asyncio.TimeoutError as e_timeout:
            logger.error(f"MCP_SERVICE_LOOP: TimeoutError in MCP service communication: {e_timeout}", exc_info=True)
        except Exception as e_generic:
            logger.error(f"MCP_SERVICE_LOOP: Generic Exception during MCP service operation (subprocess might have failed): {e_generic}", exc_info=True)
        finally:
            app_state.service_ready = False
            logger.info("MCP_SERVICE_LOOP: Connection lost or FastMCP server subprocess ended. Will attempt to reconnect after a 10s delay.")
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

async def submit_search_request(query: str) -> str: # Changed to async
    request_id = f"req_{time.time()}"
    await request_queue.put({"id": request_id, "type": "search", "query": query}) # Changed to await put
    return request_id

async def wait_for_response(request_id: str, timeout: int = 45) -> Dict: # Changed to async
    start_time = time.time()
    try:
        while time.time() - start_time < timeout:
            try:
                item = await asyncio.wait_for(response_queue.get(), timeout=0.05)
                if item.get("id") == request_id:
                    response_queue.task_done() 
                    return item
                else:
                    await response_queue.put(item) 
            except asyncio.TimeoutError: 
                pass 
            except asyncio.QueueEmpty: 
                pass
            await asyncio.sleep(0.05) 
    except Exception as e: 
        logger.error(f"Error in wait_for_response for {request_id}: {e}", exc_info=True)
    
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

def extract_search_results(response_content: Any) -> Dict:
    logger.debug(f"extract_search_results: Input type: {type(response_content)}, content preview: {str(response_content)[:200]}")

    if isinstance(response_content, dict):
        logger.debug("extract_search_results: Input is dict, returning as-is.")
        return response_content
    
    elif isinstance(response_content, list):
        logger.debug(f"extract_search_results: Input is list with {len(response_content)} items. First item type: {type(response_content[0]) if response_content else 'N/A'}")
        if len(response_content) == 1:
            item = response_content[0]
            if isinstance(item, dict):
                logger.debug("extract_search_results: Extracting single dict from list.")
                return item
            # Duck-typing for TextContent-like objects from MCP
            elif hasattr(item, 'text') and isinstance(item.text, str):
                logger.debug("extract_search_results: Item in list is TextContent-like, parsing item.text.")
                try:
                    parsed_dict = json.loads(item.text)
                    return parsed_dict
                except json.JSONDecodeError as e:
                    logger.error(f"extract_search_results: JSONDecodeError parsing item.text from list: {e}. Text was: {item.text[:200]}")
                    return {"status": "error", "message": "Failed to parse JSON from TextContent in list."}
            else:
                logger.error(f"extract_search_results: Single item in list is not a dict or TextContent-like. Type: {type(item)}")
                return {"status": "error", "message": "Unexpected item type in single-item results list."}
        elif len(response_content) > 1:
            # This case might indicate multiple ContentParts; typically we'd expect one TextContent for simple JSON.
            # For now, try to process the first if it's TextContent-like.
            logger.warning(f"extract_search_results: Multiple items in list ({len(response_content)}), attempting to process first item if TextContent-like.")
            item = response_content[0]
            if hasattr(item, 'text') and isinstance(item.text, str):
                logger.debug("extract_search_results: First item in multi-item list is TextContent-like, parsing item.text.")
                try:
                    return json.loads(item.text)
                except json.JSONDecodeError as e:
                    logger.error(f"extract_search_results: JSONDecodeError parsing item.text from multi-item list: {e}. Text was: {item.text[:200]}")
                    return {"status": "error", "message": "Failed to parse JSON from first TextContent in multi-item list."}
            else: # Fallback or if first item isn't what we expect
                logger.error(f"extract_search_results: First item in multi-item list is not TextContent-like or unhandled structure. Type: {type(item)}")
                return {"status": "error", "message": "Unhandled structure in multi-item results list."}
        else: # len(response_content) == 0
            logger.error("extract_search_results: Empty list received.")
            return {"status": "error", "message": "Empty results list received from tool."} # Clarified message

    # Fallback for direct TextContent-like object (if not wrapped in a list by MCP in some scenario)
    elif hasattr(response_content, 'text') and isinstance(response_content.text, str):
        logger.debug("extract_search_results: Input is TextContent-like (not in list), parsing .text.")
        try:
            return json.loads(response_content.text)
        except json.JSONDecodeError as e:
            logger.error(f"extract_search_results: JSONDecodeError parsing .text: {e}. Text was: {response_content.text[:200]}")
            return {"status": "error", "message": "Failed to parse search JSON from .text."}
            
    elif isinstance(response_content, str):
        logger.debug("extract_search_results: Input is string, attempting JSON parse.")
        try:
            return json.loads(response_content)
        except json.JSONDecodeError as e:
            logger.error(f"extract_search_results: JSONDecodeError from string: {e}")
            return {"status": "error", "message": "Failed to parse search JSON string."}
            
    else:
        logger.error(f"extract_search_results: Unhandled type: {type(response_content)}. Content: {str(response_content)[:200]}...")
        return {"status": "error", "message": f"Search result was not a recognized format. Type: {type(response_content)}"}


def format_search_results_for_prompt(results_data, query, max_results=3):
    if not isinstance(results_data, dict) or results_data.get("status") == "error":
        return f"Search for '{query}': {results_data.get('message', 'Error or no valid results structure.')}"
    
    # The server_search.py tool now returns a dict with "organic_results" key directly.
    organic = results_data.get('organic_results', [])
    
    # This old check might not be needed if server_search.py is consistent
    # if not organic and isinstance(results_data, list): 
    #     organic = results_data # This would imply results_data itself was the list of items

    if organic and isinstance(organic, list):
        # Ensure items in organic are dicts before trying to get 'title', 'snippet', 'link'
        formatted_results = []
        for i, item in enumerate(organic[:max_results]):
            if isinstance(item, dict):
                title = item.get('title', 'N/A')
                snippet = item.get('snippet', 'N/A')
                link = item.get('link', 'N/A')
                formatted_results.append(f"{i+1}. {title}\n   {snippet}\n   Source: {link}")
            else:
                logger.warning(f"format_search_results_for_prompt: Skipping non-dict item in organic_results: {item}")
        if not formatted_results:
             return f"Search for '{query}' returned organic results, but items were not in the expected format."
        return f"Web search results for '{query}':\n" + "\n".join(formatted_results)
    elif not organic: # organic is empty or not a list
        # Check if other keys like 'top_stories' or 'people_also_ask' have content,
        # if you want to use them as fallback. For now, just report based on organic.
        answer = results_data.get('answer_box', {}).get('answer')
        if answer:
            return f"Web search results for '{query}':\n{answer}"
        return f"Search for '{query}' returned no specific organic results. Full data: {str(results_data)[:200]}"
    else: # organic is not a list (should not happen if server_search.py is correct)
        return f"Search for '{query}' returned organic data in an unexpected format. Data: {str(results_data)[:200]}"


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
            valid_models_info = [m for m in resp.models if hasattr(m, 'model') and isinstance(m.model, str) and m.model]
            if not valid_models_info:
                logger.warning("No valid Ollama models found after filtering.")
            else:
                non_embed_models = [
                    m.model for m in valid_models_info
                    if 'embed' not in (m.details.family.lower() if hasattr(m, 'details') and hasattr(m.details, 'family') and m.details.family else "")
                    and 'embed' not in m.model.lower()
                ]
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
    elif obj_id and model_name:
        current_db_model = conversations_collection.find_one({"_id": obj_id}, {"ollama_model_name": 1})
        if current_db_model and not current_db_model.get("ollama_model_name"):
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
            assist_err_msg_obj = ChatMessage(role="assistant", content="⚠️ Web search unavailable at the moment.")
        else:
            logger.info(f"[API_CHAT] Search active for: '{user_msg_content}'")
            try:
                req_id = await submit_search_request(user_msg_content) 
                logger.debug(f"[API_CHAT] Submitted search request with ID: {req_id}")

                mcp_resp = await wait_for_response(req_id, timeout=90) 
                logger.debug(f"[API_CHAT] Received MCP response: {mcp_resp}")

                if mcp_resp.get("status") == "error" and mcp_resp.get("error") == "Request timed out": # Check for timeout specifically
                    logger.error(f"[API_CHAT] MCP response timed out: {mcp_resp.get('error')}")
                    raise Exception("Search request timed out.") # Specific exception for timeout
                elif mcp_resp.get("status") == "error": # Other errors from MCP tool call itself
                    logger.error(f"[API_CHAT] MCP response indicates tool error: {mcp_resp.get('error')}")
                    raise Exception(mcp_resp.get("error", "MCP search tool returned an error status"))

                raw_data = mcp_resp.get("data")
                logger.debug(f"[API_CHAT] Raw search data type: {type(raw_data)}, content preview: {str(raw_data)[:500]}")

                extracted_results_data = extract_search_results(raw_data)
                logger.debug(f"[API_CHAT] Extracted results: {extracted_results_data}")

                if extracted_results_data.get("status") == "error":
                    logger.error(f"[API_CHAT] Extracted results indicate error: {extracted_results_data.get('message')}")
                    raise Exception(extracted_results_data.get("message", "Failed to extract or parse search results."))

                search_summary_text = format_search_results_for_prompt(extracted_results_data, user_msg_content)
                logger.debug(f"[API_CHAT] Formatted search summary (first 200 chars): {search_summary_text[:200]}")
                
                if "Error or no valid results structure" in search_summary_text or \
                   "returned no specific organic results" in search_summary_text or \
                   "data in an unexpected format" in search_summary_text or \
                   "items were not in the expected format" in search_summary_text:
                    logger.warning(f"[API_CHAT] Search summary indicates issues or no results: {search_summary_text}")
                    # Potentially raise an exception or set assist_err_msg_obj here if this is critical
                    # For now, we'll let it proceed but the LLM might not get useful context.
                    # If the search itself was "successful" but yielded no usable items, that's different from a technical error.

                search_html_indicator = f"<div class='search-indicator-custom'><b>🔍 Web Search:</b> Results for \"{user_msg_content}\" were used.</div>"
                prompt_llm = (f"Based on the following web search results for '{user_msg_content}':\n{search_summary_text}\n\n"
                              f"Please answer the user's original question: '{user_msg_content}'")
                logger.info(f"[API_CHAT] Search processing successful, enhanced prompt created")

            except Exception as e:
                logger.error(f"[API_CHAT] Search processing error: {e}", exc_info=True)
                assist_err_msg_obj = ChatMessage(role="assistant", content=f"⚠️ Search failed: {str(e)[:100]}")

    if assist_err_msg_obj:
        ui_history.append(assist_err_msg_obj)
        if obj_id: conversations_collection.update_one({"_id": obj_id}, {"$push": {"messages": assist_err_msg_obj.model_dump(exclude_none=True)}, "$set": {"updated_at": datetime.now(timezone.utc)}})

    llm_history.append({"role": "user", "content": prompt_llm})
    model_resp_content = await chat_with_ollama(llm_history, model_name=model_name)

    if model_resp_content:
        assist_ui_resp = model_resp_content; is_html_resp = False
        if search_html_indicator and not assist_err_msg_obj: # Only add indicator if search didn't result in an error message
            assist_ui_resp = f"{search_html_indicator}\n\n{model_resp_content}"; is_html_resp = True
        assist_chat_msg = ChatMessage(role="assistant", content=assist_ui_resp, is_html=is_html_resp)
        ui_history.append(assist_chat_msg)
        assist_msg_to_save = assist_chat_msg.model_dump(exclude_none=True)
        assist_msg_to_save["raw_content_for_llm"] = model_resp_content
        if obj_id: conversations_collection.update_one({"_id": obj_id}, {"$push": {"messages": assist_msg_to_save}, "$set": {"updated_at": datetime.now(timezone.utc)}})
    elif not assist_err_msg_obj: # Only add LLM error if no search error message was already prepared
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
        pass
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

@app.get("/api/conversations", response_model=List[ConversationListItem], response_model_by_alias=False)
async def list_conversations():
    if conversations_collection is None: raise HTTPException(status_code=503, detail="MongoDB unavailable.")
    try:
        cursor = conversations_collection.find({}, {"messages": 0}).sort("updated_at", -1).limit(50)
        convs_data = list(cursor)
        conv_list_items = []
        default_model_name = None
        for db_conv_doc in convs_data:
            message_count_query = {"_id": db_conv_doc["_id"], "messages.0": {"$exists": True}}
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
        return Response(status_code=status.HTTP_204_NO_CONTENT)
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
        sys.exit(0)
    signal.signal(signal.SIGINT, graceful_shutdown_handler)
    signal.signal(signal.SIGTERM, graceful_shutdown_handler)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
