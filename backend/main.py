import asyncio
import json
import re
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel, Field
from bson import ObjectId
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
import ollama
import threading
import time
import queue

# --- Environment Setup ---
MCP_SERVER_SCRIPT = "server_search.py"
if not os.path.exists(MCP_SERVER_SCRIPT):
    if os.path.exists(os.path.join("..", MCP_SERVER_SCRIPT)):
        MCP_SERVER_SCRIPT = os.path.join("..", MCP_SERVER_SCRIPT)
    elif not os.path.exists(os.path.join(os.path.dirname(__file__), MCP_SERVER_SCRIPT)):
        logging.error(f"MCP server script '{MCP_SERVER_SCRIPT}' not found.")
    else:
        MCP_SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), MCP_SERVER_SCRIPT)

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

# --- MongoDB Setup ---
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DATABASE_NAME = os.getenv('MONGODB_DATABASE_NAME', 'mcp_chat_db')
MONGODB_COLLECTION_NAME = os.getenv('MONGODB_COLLECTION_NAME', 'conversations')

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

# --- State Management (Primarily for MCP service status and transient LLM context) ---
class AppState:
    def __init__(self):
        self.current_date = datetime.now(timezone.utc) # Use timezone-aware datetime
        self.service_ready = False
        # llm_conversation_history is now managed per-request, loaded from/saved to DB

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
    try:
        server_params = StdioServerParameters(
            command=sys.executable, args=[MCP_SERVER_SCRIPT], env=None
        )
        logger.info(f"Attempting to start MCP server with: {sys.executable} {MCP_SERVER_SCRIPT}")
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_response = await session.list_tools()
                if not any(tool.name == "web_search" for tool in tools_response.tools):
                    logger.error("Error: Required 'web_search' tool not found.")
                    app_state.service_ready = False
                    return
                app_state.service_ready = True
                logger.info("MCP service initialized and ready")
                while True:
                    try:
                        if not request_queue.empty():
                            request_data = request_queue.get_nowait()
                            if request_data["type"] == "search":
                                query = request_data["query"]
                                request_id = request_data["id"]
                                try:
                                    result = await session.call_tool("web_search", {"query": query})
                                    response_queue.put({"id": request_id, "type": "search_result", "status": "success", "data": result.content})
                                except Exception as e:
                                    logger.error(f"Error in search tool call: {e}")
                                    response_queue.put({"id": request_id, "type": "search_result", "status": "error", "error": str(e)})
                    except queue.Empty:
                        pass
                    await asyncio.sleep(0.1)
    except Exception as e:
        logger.error(f"Error in MCP service: {e}", exc_info=True)
    finally:
        app_state.service_ready = False
        logger.info("MCP service stopped")

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
async def chat_with_ollama(messages: List[Dict[str, str]]) -> Optional[str]:
    try:
        logger.info(f"[Ollama] Sending prompt. History length: {len(messages)}")
        if messages: logger.debug(f"[Ollama] Last message: {messages[-1]['content'][:200]}")
        response = await asyncio.to_thread(ollama.chat, model='qwen2.5:14b', messages=messages)
        if response and "message" in response and "content" in response["message"]:
            return response["message"]["content"]
        logger.warning(f"[Ollama] Unexpected response: {response}")
        return None
    except Exception as e:
        logger.error(f"[Ollama] Error: {e}", exc_info=True)
        return None

# --- Search Result Processing ---
def extract_search_results(response_content):
    # Simplified extraction logic
    if hasattr(response_content, 'text'): # MCP TextContent
        try: return json.loads(response_content.text)
        except json.JSONDecodeError: return response_content.text
    if isinstance(response_content, list) and len(response_content) > 0 and hasattr(response_content[0], 'text'):
        try: return json.loads(response_content[0].text)
        except json.JSONDecodeError: return response_content[0].text
    if isinstance(response_content, dict): return response_content
    return response_content


def format_search_results_for_prompt(results, query, max_results=3):
    if isinstance(results, str):
        try: results = json.loads(results)
        except json.JSONDecodeError: return f"Web search for '{query}':\n{results[:1000]}..."
    
    organic_results = []
    if isinstance(results, dict):
        organic_results = results.get('organic', []) or results.get('organic_results', [])
    elif isinstance(results, list) and results and isinstance(results[0], dict):
        organic_results = results

    if organic_results:
        formatted = "\n".join(
            f"{i+1}. {item.get('title', 'N/A')}\n   {item.get('snippet', 'N/A')}\n   Source: {item.get('link', 'N/A')}"
            for i, item in enumerate(organic_results[:max_results])
        )
        return f"Web search results for '{query}':\n{formatted}"
    return f"Search for '{query}' returned no recognizable organic results or an error."


# --- FastAPI Application ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
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
    # chat_history from client is UI history, not used for LLM context if conversation_id exists
    chat_history: List[ChatMessage] 
    use_search: bool
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    conversation_id: str
    chat_history: List[ChatMessage] # Full updated UI history for this turn

class ConversationListItem(BaseModel):
    id: str = Field(alias="_id")
    title: Optional[str] = "New Chat"
    created_at: datetime
    updated_at: datetime
    message_count: int

    class Config:
        populate_by_name = True 
        json_encoders = {ObjectId: str, datetime: lambda dt: dt.isoformat()}


async def process_chat_request(payload: ChatPayload) -> ChatResponse:
    if conversations_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB service not available.")

    user_message_content = payload.user_message
    conversation_id = payload.conversation_id
    
    llm_conversation_history: List[Dict[str, str]] = []
    ui_messages_for_response: List[ChatMessage] = []

    if conversation_id:
        conv_object_id = ObjectId(conversation_id) if ObjectId.is_valid(conversation_id) else None
        if not conv_object_id:
             raise HTTPException(status_code=400, detail="Invalid conversation_id format.")
        
        conversation = conversations_collection.find_one({"_id": conv_object_id})
        if conversation:
            for msg_data in conversation.get("messages", []):
                if 'role' in msg_data and 'content' in msg_data:
                    llm_conversation_history.append({"role": msg_data["role"], "content": msg_data.get("raw_content_for_llm", msg_data["content"])})
                ui_messages_for_response.append(ChatMessage(**msg_data))
        else:
            raise HTTPException(status_code=404, detail=f"Conversation with ID '{conversation_id}' not found.")
    else: 
        new_conv_doc = {
            "title": f"Chat started {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "messages": []
        }
        result = conversations_collection.insert_one(new_conv_doc)
        conversation_id = str(result.inserted_id)

    user_chat_message = ChatMessage(role="user", content=user_message_content)
    ui_messages_for_response.append(user_chat_message)

    if user_message_content.lower() == '#clear':
        llm_conversation_history = [] 
        assistant_response_content = "Chat context cleared. Previous messages won't be used for the next response in this session."
        assistant_chat_message = ChatMessage(role="assistant", content=assistant_response_content)
        ui_messages_for_response.append(assistant_chat_message)
        
        if conversation_id:
            conversations_collection.update_one(
                {"_id": ObjectId(conversation_id)},
                {
                    "$push": {"messages": {"$each": [
                        user_chat_message.model_dump(exclude_none=True),
                        assistant_chat_message.model_dump(exclude_none=True)
                    ]}},
                    "$set": {"updated_at": datetime.now(timezone.utc)}
                }
            )
        return ChatResponse(conversation_id=conversation_id, chat_history=ui_messages_for_response)

    prompt_for_llm = user_message_content
    raw_content_for_llm_user_message = user_message_content 
    search_performed_indicator_html = None

    if payload.use_search and app_state.service_ready:
        query = user_message_content
        logger.info(f"[MCP] Search active. Querying for: '{query}'")
        try:
            request_id = submit_search_request(query)
            response = wait_for_response(request_id)
            if response["status"] == "error": raise Exception(response.get("error", "Unknown search error"))
            
            extracted_results = extract_search_results(response["data"])
            search_results_text = format_search_results_for_prompt(extracted_results, query)
            
            search_performed_indicator_html = f"<div class='search-indicator-custom'><b>🔍 Web Search:</b> Results for \"{query}\" used.</div>"
            
            prompt_for_llm = (
                f"Today's date is {app_state.formatted_date}. Search results for '{query}':\n{search_results_text}\n\n"
                f"Using these search results, answer: '{query}'"
            )
        except Exception as e:
            logger.error(f"[MCP] Search error: {e}", exc_info=True)
            error_msg_obj = ChatMessage(role="assistant", content=f"⚠️ Search failed for '{query}'. I'll answer from my knowledge.")
            ui_messages_for_response.append(error_msg_obj)
            # Note: This transient error message is not saved to DB here.

    llm_conversation_history.append({"role": "user", "content": prompt_for_llm})
    
    user_message_to_save_dict = user_chat_message.model_dump(exclude_none=True)
    user_message_to_save_dict["raw_content_for_llm"] = prompt_for_llm 

    if conversation_id: 
        conversations_collection.update_one(
            {"_id": ObjectId(conversation_id)},
            {"$push": {"messages": user_message_to_save_dict}, "$set": {"updated_at": datetime.now(timezone.utc)}}
        )

    model_response_content = await chat_with_ollama(llm_conversation_history)

    if model_response_content:
        llm_conversation_history.append({"role": "assistant", "content": model_response_content})
        
        assistant_response_for_ui = model_response_content
        is_html = False
        if search_performed_indicator_html:
            assistant_response_for_ui = f"{search_performed_indicator_html}\n\n{model_response_content}"
            is_html = True
        
        assistant_chat_message = ChatMessage(role="assistant", content=assistant_response_for_ui, is_html=is_html)
        ui_messages_for_response.append(assistant_chat_message)

        assistant_message_to_save_dict = assistant_chat_message.model_dump(exclude_none=True)
        assistant_message_to_save_dict["raw_content_for_llm"] = model_response_content

        if conversation_id:
            conversations_collection.update_one(
                {"_id": ObjectId(conversation_id)},
                {"$push": {"messages": assistant_message_to_save_dict}, "$set": {"updated_at": datetime.now(timezone.utc)}}
            )
    else:
        err_msg_content = "Sorry, I couldn't generate a response."
        error_chat_message = ChatMessage(role="assistant", content=err_msg_content)
        ui_messages_for_response.append(error_chat_message)
        # Note: This assistant error is not saved to DB here.

    return ChatResponse(conversation_id=conversation_id, chat_history=ui_messages_for_response)


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
    return {"service_ready": app_state.service_ready, "ollama_model": "qwen2.5:14b", "db_connected": conversations_collection is not None}

@app.get("/api/conversations", response_model=List[ConversationListItem])
async def list_conversations():
    if conversations_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB service not available.")
    try:
        convs_cursor = conversations_collection.find(
            {}, {"messages": 0} 
        ).sort("updated_at", -1).limit(50) 
        
        conversation_list = []
        for conv_data in convs_cursor:
            # N+1 query pattern for message_count. For production, consider:
            # 1. Storing message_count on the conversation document and updating it with $inc.
            # 2. Using a MongoDB aggregation pipeline to calculate counts efficiently.
            full_conv_doc = conversations_collection.find_one({"_id": conv_data["_id"]}, {"messages": 1})
            message_count = len(full_conv_doc.get("messages", [])) if full_conv_doc else 0
            
            # Pydantic will use the alias "_id" for "id" field upon validation.
            # Ensure all fields required by ConversationListItem are present in conv_data or defaulted.
            conv_item_data = {**conv_data, "message_count": message_count}
            conversation_list.append(ConversationListItem.model_validate(conv_item_data))
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
    time.sleep(2) 
    logger.info("FastAPI backend setup complete. Run with: uvicorn backend.main:app --reload --port 8000")

if __name__ == "__main__":
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, initiating shutdown...")
        if mongo_client:
            mongo_client.close()
            logger.info("MongoDB connection closed.")
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main_backend()
