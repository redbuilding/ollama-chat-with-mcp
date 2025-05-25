import asyncio

import json

import re

import logging

import os

import signal

import sys

from datetime import datetime

from typing import Optional, List, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException, Request

from fastapi.staticfiles import StaticFiles

from fastapi.middleware.cors import CORSMiddleware

import uvicorn



from mcp import ClientSession, StdioServerParameters, types

from mcp.client.stdio import stdio_client

import ollama

import threading

import time

import queue



# --- Environment Setup ---

# Ensure server_search.py is in the same directory or adjust the path

MCP_SERVER_SCRIPT = "server_search.py"

# Check if MCP_SERVER_SCRIPT exists

if not os.path.exists(MCP_SERVER_SCRIPT):

    # Try looking in the parent directory if main.py is in a subdirectory like 'app'

    if os.path.exists(os.path.join("..", MCP_SERVER_SCRIPT)):

        MCP_SERVER_SCRIPT = os.path.join("..", MCP_SERVER_SCRIPT)

    # If still not found, log an error. The application might fail later.

    elif not os.path.exists(os.path.join(os.path.dirname(__file__), MCP_SERVER_SCRIPT)):

        logging.error(f"MCP server script '{MCP_SERVER_SCRIPT}' not found in current directory or parent.
Please ensure it's correctly placed.")

        # For robustness, you might want to exit or handle this more gracefully

        # For now, we'll let it proceed and potentially fail at MCP client init

    else:

        MCP_SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), MCP_SERVER_SCRIPT)





# Create logs directory if it doesn't exist

os.makedirs('logs', exist_ok=True)



# Configure logging

logging.basicConfig(

    level=logging.INFO,

    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',

    handlers=[

        logging.FileHandler(f"logs/mcp_backend_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),

        logging.StreamHandler()

    ]

)

logger = logging.getLogger("mcp_backend")



# --- Communication Queues (for MCP service interaction) ---

request_queue = queue.Queue()

response_queue = queue.Queue()



# --- State Management ---

class AppState:

    def __init__(self):

        self.llm_conversation_history = [] # History for the LLM

        self.current_date = datetime.now()

        self.service_ready = False



    @property

    def formatted_date(self):

        return f"{self.current_date.strftime('%B')} {self.current_date.strftime('%d').lstrip('0')},
{self.current_date.strftime('%Y')}"



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

            command=sys.executable, # Use current python interpreter

            args=[MCP_SERVER_SCRIPT],

            env=None

        )

        logger.info(f"Attempting to start MCP server with: {sys.executable} {MCP_SERVER_SCRIPT}")



        async with stdio_client(server_params) as (read, write):

            async with ClientSession(read, write) as session:

                await session.initialize()

                tools_response = await session.list_tools()

                logger.info(f"Available Tools: {[tool.name for tool in tools_response.tools]}")



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

                                logger.info(f"Processing search request: {query}")

                                try:

                                    result = await session.call_tool("web_search", {"query": query})

                                    response_queue.put({

                                        "id": request_id, "type": "search_result",

                                        "status": "success", "data": result.content

                                    })

                                except Exception as e:

                                    logger.error(f"Error in search tool call: {e}")

                                    response_queue.put({

                                        "id": request_id, "type": "search_result",

                                        "status": "error", "error": str(e)

                                    })

                    except queue.Empty:

                        pass

                    await asyncio.sleep(0.1)

    except Exception as e:

        logger.error(f"Error in MCP service: {e}")

        logger.exception("Detailed error:")

    finally:

        app_state.service_ready = False

        logger.info("MCP service stopped")



# --- Utility Functions for MCP Interaction ---

def submit_search_request(query: str) -> str:

    request_id = f"req_{time.time()}"

    request_queue.put({"id": request_id, "type": "search", "query": query})

    logger.info(f"Submitted search request: {query} (ID: {request_id})")

    return request_id



def wait_for_response(request_id: str, timeout: int = 30) -> Dict:

    start_time = time.time()

    while time.time() - start_time < timeout:

        try:

            items_in_queue = []

            found_response = None

            while not response_queue.empty():

                item = response_queue.get_nowait()

                if item.get("id") == request_id:

                    found_response = item

                    break

                else:

                    items_in_queue.append(item)



            for item in items_in_queue: # Put back other items

                response_queue.put(item)



            if found_response:

                return found_response

        except queue.Empty:

            pass

        time.sleep(0.5)

    return {"id": request_id, "type": "search_result", "status": "error", "error": "Request timed out"}



# --- Ollama Interaction ---

async def chat_with_ollama(messages: List[Dict[str, str]]) -> Optional[str]:

    try:

        logger.info(f"[Ollama] Sending prompt to model. History length: {len(messages)}")

        # Log the last message content for debugging

        if messages:

            logger.debug(f"[Ollama] Last message content: {messages[-1]['content'][:200]}")



        response = await asyncio.to_thread(

            ollama.chat,

            model='qwen2.5:14b', # Ensure this model is available in your Ollama setup

            messages=messages

        )

        if response and "message" in response and "content" in response["message"]:

            return response["message"]["content"]

        logger.warning(f"[Ollama] Unexpected response structure: {response}")

        return None

    except Exception as e:

        logger.error(f"[Ollama] Error: {e}")

        logger.exception("Detailed Ollama error:")

        return None



# --- Search Result Processing (largely unchanged) ---

def extract_search_results(response_content):

    if hasattr(response_content, '__class__') and response_content.__class__.__name__ == 'TextContent':

        if hasattr(response_content, 'text'):

            try: return json.loads(response_content.text)

            except json.JSONDecodeError: return response_content.text

    if isinstance(response_content, list) and len(response_content) > 0:

        first_item = response_content[0]

        if hasattr(first_item, '__class__') and first_item.__class__.__name__ == 'TextContent':

            if hasattr(first_item, 'text'):

                try: return json.loads(first_item.text)

                except json.JSONDecodeError: return first_item.text

        if isinstance(first_item, dict) and 'text' in first_item:

            try: return json.loads(first_item['text'])

            except json.JSONDecodeError: return first_item['text']

    if isinstance(response_content, dict):

        if 'status' in response_content and response_content['status'] == 'success':

            return {'organic': response_content.get('organic_results', []),

                    'topStories': response_content.get('top_stories', []),

                    'peopleAlsoAsk': response_content.get('people_also_ask', [])}

        if 'organic' in response_content: return response_content

    return response_content



def format_search_results_for_prompt(results, query, max_results=3):

    result_type = type(results).__name__

    if isinstance(results, str):

        try: results = json.loads(results)

        except json.JSONDecodeError: return f"Web search results for '{query}':\n{results[:1000]}..."

    if isinstance(results, dict):

        if 'error' in results: return f"Search failed: {results['error']}"

        organic_results = results.get('organic', []) or results.get('organic_results', [])

        if organic_results:

            formatted = "\n".join(

                f"{i+1}. {item.get('title', 'N/A')}\n   {item.get('snippet', 'N/A')}\n   Source:
{item.get('link', 'N/A')}"

                for i, item in enumerate(organic_results[:max_results])

            )

            return f"Web search results for '{query}':\n{formatted}"

    if isinstance(results, list) and results and isinstance(results[0], dict) and 'title' in results[0]:

        formatted = "\n".join(

            f"{i+1}. {item.get('title', 'N/A')}\n   {item.get('snippet', 'N/A')}\n   Source: {item.get('link',
'N/A')}"

            for i, item in enumerate(results[:max_results])

        )

        return f"Web search results for '{query}':\n{formatted}"

    return f"Search returned results in an unrecognized format ({result_type}) for '{query}'"



# --- FastAPI Application ---

app = FastAPI()



# CORS Middleware

app.add_middleware(

    CORSMiddleware,

    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], # Adjust for your frontend dev port

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],

)



class ChatRequest(Request):

    user_message: str

    chat_history: List[Dict[str, str]] # History from the client's perspective

    use_search: bool



async def process_chat_request(payload: Dict) -> List[Dict[str, str]]:

    user_message = payload.get("user_message")

    client_chat_history = payload.get("chat_history", [])

    use_search = payload.get("use_search", False)



    logger.info(f"Processing message: '{user_message[:50]}...', use_search: {use_search}")



    # Current UI history starts with what client sent

    current_ui_history = [msg.copy() for msg in client_chat_history]

    current_ui_history.append({"role": "user", "content": user_message})



    # Handle #clear command

    if user_message.lower() == '#clear':

        app_state.llm_conversation_history = [] # Clear LLM context

        logger.info("Chat history cleared by #clear command.")

        return [{"role": "assistant", "content": "Chat history cleared."}]



    # Handle exit/quit (client should handle this, but good to have a note)

    if user_message.lower() in ['exit', 'quit']:

        logger.info("Exit/quit command received.")

        # Client should handle UI, this is just a polite server response

        return current_ui_history + [{"role": "assistant", "content": "Goodbye!"}]



    # Prepare for LLM

    prompt_for_llm = user_message

    search_performed_info = None # To store info about search if performed



    if use_search:

        if not app_state.service_ready:

            # Add a message to UI history about service not ready

            current_ui_history.append({"role": "assistant", "content": "Search service is not available. I'll
answer based on my training data only."})

            # LLM will get the original user_message

        else:

            query = user_message # Assume the whole message is the query when search is toggled

            logger.info(f"[MCP] Search active. Querying for: '{query}'")

            try:

                request_id = submit_search_request(query)

                response = wait_for_response(request_id)



                if response["status"] == "error":

                    raise Exception(response.get("error", "Unknown search error"))



                extracted_results = extract_search_results(response["data"])

                search_results_text = format_search_results_for_prompt(extracted_results, query)



                search_performed_info = {"query": query} # Store query for display message



                prompt_for_llm = (

                    f"Today's date is {app_state.formatted_date}. Here are CURRENT search results from
{app_state.month_year} about '{query}':\n\n"

                    f"{search_results_text}\n\n"

                    f"Using these UP-TO-DATE search results, provide a comprehensive answer to the implicit or
explicit question in '{query}'.\n\n"

                    f"Note: These search results ARE CURRENT as of {app_state.month_year} and contain real
information about events in {app_state.year}."

                )

                logger.info(f"Prompt for LLM (with search results for '{query}'): {prompt_for_llm[:200]}...")



            except Exception as e:

                logger.error(f"[MCP] Search error: {e}")

                logger.exception("Detailed search error:")

                current_ui_history.append({"role": "assistant", "content": f"❌ Search failed for '{query}'.
I'll try to answer without web data."})

                # Fallback: LLM gets original user_message, but we can prepend a note about failed search

                prompt_for_llm = f"Search was attempted for '{query}' but it failed. Please answer based on
your training data: {user_message}"



    # Update LLM conversation history

    # Add the (potentially augmented) user message that LLM will see

    app_state.llm_conversation_history.append({"role": "user", "content": prompt_for_llm})



    # Get model response

    model_response_content = await chat_with_ollama(app_state.llm_conversation_history)



    if model_response_content:

        app_state.llm_conversation_history.append({"role": "assistant", "content": model_response_content})



        # Format assistant response for UI

        assistant_response_for_ui = model_response_content

        if search_performed_info:

            # Prepend search indicator using HTML (React will use dangerouslySetInnerHTML)

            # Ensure your frontend CSS handles 'search-indicator-custom'

            indicator_html = f"<div class='search-indicator-custom'><b>🔍 Web Search Tool:</b> Results for
\"{search_performed_info['query']}\" were used.</div>"

            assistant_response_for_ui = f"{indicator_html}\n\n{model_response_content}"

            current_ui_history.append({"role": "assistant", "content": assistant_response_for_ui, "is_html":
True})

        else:

            current_ui_history.append({"role": "assistant", "content": assistant_response_for_ui})



    else:

        # Error response if Ollama failed

        err_msg = "Sorry, I couldn't generate a response at this time."

        current_ui_history.append({"role": "assistant", "content": err_msg})

        # Add a placeholder to LLM history so it knows an attempt was made

        app_state.llm_conversation_history.append({"role": "assistant", "content": "[Error: No response
generated]"})



    return current_ui_history





@app.post("/api/chat")

async def chat_endpoint(request: Request):

    try:

        payload = await request.json()

        if not all(k in payload for k in ["user_message", "chat_history", "use_search"]):

            raise HTTPException(status_code=400, detail="Missing required fields in chat request")



        updated_history = await process_chat_request(payload)

        return {"chat_history": updated_history}

    except json.JSONDecodeError:

        logger.error("Invalid JSON received in /api/chat")

        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    except Exception as e:

        logger.error(f"Error in /api/chat: {e}")

        logger.exception("Detailed /api/chat error:")

        raise HTTPException(status_code=500, detail="Internal server error")



@app.get("/api/status")

async def get_status():

    return {"service_ready": app_state.service_ready, "ollama_model": "qwen2.5:14b"}



# Serve static files for React frontend (optional, for production)

# This assumes your React app is built into 'frontend/dist'

# and this script is run from the project root, or paths are adjusted.

# For development, frontend is usually served by its own dev server.

# Check if 'frontend/dist' exists relative to this script's location

frontend_dist_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'dist')

if os.path.exists(frontend_dist_path):

    logger.info(f"Serving static files from: {frontend_dist_path}")

    app.mount("/", StaticFiles(directory=frontend_dist_path, html=True), name="static")

else:

    logger.warning(f"Frontend build directory not found at {frontend_dist_path}. Static file serving will be
disabled. This is normal for development if frontend is on a separate server.")





# --- Main Application Logic ---

def main_backend():

    # Start MCP service in a separate thread

    service_thread = threading.Thread(target=run_mcp_service, daemon=True)

    service_thread.start()

    logger.info("MCP service thread started")



    # Wait a moment for the service to potentially start up

    # The /api/status endpoint can be polled by frontend for readiness

    time.sleep(2)



    logger.info("Starting FastAPI server...")

    # Configuration for Uvicorn can be done here or via command line

    # Example: uvicorn.run(app, host="0.0.0.0", port=8000)

    # Running via `uvicorn main:app --reload` is typical for development.

    # This main() function is more for conceptual structure or if you embed uvicorn.

    # For this setup, we'll assume uvicorn is run from the command line.

    print("FastAPI backend setup complete. Run with: uvicorn backend.main:app --reload --port 8000")





if __name__ == "__main__":

    # Set up signal handlers for graceful shutdown (optional, uvicorn handles some)

    def signal_handler(sig, frame):

        logger.info(f"Received signal {sig}, initiating shutdown...")

        # Perform any cleanup if necessary

        # MCP service thread is daemon, should exit with main

        sys.exit(0)



    signal.signal(signal.SIGINT, signal_handler)

    signal.signal(signal.SIGTERM, signal_handler)



    # This part is mostly for when you run `python backend/main.py` directly

    # and want to start uvicorn programmatically.

    # Typically, you'd run `uvicorn backend.main:app --reload --port 8000` from terminal.

    main_backend()

    # If running programmatically:

    # uvicorn.run(app, host="0.0.0.0", port=8000)