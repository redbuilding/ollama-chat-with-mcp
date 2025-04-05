import asyncio
import json
import re
import logging
import os
import signal
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import gradio as gr
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
import ollama
import threading
import time
import queue

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/mcp_client_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mcp_client")

# --- Communication Queues ---
# Queue for requests from UI to service
request_queue = queue.Queue()
# Queue for responses from service to UI
response_queue = queue.Queue()

# --- State Management ---
class AppState:
    def __init__(self):
        self.chat_history = []
        self.current_date = datetime.now()
        self.service_ready = False

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

# --- MCP Service ---
def run_mcp_service():
    """Run the MCP service in a separate thread"""
    asyncio.run(mcp_service_loop())

async def mcp_service_loop():
    """Main service loop that handles MCP communication"""
    logger.info("Starting MCP service loop")

    try:
        # Initialize MCP connection
        server_params = StdioServerParameters(
            command="python",
            args=["server.py"],
            env=None
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Check tools
                tools_response = await session.list_tools()
                logger.info(f"Available Tools: {[tool.name for tool in tools_response.tools]}")

                if not any(tool.name == "web_search" for tool in tools_response.tools):
                    logger.error("Error: Required 'web_search' tool not found.")
                    return

                # Mark service as ready
                app_state.service_ready = True
                logger.info("MCP service initialized and ready")

                # Process requests from the queue
                while True:
                    # Check for requests (non-blocking)
                    try:
                        if not request_queue.empty():
                            request = request_queue.get_nowait()

                            # Process different request types
                            if request["type"] == "search":
                                query = request["query"]
                                request_id = request["id"]

                                logger.info(f"Processing search request: {query}")
                                try:
                                    # Call search tool
                                    result = await session.call_tool("web_search", {"query": query})

                                    # Put result in response queue
                                    response_queue.put({
                                        "id": request_id,
                                        "type": "search_result",
                                        "status": "success",
                                        "data": result.content
                                    })

                                except Exception as e:
                                    logger.error(f"Error in search: {e}")
                                    # Put error in response queue
                                    response_queue.put({
                                        "id": request_id,
                                        "type": "search_result",
                                        "status": "error",
                                        "error": str(e)
                                    })
                    except queue.Empty:
                        pass

                    # Wait a bit before checking again
                    await asyncio.sleep(0.1)

    except Exception as e:
        logger.error(f"Error in MCP service: {e}")
        logger.exception("Detailed error:")
        app_state.service_ready = False
    finally:
        app_state.service_ready = False
        logger.info("MCP service stopped")

# --- Utility Functions ---
def submit_search_request(query: str) -> str:
    """Submit a search request to the service and return a request ID"""
    request_id = f"req_{time.time()}"

    request_queue.put({
        "id": request_id,
        "type": "search",
        "query": query
    })

    logger.info(f"Submitted search request: {query} (ID: {request_id})")
    return request_id

def wait_for_response(request_id: str, timeout: int = 30) -> Dict:
    """Wait for a response with the specified request ID"""
    start_time = time.time()

    while time.time() - start_time < timeout:
        # Check if response is available
        try:
            # Look through all items in the queue
            items = []
            response = None

            while not response_queue.empty():
                item = response_queue.get_nowait()
                if item["id"] == request_id:
                    response = item
                else:
                    # Put back items that don't match
                    items.append(item)

            # Put back unmatched items
            for item in items:
                response_queue.put(item)

            if response:
                return response

        except queue.Empty:
            pass

        # Wait a bit before checking again
        time.sleep(0.5)

    # If we get here, we timed out
    return {
        "id": request_id,
        "type": "search_result",
        "status": "error",
        "error": "Request timed out"
    }

# --- Ollama Interaction ---
async def chat_with_ollama(messages: List[Dict[str, str]]) -> Optional[str]:
    """Sends a conversation history to a local Ollama model and returns the response."""
    try:
        logger.info("[Ollama] Sending prompt...")
        response = await asyncio.to_thread(
            ollama.chat,
            model='qwen2.5:14b',
            messages=messages
        )
        if response and "message" in response and "content" in response["message"]:
            return response["message"]["content"]
        return None
    except Exception as e:
        logger.error(f"[Ollama] Error: {e}")
        return None

# --- Search Result Processing ---
def extract_search_results(response_content):
    """Extract search results from various possible response formats"""
    # Handle TextContent objects directly
    if hasattr(response_content, '__class__') and response_content.__class__.__name__ == 'TextContent':
        if hasattr(response_content, 'text'):
            try:
                # Try to parse the text field as JSON
                return json.loads(response_content.text)
            except json.JSONDecodeError:
                return response_content.text

    # Handle list of TextContent objects
    if isinstance(response_content, list) and len(response_content) > 0:
        first_item = response_content[0]

        # Check if the first item is a TextContent object
        if hasattr(first_item, '__class__') and first_item.__class__.__name__ == 'TextContent':
            if hasattr(first_item, 'text'):
                try:
                    # Try to parse the text field as JSON
                    return json.loads(first_item.text)
                except json.JSONDecodeError:
                    return first_item.text

        # If it's a dictionary with 'text' key (representation of TextContent)
        if isinstance(first_item, dict) and 'text' in first_item:
            try:
                # Try to parse the text field as JSON
                return json.loads(first_item['text'])
            except json.JSONDecodeError:
                return first_item['text']

    # Standard dictionary handling
    if isinstance(response_content, dict):
        if 'status' in response_content:
            if response_content['status'] == 'success':
                return {
                    'organic': response_content.get('organic_results', []),
                    'topStories': response_content.get('top_stories', []),
                    'peopleAlsoAsk': response_content.get('people_also_ask', [])
                }
            return {'error': response_content.get('message', 'Unknown error')}

        # Original server response format
        if 'organic' in response_content:
            return response_content

    # If all else fails, return as is
    return response_content

def format_search_results_for_prompt(results, query, max_results=3):
    """Format search results into a readable text for the prompt"""
    # Debug the results structure to help diagnose issues
    result_type = type(results).__name__

    # Handle string results that might be JSON
    if isinstance(results, str):
        try:
            results = json.loads(results)
        except json.JSONDecodeError:
            # Not JSON, just use the string
            return f"Web search results for '{query}':\n{results[:1000]}..."

    # Handle dictionary results
    if isinstance(results, dict):
        # Check for error
        if 'error' in results:
            return f"Search failed: {results['error']}"

        # Try different possible structures
        organic_results = results.get('organic', []) or results.get('organic_results', [])

        if organic_results:
            formatted_results = "\n".join(
                f"{i+1}. {item.get('title', 'N/A')}\n   {item.get('snippet', 'N/A')}\n   Source: {item.get('link', 'N/A')}"
                for i, item in enumerate(organic_results[:max_results])
            )
            return f"Web search results for '{query}':\n{formatted_results}"

    # If we got here and have a list, try to process it
    if isinstance(results, list):
        # Check if it's a list of search results
        if results and isinstance(results[0], dict) and 'title' in results[0]:
            formatted_results = "\n".join(
                f"{i+1}. {item.get('title', 'N/A')}\n   {item.get('snippet', 'N/A')}\n   Source: {item.get('link', 'N/A')}"
                for i, item in enumerate(results[:max_results])
            )
            return f"Web search results for '{query}':\n{formatted_results}"

    # Fallback for unrecognized format
    return f"Search returned results in an unrecognized format ({result_type}) for '{query}'"

# --- Chat Processing Functions ---
async def process_message(user_message: str, chat_history: List) -> List:
    """Process user message and update chat history"""
    logger.info(f"Processing message: {user_message[:50]}...")

    # Initialize chat history if it's empty or in the wrong format
    if not chat_history:
        chat_history = []

    # Convert chat_history format if needed for compatibility
    if chat_history and isinstance(chat_history[0], list):
        # Convert from tuples to message dict format
        formatted_history = []
        for item in chat_history:
            if isinstance(item, list) and len(item) == 2:
                formatted_history.append({"role": "user", "content": item[0]})
                formatted_history.append({"role": "assistant", "content": item[1]})
        chat_history = formatted_history

    # Check if MCP service is ready
    if not app_state.service_ready:
        new_history = chat_history.copy()
        new_history.append({"role": "user", "content": user_message})
        new_history.append({"role": "assistant", "content": "Search service is not available. I'll answer based on my training data only."})
        yield new_history

    # Handle special commands
    if user_message.lower() in ['exit', 'quit']:
        new_history = chat_history.copy()
        new_history.append({"role": "user", "content": user_message})
        new_history.append({"role": "assistant", "content": "Goodbye! Refresh the page to start a new session."})
        yield new_history
        return

    if user_message.lower() == '#clear':
        app_state.chat_history = []
        new_history = [{"role": "assistant", "content": "Chat history cleared."}]
        yield new_history
        return

    final_prompt = user_message
    search_used = False
    search_query = None
    search_results_text = None

    # Add user message to chat history
    new_chat_history = chat_history.copy()
    new_chat_history.append({"role": "user", "content": user_message})

    # Parse for search pattern: #search for "query" and [question]
    search_pattern = r'#search for "(.*?)"(?: and (.*))?'
    search_match = re.search(search_pattern, user_message, re.IGNORECASE)

    # Modified search handling
    if search_match and app_state.service_ready:
        query = search_match.group(1).strip()
        user_question = search_match.group(2).strip() if search_match.group(2) else None
        search_query = query
        search_used = True

        logger.info(f"[MCP] Searching: '{query}', User question: '{user_question}'")
        # Add a loading message
        new_chat_history.append({"role": "assistant", "content": "🔍 Searching the web..."})
        yield new_chat_history

        try:
            # Submit search request
            request_id = submit_search_request(query)

            # Wait for response
            response = wait_for_response(request_id)

            if response["status"] == "error":
                raise Exception(response["error"])

            # Extract structured data from the response
            extracted_results = extract_search_results(response["data"])

            # Format the results for the prompt
            search_results_text = format_search_results_for_prompt(extracted_results, query)

            # Create the final prompt based on whether there's a user question
            if user_question:
                final_prompt = (
                    f"Today's date is {app_state.formatted_date}. Here are CURRENT search results from {app_state.month_year} about '{query}':\n\n"
                    f"{search_results_text}\n\n"
                    f"Using these UP-TO-DATE search results, answer this question: {user_question}\n\n"
                    f"Note: These search results ARE CURRENT as of {app_state.month_year} and contain real information about events in {app_state.year}."
                )
            else:
                final_prompt = (
                    f"Today's date is {app_state.formatted_date}. Here are CURRENT search results from {app_state.month_year} about '{query}':\n\n"
                    f"{search_results_text}\n\n"
                    f"Using these UP-TO-DATE search results, summarize what they tell us about '{query}'.\n\n"
                    f"Note: These search results ARE CURRENT as of {app_state.month_year} and contain real information about events in {app_state.year}."
                )

        except Exception as e:
            logger.error(f"[MCP] Search error: {e}")
            logger.exception("Detailed error:")
            # Replace loading message with error message
            new_chat_history[-1] = {"role": "assistant", "content": "❌ Search failed. I'll try to answer without web data."}
            yield new_chat_history
            final_prompt = f"Search unavailable for '{query}'. Please answer without web data: {user_question or query}"

    # Add user message to LLM history
    app_state.chat_history.append({"role": "user", "content": final_prompt})

    # Remove loading message if present
    if search_used and len(new_chat_history) > 0 and isinstance(new_chat_history[-1], dict) and "Searching the web..." in new_chat_history[-1].get("content", ""):
        new_chat_history.pop()

    # Placeholder for response while waiting for the model
    new_chat_history.append({"role": "assistant", "content": "Thinking..."})
    yield new_chat_history

    # Get model response with full history
    model_response = await chat_with_ollama(app_state.chat_history)

    # Update the final chat history
    if model_response:
        # Format the response for display with tool indicators
        if search_used and search_results_text:
            # Create a formatted response with tool indicator
            formatted_response = f"<div style='padding: 5px; margin-bottom: 10px; font-size: 0.8em; color: #555;'><b>🔍 Web Search Tool:</b> Results for \"{search_query}\" were used</div>\n\n{model_response}"
            # Add model response to history
            app_state.chat_history.append({"role": "assistant", "content": model_response})
            new_chat_history[-1] = {"role": "assistant", "content": formatted_response}
        else:
            # Regular response
            app_state.chat_history.append({"role": "assistant", "content": model_response})
            new_chat_history[-1] = {"role": "assistant", "content": model_response}
    else:
        # Error response
        new_chat_history[-1] = {"role": "assistant", "content": "Sorry, I couldn't generate a response."}

    yield new_chat_history

# --- Gradio Interface ---
def create_interface():
    logger.info("Creating Gradio interface")
    with gr.Blocks(css="""
        body {
            font-size: 16px;
        }
        h1 {
            font-size: 2.2em;
        }
        h3 {
            font-size: 1.4em;
        }
        .search-indicator {
            background-color: #f0f7ff;
            border-left: 4px solid #3498db;
            padding: 5px;
            margin-bottom: 10px;
            font-size: 0.8em;
            color: #333;
        }
        .container {
            max-width: 800px;
            margin: auto;
            font-size: 1.05 em;
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
            padding: 12px;
            background: linear-gradient(to right, #667eea, #764ba2);
            color: white;
            border-radius: 10px;
        }
        .header h1 { font-size: 2.2em; margin-bottom: 8px; }
        .header p { font-size: 1.1em; }
        .footer {
            text-align: center;
            margin-top: 20px;
            font-size: 0.95em;
            color: #444;
            padding: 10px;
        }
        .instructions {
            background-color: #e7e7e7; /* More neutral gray background */
            padding: 18px;
            border-radius: 8px;
            margin-bottom: 20px;
            color: #333;
        }
        .instructions ul {
            margin: 0 !important;
            padding-left: 0 !important;
        }
        .instructions li {
            margin: 0 !important;
            padding-left: 0 !important;
            color: #333;
        }
        .instructions li p {
            margin: 0 !important;
            padding-left: 0 !important;
            color: #333 !important;
        }
        .instructions h3 {
            margin-bottom: 15px;
            color: #222;
        }
        .instruction-item {
            margin-left: 0 !important;
            padding-left: 0 !important;
            margin-bottom: 12px;
            display: block;
            line-height: 1.5;
            color: #444 !important;
        }
        .instruction-item p {
            color: #333 !important;
        }
        code {
            background-color: #e0e0e0;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
            color: #333;
        }
        .context {
            color: #777777;
        }
        """) as demo:

        gr.HTML("""
        <div class="header">
            <h1>Ollama Chat with MCP</h1>
            <p>Chat with an AI that can search the web for up-to-date information</p>
        </div>
        """)

        with gr.Column(elem_classes="container"):
            gr.Markdown("""
            <div class="instructions">
                <h3>How to use this chat:</h3>
                <ul>
                    <li class="instruction-item"><p>Regular questions: Just type your question normally</p></li>
                    <li class="instruction-item"><p>Web search: Use <code>#search for "query" and your question</code></p></li>
                    <li class="instruction-item"><p>Clear history: Type <code>#clear</code> to reset the conversation</p></li>
                    <li class="instruction-item"><p>Exit: Type <code>exit</code> or <code>quit</code> to end the session</p></li>
                </ul>
            </div>
            """)

            # Service status indicator
            status = gr.Markdown(
                "🔄 Connecting to search service..." if not app_state.service_ready else
                "✅ Search service ready",
                elem_id="service-status"
            )

            # Fixed Chatbot component with updated parameters
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500,
                type="messages",  # Use messages format instead of tuples
                show_label=True,
                elem_id="chatbot"
            )

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Enter your message or #search for \"topic\" and your question...",
                    container=False,
                    scale=9
                )
                send = gr.Button("Send", scale=1)

            gr.HTML("""
            <div class="footer">
                <p>This chat assistant uses the MCP protocol to access a web search tool and local models via Ollama.</p>
                <script>
                    document.getElementById('current-date').textContent = new Date().toLocaleDateString('en-US', {
                        year: 'numeric', month: 'long', day: 'numeric'
                    });
                </script>
            </div>
            """)

            # Update service status
            def update_status():
                return "✅ Search service ready" if app_state.service_ready else "❌ Search service not available"

            # Initial status update
            demo.load(update_status, outputs=status)

            # Periodic status update using a different approach
            # Method 1: Try with gr.Interval
            try:
                status_update = gr.Interval(5, update_status)
                status_update.start()
                status_update.targets = [status]
            except (AttributeError, TypeError):
                # Method 2: Alternative approach
                try:
                    refresh = gr.Button("↻", size="sm", visible=False)
                    refresh.click(fn=update_status, outputs=status)

                    # Create a fake interval with JavaScript
                    gr.HTML("""
                    <script>
                    setInterval(function() {
                        document.querySelector('button[aria-label="↻"]').click();
                    }, 5000);
                    </script>
                    """)
                except Exception as e:
                    logger.warning(f"Could not set up periodic status updates: {e}")

            # Set up event handlers
            send_event = msg.submit(
                fn=process_message,
                inputs=[msg, chatbot],
                outputs=[chatbot],
                api_name="chat"
            ).then(
                lambda: "", # Clear input after sending
                None,
                [msg]
            )

            send.click(
                fn=process_message,
                inputs=[msg, chatbot],
                outputs=[chatbot],
                api_name="send"
            ).then(
                lambda: "", # Clear input after sending
                None,
                [msg]
            )

            # Add a loading message during processing
            send_event.then(lambda: gr.update(interactive=False), None, [send])
            send_event.then(lambda: gr.update(interactive=True), None, [send])

    logger.info("Gradio interface created successfully")
    return demo

# --- Main Application Logic ---
def main():
    try:
        # Start MCP service in a separate thread
        service_thread = threading.Thread(target=run_mcp_service, daemon=True)
        service_thread.start()
        logger.info("MCP service thread started")

        # Wait a moment for the service to start up
        time.sleep(2)

        # Create and launch the Gradio interface
        logger.info("Starting Gradio interface...")
        demo = create_interface()
        demo.queue()
        demo.launch(server_name="0.0.0.0", share=False)

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        logger.exception("Detailed error:")
    finally:
        # Cleanup
        logger.info("Shutting down...")
        logger.info("Application shutdown complete")

if __name__ == "__main__":
    # Set up signal handlers
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, exiting...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run the application
    main()
