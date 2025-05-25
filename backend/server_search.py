# mcp_web_search_server.py
import os
import json
import asyncio
from dotenv import load_dotenv, find_dotenv 
import httpx
from mcp.server.fastmcp import FastMCP
from mcp.transport.tcp import TCPTransportConfiguration # Import TCP transport
import logging 
import sys 

# Get a logger for this module specifically for setup and script-level messages
script_logger = logging.getLogger("server_search_tcp_script") # Renamed logger for clarity
script_logger.setLevel(logging.INFO)
if not script_logger.hasHandlers(): 
    stderr_handler = logging.StreamHandler(sys.stderr) 
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [SERVER_SEARCH_TCP] %(message)s') 
    stderr_handler.setFormatter(formatter)
    script_logger.addHandler(stderr_handler)
    script_logger.propagate = False 

script_logger.info(f"Script starting. Python Executable: {sys.executable}")
script_logger.info(f"Python sys.path: {sys.path}")
script_logger.info(f"Current Working Directory (CWD): {os.getcwd()}")
script_logger.info(f"File __file__: {__file__}")
script_logger.info(f"Absolute path of __file__: {os.path.abspath(__file__)}")

# Define TCP server host and port
TCP_HOST = "localhost"
TCP_PORT = 9000 # You can choose a different port if needed

# Create an MCP server instance using FastMCP
# Configure it to use TCP transport
mcp = FastMCP(
    name="WebSearchTCPServer", # Renamed for clarity
    version="0.1.0",
    display_name="Web Search Server (Serper.dev via TCP)",
    description="Provides web search functionality via the Serper.dev API over TCP.",
    transports=[
        TCPTransportConfiguration(host=TCP_HOST, port=TCP_PORT)
    ]
)
script_logger.info(f"FastMCP instance created, configured for TCP on {TCP_HOST}:{TCP_PORT}.")

# Load environment variables from .env file
dotenv_path = find_dotenv(usecwd=False, raise_error_if_not_found=False)
if dotenv_path:
    script_logger.info(f"Loading .env file from: {dotenv_path}")
    load_dotenv(dotenv_path)
else:
    script_logger.warning("No .env file found by find_dotenv(). Relying on default load_dotenv() or existing environment variables.")
    load_dotenv() 

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
if SERPER_API_KEY and len(SERPER_API_KEY) > 5 : 
    script_logger.info(f"SERPER_API_KEY found (length: {len(SERPER_API_KEY)}).")
else:
    script_logger.critical(f"SERPER_API_KEY environment variable NOT FOUND or is too short (length: {len(SERPER_API_KEY) if SERPER_API_KEY else 0}). This script will now raise ValueError.")
    raise ValueError("SERPER_API_KEY environment variable not set or invalid. Please ensure it's in your .env file or environment.")

SERPER_API_URL = "https://google.serper.dev/search"
script_logger.info("Environment variables processed. SERPER_API_URL configured.")


# Register an asynchronous tool for performing web searches
@mcp.tool()
async def web_search(query: str) -> dict:
    """
    Performs a web search using the Serper.dev API and returns the JSON response.
    
    Args:
        query: The search query string
        
    Returns:
        A standardized dictionary containing search results or error information
    """
    if not query:
        mcp.logger.warning("web_search called with empty query.") # mcp.logger is fine inside tools
        return {
            "status": "error",
            "message": "Missing required parameter 'query' for web_search tool.",
            "results": []
        }
    
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }
    payload = json.dumps({"q": query})
    
    try:
        mcp.logger.info(f"Performing search for: '{query}'")
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(SERPER_API_URL, headers=headers, data=payload)
        response.raise_for_status()
        search_results = response.json()
        mcp.logger.info(f"Search successful for: '{query}'")
        
        return {
            "status": "success",
            "message": f"Search completed for: {query}",
            "query": query,
            "organic_results": search_results.get("organic", []),
            "top_stories": search_results.get("topStories", []),
            "people_also_ask": search_results.get("peopleAlsoAsk", [])
        }
    except httpx.TimeoutException:
        mcp.logger.error(f"Timeout calling Serper.dev API for query '{query}'")
        return { "status": "error", "message": f"Timeout: {query}", "results": [] }
    except httpx.HTTPStatusError as e:
        error_detail = e.response.text
        try: error_detail = e.response.json()
        except json.JSONDecodeError: pass
        mcp.logger.error(f"HTTP error for query '{query}': {e}. Response: {error_detail}")
        return { "status": "error", "message": f"API Error: {error_detail}", "results": []}
    except Exception as e:
        mcp.logger.exception(f"Unexpected error during web_search for query '{query}': {e}")
        return { "status": "error", "message": str(e), "results": [] }

script_logger.info("web_search tool defined.")

if __name__ == "__main__":
    if not logging.getLogger().hasHandlers() and not script_logger.hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [SERVER_SEARCH_TCP __main__] %(message)s')
    
    script_logger.info(f"Attempting to start MCP TCP server on {TCP_HOST}:{TCP_PORT}...")
    try:
        mcp.run() # This will now start the TCP server based on FastMCP config
    except Exception as e: 
        script_logger.exception("MCP TCP server crashed during run.")
    finally:
        script_logger.info("MCP TCP server finished or crashed.")
else:
    script_logger.info("Script is NOT being run as __main__. If this is the MCP server, it needs to be run directly.")
