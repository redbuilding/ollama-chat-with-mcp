# mcp_web_search_server.py
import os
import json
import asyncio
from dotenv import load_dotenv, find_dotenv # Import find_dotenv
import httpx
from mcp.server.fastmcp import FastMCP
import logging # Import standard logging

# Get a logger for this module specifically for setup and script-level messages
# This logger will be used for messages outside of mcp.run() or mcp.tool() contexts
script_logger = logging.getLogger("server_search_script")
# Basic configuration for the script_logger if this script is run directly.
# If run as a subprocess, its logs to stderr should be captured by main.py's MCP client logger.
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# Create an MCP server instance using FastMCP
mcp = FastMCP(
    name="WebSearchServer",
    version="0.1.0",
    display_name="Web Search Server (Serper.dev)",
    description="Provides web search functionality via the Serper.dev API."
)

# Load environment variables from .env file
# Make .env loading more robust: search from script directory upwards
dotenv_path = find_dotenv(usecwd=False, raise_error_if_not_found=False)
if dotenv_path:
    script_logger.info(f"Loading .env file from: {dotenv_path}")
    load_dotenv(dotenv_path)
else:
    script_logger.warning("No .env file found by find_dotenv(). Relying on default load_dotenv() behavior or existing environment variables.")
    load_dotenv() # Try default behavior (e.g. CWD)

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
if not SERPER_API_KEY:
    script_logger.critical("SERPER_API_KEY environment variable not set. Please ensure it's in your .env file or environment.")
    # Raising an error here will cause the script to exit.
    raise ValueError("SERPER_API_KEY environment variable not set. Please create a .env file with your key or set it in the environment.")

SERPER_API_URL = "https://google.serper.dev/search"


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
    # Inside the tool, mcp.logger is expected to be available and configured by the MCP framework.
    if not query:
        mcp.logger.warning("web_search called with empty query.")
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
        
        # Format the response in a standardized structure
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
        return {
            "status": "error",
            "message": f"Timeout calling Serper.dev API for query: {query}",
            "results": []
        }
    except httpx.HTTPStatusError as e:
        error_detail_text = e.response.text
        try:
            error_detail_json = e.response.json()
            mcp.logger.error(f"HTTP error for query '{query}': {e}. Response: {error_detail_json}")
            error_detail = error_detail_json
        except json.JSONDecodeError:
            mcp.logger.error(f"HTTP error for query '{query}': {e}. Response (not JSON): {error_detail_text}")
            error_detail = error_detail_text
        return {
            "status": "error",
            "message": f"Error calling Serper.dev API: {error_detail}",
            "results": []
        }
    except Exception as e:
        mcp.logger.exception(f"Unexpected error during web_search for query '{query}': {e}")
        return {
            "status": "error",
            "message": str(e),
            "results": []
        }

if __name__ == "__main__":
    # Use the script_logger for messages in this block.
    script_logger.info("Starting Web Search MCP Server (server_search.py directly)...")
    try:
        mcp.run()
    except Exception as e: 
        script_logger.exception("Web Search MCP Server (server_search.py directly) crashed during run.")
    finally:
        script_logger.info("Web Search MCP Server (server_search.py directly) stopped.")
