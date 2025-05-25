# mcp_web_search_server.py
import os
import json
from dotenv import load_dotenv, find_dotenv
import httpx
from mcp.server.fastmcp import FastMCP
import logging
import sys

# Get a logger for this module specifically for setup and script-level messages
script_logger = logging.getLogger("server_search_script")
script_logger.setLevel(logging.INFO)
if not script_logger.hasHandlers():
    stderr_handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [SERVER_SEARCH] %(message)s')
    stderr_handler.setFormatter(formatter)
    script_logger.addHandler(stderr_handler)
    script_logger.propagate = False

script_logger.info(f"Script starting. Python Executable: {sys.executable}")
script_logger.info(f"Current Working Directory (CWD): {os.getcwd()}")

# Load environment variables from .env file
dotenv_path = find_dotenv(usecwd=False, raise_error_if_not_found=False)
if dotenv_path:
    script_logger.info(f"Loading .env file from: {dotenv_path}")
    load_dotenv(dotenv_path)
else:
    script_logger.warning("No .env file found by find_dotenv(). Relying on default load_dotenv() or existing environment variables.")
    load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
if SERPER_API_KEY and len(SERPER_API_KEY) > 5:
    script_logger.info(f"SERPER_API_KEY found (length: {len(SERPER_API_KEY)}).")
else:
    script_logger.critical(f"SERPER_API_KEY environment variable NOT FOUND or is too short.")
    raise ValueError("SERPER_API_KEY environment variable not set or invalid. Please ensure it's in your .env file or environment.")

SERPER_API_URL = "https://google.serper.dev/search"
script_logger.info("Environment variables processed. SERPER_API_URL configured.")

# Create an MCP server instance using FastMCP
# Note: Remove the explicit transport configuration - FastMCP handles this
mcp = FastMCP(
    name="WebSearchServer",
    version="0.1.0",
    display_name="Web Search Server (Serper.dev)",
    description="Provides web search functionality via the Serper.dev API.",
)
script_logger.info("FastMCP instance created.")

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
        script_logger.warning("web_search called with empty query.")
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
        script_logger.info(f"Performing search for: '{query}'")
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(SERPER_API_URL, headers=headers, data=payload)
        response.raise_for_status()
        search_results = response.json()
        script_logger.info(f"Search successful for: '{query}'")

        # Return the structured dict
        return {
            "status": "success",
            "message": f"Search completed for: {query}",
            "query": query,
            "organic_results": search_results.get("organic", []),
            "top_stories": search_results.get("topStories", []),
            "people_also_ask": search_results.get("peopleAlsoAsk", [])
        }
    except httpx.TimeoutException:
        script_logger.error(f"Timeout calling Serper.dev API for query '{query}'")
        return {"status": "error", "message": f"Timeout performing search for: {query}", "results": []} # Updated message
    except httpx.HTTPStatusError as e:
        error_detail = e.response.text
        try:
            error_detail = e.response.json()
        except json.JSONDecodeError:
            pass # Keep error_detail as text if not JSON
        script_logger.error(f"HTTP error for query '{query}': {e.response.status_code}. Response: {error_detail}")
        return {"status": "error", "message": f"API Error ({e.response.status_code}): {error_detail}", "results": []} # Updated message
    except Exception as e:
        script_logger.exception(f"Unexpected error during web_search for query '{query}': {e}")
        return {"status": "error", "message": str(e), "results": []}

script_logger.info("web_search tool defined.")

# For FastMCP, we don't need a main block - the fastmcp CLI handles server startup
# But keep this for direct execution if needed
if __name__ == "__main__":
    script_logger.info("Note: This FastMCP server should be run using 'fastmcp dev server_search.py' for development.")
    script_logger.info("If you need to run it directly, consider using the standard MCP server pattern instead.")
