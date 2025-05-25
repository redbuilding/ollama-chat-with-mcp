# mcp_web_search_server.py
import os
import json
import asyncio
from dotenv import load_dotenv
import httpx
from mcp.server.fastmcp import FastMCP
import logging # Import standard logging

# Load environment variables from .env file
load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
if not SERPER_API_KEY:
    # Use logging for this critical error before mcp.logger might be available
    logging.critical("SERPER_API_KEY environment variable not set. Please create a .env file with your key.")
    raise ValueError("SERPER_API_KEY environment variable not set. Please create a .env file with your key.")

SERPER_API_URL = "https://google.serper.dev/search"

# Create an MCP server instance using FastMCP
mcp = FastMCP(
    name="WebSearchServer",
    version="0.1.0",
    display_name="Web Search Server (Serper.dev)",
    description="Provides web search functionality via the Serper.dev API."
)

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
    # FastMCP will set up its own logger, which is accessible via mcp.logger
    # The initial print statement is replaced by a log after mcp instance is created.
    mcp.logger.info("Starting Web Search MCP Server...")
    try:
        mcp.run()
    except Exception as e:
        mcp.logger.exception("Web Search MCP Server crashed.")
    finally:
        mcp.logger.info("Web Search MCP Server stopped.")
