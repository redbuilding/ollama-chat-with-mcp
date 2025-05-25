# mcp_web_search_server.py
import os
import json
import asyncio
from dotenv import load_dotenv
import httpx
from mcp.server.fastmcp import FastMCP

# Load environment variables from .env file
load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
if not SERPER_API_KEY:
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
        print(f"[Server] Performing search for: '{query}'")
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(SERPER_API_URL, headers=headers, data=payload)
        response.raise_for_status()
        search_results = response.json()
        print(f"[Server] Search successful for: '{query}'")
        
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
        print(f"[Server] Timeout calling Serper.dev API for query '{query}'")
        return {
            "status": "error",
            "message": f"Timeout calling Serper.dev API for query: {query}",
            "results": []
        }
    except httpx.HTTPStatusError as e:
        print(f"[Server] HTTP error for query '{query}': {e}")
        try:
            error_detail = e.response.json()
        except json.JSONDecodeError:
            error_detail = e.response.text
        return {
            "status": "error",
            "message": f"Error calling Serper.dev API: {error_detail}",
            "results": []
        }
    except Exception as e:
        print(f"[Server] Unexpected error: {e}")
        return {
            "status": "error",
            "message": str(e),
            "results": []
        }

if __name__ == "__main__":
    print("Starting Web Search MCP Server...")
    mcp.run()
    print("Server stopped.")