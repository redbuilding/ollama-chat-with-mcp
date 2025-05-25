import asyncio
import json
import re
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
import ollama

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"mcp_client_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mcp_client")

# --- Ollama Interaction ---
async def chat_with_ollama(messages: List[Dict[str, str]]) -> Optional[str]:
    """Sends a conversation history to a local Ollama  model and returns the response."""
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

# Improved extract_search_results function
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

# Improved format_search_results_for_prompt function
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

# --- Main Application Logic ---
async def main():
    current_date = datetime.now()
    current_month = current_date.strftime("%B")  # Full month name (e.g., "April")
    current_day = current_date.strftime("%d").lstrip("0")  # Day without leading zero
    current_year = current_date.strftime("%Y")  # 4-digit year
    current_date_formatted = f"{current_month} {current_day}, {current_year}"

    server_params = StdioServerParameters(
        command="python",
        args=["server_search.py"],
        env=None
    )

    # Initialize chat history
    chat_history = []

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_response = await session.list_tools()
            logger.info(f"Available Tools: {[tool.name for tool in tools_response.tools]}")

            if not any(tool.name == "web_search" for tool in tools_response.tools):
                logger.error("Error: Required 'web_search' tool not found.")
                return

            # --- Chat Loop ---
            print("\n--- Chat with model ---")
            print("Type your message. Use '#search for \"query\" and ask something' format.")
            print("Commands: #clear - Clear chat history, exit/quit - Exit the application")

            while True:
                try:
                    user_input = input("\nYou: ").strip()
                except EOFError:
                    print("\nExiting...")
                    break

                if user_input.lower() in ['exit', 'quit']:
                    break

                if user_input.lower() == '#clear':
                    chat_history = []
                    print("[System] Chat history cleared.")
                    continue

                # --- Advanced MCP Tool Handling ---
                final_prompt = user_input

                # Parse for search pattern: #search for "query" and [question]
                search_pattern = r'#search for "(.*?)"(?: and (.*))?'
                search_match = re.search(search_pattern, user_input, re.IGNORECASE)

                # Modified search handling in main()
                if search_match:
                    query = search_match.group(1).strip()
                    user_question = search_match.group(2).strip() if search_match.group(2) else None

                    logger.info(f"[MCP] Searching: '{query}', User question: '{user_question}'")
                    print(f"\n[MCP] Searching: '{query}'...")

                    try:
                        response = await session.call_tool("web_search", {"query": query})

                        # Log raw response type
                        logger.info(f"[MCP] Raw search response type: {type(response.content)}")

                        # Extract structured data from the response
                        extracted_results = extract_search_results(response.content)
                        logger.info(f"[MCP] Extracted results type: {type(extracted_results)}")
                        logger.info(f"[MCP] Extracted results: {extracted_results}")

                        # Format the results for the prompt
                        results_text = format_search_results_for_prompt(extracted_results, query)
                        logger.info(f"[MCP] Formatted results for prompt: {results_text[:500]}...")

                        # Create the final prompt based on whether there's a user question
                        if user_question:
                            final_prompt = (
                                f"Today's date is {current_date_formatted}. Here are CURRENT search results from {current_month} {current_year} about '{query}':\n\n"
                                f"{results_text}\n\n"
                                f"Using these UP-TO-DATE search results, answer this question: {user_question}\n\n"
                                f"Note: These search results ARE CURRENT as of {current_month} {current_year} and contain real information about events in {current_year}."
                            )
                        else:
                            final_prompt = (
                                f"Today's date is {current_date_formatted}. Here are CURRENT search results from {current_month} {current_year} about '{query}':\n\n"
                                f"{results_text}\n\n"
                                f"Using these UP-TO-DATE search results, summarize what they tell us about '{query}'.\n\n"
                                f"Note: These search results ARE CURRENT as of {current_month} {current_year} and contain real information about events in {current_year}."
                            )

                    except Exception as e:
                        logger.error(f"[MCP] Connection error: {e}")
                        logger.exception("Detailed error:")
                        final_prompt = f"Search unavailable for '{query}'. Please answer without web data: {user_question or query}"

                # Add user message to history
                chat_history.append({"role": "user", "content": final_prompt})

                # Get model response with full history
                if model_response := await chat_with_ollama(chat_history):
                    print(f"\nLLM: {model_response}")
                    # Add model response to history
                    chat_history.append({"role": "assistant", "content": model_response})
                else:
                    print("\nLLM: Sorry, I couldn't generate a response.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
