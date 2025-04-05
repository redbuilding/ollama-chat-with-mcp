# 🔍 🤖 🌐 Ollama Chat with MCP

A powerful demonstration of integrating local LLMs with real-time web search capabilities using the Model Context Protocol (MCP).

## Overview

Ollama Chat with MCP showcases how to extend a local language model's capabilities through tool use. This application combines the power of locally running LLMs via Ollama with up-to-date web search functionality provided by an MCP server.

The project consists of three main components:
- **MCP Web Search Server**: Provides web search functionality using the Serper.dev API
- **Terminal Client**: A CLI interface for chat and search interactions
- **Web Frontend**: A user-friendly Gradio-based web interface

By using this architecture, the application demonstrates how MCP enables local models to access external tools and data sources, significantly enhancing their capabilities.

## Features

- 🔎 **Web-enhanced chat**: Access real-time web search results during conversation
- 🧠 **Local model execution**: Uses Ollama to run models entirely on your own hardware
- 🔌 **MCP integration**: Demonstrates practical implementation of the Model Context Protocol
- 🌐 **Dual interfaces**: Choose between terminal CLI or web-based GUI
- 📊 **Structured search results**: Clean formatting of web search data for optimal context
- 🔄 **Conversation memory**: Maintains context throughout the chat session

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com/) installed and running locally
- A [Serper.dev](https://serper.dev/) API key (free tier available)
- Internet connection for web searches

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/redbuilding/ollama-chat-with-mcp.git
   cd ollama-chat-with-mcp
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your Serper.dev API key:
   ```
   SERPER_API_KEY=your_serper_api_key_here
   ```

4. Ensure Ollama is installed and the hardcoded model is available (default qwen2.5:14b):
   ```bash
   ollama pull qwen2.5:14b
   ```

## Usage

### Starting the Web Interface

To use the web-based interface:

```bash
python chat_frontend.py
```

This will start the Gradio web interface, typically accessible at http://localhost:7860

### Using the Terminal Client

To use the command-line interface:

```bash
python chat_client.py
```

### Search Commands

In both interfaces, you can use special commands to trigger web searches:

- Search and summarize: `#search for "financial market outlook April 2025"`
- Search and answer a question: `#search for "reality TV this week" and what happened recently?`

### Other Commands

- Clear conversation history: `#clear`
- Exit the application: `exit` or `quit`

## How It Works

1. The MCP server exposes a web search capability as a tool
2. When a user requests search information, the client sends a query to the MCP server
3. The server processes the request through Serper.dev and returns formatted results
4. The client constructs an enhanced prompt including the search results
5. The local Ollama model receives this prompt and generates an informed response
6. The response is displayed to the user with search attribution

## Architecture

```
┌────────── ─┐      ┌──────────┐      ┌────── ┐
│  Chat Clients  │   <---->│   MCP Server    │<---->│ Serper.dev│
│ (CLI or Gradio)│         │  (server.py)    │      │  API      │
└───────┬─── ┘      └──────────┘      └───────┘
        │
        ▼
┌─────────┐
│     Ollama     │
│  Local LLM     │
└─────────┘
```

## File Structure

- `server.py` - MCP server with web search tool
- `chat_client.py` - Terminal-based chat client
- `chat_frontend.py` - Gradio web interface client
- `requirements.txt` - Project dependencies
- `.env` - Configuration for API keys (create this file & add your key for Serper)

## Customization

- Change the Ollama model by modifying the model name in the chat client files
- Adjust the number of search results by changing the `max_results` parameter
- Modify the prompt templates to better suit your specific use case

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
