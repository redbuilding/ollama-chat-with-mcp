# 🔍 🤖 🌐 Ollama Chat with MCP (Refactored)

A powerful demonstration of integrating local LLMs with real-time web search capabilities and other tools using the Model Context Protocol (MCP), featuring a modern web interface and persistent conversations.

## Overview

Ollama Chat with MCP showcases how to extend a local language model's capabilities through tool use. This application combines the power of locally running LLMs via Ollama with up-to-date web search functionality and database querying, all managed through a robust backend and a user-friendly React frontend. Conversations are persisted using MongoDB.

The project consists of several key components:
- **Backend (FastAPI)**: Manages chat logic, Ollama interactions, MCP service communication, and conversation persistence.
- **Frontend (React)**: A modern, responsive web interface for users to interact with the chat application.
- **MCP Web Search Server**: Provides web search functionality using the Serper.dev API.
- **MCP SQL Server**: Provides a tool to query a MySQL database (example).
- **MongoDB**: Stores conversation history and user data.

This architecture demonstrates how MCP enables local models to access external tools and data sources, significantly enhancing their capabilities, now with a more scalable and feature-rich setup.

## Features

- 🔎 **Web-enhanced chat**: Access real-time web search results during conversation.
- 💾 **Persistent Conversations**: Chat history is saved in MongoDB, allowing users to resume conversations.
- 🧠 **Local model execution**: Uses Ollama to run models entirely on your own hardware.
- 🔌 **MCP integration**: Demonstrates practical implementation of the Model Context Protocol for multiple tools.
- 💻 **Modern Web Interface**: Built with React for a responsive and interactive user experience.
- 📊 **Structured search results**: Clean formatting of web search data for optimal context.
- ⚙️ **Backend API**: FastAPI backend providing robust API endpoints for chat and conversation management.
- 🗃️ **SQL Querying Tool**: Example MCP tool for interacting with a MySQL database.
- 🔄 **Conversation Management**: List, rename, and delete conversations.

## Requirements

- Python 3.11+
- Node.js (v18+) and npm/yarn for the frontend
- [Ollama](https://ollama.com/) installed and running locally
- A [Serper.dev](https://serper.dev/) API key (free tier available)
- MongoDB instance (local or cloud)
- MySQL server (optional, for the SQL querying tool)
- Internet connection for web searches and package downloads

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/redbuilding/ollama-chat-with-mcp.git
    cd ollama-chat-with-mcp
    ```

2.  **Set up Backend:**
    *   Navigate to the backend directory:
        ```bash
        cd backend
        ```
    *   Install Python dependencies:
        ```bash
        pip install -r requirements.txt
        ```
        *(Note: You might need to create a `requirements.txt` for the backend if it's not already there, based on imports in `main.py`, `server_search.py`, `server_mysql.py`)*
    *   Create a `.env` file in the `backend` directory (or project root, depending on configuration) with your API keys and database URI:
        ```
        SERPER_API_KEY=your_serper_api_key_here
        # MONGODB_URI=your_mongodb_connection_string # Add if backend/main.py uses it for DB
        # DB_HOST=localhost # For server_mysql.py
        # DB_USER=root
        # DB_PASSWORD=
        # DB_NAME=sample_db
        ```

3.  **Set up Frontend:**
    *   Navigate to the frontend directory:
        ```bash
        cd ../frontend
        ```
    *   Install Node.js dependencies:
        ```bash
        npm install
        # or
        # yarn install
        ```

4.  **Ensure Ollama is installed and a model is available:**
    The application might default to a specific model (e.g., qwen2:7b, llama3). Pull your desired model:
    ```bash
    ollama pull qwen2:7b
    # or your preferred model
    ```
    You can select the model in the UI if supported, or configure it in the backend.

## Usage

1.  **Start the MCP Servers:**
    These servers provide tools for the LLM. They need to be running for the backend to utilize their capabilities.
    *   **Web Search Server:**
        ```bash
        # From the project root directory
        python backend/server_search.py
        ```
    *   **SQL Server (Optional):**
        ```bash
        # From the project root directory
        python backend/server_mysql.py
        ```
    *(Note: The main backend application might manage these MCP services as part of its startup, or they might need to be run as separate processes. Adjust based on `backend/main.py`'s design.)*

2.  **Start the Backend Server:**
    Navigate to the `backend` directory (if not already there) and run the FastAPI application:
    ```bash
    # From the backend directory
    uvicorn main:app --reload --port 8000
    ```
    The backend API will typically be available at `http://localhost:8000`.

3.  **Start the Frontend Development Server:**
    Navigate to the `frontend` directory and run:
    ```bash
    npm run dev
    # or
    # yarn dev
    ```
    The web interface will typically be accessible at `http://localhost:5173` (or another port specified by Vite/Create React App).

### Interacting with the Application

-   Open your browser to the frontend URL.
-   Use the chat interface to send messages.
-   Web search can be triggered by specific phrasing (e.g., "search the web for...") or a dedicated search toggle, depending on UI implementation.
-   Manage conversations using the sidebar (create new, select, rename, delete).

### Legacy Clients (Phasing Out)

The original `chat_client.py` (terminal) and `chat_frontend.py` (Gradio) are still in the repository but are not part of the main refactored application. They may not work correctly with the new backend or may be removed in future updates.
-   Terminal Client: `python chat_client.py`
-   Gradio Web Interface: `python chat_frontend.py`

## How It Works

1.  The **User** interacts with the **React Frontend**.
2.  The **Frontend** sends requests (chat messages, conversation management) to the **FastAPI Backend** API.
3.  For chat messages, the **Backend** processes the request:
    *   It may interact with **Ollama** to get a response from the local LLM.
    *   If the LLM or user input indicates a need for a tool (e.g., web search, SQL query), the backend communicates with the relevant **MCP Server** (e.g., `server_search.py`, `server_mysql.py`).
    *   The MCP Server executes the tool (e.g., calls Serper.dev API, queries MySQL).
    *   Results from the MCP Server are returned to the Backend.
    *   The Backend may re-prompt Ollama with the tool's results to generate a final, informed response.
4.  The **Backend** stores/retrieves conversation history from **MongoDB**.
5.  The final response is sent back to the **Frontend** and displayed to the user.

## File Structure (Simplified)

```
ollama-chat-with-mcp/
├── backend/
│   ├── main.py             # FastAPI application, API endpoints, chat logic
│   ├── server_search.py    # MCP server for web search
│   ├── server_mysql.py     # MCP server for SQL queries
│   ├── requirements.txt    # Python dependencies for backend
│   └── .env.example        # Example environment variables for backend
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # Main React application component
│   │   ├── components/     # React UI components (ChatInput, ChatMessage, etc.)
│   │   ├── services/       # API interaction logic (api.js)
│   │   └── main.jsx        # Entry point for React app
│   ├── package.json        # Node.js dependencies and scripts
│   ├── vite.config.js      # Vite configuration (if using Vite)
│   └── public/             # Static assets
├── chat_client.py          # (Legacy) Terminal-based chat client
├── chat_frontend.py        # (Legacy) Gradio web interface client
├── README.md               # This file
└── .env.example            # Project-level example environment variables (if any)
```

## Customization

-   **Ollama Model**: Change the default Ollama model in the backend (`backend/main.py` or its configuration) or select from available models in the UI.
-   **Search Results**: Adjust the number of search results processed in `backend/main.py` or `backend/server_search.py`.
-   **Prompt Engineering**: Modify the prompts sent to Ollama in `backend/main.py` to tailor responses.
-   **Styling**: Customize the frontend appearance by modifying CSS files or styles within the React components in `frontend/src/`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
