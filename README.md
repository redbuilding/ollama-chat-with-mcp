# 🔍 🤖 🌐 OhSee (Ollama Chat with MCP)

A powerful demonstration of integrating local LLMs with real-time web search capabilities and database querying using the Model Context Protocol (MCP), featuring a modern web interface and persistent conversations.

## Overview

Ollama Chat with MCP showcases how to extend a local language model's capabilities through tool use. This application combines the power of locally running LLMs via Ollama with up-to-date web search functionality and database querying, all managed through a robust backend and a user-friendly React frontend. Conversations are persisted using MongoDB.

The project consists of several key components:

- **Backend (FastAPI)**: Manages chat logic, Ollama interactions, MCP service communication (including starting and managing MCP services like web search and SQL querying), and conversation persistence.

- **Frontend (React)**: A modern, responsive web interface for users to interact with the chat application.

- **MCP Web Search Server Module**: Provides web search functionality using the Serper.dev API, run as a service by the backend.

- **MCP SQL Server Module**: Provides a tool to query a MySQL database, run as a service by the backend.

- **MongoDB**: Stores conversation history and user data.

This architecture demonstrates how MCP enables local models to access external tools and data sources, significantly enhancing their capabilities, now with a more scalable and feature-rich setup.

## Features

- 🔎 **Web-enhanced chat**: Access real-time web search results during conversation.
- 💾 **Persistent Conversations**: Chat history is saved in MongoDB, allowing users to resume conversations.
- 🧠 **Local model execution**: Uses Ollama to run models entirely on your own hardware.
- 🔌 **MCP integration**: Demonstrates practical implementation of the Model Context Protocol for multiple tools, managed by the main backend.
- 💻 **Modern Web Interface**: Built with React for a responsive and interactive user experience.
- 📊 **Structured search results**: Clean formatting of web search data for optimal context.
- ⚙️ **Backend API**: FastAPI backend providing robust API endpoints for chat and conversation management.
- 🗃️ **SQL Querying Tool**: MCP tool for interacting with a MySQL database, with improved schema handling and query retry logic.
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
    *   Create a `.env` file in the `backend` directory. This is where the backend and its managed MCP services will look for environment variables:
        ```dotenv
        # For Web Search (server_search.py)
        SERPER_API_KEY=your_serper_api_key_here

        # For MongoDB (main.py)
        MONGODB_URI=mongodb://localhost:27017/
        MONGODB_DATABASE_NAME=mcp_chat_db

        # For MySQL Database Querying (server_mysql.py)
        DB_HOST=localhost
        DB_USER=your_db_user
        DB_PASSWORD=your_db_password
        DB_NAME=your_db_name
        ```

3.  **Set up Frontend:**
    *   Navigate to the frontend directory:
        ```bash
        cd ../frontend
        ```
        (If you were in `backend/`, otherwise navigate from project root: `cd frontend`)
    *   Install Node.js dependencies:
        ```bash
        npm install
        # or
        # yarn install
        ```

4.  **Ensure Ollama is installed and a model is available:**
    The application might default to a specific model (e.g., `llama3.1`). Pull your desired model:
    ```bash
    ollama pull llama3.1
    # or your preferred model like llama3, mistral, etc.
    ```
    You can select the model in the UI if supported, or configure a default in the backend's `.env` file.

## Usage

1.  **Ensure Prerequisites are Running:**
    *   **Ollama**: Must be running.
    *   **MongoDB**: Your MongoDB instance must be accessible.
    *   **MySQL Server** (if using the SQL tool): Your MySQL server must be running and accessible with the credentials provided in `.env`.

2.  **Start the Backend Server:**
    Navigate to the `backend` directory and run the FastAPI application:
    ```bash
    # From the backend directory
    uvicorn main:app --reload --port 8000
    ```
    The backend API will typically be available at `http://localhost:8000`.
    The FastAPI application will automatically start and manage the MCP services (Web Search and SQL Querying) as background processes using its lifespan manager. You do **not** need to run `server_search.py` or `server_mysql.py` separately.

3.  **Start the Frontend Development Server:**
    Navigate to the `frontend` directory and run:
    ```bash
    npm run dev
    # or
    # yarn dev
    ```
    The web interface will typically be accessible at `http://localhost:5173` (or another port specified by Vite).

### Interacting with the Application

-   Open your browser to the frontend URL (e.g., `http://localhost:5173`).
-   Use the chat interface to send messages.
-   Toggle "Use Web Search" or "Use Database" switches in the UI to enable these tools for your next message.
-   Manage conversations using the sidebar (create new, select, rename, delete).

### Legacy Clients (Phasing Out)

The original `chat_client.py` (terminal) and `chat_frontend.py` (Gradio) are still in the repository but are not part of the main refactored application. They are not maintained and may not work correctly with the new backend, and will be phased out in a future update.

## How It Works

1.  The **User** interacts with the **React Frontend**.
2.  The **Frontend** sends requests (chat messages, conversation management) to the **FastAPI Backend** API.
3.  For chat messages, the **Backend** (`main.py`) processes the request:
    *   It may interact with **Ollama** to get a response from the local LLM.
    *   If the user enables a tool (e.g., web search, SQL query) via the UI:
        *   The backend prepares the necessary context (e.g., fetching database schema for SQL).
        *   It may prompt the Ollama model to generate a tool-specific input (e.g., a SQL query).
        *   The backend then communicates with the relevant **MCP Service Module** (managed as a subprocess, e.g., `server_search.py` logic for web search, `server_mysql.py` logic for SQL execution).
        *   The MCP Service Module executes the tool (e.g., calls Serper.dev API, queries MySQL).
        *   Results from the MCP Service Module are returned to the Backend.
        *   The Backend may re-prompt Ollama with the tool's results to generate a final, informed response.
4.  The **Backend** stores/retrieves conversation history from **MongoDB**.
5.  The final response is sent back to the **Frontend** and displayed to the user.


## Customization

-   **Ollama Model**: Select from available models in the UI. Note that models well suited to coding may work better with the MCP SQL chat (e.g., codestral).
-   **Search Results**: Adjust the number of search results processed in `backend/main.py`.
-   **Database Schema Context**: Modify `MAX_TABLES_FOR_SCHEMA_CONTEXT` in `backend/main.py` to control how many tables' schemas are sent to the LLM.
-   **Prompt Engineering**: Modify the system prompts sent to Ollama in `backend/main.py` to tailor responses and SQL generation.
-   **Styling**: Customize the frontend appearance by modifying CSS files or styles within the React components in `frontend/src/`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
