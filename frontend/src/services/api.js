import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

export const sendMessage = async (userMessage, chatHistory, useSearch, useDatabase, conversationId = null, ollamaModelName = null) => {
  try {
    const payload = {
      user_message: userMessage,
      chat_history: chatHistory,
      use_search: useSearch,
      use_database: useDatabase, // New flag
      conversation_id: conversationId,
    };
    if (ollamaModelName && !conversationId) {
      payload.ollama_model_name = ollamaModelName;
    }
    const response = await apiClient.post("/chat", payload);
    return response.data;
  } catch (error) {
    console.error(
      "Error sending message:",
      error.response ? error.response.data : error.message,
    );
    throw error.response
      ? error.response.data
      : new Error("Network error or server unavailable");
  }
};

export const getServiceStatus = async () => {
  try {
    const response = await apiClient.get("/status");
    // Expected: { 
    //   db_connected: bool, 
    //   ollama_available: bool,
    //   mcp_services: { 
    //     web_search_service: { ready: bool }, 
    //     mysql_db_service: { ready: bool } 
    //   } 
    // }
    return response.data;
  } catch (error) {
    console.error(
      "Error fetching service status:",
      error.response ? error.response.data : error.message,
    );
    throw error.response
      ? error.response.data
      : new Error("Network error or server unavailable");
  }
};

export const getOllamaModels = async () => {
  try {
    const response = await apiClient.get("/ollama-models");
    return response.data;
  } catch (error) {
    console.error(
      "Error fetching Ollama models:",
      error.response ? error.response.data : error.message,
    );
    throw error.response
      ? error.response.data
      : new Error("Network error or server unavailable fetching Ollama models");
  }
};

export const getConversations = async () => {
  try {
    const response = await apiClient.get("/conversations");
    return response.data;
  } catch (error) {
    console.error(
      "Error fetching conversations:",
      error.response ? error.response.data : error.message,
    );
    throw error.response
      ? error.response.data
      : new Error("Network error or server unavailable");
  }
};

export const getConversationMessages = async (conversationId) => {
  try {
    const response = await apiClient.get(`/conversations/${conversationId}`);
    return response.data;
  } catch (error) {
    console.error(
      `Error fetching messages for conversation ${conversationId}:`,
      error.response ? error.response.data : error.message,
    );
    throw error.response
      ? error.response.data
      : new Error("Network error or server unavailable");
  }
};

export const deleteConversation = async (conversationId) => {
  try {
    const response = await apiClient.delete(`/conversations/${conversationId}`);
    return response.data; 
  } catch (error) {
    console.error(
      `Error deleting conversation ${conversationId}:`,
      error.response ? error.response.data : error.message,
    );
    throw error.response
      ? error.response.data
      : new Error("Network error or server unavailable");
  }
};

export const renameConversation = async (conversationId, newTitle) => {
  try {
    const response = await apiClient.put(`/conversations/${conversationId}/rename`, { new_title: newTitle });
    return response.data; 
  } catch (error) {
    console.error(
      `Error renaming conversation ${conversationId}:`,
      error.response ? error.response.data : error.message,
    );
    throw error.response 
      ? error.response.data 
      : new Error("Network error or server unavailable");
  }
};
