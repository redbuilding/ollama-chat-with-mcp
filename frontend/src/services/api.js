import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

export const sendMessage = async (userMessage, chatHistory, useSearch, conversationId = null, ollamaModelName = null) => {
  try {
    const payload = {
      user_message: userMessage,
      chat_history: chatHistory, // Current UI history, backend might load canonical from DB
      use_search: useSearch,
      conversation_id: conversationId,
    };
    if (ollamaModelName && !conversationId) { // Only send model for new chats
      payload.ollama_model_name = ollamaModelName;
    }
    const response = await apiClient.post("/chat", payload);
    // Response should now be { conversation_id: "...", chat_history: [...], ollama_model_name: "..." }
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
    // Expected: { service_ready: bool, db_connected: bool, ollama_available: bool }
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
    // Expected: ["model1:latest", "model2:latest", ...]
    return response.data;
  } catch (error) {
    console.error(
      "Error fetching Ollama models:",
      error.response ? error.response.data : error.message,
    );
    // Return an empty array or rethrow specific error type if needed by UI
    // For now, rethrowing to be handled by the caller
    throw error.response
      ? error.response.data
      : new Error("Network error or server unavailable fetching Ollama models");
  }
};

export const getConversations = async () => {
  try {
    const response = await apiClient.get("/conversations");
    // Expected: [{ id: "...", title: "...", created_at: "...", updated_at: "...", message_count: 0, ollama_model_name: "..." }, ...]
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
    // Expected: [{ role: "...", content: "...", is_html: false, timestamp: "..." }, ...]
    // This endpoint currently does not return the model name for the conversation,
    // it's assumed to be known from the conversation list or the chat response.
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
    return response.data; // Typically 204 No Content, so data might be undefined or empty
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
