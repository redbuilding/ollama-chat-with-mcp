import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

export const sendMessage = async (userMessage, chatHistory, useSearch, conversationId = null) => {
  try {
    const response = await apiClient.post("/chat", {
      user_message: userMessage,
      chat_history: chatHistory, // Current UI history, backend might load canonical from DB
      use_search: useSearch,
      conversation_id: conversationId,
    });
    // Response should now be { conversation_id: "...", chat_history: [...] }
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
    // Expected: { service_ready: bool, ollama_model: str, db_connected: bool }
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

export const getConversations = async () => {
  try {
    const response = await apiClient.get("/conversations");
    // Expected: [{ id: "...", title: "...", created_at: "...", updated_at: "...", message_count: 0 }, ...]
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
