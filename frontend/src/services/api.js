import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

const apiClient = axios.create({
  baseURL: API_URL,

  headers: {
    "Content-Type": "application/json",
  },
});

export const sendMessage = async (userMessage, chatHistory, useSearch) => {
  try {
    const response = await apiClient.post("/chat", {
      user_message: userMessage,

      chat_history: chatHistory,

      use_search: useSearch,
    });

    return response.data.chat_history;
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
