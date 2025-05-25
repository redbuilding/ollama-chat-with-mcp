import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import ChatMessage from './components/ChatMessage';
import ChatInput from './components/ChatInput';
import ConversationSidebar from './components/ConversationSidebar';
import { 
  sendMessage, 
  getServiceStatus, 
  getConversations, 
  getConversationMessages 
} from './services/api';
import { AlertTriangle, Wifi, Server, Database, Loader2 } from 'lucide-react';

const App = () => {
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false); // For message sending
  const [error, setError] = useState(null); // General errors for chat window (send/load specific chat)
  const [isSearchActive, setIsSearchActive] = useState(false);
  
  // Service Status
  const [mcpServiceReady, setMcpServiceReady] = useState(false);
  const [ollamaModel, setOllamaModel] = useState('');
  const [dbConnected, setDbConnected] = useState(false);

  // Conversation Management
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [conversations, setConversations] = useState([]);
  const [isConversationsLoading, setIsConversationsLoading] = useState(true);
  const [conversationsError, setConversationsError] = useState(null); // Specific error for conversations list
  const [isChatHistoryLoading, setIsChatHistoryLoading] = useState(false);


  const chatContainerRef = useRef(null);

  const initialWelcomeMessage = useMemo(() => ({ 
    role: 'assistant', 
    content: "Hello! I'm your AI assistant. Toggle the search icon to enable web search for up-to-date answers. Select 'New Chat' to begin or choose a past conversation.",
    timestamp: new Date().toISOString() 
  }), []);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatHistory]);

  const fetchServiceStatus = useCallback(async () => {
    try {
      const status = await getServiceStatus();
      setMcpServiceReady(status.service_ready);
      setOllamaModel(status.ollama_model || 'N/A');
      setDbConnected(status.db_connected);
      if (!status.db_connected) { // If DB is not connected, reflect this in conversations error
        setConversationsError("Database not connected. History is unavailable.");
      }
    } catch (err) {
      setError('Failed to connect to backend services. Status polling stopped.');
      setMcpServiceReady(false);
      setDbConnected(false);
      setConversationsError("Failed to fetch service status. History may be unavailable.");
    }
  }, []);

  useEffect(() => {
    fetchServiceStatus();
    const intervalId = setInterval(fetchServiceStatus, 15000);
    return () => clearInterval(intervalId);
  }, [fetchServiceStatus]);

  const fetchConversationsList = useCallback(async () => {
    if (!dbConnected) {
      setConversations([]);
      setIsConversationsLoading(false);
      // conversationsError might be already set by fetchServiceStatus if db is not connected
      if (!conversationsError) setConversationsError("Database not connected. History is unavailable.");
      return;
    }
    setIsConversationsLoading(true);
    setConversationsError(null); // Clear previous error before trying again
    try {
      const convs = await getConversations();
      setConversations(convs || []);
      if (!convs || convs.length === 0) {
        // Optional: set a specific message if list is empty but no error, e.g. "No past conversations."
        // This is handled in sidebar, but good to be aware of.
      }
    } catch (err) {
      const errorDetail = err.detail || err.message || 'Failed to fetch conversations list.';
      setConversationsError(errorDetail);
      setConversations([]); 
    } finally {
      setIsConversationsLoading(false);
    }
  }, [dbConnected, conversationsError]); // Added conversationsError to dependencies to re-evaluate if it changes

  useEffect(() => {
    // Fetch conversations only if DB is connected.
    // The fetchConversationsList callback itself checks dbConnected,
    // but this condition here prevents an unnecessary call if dbConnected is false initially.
    if (dbConnected) {
        fetchConversationsList();
    } else {
        // Ensure list is clear and loading is false if DB is not connected
        setConversations([]);
        setIsConversationsLoading(false);
        if (!conversationsError) { // If no specific error yet, set the default one
             setConversationsError("Database not connected. History is unavailable.");
        }
    }
  }, [dbConnected, fetchConversationsList, conversationsError]);


  useEffect(() => {
    const loadMessages = async () => {
      if (currentConversationId) {
        setIsChatHistoryLoading(true);
        setError(null); // Clear general chat window error before loading messages
        try {
          const messages = await getConversationMessages(currentConversationId);
          setChatHistory(messages || []);
        } catch (err) {
          const errorDetail = err.detail || err.message || `Failed to load messages for conversation.`;
          setError(errorDetail); // Set general chat window error
          setChatHistory([initialWelcomeMessage]); 
        } finally {
          setIsChatHistoryLoading(false);
        }
      } else {
        setChatHistory([initialWelcomeMessage]);
        setIsChatHistoryLoading(false);
      }
    };
    loadMessages();
  }, [currentConversationId, initialWelcomeMessage]);


  const handleSendMessage = async (userInput) => {
    setIsLoading(true);
    setError(null); // Clear general error

    try {
      const response = await sendMessage(userInput, chatHistory, isSearchActive, currentConversationId);
      
      setChatHistory(response.chat_history || []);
      
      if (response.conversation_id && response.conversation_id !== currentConversationId) {
        setCurrentConversationId(response.conversation_id);
        if (!currentConversationId || currentConversationId !== response.conversation_id) {
            await fetchConversationsList(); 
        }
      } else if (!response.conversation_id && currentConversationId) {
        setCurrentConversationId(null); // Should not happen ideally
        await fetchConversationsList();
      } else if (response.conversation_id === currentConversationId) {
        // If message sent in existing conversation, refresh list to update its timestamp/message_count
        await fetchConversationsList();
      }
      
    } catch (err) {
      const errorMessage = err.detail || err.message || 'Failed to send message.';
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectConversation = (conversationId) => {
    if (conversationId !== currentConversationId) {
      setCurrentConversationId(conversationId);
    }
  };

  const handleNewChat = () => {
    setCurrentConversationId(null);
    setChatHistory([initialWelcomeMessage]); 
    setIsSearchActive(false); 
    setError(null); // Clear general error
    // conversationsError remains as is, should not be affected by starting a new chat UI-wise
  };

  const toggleSearch = () => {
    setIsSearchActive(prev => !prev);
  };

  return (
    <div className="flex h-screen bg-brand-main-bg text-brand-text-primary">
      <ConversationSidebar
        conversations={conversations}
        currentConversationId={currentConversationId}
        onSelectConversation={handleSelectConversation}
        onNewChat={handleNewChat}
        isLoading={isConversationsLoading}
        dbConnected={dbConnected}
        conversationsError={conversationsError} 
      />
      <div className="flex flex-col flex-grow h-screen">
        {/* Header */}
        <header className="p-4 bg-brand-surface-bg shadow-md border-b border-gray-700">
          <h1 className="text-xl font-semibold text-brand-purple">Ollama Chat with MCP</h1>
          <div className="text-xs text-brand-text-secondary flex items-center space-x-4 mt-1">
              <span className={`flex items-center ${mcpServiceReady ? 'text-brand-success-green' : 'text-brand-alert-red'}`}>
                  {mcpServiceReady ? <Wifi size={14} className="mr-1" /> : <AlertTriangle size={14} className="mr-1" />}
                  Search: {mcpServiceReady ? 'Ready' : 'Unavailable'}
              </span>
              <span className="flex items-center text-brand-text-secondary">
                  <Server size={14} className="mr-1" />
                  Model: {ollamaModel}
              </span>
              <span className={`flex items-center ${dbConnected ? 'text-brand-success-green' : 'text-brand-alert-red'}`}>
                  <Database size={14} className="mr-1" />
                  DB: {dbConnected ? 'Connected' : 'Disconnected'}
              </span>
          </div>
        </header>

        {/* Chat Messages */}
        <div ref={chatContainerRef} className="flex-grow p-4 overflow-y-auto space-y-2 bg-brand-main-bg">
          {isChatHistoryLoading && (
            <div className="flex justify-center items-center h-full">
              <Loader2 size={32} className="animate-spin text-brand-purple" />
            </div>
          )}
          {!isChatHistoryLoading && chatHistory.map((msg, index) => (
            <ChatMessage key={msg.timestamp ? `${msg.timestamp}-${index}` : index} message={msg} />
          ))}
          {isLoading && !isChatHistoryLoading && ( 
            <div className="flex justify-start mb-4 animate-pulse">
              <div className="max-w-[70%] p-3 rounded-lg shadow bg-brand-surface-bg text-brand-text-primary rounded-bl-none">
                Thinking...
              </div>
            </div>
          )}
        </div>

        {/* Error Display for main chat window */}
        {error && (
          <div className="p-3 bg-brand-alert-red text-white text-sm flex items-center justify-center">
            <AlertTriangle size={18} className="mr-2" /> {error}
          </div>
        )}

        {/* Chat Input */}
        <ChatInput
          onSendMessage={handleSendMessage}
          isLoading={isLoading || isChatHistoryLoading} 
          isSearchActive={isSearchActive}
          onToggleSearch={toggleSearch}
          disabled={isChatHistoryLoading || isLoading} 
        />
      </div>
    </div>
  );
};

export default App;
