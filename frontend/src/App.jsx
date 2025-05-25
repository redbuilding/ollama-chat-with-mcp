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
  const [isLoading, setIsLoading] = useState(false); 
  const [error, setError] = useState(null); 
  const [isSearchActive, setIsSearchActive] = useState(false);
  
  const [mcpServiceReady, setMcpServiceReady] = useState(false);
  const [ollamaModel, setOllamaModel] = useState('');
  const [dbConnected, setDbConnected] = useState(false);

  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [conversations, setConversations] = useState([]);
  const [isConversationsLoading, setIsConversationsLoading] = useState(true);
  const [conversationsError, setConversationsError] = useState(null); 
  const [isChatHistoryLoading, setIsChatHistoryLoading] = useState(false);

  const chatContainerRef = useRef(null);

  const initialWelcomeMessage = useMemo(() => {
    return { 
      role: 'assistant', 
      content: "Hello! I'm your AI assistant. Toggle the search icon to enable web search for up-to-date answers. Select 'New Chat' to begin or choose a past conversation.",
      timestamp: new Date().toISOString() 
    };
  }, []);

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
      
      const newDbConnected = status.db_connected;
      setDbConnected(newDbConnected); 

      if (!newDbConnected) {
        setConversationsError("Database not connected. History is unavailable.");
      } else {
        if (conversationsError === "Database not connected. History is unavailable.") {
            setConversationsError(null);
        } else if (conversationsError === "Failed to fetch service status. History may be unavailable.") {
            setConversationsError(null);
        }
      }
    } catch (err) {
      setMcpServiceReady(false);
      setDbConnected(false); 
      setConversationsError("Failed to fetch service status. History may be unavailable.");
    }
  }, [conversationsError, dbConnected]); 

  useEffect(() => {
    fetchServiceStatus(); 
    const intervalId = setInterval(fetchServiceStatus, 15000);
    return () => {
      clearInterval(intervalId);
    };
  }, [fetchServiceStatus]);

  const fetchConversationsList = useCallback(async () => {
    setIsConversationsLoading(true);
    setConversationsError(null); 
    try {
      const convs = await getConversations();
      setConversations(convs || []);
    } catch (err) {
      const errorDetail = err.detail || err.message || 'Failed to fetch conversations list.';
      setConversationsError(errorDetail);
      setConversations([]); 
    } finally {
      setIsConversationsLoading(false);
    }
  }, []); 

  useEffect(() => {
    if (dbConnected) {
        fetchConversationsList();
    } else {
        setConversations([]);
        setIsConversationsLoading(false);
        if (conversationsError !== "Failed to fetch service status. History may be unavailable.") {
             setConversationsError("Database not connected. History is unavailable.");
        }
    }
  }, [dbConnected, fetchConversationsList]); 


  useEffect(() => {
    const loadMessages = async () => {
      if (currentConversationId) {
        setIsChatHistoryLoading(true);
        setError(null); 
        try {
          const messages = await getConversationMessages(currentConversationId);
          setChatHistory(messages || []);
        } catch (err) {
          const errorDetail = err.detail || err.message || `Failed to load messages for conversation.`;
          setError(errorDetail); 
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
    setError(null); 

    try {
      const response = await sendMessage(userInput, chatHistory, isSearchActive, currentConversationId);
      
      setChatHistory(response.chat_history || []);
      
      if (response.conversation_id && response.conversation_id !== currentConversationId) {
        setCurrentConversationId(response.conversation_id);
        await fetchConversationsList(); 
      } else if (!response.conversation_id && currentConversationId) {
        setCurrentConversationId(null);
        await fetchConversationsList();
      } else if (response.conversation_id === currentConversationId) {
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
    setError(null); 
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
