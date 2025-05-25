import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import ChatMessage from './components/ChatMessage';
import ChatInput from './components/ChatInput';
import ConversationSidebar from './components/ConversationSidebar';
import { 
  sendMessage, 
  getServiceStatus, 
  getConversations, 
  getConversationMessages,
  getOllamaModels,
  deleteConversation,
  renameConversation // Import new API function
} from './services/api';
import { AlertTriangle, Wifi, Database, Loader2, BrainCircuit, PanelLeftClose, PanelRightOpen } from 'lucide-react';

const App = () => {
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false); 
  const [error, setError] = useState(null); 
  const [isSearchActive, setIsSearchActive] = useState(false);
  
  const [mcpServiceReady, setMcpServiceReady] = useState(false);
  const [dbConnected, setDbConnected] = useState(false);
  const [ollamaAvailable, setOllamaAvailable] = useState(false);

  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [conversations, setConversations] = useState([]);
  const [isConversationsLoading, setIsConversationsLoading] = useState(true);
  const [conversationsError, setConversationsError] = useState(null); 
  const [isChatHistoryLoading, setIsChatHistoryLoading] = useState(false);

  const [availableOllamaModels, setAvailableOllamaModels] = useState([]);
  const [selectedOllamaModel, setSelectedOllamaModel] = useState(''); 
  const [ollamaModelsError, setOllamaModelsError] = useState(null);

  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false); // New state for sidebar

  const chatContainerRef = useRef(null);

  const initialWelcomeMessage = useMemo(() => {
    return { 
      role: 'assistant', 
      content: "Hello! I'm your AI assistant. Select a model for new chats from the dropdown above. Toggle the search icon to enable web search. Select 'New Chat' to begin or choose a past conversation.",
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
      setOllamaAvailable(status.ollama_available);
      
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
      setOllamaAvailable(false);
      setConversationsError("Failed to fetch service status. History may be unavailable.");
    }
  }, [conversationsError]); 

  useEffect(() => {
    fetchServiceStatus(); 
    const intervalId = setInterval(fetchServiceStatus, 15000);
    return () => {
      clearInterval(intervalId);
    };
  }, [fetchServiceStatus]);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const models = await getOllamaModels();
        setAvailableOllamaModels(models || []);
        if (models && models.length > 0) {
          const preferredModel = models.find(m => !m.toLowerCase().includes("embed") && (m.toLowerCase().includes("instruct") || m.toLowerCase().includes("chat"))) || models[0];
          setSelectedOllamaModel(preferredModel);
        } else {
          setSelectedOllamaModel('');
        }
        setOllamaModelsError(null);
      } catch (err) {
        console.error("Failed to fetch Ollama models:", err);
        setOllamaModelsError(err.detail || err.message || "Failed to load models.");
        setAvailableOllamaModels([]);
        setSelectedOllamaModel('');
      }
    };
    if (ollamaAvailable) { 
        fetchModels();
    } else {
        setAvailableOllamaModels([]);
        setSelectedOllamaModel('');
        setOllamaModelsError("Ollama server unavailable.");
    }
  }, [ollamaAvailable]);


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
  }, [dbConnected, fetchConversationsList, conversationsError]); 


  useEffect(() => {
    const loadMessages = async () => {
      if (currentConversationId) {
        setIsChatHistoryLoading(true);
        setError(null); 
        try {
          const messages = await getConversationMessages(currentConversationId);
          setChatHistory(messages || []);
          const currentConv = conversations.find(c => c.id === currentConversationId);
          if (currentConv && currentConv.ollama_model_name) {
            // setSelectedOllamaModel(currentConv.ollama_model_name); // Not strictly needed if model display is text-only for existing
          }
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
  }, [currentConversationId, initialWelcomeMessage, conversations]);


  const handleSendMessage = async (userInput) => {
    setIsLoading(true);
    setError(null); 

    const modelForThisMessage = currentConversationId ? null : selectedOllamaModel;

    if (!modelForThisMessage && !currentConversationId && availableOllamaModels.length > 0) {
        setError("Please select an Ollama model for this new chat.");
        setIsLoading(false);
        return;
    }
    if (!modelForThisMessage && !currentConversationId && availableOllamaModels.length === 0 && ollamaAvailable) {
        setError("No Ollama models available. Cannot start chat.");
        setIsLoading(false);
        return;
    }
    if(!ollamaAvailable){
        setError("Ollama server is not available. Cannot start chat.");
        setIsLoading(false);
        return;
    }


    try {
      const response = await sendMessage(userInput, chatHistory, isSearchActive, currentConversationId, modelForThisMessage);
      
      setChatHistory(response.chat_history || []);
      
      if (response.conversation_id && response.conversation_id !== currentConversationId) {
        setCurrentConversationId(response.conversation_id);
        await fetchConversationsList(); 
      } else if (response.conversation_id === currentConversationId) {
        await fetchConversationsList(); // Refresh for message count or title changes
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
    if (availableOllamaModels.length > 0 && !selectedOllamaModel) {
        const preferredModel = availableOllamaModels.find(m => !m.toLowerCase().includes("embed") && (m.toLowerCase().includes("instruct") || m.toLowerCase().includes("chat"))) || availableOllamaModels[0];
        setSelectedOllamaModel(preferredModel);
    }
  };

  const handleDeleteConversation = async (conversationIdToDelete) => {
    setError(null);
    try {
      await deleteConversation(conversationIdToDelete);
      setConversations(prevConvs => prevConvs.filter(conv => conv.id !== conversationIdToDelete));
      if (currentConversationId === conversationIdToDelete) {
        handleNewChat(); 
      }
    } catch (err) {
      const errorMessage = err.detail || err.message || 'Failed to delete conversation.';
      setError(errorMessage); 
    }
  };

  const handleRenameConversation = async (conversationIdToRename, newTitle) => {
    setError(null);
    try {
      const updatedConversation = await renameConversation(conversationIdToRename, newTitle);
      setConversations(prevConvs => 
        prevConvs.map(conv => 
          conv.id === conversationIdToRename 
            ? { ...conv, title: updatedConversation.title, updated_at: updatedConversation.updated_at } 
            : conv
        ).sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at)) // Re-sort if needed
      );
    } catch (err) {
      const errorMessage = err.detail || err.message || 'Failed to rename conversation.';
      setError(errorMessage);
      // Optionally re-fetch to revert optimistic update or ensure consistency
      // fetchConversationsList(); 
    }
  };
  
  const toggleSidebarCollapse = () => {
    setIsSidebarCollapsed(prev => !prev);
  };

  const toggleSearch = () => {
    setIsSearchActive(prev => !prev);
  };
  
  const currentConversationDetails = useMemo(() => {
    return conversations.find(c => c.id === currentConversationId);
  }, [conversations, currentConversationId]);

  const modelForDisplay = currentConversationDetails?.ollama_model_name || selectedOllamaModel || 'N/A';


  return (
    <div className="flex h-screen bg-brand-main-bg text-brand-text-primary overflow-hidden">
      <ConversationSidebar
        conversations={conversations}
        currentConversationId={currentConversationId}
        onSelectConversation={handleSelectConversation}
        onNewChat={handleNewChat}
        onDeleteConversation={handleDeleteConversation}
        onRenameConversation={handleRenameConversation}
        isLoading={isConversationsLoading}
        dbConnected={dbConnected}
        conversationsError={conversationsError}
        isCollapsed={isSidebarCollapsed}
        onToggleCollapse={toggleSidebarCollapse}
      />
      <div className={`flex flex-col flex-grow h-screen transition-all duration-300 ease-in-out ${isSidebarCollapsed ? 'ml-0' : 'ml-0'}`}> {/* Sidebar width is handled internally now */}
        {/* Header */}
        <header className="p-4 bg-brand-surface-bg shadow-md border-b border-gray-700 flex items-center justify-between">
          <div className="flex items-center">
            <button 
              onClick={toggleSidebarCollapse} 
              className="p-2 mr-2 rounded-md hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-brand-purple"
              title={isSidebarCollapsed ? "Open sidebar" : "Close sidebar"}
            >
              {isSidebarCollapsed ? <PanelRightOpen size={20} /> : <PanelLeftClose size={20} />}
            </button>
            <h1 className="text-xl font-semibold text-brand-purple">Ollama Chat with MCP</h1>
          </div>
          <div className="text-xs text-brand-text-secondary flex items-center space-x-2 sm:space-x-4 flex-wrap">
              <span className={`flex items-center ${mcpServiceReady ? 'text-brand-success-green' : 'text-brand-alert-red'}`}>
                  {mcpServiceReady ? <Wifi size={14} className="mr-1" /> : <AlertTriangle size={14} className="mr-1" />}
                  Search: {mcpServiceReady ? 'Ready' : 'Unavailable'}
              </span>
              
              <div className="flex items-center">
                <BrainCircuit size={14} className={`mr-1 ${ollamaAvailable ? 'text-brand-purple' : 'text-brand-alert-red'}`} />
                {!currentConversationId ? (
                  <select
                    value={selectedOllamaModel}
                    onChange={(e) => setSelectedOllamaModel(e.target.value)}
                    disabled={!availableOllamaModels.length || isLoading || isChatHistoryLoading || !ollamaAvailable}
                    className="bg-brand-surface-bg text-brand-text-secondary text-xs p-1 rounded border border-gray-600 focus:outline-none focus:ring-1 focus:ring-brand-purple max-w-[150px] sm:max-w-[200px] truncate"
                    title={modelForDisplay}
                  >
                    {ollamaModelsError && <option value="">{ollamaModelsError}</option>}
                    {!ollamaModelsError && !ollamaAvailable && <option value="">Ollama N/A</option>}
                    {!ollamaModelsError && ollamaAvailable && availableOllamaModels.length === 0 && <option value="">No models</option>}
                    {availableOllamaModels.map(model => (
                      <option key={model} value={model}>{model}</option>
                    ))}
                  </select>
                ) : (
                  <span className="text-brand-text-secondary max-w-[150px] sm:max-w-[200px] truncate" title={modelForDisplay}>
                    Model: {modelForDisplay}
                  </span>
                )}
              </div>

              <span className={`flex items-center ${dbConnected ? 'text-brand-success-green' : 'text-brand-alert-red'}`}>
                  <Database size={14} className="mr-1" />
                  DB: {dbConnected ? 'Connected' : 'Disconnected'}
              </span>
          </div>
        </header>

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
                Thinking with {currentConversationDetails?.ollama_model_name || selectedOllamaModel}...
              </div>
            </div>
          )}
        </div>

        {error && (
          <div className="p-3 bg-brand-alert-red text-white text-sm flex items-center justify-center">
            <AlertTriangle size={18} className="mr-2" /> {error}
          </div>
        )}

        <ChatInput
          onSendMessage={handleSendMessage}
          isLoading={isLoading || isChatHistoryLoading} 
          isSearchActive={isSearchActive}
          onToggleSearch={toggleSearch}
          disabled={isChatHistoryLoading || isLoading || (!selectedOllamaModel && !currentConversationId && availableOllamaModels.length > 0 && ollamaAvailable) || !ollamaAvailable}
          placeholder={
            !ollamaAvailable ? "Ollama server unavailable..." :
            (!selectedOllamaModel && !currentConversationId && availableOllamaModels.length > 0) ? "Select a model to begin..." :
            (isSearchActive ? "Enter web search query..." : "Type your message...")
          }
        />
      </div>
    </div>
  );
};

export default App;
