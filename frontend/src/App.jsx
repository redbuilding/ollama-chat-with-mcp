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
  console.log('[App Render] Component rendering/re-rendering.');

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
    console.log('[App Memo] Recalculating initialWelcomeMessage.');
    return { 
      role: 'assistant', 
      content: "Hello! I'm your AI assistant. Toggle the search icon to enable web search for up-to-date answers. Select 'New Chat' to begin or choose a past conversation.",
      timestamp: new Date().toISOString() 
    };
  }, []);

  useEffect(() => {
    console.log('[App Effect] Chat history changed, scrolling to bottom.');
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatHistory]);

  const fetchServiceStatus = useCallback(async () => {
    console.log('[App Callback] fetchServiceStatus: Initiated.');
    try {
      const status = await getServiceStatus();
      console.log('[App Callback] fetchServiceStatus: API response status:', status);
      setMcpServiceReady(status.service_ready);
      setOllamaModel(status.ollama_model || 'N/A');
      
      const newDbConnected = status.db_connected;
      console.log(`[App Callback] fetchServiceStatus: Current dbConnected: ${dbConnected}, New from API: ${newDbConnected}`);
      setDbConnected(newDbConnected); // This will trigger the other useEffect if value changes

      if (!newDbConnected) {
        console.log('[App Callback] fetchServiceStatus: DB not connected from status. Setting conversationsError.');
        setConversationsError("Database not connected. History is unavailable.");
      } else {
        // If DB connection is restored, clear the specific "DB not connected" error
        // This allows the other useEffect to trigger fetchConversationsList
        if (conversationsError === "Database not connected. History is unavailable.") {
            console.log('[App Callback] fetchServiceStatus: DB connected. Clearing "Database not connected..." error.');
            setConversationsError(null);
        } else if (conversationsError === "Failed to fetch service status. History may be unavailable.") {
            console.log('[App Callback] fetchServiceStatus: DB connected. Clearing "Failed to fetch service status..." error.');
            setConversationsError(null);
        }
      }
    } catch (err) {
      console.error('[App Callback] fetchServiceStatus: Error fetching service status:', err);
      setMcpServiceReady(false);
      setDbConnected(false); 
      console.log('[App Callback] fetchServiceStatus: Error occurred. Setting dbConnected to false and conversationsError.');
      setConversationsError("Failed to fetch service status. History may be unavailable.");
    }
  }, [conversationsError, dbConnected]); // dbConnected added to ensure current value is available for comparison if needed

  useEffect(() => {
    console.log('[App Effect] Mounting or fetchServiceStatus changed. Initial call and setting interval for fetchServiceStatus.');
    fetchServiceStatus(); // Initial call
    const intervalId = setInterval(fetchServiceStatus, 15000);
    return () => {
      console.log('[App Effect] Cleaning up fetchServiceStatus interval.');
      clearInterval(intervalId);
    };
  }, [fetchServiceStatus]);

  const fetchConversationsList = useCallback(async () => {
    console.log('[App Callback] fetchConversationsList: Initiated.');
    setIsConversationsLoading(true);
    setConversationsError(null); 
    try {
      const convs = await getConversations();
      console.log('[App Callback] fetchConversationsList: API response conversations:', convs);
      setConversations(convs || []);
    } catch (err) {
      const errorDetail = err.detail || err.message || 'Failed to fetch conversations list.';
      console.error('[App Callback] fetchConversationsList: Error fetching conversations:', errorDetail, err);
      setConversationsError(errorDetail);
      setConversations([]); 
    } finally {
      console.log('[App Callback] fetchConversationsList: Setting isConversationsLoading to false.');
      setIsConversationsLoading(false);
    }
  }, []); 

  useEffect(() => {
    console.log(`[App Effect] dbConnected or fetchConversationsList changed. dbConnected: ${dbConnected}`);
    if (dbConnected) {
        console.log('[App Effect] DB is connected. Calling fetchConversationsList.');
        fetchConversationsList();
    } else {
        console.log('[App Effect] DB is NOT connected. Clearing conversations, setting loading false.');
        setConversations([]);
        setIsConversationsLoading(false);
        // Set error only if not already a more specific error from fetchServiceStatus
        if (conversationsError !== "Failed to fetch service status. History may be unavailable.") {
             console.log('[App Effect] Setting conversationsError to "Database not connected..."');
             setConversationsError("Database not connected. History is unavailable.");
        } else {
            console.log('[App Effect] conversationsError is already "Failed to fetch service status...", not overwriting.');
        }
    }
  }, [dbConnected, fetchConversationsList]); 


  useEffect(() => {
    console.log(`[App Effect] currentConversationId or initialWelcomeMessage changed. currentConversationId: ${currentConversationId}`);
    const loadMessages = async () => {
      if (currentConversationId) {
        console.log(`[App Effect] Loading messages for conversation ID: ${currentConversationId}`);
        setIsChatHistoryLoading(true);
        setError(null); 
        try {
          const messages = await getConversationMessages(currentConversationId);
          console.log(`[App Effect] Messages for ${currentConversationId}:`, messages);
          setChatHistory(messages || []);
        } catch (err) {
          const errorDetail = err.detail || err.message || `Failed to load messages for conversation.`;
          console.error(`[App Effect] Error loading messages for ${currentConversationId}:`, errorDetail, err);
          setError(errorDetail); 
          setChatHistory([initialWelcomeMessage]); 
        } finally {
          console.log(`[App Effect] Finished loading messages for ${currentConversationId}. Setting isChatHistoryLoading to false.`);
          setIsChatHistoryLoading(false);
        }
      } else {
        console.log('[App Effect] No currentConversationId. Setting chat history to initialWelcomeMessage.');
        setChatHistory([initialWelcomeMessage]);
        setIsChatHistoryLoading(false);
      }
    };
    loadMessages();
  }, [currentConversationId, initialWelcomeMessage]);


  const handleSendMessage = async (userInput) => {
    console.log(`[App Handler] handleSendMessage: User input: "${userInput}", currentConversationId: ${currentConversationId}`);
    setIsLoading(true);
    setError(null); 

    try {
      const response = await sendMessage(userInput, chatHistory, isSearchActive, currentConversationId);
      console.log('[App Handler] handleSendMessage: API response:', response);
      
      setChatHistory(response.chat_history || []);
      
      if (response.conversation_id && response.conversation_id !== currentConversationId) {
        console.log(`[App Handler] handleSendMessage: New conversation ID ${response.conversation_id}. Updating currentConversationId and fetching list.`);
        setCurrentConversationId(response.conversation_id);
        await fetchConversationsList(); 
      } else if (!response.conversation_id && currentConversationId) {
        console.warn('[App Handler] handleSendMessage: Backend did not return conversation_id. Resetting currentConversationId.');
        setCurrentConversationId(null);
        await fetchConversationsList();
      } else if (response.conversation_id === currentConversationId) {
        console.log('[App Handler] handleSendMessage: Message sent in existing conversation. Refreshing conversation list.');
        await fetchConversationsList();
      }
      
    } catch (err) {
      const errorMessage = err.detail || err.message || 'Failed to send message.';
      console.error('[App Handler] handleSendMessage: Error:', errorMessage, err);
      setError(errorMessage);
    } finally {
      console.log('[App Handler] handleSendMessage: Setting isLoading to false.');
      setIsLoading(false);
    }
  };

  const handleSelectConversation = (conversationId) => {
    console.log(`[App Handler] handleSelectConversation: Selected ID: ${conversationId}, Current ID: ${currentConversationId}`);
    if (conversationId !== currentConversationId) {
      setCurrentConversationId(conversationId);
    }
  };

  const handleNewChat = () => {
    console.log('[App Handler] handleNewChat: Initiated.');
    setCurrentConversationId(null);
    setChatHistory([initialWelcomeMessage]); 
    setIsSearchActive(false); 
    setError(null); 
  };

  const toggleSearch = () => {
    console.log(`[App Handler] toggleSearch: Current isSearchActive: ${isSearchActive}, toggling.`);
    setIsSearchActive(prev => !prev);
  };

  console.log('[App Render] States before rendering return:', {
    dbConnected,
    conversationsError,
    isConversationsLoading,
    conversationsCount: conversations.length,
    currentConversationId,
    error,
    isChatHistoryLoading,
  });

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
