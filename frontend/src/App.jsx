import React, { useState, useEffect, useRef } from 'react';

import ChatMessage from './components/ChatMessage';

import ChatInput from './components/ChatInput';

import { sendMessage, getServiceStatus } from './services/api';

import { AlertTriangle, Wifi, WifiOff, Server } from 'lucide-react';



const App = () => {

  const [chatHistory, setChatHistory] = useState([]);

  const [isLoading, setIsLoading] = useState(false);

  const [error, setError] = useState(null);

  const [isSearchActive, setIsSearchActive] = useState(false);

  const [mcpServiceReady, setMcpServiceReady] = useState(false);

  const [ollamaModel, setOllamaModel] = useState('');



  const chatContainerRef = useRef(null);



  useEffect(() => {

    // Scroll to bottom when new messages are added

    if (chatContainerRef.current) {

      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;

    }

  }, [chatHistory]);



  useEffect(() => {

    // Fetch initial service status

    const fetchStatus = async () => {

      try {

        const status = await getServiceStatus();

        setMcpServiceReady(status.service_ready);

        setOllamaModel(status.ollama_model || 'N/A');

        if (!status.service_ready) {

            // You could add an initial message or keep it as a status bar info

            // setChatHistory(prev => [...prev, {role: 'assistant', content: "MCP search service is currently
unavailable."}]);

        }

      } catch (err) {

        setError('Failed to connect to backend services. Please try again later.');

        setMcpServiceReady(false);

      }

    };

    fetchStatus();

    // Poll for status updates (optional, good for dynamic service readiness)

    const intervalId = setInterval(fetchStatus, 15000); // Poll every 15 seconds

    return () => clearInterval(intervalId);

  }, []);



  const handleSendMessage = async (userInput) => {

    setIsLoading(true);

    setError(null);



    const newHistory = [...chatHistory, { role: 'user', content: userInput }];

    // Optimistically update UI with user message

    // The full history including assistant response will come from backend

    // setChatHistory(newHistory);



    try {

      // The backend will return the full history including the user's message and assistant's response

      const updatedHistoryFromServer = await sendMessage(userInput, chatHistory, isSearchActive);

      setChatHistory(updatedHistoryFromServer);



      if (userInput.toLowerCase() === '#clear') {

        // Backend handles clearing its context; frontend just updates with the "cleared" message

        // Or, if backend returns empty history on #clear:

        // setChatHistory([{ role: 'assistant', content: 'Chat history cleared.' }]);

      }



    } catch (err) {

      const errorMessage = err.detail || err.message || 'Failed to send message. Please check your connection
or try again.';

      setError(errorMessage);

      // Add error message to chat history to make it visible to user

      setChatHistory(prev => [...prev, {role: 'assistant', content: `Error: ${errorMessage}`}]);

    } finally {

      setIsLoading(false);

    }

  };



  const toggleSearch = () => {

    setIsSearchActive(prev => !prev);

  };



  // Initial welcome message

  useEffect(() => {

    setChatHistory([

      { role: 'assistant', content: "Hello! I'm your AI assistant. Toggle the search icon to enable web search
for up-to-date answers. Type #clear to reset our conversation." }

    ]);

  }, []);





  return (

    <div className="flex flex-col h-screen bg-brand-main-bg text-brand-text-primary">

      {/* Header */}

      <header className="p-4 bg-brand-surface-bg shadow-md border-b border-gray-700">

        <h1 className="text-xl font-semibold text-brand-purple">Ollama Chat with MCP</h1>

        <div className="text-xs text-brand-text-secondary flex items-center space-x-4 mt-1">

            <span className={`flex items-center ${mcpServiceReady ? 'text-brand-success-green' :
'text-brand-alert-red'}`}>

                {mcpServiceReady ? <Wifi size={14} className="mr-1" /> : <WifiOff size={14} className="mr-1"
/>}

                MCP Search: {mcpServiceReady ? 'Ready' : 'Unavailable'}

            </span>

            <span className="flex items-center text-brand-text-secondary">

                <Server size={14} className="mr-1" />

                LLM: {ollamaModel}

            </span>

        </div>

      </header>



      {/* Chat Messages */}

      <div ref={chatContainerRef} className="flex-grow p-4 overflow-y-auto space-y-2 bg-brand-main-bg">

        {chatHistory.map((msg, index) => (

          <ChatMessage key={index} message={msg} />

        ))}

        {isLoading && chatHistory.length > 0 && ( // Show thinking indicator only if there's prior history

          <div className="flex justify-start mb-4 animate-pulse">

            <div className="max-w-[70%] p-3 rounded-lg shadow bg-brand-surface-bg text-brand-text-primary
rounded-bl-none">

              Thinking...

            </div>

          </div>

        )}

      </div>



      {/* Error Display */}

      {error && (

        <div className="p-3 bg-brand-alert-red text-white text-sm flex items-center justify-center">

          <AlertTriangle size={18} className="mr-2" /> {error}

        </div>

      )}



      {/* Chat Input */}

      <ChatInput

        onSendMessage={handleSendMessage}

        isLoading={isLoading}

        isSearchActive={isSearchActive}

        onToggleSearch={toggleSearch}

      />

    </div>

  );

};



export default App;
