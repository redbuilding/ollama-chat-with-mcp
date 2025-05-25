import React, { useState } from 'react';
import { Send, Search, Zap, Database, Server } from 'lucide-react'; // Added Database, Server icons

const ChatInput = ({ 
  onSendMessage, 
  isLoading, 
  isSearchActive, 
  onToggleSearch,
  isDatabaseActive, // New prop
  onToggleDatabase, // New prop
  disabled,
  placeholder
}) => {
  const [inputValue, setInputValue] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (inputValue.trim() && !isLoading && !disabled) {
      onSendMessage(inputValue.trim());
      setInputValue('');
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="sticky bottom-0 left-0 right-0 p-4 bg-brand-main-bg border-t border-brand-surface-bg"
    >
      <div className="flex items-center bg-brand-surface-bg rounded-lg p-2 shadow-md">
        {/* Search Toggle Button */}
        <button
          type="button"
          onClick={() => {
            onToggleSearch();
            if (isDatabaseActive && !isSearchActive) onToggleDatabase(); // Deactivate DB if activating Search
          }}
          title={isSearchActive ? "Disable Web Search" : "Enable Web Search"}
          disabled={disabled}
          className={`p-2 rounded-md mr-2 transition-colors duration-200 focus:outline-none focus:ring-2
            focus:ring-brand-purple ${
            isSearchActive ? 'bg-brand-accent text-brand-main-bg hover:bg-yellow-400' : 'bg-gray-700 text-brand-text-secondary hover:bg-gray-600'
          } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          {isSearchActive ? <Zap size={20} /> : <Search size={20} />}
        </button>

        {/* Database Toggle Button */}
        <button
          type="button"
          onClick={() => {
            onToggleDatabase();
            if (isSearchActive && !isDatabaseActive) onToggleSearch(); // Deactivate Search if activating DB
          }}
          title={isDatabaseActive ? "Disable Database Query" : "Enable Database Query"}
          disabled={disabled}
          className={`p-2 rounded-md mr-2 transition-colors duration-200 focus:outline-none focus:ring-2
            focus:ring-brand-purple ${
            isDatabaseActive ? 'bg-brand-blue text-white hover:bg-blue-500' : 'bg-gray-700 text-brand-text-secondary hover:bg-gray-600'
          } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          {isDatabaseActive ? <Server size={20} /> : <Database size={20} />}
        </button>
        
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder={placeholder}
          className="flex-grow p-2 bg-transparent text-brand-text-primary focus:outline-none placeholder-brand-text-secondary"
          disabled={isLoading || disabled}
        />
        <button
          type="submit"
          disabled={isLoading || !inputValue.trim() || disabled}
          className="p-2 ml-2 rounded-md bg-brand-purple text-white hover:bg-brand-button-grad-to
            focus:outline-none focus:ring-2 focus:ring-brand-blue disabled:opacity-50 transition-colors duration-200"
        >
          {isLoading ? (
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
          ) : (
            <Send size={20} />
          )}
        </button>
      </div>
       {isSearchActive && (
        <p className="text-xs text-brand-accent mt-2 ml-12 sm:ml-28"> {/* Adjusted margin for two buttons */}
          ⚡ Web search is active. Your message will be used as a search query.
        </p>
      )}
      {isDatabaseActive && (
        <p className="text-xs text-brand-blue mt-2 ml-12 sm:ml-28"> {/* Adjusted margin */}
          💾 Database query is active. Your message will be interpreted to query the database.
        </p>
      )}
      {!isSearchActive && !isDatabaseActive && (
         <p className="text-xs text-brand-text-secondary mt-2 ml-12 sm:ml-28 h-4"> {/* Placeholder for consistent height */}
         </p>
      )}
    </form>
  );
};

export default ChatInput;
