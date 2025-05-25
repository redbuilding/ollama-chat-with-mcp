import React, { useState } from 'react';
import { Send, Search, Zap } from 'lucide-react'; // Zap for search active, Search for inactive

const ChatInput = ({ onSendMessage, isLoading, isSearchActive, onToggleSearch }) => {
  const [inputValue, setInputValue] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (inputValue.trim() && !isLoading) {
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
        <button
          type="button"
          onClick={onToggleSearch}
          title={isSearchActive ? "Disable Web Search" : "Enable Web Search"}
          className={`p-2 rounded-md mr-2 transition-colors duration-200 focus:outline-none focus:ring-2
focus:ring-brand-purple ${
            isSearchActive ? 'bg-brand-accent text-brand-main-bg hover:bg-yellow-400' : 'bg-gray-700 text-brand-text-secondary hover:bg-gray-600'
          }`}
        >
          {isSearchActive ? <Zap size={20} /> : <Search size={20} />}
        </button>
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder={isSearchActive ? "Enter web search query..." : "Type your message..."}
          className="flex-grow p-2 bg-transparent text-brand-text-primary focus:outline-none
placeholder-brand-text-secondary"
          disabled={isLoading}
        />
        <button
          type="submit"
          disabled={isLoading || !inputValue.trim()}
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
        <p className="text-xs text-brand-accent mt-2 ml-12">
          ⚡ Web search is active. Your message will be used as a search query.
        </p>
      )}
    </form>
  );
};

export default ChatInput;
