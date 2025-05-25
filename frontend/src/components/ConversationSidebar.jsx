import React from 'react';
import { PlusSquare, MessageSquare, Loader2, AlertTriangle } from 'lucide-react';

const ConversationSidebar = ({
  conversations,
  currentConversationId,
  onSelectConversation,
  onNewChat,
  isLoading, // This is isConversationsLoading from App.jsx
  dbConnected,
  conversationsError,
}) => {
  const formatDate = (isoString) => {
    if (!isoString) return '';
    return new Date(isoString).toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
    });
  };

  return (
    <div className="w-64 bg-brand-surface-bg p-4 flex flex-col border-r border-gray-700 h-full">
      <button
        onClick={onNewChat}
        className="flex items-center justify-center w-full p-2 mb-4 bg-brand-purple text-white rounded-md hover:bg-brand-button-grad-to transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-brand-blue"
      >
        <PlusSquare size={20} className="mr-2" />
        New Chat
      </button>
      <h2 className="text-sm font-semibold text-brand-text-secondary mb-2 px-2">History</h2>
      
      {isLoading && (
        <div className="flex items-center justify-center py-4">
          <Loader2 size={24} className="animate-spin text-brand-purple" />
        </div>
      )}

      {!isLoading && conversationsError && (
        <div className="text-xs text-brand-alert-red p-2 rounded bg-red-900/30 mb-2 flex items-start">
          <AlertTriangle size={16} className="mr-2 flex-shrink-0 mt-0.5" /> 
          <span>Error: {conversationsError}</span>
        </div>
      )}

      {!isLoading && !conversationsError && dbConnected && conversations.length === 0 && (
        <p className="text-xs text-brand-text-secondary px-2">
          No past conversations.
        </p>
      )}

      <div className="flex-grow overflow-y-auto space-y-1 pr-1 -mr-1">
        {!isLoading && !conversationsError && dbConnected &&
          conversations
            .filter(conv => {
              const isValid = conv && typeof conv.id === 'string' && conv.id.trim() !== '';
              return isValid;
            }) 
            .map((conv) => {
              return (
                <button
                  key={conv.id} 
                  onClick={() => onSelectConversation(conv.id)}
                  className={`w-full flex items-start text-left p-2 rounded-md text-sm transition-colors duration-150 focus:outline-none
                    ${
                      currentConversationId === conv.id
                        ? 'bg-brand-blue text-white'
                        : 'text-brand-text-secondary hover:bg-gray-700 hover:text-brand-text-primary'
                    }`}
                >
                  <MessageSquare size={16} className="mr-2 mt-0.5 flex-shrink-0" />
                  <div className="flex-grow overflow-hidden">
                    <p className="truncate font-medium">
                      {conv.title || `Chat from ${formatDate(conv.created_at)}`}
                    </p>
                    <p className={`text-xs truncate ${currentConversationId === conv.id ? 'text-blue-200' : 'text-gray-500'}`}>
                      {conv.message_count} messages - {formatDate(conv.updated_at)}
                    </p>
                  </div>
                </button>
              );
            })}
      </div>
    </div>
  );
};

export default ConversationSidebar;
