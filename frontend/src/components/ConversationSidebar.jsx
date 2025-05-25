import React, { useState, useEffect, useRef } from 'react';
import { PlusSquare, MessageSquare, Loader2, AlertTriangle, Trash2, Pencil, Check, X, PanelLeftClose, PanelRightOpen } from 'lucide-react';

const ConversationSidebar = ({
  conversations,
  currentConversationId,
  onSelectConversation,
  onNewChat,
  onDeleteConversation,
  onRenameConversation, // New prop
  isLoading,
  dbConnected,
  conversationsError,
  isCollapsed,       // New prop
  onToggleCollapse   // New prop
}) => {
  const [editingConversationId, setEditingConversationId] = useState(null);
  const [currentEditingTitle, setCurrentEditingTitle] = useState('');
  const editInputRef = useRef(null);

  const formatDate = (isoString) => {
    if (!isoString) return '';
    return new Date(isoString).toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
    });
  };

  const handleEditClick = (e, conv) => {
    e.stopPropagation();
    setEditingConversationId(conv.id);
    setCurrentEditingTitle(conv.title || '');
  };

  const handleRenameSave = (e) => {
    e.stopPropagation();
    if (editingConversationId && currentEditingTitle.trim()) {
      onRenameConversation(editingConversationId, currentEditingTitle.trim());
    }
    setEditingConversationId(null);
    setCurrentEditingTitle('');
  };

  const handleRenameCancel = (e) => {
    e.stopPropagation();
    setEditingConversationId(null);
    setCurrentEditingTitle('');
  };

  const handleEditKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleRenameSave(e);
    } else if (e.key === 'Escape') {
      handleRenameCancel(e);
    }
  };
  
  useEffect(() => {
    if (editingConversationId && editInputRef.current) {
      editInputRef.current.focus();
      editInputRef.current.select();
    }
  }, [editingConversationId]);


  const handleDeleteClick = (e, conversationId) => {
    e.stopPropagation();
    if (window.confirm('Are you sure you want to delete this conversation? This action cannot be undone.')) {
      onDeleteConversation(conversationId);
    }
  };

  return (
    <div 
      className={`bg-brand-surface-bg flex flex-col border-r border-gray-700 h-full transition-all duration-300 ease-in-out
                  ${isCollapsed ? 'w-16 p-2' : 'w-64 p-4'}`}
    >
      <button
        onClick={onNewChat}
        className={`flex items-center justify-center w-full p-2 mb-4 bg-brand-purple text-white rounded-md hover:bg-brand-button-grad-to transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-brand-blue
                    ${isCollapsed ? 'px-0' : ''}`}
        title={isCollapsed ? "New Chat" : ""}
      >
        <PlusSquare size={20} className={isCollapsed ? '' : 'mr-2'} />
        {!isCollapsed && <span>New Chat</span>}
      </button>
      
      {!isCollapsed && (
        <h2 className="text-sm font-semibold text-brand-text-secondary mb-2 px-2">History</h2>
      )}
      
      {isLoading && (
        <div className="flex items-center justify-center py-4">
          <Loader2 size={24} className="animate-spin text-brand-purple" />
        </div>
      )}

      {!isLoading && conversationsError && !isCollapsed && (
        <div className="text-xs text-brand-alert-red p-2 rounded bg-red-900/30 mb-2 flex items-start">
          <AlertTriangle size={16} className="mr-2 flex-shrink-0 mt-0.5" /> 
          <span>Error: {conversationsError}</span>
        </div>
      )}

      {!isLoading && !conversationsError && dbConnected && conversations.length === 0 && !isCollapsed && (
        <p className="text-xs text-brand-text-secondary px-2">
          No past conversations.
        </p>
      )}

      <div className={`flex-grow overflow-y-auto space-y-1 ${isCollapsed ? '' : 'pr-1 -mr-1'}`}>
        {!isLoading && !conversationsError && dbConnected &&
          conversations
            .filter(conv => conv && typeof conv.id === 'string' && conv.id.trim() !== '') 
            .map((conv) => {
              const isEditingThis = editingConversationId === conv.id;
              return (
                <div key={conv.id} className="relative group">
                  <button
                    onClick={() => !isEditingThis && onSelectConversation(conv.id)}
                    className={`w-full flex items-start text-left p-2 rounded-md text-sm transition-colors duration-150 focus:outline-none
                      ${ currentConversationId === conv.id && !isEditingThis ? 'bg-brand-blue text-white' : 'text-brand-text-secondary hover:bg-gray-700 hover:text-brand-text-primary'}
                      ${isCollapsed ? 'justify-center' : ''}
                      ${isEditingThis ? 'bg-gray-700' : ''}`}
                    title={isCollapsed ? (conv.title || `Chat from ${formatDate(conv.created_at)}`) : ""}
                  >
                    <MessageSquare size={16} className={`${isCollapsed ? '' : 'mr-2'} mt-0.5 flex-shrink-0`} />
                    {!isCollapsed && (
                      <div className="flex-grow overflow-hidden">
                        {isEditingThis ? (
                          <div className="flex items-center w-full">
                            <input
                              ref={editInputRef}
                              type="text"
                              value={currentEditingTitle}
                              onChange={(e) => setCurrentEditingTitle(e.target.value)}
                              onKeyDown={handleEditKeyDown}
                              onClick={(e) => e.stopPropagation()} // Prevent select on click
                              className="flex-grow bg-gray-600 text-white text-sm p-1 rounded-l-md focus:outline-none focus:ring-1 focus:ring-brand-purple"
                            />
                            <button onClick={handleRenameSave} className="p-1.5 bg-green-500 hover:bg-green-600 text-white rounded-r-md focus:outline-none">
                              <Check size={14} />
                            </button>
                            <button onClick={handleRenameCancel} className="p-1.5 bg-red-500 hover:bg-red-600 text-white ml-1 rounded-md focus:outline-none">
                              <X size={14} />
                            </button>
                          </div>
                        ) : (
                          <>
                            <p className="truncate font-medium">
                              {conv.title || `Chat from ${formatDate(conv.created_at)}`}
                            </p>
                            <p className={`text-xs truncate ${currentConversationId === conv.id ? 'text-blue-200' : 'text-gray-500'}`}>
                              {conv.message_count} messages - {formatDate(conv.updated_at)}
                            </p>
                          </>
                        )}
                      </div>
                    )}
                  </button>
                  {!isEditingThis && !isCollapsed && (
                    <div className="absolute top-1/2 right-1 transform -translate-y-1/2 flex items-center opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity duration-150 z-10">
                      <button
                        onClick={(e) => handleEditClick(e, conv)}
                        className="p-1.5 rounded-md text-brand-text-secondary hover:text-brand-blue hover:bg-gray-600"
                        title="Rename chat"
                        aria-label="Rename chat"
                      >
                        <Pencil size={14} />
                      </button>
                      <button
                        onClick={(e) => handleDeleteClick(e, conv.id)}
                        className="p-1.5 rounded-md text-brand-text-secondary hover:text-brand-alert-red hover:bg-gray-600"
                        title="Delete chat"
                        aria-label="Delete chat"
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  )}
                </div>
              );
            })}
      </div>
      {/* Toggle button is now in App.jsx header */}
    </div>
  );
};

export default ConversationSidebar;
