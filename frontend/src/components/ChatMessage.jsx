import React from 'react';
import { User, Bot, Clipboard, Check } from 'lucide-react';
import { useCopyToClipboard } from '../hooks/useCopyToClipboard';
import CodeBlock from './CodeBlock'; // Assuming CodeBlock component exists

// Function to parse message content for code blocks
const parseMessageContent = (content) => {
  const parts = [];
  let lastIndex = 0;
  // Regex to find markdown-style code blocks
  const codeBlockRegex = /```(\w*)\n([\s\S]*?)```/g;
  let match;

  while ((match = codeBlockRegex.exec(content)) !== null) {
    // Text before code block
    if (match.index > lastIndex) {
      parts.push({ type: 'text', content: content.substring(lastIndex, match.index) });
    }
    // Code block
    parts.push({ type: 'code', language: match[1] || '', code: match[2].trim() });
    lastIndex = match.index + match[0].length;
  }

  // Remaining text after the last code block
  if (lastIndex < content.length) {
    parts.push({ type: 'text', content: content.substring(lastIndex) });
  }

  // If no code blocks found, the whole content is text
  if (parts.length === 0 && content) {
    parts.push({ type: 'text', content });
  }

  return parts;
};


const ChatMessage = ({ message }) => {
  const { role, content, is_html } = message;
  const isUser = role === 'user';
  const [copied, copy] = useCopyToClipboard();

  // Extract text content for copying, stripping HTML if necessary
  const getTextContentForCopy = (htmlContent) => {
    if (!htmlContent) return '';
    if (is_html) {
      const tempDiv = document.createElement('div');
      tempDiv.innerHTML = htmlContent;
      // A common pattern is an indicator div then the actual message.
      // Try to get text from all child nodes.
      let text = "";
      tempDiv.childNodes.forEach(node => {
        if (node.textContent) {
          text += node.textContent + "\n";
        }
      });
      return text.trim();
    }
    return htmlContent;
  };

  const contentToCopy = getTextContentForCopy(content);
  const messageParts = is_html ? [] : parseMessageContent(content);


  return (
    <div className={`flex animate-slide-up ${isUser ? 'justify-end' : 'justify-start'} mb-4 group`}>
      <div
        className={`max-w-[80%] p-3 rounded-lg shadow ${
          isUser ? 'bg-brand-blue text-white rounded-br-none' : 'bg-brand-surface-bg text-brand-text-primary rounded-bl-none'
        }`}
      >
        <div className="flex items-start mb-1">
          {isUser ? (
            <User size={20} className="mr-2 text-white flex-shrink-0" />
          ) : (
            <Bot size={20} className="mr-2 text-brand-purple flex-shrink-0" />
          )}
          <span className="font-semibold text-sm">{isUser ? 'You' : 'Assistant'}</span>
        </div>

        {is_html ? (
          <div className="prose prose-sm prose-invert max-w-none" dangerouslySetInnerHTML={{ __html: content }} />
        ) : (
          messageParts.map((part, index) =>
            part.type === 'code' ? (
              <CodeBlock key={index} language={part.language} code={part.code} />
            ) : (
              <p key={index} className="whitespace-pre-wrap text-sm leading-relaxed">
                {part.content}
              </p>
            )
          )
        )}

        <button
          onClick={() => copy(contentToCopy)}
          className="opacity-0 group-hover:opacity-100 transition-opacity duration-300 mt-2 flex items-center text-xs p-1 rounded hover:bg-opacity-20 hover:bg-gray-500"
          aria-label="Copy message"
        >
          {copied ? (
            <Check size={14} className={isUser ? "text-green-300" : "text-brand-success-green"} />
          ) : (
            <Clipboard size={14} className={isUser ? "text-gray-300" : "text-brand-text-secondary"} />
          )}
          <span className={`ml-1 text-xs ${isUser ? "text-gray-300" : "text-brand-text-secondary"}`}>
            {copied ? 'Copied!' : 'Copy'}
          </span>
        </button>
      </div>
    </div>
  );
};

export default ChatMessage;
