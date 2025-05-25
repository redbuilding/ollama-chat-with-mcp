import React from "react";

import { Clipboard, Check } from "lucide-react";

import { useCopyToClipboard } from "../hooks/useCopyToClipboard";

const CodeBlock = ({ language, code }) => {
  const [copied, copy] = useCopyToClipboard();

  // Basic language detection for display

  const displayLanguage = language || "code";

  return (
    <div className="relative my-2 rounded-md bg-gray-800 border border-brand-surface-bg group">
      <div
        className="flex items-center justify-between px-4 py-2 text-xs text-brand-text-secondary border-b
border-brand-surface-bg"
      >
        <span>{displayLanguage}</span>

        <button
          onClick={() => copy(code)}
          className="flex items-center p-1 rounded hover:bg-brand-surface-bg transition-colors"
          aria-label="Copy code"
        >
          {copied ? (
            <Check size={16} className="text-brand-success-green" />
          ) : (
            <Clipboard size={16} />
          )}

          <span className="ml-1 text-xs">{copied ? "Copied!" : "Copy"}</span>
        </button>
      </div>

      <pre className="p-4 overflow-x-auto bg-transparent border-none m-0">
        <code className={`language-${language}`}>{code}</code>
      </pre>
    </div>
  );
};

export default CodeBlock;
