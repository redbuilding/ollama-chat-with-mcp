import { useState, useCallback } from "react";

export function useCopyToClipboard(timeout = 2000) {
  const [copied, setCopied] = useState(false);

  const copy = useCallback(
    async (text) => {
      if (!navigator?.clipboard) {
        console.warn("Clipboard not supported");

        // Fallback for http connections or older browsers

        try {
          const textArea = document.createElement("textarea");

          textArea.value = text;

          textArea.style.position = "fixed"; // Prevent scrolling to bottom of page in MS Edge.

          document.body.appendChild(textArea);

          textArea.focus();

          textArea.select();

          document.execCommand("copy");

          setCopied(true);

          setTimeout(() => setCopied(false), timeout);
        } catch (err) {
          console.error("Fallback copy failed:", err);

          setCopied(false);
        } finally {
          const textArea = document.querySelector("textarea");

          if (textArea) document.body.removeChild(textArea);
        }

        return;
      }

      try {
        await navigator.clipboard.writeText(text);

        setCopied(true);

        setTimeout(() => setCopied(false), timeout);
      } catch (error) {
        console.error("Copy to clipboard failed:", error);

        setCopied(false);
      }
    },
    [timeout],
  );

  return [copied, copy];
}
