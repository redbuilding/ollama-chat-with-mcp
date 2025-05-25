import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],

  css: {
    // Explicitly point to postcss.config.js relative to the new root.
    // Vite should auto-detect 'postcss.config.js' in the project root,
    // but being explicit can be safer if auto-detection fails.
    postcss: "./postcss.config.js", // This path is now relative to 'frontend/'
  },

  // Other Vite options like server, build, resolve, etc.,
  // will also now operate relative to the new root ('frontend/').
  // For example, `build.outDir` will default to 'frontend/dist'.
  // The entry point for `index.html` (`<script type="module" src="/src/main.jsx"></script>`)
  // will correctly resolve to `frontend/src/main.jsx`.
});
