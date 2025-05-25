import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
// Node.js 'path' module is not strictly necessary for '..' but can be used for more complex resolutions.
// For `root: '..'`, Vite handles it directly.

// https://vitejs.dev/config/
export default defineConfig({
  // If vite.config.js is in 'frontend/src/', and your project root
  // (containing index.html, tailwind.config.js, postcss.config.js) is 'frontend/',
  // set the root to the parent directory.
  root: '..', // Sets Vite's project root to the parent directory (e.g., 'frontend/')

  plugins: [react()],

  css: {
    // Explicitly point to postcss.config.js relative to the new root.
    // Vite should auto-detect 'postcss.config.js' in the project root,
    // but being explicit can be safer if auto-detection fails.
    postcss: './postcss.config.js' // This path is now relative to 'frontend/'
  },

  // Other Vite options like server, build, resolve, etc.,
  // will also now operate relative to the new root ('frontend/').
  // For example, `build.outDir` will default to 'frontend/dist'.
  // The entry point for `index.html` (`<script type="module" src="/src/main.jsx"></script>`)
  // will correctly resolve to `frontend/src/main.jsx`.
})
