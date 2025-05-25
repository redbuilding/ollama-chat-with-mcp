/** @type {import('tailwindcss').Config} */

export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],

  darkMode: "class", // Using class strategy for dark mode

  theme: {
    extend: {
      colors: {
        brand: {
          purple: "#A78BFA", // Primary Purple

          blue: "#60A5FA", // Primary Blue

          accent: "#FCD34D", // Primary Accent (Yellow-Orange)

          "main-bg": "#121212", // Main Background

          "surface-bg": "#1E1E1E", // Surface Layer (cards, content containers)

          "text-primary": "#FFFFFF", // Primary Text (White)

          "text-secondary": "#D1D5DB", // Secondary Text (Soft Gray)

          "text-accent": "#A78BFA", // Accent Text (Purple)

          "success-green": "#10B981", // Success Green

          "stat-blue": "#3B82F6", // Stat Blue (Interactive elements)

          "alert-red": "#EF4444", // Alert Red

          "button-grad-from": "#7C3AED",

          "button-grad-to": "#8B5CF6",
        },
      },

      animation: {
        "fade-in": "fadeIn 0.5s ease-out",

        "slide-up": "slideUp 0.5s ease-out",
      },

      keyframes: {
        fadeIn: {
          "0%": { opacity: 0 },

          "100%": { opacity: 1 },
        },

        slideUp: {
          "0%": { opacity: 0, transform: "translateY(20px)" },

          "100%": { opacity: 1, transform: "translateY(0)" },
        },
      },
    },
  },

  plugins: [],
};
