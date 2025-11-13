/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: "#7C83FD",
          light: "#A39BFF",
          dark: "#5A5AD6",
        },
        surface: "#1E1E2E",
        accent: "#F6C177",
      },
    },
  },
  plugins: [],
}

