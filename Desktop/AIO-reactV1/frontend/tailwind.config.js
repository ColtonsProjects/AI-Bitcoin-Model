/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        customGray: 'rgb(18,18,18)',  // Define custom color
      },
    },
  },
  plugins: [],
}

