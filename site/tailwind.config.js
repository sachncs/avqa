/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          "Manrope",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        display: ["Manrope", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: [
          "JetBrains Mono",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Monaco",
          "Consolas",
          "monospace",
        ],
      },
      colors: {
        ink: {
          50: "#F5F3E9",
          100: "#E8E6D8",
          200: "#D1D3C4",
          300: "#B3BBA9",
          400: "#939E8C",
          500: "#83907E",
          600: "#566351",
          700: "#39463A",
          800: "#253329",
          900: "#17221A",
          950: "#0D1510",
        },
        accent: {
          DEFAULT: "#E9784F",
          50: "#FFF2E9",
          100: "#FFE1D1",
          200: "#FFC2A5",
          300: "#F5A17C",
          400: "#E9784F",
          500: "#C95836",
          600: "#A5422A",
          700: "#7E3426",
          800: "#59271E",
          900: "#3A1C17",
        },
        glow: {
          DEFAULT: "#B9D692",
          soft: "#D5E7B8",
        },
      },
      letterSpacing: {
        tightest: "-0.045em",
      },
      boxShadow: {
        glow: "0 0 0 3px rgba(233,120,79,0.16)",
        card: "0 1px 0 rgba(255,255,255,0.035) inset, 0 20px 50px -32px rgba(0,0,0,0.6)",
      },
      backgroundImage: {
        "grid-faint":
          "linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px)",
        "radial-fade":
          "radial-gradient(ellipse at top, rgba(185,214,146,0.08), transparent 58%)",
      },
      keyframes: {
        float: {
          "0%,100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-6px)" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        pulse_soft: {
          "0%,100%": { opacity: "0.4" },
          "50%": { opacity: "0.9" },
        },
      },
      animation: {
        float: "float 6s ease-in-out infinite",
        shimmer: "shimmer 3s linear infinite",
        "pulse-soft": "pulse_soft 4s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
