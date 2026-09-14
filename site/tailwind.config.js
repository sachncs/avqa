/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        display: [
          "Inter Display",
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "sans-serif",
        ],
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
          50: "#F7F8FA",
          100: "#EEEFF3",
          200: "#D6D9E2",
          300: "#B1B6C7",
          400: "#8088A1",
          500: "#5C6580",
          600: "#3F4763",
          700: "#2A3047",
          800: "#171B2D",
          900: "#0C0F1C",
          950: "#06080F",
        },
        accent: {
          DEFAULT: "#7C8CFF",
          50: "#F1F3FF",
          100: "#E0E5FF",
          200: "#C2CAFF",
          300: "#9AA8FF",
          400: "#7C8CFF",
          500: "#5F6EF0",
          600: "#4753D6",
          700: "#3742AB",
          800: "#2B3385",
          900: "#1F2461",
        },
        glow: {
          DEFAULT: "#7DF9FF",
          soft: "#A0E9FF",
        },
      },
      letterSpacing: {
        tightest: "-0.045em",
      },
      boxShadow: {
        glow: "0 0 80px -20px rgba(124,140,255,0.45)",
        card: "0 1px 0 rgba(255,255,255,0.04) inset, 0 30px 80px -30px rgba(0,0,0,0.6)",
      },
      backgroundImage: {
        "grid-faint":
          "linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px)",
        "radial-fade":
          "radial-gradient(ellipse at top, rgba(124,140,255,0.18), transparent 60%)",
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