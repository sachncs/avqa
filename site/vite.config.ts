import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import fs from "node:fs";
import path from "node:path";

const versionSource = fs.readFileSync(
  path.resolve(import.meta.dirname, "../src/avqa/version.py"),
  "utf8",
);
const versionMatch = versionSource.match(/__version__ = "([^"]+)"/);
if (versionMatch === null) {
  throw new Error("Unable to read canonical AVQA version from src/avqa/version.py");
}
const avqaVersion = versionMatch[1];

export default defineConfig({
  plugins: [react()],
  define: {
    "import.meta.env.VITE_AVQA_VERSION": JSON.stringify(avqaVersion),
  },
  resolve: {
    alias: {
      "@": path.resolve(import.meta.dirname, "./src"),
    },
  },
  base: "/avqa/",
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: false,
    target: "es2022",
  },
  server: {
    host: true,
    port: 5173,
  },
});
