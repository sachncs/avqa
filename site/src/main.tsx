import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import DocsApp from "./DocsApp";
import "./index.css";

const isDocs = window.location.pathname.startsWith("/avqa/docs");

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    {isDocs ? <DocsApp /> : <App />}
  </StrictMode>
);
