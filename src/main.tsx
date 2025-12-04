import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import "./index.css";
import { setupMockTTSAPI } from "@/utils/mockTTSApi";
import { initGoogleTranslate } from "@/lib/googleTranslate";

// Initialize mock TTS API
setupMockTTSAPI();

// Load Google Translate once at startup
initGoogleTranslate();

createRoot(document.getElementById("root")!).render(<App />);
