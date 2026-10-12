import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In dev, the API runs separately: uvicorn aitext.api:app --app-dir ../src --port 8000
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});
