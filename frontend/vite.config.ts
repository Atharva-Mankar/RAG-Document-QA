import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Requests to /api/... are forwarded to the FastAPI backend.
    //
    // The FastAPI routers are registered WITHOUT an /api prefix
    // (health -> /health, documents -> /documents, chat -> /chat),
    // so `rewrite` strips the /api prefix before forwarding:
    //
    //   /api/health  ->  http://127.0.0.1:8000/health
    //   /api/chat    ->  http://127.0.0.1:8000/chat
    //   /api/documents -> http://127.0.0.1:8000/documents
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
