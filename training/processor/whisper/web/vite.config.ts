import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const reviewServer = process.env.DAIYA_LABEL_SERVER ?? 'http://127.0.0.1:8765';

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    proxy: {
      '/api': { target: reviewServer, changeOrigin: true },
    },
  },
});
