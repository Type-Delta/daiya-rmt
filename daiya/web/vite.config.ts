import fs from 'node:fs';
import path from 'node:path';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// HTTPS for LAN phone testing — getUserMedia needs a secure context off-localhost.
// Convention: drop a cert at certs/dev.crt + certs/dev.key (e.g. from mkcert),
// or point DAIYA_TLS_CERT / DAIYA_TLS_KEY at existing files. No cert → plain HTTP.
function devHttps() {
  const cert = process.env.DAIYA_TLS_CERT ?? path.resolve(__dirname, 'certs/dev.crt');
  const key = process.env.DAIYA_TLS_KEY ?? path.resolve(__dirname, 'certs/dev.key');
  if (!fs.existsSync(cert) || !fs.existsSync(key)) return undefined;
  return { cert: fs.readFileSync(cert), key: fs.readFileSync(key) };
}

// The FastAPI backend; /ws and /api are proxied there in dev.
const server = process.env.DAIYA_SERVER ?? 'http://127.0.0.1:8000';

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    https: devHttps(),
    proxy: {
      '/ws': { target: server, ws: true, changeOrigin: true },
      '/api': { target: server, changeOrigin: true },
    },
  },
});
