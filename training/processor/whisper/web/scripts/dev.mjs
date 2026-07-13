import { spawn } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { startLabelServer } from './label-server.mjs';

const labelServer = startLabelServer();
const webRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const vite = spawn(process.execPath, [resolve(webRoot, 'node_modules', 'vite', 'bin', 'vite.js'), '--host'], { cwd: webRoot, stdio: 'inherit' });

function stop() {
  vite.kill();
  labelServer.kill();
}

process.on('SIGINT', stop);
process.on('SIGTERM', stop);
vite.on('exit', (code) => {
  labelServer.kill();
  process.exit(code ?? 0);
});
