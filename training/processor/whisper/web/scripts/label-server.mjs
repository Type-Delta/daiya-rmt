import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const webRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const python = process.env.PYTHON ?? 'python';

export function startLabelServer() {
  const child = spawn(python, ['server.py'], { cwd: webRoot, stdio: 'inherit' });
  child.on('error', (error) => {
    console.error(`Unable to start the local labeling server with ${python}: ${error.message}`);
  });
  return child;
}
