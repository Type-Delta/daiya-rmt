import { startLabelServer } from './label-server.mjs';

const labelServer = startLabelServer();
process.on('SIGINT', () => labelServer.kill());
process.on('SIGTERM', () => labelServer.kill());
