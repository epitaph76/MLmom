import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import { setTimeout as delay } from 'node:timers/promises';
import path from 'node:path';
import { prepareRuntime, python, root } from './config.mjs';

const children = [];
let stopping = false;
const cfg = await prepareRuntime();
const webPackage = createRequire(import.meta.url).resolve('@agegr/pi-web/package.json');
const webDir = path.dirname(webPackage);
const webRequire = createRequire(webPackage);
const nextBin = webRequire.resolve('next/dist/bin/next');
const { rotatePreviewSecrets } = webRequire('./bin/rotate-preview-secrets');

export async function stop() {
  if (stopping) return;
  stopping = true;
  await Promise.all(children.map(child => new Promise(resolve => {
    if (child.exitCode !== null) return resolve();
    const timer = setTimeout(() => { child.kill('SIGKILL'); resolve(); }, 4000);
    timer.unref();
    child.once('exit', () => { clearTimeout(timer); resolve(); });
    child.kill('SIGTERM');
  })));
  if (process.connected) process.disconnect();
}

function start(executable, args, extraEnv = {}, cwd = root) {
  const child = spawn(executable, args, { cwd, stdio: 'inherit', env: { ...process.env, ...extraEnv } });
  children.push(child);
  child.on('error', async error => { console.error(error.message); process.exitCode = 1; await stop(); });
  child.on('exit', async code => {
    if (!stopping) { console.error(`Service stopped (${code})`); process.exitCode = code || 1; await stop(); }
  });
  return child;
}

async function ready(url) {
  for (let i = 0; i < 100 && !stopping; i++) {
    try {
      const response = await fetch(`${url}/health`, { signal: AbortSignal.timeout(500) });
      if (response.ok && (await response.json()).status === 'ok') return;
    } catch { /* Allow service startup. */ }
    await delay(100);
  }
  if (!stopping) throw new Error(`Service unavailable: ${url}`);
}

// Refuse occupied ports so a second launcher never adopts another process.
async function assertFree(host, port) {
  const { createServer } = await import('node:net');
  await new Promise((resolve, reject) => {
    const server = createServer();
    server.once('error', () => reject(new Error(`Address already in use: ${host}:${port}`)));
    server.listen(port, host, () => server.close(resolve));
  });
}

for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => void stop());
// The test harness uses IPC to request the same shutdown path on Windows and Linux.
if (process.send) process.on('message', message => { if (message === 'shutdown') void stop(); });
try {
  await assertFree(cfg.backendHost, cfg.backendPort);
  await assertFree(cfg.webHost, cfg.webPort);
  if (cfg.mode === 'mock') await assertFree(cfg.researchHost, cfg.researchPort);
  start(python, ['-m', 'uvicorn', 'mlmom.backend:app', '--host', cfg.backendHost, '--port', String(cfg.backendPort)]);
  if (cfg.mode === 'mock') start(python, ['-m', 'uvicorn', 'mlmom.research_mock:app', '--host', cfg.researchHost, '--port', String(cfg.researchPort)]);
  await Promise.all([ready(cfg.backendUrl), ready(cfg.researchUrl)]);
  if (!stopping) {
    // Match upstream secret rotation, but own the Next process directly for Windows shutdown.
    const rotation = rotatePreviewSecrets(path.join(webDir, '.next'));
    if (!rotation.ok) throw new Error('Pi Web preview secret rotation failed');
    start(process.execPath, [nextBin, 'start', '-H', cfg.webHost, '-p', String(cfg.webPort)], {
      PI_WEB_NO_OPEN: '1', PI_WEB_SKIP_VERSION_CHECK: '1',
    }, webDir);
    console.log(`MLmom: http://${cfg.webHost}:${cfg.webPort} — select project ${root}`);
  }
} catch (error) { console.error(error.message); process.exitCode = 1; await stop(); }
