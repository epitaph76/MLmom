import { spawn, type ChildProcess } from 'node:child_process';
import { createServer } from 'node:net';
import { setTimeout as sleep } from 'node:timers/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const root = fileURLToPath(new URL('../../../', import.meta.url));
const python = path.join(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');

export async function port() {
  const server = createServer();
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('No port');
  await new Promise<void>(resolve => server.close(() => resolve()));
  return address.port;
}

export async function service(module: string, extraEnv: Record<string, string> = {}) {
  const selectedPort = await port();
  let logs = '';
  const child = spawn(python, ['-m', 'uvicorn', module, '--host', '127.0.0.1', '--port', String(selectedPort)], {
    cwd: root, env: { ...process.env, ...extraEnv }, stdio: ['ignore', 'pipe', 'pipe'],
  });
  child.stdout!.on('data', chunk => { logs += chunk; });
  child.stderr!.on('data', chunk => { logs += chunk; });
  let spawnError: Error | undefined;
  child.on('error', error => { spawnError = error; });
  const url = `http://127.0.0.1:${selectedPort}`;
  for (let i = 0; i < 100; i++) {
    if (spawnError || child.exitCode !== null) throw new Error(`Service failed: ${spawnError?.message ?? logs}`);
    try {
      if ((await fetch(url + '/health', { signal: AbortSignal.timeout(300) })).ok) return { child, url };
    } catch { /* startup */ }
    await sleep(50);
  }
  await stop(child);
  throw new Error(`Service did not start: ${logs}`);
}

export async function stop(child: ChildProcess) {
  if (child.exitCode !== null || child.signalCode !== null) return;
  await new Promise<void>(resolve => {
    const timer = setTimeout(() => { child.kill('SIGKILL'); resolve(); }, 3000);
    child.once('exit', () => { clearTimeout(timer); resolve(); });
    child.kill('SIGTERM');
  });
}
