import assert from 'node:assert/strict';
import { fork } from 'node:child_process';
import { test } from 'node:test';
import { setTimeout as sleep } from 'node:timers/promises';
import { randomUUID } from 'node:crypto';
import path from 'node:path';
import { port, root } from './helpers.js';

test('launcher serves Pi Web and APIs, then closes all owned listeners', { timeout: 30000 }, async () => {
  const [backendPort, researchPort, webPort] = [await port(), await port(), await port()];
  const urls = [backendPort, researchPort, webPort].map(port => `http://127.0.0.1:${port}`);
  const runtimeDir = path.join(root, '.local', `launcher-${randomUUID()}`);
  let output = '';
  const child = fork(path.join(root, 'scripts/start.mjs'), [], {
    cwd: root, execArgv: [], stdio: ['ignore', 'pipe', 'pipe', 'ipc'],
    env: { ...process.env, ROUTERAI_API_KEY: '', PI_CODING_AGENT_DIR: runtimeDir,
      BACKEND_HOST: '127.0.0.1', RESEARCH_HOST: '127.0.0.1', PI_WEB_HOSTNAME: '127.0.0.1',
      BACKEND_PORT: String(backendPort), RESEARCH_PORT: String(researchPort), PI_WEB_PORT: String(webPort),
      BACKEND_URL: urls[0], RESEARCH_URL: urls[1], RESEARCH_MODE: 'mock' },
  });
  child.stdout!.on('data', chunk => { output += chunk; });
  child.stderr!.on('data', chunk => { output += chunk; });
  const exited = new Promise<number | null>(resolve => child.once('exit', resolve));
  try {
    let ready = false;
    for (let i = 0; i < 150; i++) {
      if (child.exitCode !== null) assert.fail(output);
      try {
        const response = await fetch(urls[2], { signal: AbortSignal.timeout(500) });
        if (response.ok && (await response.text()).includes('Pi Web')) { ready = true; break; }
      } catch { /* startup */ }
      await sleep(100);
    }
    assert.ok(ready, output);
    for (const url of urls.slice(0, 2)) assert.equal((await fetch(url + '/health')).status, 200);
    child.send('shutdown');
    const result = await Promise.race([exited, sleep(7000).then(() => 'timeout')]);
    assert.equal(result, 0, output);
    for (const url of urls) await assert.rejects(fetch(url, { signal: AbortSignal.timeout(500) }));
  } finally {
    if (child.connected) child.send('shutdown');
    if (child.exitCode === null && child.signalCode === null) {
      await Promise.race([exited, sleep(5000)]);
      if (child.exitCode === null && child.signalCode === null) child.kill();
    }
  }
});
