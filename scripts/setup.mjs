import { spawn } from 'node:child_process';
import { access, copyFile } from 'node:fs/promises';
import path from 'node:path';
import { python, root } from './config.mjs';

const run = (executable, args) => new Promise((resolve, reject) => {
  const child = spawn(executable, args, { cwd: root, stdio: 'inherit' });
  child.on('error', reject);
  child.on('exit', code => code === 0 ? resolve() : reject(new Error(`Setup exited ${code}`)));
});
try {
  try { await access(python); } catch { await run(process.env.PYTHON || 'python', ['-m', 'venv', '.venv']); }
  await run(python, ['-m', 'pip', 'install', '-r', 'requirements-dev.lock']);
  await run(python, ['-m', 'pip', 'install', '--no-deps', '-e', '.']);
  const env = path.join(root, '.env');
  try { await access(env); } catch { await copyFile(path.join(root, '.env.example'), env); }
  console.log('Ready. Set ROUTERAI_API_KEY in .env, then npm start.');
} catch (error) { console.error(error.message); process.exitCode = 1; }
