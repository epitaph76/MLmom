import { spawn } from 'node:child_process';
import { python, root } from './config.mjs';

const child = spawn(python, process.argv.slice(2), { cwd: root, stdio: 'inherit' });
child.on('error', () => { console.error('Python environment missing; run npm run setup'); process.exitCode = 1; });
child.on('exit', code => { process.exitCode = code ?? 1; });
