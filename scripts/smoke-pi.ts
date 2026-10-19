import { mkdir } from 'node:fs/promises';
import path from 'node:path';
import { offlineSession } from './offline-session.js';
import { root } from '../packages/pi/test/helpers.js';

await mkdir(path.join(root, '.local'), { recursive: true });
const { session } = await offlineSession();
try {
  session.subscribe(event => {
    if (event.type === 'tool_execution_end') {
      if (event.isError) throw new Error(`Failed tool ${event.toolName}`);
      console.log(`${event.toolName}: OK`);
    }
  });
  await session.prompt('Test research integration');
  if (!session.getLastAssistantText()?.includes('complete')) throw new Error('Pi smoke failed');
  console.log('Pi discovery and research tool loop: OK (offline LLM, mock research)');
} finally { session.dispose(); }
