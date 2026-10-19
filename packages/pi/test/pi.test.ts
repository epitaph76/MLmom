import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import path from 'node:path';
import { test } from 'node:test';
import { offlineSession } from '../../../scripts/offline-session.js';
import { root, service, stop } from './helpers.js';

test('Pi discovers all eight tools and executes research through the actual agent loop', async () => {
  const research = await service('mlmom.research_mock:app');
  process.env.RESEARCH_URL = research.url;
  await mkdir(path.join(root, '.local'), { recursive: true });
  let session: Awaited<ReturnType<typeof offlineSession>>['session'] | undefined;
  try {
    ({ session } = await offlineSession());
    const names = session.getAllTools().map(tool => tool.name);
    for (const name of ['research', 'get_research_status', 'cancel_research', 'get_capabilities', 'profile_data', 'train', 'evaluate', 'get_history']) assert.ok(names.includes(name), name);
    const results: unknown[] = [];
    session.subscribe(event => { if (event.type === 'tool_execution_end') { assert.equal(event.isError, false); results.push(event.result); } });
    await session.prompt('Run the offline research transport check');
    assert.equal(results.length, 2);
    const final = results[1] as { details: { status: string; mock: boolean } };
    assert.equal(final.details.status, 'succeeded');
    assert.equal(final.details.mock, true);
    assert.match(session.getLastAssistantText() ?? '', /complete/);
  } finally { session?.dispose(); await stop(research.child); }
});
