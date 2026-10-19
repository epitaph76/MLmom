import { fauxAssistantMessage, fauxProvider, fauxToolCall } from '@earendil-works/pi-ai';
import { createAgentSession, ModelRuntime, SessionManager, SettingsManager } from '@earendil-works/pi-coding-agent';
import { mkdtemp } from 'node:fs/promises';
import path from 'node:path';
import { root } from '../packages/pi/test/helpers.js';

/** Runs the actual Pi loop and discovered extension with scripted, offline LLM messages. */
export async function offlineSession() {
  const dir = await mkdtemp(path.join(root, '.local', 'offline-'));
  const runtime = await ModelRuntime.create({ modelsPath: null, authPath: path.join(dir, 'auth.json'), modelsStorePath: path.join(dir, 'models-cache.json'), refreshOnCreate: false });
  const faux = fauxProvider({ provider: 'mlmom-offline', models: [{ id: 'scripted' }] });
  runtime.registerNativeProvider(faux.provider);
  const { session, extensionsResult } = await createAgentSession({
    cwd: root, agentDir: dir, modelRuntime: runtime, model: faux.getModel(), noTools: 'builtin',
    sessionManager: SessionManager.inMemory(),
    settingsManager: SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: false } }),
  });
  if (extensionsResult.errors.length) throw new Error(JSON.stringify(extensionsResult.errors));
  faux.setResponses([
    fauxAssistantMessage(fauxToolCall('research', { task: 'Offline infrastructure smoke test' }), { stopReason: 'toolUse' }),
    context => {
      const last = context.messages.at(-1);
      if (!last || last.role !== 'toolResult') throw new Error('Expected research result');
      const block = last.content.find(block => block.type === 'text');
      if (!block || block.type !== 'text') throw new Error('Missing research response');
      const job = JSON.parse(block.text);
      return fauxAssistantMessage(fauxToolCall('get_research_status', { job_id: job.job_id, wait: true }), { stopReason: 'toolUse' });
    },
    fauxAssistantMessage('Offline MLmom integration complete. Research was mocked.'),
  ]);
  return { session, faux };
}
