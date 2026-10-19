import assert from 'node:assert/strict';
import { createAgentSession, ModelRuntime, SessionManager, SettingsManager } from '@earendil-works/pi-coding-agent';
import { prepareRuntime, root } from './config.mjs';

if (!process.env.ROUTERAI_API_KEY) throw new Error('Set ROUTERAI_API_KEY in .env before running this explicit network test');
const cfg = await prepareRuntime();
const runtime = await ModelRuntime.create({ modelsPath: `${cfg.agentDir}/models.json`, authPath: `${cfg.agentDir}/auth.json`, modelsStorePath: `${cfg.agentDir}/models-cache.json`, refreshOnCreate: false });
if (runtime.getError()) throw new Error(runtime.getError());
const model = runtime.getModel('routerai', process.env.LLM_MODEL || 'deepseek/deepseek-v4.1-flash-1017');
assert.ok(model, 'Configured model must be available');
const context = { messages: [{ role: 'user' as const, content: 'Reply with the single word OK.', timestamp: Date.now() }] };
const options = { maxTokens: 128, signal: AbortSignal.timeout(30000) };
const response = await runtime.completeSimple(model, context, options);
assert.ok(response.stopReason !== 'error', response.errorMessage || 'Completion failed');
assert.ok(response.content.some(block => block.type === 'text' && block.text.trim()), 'Missing response text');
console.log('RouterAI response: OK');

let deltas = 0;
const stream = runtime.streamSimple(model, context, { ...options, signal: AbortSignal.timeout(30000) });
for await (const event of stream) {
  if (event.type === 'text_delta') deltas++;
  if (event.type === 'error') throw new Error(event.error.errorMessage || 'Streaming failed');
}
assert.ok(deltas > 0, 'No text streaming events received');
console.log('RouterAI streaming: OK');

const { session, extensionsResult } = await createAgentSession({
  cwd: root, agentDir: cfg.agentDir, modelRuntime: runtime, model,
  tools: ['get_capabilities'], thinkingLevel: 'off', sessionManager: SessionManager.inMemory(),
  settingsManager: SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: false } }),
});
assert.equal(extensionsResult.errors.length, 0, 'Extension must load');
let tools = 0;
const timer = setTimeout(() => void session.abort(), 30000);
try {
  session.subscribe(event => {
    if (event.type === 'tool_execution_end' && event.toolName === 'get_capabilities' && !event.isError) tools++;
  });
  await session.prompt('Call get_capabilities exactly once, then describe its stage in one sentence. Do not answer before calling the tool.');
  assert.equal(tools, 1, 'Model must execute get_capabilities exactly once');
  assert.ok(session.getLastAssistantText()?.trim(), 'Missing final model response');
  console.log('RouterAI Pi tool calling: OK');
} finally { clearTimeout(timer); session.dispose(); }
