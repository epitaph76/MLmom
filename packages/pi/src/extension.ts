import { randomUUID } from 'node:crypto';
import { Type } from '@earendil-works/pi-ai';
import { defineTool, type ExtensionAPI } from '@earendil-works/pi-coding-agent';
import { BackendClient, ResearchClient } from './client.js';

const textResult = (value: unknown) => ({ content: [{ type: 'text' as const, text: JSON.stringify(value) }], details: value });

function clients() {
  const options = {
    timeoutMs: Number(process.env.HTTP_TIMEOUT_MS || 10000),
    getRetries: Number(process.env.HTTP_GET_RETRIES || 2),
  };
  return {
    research: new ResearchClient(process.env.RESEARCH_URL || 'http://127.0.0.1:8011', options),
    backend: new BackendClient(process.env.BACKEND_URL || 'http://127.0.0.1:8010', options),
  };
}

export default function mlmom(pi: ExtensionAPI) {
  pi.registerTool(defineTool({
    name: 'research', label: 'Research', description: 'Start asynchronous ML research. Returns job_id; use get_research_status. mock=true means synthetic fixtures, not real research.',
    parameters: Type.Object({
      request_id: Type.Optional(Type.String({ minLength: 1, maxLength: 128 })),
      task: Type.String({ minLength: 1, maxLength: 20000 }),
      task_type: Type.Optional(Type.String()), metric: Type.Optional(Type.String()),
      dataset_summary: Type.Optional(Type.Record(Type.String(), Type.Unknown())),
      constraints: Type.Optional(Type.Record(Type.String(), Type.Unknown())),
    }),
    async execute(_id, params, signal) {
      return textResult(await clients().research.create({ ...params, request_id: params.request_id ?? randomUUID() }, signal));
    },
  }));
  pi.registerTool(defineTool({
    name: 'get_research_status', label: 'Research status', description: 'Get research state or wait for completion. Stopping a wait requests cancellation; polling timeout or lost connection does not cancel the job.',
    parameters: Type.Object({ job_id: Type.String({ minLength: 1 }), wait: Type.Optional(Type.Boolean()) }),
    async execute(_id, params, signal) {
      const { research } = clients();
      const job = params.wait ? await research.waitForJob(params.job_id, {
        signal, cancelOnAbort: true,
        pollIntervalMs: Number(process.env.RESEARCH_POLL_INTERVAL_MS || 500),
        pollTimeoutMs: Number(process.env.RESEARCH_POLL_TIMEOUT_MS || 60000),
      }) : await research.get(params.job_id, signal);
      return textResult(job);
    },
  }));
  pi.registerTool(defineTool({
    name: 'cancel_research', label: 'Cancel research', description: 'Explicitly cancel a research job. Completed results are preserved.',
    parameters: Type.Object({ job_id: Type.String({ minLength: 1 }) }),
    async execute(_id, params, signal) { return textResult(await clients().research.cancel(params.job_id, signal)); },
  }));
  pi.registerTool(defineTool({
    name: 'get_capabilities', label: 'ML capabilities', description: 'Check implemented ML capabilities before attempting training.', parameters: Type.Object({}),
    async execute(_id, _params, signal) { return textResult(await clients().backend.capabilities(signal)); },
  }));
  pi.registerTool(defineTool({
    name: 'get_history', label: 'Experiment history', description: 'Get actual experiment history. Currently empty; training is not implemented.', parameters: Type.Object({}),
    async execute(_id, _params, signal) { return textResult(await clients().backend.history(signal)); },
  }));
  pi.registerTool(defineTool({
    name: 'profile_data', label: 'Profile data', description: 'Reserved data profiling interface. Currently fails with NOT_IMPLEMENTED.',
    parameters: Type.Object({ dataset_ref: Type.String({ minLength: 1 }) }),
    async execute(_id, params, signal) { return textResult(await clients().backend.profile(params, signal)); },
  }));
  pi.registerTool(defineTool({
    name: 'train', label: 'Train', description: 'Reserved training interface accepting a structured recipe. Currently fails with NOT_IMPLEMENTED.',
    parameters: Type.Object({ dataset_ref: Type.String({ minLength: 1 }), recipe: Type.Record(Type.String(), Type.Unknown()) }),
    async execute(_id, params, signal) { return textResult(await clients().backend.train(params, signal)); },
  }));
  pi.registerTool(defineTool({
    name: 'evaluate', label: 'Evaluate', description: 'Reserved independent evaluation interface. Currently fails with NOT_IMPLEMENTED.',
    parameters: Type.Object({ experiment_id: Type.String({ minLength: 1 }) }),
    async execute(_id, params, signal) { return textResult(await clients().backend.evaluate(params, signal)); },
  }));
}
