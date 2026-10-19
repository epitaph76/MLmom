import { setTimeout as sleep } from 'node:timers/promises';
import type { components as Research } from './research.generated.js';
import type { components as Backend } from './backend.generated.js';
import { validatePayload } from './validation.js';

export type ResearchRequest = Research['schemas']['ResearchRequest'];
export type ResearchJob = Research['schemas']['ResearchJob'];
export type ApiError = Research['schemas']['ApiError'];
export type MockScenario = 'success' | 'empty' | 'failed';

export class HttpError extends Error {
  constructor(public status: number, public body: ApiError) { super(`${body.code}: ${body.message}`); }
}

export class PollTimeoutError extends Error {
  constructor(public jobId: string) { super(`Polling timed out for ${jobId}; job was not cancelled. Query it again with get_research_status.`); }
}

export interface ClientOptions {
  timeoutMs?: number;
  getRetries?: number;
  retryDelayMs?: number;
  fetch?: typeof fetch;
}

function nonnegative(value: number, name: string, allowZero = false) {
  if (!Number.isSafeInteger(value) || value < (allowZero ? 0 : 1)) throw new Error(`Invalid ${name}`);
  return value;
}

export class HttpClient {
  readonly timeoutMs: number;
  readonly getRetries: number;
  private readonly retryDelayMs: number;
  private readonly fetcher: typeof fetch;
  private readonly base: string;

  constructor(baseUrl: string, private readonly api: 'research' | 'backend', options: ClientOptions = {}) {
    const url = new URL(baseUrl);
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) throw new Error('Invalid service URL');
    this.base = baseUrl.replace(/\/$/, '');
    this.timeoutMs = nonnegative(options.timeoutMs ?? 10000, 'timeoutMs');
    this.getRetries = nonnegative(options.getRetries ?? 2, 'getRetries', true);
    this.retryDelayMs = nonnegative(options.retryDelayMs ?? 100, 'retryDelayMs', true);
    this.fetcher = options.fetch ?? fetch;
  }

  async request<T>(method: 'GET' | 'POST', path: string, schema: string, body?: unknown, signal?: AbortSignal, headers: Record<string, string> = {}): Promise<T> {
    const attempts = method === 'GET' ? this.getRetries + 1 : 1;
    for (let attempt = 0; attempt < attempts; attempt++) {
      signal?.throwIfAborted();
      try {
        const requestSignal = AbortSignal.any([AbortSignal.timeout(this.timeoutMs), ...(signal ? [signal] : [])]);
        const response = await this.fetcher(this.base + path, {
          method, signal: requestSignal,
          headers: { accept: 'application/json', ...(body === undefined ? {} : { 'content-type': 'application/json' }), ...headers },
          ...(body === undefined ? {} : { body: JSON.stringify(body) }),
        });
        const payload: unknown = await response.json();
        if (!response.ok) {
          validatePayload(this.api, 'ApiError', payload);
          throw new HttpError(response.status, payload as ApiError);
        }
        validatePayload(this.api, schema, payload);
        return payload as T;
      } catch (error) {
        if (signal?.aborted) throw signal.reason;
        const transient = error instanceof HttpError ? error.body.retryable && (error.status === 429 || error.status >= 500) :
          error instanceof TypeError || (error instanceof Error && error.name === 'TimeoutError');
        if (!transient || attempt + 1 === attempts) throw error;
        await sleep(this.retryDelayMs * (attempt + 1), undefined, { signal });
      }
    }
    throw new Error('Unreachable');
  }
}

export class ResearchClient extends HttpClient {
  constructor(baseUrl: string, options: ClientOptions = {}) { super(baseUrl, 'research', options); }

  create(request: ResearchRequest, signal?: AbortSignal, scenario?: MockScenario) {
    validatePayload('research', 'ResearchRequest', request);
    return this.request<ResearchJob>('POST', '/v1/research/jobs', 'ResearchJob', request, signal,
      scenario ? { 'x-mock-scenario': scenario } : {});
  }

  get(jobId: string, signal?: AbortSignal) {
    return this.request<ResearchJob>('GET', `/v1/research/jobs/${encodeURIComponent(jobId)}`, 'ResearchJob', undefined, signal);
  }

  cancel(jobId: string, signal?: AbortSignal) {
    return this.request<ResearchJob>('POST', `/v1/research/jobs/${encodeURIComponent(jobId)}/cancel`, 'ResearchJob', undefined, signal);
  }

  async waitForJob(jobId: string, options: { signal?: AbortSignal; pollIntervalMs?: number; pollTimeoutMs?: number; cancelOnAbort?: boolean } = {}) {
    const interval = nonnegative(options.pollIntervalMs ?? 500, 'pollIntervalMs');
    const timeout = nonnegative(options.pollTimeoutMs ?? 60000, 'pollTimeoutMs');
    const deadline = AbortSignal.timeout(timeout);
    const signal = AbortSignal.any([deadline, ...(options.signal ? [options.signal] : [])]);
    try {
      while (true) {
        const job = await this.get(jobId, signal);
        if (['succeeded', 'failed', 'cancelled'].includes(job.status)) return job;
        await sleep(interval, undefined, { signal });
      }
    } catch (error) {
      if (options.signal?.aborted) {
        if (options.cancelOnAbort) {
          try { await this.cancel(jobId); } catch { throw new Error(`Wait aborted; cancellation of ${jobId} could not be confirmed. Query job status.`, { cause: error }); }
        }
        throw options.signal.reason;
      }
      if (deadline.aborted) throw new PollTimeoutError(jobId);
      throw error; // Network failure never implies server-side cancellation.
    }
  }
}

export class BackendClient extends HttpClient {
  constructor(baseUrl: string, options: ClientOptions = {}) { super(baseUrl, 'backend', options); }
  capabilities(signal?: AbortSignal) { return this.request<Backend['schemas']['Capabilities']>('GET', '/v1/capabilities', 'Capabilities', undefined, signal); }
  history(signal?: AbortSignal) { return this.request<Backend['schemas']['History']>('GET', '/v1/history', 'History', undefined, signal); }
  profile(body: Backend['schemas']['ProfileRequest'], signal?: AbortSignal) { return this.request('POST', '/v1/profile', 'ApiError', body, signal); }
  train(body: Backend['schemas']['TrainRequest'], signal?: AbortSignal) { return this.request('POST', '/v1/train', 'ApiError', body, signal); }
  evaluate(body: Backend['schemas']['EvaluateRequest'], signal?: AbortSignal) { return this.request('POST', '/v1/evaluate', 'ApiError', body, signal); }
}
