import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { readFile, readdir } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import { BackendClient, HttpError, PollTimeoutError, ResearchClient } from '../src/client.js';
import { validatePayload } from '../src/validation.js';
import { service, stop } from './helpers.js';

let research: Awaited<ReturnType<typeof service>>;
let backend: Awaited<ReturnType<typeof service>>;
before(async () => {
  research = await service('mlmom.research_mock:app', { RESEARCH_MOCK_DELAY_MS: '100' });
  backend = await service('mlmom.backend:app');
});
after(async () => { if (research) await stop(research.child); if (backend) await stop(backend.child); });

test('shared examples validate with the TypeScript runtime schema', async () => {
  const dir = new URL('../../../contracts/examples/', import.meta.url);
  for (const file of await readdir(dir)) {
    const schema = file === 'request.json' ? 'ResearchRequest' : file === 'error.json' ? 'ApiError' : 'ResearchJob';
    validatePayload('research', schema, JSON.parse(await readFile(new URL(file, dir), 'utf8')));
  }
});

test('HTTP lifecycle, idempotency, empty research, failure and terminal cancellation', async () => {
  const client = new ResearchClient(research.url);
  for (const scenario of ['success', 'empty', 'failed'] as const) {
    const body = { request_id: randomUUID(), task: 'transport test' };
    const initial = await client.create(body, undefined, scenario);
    assert.equal((await client.create(body, undefined, scenario)).job_id, initial.job_id);
    await assert.rejects(client.create({ ...body, task: 'other' }, undefined, scenario), (error: unknown) => error instanceof HttpError && error.status === 409);
    const final = await client.waitForJob(initial.job_id, { pollIntervalMs: 10 });
    assert.equal(final.status, scenario === 'failed' ? 'failed' : 'succeeded');
    assert.equal(final.mock, true);
    if (scenario !== 'failed') assert.equal(final.result?.cards.length, scenario === 'empty' ? 0 : 1);
    assert.deepEqual(await client.cancel(initial.job_id), final);
  }
  await assert.rejects(client.get('missing'), (error: unknown) => error instanceof HttpError && error.status === 404);
});

test('explicit abort while waiting propagates cancellation', async () => {
  const client = new ResearchClient(research.url);
  const job = await client.create({ request_id: randomUUID(), task: 'cancel test' });
  const controller = new AbortController();
  const pending = client.waitForJob(job.job_id, { signal: controller.signal, cancelOnAbort: true, pollIntervalMs: 10 });
  controller.abort();
  await assert.rejects(pending);
  assert.equal((await client.get(job.job_id)).status, 'cancelled');
  assert.equal((await client.cancel(job.job_id)).status, 'cancelled');
});

test('polling timeout leaves the server job running and recoverable', async () => {
  const client = new ResearchClient(research.url);
  const job = await client.create({ request_id: randomUUID(), task: 'timeout test' });
  await assert.rejects(client.waitForJob(job.job_id, { pollTimeoutMs: 10, pollIntervalMs: 5 }), PollTimeoutError);
  const recovered = await client.waitForJob(job.job_id, { pollIntervalMs: 10 });
  assert.equal(recovered.status, 'succeeded');
});

test('GET retries are bounded; POST and nonretryable errors are never repeated', async () => {
  let calls = 0;
  const unavailable = new ResearchClient('http://example.invalid', { getRetries: 2, retryDelayMs: 0,
    fetch: (async () => { calls++; throw new TypeError('offline'); }) as typeof fetch });
  await assert.rejects(unavailable.get('x'));
  assert.equal(calls, 3);
  calls = 0;
  await assert.rejects(unavailable.create({ request_id: 'r', task: 't' }));
  assert.equal(calls, 1);
  calls = 0;
  const forbidden = new ResearchClient('http://example.invalid', { fetch: (async () => {
    calls++;
    return Response.json({ code: 'NO', message: 'No', retryable: false, request_id: null, mock: false }, { status: 404 });
  }) as typeof fetch });
  await assert.rejects(forbidden.get('x'), HttpError);
  assert.equal(calls, 1);
});

test('request timeouts retry GET; disconnect never sends cancel', async () => {
  const methods: string[] = [];
  const client = new ResearchClient('http://example.invalid', { timeoutMs: 10, getRetries: 1, retryDelayMs: 0,
    fetch: (async (_url, init) => {
      methods.push(init?.method ?? '');
      return new Promise((_resolve, reject) => {
        const timer = setTimeout(() => reject(new Error('test guard')), 500);
        init?.signal?.addEventListener('abort', () => { clearTimeout(timer); reject(init.signal?.reason); }, { once: true });
      });
    }) as typeof fetch });
  await assert.rejects(client.waitForJob('x', { cancelOnAbort: true }), { name: 'TimeoutError' });
  assert.deepEqual(methods, ['GET', 'GET']);
});

test('reject malformed server responses and impossible result states', async () => {
  const client = new ResearchClient('http://example.invalid', { fetch: (async () => Response.json({ status: 'succeeded' })) as typeof fetch });
  await assert.rejects(client.get('x'), /Invalid research/);
  assert.throws(() => validatePayload('research', 'ResearchJob', { job_id: 'x', request_id: 'r', status: 'succeeded' }), /Invalid research job state/);
});

test('backend exposes actual capabilities and refuses unsupported operations', async () => {
  const client = new BackendClient(backend.url);
  assert.equal((await client.capabilities()).stage, 'infrastructure');
  assert.deepEqual(await client.history(), { experiments: [] });
  for (const operation of [() => client.profile({ dataset_ref: 'x' }), () => client.train({ dataset_ref: 'x', recipe: {} }), () => client.evaluate({ experiment_id: 'x' })]) {
    await assert.rejects(operation(), (error: unknown) => error instanceof HttpError && error.status === 501 && error.body.code === 'NOT_IMPLEMENTED');
  }
});
