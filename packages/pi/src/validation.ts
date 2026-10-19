import { Ajv2020 } from 'ajv/dist/2020.js';
import addFormats from 'ajv-formats';
import research from '../../../contracts/research.openapi.json' with { type: 'json' };
import backend from '../../../contracts/backend.openapi.json' with { type: 'json' };

const ajv = new Ajv2020({ strict: false, allErrors: true });
addFormats.default(ajv);
const cache = new Map<string, ReturnType<typeof ajv.compile>>();

export function validatePayload(api: 'research' | 'backend', name: string, value: unknown): void {
  const key = `${api}:${name}`;
  let validator = cache.get(key);
  if (!validator) {
    const doc = api === 'research' ? research : backend;
    const definitions = JSON.parse(JSON.stringify(doc.components.schemas).replaceAll('#/components/schemas/', '#/$defs/'));
    validator = ajv.compile({ $defs: definitions, $ref: `#/$defs/${name}` });
    cache.set(key, validator);
  }
  if (!validator(value)) throw new Error(`Invalid ${key} payload: ${ajv.errorsText(validator.errors)}`);
  if (api === 'research' && name === 'ResearchJob') {
    const job = value as { status: string; result: any; error: unknown };
    if ((job.status === 'succeeded') !== (job.result != null) || (job.status === 'failed') !== (job.error != null)) {
      throw new Error('Invalid research job state');
    }
    if (job.result) {
      const ids = new Set(job.result.cards.map((card: { id: string }) => card.id));
      const hypothesisIds = new Set(job.result.hypotheses.map((h: { id: string }) => h.id));
      if (ids.size !== job.result.cards.length || hypothesisIds.size !== job.result.hypotheses.length ||
          job.result.hypotheses.some((h: { card_ids: string[] }) => h.card_ids.some(id => !ids.has(id)))) {
        throw new Error('Invalid research card references');
      }
    }
  }
}
