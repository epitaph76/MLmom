"""Durable rejection counts shared by inner repairs and outer planning retries."""
from __future__ import annotations

import hashlib
import json


class PlanningRejections:
    def __init__(self, progress, persist, maximum):
        self.records = progress.setdefault('rejections', {})
        self.persist = persist
        self.maximum = maximum
        self.variants = {}

    def key(self, hypothesis):
        data = hypothesis.model_dump(mode='json')
        identity = {key: data.get(key) for key in
                    ('description', 'based_on', 'kind', 'model_family', 'family_tools')}
        identity['description'] = ' '.join(identity['description'].lower().split())
        identity['backend'] = self.variants.get(hypothesis.hypothesis_id)
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def record(self, hypothesis):
        return self.records.get(self.key(hypothesis), {})

    def blocked(self, hypothesis):
        return self.record(hypothesis).get('blocked', False)

    def reject(self, hypothesis, error, plan=None):
        key = self.key(hypothesis)
        previous = self.records.get(key, {})
        signature = ' '.join(str(error).split())
        executable = plan.model_dump(mode='json',exclude={'hypothesis','agent_notes'}) if plan is not None else None
        fingerprint = hashlib.sha256(json.dumps(executable,sort_keys=True).encode()).hexdigest()
        unchanged = signature == previous.get('signature') and fingerprint == previous.get('plan_fingerprint')
        repeated = previous.get('repeated', 0) + 1 if unchanged else 1
        attempts = previous.get('attempts', 0) + 1
        self.records[key] = dict(hypothesis_id=hypothesis.hypothesis_id,
            description=hypothesis.description, attempts=attempts, repeated=repeated,
            backend=self.variants.get(hypothesis.hypothesis_id),
            signature=signature, plan_fingerprint=fingerprint,error=str(error)[:2000],
            blocked=repeated >= min(2, self.maximum))
        self.persist()
        return self.blocked(hypothesis)

    def summaries(self):
        return [{key: value for key, value in record.items() if key != 'signature'}
                for record in self.records.values()]
