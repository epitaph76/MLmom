"""Detect unchanged failed executions, without judging an agent's explanation."""
from .execution import config_hash


def repair_allowed(store, key, candidate, plan, error, attempts, maximum=None):
    if maximum is not None and attempts >= maximum:
        return False
    lines=[line.strip() for line in str(error).splitlines() if line.strip()]
    signature=config_hash(dict(code_hash=candidate.code_hash,configuration=plan.model_dump(mode='json',
                               exclude={'hypothesis','agent_notes','priority','hpo_trial_budget'}),
                               error=lines[-1] if lines else 'Unknown runtime error'))
    previous=store.get('repair_progress',key) or {}
    repeated=previous.get('repeated',0)+1 if previous.get('signature')==signature else 1
    store.put('repair_progress',key,dict(signature=signature,repeated=repeated,attempts=attempts+1))
    if repeated > 1:
        store.event('repair_stalled',key,dict(reason='Unchanged source/configuration repeated the same failure'))
        return False
    return True
