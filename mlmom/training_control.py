"""Checkpoint decisions in the controller; workers never receive LLM credentials."""
from __future__ import annotations

import math
import os
import uuid
import time
from pathlib import Path

from .agents import role_window
from .persistence import atomic_json
from .pipeline_contracts import TrainingCheckpoint, TrainingDecision
from .tracing import emit


def commit_mailbox(path, value):
    """First writer wins, including controller/worker timeout races."""
    import json
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        atomic_json(temporary,value)
        try:os.link(temporary,path)
        except FileExistsError:pass
    finally:
        temporary.unlink(missing_ok=True)
    return json.loads(path.read_text(encoding='utf-8'))


def validate_decision(decision, checkpoint):
    decision = TrainingDecision.model_validate(decision)
    if decision.action == 'continue' and (checkpoint.remaining_seconds <= 0 or checkpoint.remaining_trials <= 0):
        raise ValueError('No remaining search budget')
    if decision.next_params and decision.action != 'continue':
        raise ValueError('Trial suggestions require continue')
    if len(decision.next_params) > checkpoint.remaining_trials:
        raise ValueError('Too many trial suggestions')
    space = checkpoint.plan.get('search_space', {})
    for params in decision.next_params:
        for key, value in params.items():
            if key not in space: raise ValueError(f'Unknown search parameter: {key}')
            spec = space[key]
            kind = spec['type']
            if kind == 'categorical':
                if not any(type(value) is type(c) and value == c for c in spec['choices']):
                    raise ValueError(f'Invalid categorical value: {key}')
            else:
                if type(value) not in ({int} if kind == 'int' else {int, float}):
                    raise ValueError(f'Invalid parameter type: {key}')
                if not math.isfinite(value) or not spec['low'] <= value <= spec['high']:
                    raise ValueError(f'Parameter outside approved bounds: {key}')
    return decision


def fallback(checkpoint, reason):
    return TrainingDecision(action=checkpoint.default_action, reason=reason)


def decide(client, checkpoint, skills, *, seconds=30):
    try:
        with role_window(client, 'training_agent', checkpoint.checkpoint_id, min(seconds, checkpoint.remaining_seconds)):
            answer = client.generate('training_agent',
                'Control an existing training run at this checkpoint only. Do not write code or change the model, '
                'metric, splits, base parameters or search distributions. Inspect internal scores, failures, trial '
                'parameters and time. Choose continue, stop (end search) or refit (end search). Both stop and '
                'refit require a fresh fit of the selected configuration on all allowed training rows before evaluation. '
                'Optional next_params are queued suggestions within the approved search space, never a new space. '
                'Respect remaining budgets. Explain decisions and put future model changes in recommendations.',
                dict(checkpoint=checkpoint.model_dump(mode='json'), skills=skills), TrainingDecision, None)
        return validate_decision(answer, checkpoint)
    except Exception as error:
        emit('training.fallback', {'checkpoint_id': checkpoint.checkpoint_id, 'reason': str(error)[:300]})
        return fallback(checkpoint, 'Programmatic policy: '+str(error)[:300])


def worker_checkpoint(root, checkpoint, *, seconds=30):
    """Atomic mailbox with a bounded wait, charged to the existing worker deadline."""
    folder = Path(root)/'control'
    request = folder/(checkpoint.checkpoint_id+'.request.json')
    response = folder/(checkpoint.checkpoint_id+'.decision.json')
    commit_mailbox(request, checkpoint)
    until = time.monotonic()+min(seconds, max(0., checkpoint.remaining_seconds)) if seconds is not None else float('inf')
    while True:
        if response.exists():
            try:
                return validate_decision(TrainingDecision.model_validate_json(response.read_text(encoding='utf-8')), checkpoint)
            except (ValueError, OSError):
                return fallback(checkpoint, 'Invalid controller decision; programmatic policy')
        if time.monotonic() >= until:
            # Persist timeout decisions too: a resumed worker must not ask again.
            decision = fallback(checkpoint, 'Controller unavailable; programmatic policy')
            return TrainingDecision.model_validate(commit_mailbox(response,decision))
        time.sleep(.05)
