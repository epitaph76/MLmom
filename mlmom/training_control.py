"""Checkpoint decisions in the controller; workers never receive LLM credentials."""
from __future__ import annotations

from .agents import GlobalAPIError
from .resources import WorkerInfrastructureError

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
    if decision.action == 'continue' and ((checkpoint.remaining_seconds is not None and checkpoint.remaining_seconds <= 0) or
                                         checkpoint.remaining_trials + decision.additional_trials <= 0):
        raise ValueError('No remaining search budget')
    if decision.additional_trials and decision.action != 'continue':
        raise ValueError('Additional trials require continue')
    if checkpoint.trial_limit is not None and decision.additional_trials > max(0, checkpoint.trial_limit-len(
            [t for t in checkpoint.trials if t.get('origin') != 'baseline'])-checkpoint.remaining_trials):
        raise ValueError('Additional trials exceed the explicit user limit')
    return decision


def fallback(checkpoint, reason):
    return TrainingDecision(action=checkpoint.default_action, reason=reason)


def decide(client, checkpoint, skills, *, seconds=30):
    try:
        allowance = min(seconds, checkpoint.remaining_seconds) if checkpoint.remaining_seconds is not None else seconds
        with role_window(client, 'training_agent', checkpoint.checkpoint_id, allowance):
            answer = client.generate('training_agent',
                'Control an existing training run at this checkpoint only. Do not write code or change the model, '
                'metric, splits, base parameters or search distributions. Inspect internal scores, failures, trial '
                'parameters and time. Choose continue, stop (end search) or refit (end search). Both stop and '
                'refit require a fresh fit of the selected configuration on all allowed training rows before evaluation. '
                'Optuna alone selects numerical trial configurations. To extend this unchanged experiment, '
                'return additional_trials with continue; respect trial_limit when supplied. '
                'A null remaining_seconds means no time deadline; use measured duration and progress. '
                'Respect remaining budgets. Explain decisions and put future model changes in recommendations.',
                dict(checkpoint=checkpoint.model_dump(mode='json'), skills=skills), TrainingDecision, None)
        return validate_decision(answer, checkpoint)
    except (GlobalAPIError, WorkerInfrastructureError):
        raise
    except Exception as error:
        emit('training.fallback', {'checkpoint_id': checkpoint.checkpoint_id, 'reason': str(error)[:300]})
        return fallback(checkpoint, 'Programmatic policy: '+str(error)[:300])


def worker_checkpoint(root, checkpoint, *, seconds=30):
    """Atomic mailbox with a bounded wait, charged to the existing worker deadline."""
    folder = Path(root)/'control'
    request = folder/(checkpoint.checkpoint_id+'.request.json')
    response = folder/(checkpoint.checkpoint_id+'.decision.json')
    commit_mailbox(request, checkpoint)
    allowance = min(seconds, max(0., checkpoint.remaining_seconds)) if seconds is not None and checkpoint.remaining_seconds is not None else seconds
    until = time.monotonic()+allowance if allowance is not None else float('inf')
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
