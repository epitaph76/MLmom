"""Shared transport recovery and role-specific thinking without token caps."""
from __future__ import annotations

import threading
import time


class NetworkCircuit:
    def __init__(self, threshold=3, cooldown=30., *, clock=time.monotonic):
        self.threshold, self.cooldown, self.clock = threshold, cooldown, clock
        self.failures = 0
        self.open_until = None
        self.probing = False
        self.generation = 0
        self.condition = threading.Condition()

    def acquire(self, remaining=None):
        until = None if remaining is None else self.clock()+remaining
        with self.condition:
            while True:
                now = self.clock()
                if until is not None and now >= until:
                    raise TimeoutError('LLM deadline reached while waiting for shared network recovery')
                if self.open_until is None:
                    return (False, self.generation)
                if now >= self.open_until and not self.probing:
                    self.probing = True
                    return (True, self.generation)
                wait = min(.25, max(.001, self.open_until-now)) if not self.probing else .25
                if until is not None:
                    wait = min(wait, max(.001, until-now))
                self.condition.wait(wait)

    def success(self, token):
        probe, generation = token
        with self.condition:
            if generation != self.generation:
                return
            if self.open_until is None or probe:
                self.failures = 0
                self.open_until = None
                self.probing = False
                self.condition.notify_all()

    def failure(self, token):
        probe, generation = token
        with self.condition:
            if generation != self.generation:
                return
            self.failures += 1
            if probe or self.failures >= self.threshold:
                self.generation += 1
                self.open_until = self.clock()+self.cooldown
                self.probing = False
                self.condition.notify_all()


def thinking_parameters(model, mode):
    if mode != 'off':
        return {}
    name = model.lower()
    if 'qwen' in name:
        return {'enable_thinking': False}
    if 'deepseek' in name:
        return {'thinking': {'type': 'disabled'}}
    return {}
