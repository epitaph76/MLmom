"""Candidate SDK. No external holdout labels or LLM credentials enter this API."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
import importlib.util
import sys
import time
import uuid
import hashlib


@dataclass
class FitContext:
    task_type: str
    metric: str
    direction: str
    seed: int
    cpus: int
    memory_gb: float
    time_limit_seconds: float
    classes: list[Any] = field(default_factory=list)
    groups: Any = None
    sample_weight: Any = None
    validation_X: Any = None
    validation_y: Any = None
    validation_groups: Any = None
    validation_weight: Any = None
    work_dir: str = '.'
    task_context: Any = None
    started: float = field(default_factory=time.monotonic)

    @property
    def remaining_seconds(self):
        return max(0., self.time_limit_seconds - (time.monotonic()-self.started))


class Candidate(Protocol):
    def fit(self, X, y, context: FitContext): ...
    def predict(self, X, prediction_spec: dict): ...
    def save(self, artifact_dir: str): ...


def load_module(code_dir: str | Path):
    root = Path(code_dir).resolve()
    path = root / 'candidate.py'
    # Stable names let ordinary pickle/joblib restore custom candidate classes.
    name = 'mlmom_candidate_' + hashlib.sha256(str(root).encode()+path.read_bytes()).hexdigest()[:24]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError('Missing candidate.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # custom estimator classes can be pickled during this process
    sys.path.insert(0, str(root))
    try:
        spec.loader.exec_module(module)
        import cloudpickle
        cloudpickle.register_pickle_by_value(module)
        for imported in list(sys.modules.values()):
            location=getattr(imported,'__file__',None)
            if location and Path(location).resolve().is_relative_to(root):
                cloudpickle.register_pickle_by_value(imported)
    finally:
        sys.path.remove(str(root))
    if not callable(getattr(module, 'create_candidate', None)) or not callable(getattr(module, 'load_candidate', None)):
        raise ValueError('Candidate must export create_candidate(config), load_candidate(artifact_dir)')
    return module
