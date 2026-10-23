"""Candidate SDK. No external holdout labels or LLM credentials enter this API."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol
import importlib.util
import sys
import time
import uuid
import hashlib
import ast
import copy


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
    timeouts_enabled: bool = True
    fit_rows: int = 0
    optimizer_steps: int = 0
    optimizer_steps_per_epoch: int | None = None
    optimizer_params: dict[str, Any] = field(default_factory=dict)
    optimizer_traces: dict[tuple[int, ...], dict[str, Any]] = field(default_factory=dict)

    def track_optimizer(self, optimizer, batch_size: int):
        """Count actual backend optimizer steps; epoch count is derived, never self-reported."""
        import math
        if self.fit_rows < 1 or batch_size < 1: raise ValueError('Optimizer tracking needs fit_rows and a positive batch_size')
        self.optimizer_steps = 0
        self.optimizer_steps_per_epoch = math.ceil(self.fit_rows/batch_size)
        identity=tuple(sorted(id(parameter) for group in optimizer.param_groups for parameter in group['params']))
        trace=dict(steps=0,steps_per_epoch=self.optimizer_steps_per_epoch,
                   params=dict(lr=float(optimizer.param_groups[0]['lr']),batch_size=batch_size))
        self.optimizer_traces[identity]=trace
        def completed(value, args, kwargs):
            self.optimizer_steps += 1
            self.optimizer_params = dict(lr=float(value.param_groups[0]['lr']),batch_size=batch_size)
            trace.update(steps=trace['steps']+1,params=self.optimizer_params)
        return optimizer.register_step_post_hook(completed)

    @property
    def remaining_seconds(self):
        if not self.timeouts_enabled:
            return float('inf')
        return max(0., self.time_limit_seconds - (time.monotonic()-self.started))


class Candidate(Protocol):
    def fit(self, X, y, context: FitContext): ...
    def predict(self, X, prediction_spec: dict): ...
    def save(self, artifact_dir: str): ...
    def export_model_params(self) -> dict: ...
    def fitted_components(self) -> dict[str, Any]: ...


class Preprocessor(Protocol):
    def fit_transform(self, X, y, context: FitContext): ...
    def transform(self, X): ...


class PreparedCandidate:
    """Compose separately authored code; every fit creates fresh transform state."""
    def __init__(self, config, create_model, create_preprocessor):
        self.config = copy.deepcopy(config)
        self.create_model = create_model
        self.create_preprocessor = create_preprocessor

    def fit(self, X, y, context):
        context.fit_rows = len(X)
        if self.config.get('ensemble_components') and self.config.get('model_params', {}).get('method') == 'stack':
            self.estimator = self.create_model(copy.deepcopy(self.config))
            fit_stack = getattr(self.estimator, 'fit_stack', None)
            if not callable(fit_stack):
                raise ValueError('Stacking requires fit_stack(X,y,context,create_preprocessor) for fold-local preprocessing')
            self.preprocessor = None
            fit_stack(X, y, context, self.create_preprocessor)
            return self
        try:
            self.preprocessor = self.create_preprocessor(copy.deepcopy(self.config))
            train_context = replace(context, validation_X=None, validation_y=None,
                                    validation_groups=None, validation_weight=None)
            values = self.preprocessor.fit_transform(X, y, train_context)
            if values.shape[0] != len(X):
                raise ValueError('Preprocessing must preserve training rows')
            valid = self.preprocessor.transform(context.validation_X) if context.validation_X is not None else None
            if valid is not None and valid.shape[0] != len(context.validation_X):
                raise ValueError('Preprocessing must preserve validation rows')
        except Exception as error:
            raise ValueError(f'preprocessing.py: {type(error).__name__}: {error}') from error
        self.estimator = self.create_model(copy.deepcopy(self.config))
        self.estimator.fit(values, y, replace(context, validation_X=valid))
        return self

    def predict(self, X, prediction_spec):
        if self.preprocessor is None:
            return self.estimator.predict(X, prediction_spec)
        try:
            values = self.preprocessor.transform(X)
            if values.shape[0] != len(X): raise ValueError('Preprocessing must preserve inference rows')
        except Exception as error:
            raise ValueError(f'preprocessing.py: {type(error).__name__}: {error}') from error
        return self.estimator.predict(values, prediction_spec)

    @property
    def model(self):
        return getattr(self.estimator, 'model', None)

    def export_model_params(self):
        export = getattr(self.estimator, 'export_model_params', None)
        return export() if export else dict(self.config.get('model_params', {}))

    def fitted_components(self):
        from .execution import fitted_components
        return fitted_components(self.estimator)

    def native_history(self):
        export = getattr(self.estimator, 'native_history', None)
        return export() if export else None

    def save(self, artifact_dir):
        import cloudpickle
        root = Path(artifact_dir)
        model_dir = root/'estimator'
        model_dir.mkdir(parents=True, exist_ok=True)
        self.estimator.save(str(model_dir))
        (root/'preprocessing.pkl').write_bytes(cloudpickle.dumps(self.preprocessor))
        (root/'prepared-config.pkl').write_bytes(cloudpickle.dumps(self.config))


def _execute_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ValueError(f'Missing source: {path.name}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    import cloudpickle
    cloudpickle.register_pickle_by_value(module)
    return module


def load_module(code_dir: str | Path):
    root = Path(code_dir).resolve()
    path = root / 'candidate.py'
    # Stable names let ordinary pickle/joblib restore custom candidate classes.
    name = 'mlmom_candidate_' + hashlib.sha256(str(root).encode()+path.read_bytes()).hexdigest()[:24]
    # Temporarily isolate sibling imports such as helpers.py between candidates.
    local_names = {p.stem for p in root.glob('*.py')}
    previous = {key: sys.modules.pop(key) for key in local_names if key in sys.modules}
    sys.path.insert(0, str(root))
    try:
        module = _execute_module(path, name)
        preprocessing = root/'preprocessing.py'
        separate = False
        if preprocessing.is_file():
            tree=ast.parse(preprocessing.read_text(encoding='utf-8'))
            separate=any(isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MLMOM_PREPROCESSOR_V1' for t in node.targets)
                         and isinstance(node.value,ast.Constant) and node.value.value is True for node in tree.body)
        if separate:
            pre_name = name+'_pre_'+hashlib.sha256(preprocessing.read_bytes()).hexdigest()[:12]
            pre_module = _execute_module(preprocessing, pre_name)
        import cloudpickle
        cloudpickle.register_pickle_by_value(module)
        for imported in list(sys.modules.values()):
            location=getattr(imported,'__file__',None)
            if location and Path(location).resolve().is_relative_to(root):
                cloudpickle.register_pickle_by_value(imported)
    finally:
        sys.path.remove(str(root))
        for key in local_names: sys.modules.pop(key, None)
        sys.modules.update(previous)
    if not callable(getattr(module, 'create_candidate', None)) or not callable(getattr(module, 'load_candidate', None)):
        raise ValueError('Candidate must export create_candidate(config), load_candidate(artifact_dir)')
    if separate:
        create_model, load_model = module.create_candidate, module.load_candidate
        factory = getattr(pre_module, 'create_preprocessor', None)
        if not callable(factory): raise ValueError('preprocessing.py must export create_preprocessor(config)')
        def create(config):
            return PreparedCandidate(config, create_model, factory)
        def load(artifact_dir):
            import cloudpickle
            directory = Path(artifact_dir)
            obj = create(cloudpickle.loads((directory/'prepared-config.pkl').read_bytes()))
            obj.preprocessor = cloudpickle.loads((directory/'preprocessing.pkl').read_bytes())
            obj.estimator = load_model(str(directory/'estimator'))
            return obj
        module.create_candidate, module.load_candidate = create, load
    return module
