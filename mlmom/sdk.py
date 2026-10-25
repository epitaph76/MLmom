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
from .composite import FrozenEnsemble


@dataclass
class FitContext:
    task_type: str
    metric: str
    direction: str
    seed: int
    cpus: int
    memory_gb: float
    time_limit_seconds: float
    stage: str = 'fit'
    fit_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    runtime_policies_applied: bool = False
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
    execution_facts_dir: str | None = None
    outer_split_id: str | None = None
    trial_number: int | None = None
    fold_number: int | None = None
    component_name: str | None = None
    parent_fit_id: str | None = None
    @property
    def remaining_seconds(self):
        if not self.timeouts_enabled:
            return float('inf')
        return max(0., self.time_limit_seconds - (time.monotonic()-self.started))


class Candidate(Protocol):
    def fit(self, X, y, context: FitContext): ...
    def predict(self, X, prediction_spec: dict): ...
    def save(self, artifact_dir: str): ...


class Preprocessor(Protocol):
    def fit_transform(self, X, y, context: FitContext): ...
    def transform(self, X): ...


class PolicyCandidate:
    """Own target/weight/output behavior independently of generated estimator code."""
    def __init__(self, estimator, config):
        from .model_spec import PolicyState
        self.estimator = estimator
        self.config = copy.deepcopy(config)
        self.policy = PolicyState(config)

    def __getattr__(self, name):
        estimator = self.__dict__.get('estimator')
        if estimator is None:
            raise AttributeError(name)
        return getattr(estimator, name)

    def fit(self, X, y, context):
        if context.task_type=='ranking' and context.groups is not None:
            import numpy as np
            order=np.argsort(context.groups,kind='stable')
            X=X.iloc[order] if hasattr(X,'iloc') else X[order]
            y=y.iloc[order] if hasattr(y,'iloc') else np.asarray(y)[order]
            context=replace(context,groups=np.asarray(context.groups)[order],
                sample_weight=np.asarray(context.sample_weight)[order] if context.sample_weight is not None else None)
        self.apply_policy=not context.runtime_policies_applied
        transformed,weights=self.policy.fit_transform(y,context.sample_weight,context.task_type) if self.apply_policy else (y,context.sample_weight)
        valid_y=self.policy.transform_target(context.validation_y) if self.apply_policy else context.validation_y
        context = replace(context, sample_weight=weights, validation_y=valid_y,runtime_policies_applied=True)
        self.estimator.fit(X, transformed, context)
        return self

    def predict(self, X, prediction_spec):
        result=self.estimator.predict(X,prediction_spec)
        return self.policy.inverse(result,prediction_spec.get('kind','value')) if getattr(self,'apply_policy',True) else result

    def export_model_params(self):
        method = getattr(self.estimator, 'export_model_params', None)
        return method() if method else copy.deepcopy(self.config.get('model_params', {}))

    def export_model_spec(self):
        from .model_spec import model_spec
        return model_spec(self.config, self.export_model_params())

    def save(self, artifact_dir):
        import cloudpickle
        self.estimator.save(artifact_dir)
        (Path(artifact_dir)/'runtime-policy.pkl').write_bytes(cloudpickle.dumps((self.config, self.policy,getattr(self,'apply_policy',True))))


def _load_policy(estimator, directory):
    path = Path(directory)/'runtime-policy.pkl'
    if not path.is_file():
        return estimator
    import cloudpickle
    saved=cloudpickle.loads(path.read_bytes())
    config,policy=saved[:2]
    result = PolicyCandidate(estimator, config)
    result.policy = policy
    result.apply_policy=saved[2] if len(saved)>2 else True
    return result


class PreparedCandidate:
    """Compose separately authored code; every fit creates fresh transform state."""
    def __init__(self, config, create_model, create_preprocessor):
        self.config = copy.deepcopy(config)
        self.create_model = create_model
        self.create_preprocessor = create_preprocessor

    def fit(self, X, y, context):
        context.fit_rows = len(X)
        if self.config.get('ensemble_components') and self.config.get('model_params', {}).get('method') == 'stack':
            from .model_spec import PolicyState
            self.estimator = self.create_model(copy.deepcopy(self.config))
            fit_stack = getattr(self.estimator, 'fit_stack', None)
            if not callable(fit_stack):
                raise ValueError('Stacking requires fit_stack(X,y,context,create_preprocessor) for fold-local preprocessing')
            self.preprocessor = None
            self.stack_policy=PolicyState(self.config)
            self.stack_apply_policy=not context.runtime_policies_applied
            transformed,weights=self.stack_policy.fit_transform(y,context.sample_weight,context.task_type) if self.stack_apply_policy else (y,context.sample_weight)
            stack_context=replace(context,sample_weight=weights,runtime_policies_applied=True,
                validation_y=self.stack_policy.transform_target(context.validation_y) if self.stack_apply_policy else context.validation_y)
            fit_stack(X, transformed, stack_context, self.create_preprocessor)
            return self
        try:
            # Registered feature definitions belong to this wrapper, not base factories.
            factory_config={k:v for k,v in self.config.items() if k not in {'feature_set','feature_set_id'}}
            self.preprocessor = self.create_preprocessor(copy.deepcopy(factory_config))
            if self.config.get('feature_set'):
                from .feature_engineering import FeatureAugmentingPreprocessor
                self.preprocessor = FeatureAugmentingPreprocessor(self.preprocessor, self.config['feature_set'])
            train_context = replace(context, validation_X=None, validation_y=None,
                                    validation_groups=None, validation_weight=None)
            values = self.preprocessor.fit_transform(X, y, train_context)
            self.base_width = values.shape[1]
            if values.shape[0] != len(X):
                raise ValueError('Preprocessing must preserve training rows')
            valid = self.preprocessor.transform(context.validation_X) if context.validation_X is not None else None
            if valid is not None and valid.shape[0] != len(context.validation_X):
                raise ValueError('Preprocessing must preserve validation rows')
        except Exception as error:
            raise ValueError(f'preprocessing.py: {type(error).__name__}: {error}') from error
        self.estimator = self.create_model(copy.deepcopy(factory_config))
        if self.config.get('contract_version') == 2:
            self.estimator = PolicyCandidate(self.estimator, self.config)
        self.estimator.fit(values, y, replace(context, validation_X=valid))
        return self

    def feature_manifest(self):
        from .feature_engineering import FeatureAugmentingPreprocessor
        if isinstance(getattr(self, 'preprocessor', None), FeatureAugmentingPreprocessor):
            return self.preprocessor.manifest()
        if getattr(self, 'preprocessor', None) is not None and hasattr(self, 'base_width'):
            return dict(base_width=self.base_width, output_width=self.base_width, names=[], programs=[], feature_set_hash=None)
        return getattr(self, '_saved_feature_manifest', None)

    def predict(self, X, prediction_spec):
        if self.preprocessor is None:
            predictions=self.estimator.predict(X, prediction_spec)
            return self.stack_policy.inverse(predictions,prediction_spec.get('kind','value')) if getattr(self,'stack_apply_policy',False) else predictions
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

    def export_model_spec(self):
        from .model_spec import model_spec
        return model_spec(self.config, self.export_model_params())

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
        try:
            from .persistence import atomic_json
            atomic_json(root/'feature-manifest.json',self.feature_manifest())
        except Exception:
            pass  # Optional descriptive metadata must not prevent saving a trained model.
        if hasattr(self,'stack_policy'):
            (root/'stack-policy.pkl').write_bytes(cloudpickle.dumps((self.stack_policy,self.stack_apply_policy)))


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
            if config.get('ensemble_sources'):
                raise ValueError('Frozen-source ensembles cannot declare shared outer preprocessing.py')
            config = dict(config, _preprocessing_hash=hashlib.sha256(preprocessing.read_bytes()).hexdigest())
            return PreparedCandidate(config, create_model, factory)
        def load(artifact_dir):
            import cloudpickle
            directory = Path(artifact_dir)
            obj = create(cloudpickle.loads((directory/'prepared-config.pkl').read_bytes()))
            obj.preprocessor = cloudpickle.loads((directory/'preprocessing.pkl').read_bytes())
            obj.estimator = _load_policy(load_model(str(directory/'estimator')), directory/'estimator')
            if (directory/'feature-manifest.json').is_file():
                import json
                obj._saved_feature_manifest=json.loads((directory/'feature-manifest.json').read_text(encoding='utf-8'))
            if (directory/'stack-policy.pkl').is_file():
                obj.stack_policy,obj.stack_apply_policy=cloudpickle.loads((directory/'stack-policy.pkl').read_bytes())
            return obj
        module.create_candidate, module.load_candidate = create, load
    else:
        create_model, load_model = module.create_candidate, module.load_candidate
        def create(config):
            if config.get('feature_set'):
                raise ValueError('Engineered features require separate preprocessing.py')
            if config.get('ensemble_sources'):
                candidate = create_model(copy.deepcopy(config))
                return candidate
            candidate = create_model(config)
            return PolicyCandidate(candidate, config) if config.get('contract_version') == 2 else candidate
        def load(directory):
            return _load_policy(load_model(directory), directory)
        module.create_candidate, module.load_candidate = create, load
    return module
