"""Versioned model behavior, executed by the SDK rather than generated code."""
from __future__ import annotations

import copy
from typing import Any, Literal

from pydantic import Field
from .contracts import StrictModel

CONTRACT_VERSION = 2


class TargetTransform(StrictModel):
    name: Literal['identity', 'log1p'] = 'identity'
    clip_min: float | None = None
    nonfinite: Literal['error', 'train_median'] = 'error'


class WeightPolicy(StrictModel):
    name: Literal['none', 'balanced', 'sqrt_balanced', 'tail_aware'] = 'none'
    quantile: float = Field(.99, gt=0, le=1)
    minimum: float = Field(1., ge=0)


class PredictionPolicy(StrictModel):
    clip_min: float | None = None
    nonfinite: Literal['error', 'train_median'] = 'error'


class ModelSpec(StrictModel):
    version: Literal[2] = 2
    tool_id: str
    backend_params: dict[str, Any] = Field(default_factory=dict)
    target_transform: TargetTransform = Field(default_factory=TargetTransform)
    weight_policy: WeightPolicy = Field(default_factory=WeightPolicy)
    prediction_policy: PredictionPolicy = Field(default_factory=PredictionPolicy)
    code_hash: str | None = None
    preprocessing_hash: str | None = None
    feature_set_hash: str | None = None


class Intervention(StrictModel):
    kind: Literal['baseline', 'model_change', 'ablation', 'hpo', 'single_configuration'] = 'model_change'
    parent_experiment_id: str | None = None
    base_candidate_id: str | None = None
    changed_fields: list[str] = Field(default_factory=list)
    operations: list[str] = Field(default_factory=list)
    preserved_conditions: list[str] = Field(default_factory=list)
    success_conditions: list[str] = Field(default_factory=list)
    rejection_conditions: list[str] = Field(default_factory=list)


POLICIES = ('target_transform', 'weight_policy', 'prediction_policy')


def model_spec(config, params=None):
    from .feature_contracts import feature_hash
    return ModelSpec(tool_id=config.get('tool_id', 'unknown'),
        backend_params=copy.deepcopy(config.get('model_params', {}) if params is None else params),
        code_hash=config.get('_code_hash'), preprocessing_hash=config.get('_preprocessing_hash'),
        feature_set_hash=feature_hash(config.get('feature_set')),
        **{key: config.get(key, {}) for key in POLICIES}).model_dump(mode='json')


def with_spec(config, spec):
    from .feature_contracts import feature_hash
    spec = ModelSpec.model_validate(spec).model_dump(mode='json')
    if spec['feature_set_hash'] != feature_hash(config.get('feature_set')):
        raise ValueError('Frozen ModelSpec differs from the feature definition')
    return dict(config, model_params=spec['backend_params'], **{key: spec[key] for key in POLICIES})


def frozen_plan(plan, run):
    from .pipeline_contracts import HypothesisPlan
    config=plan.model_dump(mode='json')
    config['model_params']=run.best_params or plan.model_params
    config['strategy']='single'
    config['search_space']={}
    if run.model_spec:
        config=with_spec(config,run.model_spec)
    return HypothesisPlan.model_validate(config)


def apply_overrides(config, overrides):
    result = copy.deepcopy(config)
    for key, value in overrides.items():
        if key.startswith('policy_params.'):
            path = key.split('.')
            if len(path) != 3 or path[1] not in POLICIES:
                raise ValueError(f'Unknown policy parameter route: {key}')
            result.setdefault(path[1], {})[path[2]] = value
        else:
            result.setdefault('model_params', {})[key] = value
    # Validate routes and values before training.
    model_spec(result)
    return result


def normalize_protocol(declared, expected):
    """Only orchestration-owned fields determine splits; annotations stay separate."""
    metadata = {key: value for key, value in declared.items() if key not in expected}
    for key, value in expected.items():
        if key in declared and declared[key] != value:
            raise ValueError(f'validation_protocol.{key}: expected {value!r}, received {declared[key]!r}')
    return dict(expected), metadata


def constant_space(space):
    if not space:
        return None
    values = {}
    for key, item in space.items():
        if item.get('type') == 'categorical':
            choices = item.get('choices', [])
            if len(choices) != 1:
                return None
            values[key] = choices[0]
        elif item.get('low') == item.get('high') and 'low' in item:
            if item.get('type')=='int' and type(item['low']) is not int:
                raise ValueError(f'search_space.{key}: integer bounds must be integers')
            if item.get('type') not in {'int','float'}:
                raise ValueError(f'search_space.{key}: unknown numeric distribution type')
            values[key] = item['low']
        else:
            return None
    return values


def hpo_child(plan, parent_experiment_id=None):
    """A search is an explicitly different intervention; retain the parent hypothesis."""
    import hashlib
    item = plan.model_copy(deep=True)
    original = item.hypothesis.description
    suffix = '-hpo'
    stem = item.hypothesis.hypothesis_id[:55]
    identity = hashlib.sha256((item.hypothesis.hypothesis_id + str(parent_experiment_id)).encode()).hexdigest()[:4]
    item.hypothesis.hypothesis_id = stem + suffix + '-' + identity
    item.hypothesis.baseline = False
    item.hypothesis.based_on_hypothesis_id = plan.hypothesis.hypothesis_id
    item.hypothesis.description = 'Tune backend/policy parameters of: ' + original
    item.intervention = Intervention(kind='hpo', parent_experiment_id=parent_experiment_id,
        base_candidate_id=plan.hypothesis.based_on, changed_fields=list(item.search_space),
        preserved_conditions=['candidate code', 'preprocessing', 'validation protocol', 'metric'],
        success_conditions=['Improve the declared primary metric against the paired parent/control'],
        rejection_conditions=['No improvement on the comparable outer split'])
    return item


class PolicyState:
    """Fold-local policies. No fitted statistics are read from another fold."""
    def __init__(self, config):
        self.spec = model_spec(config)
        self.median = None

    def fit_transform(self, y, weights, task_type):
        import numpy as np
        import pandas as pd
        if y is None:
            if self.spec['target_transform']['name'] != 'identity' or self.spec['weight_policy']['name'] != 'none':
                raise ValueError('Target/weight policies require training labels')
            return y, weights
        target = self.spec['target_transform']
        policy = self.spec['weight_policy']
        regression = 'regression' in task_type
        if not regression and (target['name'] != 'identity' or target['clip_min'] is not None or target['nonfinite'] != 'error'):
            raise ValueError('Target transforms are supported for regression only')
        values = np.asarray(y)
        self.median = np.nanmedian(np.where(np.isfinite(values.astype(float)), values, np.nan), axis=0) if regression else None
        transformed = self.transform_target(y)
        custom = None
        if policy['name'] in {'balanced', 'sqrt_balanced'}:
            if task_type not in {'binary_classification', 'multiclass_classification'}:
                raise ValueError('Class balancing requires single-target classification')
            labels = pd.Series(values)
            counts = labels.value_counts()
            custom = labels.map(len(labels) / (len(counts) * counts)).to_numpy(dtype=float)
            if policy['name'] == 'sqrt_balanced':
                custom = np.sqrt(custom)
        elif policy['name'] == 'tail_aware':
            if not regression or values.ndim != 1:
                raise ValueError('tail_aware weights require single-target regression')
            numeric = np.asarray(y, dtype=float)
            finite = numeric[np.isfinite(numeric)]
            q = np.quantile(finite, policy['quantile']) if len(finite) else 0.
            custom = np.maximum(policy['minimum'], 1 + numeric/q) if q > 0 else np.ones(len(values))
            custom = np.where(np.isfinite(custom), custom, 1.)
        if custom is not None:
            weights = custom if weights is None else custom * np.asarray(weights)
        return transformed, weights

    def transform_target(self, y):
        import numpy as np
        import pandas as pd
        if y is None:
            return None
        target = self.spec['target_transform']
        if target == TargetTransform().model_dump():
            return y
        values = np.asarray(y, dtype=float).copy()
        if target['nonfinite'] == 'train_median':
            values = np.where(np.isfinite(values), values, self.median)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite target under declared target policy')
        if target['clip_min'] is not None:
            values = np.maximum(values, target['clip_min'])
        if target['name'] == 'log1p':
            if np.any(values <= -1):
                raise ValueError('log1p requires targets > -1; declare clip_min explicitly')
            values = np.log1p(values)
        if isinstance(y, pd.Series):
            return pd.Series(values, index=y.index, name=y.name)
        if isinstance(y, pd.DataFrame):
            return pd.DataFrame(values, index=y.index, columns=y.columns)
        return values

    def inverse(self, predictions, kind):
        import numpy as np
        if kind != 'value':
            return predictions
        values = np.asarray(predictions)
        if self.spec['target_transform']['name'] == 'log1p':
            values = np.expm1(values)
        policy = self.spec['prediction_policy']
        if policy['nonfinite'] == 'train_median':
            values = np.where(np.isfinite(values), values, self.median)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite predictions under declared prediction policy')
        if policy['clip_min'] is not None:
            values = np.maximum(values, policy['clip_min'])
        return values
