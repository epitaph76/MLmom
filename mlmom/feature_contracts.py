"""Versioned, non-executable definitions and evidence for feature search."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import Field, model_validator
from .contracts import StrictModel


class FeatureSpec(StrictModel):
    name: str = Field(pattern=r'^[A-Za-z][A-Za-z0-9_]{0,63}$')
    op: Literal['missing', 'log', 'ratio', 'difference', 'product', 'date_part',
                'frequency', 'group_count', 'group_mean', 'group_std', 'group_min', 'group_max']
    columns: list[str] = Field(default_factory=list, max_length=2)
    by: str | None = None
    part: Literal['year', 'month', 'day', 'dayofweek', 'hour'] | None = None

    @model_validator(mode='after')
    def arguments(self):
        count = 0 if self.op == 'group_count' else 2 if self.op in {'ratio', 'difference', 'product'} else 1
        if len(self.columns) != count:
            raise ValueError(f'{self.op} requires {count} columns')
        if self.op.startswith('group_') != (self.by is not None):
            raise ValueError('Only group operations require by')
        if (self.op == 'date_part') != (self.part is not None):
            raise ValueError('Only date_part requires part')
        return self

    def expression_key(self):
        value = self.model_dump(exclude={'name'})
        if self.op == 'product':
            value['columns'] = sorted(value['columns'])
        return json.dumps(value, sort_keys=True, separators=(',', ':'))


class FeatureSet(StrictModel):
    version: Literal[1] = 1
    feature_set_id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,64}$')
    features: list[FeatureSpec] = Field(default_factory=list, max_length=16)

    @model_validator(mode='after')
    def distinct(self):
        if len({f.name for f in self.features}) != len(self.features):
            raise ValueError('Duplicate feature names')
        if len({f.expression_key() for f in self.features}) != len(self.features):
            raise ValueError('Duplicate feature expressions')
        return self


def feature_hash(value):
    if not value:
        return None
    item = FeatureSet.model_validate(value)
    if not item.features:
        return None
    payload = [f.model_dump(mode='json') for f in item.features]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class FeatureProposal(StrictModel):
    action: Literal['add', 'remove', 'stop']
    rationale: str = Field(min_length=1, max_length=2000)
    features: list[FeatureSpec] = Field(default_factory=list, max_length=8)
    remove: list[str] = Field(default_factory=list, max_length=16)

    @model_validator(mode='after')
    def one_intervention(self):
        if self.action == 'add' and (not self.features or self.remove):
            raise ValueError('Add one related group of features, without removals')
        if self.action == 'remove' and (not self.remove or self.features):
            raise ValueError('Remove one related group, without additions')
        if self.action == 'stop' and (self.features or self.remove):
            raise ValueError('Stop cannot change features')
        return self


class FeatureEvidence(StrictModel):
    session_id: str = 'initial'
    round_id: str = 'r1'
    feature_set_id: str
    parent_feature_set_id: str
    fingerprint: str
    status: Literal['accepted', 'rejected', 'quarantine', 'failed', 'screened_rejected']
    reason: str
    mean_delta: float | None = Field(None, allow_inf_nan=False)
    fold_deltas: list[float] = Field(default_factory=list)
    baseline_mean_delta: float | None = Field(None, allow_inf_nan=False)
    baseline_fold_deltas: list[float] = Field(default_factory=list)
    protocol_hash: str
    result_ref: str | None = None
    oof_ref: str | None = None
    duration_seconds: float = 0


class FeatureExperimentPlan(StrictModel):
    feature_set_id: str
    reference_feature_set_id: str
    rationale: str = Field(min_length=1, max_length=2000)


class FeatureSession(StrictModel):
    session_id: str = 'initial'
    round_id: str = 'r1'
    status: Literal['searching', 'selected', 'completed', 'skipped'] = 'searching'
    baseline_candidate_id: str
    model_hash: str
    train_hash: str
    protocol_hash: str
    protocol_ref: str
    max_configurations: int = 6
    plateau_limit: int = 3
    attempts: int = 0
    stagnant: int = 0
    proposal_calls: int = 0
    attempted_hashes: list[str] = Field(default_factory=list)
    baseline_feature_set_id: str = 'empty'
    best_feature_set_id: str = 'empty'
    best_result: dict[str, Any] | None = None
    pending: dict[str, Any] | None = None
    winner: dict[str, Any] | None = None
    reason: str | None = None
    outer_experiment_id: str | None = None
