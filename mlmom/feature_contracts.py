"""Versioned Python feature programs, optional DSL definitions and measured evidence."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import Field, model_validator
from .contracts import StrictModel
from .agent_contracts import AgentNarrative


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


class FeatureProgram(StrictModel):
    """Agent-authored, fold-fitted transformation executed inside the ML worker."""
    name: str = Field(pattern=r'^[A-Za-z][A-Za-z0-9_]{0,63}$')
    source: str = Field(min_length=1, description='Python module defining create_features(config); the object implements fit_transform(X,y,context) and transform(X).')
    config: dict[str, Any] = Field(default_factory=dict)

    def expression_key(self):
        return json.dumps(self.model_dump(mode='json'), sort_keys=True, separators=(',', ':'))


class FeatureSet(StrictModel):
    version: Literal[1] = 1
    feature_set_id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,64}$')
    features: list[FeatureSpec] = Field(default_factory=list)
    programs: list[FeatureProgram] = Field(default_factory=list)

    @model_validator(mode='after')
    def distinct(self):
        names = [f.name for f in [*self.features, *getattr(self, 'programs', [])]]
        if len(set(names)) != len(names):
            raise ValueError('Duplicate feature names')
        return self


def feature_hash(value):
    if not value:
        return None
    item = FeatureSet.model_validate(value.model_dump(mode='python') if isinstance(value, FeatureSet) else value)
    if not item.features and not item.programs:
        return None
    payload = [f.model_dump(mode='json') for f in item.features]
    if item.programs:
        payload = dict(features=payload, programs=[p.model_dump(mode='json') for p in item.programs])
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class FeatureProposal(AgentNarrative):
    action: Literal['add', 'remove', 'replace', 'stop']
    rationale: str = Field(min_length=1, max_length=2000)
    features: list[FeatureSpec] = Field(default_factory=list)
    programs: list[FeatureProgram] = Field(default_factory=list)
    remove: list[str] = Field(default_factory=list)

    @model_validator(mode='before')
    @classmethod
    def legacy_replacement(cls, value):
        # A former add+remove response has one unambiguous atomic meaning.
        if isinstance(value, dict) and value.get('action') == 'add' and (value.get('features') or value.get('programs')) and value.get('remove'):
            return dict(value, action='replace')
        return value

    @model_validator(mode='after')
    def one_intervention(self):
        additions = bool(self.features or self.programs)
        if self.action == 'add' and (not additions or self.remove):
            raise ValueError('Add one related group of features, without removals')
        if self.action == 'remove' and (not self.remove or additions):
            raise ValueError('Remove one related group, without additions')
        if self.action == 'replace' and (not self.remove or not additions):
            raise ValueError('Replace requires both feature names to remove and definitions to add')
        if self.action == 'stop' and (additions or self.remove):
            raise ValueError('Stop cannot change features')
        return self


def feature_configuration_keys(definition):
    return sorted(f.expression_key() for f in [*definition.features, *getattr(definition, 'programs', [])])


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
    proposal: FeatureProposal | None = None


class FeatureExperimentPlan(AgentNarrative):
    feature_set_id: str
    reference_feature_set_id: str
    rationale: str = Field(min_length=1, max_length=5000)


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
