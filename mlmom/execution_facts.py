"""Observed fit metadata and a common, non-narrative view for every role."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import Field
from .contracts import StrictModel


RUNTIME_CONTRACT_VERSION = 7


class FitExecutionFacts(StrictModel):
    version: Literal[1] = 1
    fit_id: str
    recorded_at: float = Field(default_factory=time.time)
    stage: str
    split_id: str | None = None
    trial_number: int | None = None
    fold_number: int | None = None
    component: str | None = None
    parent_fit_id: str | None = None
    fit_rows: int | None = None
    status: Literal['observed', 'unknown'] = 'unknown'
    estimator: str | None = None
    parameters: dict[str, Any] | None = None
    parameter_source: str = 'unknown'
    base_columns: int | None = None
    derived_columns: int | None = None
    total_columns: int | None = None
    feature_programs: list[dict[str, Any]] | None = None
    feature_names: list[str] | None = None
    feature_set_hash: str | None = None
    feature_execution_owner: str | None = None
    code_hash: str | None = None
    preprocessing_hash: str | None = None
    reason: str | None = None


def read_fit_facts(root):
    """One atomic file per fit also survives errors after fitting and worker restarts."""
    values = []
    for path in sorted((Path(root) / 'fit-execution').glob('*.json')):
        try:
            value=json.loads(path.read_text(encoding='utf-8'))
            values.append(FitExecutionFacts.model_validate(value).model_dump(mode='json'))
        except (OSError, ValueError):
            continue
    return sorted(values, key=lambda value: value.get('recorded_at', 0))


def record_fit(model, config, context):
    """Best-effort observation, never a new estimator audit or execution gate."""
    from .persistence import atomic_json
    facts = FitExecutionFacts(fit_id=context.fit_id,
        stage='submission_refit' if context.stage=='final_full_data_refit' else context.stage,
        split_id=context.outer_split_id, trial_number=context.trial_number,
        fold_number=context.fold_number, component=context.component_name,
        parent_fit_id=context.parent_fit_id, fit_rows=context.fit_rows,
        code_hash=config.get('_code_hash'), preprocessing_hash=config.get('_preprocessing_hash'))
    try:actual_config = getattr(model, 'config', config)
    except Exception:actual_config = config
    if not isinstance(actual_config,dict):actual_config=config
    facts.preprocessing_hash = actual_config.get('_preprocessing_hash', facts.preprocessing_hash)
    try:
        facts.parameters = copy.deepcopy(actual_config.get('model_params', {}))
        facts.parameter_source = 'executed_config'
        export = getattr(model, 'export_model_params', None)
        if callable(export):
            values = export()
            if isinstance(values, dict):
                json.dumps(values, allow_nan=False)
                facts.parameters, facts.parameter_source = values, 'export_model_params'
    except Exception:
        pass
    try:
        estimator = getattr(model, 'model', None)
        if estimator is not None:
            facts.estimator = type(estimator).__name__
    except Exception:
        pass
    try:
        manifest = getattr(model, 'feature_manifest', None)
        value = manifest() if callable(manifest) else None
        if value is not None:
            facts.status = 'observed'
            facts.base_columns = value.get('base_width')
            facts.total_columns = value.get('output_width')
            if facts.base_columns is not None and facts.total_columns is not None:
                facts.derived_columns = facts.total_columns - facts.base_columns
            facts.feature_programs = value.get('programs')
            facts.feature_names = value.get('names')
            facts.feature_set_hash = value.get('feature_set_hash')
            facts.feature_execution_owner = 'SDK'
        else:
            facts.reason = 'Feature manifest unavailable; feature execution is unknown'
    except Exception as error:
        facts.reason = 'Feature manifest unavailable: ' + str(error)[:300]
    try:
        # Generated/native exports may contain values unsuitable for metadata JSON.
        value = json.loads(facts.model_dump_json())
        if context.execution_facts_dir:
            atomic_json(Path(context.execution_facts_dir) / (facts.fit_id + '.json'), value)
        return value
    except Exception:
        return None


def configuration_diff(before, after):
    """Compute changes from configurations, not from an agent's description."""
    fields = ('tool_id', 'model_params', 'preprocessing', 'target_transform', 'weight_policy',
              'prediction_policy', 'stopping_policy', 'ensemble_sources')
    changes = {key: dict(before=before.get(key), after=after.get(key)) for key in fields
               if before.get(key) != after.get(key)}
    def entries(config):
        definition = config.get('feature_set') or {}
        return {item['name']: item for key in ('features', 'programs') for item in definition.get(key, [])}
    old, new = entries(before), entries(after)
    changes['features'] = dict(added=sorted(new.keys()-old.keys()), removed=sorted(old.keys()-new.keys()),
        replaced=sorted(key for key in old.keys() & new.keys() if old[key] != new[key]))
    return changes


def observed_execution(facts, experiment_id=None, related_fits=None):
    latest={}
    for fact in facts:
        latest[fact['stage'],fact.get('component')]=fact
    return dict(status='observed' if facts else 'unknown',fit_count=len(facts),fits=list(latest.values()),
        evidence_ref=('execution:'+experiment_id) if facts and experiment_id else None,related_fits=related_fits or [])


def execution_context(plan, run=None, evaluation=None, *, candidate=None, related_fits=None):
    """A lossless current configuration, scoped measurements and explicitly historical prose."""
    from .execution_contract import execution_contract
    from .model_spec import frozen_plan
    from .pipeline_contracts import HypothesisPlan, ExperimentRun
    raw_run = run.model_dump(mode='json') if hasattr(run, 'model_dump') else (run or {})
    try:
        plan = plan if isinstance(plan, HypothesisPlan) else HypothesisPlan.model_validate(plan)
        effective = frozen_plan(plan, ExperimentRun.model_validate(raw_run)) if raw_run.get('experiment_id') else plan
        current = effective.model_dump(mode='json')
        contract = execution_contract(effective)
        difference = raw_run.get('configuration_diff') or plan.configuration_diff
    except ValueError:
        # Legacy/minimal training checkpoints remain valid control inputs.
        current = copy.deepcopy(plan.model_dump(mode='json') if hasattr(plan,'model_dump') else plan)
        contract, difference = None, raw_run.get('configuration_diff', {})
    if raw_run.get('best_params') and not raw_run.get('experiment_id'):
        current['model_params'] = raw_run['best_params']
    history = current.pop('hypothesis', None)
    # Narrative annotations never define the model that was executed.
    current.pop('agent_notes', None)
    facts = raw_run.get('execution_facts', [])
    return dict(current_configuration=current, execution_contract=contract,
        observed_execution=observed_execution(facts,raw_run.get('experiment_id'),related_fits),
        measurements=dict(internal={key:raw_run.get(key) for key in
            ('internal_score', 'score_kind', 'split_id', 'validation_protocol_ref', 'fold_results', 'score_std', 'oof_score', 'oof_score_kind')},
            evaluation=evaluation.model_dump(mode='json') if hasattr(evaluation, 'model_dump') else evaluation),
        configuration_diff=difference,
        source_identity={key:(candidate or {}).get(key) for key in ('code_hash', 'preprocessing_hash')},
        historical_baseline=dict(role='historical description; not the current model composition', hypothesis=history))


def related_fit_observations(store, experiment_id):
    values=[]
    if store is not None:
        for run in store.list('submission_refit'):
            source=store.get('submission_refit_source',run['experiment_id']) or {}
            if source.get('champion_experiment_id') == experiment_id:
                values.extend(run.get('execution_facts', []))
    return values
