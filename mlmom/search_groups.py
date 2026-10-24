"""Logical search groups, independent of worker admission and outer evaluation."""
from __future__ import annotations

import hashlib
from .pipeline_contracts import Hypothesis, HypothesisPlan, SearchGroupPlan

SUPERVISED = {'regression', 'multioutput_regression', 'binary_classification',
              'multiclass_classification', 'multilabel_classification'}
BOOSTERS = ('lightgbm', 'catboost', 'xgboost')


def compatible_boosters(tools, task_type):
    return [key for key in BOOSTERS if (tool := tools.get(key)) and
            task_type in tool.task_types and not tool.native_hpo and
            (tool.available or tool.reason == 'adapter probe required')]


def initial_boosting_resources(budget, variants, max_workers):
    """Share capacity across initial fits; small machines retain the same queue."""
    slots = max(1, min(variants, max_workers, budget.available_cpus,
                       max(1, int(budget.available_memory_gb))))
    # Round down so simultaneous leases cannot exceed the memory ledger.
    memory = int(budget.available_memory_gb / slots * 1024) / 1024
    return dict(workers=slots, cpus=max(1, budget.available_cpus // slots),
                memory_gb=max(1 / 1024, memory))


def variant_id(parent, variant):
    digest = hashlib.sha256((parent + ':' + variant).encode()).hexdigest()[:6]
    return f'{parent[:40]}-{variant[:15]}-{digest}'[:64]


def expand_families(hypotheses, tools, task_type, mode='all'):
    """Stable child identities let baseline/alternatives and resume share one group."""
    expanded, metadata, skipped = [], {}, []
    for parent in hypotheses:
        if parent.kind != 'family':
            if mode != 'alternatives' or not parent.baseline:
                expanded.append(parent)
            continue
        family = (parent.model_family or '').lower()
        requested = list(BOOSTERS) if family in {'boosting', 'gradient_boosting', 'gbdt', 'boosted_trees'} else parent.family_tools
        if not requested:
            raise ValueError('A model family requires model_family=gradient_boosting or explicit family_tools')
        compatible = []
        for key in dict.fromkeys(requested):
            tool = tools.get(key)
            if (not tool or task_type not in tool.task_types or tool.native_hpo or
                    (not tool.available and tool.reason != 'adapter probe required')):
                skipped.append(dict(group_id=parent.hypothesis_id, tool_id=key,
                                    reason=tool.reason if tool else 'unknown backend'))
            else:
                compatible.append(key)
        for index, key in enumerate(compatible[:3]):
            child = parent.model_copy(deep=True, update=dict(
                hypothesis_id=variant_id(parent.hypothesis_id, key), kind='single',
                baseline=parent.baseline and index == 0,
                based_on_hypothesis_id=parent.hypothesis_id))
            metadata[child.hypothesis_id] = (parent, key)
            if mode == 'baseline' and not child.baseline:
                continue
            if mode == 'alternatives' and child.baseline:
                continue
            expanded.append(child)
    return expanded, metadata, skipped


def normalize_single_group(plan):
    """A standalone hypothesis owns its group; historical lineage remains separate."""
    if plan.search_group_kind != 'single' or plan.search_group_id in {None, plan.hypothesis.hypothesis_id}:
        return plan
    notes = dict(plan.agent_notes)
    notes.setdefault('requested_search_group_id', plan.search_group_id)
    return plan.model_copy(update=dict(search_group_id=plan.hypothesis.hypothesis_id, agent_notes=notes))


def collect_groups(plans, skipped=()):
    groups = {}
    for value in plans:
        plan = value if isinstance(value, HypothesisPlan) else HypothesisPlan.model_validate(value)
        plan = normalize_single_group(plan)
        key = plan.search_group_id or plan.hypothesis.hypothesis_id
        if key not in groups:
            groups[key] = SearchGroupPlan(hypothesis=(plan.parent_hypothesis or plan.hypothesis)
                if plan.search_group_kind != 'single' else plan.hypothesis,
                kind=plan.search_group_kind, variants=[plan],
                skipped=[s for s in skipped if s.get('group_id') == key])
        else:
            groups[key].variants.append(plan)
    return [SearchGroupPlan.model_validate(g.model_dump()) for g in groups.values()]


def internal_winner(rows, direction):
    import math
    eligible = [r for r in rows if r['run'].get('role', 'evaluation') == 'evaluation' and r['run']['status'] == 'completed' and
                r['run'].get('internal_score') is not None and math.isfinite(r['run']['internal_score'])]
    if not eligible:
        return None
    def order(row):
        plan=row.get('plan',{})
        method=plan.get('model_params',{}).get('method','mean')
        return (row['run']['internal_score']*(1 if direction=='minimize' else -1),
                {'mean':0,'median':1,'weighted':2,'stack':3}.get(method,0),
                {'fast':0,'medium':1,'slow':2}.get(plan.get('cost_tag','fast'),0),
                plan.get('strategy')=='optuna',row['run'].get('experiment_id',''))
    return min(eligible,key=order)


def limit_parent_groups(plans, limit):
    groups=collect_groups(plans)
    groups.sort(key=lambda g:sum({'fast':1,'medium':2,'slow':3}[p.cost_tag] for p in g.variants))
    chosen={g.hypothesis.hypothesis_id for g in groups[:limit]}
    return [p for p in plans if (p.get('search_group_id') or p['hypothesis']['hypothesis_id']) in chosen]


def component_pool(store, round_number):
    """Only completed earlier cycles; deduplicate frozen configurations, not backend names."""
    completed = {r['round_id'] for r in store.list('cycle') if r.get('evaluation_completed', r.get('completed'))}
    records=store.list('component')
    # Existing contract-v2 runs may predate the dedicated component index.
    indexed={row['experiment_id'] for row in records}
    from .pipeline_contracts import CandidateVersion, ExperimentRun
    from .execution import require_admitted, config_hash
    from .registry import NATIVE_VALIDATION
    import json
    from pathlib import Path
    for run in store.list('experiment'):
        if (run.get('role', 'evaluation') != 'evaluation' or run['experiment_id'] in indexed or run.get('round_id') not in completed or
            run.get('status')!='completed' or run.get('internal_score') is None or run.get('incumbent')):
            continue
        candidate=store.get('candidate',run['candidate_id'])
        if not candidate:continue
        try:
            plan=store.get('training_plan',run['experiment_id']) or json.loads(Path(candidate['plan_ref']).read_text(encoding='utf-8'))
            if plan.get('ensemble_components') or plan.get('ensemble_sources') or plan['tool_id'] in NATIVE_VALIDATION:continue
            require_admitted(ExperimentRun.model_validate(run),CandidateVersion.model_validate(candidate))
            records.append(dict(experiment_id=run['experiment_id'],round_id=run['round_id'],run=run,
                candidate=candidate,plan=plan,evaluation=store.get('evaluation',run['experiment_id']),
                configuration_hash=config_hash(run.get('model_spec') or run['best_params'])))
        except (OSError,ValueError,KeyError):
            continue
    seen, pool = set(), []
    for row in reversed(records):
        if row.get('run', {}).get('role', 'evaluation') != 'evaluation':
            continue
        if row['round_id'] not in completed or int(row['round_id'][1:]) >= round_number:
            continue
        identity = (row['candidate']['code_hash'], row['configuration_hash'])
        if identity not in seen:
            pool.append(row)
            seen.add(identity)
    return pool


def component_brief(row):
    from .planning_context import evaluation_summary
    return dict(experiment_id=row['experiment_id'], candidate_id=row['candidate']['candidate_id'],
        tool_id=row['plan']['tool_id'], hypothesis=row['plan']['hypothesis']['description'],
        model_params=row['run']['best_params'], round_id=row['round_id'], split_id=row['run']['split_id'],
        internal_score=row['run']['internal_score'], score_kind=row['run'].get('score_kind'),
        validation_protocol_ref=row['run'].get('validation_protocol_ref'),
        configuration_hash=row['configuration_hash'], duration_seconds=row['run']['duration_seconds'],
        evaluation=evaluation_summary(row['evaluation']) if row.get('evaluation') else None)


def configure_composite(plan, sources, max_trials):
    """The planner cannot rewrite previously verified component configurations."""
    from .model_spec import TargetTransform, WeightPolicy, PredictionPolicy
    plan.ensemble_sources = sources
    plan.ensemble_components = [dict(name=s['name'], tool_id=s['config']['tool_id'],
        model_params=s['config']['model_params']) for s in sources]
    plan.tool_id = sources[0]['config']['tool_id']
    plan.model_params = dict(method='mean', oof_folds=3, weight_trials=max_trials,
                            **{s['name']:s['config']['model_params'] for s in sources})
    plan.target_transform = TargetTransform()
    plan.weight_policy = WeightPolicy()
    plan.prediction_policy = PredictionPolicy()
    plan.preprocessing = []
    plan.strategy = 'single'
    plan.search_space = {}
    plan.parameter_routes = {}
    plan.stopping_policy = 'fixed'
    plan.component_stopping_policies = {}
    plan.max_trials = 1
    plan.skill_ids = list(dict.fromkeys(['ensembles',
        *[k for s in sources for k in s['config']['skill_ids']]]))
    return plan


def ensemble_recipes(plan, task_type):
    parent = plan.hypothesis.model_copy(deep=True)
    methods = ['mean', 'median', 'weighted', 'stack'] if 'regression' in task_type else ['mean', 'weighted', 'stack']
    variants = []
    for method in methods:
        child = plan.model_copy(deep=True)
        child.hypothesis.hypothesis_id = variant_id(parent.hypothesis_id, method)
        child.hypothesis.based_on_hypothesis_id = parent.hypothesis_id
        child.hypothesis.baseline = False
        child.search_group_id = parent.hypothesis_id
        child.search_group_kind = 'ensemble'
        child.parent_hypothesis = parent
        child.model_params['method'] = method
        child.cost_tag = 'slow' if method in {'weighted', 'stack'} else 'medium'
        variants.append(child)
    return SearchGroupPlan(hypothesis=parent, kind='ensemble', variants=variants)
