"""Hypothesis planning subgraph and deterministic plan validation."""
from __future__ import annotations

from .agents import GlobalAPIError
from .resources import WorkerInfrastructureError

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from .pipeline_contracts import (HypothesisBatch, PlanBatch, HypothesisPlan, ValidationReport,
                                  BudgetSnapshot, TaskContext)
from .evaluation import prediction_kind
from .registry import LIGHTAUTOML_OBJECTIVES
from .agents import LLMUnavailable, StructuredOutputError, role_window
from .tracing import bind, emit
from .model_spec import normalize_protocol, constant_space, apply_overrides, Intervention
from functools import lru_cache


@lru_cache(maxsize=1)
def categorical_probe():
    """A small executable counterexample, using no user data or API tokens."""
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    train=pd.DataFrame({'category':pd.Categorical(['a','b']*16)})
    estimator=lgb.LGBMRegressor(n_estimators=3,min_child_samples=1,num_leaves=3,n_jobs=1,verbosity=-1)
    estimator.fit(train,np.arange(32,dtype=float))
    incoming=pd.DataFrame({'category':pd.Categorical(['a','unseen',None],categories=['a','b'])})
    values=estimator.predict(incoming)
    return dict(probe_id='lightgbm_native_categorical_unseen_nan',backend_version=lgb.__version__,
                status='passed' if np.isfinite(values).all() else 'failed',synthetic=True,
                known_unseen_missing_predictions_finite=bool(np.isfinite(values).all()))


def ground_semantic_report(report, plan, context):
    """LLM concerns need inspectable plan evidence; backend guesses cannot veto."""
    import re
    errors=[]
    warnings=list(report.warnings)
    data=plan.model_dump(mode='json')
    canonical={'prediction_kind':prediction_kind(context.task.primary_metric,context.task.task_type)}
    for issue in report.issues:
        path=issue.get('field','')
        observed=data
        try:
            for key in path.split('.'):
                observed=observed[int(key)] if isinstance(observed,list) else observed[key]
        except (KeyError,TypeError,ValueError,IndexError):
            warnings.append('Ungrounded semantic issue: '+str(issue))
            continue
        reason=issue.get('reason','Semantic constraint violation')
        if issue.get('rule_id')=='task_contract' and path in canonical and observed!=canonical[path]:
            errors.append(path+': '+reason)
        elif (issue.get('rule_id')=='holdout_training' and isinstance(observed,str)
              and issue.get('source_quote') and issue['source_quote'] in observed
              and re.search(r'fit|train',issue['source_quote'],re.I)
              and re.search(r'holdout\s+labels|external\s+holdout|test\s+labels',issue['source_quote'],re.I)):
            errors.append(path+': '+reason)
        else:
            warnings.append('Semantic concern requires a reproducible check: '+reason)
    if not errors:
        warnings.extend('Unverified semantic concern: '+message for message in report.errors)
        if not report.valid and not report.errors:
            warnings.append('Semantic rejection had no grounded field/evidence')
    return ValidationReport(valid=not errors,errors=errors,warnings=warnings,issues=report.issues)


def validate_plan(plan: HypothesisPlan, context: TaskContext, tools: dict, skills, budget: BudgetSnapshot,
                  *, lightweight=False, initial_experiment_seconds=300, evaluation_reserve_seconds=0) -> ValidationReport:
    errors = []
    if plan.feature_set_id:
        if not plan.feature_set or plan.feature_set.feature_set_id != plan.feature_set_id:
            errors.append('feature_set_id requires a registered feature definition')
        else:
            from .feature_engineering import validate_features
            try: validate_features(plan.feature_set,context.feature_columns)
            except ValueError as error: errors.append(str(error))
        if plan.ensemble_components or plan.ensemble_sources:
            errors.append('Shared engineered features are unsupported for ensembles; use frozen components')
    elif plan.feature_set:
        errors.append('feature_set requires an explicit feature_set_id')
    expected_protocol=dict(strategy=str(context.task.validation_strategy),folds=context.task.validation_folds,
                           duplicate_policy=getattr(context.task,'duplicate_policy','allow'))
    try:
        plan.validation_protocol,metadata=normalize_protocol(plan.validation_protocol,expected_protocol)
        plan.validation_metadata.update(metadata)
    except ValueError as error:
        errors.append(str(error))
    if plan.model_params.get('class_weight') == 'sqrt_balanced':
        errors.append('model_params.class_weight: sqrt_balanced is a runtime weight_policy, not a native backend parameter')
    if plan.weight_policy.name != 'none' and plan.model_params.get('class_weight') is not None:
        errors.append('weight_policy conflicts with native model_params.class_weight; declare one source of weights')
    if plan.target_transform.name!='identity' and 'regression' not in context.task.task_type:
        errors.append('target_transform requires a regression task')
    if plan.weight_policy.name in {'balanced','sqrt_balanced'} and context.task.task_type not in {'binary_classification','multiclass_classification'}:
        errors.append('weight_policy class balancing requires single-target classification')
    if plan.weight_policy.name=='tail_aware' and context.task.task_type!='regression':
        errors.append('weight_policy.tail_aware requires single-target regression')
    text=' '.join([plan.hypothesis.description,plan.hypothesis.rationale,*plan.preprocessing]).lower()
    if ('duplicate-aware' in text or 'duplicate aware' in text) and getattr(context.task,'duplicate_policy','allow') != 'group_features':
        errors.append('duplicate-aware validation requires confirmed duplicate_policy=group_features; narrative cannot change the protocol')
    if plan.stopping_policy=='early_stopping' and str(context.task.validation_strategy)!='holdout':
        errors.append('early_stopping requires the confirmed holdout strategy; CV has no stopping validation set')
    if any(v=='early_stopping' for v in plan.component_stopping_policies.values()) and str(context.task.validation_strategy)!='holdout':
        errors.append('Component early_stopping requires holdout')
    if plan.change_kind=='single_change_ablation' and (plan.strategy!='single' or plan.search_space):
        errors.append('Single-change ablation must freeze model parameters without additional HPO')
    tool = tools.get(plan.tool_id)
    if not tool or not tool.available: errors.append(f'Tool unavailable: {plan.tool_id}')
    else:
        if context.task.validation_strategy not in tool.validation_strategies:
            errors.append('Tool has no verified support for the confirmed internal validation strategy')
        if tool.native_hpo and context.task.group_columns:
            errors.append('Native group-aware validation has not been verified')
        if context.task.task_type not in tool.task_types: errors.append('Tool does not support the task type')
        if plan.weight_policy.name!='none' and not tool.supports_weights:errors.append('Backend has no verified sample-weight support')
        if context.task.group_columns and not tool.supports_groups and context.task.task_type == 'ranking': errors.append('Ranking requires a group-aware adapter')
        if context.task.weight_column and not tool.supports_weights: errors.append('Adapter does not support sample weights')
        if plan.strategy == 'native' and not tool.native_hpo: errors.append('Native HPO is not supported')
        if tool.native_hpo and plan.strategy != 'native': errors.append('Native AutoML requires strategy=native')
        if plan.strategy == 'native' and plan.patience_trials is not None and not tool.supports_patience:
            errors.append('Native adapter has no trial-patience callback; set patience_trials=null')
        if not set(tool.skill_ids) <= set(plan.skill_ids): errors.append('Required library skills are missing')
    if lightweight:
        if plan.strategy != 'single' or plan.max_trials != 1:
            errors.append('Lightweight plans require strategy=single and max_trials=1; search_space is reserved for later HPO')
        if tool and (tool.native_hpo or tool.cost_tag == 'slow'):
            errors.append('Lightweight plans cannot use native AutoML or slow backends')
        if getattr(budget,'timeouts_enabled',True) and plan.time_limit_seconds > initial_experiment_seconds:
            errors.append('Lightweight experiment exceeds the initial experiment limit')
    if plan.cpus > budget.available_cpus or plan.memory_gb > budget.available_memory_gb:
        errors.append('Requested resources exceed worker capacity')
    if getattr(budget,'timeouts_enabled',True) and plan.time_limit_seconds + max(0,evaluation_reserve_seconds) > budget.research_seconds:
        errors.append('Experiment exceeds research deadline including evaluation reserve')
    if plan.prediction_kind != prediction_kind(context.task.primary_metric, context.task.task_type):
        errors.append('prediction_kind does not match the primary metric')
    try: skills.load(plan.skill_ids, context.task.task_type)
    except (GlobalAPIError, WorkerInfrastructureError):
        raise
    except Exception as error: errors.append(f'Skills: {error}')
    for name, spec in plan.search_space.items():
        if name.startswith('policy_params.'):
            path=name.split('.')
            policy_fields={'target_transform':set(item for item in type(plan.target_transform).model_fields),
                           'weight_policy':set(item for item in type(plan.weight_policy).model_fields),
                           'prediction_policy':set(item for item in type(plan.prediction_policy).model_fields)}
            if len(path)!=3 or path[1] not in policy_fields or path[2] not in policy_fields[path[1]]:
                errors.append('Unknown custom policy parameter route: '+name)
        if not isinstance(spec.get('type'),str) or spec.get('type') not in {'int','float','categorical'}: errors.append(f'Invalid search type: {name}')
        elif spec['type'] == 'categorical':
            choices=spec.get('choices')
            if not isinstance(choices,list) or not choices or any(type(x) not in {str,int,float,bool,type(None)} for x in choices):errors.append(f'Invalid search choices: {name}')
        else:
            low, high = spec.get('low'), spec.get('high')
            if not isinstance(low,(int,float)) or not isinstance(high,(int,float)) or low > high:
                errors.append(f'Invalid search bounds: {name}')
            elif spec.get('log') and low <= 0: errors.append(f'Log search requires positive bounds: {name}')
            if spec['type']=='int' and (type(low) is not int or type(high) is not int):errors.append(f'Integer search requires integer bounds: {name}')
        if 'log' in spec and type(spec['log']) is not bool:errors.append(f'Log flag must be boolean: {name}')
    if plan.strategy == 'optuna' and not plan.search_space: errors.append('Optuna requires a search space')
    reserved={'n_jobs','num_threads','thread_count','num_cpus','num_gpus','device','device_type','task_type',
              'pretrained','presets','hyperparameters','general_params','reader_params','estimator_list',
              'time_budget','time_limit','timeout','gpu_ids','starting_points','__native_frozen','__ensemble_frozen',
              'log_file_name','log_type'}
    if tool and tool.native_hpo:reserved.add('max_iter')
    if reserved & (set(plan.model_params)|set(plan.search_space)):
        errors.append('Model/search parameters attempt to override infrastructure or pretrained policy')
    if plan.tool_id=='statsmodels' and context.task.task_type!='regression':
        errors.append('The statsmodels OLS adapter supports single-output regression')
    if tool and tool.native_hpo:
        from .evaluation import metric_name
        accepted={'rmse','mae','f1_macro','roc_auc','accuracy','log_loss'}
        if plan.tool_id=='lightautoml':accepted={'rmse','roc_auc','log_loss'}
        if metric_name(context.task.primary_metric) not in accepted:errors.append('Native adapter cannot optimize this metric')
        if plan.tool_id=='lightautoml' and metric_name(context.task.primary_metric)!=LIGHTAUTOML_OBJECTIVES.get(context.task.task_type):
            errors.append('LightAutoML native objective must match its task: regression=rmse, binary=roc_auc, multiclass=log_loss')
    if plan.ensemble_components:
        for key in plan.search_space:
            if key.startswith('policy_params.'):continue
            if key not in plan.parameter_routes:
                errors.append(f'Flat ensemble HPO parameter {key} requires parameter_routes to a named component')
        component_names={entry.get('name',entry.get('tool_id')) for entry in plan.ensemble_components}
        if len(component_names)!=len(plan.ensemble_components):
            errors.append('Ensemble component names must be unique; add name when using the same backend twice')
        if set(plan.parameter_routes.values())-component_names:
            errors.append('parameter_routes names an undeclared ensemble component')
        if len(plan.ensemble_components)<2:errors.append('At least two ensemble components are required')
        if context.task.task_type not in {'regression','multioutput_regression','binary_classification','multiclass_classification','multilabel_classification'}:
            errors.append('Ensemble SDK supports supervised regression/classification')
        if 'ensembles' not in plan.skill_ids: errors.append('Ensembles require the ensembles skill')
        for component in plan.ensemble_components:
            entry = tools.get(component.get('tool_id'))
            if not entry or not entry.available or context.task.task_type not in entry.task_types:
                errors.append('Unavailable/incompatible ensemble component')
            if entry and entry.native_hpo:
                errors.append('Native AutoML ensemble components have no verified frozen refit protocol')
            if lightweight and entry and entry.cost_tag == 'slow':
                errors.append('Lightweight ensemble components cannot use slow backends')
            if entry and not set(entry.skill_ids)<=set(plan.skill_ids):errors.append('Required ensemble component skills are missing')
            if entry and context.task.validation_strategy not in entry.validation_strategies:
                errors.append('Ensemble component has no verified support for the internal validation strategy')
            if entry and context.task.weight_column and not entry.supports_weights:errors.append('Ensemble component cannot honor sample weights')
            if reserved & set(component.get('model_params',{})):errors.append('Ensemble component overrides infrastructure policy')
    if plan.ensemble_sources:
        from .model_spec import TargetTransform, WeightPolicy, PredictionPolicy
        names=[s.get('name') for s in plan.ensemble_sources]
        if len(names)<2 or len(set(names))!=len(names) or set(names)!={s.get('name') for s in plan.ensemble_components}:
            errors.append('Frozen ensemble sources must match two or three distinct declared components')
        if plan.target_transform!=TargetTransform() or plan.weight_policy!=WeightPolicy() or plan.prediction_policy!=PredictionPolicy():
            errors.append('Frozen-source ensembles keep policies inside each component; outer policies must be identity')
        if plan.strategy!='single' or plan.search_space or plan.preprocessing:
            errors.append('Frozen-source recipes require single execution and component-owned preprocessing')
        if plan.model_params.get('method') not in {'mean','median','weighted','stack'}:
            errors.append('Unsupported frozen-source ensemble recipe')
        if plan.model_params.get('method')=='median' and 'classification' in context.task.task_type:
            errors.append('Median is a regression-only recipe')
        for source in plan.ensemble_sources:
            cfg=source.get('config',{})
            if cfg.get('ensemble_components') or cfg.get('ensemble_sources'):
                errors.append('Recursive ensembles are not admitted')
            if not source.get('candidate_id') or not source.get('experiment_id') or not source.get('code_hash'):
                errors.append('Frozen ensemble source requires verified candidate/experiment/hash references')
    return ValidationReport(valid=not errors, errors=errors)


class PlanningState(TypedDict, total=False):
    payload: dict
    hypotheses: dict
    plans: list[dict]
    attempts: int
    errors: list[dict]
    valid: bool
    warnings: list[dict]
    groups: list[dict]
    skipped: list[dict]


def build_planning_graph(client, context, tools, skills, budget, max_attempts=3, max_candidates=2,
                         role_tools=None, known_candidates=None, *, lightweight=None,
                         initial_experiment_seconds=300, hypothesis_ids=None, planning_mode='all',
                         probe_tool=None, evaluation_reserve_seconds=0, candidate_hypotheses=None,
                         hypothesis_seconds=210, planner_seconds=210, validator_seconds=210,
                         saved_progress=None,save_progress=None,initial_max_workers=3):
    """Plan hypotheses independently; callers can publish a baseline before alternatives.

    Pass a previous output's ``hypotheses`` in the input state to reuse the same
    batch with planning_mode='alternatives', without generating new hypotheses.
    Only baseline mode defers variants. All and alternatives modes finish every
    selected family member; an accepted plan must not truncate the search group.
    """
    if not 1 <= max_candidates <= 2: raise ValueError('max_candidates must be between 1 and 2')
    if planning_mode not in {'all', 'baseline', 'alternatives', 'initial'}: raise ValueError('Unknown planning mode')
    from .search_groups import compatible_boosters, initial_boosting_resources
    initial_resources = (initial_boosting_resources(budget,
        len(compatible_boosters(tools, context.task.task_type)), initial_max_workers)
        if planning_mode == 'initial' else None)
    worker_budget = budget.model_copy(update=dict(available_cpus=initial_resources['cpus'],
        available_memory_gb=initial_resources['memory_gb'])) if initial_resources else budget
    progress=dict(saved_progress or {})
    progress.setdefault('proposed',{})
    progress.setdefault('accepted',{})
    def persist():
        if save_progress:save_progress(progress)

    def is_light(state):
        return lightweight if lightweight is not None else state['payload'].get('round_number',1) == 1

    def for_hypothesis(hypothesis, function, *args, **kwargs):
        with bind(hypothesis_id=hypothesis.hypothesis_id):
            return function(*args, **kwargs)

    def generate(role, instruction, payload, schema, role_tools=None):
        if initial_resources:
            instruction += (' This is the required first-cycle boosting comparison. Plan one fixed '
                'fit for each required_tool_id with backend-specific parameters and preprocessing. '
                'All members use the same confirmed folds, metric and seed. CPU/RAM are assigned '
                'by the runtime; use budget.available_cpus and budget.available_memory_gb per worker. '
                'Feature search and HPO start only after the initial family fits finish. ')
            payload = dict(payload, budget=worker_budget.model_dump(), initial_boosting_sweep=initial_resources)
        instruction += (' Hard timeouts apply only when budget.timeouts_enabled=true. Otherwise time fields '
            'are estimates, not cutoffs; CPU, memory and trial-count limits still apply. ')
        instruction += (' Read planning_brief for task purpose, quoted domain details and prioritized dataset evidence. '
            'Canonical task and planning_brief.requirements are authoritative; never replace them with narrative '
            'assumptions. Account for mandatory_findings even when not selected in ranked_findings. '
            'Proposed experiments are not measured benefits. ')
        instruction += (' technique_skills lists compatible preprocessing/modeling techniques as metadata. '
            'Choose only techniques justified by the profile, findings or hypothesis. Planner must describe '
            'the intended operations in preprocessing/model assignment and include their skill_id in '
            'plan.skill_ids alongside the backend skill. Runtime sends implementation instructions to '
            'the owning role; preprocessing.py preserves rows. Runtime owns declared target/weight/output policies; '
            'custom estimator operations belong to coding_agent. Catalog availability alone is not evidence of benefit. ')
        if role_tools is not None:
            instruction += 'Use read_task if a task nuance needs clarification. '
        payload=dict(payload,technique_skills=skills.technique_catalog(context.task.task_type),
                     skills=skills.list(role=role),
                     skill_instructions=skills.load([],context.task.task_type,role=role))
        if role_tools is not None:
            from .tools import ToolSet
            scoped=ToolSet()
            scoped.tools=dict(role_tools.tools)
            scoped.register('list_skills',lambda:skills.list(role=role),'List this role instruction packages.')
            def read_skills(skill_ids: list[str]) -> list[dict]:
                return skills.load(skill_ids,context.task.task_type,role=role)
            scoped.register('read_skills',read_skills,'Read instructions available to this role only.')
            role_tools=scoped
        key = str(payload.get('round_number',1))
        key += f":p{payload.get('planning_attempt',0)}"
        if role != 'hypothesis_generator':
            key += ':' + payload['hypotheses']['hypotheses'][0]['hypothesis_id']
        seconds = {'hypothesis_generator':hypothesis_seconds,'planner':planner_seconds,
                   'plan_validator':validator_seconds}[role]
        with role_window(client, role, key, seconds):
            return client.generate(role,instruction,payload,schema,role_tools)

    def hypotheses(state):
        errors=[]
        for _ in range(max_attempts):
            try:
                if state.get('hypotheses'):
                    batch=HypothesisBatch.model_validate(state['hypotheses'])
                else:
                    batch = generate('hypothesis_generator',
                        f'Propose 1 to {max_candidates} distinct tabular ML hypotheses. Round 1 or first_candidate_required=true requires a fast baseline. '
                        'A hypothesis tests an approach, not one library. For a boosting comparison set kind=family, '
                        'model_family=gradient_boosting: the runner compares compatible LightGBM, CatBoost and XGBoost '
                        'inside that one hypothesis before tuning the winner. Other families specify up to three family_tools. '
                        'Do not spend separate hypothesis slots on equivalent boosting implementations. '
                        'When lightweight=true, propose simple fixed-configuration fits only: no HPO, AutoML, neural '
                        'networks. Prefer a cheap single-model baseline; ensembles are permitted when justified. The coder implements the baseline directly using the selected library '
                        'and simple fold-local preprocessing; reference skills provide implementation examples. '
                        'The runner performs the confirmed internal validation; candidate.fit must not repeat it. '
                        'Use supplied history; empty history is normal before the first fit. '
                        'After baseline, cite numeric observations or previous experiments in rationale and propose '
                        'a change in model logic or preprocessing. Parameter trials belong INSIDE a hypothesis. '
                        'based_on is a verified candidate_id only; use based_on_hypothesis_id for hypothesis lineage. '
                        'Declare target_transform, weight_policy and prediction_policy separately from native model_params. '
                        'Runtime owns these operations; never ask the coder to apply them a second time.',
                        dict(state['payload'],lightweight=is_light(state),feedback=errors), HypothesisBatch, role_tools)
                if len(batch.hypotheses)>max_candidates: raise ValueError('Too many hypotheses for this round')
                ids=[h.hypothesis_id for h in batch.hypotheses]
                if len(ids)!=len(set(ids)): raise ValueError('Duplicate hypothesis ID')
                if (state['payload'].get('round_number',1)==1 or state['payload'].get('first_candidate_required')) and not any(h.baseline for h in batch.hypotheses):
                    raise ValueError('Round 1 requires a baseline')
                if planning_mode == 'initial':
                    parent = next(h for h in batch.hypotheses if h.baseline).model_copy(deep=True)
                    parent.kind, parent.model_family = 'family', 'gradient_boosting'
                    parent.family_tools = ['lightgbm', 'catboost', 'xgboost']
                    parent.description = ('Compare fixed-configuration LightGBM, CatBoost and XGBoost '
                        'as the initial tabular baseline. Implement exactly required_tool_id with its own '
                        'native parameters and cheap fold-local preprocessing; preserve the canonical '
                        'task, feature availability, metric, folds and seed. No HPO in these initial fits.')
                    diversity = ' The first cycle must retain diverse verified boosting models for later ensembles.'
                    if not parent.rationale.endswith(diversity):
                        parent.rationale += diversity
                    batch = HypothesisBatch(hypotheses=[parent])
                progress['hypotheses']=batch.model_dump(mode='json');persist()
                return {'hypotheses':batch.model_dump(mode='json'), 'attempts':0, 'errors':[]}
            except ValueError as error:
                errors=[str(error)]
                if state.get('hypotheses'): break
        raise ValueError('Hypothesis generation failed: '+'; '.join(errors))

    def plan(state):
        accepted=[]; rejected=[]; attempts=0; warnings=[]
        source=HypothesisBatch.model_validate(state['hypotheses']).hypotheses
        chosen=[h for h in source if hypothesis_ids is None or h.hypothesis_id in hypothesis_ids]
        if planning_mode in {'baseline', 'initial'}: chosen=[h for h in chosen if h.baseline][:1]
        elif planning_mode=='alternatives':
            baseline=next((h.hypothesis_id for h in source if h.baseline),None)
            chosen=[h for h in chosen if h.hypothesis_id!=baseline or h.kind=='family']
        chosen.sort(key=lambda h:not h.baseline)
        from .search_groups import expand_families, collect_groups, configure_composite
        chosen, family_metadata, skipped = expand_families(chosen,tools,context.task.task_type,
            'all' if planning_mode == 'initial' else planning_mode)
        for hypothesis in chosen:
            feedback=[]
            for _ in range(max_attempts):
                attempts+=1
                validating=False
                try:
                    saved=progress['proposed'].get(hypothesis.hypothesis_id)
                    batch = PlanBatch(plans=[saved]) if saved else for_hypothesis(hypothesis,generate,'planner',
                        'Turn exactly the supplied hypothesis into one executable plan, preserving its ID, '
                        'When required_tool_id is supplied, use exactly that backend; it is one member of a family search. '
                        'When ensemble_sources is supplied, plan a frozen-source composition via SDK FrozenEnsemble; '
                        'do not recreate base model code or share preprocessing between components. '
                        'meaning and baseline flag. Use available tools and skills, or an installed backend whose '
                        'reason is exactly "adapter probe required": selected backends will be probed before '
                        'validation. Never select missing libraries or version mismatches. Native AutoML requires '
                        'strategy=native and patience_trials=null. Your role receives its own instructions and may read '
                        'available shared contracts only. Backend implementation references belong to the coders. '
                        'Use supplied context to return final JSON first; '
                        'Select backend skills, hpo for Optuna, feature_engineering for new features, '
                        'imbalance for supported classification or regression weighting/thresholds. Ensembles are allowed; include the ensembles and component skills. '
                        'When lightweight=true require strategy=single, max_trials=1, no '
                        'native AutoML or slow backend, and an optional time_limit_seconds of '
                        f'{initial_experiment_seconds}. Never override orchestration resources; '
                        'exclude targets, IDs and evaluation data from features. Always use execution_mode=custom: '
                        'the preprocessing agent implements transforms; the model coder implements the estimator. In lightweight mode '
                        'keep preprocessing cheap and parameters fixed for the initial fit. Declare a small compatible '
                        'search_space even for a single baseline: it is used only after the first CSV is saved. '
                        'Describe required transforms in preprocessing, including column names and prediction '
                        'constraints. The runner executes the confirmed internal validation strategy; '
                        'budget all folds plus a mandatory full-train refit. Do not implement CV in candidate.fit. '
                        'Native AutoML requires a verified validation_strategies entry matching the confirmed task. '
                        'Use strategy=optuna only for an explicitly declared HPO intervention; keep fixed ablations single. '
                        'Declare target_transform, weight_policy and prediction_policy as runtime policies, never native model_params. '
                        'Route custom search dimensions as policy_params.<policy>.<field>. '
                        'never implement HPO in candidate.fit. Respect improvement_policy; stacking meta learners require OOF base predictions. '
                        'Backend skills are references, not hidden runtime adapters. The code must consume every '
                        'declared model/search parameter. For text specify a sparse linear estimator with C for '
                        'classification or alpha for regression. '
                        'based_on must be one of known_candidates; hypothesis lineage uses based_on_hypothesis_id.',
                        dict(state['payload'],lightweight=is_light(state),
                             required_tool_id=family_metadata.get(hypothesis.hypothesis_id,(None,None))[1],
                             hypotheses={'hypotheses':[hypothesis.model_dump(mode='json')]},
                             feedback=feedback), PlanBatch, role_tools)
                    if len(batch.plans)!=1: raise ValueError('Return exactly one plan for the supplied hypothesis')
                    item=batch.plans[0]
                    requested_cpus = item.cpus
                    if initial_resources:
                        item.cpus = initial_resources['cpus']
                        item.memory_gb = initial_resources['memory_gb']
                    # Definitions come from runtime memory, never from a planner-written formula.
                    registry={v['definition']['feature_set_id']:v['definition'] for v in state['payload'].get('feature_sets',[])}
                    if item.feature_set_id:
                        if item.feature_set_id not in registry:
                            raise ValueError('Unknown/unaccepted feature_set_id')
                        from .feature_contracts import FeatureSet
                        registered=FeatureSet.model_validate(registry[item.feature_set_id])
                        if item.feature_set is not None and item.feature_set != registered:
                            raise ValueError('Planner changed a registered feature definition')
                        item.feature_set=registered
                    elif item.feature_set is not None:
                        raise ValueError('Planner must select a registered feature_set_id')
                    if hypothesis.hypothesis_id in family_metadata:
                        parent, required_tool = family_metadata[hypothesis.hypothesis_id]
                        if item.tool_id != required_tool:
                            raise ValueError('Family variant requires tool_id='+required_tool)
                        item.search_group_id=parent.hypothesis_id
                        item.search_group_kind='family'
                        item.parent_hypothesis=parent
                        item.strategy='single';item.max_trials=1
                    if state['payload'].get('ensemble_sources'):
                        item=configure_composite(item,state['payload']['ensemble_sources'],
                                                 state['payload'].get('ensemble_weight_trials',15))
                    champion=state['payload'].get('champion')
                    if champion and item.hypothesis.based_on==champion['candidate_id']:
                        digest=champion.get('configuration_hash')
                        if item.base_config_hash and item.base_config_hash != digest:
                            raise ValueError('base_config_hash differs from the authoritative Champion snapshot')
                        item.base_config_hash=digest
                        if item.change_kind=='single_change_ablation':
                            import copy
                            if item.tool_id != champion['plan']['tool_id']:
                                raise ValueError('Single-change preprocessing ablation must retain Champion backend')
                            item.model_params=copy.deepcopy(champion['model_params'])
                            item.strategy='single';item.search_space={};item.max_trials=1
                            old=champion['plan']
                            policy_types={'target_transform':type(item.target_transform),'weight_policy':type(item.weight_policy),
                                          'prediction_policy':type(item.prediction_policy)}
                            for key,cls in policy_types.items():
                                if key not in item.model_fields_set:
                                    setattr(item,key,cls.model_validate(old.get(key,{})))
                            fields=('preprocessing','target_transform','weight_policy','prediction_policy','ensemble_components','stopping_policy','feature_set_id')
                            current=item.model_dump(mode='json')
                            changed=[key for key in fields if current[key]!=old.get(key,current[key] if key=='stopping_policy' else
                                {} if key in policy_types else []) and
                                not (key in policy_types and current[key]==policy_types[key]().model_dump() and key not in old)]
                            allowed={key.split('.')[0] for key in item.intervention.changed_fields}
                            if allowed and set(changed)-allowed:
                                raise ValueError('Undeclared ablation changes: '+', '.join(sorted(set(changed)-allowed)))
                            if len(changed)>1:
                                raise ValueError('Single-change ablation modifies multiple policy/feature groups: '+', '.join(changed))
                            item.intervention.changed_fields=changed
                    elif item.change_kind=='single_change_ablation':
                        raise ValueError('Single-change ablation requires based_on=current Champion candidate_id')
                    actual_protocol=dict(strategy=str(context.task.validation_strategy),
                        folds=context.task.validation_folds,duplicate_policy=getattr(context.task,'duplicate_policy','allow'))
                    item.validation_protocol,metadata=normalize_protocol(item.validation_protocol,actual_protocol)
                    item.validation_metadata.update(metadata)
                    fixed=constant_space(item.search_space) if item.strategy=='optuna' else None
                    if fixed is not None:
                        normalized=apply_overrides(item.model_dump(mode='json'),fixed)
                        item=HypothesisPlan.model_validate(dict(normalized,strategy='single',search_space={},max_trials=1))
                        item.intervention.kind='single_configuration'
                    if item.hypothesis.baseline:
                        item.intervention.kind='baseline'
                    elif item.change_kind=='single_change_ablation':
                        item.intervention.kind='ablation'
                    elif item.strategy=='optuna':
                        item.intervention.kind='hpo'
                    item.intervention.base_candidate_id=item.hypothesis.based_on
                    if not item.intervention.operations:
                        item.intervention.operations=list(item.preprocessing)
                    if not item.intervention.changed_fields:
                        item.intervention.changed_fields=list(item.search_space)
                    if not item.intervention.preserved_conditions:
                        item.intervention.preserved_conditions=['canonical task','metric','validation protocol','data isolation']
                    if not item.intervention.success_conditions:
                        item.intervention.success_conditions=['Improve the paired primary metric; no causal claim without a comparable control']
                    if not item.intervention.rejection_conditions:
                        item.intervention.rejection_conditions=['Violation of the declared intervention or its subgroup constraints']
                    normalized=[]
                    for name in ('n_jobs','num_threads','thread_count','num_cpus'):
                        if item.model_params.get(name)==item.cpus or (initial_resources and item.model_params.get(name)==requested_cpus):
                            item.model_params.pop(name);normalized.append(name)
                    if normalized:
                        emit('planning.normalized_resources',{'parameters':normalized,'cpus':item.cpus})
                    if item.hypothesis.hypothesis_id!=hypothesis.hypothesis_id or item.hypothesis.baseline!=hypothesis.baseline:
                        raise ValueError('Plan must preserve the hypothesis ID and baseline flag')
                    if item.hypothesis.based_on and known_candidates is not None and item.hypothesis.based_on not in known_candidates:
                        alias=item.hypothesis.based_on
                        lineage={h.hypothesis_id for h in source} | set((candidate_hypotheses or {}).values())
                        if alias not in lineage:
                            raise ValueError('based_on must reference an existing verified candidate_id')
                        matches=[key for key,value in (candidate_hypotheses or {}).items() if value==alias and key in known_candidates]
                        item.hypothesis.based_on_hypothesis_id=alias
                        item.hypothesis.based_on=matches[-1] if matches else None
                        emit('planning.normalized_lineage',item.hypothesis)
                    if probe_tool is not None:
                        # Probe only the proposed backends, never the full heavy
                        # registry. Forbidden initial plans must not trigger
                        # expensive capability checks before rejection.
                        tool=tools.get(item.tool_id)
                        forbidden_initial=is_light(state) and (
                            (tool and (tool.native_hpo or tool.cost_tag=='slow')))
                        if not forbidden_initial:
                            selected=[item.tool_id,*[c.get('tool_id') for c in item.ensemble_components]]
                            for key in dict.fromkeys(selected):
                                selected_tool=tools.get(key)
                                if selected_tool and not selected_tool.available and selected_tool.reason=='adapter probe required':
                                    for_hypothesis(hypothesis,probe_tool,key)
                    report=for_hypothesis(hypothesis,validate_plan,item,context,tools,skills,worker_budget,lightweight=is_light(state),
                                         initial_experiment_seconds=initial_experiment_seconds,
                                         evaluation_reserve_seconds=evaluation_reserve_seconds)
                    if not report.valid: raise ValueError('; '.join(report.errors))
                    progress['proposed'][hypothesis.hypothesis_id]=item.model_dump(mode='json');persist()
                    if progress['accepted'].get(hypothesis.hypothesis_id)==item.model_dump(mode='json'):
                        accepted.append(item.model_dump(mode='json'))
                        emit('planning.validation_reused',{'hypothesis_id':hypothesis.hypothesis_id})
                        break
                    instructions=for_hypothesis(hypothesis,skills.load,[],context.task.task_type,role='plan_validator')
                    probes=[]
                    if item.tool_id=='lightgbm' and any(word in ' '.join(item.preprocessing).lower() for word in ('categor','unseen')):
                        try:
                            probes.append(categorical_probe())
                        except WorkerInfrastructureError:
                            raise
                        except Exception as error:
                            probes.append(dict(probe_id='lightgbm_native_categorical_unseen_nan',status='unavailable',error_type=type(error).__name__))
                        emit('planning.capability_probes',probes)
                    validating=True
                    report=for_hypothesis(hypothesis,generate,'plan_validator',
                        'min_delta is a legacy absolute comparison field, NOT the batched HPO continuation threshold. '
                        'min_delta=0 is correct: the runner separately enforces hpo_gain_threshold=0.005 and keeps tiny gains. '
                        'Review feasibility, leakage and methodological validity against the supplied skill '
                        'instructions. Return a concise validation JSON; never rewrite plans. '
                        'Separate preprocessing and model coders implement the candidate. Check that transforms and parameters can '
                        'implement every claim, with no implicit adapter behavior. The runner owns splits and HPO. '
                        'Require explicit operations and columns for changes such as treating numeric years as '
                        'categories. Single baseline search_space is for later HPO. '
                        'Blocking issues must include rule_id, field, reason and source_quote in issues. '
                        'Use task_contract for canonical task violations or holdout_training for a quoted instruction to train on external labels. '
                        'Unproven library incompatibilities are warnings. Runtime executes declared target/weight/output policies.',
                        dict(round_number=state['payload'].get('round_number',1),
                             planning_brief=state['payload'].get('planning_brief'),
                             planning_attempt=state['payload'].get('planning_attempt',0),lightweight=is_light(state),
                             task=context.task.model_dump(mode='json') if hasattr(context.task,'model_dump') else
                                  dict(task_type=context.task.task_type,primary_metric=context.task.primary_metric),
                             features=getattr(context,'feature_columns',[]),budget=budget.model_dump(),
                             plans=[item.model_dump(mode='json')],
                             capability_probes=probes,
                             hypotheses={'hypotheses':[hypothesis.model_dump(mode='json')]},
                             skill_instructions=[s for s in instructions if s['skill_id']!='candidate_sdk']),ValidationReport)
                    report=ground_semantic_report(report,item,context)
                    warnings.extend(report.warnings)
                    if not report.valid or report.errors:
                        raise ValueError('; '.join(report.errors) or 'Semantic validator rejected the plan without an explanation')
                    accepted.append(item.model_dump(mode='json'))
                    progress['accepted'][hypothesis.hypothesis_id]=item.model_dump(mode='json');persist()
                    break
                except StructuredOutputError as error:
                    if not validating:
                        feedback=[str(error)]
                        continue
                    rejected.append({'hypothesis_id':hypothesis.hypothesis_id,'errors':[str(error)]})
                    break
                except (GlobalAPIError, WorkerInfrastructureError):
                    raise
                except LLMUnavailable as error:
                    rejected.append({'hypothesis_id':hypothesis.hypothesis_id,'errors':[str(error)]})
                    break
                except ValueError as error:
                    feedback=[str(error)]
                    progress['proposed'].pop(hypothesis.hypothesis_id,None)
                    progress['accepted'].pop(hypothesis.hypothesis_id,None);persist()
            else: rejected.append({'hypothesis_id':hypothesis.hypothesis_id,'errors':feedback})
        # A rejected alternative never discards an already accepted baseline.
        if planning_mode == 'initial' and accepted and not any(p['hypothesis']['baseline'] for p in accepted):
            # A failed first backend must not discard the other valid starters.
            accepted[0]['hypothesis']['baseline'] = True
        valid=bool(accepted)
        if (state['payload'].get('round_number',1)==1 or state['payload'].get('first_candidate_required')) and planning_mode!='alternatives':
            valid=valid and any(p['hypothesis']['baseline'] for p in accepted)
        for rejection in rejected:
            if rejection['hypothesis_id'] in family_metadata:
                parent,tool_id=family_metadata[rejection['hypothesis_id']]
                skipped.append(dict(group_id=parent.hypothesis_id,variant_id=rejection['hypothesis_id'],
                                    tool_id=tool_id,reason='; '.join(rejection['errors'])))
        return {'plans':accepted,'groups':[g.model_dump(mode='json') for g in collect_groups(accepted,skipped)],
                'skipped':skipped,'attempts':attempts,'errors':rejected,'valid':valid,'warnings':warnings}

    graph = StateGraph(PlanningState)
    graph.add_node('hypotheses',hypotheses); graph.add_node('plan',plan)
    graph.add_edge(START,'hypotheses'); graph.add_edge('hypotheses','plan'); graph.add_edge('plan',END)
    return graph.compile()
