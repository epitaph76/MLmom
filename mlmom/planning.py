"""Hypothesis planning subgraph and deterministic plan validation."""
from __future__ import annotations

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from .pipeline_contracts import (HypothesisBatch, PlanBatch, HypothesisPlan, ValidationReport,
                                  BudgetSnapshot, TaskContext)
from .evaluation import prediction_kind


def validate_plan(plan: HypothesisPlan, context: TaskContext, tools: dict, skills, budget: BudgetSnapshot) -> ValidationReport:
    errors = []
    tool = tools.get(plan.tool_id)
    if not tool or not tool.available: errors.append(f'Tool unavailable: {plan.tool_id}')
    else:
        if context.task.task_type not in tool.task_types: errors.append('Tool does not support the task type')
        if context.task.group_columns and not tool.supports_groups and context.task.task_type == 'ranking': errors.append('Ranking requires a group-aware adapter')
        if context.task.weight_column and not tool.supports_weights: errors.append('Adapter does not support sample weights')
        if plan.strategy == 'native' and not tool.native_hpo: errors.append('Native HPO is not supported')
        if tool.native_hpo and plan.strategy == 'optuna': errors.append('Do not nest Optuna around native AutoML')
        if plan.strategy == 'native' and plan.patience_trials is not None and not tool.supports_patience:
            errors.append('Native adapter has no trial-patience callback; set patience_trials=null')
        if not set(tool.skill_ids) <= set(plan.skill_ids): errors.append('Required library skills are missing')
    if plan.cpus > budget.available_cpus or plan.memory_gb > budget.available_memory_gb:
        errors.append('Requested resources exceed worker capacity')
    if plan.time_limit_seconds > budget.research_seconds: errors.append('Experiment exceeds research deadline')
    if plan.prediction_kind != prediction_kind(context.task.primary_metric, context.task.task_type):
        errors.append('prediction_kind does not match the primary metric')
    try: skills.load(plan.skill_ids, context.task.task_type)
    except Exception as error: errors.append(f'Skills: {error}')
    for name, spec in plan.search_space.items():
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
              'log_file_name','log_type','max_iter'}
    if reserved & (set(plan.model_params)|set(plan.search_space)):
        errors.append('Model/search parameters attempt to override infrastructure or pretrained policy')
    if plan.tool_id=='statsmodels' and context.task.task_type!='regression':
        errors.append('The statsmodels OLS adapter supports single-output regression')
    if tool and tool.native_hpo:
        from .evaluation import metric_name
        accepted={'rmse','mae','f1_macro','roc_auc','accuracy','log_loss'}
        if plan.tool_id=='lightautoml':accepted={'rmse','roc_auc','log_loss'}
        if metric_name(context.task.primary_metric) not in accepted:errors.append('Native adapter cannot optimize this metric')
    if plan.ensemble_components:
        if context.task.task_type not in {'regression','multioutput_regression','binary_classification','multiclass_classification','multilabel_classification'}:
            errors.append('Ensemble SDK supports supervised regression/classification')
        if 'ensembles' not in plan.skill_ids: errors.append('Ensembles require the ensembles skill')
        for component in plan.ensemble_components:
            entry = tools.get(component.get('tool_id'))
            if not entry or not entry.available or context.task.task_type not in entry.task_types:
                errors.append('Unavailable/incompatible ensemble component')
            if entry and context.task.weight_column and not entry.supports_weights:errors.append('Ensemble component cannot honor sample weights')
            if reserved & set(component.get('model_params',{})):errors.append('Ensemble component overrides infrastructure policy')
    return ValidationReport(valid=not errors, errors=errors)


class PlanningState(TypedDict, total=False):
    payload: dict
    hypotheses: dict
    plans: list[dict]
    attempts: int
    errors: list[str]
    valid: bool


def build_planning_graph(client, context, tools, skills, budget, max_attempts=3, max_candidates=3, role_tools=None, known_candidates=None):
    def hypotheses(state):
        errors=[]
        for _ in range(max_attempts):
            try:
                batch = client.generate('hypothesis_generator',
                    f'Propose 1 to {max_candidates} distinct tabular ML hypotheses. Round 1 must contain a fast baseline. '
                    'Use history to improve strong approaches and avoid failed configurations.',
                    dict(state['payload'],feedback=errors), HypothesisBatch, role_tools)
                if len(batch.hypotheses)>max_candidates:raise ValueError('Too many hypotheses for this run')
                return {'hypotheses':batch.model_dump(mode='json'), 'attempts':0, 'errors':[]}
            except ValueError as error:errors=[str(error)]
        raise ValueError('Hypothesis generation failed: '+'; '.join(errors))

    def plan(state):
        try:
            batch = client.generate('planner',
                'Turn the supplied hypotheses into executable plans, preserving their IDs and meaning. '
                'Use only available tools and skills. Native AutoML must set patience_trials=null. '
                'Core candidate_sdk, validation, metrics and preprocessing skills load automatically. '
                'Select the backend skill; add hpo for Optuna, feature_engineering for new features, '
                'imbalance for weighting/threshold changes, and ensembles plus component backend skills for ensembles. '
                'Start with a cheap baseline, then a booster; native AutoML is an alternative search backend. '
                'Do not include targets, IDs or evaluation data in features. A fast baseline uses strategy=single. '
                'Model parameters passed to adapters must not override CPU, network, GPU or orchestration settings.',
                dict(state['payload'], hypotheses=state['hypotheses'], previous_plans=state.get('plans',[]),
                     feedback=state.get('errors',[])), PlanBatch, role_tools)
            plans = [p.model_dump(mode='json') for p in batch.plans][:max_candidates]
            return {'plans':plans, 'attempts':state.get('attempts',0)+1, 'errors':[]}
        except ValueError as error:
            return {'plans':[], 'attempts':state.get('attempts',0)+1,'errors':[str(error)]}

    def validate(state):
        errors = list(state.get('errors',[]))
        ids = set()
        for raw in state['plans']:
            item = HypothesisPlan.model_validate(raw)
            if item.hypothesis.hypothesis_id in ids: errors.append('Duplicate hypothesis ID')
            ids.add(item.hypothesis.hypothesis_id)
            if item.hypothesis.based_on and known_candidates is not None and item.hypothesis.based_on not in known_candidates:
                errors.append('based_on must reference an existing candidate_id')
            errors.extend(validate_plan(item,context,tools,skills,budget).errors)
        source = {h['hypothesis_id'] for h in state['hypotheses']['hypotheses']}
        if ids != source: errors.append('Plans must preserve the complete hypothesis set')
        if state['payload'].get('round_number') == 1 and not any(p['hypothesis']['baseline'] for p in state['plans']):
            errors.append('Round 1 requires a baseline')
        if not state['plans']: errors.append('No plans generated')
        return {'errors':errors,'valid':not errors}

    def semantic(state):
        report = client.generate('plan_validator',
            'Review feasibility, leakage, diversity and methodological validity. Return findings; never rewrite plans.',
            dict(state['payload'],plans=state['plans']), ValidationReport)
        if not report.valid and not report.errors:
            report.errors.append('Semantic validator rejected the plan without an explanation')
        return {'valid':report.valid and not report.errors,'errors':report.errors}

    def route(state):
        return 'finish' if state['valid'] or state['attempts'] >= max_attempts else 'plan'

    graph = StateGraph(PlanningState)
    for name,node in [('hypotheses',hypotheses),('plan',plan),('validate',validate),('semantic',semantic)]: graph.add_node(name,node)
    graph.add_edge(START,'hypotheses'); graph.add_edge('hypotheses','plan'); graph.add_edge('plan','validate')
    graph.add_conditional_edges('validate',lambda s:'semantic' if s['valid'] else route(s),{'semantic':'semantic','plan':'plan','finish':END})
    graph.add_conditional_edges('semantic',route,{'finish':END,'plan':'plan'})
    return graph.compile()
