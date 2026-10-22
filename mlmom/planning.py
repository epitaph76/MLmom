"""Hypothesis planning subgraph and deterministic plan validation."""
from __future__ import annotations

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from .pipeline_contracts import (HypothesisBatch, PlanBatch, HypothesisPlan, ValidationReport,
                                  BudgetSnapshot, TaskContext)
from .evaluation import prediction_kind
from .registry import LIGHTAUTOML_OBJECTIVES
from .agents import LLMUnavailable, StructuredOutputError, role_window
from .tracing import bind, emit


def validate_plan(plan: HypothesisPlan, context: TaskContext, tools: dict, skills, budget: BudgetSnapshot,
                  *, lightweight=False, initial_experiment_seconds=300, evaluation_reserve_seconds=0) -> ValidationReport:
    errors = []
    if plan.ensemble_components or 'ensembles' in plan.skill_ids:
        errors.append('Candidate ensembles and AutoML blending/stacking are disabled')
    tool = tools.get(plan.tool_id)
    if not tool or not tool.available: errors.append(f'Tool unavailable: {plan.tool_id}')
    else:
        if context.task.validation_strategy not in tool.validation_strategies:
            errors.append('Tool has no verified support for the confirmed internal validation strategy')
        if tool.native_hpo and context.task.group_columns:
            errors.append('Native group-aware validation has not been verified')
        if context.task.task_type not in tool.task_types: errors.append('Tool does not support the task type')
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
        if plan.ensemble_components: errors.append('Lightweight plans cannot contain ensembles')
        if tool and (tool.native_hpo or tool.cost_tag == 'slow'):
            errors.append('Lightweight plans cannot use native AutoML or slow backends')
        if plan.time_limit_seconds > min(300, initial_experiment_seconds):
            errors.append('Lightweight experiment exceeds the initial experiment limit')
    if plan.cpus > budget.available_cpus or plan.memory_gb > budget.available_memory_gb:
        errors.append('Requested resources exceed worker capacity')
    if plan.time_limit_seconds + max(0,evaluation_reserve_seconds) > budget.research_seconds:
        errors.append('Experiment exceeds research deadline including evaluation reserve')
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
    reserved |= {'ensemble','fit_weighted_ensemble','num_bag_folds','num_stack_levels','auto_stack'}
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
    errors: list[dict]
    valid: bool
    warnings: list[dict]


def build_planning_graph(client, context, tools, skills, budget, max_attempts=3, max_candidates=2,
                         role_tools=None, known_candidates=None, *, lightweight=None,
                         initial_experiment_seconds=300, hypothesis_ids=None, planning_mode='all',
                         probe_tool=None, evaluation_reserve_seconds=0, candidate_hypotheses=None,
                         hypothesis_seconds=210, planner_seconds=210, validator_seconds=210,
                         saved_progress=None,save_progress=None):
    """Plan hypotheses independently; callers can publish a baseline before alternatives.

    Pass a previous output's ``hypotheses`` in the input state to reuse the same
    batch with planning_mode='alternatives', without generating new hypotheses.
    """
    if not 1 <= max_candidates <= 2: raise ValueError('max_candidates must be between 1 and 2')
    if planning_mode not in {'all', 'baseline', 'alternatives'}: raise ValueError('Unknown planning mode')
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
        instruction += (' Read planning_brief for task purpose, quoted domain details and prioritized dataset evidence. '
            'Canonical task and planning_brief.requirements are authoritative; never replace them with narrative '
            'assumptions. Account for mandatory_findings even when not selected in ranked_findings. '
            'Proposed experiments are not measured benefits. ')
        instruction += (' technique_skills lists compatible preprocessing/modeling techniques as metadata. '
            'Choose only techniques justified by the profile, findings or hypothesis. Planner must describe '
            'the intended operations in preprocessing/model assignment and include their skill_id in '
            'plan.skill_ids alongside the backend skill. Runtime sends implementation instructions to '
            'the owning role; preprocessing.py preserves rows, while sampling and target transforms '
            'belong to coding_agent. Catalog availability alone is not evidence of benefit. ')
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
                        'When lightweight=true, propose simple fixed-configuration fits only: no HPO, AutoML, neural '
                        'networks or ensembles. The coder implements a cheap baseline directly using the selected library '
                        'and simple fold-local preprocessing; reference skills provide implementation examples. '
                        'No internal cross-validation. '
                        'Use supplied history; empty history is normal before the first fit. '
                        'After baseline, cite numeric observations or previous experiments in rationale and propose '
                        'a change in model logic or preprocessing. Parameter trials belong INSIDE a hypothesis. '
                        'based_on is a verified candidate_id only; use based_on_hypothesis_id for hypothesis lineage.',
                        dict(state['payload'],lightweight=is_light(state),feedback=errors), HypothesisBatch, role_tools)
                if len(batch.hypotheses)>max_candidates: raise ValueError('Too many hypotheses for this round')
                ids=[h.hypothesis_id for h in batch.hypotheses]
                if len(ids)!=len(set(ids)): raise ValueError('Duplicate hypothesis ID')
                if (state['payload'].get('round_number',1)==1 or state['payload'].get('first_candidate_required')) and not any(h.baseline for h in batch.hypotheses):
                    raise ValueError('Round 1 requires a baseline')
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
        if planning_mode=='baseline': chosen=[h for h in chosen if h.baseline][:1]
        elif planning_mode=='alternatives':
            baseline=next((h.hypothesis_id for h in source if h.baseline),None)
            chosen=[h for h in chosen if h.hypothesis_id!=baseline]
        chosen.sort(key=lambda h:not h.baseline)
        for hypothesis in chosen:
            feedback=[]
            for _ in range(max_attempts):
                attempts+=1
                validating=False
                try:
                    saved=progress['proposed'].get(hypothesis.hypothesis_id)
                    batch = PlanBatch(plans=[saved]) if saved else for_hypothesis(hypothesis,generate,'planner',
                        'Turn exactly the supplied hypothesis into one executable plan, preserving its ID, '
                        'meaning and baseline flag. Use available tools and skills, or an installed backend whose '
                        'reason is exactly "adapter probe required": selected backends will be probed before '
                        'validation. Never select missing libraries or version mismatches. Native AutoML requires '
                        'strategy=native and patience_trials=null. Your role receives its own instructions and may read '
                        'available shared contracts only. Backend implementation references belong to the coders. '
                        'Use supplied context to return final JSON first; '
                        'Select backend skills, hpo for Optuna, feature_engineering for new features, '
                        'imbalance for weighting/thresholds. Candidate ensembles and AutoML blending/stacking are disabled. '
                        'When lightweight=true require strategy=single, max_trials=1, no '
                        'ensemble, no native AutoML or slow backend, and time_limit_seconds at most '
                        f'{min(300,initial_experiment_seconds)}. Never override orchestration resources; '
                        'exclude targets, IDs and evaluation data from features. Always use execution_mode=custom: '
                        'the preprocessing agent implements transforms; the model coder implements the estimator. In lightweight mode '
                        'keep preprocessing cheap and parameters fixed for the initial fit. Declare a small compatible '
                        'search_space even for a single baseline: it is used only after the first CSV is saved. '
                        'Describe required transforms in preprocessing, including column names and prediction '
                        'constraints. The runner executes the confirmed internal validation strategy; '
                        'budget all folds plus a mandatory full-train refit. Do not implement CV in candidate.fit. '
                        'Native AutoML requires a verified validation_strategies entry matching the confirmed task. '
                        'After baseline use strategy=optuna with a small compatible search_space for ordinary estimators; '
                        'never implement HPO in candidate.fit. Respect improvement_policy and do not use ensembles. '
                        'Backend skills are references, not hidden runtime adapters. The code must consume every '
                        'declared model/search parameter. For text specify a sparse linear estimator with C for '
                        'classification or alpha for regression. '
                        'based_on must be one of known_candidates; hypothesis lineage uses based_on_hypothesis_id.',
                        dict(state['payload'],lightweight=is_light(state),
                             hypotheses={'hypotheses':[hypothesis.model_dump(mode='json')]},
                             feedback=feedback), PlanBatch, role_tools)
                    if len(batch.plans)!=1: raise ValueError('Return exactly one plan for the supplied hypothesis')
                    item=batch.plans[0]
                    normalized=[]
                    for name in ('n_jobs','num_threads','thread_count','num_cpus'):
                        if item.model_params.get(name)==item.cpus:
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
                        forbidden_initial=is_light(state) and (item.ensemble_components or
                            (tool and (tool.native_hpo or tool.cost_tag=='slow')))
                        if not forbidden_initial and not item.ensemble_components and 'ensembles' not in item.skill_ids:
                            selected=[item.tool_id,*[c.get('tool_id') for c in item.ensemble_components]]
                            for key in dict.fromkeys(selected):
                                selected_tool=tools.get(key)
                                if selected_tool and not selected_tool.available and selected_tool.reason=='adapter probe required':
                                    for_hypothesis(hypothesis,probe_tool,key)
                    report=for_hypothesis(hypothesis,validate_plan,item,context,tools,skills,budget,lightweight=is_light(state),
                                         initial_experiment_seconds=initial_experiment_seconds,
                                         evaluation_reserve_seconds=evaluation_reserve_seconds)
                    if not report.valid: raise ValueError('; '.join(report.errors))
                    progress['proposed'][hypothesis.hypothesis_id]=item.model_dump(mode='json');persist()
                    if progress['accepted'].get(hypothesis.hypothesis_id)==item.model_dump(mode='json'):
                        accepted.append(item.model_dump(mode='json'))
                        emit('planning.validation_reused',{'hypothesis_id':hypothesis.hypothesis_id})
                        break
                    instructions=for_hypothesis(hypothesis,skills.load,[],context.task.task_type,role='plan_validator')
                    validating=True
                    report=for_hypothesis(hypothesis,generate,'plan_validator',
                        'min_delta is a legacy absolute comparison field, NOT the batched HPO continuation threshold. '
                        'min_delta=0 is correct: the runner separately enforces hpo_gain_threshold=0.005 and keeps tiny gains. '
                        'Review feasibility, leakage and methodological validity against the supplied skill '
                        'instructions. Return a concise validation JSON; never rewrite plans. '
                        'Separate preprocessing and model coders implement the candidate. Check that transforms and parameters can '
                        'implement every claim, with no implicit adapter behavior. The runner owns splits and HPO. '
                        'Require explicit operations and columns for changes such as treating numeric years as '
                        'categories. Single baseline search_space is for later HPO.',
                        dict(round_number=state['payload'].get('round_number',1),
                             planning_brief=state['payload'].get('planning_brief'),
                             planning_attempt=state['payload'].get('planning_attempt',0),lightweight=is_light(state),
                             task=context.task.model_dump(mode='json') if hasattr(context.task,'model_dump') else
                                  dict(task_type=context.task.task_type,primary_metric=context.task.primary_metric),
                             features=getattr(context,'feature_columns',[]),budget=budget.model_dump(),
                             plans=[item.model_dump(mode='json')],
                             hypotheses={'hypotheses':[hypothesis.model_dump(mode='json')]},
                             skill_instructions=[s for s in instructions if s['skill_id']!='candidate_sdk']),ValidationReport)
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
                except LLMUnavailable as error:
                    rejected.append({'hypothesis_id':hypothesis.hypothesis_id,'errors':[str(error)]})
                    break
                except ValueError as error:
                    feedback=[str(error)]
                    progress['proposed'].pop(hypothesis.hypothesis_id,None)
                    progress['accepted'].pop(hypothesis.hypothesis_id,None);persist()
            else: rejected.append({'hypothesis_id':hypothesis.hypothesis_id,'errors':feedback})
        # A rejected alternative never discards an already accepted baseline.
        valid=bool(accepted)
        if (state['payload'].get('round_number',1)==1 or state['payload'].get('first_candidate_required')) and planning_mode!='alternatives':
            valid=valid and any(p['hypothesis']['baseline'] for p in accepted)
        return {'plans':accepted,'attempts':attempts,'errors':rejected,'valid':valid,'warnings':warnings}

    graph = StateGraph(PlanningState)
    graph.add_node('hypotheses',hypotheses); graph.add_node('plan',plan)
    graph.add_edge(START,'hypotheses'); graph.add_edge('hypotheses','plan'); graph.add_edge('plan',END)
    return graph.compile()
