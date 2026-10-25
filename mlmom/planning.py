"""Agent planning and review, with technical resource admission."""
from __future__ import annotations

from .agents import GlobalAPIError
from .resources import WorkerInfrastructureError

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from .pipeline_contracts import (HypothesisBatch, PlanBatch, HypothesisPlan, ValidationReport,
                                  BudgetSnapshot, TaskContext)
from .evaluation import prediction_kind
from .agents import LLMUnavailable, StructuredOutputError, role_window
from .tracing import bind, emit
from .model_spec import normalize_protocol, constant_space, apply_overrides, Intervention
from .planning_assignments import normalize_cpu_hints
from .planning_retries import PlanningRejections
from .execution_contract import execution_contract, plan_field_sources


def validate_plan(plan: HypothesisPlan, context: TaskContext, tools: dict, skills,
                  budget: BudgetSnapshot, *, lightweight=False,
                  initial_experiment_seconds=300, evaluation_reserve_seconds=0) -> ValidationReport:
    errors = []
    if plan.cpus > budget.available_cpus or plan.memory_gb > budget.available_memory_gb:
        errors.append('Requested resources exceed worker capacity')
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
                          saved_progress=None,save_progress=None,initial_max_workers=3,group_parent=None):
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

    rejections = PlanningRejections(progress, persist, max_attempts)

    def is_light(state):
        return lightweight if lightweight is not None else state['payload'].get('round_number',1) == 1

    def for_hypothesis(hypothesis, function, *args, **kwargs):
        with bind(hypothesis_id=hypothesis.hypothesis_id):
            return function(*args, **kwargs)

    def generate(role, instruction, payload, schema, role_tools=None):
        if initial_resources:
            instruction += (' This is the required first-cycle boosting comparison. '
                'All members use the same confirmed folds, metric and seed. CPU/RAM are assigned '
                'by the runtime; use budget.available_cpus and budget.available_memory_gb per worker. '
                'Feature proposals are prepared early; measure features on the first control before long HPO. ')
            if role == 'hypothesis_generator':
                instruction += (' Propose one cheap boosting-family baseline. The runtime expands it '
                    'into separate backend-specific planning and review calls. ')
            elif role == 'planner':
                instruction += (' This call plans only required_tool_id: return exactly one fixed-fit '
                    'plan for that member. The runtime plans the other family members separately. ')
            elif role == 'plan_validator':
                instruction += (' Review only the supplied single family member. The runtime plans '
                    'and checks sibling coverage separately; absent sibling plans are expected here. ')
            payload = dict(payload, budget=worker_budget.agent_view(), initial_boosting_sweep=initial_resources)
        instruction += (' Hard timeouts apply only when budget.timeouts_enabled=true. Otherwise time fields '
            'are estimates, not cutoffs; CPU, memory and trial-count limits still apply. ')
        instruction += (' Historical durations are advisory, not admission tests. A source experiment wall time '
            'may include a whole HPO study and is not one frozen refit cost. Actual scheduling owns parallelism. ')
        instruction += (' Read planning_brief for task purpose, quoted domain details and prioritized dataset evidence. '
            'Canonical task and planning_brief.requirements are authoritative; never replace them with narrative '
            'assumptions. Account for mandatory_findings even when not selected in ranked_findings. '
            'Proposed experiments are not measured benefits. ')
        if role == 'plan_validator':
            instruction += (' technique_skills is reference metadata for checking declared assignments. '
                'Review the existing plan; do not select extra techniques or redesign the baseline. '
                'Distinguish a concrete contract violation from an optional modeling experiment. ')
        else:
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
        state['payload']['rejected_hypotheses'] = rejections.summaries()[-12:]
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
                        'When lightweight=true, prefer inexpensive fixed-configuration fits before long searches. '
                        'Choose families for the actual task; a boosting family is optional. The coder implements the baseline using the selected library '
                        'and simple fold-local preprocessing; reference skills provide implementation examples. '
                        'The runner performs the confirmed internal validation; candidate.fit must not repeat it. '
                        'Use supplied history; empty history is normal before the first fit. '
                        'Do not repeat rejected_hypotheses marked blocked; propose a different approach. '
                        'After baseline, cite numeric observations or previous experiments in rationale and propose '
                        'a change in model logic or preprocessing. Parameter trials belong INSIDE a hypothesis. '
                        'based_on is a saved candidate_id; use based_on_hypothesis_id for hypothesis lineage. '
                        'Declare target_transform, weight_policy and prediction_policy separately from native model_params. '
                        'Runtime owns these operations; never ask the coder to apply them a second time.',
                        dict(state['payload'],lightweight=is_light(state),feedback=errors), HypothesisBatch, role_tools)
                if len(batch.hypotheses)>max_candidates: raise ValueError('Too many hypotheses for this round')
                if batch.agent_notes:
                    import copy
                    for hypothesis in batch.hypotheses:
                        hypothesis.agent_notes['batch'] = copy.deepcopy(batch.agent_notes)
                ids=[h.hypothesis_id for h in batch.hypotheses]
                if len(ids)!=len(set(ids)): raise ValueError('Duplicate hypothesis ID')
                if (state['payload'].get('round_number',1)==1 or state['payload'].get('first_candidate_required')) and not any(h.baseline for h in batch.hypotheses):
                    batch.hypotheses[0].baseline = True
                if planning_mode == 'initial':
                    parent = next(h for h in batch.hypotheses if h.baseline).model_copy(deep=True)
                    parent.kind, parent.model_family = 'family', 'gradient_boosting'
                    parent.family_tools = ['lightgbm', 'catboost', 'xgboost']
                    parent.description = ('Compare fixed-configuration LightGBM, CatBoost and XGBoost '
                        'as the initial tabular baseline. Implement exactly required_tool_id with its own '
                        'native parameters and cheap fold-local preprocessing; preserve the canonical '
                        'task, feature availability, metric, folds and seed. No HPO in these initial fits.')
                    diversity = ' The first cycle must retain diverse boosting models for later ensembles.'
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
        from .search_groups import expand_families, collect_groups, configure_composite, bind_group
        chosen, family_metadata, skipped = expand_families(chosen,tools,context.task.task_type,
            'all' if planning_mode == 'initial' else planning_mode)
        rejections.variants = {key: tool_id for key, (_, tool_id) in family_metadata.items()}

        def owned_group(item, hypothesis):
            if hypothesis.hypothesis_id in family_metadata:
                parent, _ = family_metadata[hypothesis.hypothesis_id]
                item.strategy, item.max_trials = 'single', 1
                return bind_group(item, parent=parent, kind='family')
            if group_parent is not None:
                return bind_group(item, parent=group_parent, kind=group_parent.kind)
            return bind_group(item, kind='ensemble' if hypothesis.kind == 'ensemble' else 'single')

        for hypothesis in chosen:
            feedback=[rejections.record(hypothesis)['error']] if rejections.record(hypothesis) else []
            repair_context={}
            if rejections.blocked(hypothesis):
                rejected.append(dict(hypothesis_id=hypothesis.hypothesis_id, errors=feedback, exhausted=True))
                continue
            for _ in range(max_attempts):
                attempts+=1
                validating=False
                item=None
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
                        'When lightweight=true prefer a cheap fixed strategy=single, max_trials=1 control. '
                        'Choose the actual backend for the task; justify costlier or native search choices. '
                        'An optional initial time estimate is '
                        f'{initial_experiment_seconds}. Never override orchestration resources; '
                        'exclude targets, IDs and evaluation data from features. Always use execution_mode=custom: '
                        'the preprocessing agent implements transforms; the model coder implements the estimator. In lightweight mode '
                        'keep preprocessing cheap and parameters fixed for the initial fit. Declare a small compatible '
                        'search_space even for a single baseline: it is used only after the first CSV is saved. '
                        'Describe required transforms in preprocessing, including column names and prediction '
                        'constraints. The runner executes the confirmed internal validation strategy; '
                        'budget all folds plus a mandatory full-train refit. Do not implement CV in candidate.fit. '
                        'Use strategy=optuna only for an explicitly declared HPO intervention; keep fixed ablations single. '
                        'Declare target_transform, weight_policy and prediction_policy as runtime policies, never native model_params. '
                         'Search-group membership and parent group identifiers are assigned by the controller; '
                         'return only the experiment fields in output_schema. '
                        'Route custom search dimensions as policy_params.<policy>.<field>. '
                        'never implement HPO in candidate.fit. Respect improvement_policy; stacking meta learners require OOF base predictions. '
                        'Backend skills are references, not hidden runtime adapters. The code must consume every '
                        'declared model/search parameter. For text specify a sparse linear estimator with C for '
                        'classification or alpha for regression. '
                        'based_on must be one of known_candidates; hypothesis lineage uses based_on_hypothesis_id.',
                        dict(state['payload'],lightweight=is_light(state),
                             required_tool_id=family_metadata.get(hypothesis.hypothesis_id,(None,None))[1],
                             hypotheses={'hypotheses':[hypothesis.model_dump(mode='json')]},
                             feedback=feedback,repair_context=repair_context), PlanBatch, role_tools)
                    if len(batch.plans)!=1: raise ValueError('Return exactly one plan for the supplied hypothesis')
                    item=HypothesisPlan.model_validate(batch.plans[0].model_dump(mode='json'))
                    if item.hypothesis.hypothesis_id != hypothesis.hypothesis_id:
                        item.agent_notes['requested_hypothesis_id'] = item.hypothesis.hypothesis_id
                        item.hypothesis.hypothesis_id = hypothesis.hypothesis_id
                    item.hypothesis.baseline = hypothesis.baseline
                    item = owned_group(item, hypothesis)
                    if batch.agent_notes:
                        import copy
                        item.agent_notes['batch'] = copy.deepcopy(batch.agent_notes)
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
                        item.feature_set=registered
                    elif item.feature_set is not None:
                        raise ValueError('Planner must select a registered feature_set_id')
                    if state['payload'].get('ensemble_sources'):
                        item.model_params = dict(state['payload'].get('ensemble_recipe',{}), **item.model_params)
                        item=configure_composite(item,state['payload']['ensemble_sources'],
                                                 state['payload'].get('ensemble_weight_trials',15))
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
                    elif item.intervention.kind=='ablation':
                        item.intervention.kind='model_change'
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
                    normalized=normalize_cpu_hints(item)
                    if normalized:
                        emit('planning.normalized_resources',{'parameters':normalized,'cpus':item.cpus})
                    if item.hypothesis.based_on and known_candidates is not None and item.hypothesis.based_on not in known_candidates:
                        alias = item.hypothesis.based_on
                        matches = [key for key, value in (candidate_hypotheses or {}).items() if value == alias and key in known_candidates]
                        item.hypothesis.based_on_hypothesis_id = alias
                        item.hypothesis.based_on = matches[-1] if matches else None
                    report=for_hypothesis(hypothesis,validate_plan,item,context,tools,skills,worker_budget,lightweight=is_light(state),
                                         initial_experiment_seconds=initial_experiment_seconds,
                                         evaluation_reserve_seconds=evaluation_reserve_seconds)
                    if not report.valid: raise ValueError('; '.join(report.errors))
                    approved = progress['accepted'].get(hypothesis.hypothesis_id)
                    if approved:
                        progress['accepted'][hypothesis.hypothesis_id] = owned_group(
                            HypothesisPlan.model_validate(approved), hypothesis).model_dump(mode='json')
                    progress['proposed'][hypothesis.hypothesis_id]=item.model_dump(mode='json');persist()
                    if progress['accepted'].get(hypothesis.hypothesis_id)==item.model_dump(mode='json'):
                        accepted.append(item.model_dump(mode='json'))
                        emit('planning.validation_reused',{'hypothesis_id':hypothesis.hypothesis_id})
                        break
                    instructions=for_hypothesis(hypothesis,skills.load,[],context.task.task_type,role='plan_validator')
                    probes=[]
                    validating=True
                    report=for_hypothesis(hypothesis,generate,'plan_validator',
                        'min_delta is a legacy absolute comparison field, NOT the batched HPO continuation threshold. '
                        'min_delta=0 is correct: the runner separately enforces hpo_gain_threshold=0.005 and keeps tiny gains. '
                        'Review feasibility, leakage and methodological validity against the supplied skill '
                        'instructions. Return a concise validation JSON; never rewrite plans. '
                        'review_scope describes this single-plan call; sibling plans are intentionally absent. '
                        'execution_contract describes the CURRENT SDK, including feature_set augmentation. '
                        'Frozen preprocessing prose is historical intent, not a ban on an active feature_set. '
                        'Use execution_contract for ownership: CPU/thread controls come from runtime FitContext, '
                        'so their absence from model_params is expected and is not a warning. '
                        'You decide methodological approval. Runtime only enforces technical boundaries and budgets. '
                        'Prioritize concrete plan-field contradictions and leakage. Optional improvements such '
                        'as circular encoding or rare-category pooling are not baseline defects without evidence. '
                        'Keep at most three distinct actionable warnings; do not limit real blocking issues. '
                        'Separate preprocessing and model coders implement the candidate. Check that transforms and parameters can '
                        'implement every claim, with no implicit adapter behavior. The runner owns splits and HPO. '
                        'Require explicit operations and columns for changes such as treating numeric years as '
                        'categories. Single baseline search_space is for later HPO. '
                        'Explain blocking issues in errors; issues and source citations are optional. '
                        'Unproven library incompatibilities are warnings. Runtime executes declared target/weight/output policies.',
                        dict(round_number=state['payload'].get('round_number',1),
                             planning_brief=state['payload'].get('planning_brief'),
                             planning_attempt=state['payload'].get('planning_attempt',0),lightweight=is_light(state),
                             task=context.task.model_dump(mode='json') if hasattr(context.task,'model_dump') else
                                  dict(task_type=context.task.task_type,primary_metric=context.task.primary_metric),
                             features=getattr(context,'feature_columns',[]),budget=budget.agent_view(),
                             plans=[item.model_dump(mode='json')],
                             field_sources=plan_field_sources(item),
                             revision_changes=[key for key,value in item.model_dump(mode='json').items()
                                 if repair_context.get('effective_plan') and value!=repair_context['effective_plan'].get(key)],
                             required_tool_id=family_metadata.get(hypothesis.hypothesis_id,(None,None))[1],
                             review_scope=dict(kind='single_plan',tool_id=item.tool_id,
                                 family_id=item.search_group_id if item.search_group_kind=='family' else None,
                                 scheduled_family_tools=[tool for parent,tool in family_metadata.values()
                                     if parent.hypothesis_id==item.search_group_id],
                                 accepted_family_tools=[plan['tool_id'] for plan in progress['accepted'].values()
                                     if item.search_group_kind=='family' and plan.get('search_group_id')==item.search_group_id],
                                 family_coverage_owner='runtime',sibling_plans_in_payload=False),
                             execution_contract=execution_contract(item),
                             capability_probes=probes,
                             hypotheses={'hypotheses':[hypothesis.model_dump(mode='json')]},
                             skill_instructions=[s for s in instructions if s['skill_id']!='candidate_sdk']),ValidationReport)
                    warnings.extend(report.warnings)
                    if not report.valid or report.errors:
                        raise ValueError('; '.join(report.errors) or 'Semantic validator rejected the plan without an explanation')
                    accepted.append(item.model_dump(mode='json'))
                    progress['accepted'][hypothesis.hypothesis_id]=item.model_dump(mode='json');persist()
                    break
                except StructuredOutputError as error:
                    feedback=[str(error)]
                    blocked=rejections.reject(hypothesis,error)
                    if not validating and not blocked:
                        continue
                    rejected.append(dict(hypothesis_id=hypothesis.hypothesis_id,errors=feedback,exhausted=blocked))
                    break
                except (GlobalAPIError, WorkerInfrastructureError):
                    raise
                except LLMUnavailable as error:
                    rejected.append({'hypothesis_id':hypothesis.hypothesis_id,'errors':[str(error)]})
                    break
                except ValueError as error:
                    feedback=[str(error)]
                    if item is not None:
                        repair_context=dict(error=str(error),effective_plan=item.model_dump(mode='json'),
                            execution_contract=execution_contract(item),field_sources=plan_field_sources(item))
                        emit('planning.repair_feedback',repair_context)
                    progress['proposed'].pop(hypothesis.hypothesis_id,None)
                    progress['accepted'].pop(hypothesis.hypothesis_id,None);persist()
                    if rejections.reject(hypothesis,error,item):
                        rejected.append(dict(hypothesis_id=hypothesis.hypothesis_id,errors=feedback,exhausted=True))
                        break
            else: rejected.append({'hypothesis_id':hypothesis.hypothesis_id,'errors':feedback})
        if not accepted and chosen and all(rejections.blocked(h) for h in chosen):
            # The next outer attempt asks for fresh ideas, retaining rejection history.
            progress['hypotheses'] = None
            for h in chosen:
                progress['proposed'].pop(h.hypothesis_id, None)
            persist()
            emit('planning.hypotheses_exhausted', {'rejections': rejections.summaries()})
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
