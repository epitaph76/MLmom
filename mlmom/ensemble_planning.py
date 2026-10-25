"""A separate ensemble proposal, using only frozen sources from completed cycles."""
from pathlib import Path
from concurrent.futures import CancelledError
import shutil

from .agents import GlobalAPIError, LLMUnavailable, StructuredOutputError, role_window
from .resources import WorkerInfrastructureError
from .pipeline_contracts import EnsembleProposal, Hypothesis, HypothesisPlan, ExperimentRun, Champion
from .planning_context import champion_snapshot
from .planning import build_planning_graph
from .search_groups import SUPERVISED, component_pool, component_brief, ensemble_recipes
from .model_spec import frozen_plan
from .persistence import tree_hash
from .ensemble_evidence import diversity_evidence


def plan_ensemble(pipeline, state):
    if getattr(pipeline,'context',None) is None:
        return dict(plans=[],reason='no_confirmed_context')
    key = f"r{state['round_number']}"
    saved = pipeline.store.get('ensemble_planning', key)
    if saved is not None:
        return saved
    def finish(**result):
        pipeline.store.put('ensemble_planning', key, result)
        return result
    if state['round_number'] < 2 or not pipeline.config.ensemble_search_enabled:
        return finish(plans=[], reason='ensemble_search_disabled_or_first_cycle')
    context = pipeline.context
    if context.task.task_type not in SUPERVISED:
        return finish(plans=[], reason='unsupported_task_type')
    pool = component_pool(pipeline.store, state['round_number'])
    if len(pool) < 2:
        return finish(plans=[], reason='fewer_than_two_compatible_components')
    snapshot = pipeline.budget.snapshot()
    if snapshot.timeouts_enabled and snapshot.research_seconds <= .5:
        return finish(plans=[], reason='research_deadline')
    available = {row['experiment_id']:row for row in pool}
    def measure_components(experiment_ids, methods=(), oof_folds=3):
        if (not 2<=len(experiment_ids)<=3 or len(set(experiment_ids))!=len(experiment_ids)
                or any(k not in available for k in experiment_ids)):
            raise ValueError('Use two or three distinct supplied component experiment IDs')
        evidence=diversity_evidence([available[k] for k in experiment_ids],context,
            methods=methods,oof_folds=oof_folds)
        from hashlib import sha256
        evidence_key=key+':'+sha256('|'.join(experiment_ids).encode()).hexdigest()[:16]
        evidence['reference']='ensemble_diversity:'+evidence_key
        pipeline.store.put('ensemble_diversity',evidence_key,evidence)
        return evidence
    def compare_oof_components(experiment_ids: list[str]):
        return measure_components(experiment_ids)
    try:
        proposal = pipeline.store.get('ensemble_proposal',key)
        if proposal is None:
            from .tools import ToolSet
            role_tools=ToolSet()
            def read_component_history(offset: int=0, limit: int=12):
                if offset<0 or not 1<=limit<=12:raise ValueError('Use offset>=0 and limit in 1..12')
                return dict(components=[component_brief(row) for row in pool[offset:offset+limit]],total=len(pool))
            role_tools.register('read_component_history',read_component_history,'Read a page of saved component metadata from completed cycles.')
            role_tools.register('compare_oof_components',compare_oof_components,
                'Diversity Evidence for two or three supplied frozen components: aligned historical inner OOF error correlation/overlap, fixed equal-mean benefit under the primary metric, and advisory refit costs. Returns explicit unavailable reasons; never trains or reads current holdout labels.')
            preview=pool[:12]
            champion_id=(state.get('champion') or {}).get('run',{}).get('experiment_id')
            anchor=next((row for row in pool if row['experiment_id']==champion_id),None)
            if anchor and anchor not in preview:preview=[anchor,*preview[:11]]
            with role_window(pipeline.client, 'ensemble_agent', key, pipeline.config.ensemble_agent_seconds):
                feedback = None
                for attempt in range(pipeline.config.max_attempts):
                    try:
                        proposal = pipeline.client.generate('ensemble_agent',
                            'Select two or three complementary saved SINGLE model experiments for one ensemble hypothesis. '
                            'Use only supplied experiment IDs. No recursive ensembles. Consider diversity, evidence and cost; '
                            'historical scores with different split IDs are not comparable. Return skip=true with a reason '
                            'when appropriate. Source duration_seconds is historical experiment wall time, possibly a whole '
                            'HPO study; do not treat its sum as frozen-refit wall time. Actual resource scheduling owns overlap. '
                            'Use compare_oof_components for shortlisted components when OOF is registered. '
                            'Return agent_notes.diversity_evidence with measurement reference, interpretation, unknowns and '
                            'decision_status=unverified_hypothesis unless the specific claim is measured. '
                            'Correlation or historical OOF blend gain does not prove an independent gain of weighted/stack recipes. '
                            'Explicitly select methods (mean, median, weighted, stack), '
                            'oof_folds and weight_trials when relevant, plus component_experiment_ids, rationale and skip; '
                            'put any skip explanation in rationale. Do not train, write code or select on current holdout labels.',
                            dict(task=context.task.model_dump(mode='json'), budget=snapshot.agent_view(),
                                 champion=champion_snapshot(Champion.model_validate(state['champion']),pipeline.store) if state.get('champion') else None,
                                 components=[component_brief(row) for row in preview],total_components=len(pool),
                                 diversity_evidence=dict(status='unverified_hypothesis',
                                     tool='compare_oof_components',scope='historical_inner_oof_diagnostic',
                                     missing_data_policy='Explain unknowns; a hypothesis may still be tested'),
                                 validation_feedback=feedback,
                                 skill_instructions=pipeline.skills.load([],context.task.task_type,role='ensemble_agent')),
                            EnsembleProposal, role_tools)
                        proposal = EnsembleProposal.model_validate(proposal.model_dump() if hasattr(proposal,'model_dump') else proposal)
                        break
                    except (StructuredOutputError, ValueError) as error:
                        if attempt + 1 == pipeline.config.max_attempts:
                            raise
                        feedback = str(error)[:2000]
            pipeline.store.put('ensemble_proposal',key,proposal)
        proposal = EnsembleProposal.model_validate(proposal.model_dump() if hasattr(proposal,'model_dump') else proposal)
        if proposal.skip:
            return finish(plans=[],reason=proposal.rationale)
        if any(key not in available for key in proposal.component_experiment_ids):
            raise ValueError('Ensemble agent selected an unregistered experiment')
        evidence=measure_components(proposal.component_experiment_ids,proposal.methods,proposal.oof_folds)
        sources = []
        for index, experiment_id in enumerate(proposal.component_experiment_ids):
            row = available[experiment_id]
            plan = frozen_plan(HypothesisPlan.model_validate(row['plan']),ExperimentRun.model_validate(row['run']))
            sources.append(dict(name=f'base_{index}', candidate_id=row['candidate']['candidate_id'],
                experiment_id=experiment_id, code_hash=row['candidate']['code_hash'], config=plan.model_dump(mode='json')))
        parent = Hypothesis(hypothesis_id=key+'-ensemble',kind='ensemble',
            description='Agent-selected recipes of complementary frozen models',rationale=proposal.rationale)
        plans, errors = [], []
        for method in dict.fromkeys(proposal.methods):
            try:
                hypothesis = parent.model_copy(update={'hypothesis_id':key+'-ensemble-'+method,
                    'description':f'Fit the selected {method} recipe with frozen components'})
                payload = pipeline.planning_payload(state)
                recipe = dict(method=method)
                if method in {'weighted','stack'}: recipe['oof_folds']=proposal.oof_folds
                if method == 'weighted': recipe['weight_trials']=proposal.weight_trials
                payload.update(ensemble_sources=sources,ensemble_recipe=recipe,
                               diversity_evidence=evidence,
                               ensemble_weight_trials=proposal.weight_trials)
                progress_key = key+':'+method
                progress = pipeline.store.get('ensemble_planning_progress',progress_key) or {}
                graph = build_planning_graph(pipeline.client,context,pipeline.tools,pipeline.skills,snapshot,
                    max_attempts=pipeline.config.max_attempts,max_candidates=1,lightweight=False,
                    evaluation_reserve_seconds=2*pipeline.config.prediction_seconds,
                    planner_seconds=pipeline.config.planner_seconds,validator_seconds=pipeline.config.validator_seconds,
                    group_parent=parent,
                    saved_progress=progress,save_progress=lambda value,k=progress_key:pipeline.store.put('ensemble_planning_progress',k,value))
                result = graph.invoke(dict(payload=payload,hypotheses={'hypotheses':[hypothesis.model_dump(mode='json')]}),
                                      {'recursion_limit':100})
                if not result.get('valid'):
                    errors.extend(dict(error, method=method) for error in result.get('errors',[]))
                    continue
                plans.append(result['plans'][0])
            except (GlobalAPIError, WorkerInfrastructureError, CancelledError):
                raise
            except Exception as error:
                failure = dict(method=method, error=str(error)[:2000], error_type=type(error).__name__)
                pipeline.store.event('ensemble_recipe_failed',key+':'+method,failure)
                errors.append(failure)
        return finish(plans=plans,reason='planned' if plans else 'ensemble_plan_rejected',errors=errors,
                      diversity_evidence=evidence)
    except (GlobalAPIError, WorkerInfrastructureError):
        raise
    except (LLMUnavailable,ValueError) as error:
        return finish(plans=[],reason='ensemble_planning_failed',error=str(error)[:2000])


def bundle_components(plan, store, workspace):
    for source in plan.ensemble_sources:
        candidate = store.get('candidate',source['candidate_id'])
        if not candidate or candidate['code_hash'] != source['code_hash']:
            raise ValueError('Missing or changed ensemble component candidate')
        root = Path(candidate['code_dir'])
        if tree_hash(root) != source['code_hash']:
            raise ValueError('Ensemble component source was modified')
        destination = Path(workspace)/'components'/source['name']
        destination.mkdir(parents=True,exist_ok=True)
        for file in root.rglob('*.py'):
            target = destination/file.relative_to(root)
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.is_file():
                shutil.copyfile(file,target)
        if tree_hash(destination) != source['code_hash']:
            raise ValueError('Bundled component source was modified')
