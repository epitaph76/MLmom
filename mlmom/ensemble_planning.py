"""A separate ensemble proposal, using only frozen sources from completed cycles."""
from pathlib import Path
import shutil

from .agents import GlobalAPIError, LLMUnavailable, role_window
from .resources import WorkerInfrastructureError
from .pipeline_contracts import EnsembleProposal, Hypothesis, HypothesisPlan, ExperimentRun
from .planning import build_planning_graph
from .search_groups import SUPERVISED, component_pool, component_brief, ensemble_recipes
from .model_spec import frozen_plan
from .persistence import tree_hash


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
    pool = [row for row in pool if (tool:=pipeline.tools.get(row['plan']['tool_id'])) and
            context.task.task_type in tool.task_types and
            (tool.available or tool.reason=='adapter probe required')]
    if len(pool) < 2:
        return finish(plans=[], reason='fewer_than_two_compatible_components')
    snapshot = pipeline.budget.snapshot()
    if snapshot.timeouts_enabled and snapshot.research_seconds < pipeline.admission_cost():
        return finish(plans=[], reason='insufficient_complete_ensemble_budget')
    try:
        proposal = pipeline.store.get('ensemble_proposal',key)
        if proposal is None:
            from .tools import ToolSet
            role_tools=ToolSet()
            def read_component_history(offset: int=0, limit: int=12):
                if offset<0 or not 1<=limit<=12:raise ValueError('Use offset>=0 and limit in 1..12')
                return dict(components=[component_brief(row) for row in pool[offset:offset+limit]],total=len(pool))
            role_tools.register('read_component_history',read_component_history,'Read a page of verified component metadata from completed cycles.')
            preview=pool[:12]
            champion_id=(state.get('champion') or {}).get('run',{}).get('experiment_id')
            anchor=next((row for row in pool if row['experiment_id']==champion_id),None)
            if anchor and anchor not in preview:preview=[anchor,*preview[:11]]
            with role_window(pipeline.client, 'ensemble_agent', key, pipeline.config.ensemble_agent_seconds):
                proposal = pipeline.client.generate('ensemble_agent',
                    'Select two or three complementary verified SINGLE model experiments for one ensemble hypothesis. '
                    'Use only supplied experiment IDs. No recursive ensembles. Consider diversity, evidence and cost; '
                    'historical scores with different split IDs are not comparable. Return skip=true with a reason '
                    'when no defensible composition exists. Do not train, write code or select on current holdout labels.',
                    dict(task=context.task.model_dump(mode='json'), budget=snapshot.model_dump(),
                         components=[component_brief(row) for row in preview],total_components=len(pool),
                         skill_instructions=pipeline.skills.load([],context.task.task_type,role='ensemble_agent')),
                    EnsembleProposal, role_tools)
            pipeline.store.put('ensemble_proposal',key,proposal)
        proposal = EnsembleProposal.model_validate(proposal.model_dump() if hasattr(proposal,'model_dump') else proposal)
        if proposal.skip:
            return finish(plans=[],reason=proposal.rationale)
        available = {r['experiment_id']:r for r in pool}
        if any(key not in available for key in proposal.component_experiment_ids):
            raise ValueError('Ensemble agent selected an unregistered experiment')
        sources = []
        for index, experiment_id in enumerate(proposal.component_experiment_ids):
            row = available[experiment_id]
            plan = frozen_plan(HypothesisPlan.model_validate(row['plan']),ExperimentRun.model_validate(row['run']))
            sources.append(dict(name=f'base_{index}', candidate_id=row['candidate']['candidate_id'],
                experiment_id=experiment_id, code_hash=row['candidate']['code_hash'], config=plan.model_dump(mode='json')))
        hypothesis = Hypothesis(hypothesis_id=key+'-ensemble',kind='ensemble',
            description='Compare combinations of complementary frozen model implementations',rationale=proposal.rationale)
        payload = pipeline.planning_payload(state)
        payload.update(ensemble_sources=sources,ensemble_weight_trials=pipeline.config.hpo_max_trials)
        progress = pipeline.store.get('ensemble_planning_progress',key) or {}
        graph = build_planning_graph(pipeline.client,context,pipeline.tools,pipeline.skills,snapshot,
            max_attempts=pipeline.config.max_attempts,max_candidates=1,lightweight=False,
            probe_tool=pipeline.probe_ml_tool,evaluation_reserve_seconds=2*pipeline.config.prediction_seconds,
            planner_seconds=pipeline.config.planner_seconds,validator_seconds=pipeline.config.validator_seconds,
            saved_progress=progress,save_progress=lambda p:pipeline.store.put('ensemble_planning_progress',key,p))
        result = graph.invoke(dict(payload=payload,hypotheses={'hypotheses':[hypothesis.model_dump(mode='json')]}),
                              {'recursion_limit':100})
        if not result.get('valid'):
            return finish(plans=[],reason='ensemble_plan_rejected',errors=result.get('errors',[]))
        group = ensemble_recipes(HypothesisPlan.model_validate(result['plans'][0]),context.task.task_type)
        pipeline.store.put('search_group',key+'-'+group.hypothesis.hypothesis_id,group)
        return finish(plans=[p.model_dump(mode='json') for p in group.variants],reason='planned')
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
