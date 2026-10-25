"""Build review inputs from execution and measurement records, independently of opinions."""
from .execution_facts import execution_context, related_fit_observations
from .evaluation import comparable_scores, relative_gain


def review_payload(row, rows, *, sources=None, store=None):
    plan, run, evaluation = row['plan'], row['run'], row['evaluation']
    key = run['experiment_id']
    intervention = run.get('intervention') or plan.get('intervention') or {}
    from_feature = plan.get('feature_set_id') and plan.get('feature_set_id') in (
        plan['hypothesis']['hypothesis_id'],plan['hypothesis'].get('based_on_hypothesis_id'))
    inherited_prose = bool(from_feature and not any(plan.get('agent_notes',{}).get(key) for key in
        ('historical_baseline_hypothesis','historical_parent_hypothesis')))
    parent_id = intervention.get('parent_experiment_id')
    parent = next((item['evaluation'] for item in rows if item['run']['experiment_id'] == parent_id), None)
    if parent is None and parent_id and store is not None:
        parent = store.get('evaluation', parent_id)
    incumbent = next((item['evaluation'] for item in rows if item['run'].get('incumbent')), None)
    baseline = next((item['evaluation'] for item in rows if item['run']['experiment_id'] != key and
        item['plan']['hypothesis']['baseline'] and item['plan']['strategy'] == 'single' and
        comparable_scores(evaluation, item['evaluation'])), None)
    reference = parent if intervention.get('kind') == 'hpo' else incumbent or parent or baseline
    common = execution_context(plan, run, evaluation, candidate=row['candidate'],
        related_fits=related_fit_observations(store, key))
    sources = dict(sources or {})
    for program in (plan.get('feature_set') or {}).get('programs', []):
        sources['feature_programs/' + program['name'] + '.py'] = program['source']
    evidence = {f'candidate:{row["candidate"]["candidate_id"]}':row['candidate'],
        f'evaluation:{key}':evaluation, f'plan:{key}':plan,
        f'execution:{key}':dict(execution_context=common, fits=run.get('execution_facts', []))}
    for item in (parent, incumbent, baseline):
        if item and item.get('experiment_id'):
            evidence['evaluation:' + item['experiment_id']] = item
    return dict(hypothesis=plan['hypothesis'], plan=plan, source=sources, evidence=evidence,
        hypothesis_prose_role=('Historical baseline rationale inherited by a controller-selected feature bundle; '
            'the current intervention is the registered feature_set, not restrictions in that old rationale')
            if inherited_prose else 'Current experiment intent',
        code_hash=row['candidate']['code_hash'], parameters=run['best_params'],
        model_spec=run.get('model_spec'), intervention=intervention,
        trial_history=dict(total=len(run.get('trials',[])),
            completed=sum(t.get('status')=='completed' for t in run.get('trials',[])),
            failed=sum(t.get('status')=='failed' for t in run.get('trials',[]))),
        execution_context=common, execution_contract=common['execution_contract'],
        internal_validation={name:run.get(name) for name in ('score_kind','internal_score','validation_protocol_ref',
            'fold_results','score_std','oof_score','oof_ref')},
        stages=dict(smoke='Synthetic API/save/load observation', inner_validation='Inner folds only',
            refit='Outer training partition only', submission_refit='All train; never comparison evidence'),
        execution_report=run.get('execution_report'), smoke_execution_report=row['candidate'].get('execution_report'),
        evaluation=evaluation, comparable_reference=reference,
        references=dict(parent_control=parent, cycle_incumbent=incumbent, baseline=baseline),
        relative_gain=relative_gain(evaluation,reference))
