"""One resumable, train-only feature session per improvement cycle."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .agents import GlobalAPIError, role_window
from .resources import WorkerInfrastructureError
from .process_identity import worker_record, recorded_process_matches
from .feature_contracts import (FeatureSet, FeatureProposal, FeatureSession, FeatureEvidence,
                               FeatureExperimentPlan, feature_hash)
from .feature_engineering import validate_features
from .persistence import atomic_json, file_hash, tree_hash
from .execution import config_hash
from .model_spec import frozen_plan, Intervention
from .pipeline_contracts import CandidateVersion, HypothesisPlan, ExperimentRun, TaskContext
from .splitting import inner_protocol, feature_context
from .tracing import emit


def feature_session_key(round_id='r1'):
    # Keep the first-round identity compatible with already checkpointed runs.
    return 'initial' if round_id == 'r1' else round_id


def session_directory(run_dir, session_id):
    root = run_dir / 'feature_search'
    return root if session_id == 'initial' else root / session_id


def session_object_key(session_id, identity):
    return identity if session_id == 'initial' else session_id + ':' + identity


def select_feature_baseline(pipeline, champion, trained, round_number):
    """Use the champion, or a verified single source if an ensemble is leading."""
    from types import SimpleNamespace
    from .search_groups import internal_winner, component_pool

    def eligible(row):
        plan, run, candidate = row['plan'], row['run'], row['candidate']
        return (not plan.get('ensemble_components') and not plan.get('ensemble_sources') and
                plan['tool_id'] not in {'flaml', 'autogluon', 'lightautoml'} and run.get('model_spec') and
                (Path(candidate['code_dir']) / 'preprocessing.py').is_file())

    if eligible(champion.model_dump(mode='json')):
        return champion.model_copy(deep=True)
    # Only compare scores measured on this cycle's train. Historical scores are
    # never ranked against current scores or against scores from other splits.
    current = [row for row in trained.values() if eligible(row)]
    selected = internal_winner(current, pipeline.context.task.metric_direction)
    if selected is None:
        sources = {s['experiment_id'] for s in champion.plan.ensemble_sources}
        pool = [row for row in component_pool(pipeline.store, round_number) if eligible(row)]
        selected = next((row for row in pool if row['experiment_id'] in sources), None)
        selected = selected or next(iter(pool), None)
    if selected is None:
        # The coordinator will persist an explicit unsupported-baseline reason.
        return champion.model_copy(deep=True)
    return SimpleNamespace(candidate=CandidateVersion.model_validate(selected['candidate']),
        plan=HypothesisPlan.model_validate(selected['plan']), run=ExperimentRun.model_validate(selected['run']))


def feature_task_view(task):
    """Keep formal requirements; initial full-data EDA prose is not feature evidence."""
    return task.model_dump(mode='json', exclude={'evidence', 'assumptions', 'risks'})


def dictionary_metadata(context):
    """Reuse declared meanings/units, without copying full-data profile statistics."""
    if not context.profile_ref or not Path(context.profile_ref).is_file():
        return []
    profile=json.loads(Path(context.profile_ref).read_text(encoding='utf-8'))
    return [{k:column.get(k) for k in ('name','description','unit','declared_dtype')}
            for column in profile.get('columns',[]) if column['name'] in context.feature_columns]


def feature_profile(request):
    """Numerical observations computed in a worker from outer train and inner OOF only."""
    from .inputs import read_table, features, targets
    context = TaskContext.model_validate(request['context'])
    frame = read_table(request['train_path'])
    X, y = features(frame, context), targets(frame, context)
    columns = []
    for name in X.columns[:100]:
        values = X[name]
        item = dict(name=name, dtype=str(values.dtype), missing=int(values.isna().sum()),
                    unique=int(values.nunique(dropna=True)))
        if pd.api.types.is_numeric_dtype(values):
            finite = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
            item['range'] = [float(finite.min()), float(finite.max())] if len(finite) else None
        columns.append(item)
    report = dict(rows=len(X), columns=columns, residual_slices=[],
                  error_weighting='unweighted', residual_scope='feature_inner_oof')
    if request.get('oof_ref'):
        oof = json.loads(Path(request['oof_ref']).read_text(encoding='utf-8'))
        indices = np.asarray(oof['indices'], dtype=int)
        prediction = np.asarray(oof['predictions'])
        if prediction.ndim == 2 and prediction.shape[1] == 1:
            prediction = prediction[:, 0]
        truth = np.asarray(y.iloc[indices])
        if context.task.task_type == 'regression':
            residual = truth - prediction
            error = np.abs(residual)
        else:
            residual = None
            if prediction.ndim == 2:
                classes = list(context.classes)
                error = 1 - prediction[np.arange(len(truth)), [classes.index(t) for t in truth]]
            elif oof['prediction_kind'] == 'probability':
                error = np.abs((truth == context.classes[-1]).astype(float) - prediction)
            else:
                error = (truth != prediction).astype(float)
        for name in X.columns[:30]:
            values = X.iloc[indices][name].reset_index(drop=True)
            if pd.api.types.is_numeric_dtype(values):
                numeric = pd.to_numeric(values, errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan)
                values = pd.qcut(numeric, 4, duplicates='drop').astype(str) if numeric.notna().any() else pd.Series('__missing__',index=values.index)
            else:
                values = values.astype(object).where(values.notna(), '__missing__').astype(str)
            for label in values.value_counts().head(8).index:
                mask = (values == label).to_numpy()
                if mask.sum() < max(5, math.ceil(.01 * len(X))):
                    continue
                item = dict(column=name, slice=str(label)[:100], rows=int(mask.sum()),
                            mean_error=float(np.mean(error[mask])))
                if residual is not None:
                    item['mean_residual'] = float(np.mean(residual[mask]))
                report['residual_slices'].append(item)
        report['residual_slices'] = sorted(report['residual_slices'], key=lambda r: -r['mean_error'])[:20]
    return dict(status='completed', profile=report)


def fixed_model_hash(spec):
    return config_hash({k: v for k, v in spec.items() if k != 'feature_set_hash'})


def verified_scores(result, protocol, expected_model_hash):
    if result.get('status') != 'completed' or not result.get('full_refit'):
        raise ValueError('Incomplete feature evaluation/refit')
    if result.get('validation_scope') != 'feature' or result.get('score_kind') != 'feature_inner_cv_mean':
        raise ValueError('Feature metric has the wrong evaluation scope')
    if fixed_model_hash(result.get('model_spec') or {}) != expected_model_hash:
        raise ValueError('Feature comparison changed the frozen model or policies')
    report = result.get('execution_report') or {}
    if report.get('status') != 'passed' or report.get('errors'):
        raise ValueError('Feature comparison has no passed execution gate')
    saved = json.loads(Path(result['validation_protocol_ref']).read_text(encoding='utf-8'))
    if saved != protocol:
        raise ValueError('Feature comparison changed fold indices')
    folds = result.get('fold_results', [])
    if len(folds) != protocol['n_splits'] or [f['number'] for f in folds] != list(range(protocol['n_splits'])):
        raise ValueError('Feature comparison has incomplete folds')
    if any(f.get('status') != 'completed' or f.get('execution_status') != 'passed' for f in folds):
        raise ValueError('Feature comparison has failed folds')
    scores = np.asarray([f['score'] for f in folds], dtype=float)
    if not np.isfinite(scores).all() or not np.isclose(result['internal_score'], scores.mean(), rtol=1e-12, atol=1e-12):
        raise ValueError('Feature comparison has inconsistent metrics')
    oof = json.loads(Path(result['oof_ref']).read_text(encoding='utf-8'))
    expected = sorted(i for f in protocol['folds'] for i in f['validation_indices'])
    expected_source = sorted(i for f in protocol['folds'] for i in f['source_validation_indices'])
    if oof['indices'] != expected or sorted(oof['source_indices']) != expected_source or len(oof['predictions']) != len(expected):
        raise ValueError('Feature comparison has incomplete OOF coverage')
    return scores


def paired_evidence(candidate, reference, protocol, expected_model_hash, direction, min_delta=0):
    left = verified_scores(candidate, protocol, expected_model_hash)
    right = verified_scores(reference, protocol, expected_model_hash)
    from .execution import _leaves
    def actual_configuration(fold):
        report=json.loads(Path(fold['execution_report_ref']).read_text(encoding='utf-8'))
        observed={}
        for name,component in _leaves(report['components']):
            entry=component.get('fit_entry')
            if not entry or not entry.get('component_type'):
                raise ValueError('Missing actual feature-reference estimator configuration')
            observed[name]=dict(component_type=entry['component_type'],params=entry['params'])
        return observed
    for current,control in zip(candidate['fold_results'],reference['fold_results']):
        if actual_configuration(current) != actual_configuration(control):
            raise ValueError('Feature comparison changed actual estimator class or constructor parameters')
    deltas = (left - right) * (1 if direction == 'maximize' else -1)
    if not np.isfinite(deltas).all():
        raise ValueError('Nonfinite feature comparison delta')
    mean = float(deltas.mean())
    stable = int((deltas > 0).sum()) >= math.ceil(2 * len(deltas) / 3)
    status = 'accepted' if mean > min_delta and stable else 'quarantine' if mean > min_delta else 'rejected'
    return status, mean, deltas.tolist()


def run_feature_session(pipeline, baseline, split, train_path):
    """Return at most one selected row. This function never reads an outer evaluation."""
    p = pipeline
    round_id = split.round_id if split is not None else 'r1'
    session_id = feature_session_key(round_id)
    object_key = lambda identity: session_object_key(session_id, identity)
    saved = p.store.get('feature_session', session_id)
    if not p.config.feature_search_enabled:
        return None
    if saved and saved['status'] in {'selected', 'completed', 'skipped'}:
        return saved.get('winner')
    context = p.context
    directory = session_directory(p.run_dir, session_id)
    directory.mkdir(parents=True, exist_ok=True)
    plan = frozen_plan(baseline.plan, baseline.run)
    plan.strategy, plan.search_space, plan.max_trials = 'single', {}, 1
    plan.stopping_policy, plan.component_stopping_policies = 'fixed', {}
    plan.search_group_kind, plan.search_group_id = 'single', None
    plan.hypothesis.baseline = False
    plan.hypothesis.based_on = baseline.candidate.candidate_id
    baseline_feature_set = plan.feature_set or FeatureSet(feature_set_id='empty')
    baseline_feature_set_id = baseline_feature_set.feature_set_id
    spec = baseline.run.model_spec or {}
    model_hash = fixed_model_hash(spec)
    train_hash = file_hash(Path(train_path))
    protocol = {}
    try:
        if context.task.task_type not in {'regression', 'binary_classification', 'multiclass_classification'}:
            raise ValueError('Task is outside feature DSL v1')
        if plan.ensemble_components or plan.ensemble_sources or plan.tool_id in {'flaml', 'autogluon', 'lightautoml'}:
            raise ValueError('Feature search requires a single ordinary baseline')
        if not spec:
            raise ValueError('Feature search requires a verified baseline')
        if not (Path(baseline.candidate.code_dir) / 'preprocessing.py').is_file():
            raise ValueError('Feature search requires separate preprocessing.py')
        frame = p.frame.iloc[split.train_indices]
        protocol = inner_protocol(frame, feature_context(context, p.config.feature_holdout_folds),
            fraction=p.config.inner_fraction, seed=split.seed + 1, outer_split_id=split.split_id,
            source_indices=split.train_indices)
        if protocol['n_splits'] < 2:
            raise ValueError('Feature stability requires multiple folds')
    except ValueError as error:
        session = FeatureSession(session_id=session_id, round_id=round_id,
            baseline_candidate_id=baseline.candidate.candidate_id, model_hash=model_hash,
            train_hash=train_hash, protocol_hash=config_hash(protocol), protocol_ref='', status='skipped', reason=str(error))
        p.store.put('feature_session', session_id, session)
        atomic_json(directory / 'session.json', session)
        emit('feature.skipped', dict(session_id=session_id, round_id=round_id, reason=str(error)))
        return None
    protocol_hash = config_hash(protocol)
    protocol_ref = directory / 'folds.json'
    if saved:
        session = FeatureSession.model_validate(saved)
        if (session.baseline_candidate_id != baseline.candidate.candidate_id or session.train_hash != train_hash or
                session.model_hash != model_hash or session.protocol_hash != protocol_hash):
            raise ValueError('Feature resume differs from the original baseline, data or folds')
    else:
        atomic_json(protocol_ref, protocol)
        session = FeatureSession(session_id=session_id, round_id=round_id,
            baseline_candidate_id=baseline.candidate.candidate_id, model_hash=model_hash,
            train_hash=train_hash, protocol_hash=protocol_hash, protocol_ref=str(protocol_ref),
            baseline_feature_set_id=baseline_feature_set_id, best_feature_set_id=baseline_feature_set_id,
            attempted_hashes=([config_hash(sorted(f.expression_key() for f in baseline_feature_set.features))]
                              if baseline_feature_set.features else ['empty']),
            max_configurations=p.config.feature_max_configurations, plateau_limit=p.config.feature_plateau_limit)
        p.store.put('feature_set', 'empty', FeatureSet(feature_set_id='empty'))
        p.store.put('feature_set', baseline_feature_set_id, baseline_feature_set, immutable=True)
        emit('feature.started', dict(session_id=session_id, round_id=round_id,
            baseline_candidate_id=session.baseline_candidate_id, baseline_feature_set_id=baseline_feature_set_id,
            protocol_hash=protocol_hash))

    def save():
        p.store.put('feature_session', session_id, session)
        atomic_json(directory / 'session.json', session)

    save()

    def evaluate(identity, current_plan):
        root = directory / identity
        root.mkdir(exist_ok=True)
        result_path = root / 'result.json'
        if result_path.exists():
            cached = json.loads(result_path.read_text(encoding='utf-8'))
            if cached.get('stop_reason') not in {'parent_exit', 'controller_closed'}:
                return cached
        if tree_hash(Path(baseline.candidate.code_dir)) != baseline.candidate.code_hash:
            raise ValueError('Frozen feature baseline source was modified')
        request = dict(action='train', code_dir=baseline.candidate.code_dir, code_hash=baseline.candidate.code_hash,
            plan=current_plan.model_dump(mode='json'), context=p.sanitized_context(train_path), train_path=str(train_path),
            seed=split.seed, inner_fraction=p.config.inner_fraction, outer_split_id=split.split_id,
            outer_train_indices=split.train_indices, training_control=False, validation_scope='feature',
            feature_protocol=protocol, feature_holdout_folds=p.config.feature_holdout_folds)
        limits=dict(seconds=min(plan.time_limit_seconds, p.budget.snapshot().research_seconds),
                    cpus=plan.cpus, memory_gb=plan.memory_gb)
        from .resources import SandboxManager, terminate_tree, require_worker_infrastructure
        if isinstance(p.sandbox, SandboxManager):
            import psutil
            worker_key='feature:'+object_key(identity)
            previous_worker=p.store.get('worker',worker_key)
            if previous_worker:
                try:
                    if recorded_process_matches(previous_worker):
                        terminate_tree(previous_worker['pid'])
                except psutil.Error:
                    pass
            job=p.sandbox.start(request,root,**limits)
            try:
                p.store.put('worker',worker_key,worker_record(job.process.pid))
            except psutil.Error:
                pass
            while job.poll() is None:
                p.sandbox.reap()
                time.sleep(.05)
            result=job.finish()
        else:
            result = p.sandbox.run(request, root, **limits)
        # Sandbox fakes and interruption failures need the same durable boundary as real workers.
        atomic_json(result_path, result)
        require_worker_infrastructure(result, root)
        return result

    try:
        if p.budget.snapshot().research_seconds < 1:
            raise ValueError('No feature evaluation budget remains')
        reference = evaluate('reference', plan)
        verified_scores(reference, protocol, model_hash)
    except (GlobalAPIError, WorkerInfrastructureError):
        raise
    except Exception as error:
        session.status, session.reason = 'skipped', 'Reference unavailable: ' + str(error)
        save()
        return None
    if session.best_result is None:
        session.best_result = reference
        save()

    while session.pending is not None or (session.attempts < session.max_configurations and session.stagnant < session.plateau_limit):
        if p.budget.snapshot().research_seconds < 1:
            session.reason = 'research_budget_exhausted'
            break
        if session.pending is None:
            if session.proposal_calls >= session.max_configurations * p.config.max_attempts:
                session.reason = 'proposal_limit'
                break
            key = str(session.proposal_calls)
            proposal_key = object_key(key)
            draft = p.store.get('feature_proposal', proposal_key)
            try:
                if draft is None:
                    profile_root = directory / ('profile-' + key)
                    if (profile_root / 'result.json').exists():
                        profile = json.loads((profile_root / 'result.json').read_text(encoding='utf-8'))
                    else:
                        profile = p.sandbox.run(dict(action='feature_profile', context=p.sanitized_context(train_path),
                            train_path=str(train_path), oof_ref=session.best_result.get('oof_ref')), profile_root,
                            seconds=min(p.config.reconnaissance_query_seconds, p.budget.snapshot().research_seconds),
                            cpus=plan.cpus, memory_gb=plan.memory_gb)
                        atomic_json(profile_root / 'result.json', profile)
                    if profile.get('status') != 'completed':
                        raise ValueError('Train-only feature profile is unavailable')
                    payload = dict(task=feature_task_view(context.task),
                        dictionary_metadata=dictionary_metadata(context),
                        task_description=Path(p.files.task).read_text(encoding='utf-8-sig'),
                        train_profile=profile['profile'],
                        current_feature_set=p.store.get('feature_set', session.best_feature_set_id),
                        history=[dict(e,definition=p.store.get('feature_set',e['feature_set_id']))
                                 for e in p.store.list('feature_evidence')
                                 if e.get('session_id', 'initial') == session_id],
                        proposal_errors=[e for e in p.store.list('feature_proposal_error')
                                         if e.get('session_id', 'initial') == session_id][-3:],
                        remaining_configurations=session.max_configurations-session.attempts,
                        skill_instructions=p.skills.load([], context.task.task_type, role='feature_agent'))
                    with role_window(p.client, 'feature_agent', session_id + ':' + key,
                                     min(p.config.feature_agent_seconds,p.budget.snapshot().research_seconds)):
                        draft = p.client.generate('feature_agent', FEATURE_PROMPT, payload, FeatureProposal).model_dump(mode='json')
                    p.store.put('feature_proposal', proposal_key, draft, immutable=True)
                proposal = FeatureProposal.model_validate(draft)
                session.proposal_calls += 1
                save()
                if proposal.action == 'stop':
                    session.reason = proposal.rationale
                    break
                parent = FeatureSet.model_validate(p.store.get('feature_set', session.best_feature_set_id))
                if proposal.action == 'remove' and not set(proposal.remove) <= {f.name for f in parent.features}:
                    raise ValueError('Removal references an unknown feature')
                members = ([*parent.features, *proposal.features] if proposal.action == 'add' else
                           [f for f in parent.features if f.name not in proposal.remove])
                definition = FeatureSet(feature_set_id='pending', features=members)
                validate_features(definition, context.feature_columns)
                fingerprint = config_hash(sorted(f.expression_key() for f in definition.features)) if definition.features else 'empty'
                if fingerprint in session.attempted_hashes:
                    raise ValueError('Equivalent feature configuration was already tested')
                definition.feature_set_id = 'fe-' + fingerprint[:16] if definition.features else 'empty'
                existing = p.store.get('feature_set', definition.feature_set_id)
                if existing:
                    # Equivalent formulas keep their immutable, previously assigned names.
                    definition = FeatureSet.model_validate(existing)
                p.store.put('feature_set', definition.feature_set_id, definition, immutable=True)
                session.pending = dict(feature_set_id=definition.feature_set_id, fingerprint=fingerprint,
                    parent_feature_set_id=parent.feature_set_id, reference=session.best_result,
                    rationale=proposal.rationale)
                # Commit the reservation BEFORE smoke or full training can start.
                session.attempts += 1
                session.attempted_hashes.append(fingerprint)
                save()
            except (GlobalAPIError, WorkerInfrastructureError):
                raise
            except Exception as error:
                session.proposal_calls = max(session.proposal_calls, int(key) + 1)
                p.store.put('feature_proposal_error', proposal_key,
                    dict(session_id=session_id, reason=str(error)[:1500], proposal=draft))
                save()
                continue
        pending = session.pending
        definition = FeatureSet.model_validate(p.store.get('feature_set', pending['feature_set_id']))
        candidate_plan = plan.model_copy(deep=True)
        candidate_plan.feature_set_id, candidate_plan.feature_set = definition.feature_set_id, definition
        candidate_plan.hypothesis.hypothesis_id = definition.feature_set_id
        candidate_plan.hypothesis.description = pending['rationale']
        candidate_plan.intervention = Intervention(kind='ablation', base_candidate_id=baseline.candidate.candidate_id,
            changed_fields=['feature_set'], operations=[pending['rationale']],
            preserved_conditions=['estimator source', 'exported model parameters', 'policies', 'folds', 'seed'])
        status, mean, deltas = 'failed', None, []
        baseline_mean, baseline_deltas = None, []
        result = {}
        try:
            evidence_key = object_key(definition.feature_set_id)
            assignment = p.store.get('feature_experiment_plan', evidence_key)
            if assignment is None:
                payload = dict(task=feature_task_view(context.task),
                    feature_set=definition.model_dump(mode='json'),
                    reference_feature_set_id=pending['parent_feature_set_id'], hypothesis=pending['rationale'],
                    frozen_model=spec, validation_scope='feature', protocol_hash=protocol_hash,
                    skill_instructions=p.skills.load([], context.task.task_type, role='planner'))
                with role_window(p.client, 'planner', 'feature:' + session_id + ':' + definition.feature_set_id,
                                 min(p.config.planner_seconds,p.budget.snapshot().research_seconds)):
                    assignment = p.client.generate('planner',
                        'Frame this single feature intervention against the supplied reference. Return FeatureExperimentPlan. '
                        'Preserve feature_set_id and reference_feature_set_id exactly. The runtime fixes model source, '
                        'exported parameters, policies, folds and seeds, and disables HPO. Do not introduce another change. '
                        'These are internal train-only experiments, with no external evidence.', payload,
                        FeatureExperimentPlan).model_dump(mode='json')
                assignment = FeatureExperimentPlan.model_validate(assignment)
                if (assignment.feature_set_id != definition.feature_set_id or
                        assignment.reference_feature_set_id != pending['parent_feature_set_id']):
                    raise ValueError('Feature Planner changed the paired assignment')
                assignment = assignment.model_dump(mode='json')
                p.store.put('feature_experiment_plan', evidence_key, assignment, immutable=True)
            result = evaluate(definition.feature_set_id, candidate_plan)
            status, mean, deltas = paired_evidence(result, pending['reference'], protocol, model_hash,
                context.task.metric_direction, p.config.feature_min_delta)
            baseline_status, baseline_mean, baseline_deltas = paired_evidence(result, reference, protocol, model_hash,
                context.task.metric_direction, p.config.feature_min_delta)
            if status == 'accepted' and baseline_status != 'accepted':
                status = baseline_status
            reason = 'Paired fold evidence: ' + status
        except (GlobalAPIError, WorkerInfrastructureError):
            raise
        except Exception as error:
            reason = str(result.get('error') or error)[:1500]
        evidence = FeatureEvidence(session_id=session_id, round_id=round_id, feature_set_id=definition.feature_set_id,
            parent_feature_set_id=pending['parent_feature_set_id'], fingerprint=pending['fingerprint'],
            status=status, reason=reason, mean_delta=mean, fold_deltas=deltas, protocol_hash=protocol_hash,
            baseline_mean_delta=baseline_mean, baseline_fold_deltas=baseline_deltas,
            result_ref=str(directory / definition.feature_set_id / 'result.json'), oof_ref=result.get('oof_ref'),
            duration_seconds=result.get('duration_seconds', 0))
        p.store.put('feature_evidence', evidence_key, evidence, immutable=True)
        atomic_json(directory / definition.feature_set_id / 'evidence.json', evidence)
        if status == 'accepted':
            session.best_result, session.best_feature_set_id = result, definition.feature_set_id
            session.stagnant = 0
        elif status != 'failed':
            session.stagnant += 1
        session.pending = None
        save()
        emit('feature.evaluated', evidence.model_dump(mode='json'))

    if session.best_feature_set_id == session.baseline_feature_set_id:
        session.status = 'completed'
        session.reason = session.reason or 'No stable inner improvement'
        save()
        return None
    try:
        status,_,_=paired_evidence(session.best_result,reference,protocol,model_hash,
                                  context.task.metric_direction,p.config.feature_min_delta)
        if status != 'accepted':
            raise ValueError('Final set is not stable against the original baseline')
    except (ValueError,KeyError,OSError) as error:
        session.status,session.reason='completed',str(error)
        save()
        return None
    definition = FeatureSet.model_validate(p.store.get('feature_set', session.best_feature_set_id))
    winner_plan = plan.model_copy(deep=True)
    winner_plan.feature_set_id, winner_plan.feature_set = definition.feature_set_id, definition
    winner_plan.hypothesis.hypothesis_id = definition.feature_set_id
    winner_plan.hypothesis.description = 'Selected stable feature set ' + definition.feature_set_id
    winner_plan.intervention = Intervention(kind='ablation', base_candidate_id=baseline.candidate.candidate_id,
        changed_fields=['feature_set'], operations=[f.expression_key() for f in definition.features])
    plan_ref = p.run_dir / 'plans' / (round_id + '-' + definition.feature_set_id + '.json')
    atomic_json(plan_ref, winner_plan)
    # Same immutable estimator/preprocessing source, a separately verified assignment.
    candidate = baseline.candidate.model_copy(deep=True)
    candidate.candidate_id = round_id + '-' + definition.feature_set_id
    candidate.hypothesis_id, candidate.plan_ref = definition.feature_set_id, str(plan_ref)
    candidate.based_on, candidate.feature_set_hash = baseline.candidate.candidate_id, feature_hash(definition)
    smoke_path = directory / 'winner-smoke' / 'result.json'
    if smoke_path.exists():
        smoke = json.loads(smoke_path.read_text(encoding='utf-8'))
    else:
        schema = [dict(name=c, semantic_type='numeric' if pd.api.types.is_numeric_dtype(frame[c]) else
                       'datetime' if pd.api.types.is_datetime64_any_dtype(frame[c]) else 'categorical')
                  for c in context.feature_columns]
        try:
            smoke = p.sandbox.run(dict(action='smoke', code_dir=candidate.code_dir, code_hash=candidate.code_hash,
                config=winner_plan.model_dump(mode='json'), task_type=context.task.task_type,
                metric=context.task.primary_metric, prediction_kind=winner_plan.prediction_kind,
                feature_schema=schema, classes=context.classes, target_columns=context.task.target_columns),
                smoke_path.parent, seconds=min(p.config.smoke_seconds, p.budget.snapshot().research_seconds),
                cpus=plan.cpus, memory_gb=plan.memory_gb)
        except WorkerInfrastructureError:
            raise
        except Exception as error:
            smoke=dict(status='failed',error=str(error))
        atomic_json(smoke_path, smoke)
    if smoke.get('status') != 'completed':
        session.status, session.reason = 'completed', 'Feature winner failed portability smoke'
        save()
        return None
    candidate.execution_report, candidate.execution_report_ref = smoke.get('execution_report'), smoke.get('execution_report_ref')
    if (candidate.execution_report or {}).get('model_spec', {}).get('feature_set_hash') != candidate.feature_set_hash:
        session.status, session.reason = 'completed', 'Feature winner smoke has mismatched provenance'
        save()
        return None
    p.store.put('candidate', candidate.candidate_id, candidate, immutable=True)
    p.store.put('plan', plan_ref.stem, winner_plan, immutable=True)
    preprocessing = p.store.get('preprocessing', Path(baseline.candidate.plan_ref).stem)
    if preprocessing:
        p.store.put('preprocessing', plan_ref.stem, preprocessing)
    run = ExperimentRun.model_validate(dict({k:v for k,v in session.best_result.items() if k in ExperimentRun.model_fields},
        experiment_id=candidate.candidate_id, round_id=split.round_id, candidate_id=candidate.candidate_id,
        split_id=split.split_id, resource_cpus=plan.cpus, resource_memory_gb=plan.memory_gb))
    session.winner = dict(candidate=candidate.model_dump(mode='json'), plan=winner_plan.model_dump(mode='json'),
                          run=run.model_dump(mode='json'))
    session.status = 'selected'
    save()
    return session.winner


def complete_feature_session(pipeline, winner, promoted):
    session_id = feature_session_key(winner['run'].get('round_id', 'r1'))
    saved = pipeline.store.get('feature_session', session_id)
    if not saved:
        return
    session = FeatureSession.model_validate(saved)
    if session.status == 'completed':
        return
    session.outer_experiment_id = winner['run']['experiment_id']
    session.reason = 'outer_improvement' if promoted else 'no_outer_improvement'
    if promoted:
        definition = winner['plan']['feature_set']
        acceptance = dict(definition=definition,
            evidence=pipeline.store.get('feature_evidence', session_object_key(session_id, definition['feature_set_id'])),
            outer_experiment_id=session.outer_experiment_id)
        pipeline.store.put('feature_registry_acceptance', session_object_key(session_id, definition['feature_set_id']),
            acceptance, immutable=True)
        if not pipeline.store.get('feature_registry', definition['feature_set_id']):
            pipeline.store.put('feature_registry', definition['feature_set_id'], acceptance, immutable=True)
    # Publish the registry before marking the session complete; replay can safely repair
    # an interrupted completion without losing a promoted set.
    session.status = 'completed'
    pipeline.store.put('feature_session', session_id, session)
    atomic_json(session_directory(pipeline.run_dir, session_id) / 'session.json', session)
    emit('feature.completed', dict(session_id=session_id, round_id=session.round_id,
        reason=session.reason, attempts=session.attempts))


FEATURE_PROMPT = '''Coordinate one feature-only experiment on the frozen baseline. Return a typed
FeatureProposal: add one related group, remove one related group, or stop with a reason.
Use only train_profile (computed from outer train and inner OOF), task semantics and supplied history.
dictionary_metadata contains declared column meanings and units, not measured effects.
Never request external holdout, test, files, arbitrary Python, target-dependent features, IDs or excluded
columns. The trusted DSL executes operations; you do not implement Python or change model parameters.
missing/log/frequency/date_part take one column; ratio/difference/product take two. date_part also
requires part=year/month/day/dayofweek/hour. group_count takes no columns and requires by;
group_mean/std/min/max take one numeric column and by. Logs use positive values; invalid arithmetic
is imputed using training-fold medians. Group maps fit on the training fold only, with stored fallbacks.
Give unique short ASCII feature names. At most 16 features in the resulting set. Do not repeat failed,
rejected or equivalent configurations. A numerical improvement must come from runtime evidence;
semantic plausibility, correlations and importance alone are not proof. No causal claims. Stop when
no useful, new, safe intervention is supported by the available evidence.'''
