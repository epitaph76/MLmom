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
                               FeatureExperimentPlan, feature_hash, feature_configuration_keys)
from .feature_engineering import validate_features
from .persistence import atomic_json, file_hash, tree_hash
from .execution import config_hash
from .model_spec import frozen_plan, Intervention
from .pipeline_contracts import CandidateVersion, HypothesisPlan, ExperimentRun, TaskContext
from .splitting import inner_protocol, feature_context
from .tracing import emit
from .execution_facts import execution_context, configuration_diff


def feature_session_key(round_id='r1'):
    # Keep the first-round identity compatible with already checkpointed runs.
    return 'initial' if round_id == 'r1' else round_id


def session_directory(run_dir, session_id):
    root = run_dir / 'feature_search'
    return root if session_id == 'initial' else root / session_id


def session_object_key(session_id, identity):
    return identity if session_id == 'initial' else session_id + ':' + identity


def select_feature_baseline(pipeline, champion, trained, round_number):
    """Use the champion, or a saved single source if an ensemble is leading."""
    from types import SimpleNamespace
    from .search_groups import internal_winner, component_pool

    def eligible(row):
        plan, run, candidate = row['plan'], row['run'], row['candidate']
        return (not plan.get('ensemble_components') and not plan.get('ensemble_sources') and
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


def feature_runtime_info():
    from importlib.metadata import version, PackageNotFoundError
    installed = {}
    for package in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'polars', 'lightgbm', 'catboost', 'xgboost',
                    'featuretools', 'splink']:
        try:
            installed[package] = version(package)
        except PackageNotFoundError:
            pass
    return dict(interface='create_features(config) -> fit_transform(X,y,context), transform(X)',
        implementations=['python', 'optional_dsl'], feature_count_limit=None,
        installed_libraries=installed, fitting_scope='current training fold only',
        transform_has_labels=False, persistence='cloudpickle state with the model',
        fit_context=dict(access='dataclass attributes',fields=['task_type','metric','direction','seed',
            'cpus','memory_gb','time_limit_seconds','timeouts_enabled','remaining_seconds','classes',
            'groups','sample_weight','work_dir'],validation_data='absent',
            untimed_python_remaining='infinity; inspect timeouts_enabled before deadline arithmetic'))


def prepare_initial_proposal(p, planning_brief):
    """Let the role contribute before model preparation; measure only on a fixed control later."""
    if not p.config.feature_search_enabled or p.store.get('feature_proposal', '0') is not None:
        return
    if p.store.get('feature_proposal_error', 'early') is not None:
        return  # The measured session receives this error and can request a corrected proposal.
    from .splitting import make_split
    from .pipeline_contracts import SplitSpec
    saved = p.store.get('split', 'r1')
    split = SplitSpec.model_validate(saved) if saved else make_split(p.frame,p.context,'r1',p.config.holdout_fraction,p.config.seed)
    p.store.put('split','r1',split,immutable=True)
    directory = p.run_dir/'splits'/'r1'
    directory.mkdir(parents=True,exist_ok=True)
    train_path = directory/'train.csv'
    if not train_path.exists(): p.frame.iloc[split.train_indices].to_csv(train_path,index=False)
    try:
        profile = p.sandbox.run(dict(action='feature_profile',context=p.sanitized_context(train_path),
            train_path=str(train_path)),p.run_dir/'feature_search'/'early-profile',
            seconds=p.config.reconnaissance_query_seconds,cpus=min(2,p.budget.cpus),memory_gb=min(2.,p.budget.memory_gb))
        if profile.get('status') != 'completed': raise ValueError(profile.get('error') or 'Feature profile unavailable')
        from .role_tools import feature_data_tools
        role_tools = feature_data_tools(p, train_path, 'initial')
        payload = dict(task=feature_task_view(p.context.task), train_profile=profile['profile'],
            train_evidence_refs=role_tools.evidence_refs,
            dictionary_metadata=dictionary_metadata(p.context), task_description=Path(p.files.task).read_text(encoding='utf-8-sig'),
            current_feature_set=FeatureSet(feature_set_id='empty').model_dump(mode='json'), history=[],
            phase='proposal_before_baseline', remaining_configurations=p.config.feature_max_configurations,
            feature_runtime=feature_runtime_info(),
            skill_instructions=p.skills.load([],p.context.task.task_type,role='feature_agent',include_references=True))
        with role_window(p.client,'feature_agent','initial:0',p.config.feature_agent_seconds):
            proposal = p.client.generate('feature_agent',FEATURE_PROMPT+' No fitted model or OOF evidence exists yet; '
                'propose a testable feature intervention from train-only facts, or stop.',payload,FeatureProposal,role_tools)
        p.store.put('feature_proposal','0',proposal,immutable=True)
        emit('feature.proposal_prepared',dict(session_id='initial',proposal=proposal.model_dump(mode='json')))
    except (GlobalAPIError, WorkerInfrastructureError):
        raise
    except Exception as error:
        p.store.put('feature_proposal_error','early',dict(session_id='initial',reason=str(error)))


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
    report = dict(rows=len(X), columns=columns,total_columns=len(X.columns),
                  columns_truncated=len(X.columns)>len(columns),column_offset=0,
                  remaining_columns_via='list_tables(table="train", offset=...)',residual_slices=[],
                  residual_columns_considered=min(30,len(X.columns)),residual_columns_truncated=len(X.columns)>30,
                  error_weighting='unweighted', residual_scope='feature_inner_oof')
    if request.get('oof_ref'):
        if context.task.task_type not in {'regression','multioutput_regression','binary_classification',
                                           'multiclass_classification','multilabel_classification'}:
            return dict(status='completed',profile=report)
        oof = json.loads(Path(request['oof_ref']).read_text(encoding='utf-8'))
        indices = np.asarray(oof['indices'], dtype=int)
        prediction = np.asarray(oof['predictions'])
        if prediction.ndim == 2 and prediction.shape[1] == 1:
            prediction = prediction[:, 0]
        truth = np.asarray(y.iloc[indices])
        if context.task.task_type in {'regression','multioutput_regression'}:
            residual = truth - prediction
            error = np.abs(residual)
            if error.ndim > 1:
                error = np.mean(error,axis=1)
                residual = np.mean(residual,axis=1)
        else:
            residual = None
            if context.task.task_type == 'multilabel_classification':
                error = np.mean(np.abs(truth-prediction),axis=1)
            elif prediction.ndim == 2:
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


def proposed_feature_set(parent, proposal, allowed_columns):
    unknown = set(proposal.remove) - {f.name for f in [*parent.features, *parent.programs]}
    if unknown:
        raise ValueError('Removal references unknown features: ' + ', '.join(sorted(unknown)))
    definition = FeatureSet(feature_set_id='pending',
        features=[f for f in parent.features if f.name not in proposal.remove] + proposal.features,
        programs=[p for p in parent.programs if p.name not in proposal.remove] + proposal.programs)
    validate_features(definition, allowed_columns)
    keys = feature_configuration_keys(definition)
    fingerprint = config_hash(keys) if keys else 'empty'
    definition.feature_set_id = 'fe-' + fingerprint[:16] if keys else 'empty'
    return definition, fingerprint


def paired_evidence(candidate, reference, protocol, expected_model_hash, direction, min_delta=0):
    """Compare measured internal means; fold deltas are diagnostics only."""
    def score(result):
        if result.get('status') != 'completed':
            raise ValueError(result.get('error') or 'Feature evaluation failed')
        value = float(result['internal_score'])
        if not math.isfinite(value):
            raise ValueError('Nonfinite feature metric')
        return value
    sign = 1 if direction == 'maximize' else -1
    mean = (score(candidate)-score(reference))*sign
    deltas = [(float(a['score'])-float(b['score']))*sign
              for a,b in zip(candidate.get('fold_results', []), reference.get('fold_results', []))]
    return ('accepted' if mean > min_delta else 'rejected'), mean, deltas


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
        if not (Path(baseline.candidate.code_dir) / 'preprocessing.py').is_file():
            raise ValueError('Feature search requires separate preprocessing.py')
        frame = p.frame.iloc[split.train_indices]
        protocol = inner_protocol(frame, feature_context(context, p.config.feature_holdout_folds),
            fraction=p.config.inner_fraction, seed=split.seed + 1, outer_split_id=split.split_id,
            source_indices=split.train_indices)
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
            attempted_hashes=([config_hash(feature_configuration_keys(baseline_feature_set))]
                              if baseline_feature_set.features or baseline_feature_set.programs else ['empty']),
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
        if reference.get('status') != 'completed':
            raise ValueError(reference.get('error') or 'Feature reference failed')
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
                    from .role_tools import feature_data_tools
                    role_tools = feature_data_tools(p, train_path, session_id)
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
                        train_evidence_refs=role_tools.evidence_refs,
                        execution_context=execution_context(baseline.plan,baseline.run),
                        feature_runtime=feature_runtime_info(), frozen_model=spec,
                        base_preprocessing=(p.store.get('preprocessing', Path(baseline.candidate.plan_ref).stem) or {}).get('spec'),
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
                        skill_instructions=p.skills.load([], context.task.task_type, role='feature_agent',include_references=True))
                    with role_window(p.client, 'feature_agent', session_id + ':' + key,
                                     min(p.config.feature_agent_seconds,p.budget.snapshot().research_seconds)):
                        draft = p.client.generate('feature_agent', FEATURE_PROMPT, payload, FeatureProposal,role_tools).model_dump(mode='json')
                    p.store.put('feature_proposal', proposal_key, draft, immutable=True)
                proposal = FeatureProposal.model_validate(draft)
                session.proposal_calls += 1
                save()
                if proposal.action == 'stop':
                    session.reason = proposal.rationale
                    break
                parent = FeatureSet.model_validate(p.store.get('feature_set', session.best_feature_set_id))
                # Construct and measure only the final set. Removal is not a separate
                # experiment and need not improve the metric on its own.
                definition, fingerprint = proposed_feature_set(parent, proposal, context.feature_columns)
                if fingerprint in session.attempted_hashes:
                    raise ValueError('Equivalent feature configuration was already tested')
                existing = p.store.get('feature_set', definition.feature_set_id)
                if existing:
                    # Equivalent formulas keep their immutable, previously assigned names.
                    definition = FeatureSet.model_validate(existing)
                p.store.put('feature_set', definition.feature_set_id, definition, immutable=True)
                session.pending = dict(feature_set_id=definition.feature_set_id, fingerprint=fingerprint,
                    parent_feature_set_id=parent.feature_set_id, reference=session.best_result,
                    rationale=proposal.rationale, proposal=proposal.model_dump(mode='json'))
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
                    reference_feature_set=p.store.get('feature_set', pending['parent_feature_set_id']),
                    proposal=pending.get('proposal'),
                    reference_feature_set_id=pending['parent_feature_set_id'], hypothesis=pending['rationale'],
                    frozen_model=spec, validation_scope='feature', protocol_hash=protocol_hash,
                    skill_instructions=p.skills.load([], context.task.task_type, role='planner'))
                with role_window(p.client, 'planner', 'feature:' + session_id + ':' + definition.feature_set_id,
                                 min(p.config.planner_seconds,p.budget.snapshot().research_seconds)):
                    assignment = p.client.generate('planner',
                        'Frame this single feature intervention against the supplied reference. Return FeatureExperimentPlan. '
                        'Preserve feature_set_id and reference_feature_set_id exactly. The runtime fixes model source, '
                        'exported parameters, policies, folds and seeds, and disables HPO. Do not introduce another change. '
                        'A replace proposal is one atomic final-set comparison; do not require an intermediate removal gain. '
                        'These are internal train-only experiments, with no external evidence.', payload,
                        FeatureExperimentPlan).model_dump(mode='json')
                assignment = FeatureExperimentPlan.model_validate(assignment)
                assignment = assignment.model_dump(mode='json')
                p.store.put('feature_experiment_plan', evidence_key, assignment, immutable=True)
            result = evaluate(definition.feature_set_id, candidate_plan)
            status, mean, deltas = paired_evidence(result, pending['reference'], protocol, model_hash,
                context.task.metric_direction, p.config.feature_min_delta)
            baseline_status, baseline_mean, baseline_deltas = paired_evidence(result, reference, protocol, model_hash,
                context.task.metric_direction, p.config.feature_min_delta)
            if status == 'accepted' and baseline_status != 'accepted':
                status = baseline_status
            reason = 'Measured internal mean: ' + status
        except (GlobalAPIError, WorkerInfrastructureError):
            raise
        except Exception as error:
            reason = str(result.get('error') or error)[:1500]
        evidence = FeatureEvidence(session_id=session_id, round_id=round_id, feature_set_id=definition.feature_set_id,
            parent_feature_set_id=pending['parent_feature_set_id'], fingerprint=pending['fingerprint'],
            status=status, reason=reason, mean_delta=mean, fold_deltas=deltas, protocol_hash=protocol_hash,
            baseline_mean_delta=baseline_mean, baseline_fold_deltas=baseline_deltas,
            result_ref=str(directory / definition.feature_set_id / 'result.json'), oof_ref=result.get('oof_ref'),
            duration_seconds=result.get('duration_seconds', 0), proposal=pending.get('proposal'))
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
        session.reason = session.reason or 'No inner improvement'
        save()
        return None
    definition = FeatureSet.model_validate(p.store.get('feature_set', session.best_feature_set_id))
    winner_plan = plan.model_copy(deep=True)
    winner_plan.feature_set_id, winner_plan.feature_set = definition.feature_set_id, definition
    winner_plan.hypothesis.hypothesis_id = definition.feature_set_id
    winner_plan.hypothesis.description = 'Selected stable feature set ' + definition.feature_set_id
    winner_plan.agent_notes['historical_baseline_hypothesis']=baseline.plan.hypothesis.model_dump(mode='json')
    winner_plan.hypothesis.rationale = ('Compare the selected SDK feature bundle against the frozen baseline; '
        'constituent proposals and their rationales are recorded in feature evidence. '
        'The bundle score does not establish the effect of individual programs.')
    winner_plan.intervention = Intervention(kind='ablation', base_candidate_id=baseline.candidate.candidate_id,
        changed_fields=['feature_set'], operations=[f.expression_key() for f in [*definition.features, *definition.programs]])
    winner_plan.configuration_diff=dict(base_candidate_id=baseline.candidate.candidate_id,
        base_experiment_id=baseline.run.experiment_id,base_parameters=plan.model_params,
        base_code_hash=baseline.candidate.code_hash,base_preprocessing_hash=baseline.candidate.preprocessing_hash,
        **configuration_diff(plan.model_dump(mode='json'),winner_plan.model_dump(mode='json')))
    plan_ref = p.run_dir / 'plans' / (round_id + '-' + definition.feature_set_id + '.json')
    atomic_json(plan_ref, winner_plan)
    # Same immutable estimator/preprocessing source, a new feature assignment.
    candidate = baseline.candidate.model_copy(deep=True)
    candidate.candidate_id = round_id + '-' + definition.feature_set_id
    candidate.hypothesis_id, candidate.plan_ref = definition.feature_set_id, str(plan_ref)
    candidate.based_on, candidate.feature_set_hash = baseline.candidate.candidate_id, feature_hash(definition)
    candidate.execution_report = candidate.execution_report_ref = None
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


FEATURE_PROMPT = '''Design one executable feature experiment on the frozen control model.
For an entity/history hypothesis follow the supplied skill's structure -> invariants ->
key/linkage quality -> relationships/time -> inference-safe family -> segmented ablation procedure.
Prioritize a short evidence/cost queue instead of enumerating every combination. Without
independent linkage labels, retain unknown error rates; feature gains do not prove identity.
Return FeatureProposal: add, remove, replace, or stop. You may write Python in programs;
the optional features DSL is a shortcut, not a limit on your methods or feature count.
Each program has name, source and config. source defines create_features(config), returning
an object with fit_transform(X,y,context) and transform(X). Return additional columns as a
DataFrame, 2D NumPy array or scipy sparse matrix. Preserve row order and names/width at inference.
Use installed libraries, raw permitted feature columns and current fold training labels.
Supervised representations, smoothed target statistics, inferred entity keys, historical
aggregates and other learned transformations are allowed. For target-derived training
columns use appropriate cross-fitting or exclusion so a row's own label is not its input.
Apply that scope to smoothing priors and fallback statistics too.
Build inference maps from this fold's train only; transform receives X without labels and
must not fit or learn from the prediction batch. Respect declared group/time boundaries.
Treat inferred identities as hypotheses to measure, not as verified IDs; lack of an official
entity ID or a matching DSL operator is not by itself a reason to abandon an experiment.
Persist learned maps/state on the object; the SDK serializes it with the model. No file,
network, environment or external-dataset reads. No access to outer holdout/test labels.
Keep the frozen control model, its parameters, folds and target/output policies unchanged.
Use train_profile, task semantics, dictionary meanings and supplied scoped history. Explain
uncertainty rather than inventing observed improvements. Consider compatible costs/resources.
Replace removes named programs/features and adds new ones atomically, without an intermediate
removal-only gain requirement. Legacy add with removals is read as replace. Names identify
programs/features; the number of returned columns is chosen by you. Runtime measures the
final set against the current accepted set and the original control on the same folds.
Actual errors and measurements return to you for correction or a different proposal.
Stop when no useful further experiment is justified, not merely because an idea needs Python.'''
