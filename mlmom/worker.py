"""Short-lived worker entrypoint. Invoked only by SandboxManager, never by an LLM."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
import uuid
import copy
from dataclasses import replace
from pathlib import Path


def failure_result(error, root):
    """Keep complete debugging evidence on disk and send the repair role one defect."""
    from .execution import FidelityError
    path = Path(root)/'traceback.log'
    path.write_text(''.join(traceback.format_exception(type(error),error,error.__traceback__)),encoding='utf-8')
    frames = traceback.extract_tb(error.__traceback__)
    authored = [frame for frame in frames if Path(frame.filename).name in {'candidate.py','preprocessing.py'}]
    frame = authored[-1] if authored else frames[-1] if frames else None
    source = dict(file=frame.filename,line=frame.lineno,function=frame.name) if frame else None
    fidelity = isinstance(error,FidelityError)
    hint = ('Repair the reported configuration, export or stopping defect; inspect execution_report_ref.' if fidelity else
            'Use apply_patch on the existing source at error_details.source; retain unrelated working code.')
    message = f'{type(error).__name__}: {str(error)[:1700]}'
    if source: message += f' [{Path(source["file"]).name}:{source["line"]}]'
    result = dict(status='failed',error=message,error_code='execution_fidelity_failed' if fidelity else 'worker_runtime_error',
                  error_details=dict(type=type(error).__name__,source=source,hint=hint),traceback_ref=str(path))
    if fidelity:
        result.update(execution_report=error.report,execution_report_ref=error.path,stop_reason='execution_fidelity_failed')
    return result


def parent_is_alive(request):
    """Check controller identity without comparing wall-clock timestamps.

    ``psutil.Process.create_time`` is wall-clock based on some platforms.  A
    host/container clock correction therefore used to kill healthy workers as
    ``parent_exit``.  Direct parenthood is authoritative; the executable path
    protects the fallback from the unlikely case of PID reuse.
    """
    import psutil
    try:
        pid=int(request['parent_pid'])
        parent=psutil.Process(pid)
        if not parent.is_running():return False
        if os.getppid()==pid:return True
        expected=request.get('parent_executable')
        if not expected:return True  # Backward-compatible request artifact.
        try:actual=parent.exe()
        except (psutil.AccessDenied,psutil.ZombieProcess):return True
        # A Windows venv launcher and psutil may report different absolute
        # paths for the same interpreter (venv shim versus base runtime).
        return Path(actual).name.casefold()==Path(expected).name.casefold()
    except (KeyError,TypeError,ValueError,psutil.NoSuchProcess,psutil.ZombieProcess):
        return False


def watchdog(request):
    import psutil
    until = time.monotonic()+request['worker_seconds']
    while True:
        alive = parent_is_alive(request)
        reason = 'parent_exit' if not alive else 'time_limit' if request.get('timeouts_enabled',True) and time.monotonic()>until else None
        try:
            processes=[psutil.Process()]+psutil.Process().children(recursive=True)
            memory=sum(p.memory_info().rss for p in processes)/1024**3
            if memory>request['worker_memory_gb']:reason='memory_limit'
        except psutil.Error:pass
        if reason:
            Path('limit.json').write_text(json.dumps({'reason':reason}),encoding='utf-8')
            from .resources import terminate_tree
            terminate_tree(os.getpid())
            return
        time.sleep(.2)


def fit_context(context, request, *, work_dir, X_valid=None, y_valid=None, groups=None, weights=None, validation_groups=None, validation_weight=None, seconds=None):
    from .sdk import FitContext
    work_dir=Path(work_dir)/uuid.uuid4().hex[:8]
    work_dir.mkdir(parents=True,exist_ok=True)
    return FitContext(task_type=context.task.task_type,metric=context.task.primary_metric,
        direction=context.task.metric_direction,seed=request.get('seed',42),
        cpus=request['worker_cpus'],memory_gb=request['worker_memory_gb'],
        time_limit_seconds=max(.1,seconds or request['worker_seconds']*.8),classes=context.classes,
        groups=groups,sample_weight=weights,validation_X=X_valid,validation_y=y_valid,
        validation_groups=validation_groups,validation_weight=validation_weight,work_dir=str(work_dir),task_context=context,
        timeouts_enabled=request.get('timeouts_enabled',True))


def write_predictions(model, X, kinds, root):
    import pandas as pd
    from .evaluation import checked_predictions
    result = {}
    for kind in dict.fromkeys(kinds):
        predictions = checked_predictions(model.predict(X,{'kind':kind}),len(X),kind)
        path = root/f'prediction-{kind}.parquet'
        pd.DataFrame(predictions).to_parquet(path,index=False)
        result[kind] = str(path)
    return result


def synthetic(kind):
    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_classification,make_regression,make_blobs
    if kind == 'clustering':
        X,y = make_blobs(n_samples=60,n_features=3,centers=3,random_state=4)
        return pd.DataFrame(X,columns=['x','z','v']), None
    if 'classification' in kind or kind == 'anomaly_detection':
        X,y = make_classification(n_samples=60,n_features=4,n_informative=3,n_redundant=0,
                                  n_classes=3 if kind=='multiclass_classification' else 2,random_state=4)
        if kind == 'multilabel_classification': y = np.column_stack([y,np.arange(60)%2])
    else:
        X,y = make_regression(n_samples=60,n_features=4,n_targets=2 if kind=='multioutput_regression' else 1,random_state=4)
        if kind=='ranking':y=np.tile(np.arange(5),12)
    return pd.DataFrame(X,columns=['x','z','v','w']), pd.DataFrame(y) if np.asarray(y).ndim==2 else pd.Series(y)


def smoke(request, root):
    from .sdk import load_module,FitContext
    from .adapters import LibraryCandidate,load_adapter
    from .evaluation import checked_predictions
    from .persistence import tree_hash
    from .execution import audit_fit, compare_roundtrip, FidelityError
    kind = request.get('task_type','regression')
    X,y = synthetic(kind)
    if request.get('feature_schema'):
        import pandas as pd
        import numpy as np
        rng=np.random.default_rng(42)
        columns={}
        for column in request['feature_schema']:
            semantic=column['semantic_type']
            if semantic in {'categorical','text'}:
                values=np.resize(['synthetic_a','synthetic_b','synthetic_c'],len(X))
            elif semantic=='boolean':values=np.arange(len(X))%2==0
            elif semantic=='datetime':values=pd.date_range('2020-01-01',periods=len(X))
            else:values=rng.uniform(1,10,len(X))
            columns[column['name']]=values
        X=pd.DataFrame(columns)
        labels=request.get('classes',[])
        if 'classification' in kind and kind!='multilabel_classification' and labels:
            count=max(len(X),len(labels)*2)
            if count>len(X):X=X.iloc[np.arange(count)%len(X)].reset_index(drop=True)
            y=pd.Series([labels[i%len(labels)] for i in range(len(X))])
        names=request.get('target_columns',[])
        if names and y is not None:
            if kind in {'multioutput_regression','multilabel_classification'}:
                base=np.asarray(y)
                if base.ndim==1:base=base[:,None]
                y=pd.DataFrame({name:base[:,i%base.shape[1]] for i,name in enumerate(names)})
            elif isinstance(y,pd.Series):y.name=names[0]
    config = dict(request['config'],_code_hash=request.get('code_hash'))
    if request.get('code_dir'):
        if request.get('code_hash') and tree_hash(Path(request['code_dir'])) != request['code_hash']:
            raise ValueError('Candidate hash mismatch')
        module = load_module(request['code_dir'])
        model = module.create_candidate(copy.deepcopy(config))
    else:
        model = LibraryCandidate(copy.deepcopy(config))
        module = None
    import numpy as np
    groups = np.repeat(np.arange(12),5).astype(str) if kind == 'ranking' else None
    context = FitContext(task_type=kind,metric=request.get('metric','rmse'),direction='minimize',
        seed=42,cpus=request['worker_cpus'],memory_gb=request['worker_memory_gb'],
        time_limit_seconds=max(1,request['worker_seconds']*.5),
        classes=sorted(y.unique().tolist()) if y is not None and getattr(y,'ndim',1)==1 and ('classification' in kind) else [],
        groups=groups,work_dir=str(root/'native'),timeouts_enabled=request.get('timeouts_enabled',True))
    Path(context.work_dir).mkdir(parents=True,exist_ok=True)
    if request.get('native_holdout'):
        from sklearn.model_selection import train_test_split
        a,b=train_test_split(np.arange(len(X)),test_size=.25,random_state=42,
                            stratify=y if 'classification' in kind else None)
        context.fit_rows=len(a)
        model.fit(X.iloc[a],y.iloc[a],replace(context,validation_X=X.iloc[b],validation_y=y.iloc[b]))
    else:
        context.fit_rows=len(X)
        model.fit(X,y,context)
    inspection_context = replace(context,validation_y=y.iloc[b]) if request.get('native_holdout') else context
    report_path=root/'execution-report.json'
    report=audit_fit(model,config.get('model_params',{}),config,inspection_context,report_path,stage='smoke')
    frozen=report['exported_params']
    frozen_config=dict(config,model_params=frozen,stopping_policy='fixed',component_stopping_policies={})
    frozen_model=module.create_candidate(copy.deepcopy(frozen_config)) if module else LibraryCandidate(copy.deepcopy(frozen_config))
    frozen_context=replace(context,work_dir=str(root/'frozen'),started=time.monotonic(),
                           time_limit_seconds=max(1,request['worker_seconds']*.25),fit_rows=len(X))
    Path(frozen_context.work_dir).mkdir(parents=True,exist_ok=True)
    frozen_model.fit(X,y,frozen_context)
    recreated=audit_fit(frozen_model,frozen,frozen_config,frozen_context,root/'recreated-report.json',stage='smoke_recreate')
    errors=compare_roundtrip(report,recreated)
    if not request.get('native_holdout') and config.get('tool_id') not in {'flaml','autogluon','lightautoml'}:
        left=np.asarray(model.predict(X.iloc[:5],{'kind':request.get('prediction_kind','value')}))
        right=np.asarray(frozen_model.predict(X.iloc[:5],{'kind':request.get('prediction_kind','value')}))
        same=np.array_equal(left,right)
        if not same and np.issubdtype(left.dtype,np.number) and np.issubdtype(right.dtype,np.number):
            same=np.allclose(left,right,rtol=1e-6,atol=1e-8)
        if not same:errors.append('export/recreate changes predictions with the same data and seed')
    report['roundtrip']='failed' if errors else 'passed'
    report['errors'].extend(errors)
    report['status']='failed' if errors else 'passed'
    from .persistence import atomic_json
    atomic_json(report_path,report)
    if errors:raise FidelityError(report,report_path)
    (root/'frozen-model').mkdir(parents=True,exist_ok=True)
    frozen_model.save(str(root/'frozen-model'))
    frozen_loaded=module.load_candidate(str(root/'frozen-model')) if module else load_adapter(str(root/'frozen-model'))
    checked_predictions(frozen_loaded.predict(X.iloc[:5],{'kind':request.get('prediction_kind','value')}),5,
                        request.get('prediction_kind','value'))
    kinds = [request.get('prediction_kind','value')]
    for prediction in kinds: checked_predictions(model.predict(X.iloc[:5],{'kind':prediction}),5,prediction)
    (root/'model').mkdir(parents=True,exist_ok=True)
    model.save(str(root/'model'))
    moved=root/f'relocated-model-{uuid.uuid4().hex[:8]}'
    (root/'model').rename(moved)
    restored = module.load_candidate(str(moved)) if module else load_adapter(str(moved))
    for prediction in kinds:
        original = np.asarray(model.predict(X.iloc[:5],{'kind':prediction}))
        after = np.asarray(restored.predict(X.iloc[:5],{'kind':prediction}))
        if original.shape != after.shape or not np.array_equal(original,after):
            if not np.issubdtype(original.dtype,np.number) or not np.allclose(original,after):
                raise ValueError('Model save/load changes predictions')
    return {'status':'completed','execution_report':report,'execution_report_ref':str(report_path)}


def train(request, root):
    import numpy as np
    import optuna
    from .pipeline_contracts import TaskContext,HypothesisPlan,TrialResult
    from .inputs import read_table,features,targets
    from .splitting import inner_protocol,group_keys
    from .sdk import load_module
    from .evaluation import score_predictions,prediction_kind,better
    from .persistence import atomic_json,tree_hash,file_hash
    context = TaskContext.model_validate(request['context'])
    plan = HypothesisPlan.model_validate(request['plan'])
    from .registry import NATIVE_VALIDATION
    if any(c.get('tool_id') in NATIVE_VALIDATION for c in plan.ensemble_components):
        raise ValueError('Native AutoML ensemble components have no verified frozen refit protocol')
    if not request.get('fixed') and plan.tool_id in NATIVE_VALIDATION:
        if (plan.strategy != 'native' or context.task.validation_strategy not in NATIVE_VALIDATION[plan.tool_id]
                or context.task.group_columns):
            raise ValueError('Native internal validation protocol is not verified for this task')
    if tree_hash(Path(request['code_dir'])) != request['code_hash']: raise ValueError('Candidate hash mismatch')
    module = load_module(request['code_dir'])
    frame = read_table(request['train_path'],table_format=context.table_formats.get(request['train_path']))
    X,y = features(frame,context), targets(frame,context)
    groups = group_keys(frame,context.task.group_columns)
    from .splitting import validation_groups
    fit_groups=groups if context.task.task_type=='ranking' else validation_groups(frame,context)
    weights = frame[context.task.weight_column].to_numpy() if context.task.weight_column else None
    kind = prediction_kind(context.task.primary_metric,context.task.task_type)
    unsupervised = context.task.task_type in {'clustering','anomaly_detection'}
    best_score = None
    best_params = dict(plan.model_params)
    trials = []
    started = time.monotonic()
    best_model = None
    search_result = {}
    config = dict(plan.model_dump(mode='json'),_code_hash=request.get('code_hash'))
    window_path = root/'training-window.json'
    timed=request.get('timeouts_enabled',True)
    if not window_path.exists():
        atomic_json(window_path, dict(deadline=time.time()+request['worker_seconds']*.85 if timed else None,timeouts_enabled=timed))
    saved_window=json.loads(window_path.read_text(encoding='utf-8'))
    if saved_window.get('timeouts_enabled',True)!=timed:raise ValueError('Resume requires the same timeout policy')
    from .time_policy import UNBOUNDED_SECONDS
    deadline = min(saved_window['deadline'],time.time()+request['worker_seconds']*.85) if timed else UNBOUNDED_SECONDS
    fit_budget = max(0., deadline-time.time())
    control_action = None
    protocol_ref = None
    best_details = {}

    def control(stage, details=None, error=None):
        nonlocal control_action
        from .pipeline_contracts import TrainingCheckpoint
        from .training_control import worker_checkpoint, fallback
        details=details or {}
        checkpoint=TrainingCheckpoint(checkpoint_id=f'{stage}-{len(trials)}',stage=stage,
            plan=plan.model_dump(mode='json'),trials=[t.model_dump(mode='json') for t in trials],
            best_params=details.get('best_params',best_params),best_score=details.get('best_score',best_score),
            remaining_seconds=max(0.,min(details.get('remaining_seconds',fit_budget),fit_budget-(time.monotonic()-started))),
            remaining_trials=details.get('remaining_trials',max(0,min(plan.max_trials,request.get('hpo_max_trials',15))-len(trials))) if plan.strategy=='optuna' else 0,
            default_action=details.get('default_action','continue' if plan.strategy=='optuna' else 'refit'),error=error)
        checkpoint=checkpoint.model_copy(update=dict(score_kind='inner_holdout' if context.task.validation_strategy=='holdout' else 'inner_cv_mean',
            outer_split_id=request.get('outer_split_id'),validation_protocol_ref=protocol_ref,
            metric_options=context.metric_options,weighting='sample_weight' if weights is not None else 'unweighted'))
        if not request.get('training_control'):
            return fallback(checkpoint,'Programmatic policy')
        decision=worker_checkpoint(root,checkpoint,seconds=request.get('training_control_seconds',30) if request.get('timeouts_enabled',True) else None)
        control_action=decision.action
        return decision

    def save_progress(model, params, score, reason):
        from .execution import audit_fit
        report_path=root/'execution-report.json'
        report=audit_fit(model,params,dict(config,stopping_policy='fixed',component_stopping_policies={}),
                         c_final,report_path,stage='full_refit',protocol_ref=protocol_ref)
        params=report['exported_params']
        artifact=root/f'model-{uuid.uuid4().hex[:12]}'
        artifact.mkdir(parents=True)
        model.save(str(artifact))
        history=model.native_history() if hasattr(model,'native_history') else None
        history_ref=None
        if history:
            history_ref=str(artifact/'native-history.json')
            atomic_json(Path(history_ref),history)
        atomic_json(root/'progress.json',dict(status='completed',model_artifact=str(artifact),
            best_params=params,internal_score=score,trials=[t.model_dump(mode='json') for t in trials],stop_reason=reason,native_history_ref=history_ref,
            study_ref=search_result.get('study_ref'),hpo_batches=search_result.get('batches',[]),
            full_refit=True,validation_protocol_ref=protocol_ref,execution_report=report,
            execution_report_ref=str(report_path),score_kind=('not_applicable' if score is None else
                'inner_holdout' if context.task.validation_strategy=='holdout' else 'inner_cv_mean'),**best_details))

    if request.get('fixed'):
        if fit_budget <= .2: raise TimeoutError('Original training deadline reached before fixed refit')
        model = module.create_candidate(copy.deepcopy(config))
        c = fit_context(context,request,work_dir=root/'fixed',groups=fit_groups,weights=weights,seconds=fit_budget)
        c.fit_rows=len(X)
        model.fit(X,None if unsupervised else y,c)
        c_final=c
        if time.time() >= deadline: raise TimeoutError('Fixed refit exceeded original deadline')
        save_progress(model,best_params,None,'fixed_configuration')
    else:
        protocol = inner_protocol(frame,context,fraction=request.get('inner_fraction',.2),
            seed=request.get('seed',42)+1,outer_split_id=request.get('outer_split_id'),
            source_indices=request.get('outer_train_indices'))
        protocol_ref = str(root/'validation-protocol.json')
        protocol_path = Path(protocol_ref)
        if protocol_path.exists() and json.loads(protocol_path.read_text(encoding='utf-8')) != protocol:
            raise ValueError('Saved validation protocol differs from this request')
        atomic_json(protocol_path,protocol)
        selection_path=root/'selection.json'
        if selection_path.exists():
            saved=json.loads(selection_path.read_text(encoding='utf-8'))
            if saved['protocol'] != protocol: raise ValueError('Incompatible trial selection protocol')
            best_score=saved['score'];best_params=saved['params'];best_details=saved['details']
        if (root/'trials.json').exists():
            trials=[TrialResult.model_validate(t) for t in json.loads((root/'trials.json').read_text(encoding='utf-8'))]
        # Reserve the existing worker window for a full-train refit. Checkpoints,
        # all folds and interrupted attempts consume the same original deadline.
        search_deadline=min(deadline-fit_budget*.3,time.time()+request.get('hpo_seconds',600))
        if not request.get('timeouts_enabled',True):search_deadline=deadline
        def objective(sampled=None, number=None, seconds=None):
            nonlocal best_score,best_params,best_details
            params = dict(plan.model_params)
            params.update(sampled or {})
            t0 = time.monotonic()
            trial_number=number if number is not None else len(trials)
            trial=TrialResult(number=trial_number,params=params,status='running',duration_seconds=0,
                kind='native_run' if plan.strategy=='native' else plan.strategy,validation_protocol_ref=protocol_ref,
                score_kind='inner_holdout' if context.task.validation_strategy=='holdout' else 'inner_cv_mean',
                outer_split_id=request.get('outer_split_id'),metric_options=context.metric_options,
                weighting='sample_weight' if weights is not None else 'unweighted')
            trials.append(trial)
            trial_root=root/f'trial-{trial_number}'
            trial_root.mkdir(exist_ok=True)
            until=min(search_deadline,time.time()+seconds) if seconds is not None else search_deadline
            predictions=[];positions=[]
            selected_params=params
            try:
                for fold in protocol['folds']:
                    a,b=np.asarray(fold['train_indices']),np.asarray(fold['validation_indices'])
                    remaining=until-time.time()
                    if remaining <= .2: raise TimeoutError('Internal validation deadline reached')
                    model = module.create_candidate(copy.deepcopy(dict(config,model_params=params)))
                    holdout=protocol['strategy']=='holdout'
                    c=fit_context(context,request,work_dir=trial_root/f"fold-{fold['number']}",
                        X_valid=X.iloc[b] if holdout else None,
                        y_valid=y.iloc[b] if holdout and not unsupervised else None,
                        groups=fit_groups[a] if fit_groups is not None else None,
                        validation_groups=groups[b] if holdout and groups is not None else None,
                        validation_weight=weights[b] if holdout and weights is not None else None,
                        weights=weights[a] if weights is not None else None,
                        seconds=remaining/(protocol['n_splits']-fold['number']))
                    fold_started=time.monotonic()
                    c.fit_rows=len(a)
                    model.fit(X.iloc[a],None if unsupervised else y.iloc[a],c)
                    from .execution import audit_fit
                    fit_report=audit_fit(model,params,config,c,trial_root/f"fold-{fold['number']}-execution.json",
                        stage='inner_fold',protocol_ref=protocol_ref)
                    prediction=np.asarray(model.predict(X.iloc[b],{'kind':kind}))
                    score=score_predictions(y.iloc[b] if y is not None else None,prediction,context,X=X.iloc[b],
                        groups=groups[b] if groups is not None else None,weights=weights[b] if weights is not None else None)
                    if time.time() >= until: raise TimeoutError('Internal fold exceeded validation deadline')
                    trial.fold_results.append(dict(number=fold['number'],score=score,status='completed',
                        duration_seconds=time.monotonic()-fold_started,execution_status=fit_report['status'],
                        execution_report_ref=str(trial_root/f"fold-{fold['number']}-execution.json"),
                        actual_components={name:{key:item.get(key) for key in ('backend','completed_iterations','retained_trees','stopping_reason')}
                                           for name,item in fit_report['components'].items()},
                        score_kind='inner_fold',weighting='sample_weight' if weights is not None else 'unweighted'))
                    trial.duration_seconds=time.monotonic()-t0
                    atomic_json(root/'trials.json',trials)
                    predictions.append(prediction);positions.extend(b.tolist())
                    if holdout:selected_params=fit_report['exported_params']
                if plan.strategy=='native' and not selected_params.get('__native_frozen'):
                    raise ValueError('Native AutoML must export a frozen configuration for search-free refit')
                order=np.argsort(positions);indices=np.asarray(positions)[order]
                combined=np.concatenate(predictions,axis=0)[order]
                trial.oof_score=score_predictions(y.iloc[indices] if y is not None else None,combined,context,X=X.iloc[indices],
                    groups=groups[indices] if groups is not None else None,weights=weights[indices] if weights is not None else None)
                trial.oof_ref=str(trial_root/'oof.json')
                atomic_json(Path(trial.oof_ref),dict(indices=indices.tolist(),
                    source_indices=np.asarray(request.get('outer_train_indices',list(range(len(frame)))))[indices].tolist(),
                    prediction_kind=kind,predictions=combined.tolist()))
                scores=[f['score'] for f in trial.fold_results]
                score=float(np.mean(scores));trial.score=score
                trial.score_std=float(np.std(scores)) if len(scores)>1 else None
                trial.status='completed';trial.duration_seconds=time.monotonic()-t0
                details={key:getattr(trial,key) for key in ('fold_results','score_std','oof_score','oof_ref')}
                trial_selection=dict(protocol=protocol,number=trial_number,score=score,params=selected_params,details=details)
                atomic_json(trial_root/'selection.json',trial_selection)
                if best_score is None or better(score,best_score,context.task.metric_direction):
                    best_score,best_params=score,selected_params
                    best_details=details
                    if plan.strategy!='optuna':atomic_json(selection_path,trial_selection)
                atomic_json(root/'trials.json',trials)
                if len(trials)==1 and plan.strategy!='optuna':
                    control('initial_fit')
                return score
            except optuna.TrialPruned: raise
            except Exception as error:
                from .execution import FidelityError
                if isinstance(error,FidelityError):
                    atomic_json(root/'last-fidelity-failure.json',error.report)
                trial.status='failed';trial.score=None;trial.error=str(error)[:2000];trial.duration_seconds=time.monotonic()-t0
                atomic_json(root/'trials.json',trials)
                if plan.strategy!='optuna':control('failure',error=str(error)[:2000])
                raise

        if plan.strategy=='optuna':
            from .hpo import run_search
            def search_checkpoint(details):
                return control('batch',details)
            search_result=run_search(objective,directory=root/'hpo',identity=dict(code=request['code_hash'],
                train_hash=file_hash(Path(request['train_path'])),seed=request.get('seed',42),inner_fraction=request.get('inner_fraction',.2),
                task=context.task.model_dump(mode='json'),metric_options=context.metric_options,
                base_params=plan.model_params,tool_id=plan.tool_id,preprocessing=plan.preprocessing,validation_protocol=protocol),
                space=plan.search_space,direction=context.task.metric_direction,seed=request.get('seed',42),
                max_trials=min(plan.max_trials,request.get('hpo_max_trials',15)),batch_size=request.get('hpo_batch_size',5),
                gain_threshold=request.get('hpo_gain_threshold',.005),seconds=max(0,search_deadline-time.time()),
                timeouts_enabled=request.get('timeouts_enabled',True),
                baseline_score=request.get('baseline_score'),checkpoint=search_checkpoint if request.get('training_control') else None,
                trial_checkpoint=(lambda details:control(details['stage'],details,error=details.get('error'))) if request.get('training_control') else None)
            control_action=search_result.get('control_action',control_action)
            metadata={t.number:t.model_dump() for t in trials}
            trials=[TrialResult(**dict(metadata.get(t['number'],{}),number=t['number'],params=dict(plan.model_params,**t['params']),
                status='completed' if t['status']=='complete' else 'failed',score=t['score'],
                duration_seconds=t['duration_seconds'],error=t.get('error'))) for t in search_result['trials']]
            atomic_json(root/'trials.json',trials)
            successful=[t for t in trials if t.status=='completed']
            if not successful:
                failure_path=root/'last-fidelity-failure.json'
                if failure_path.exists():
                    from .execution import FidelityError
                    raise FidelityError(json.loads(failure_path.read_text(encoding='utf-8')),failure_path)
                raise ValueError('No successful internal trial: '+'; '.join(t.error or '' for t in trials)[-2500:])
            winner=(min if context.task.metric_direction=='minimize' else max)(successful,key=lambda t:t.score)
            chosen=json.loads((root/f'trial-{winner.number}'/'selection.json').read_text(encoding='utf-8'))
            if chosen['protocol']!=protocol or chosen['score']!=winner.score:
                raise ValueError('Completed study trial differs from saved validation evidence')
            best_score,best_params,best_details=chosen['score'],chosen['params'],chosen['details']
            atomic_json(selection_path,chosen)
        elif best_score is None: objective()
        else: control('initial_fit')  # Reuse the completed trial and any committed decision.
        if best_score is None: raise ValueError('No successful internal trial')
        if deadline-time.time() <= .2: raise TimeoutError('No time for mandatory full-train refit')
        model = module.create_candidate(copy.deepcopy(dict(config,model_params=best_params)))
        c_final=fit_context(context,request,work_dir=root/'refit-round',
            groups=fit_groups,weights=weights,seconds=deadline-time.time())
        c_final.fit_rows=len(X)
        model.fit(X,None if unsupervised else y,c_final)
        if time.time() >= deadline: raise TimeoutError('Full-train refit exceeded original deadline')
        save_progress(model,best_params,best_score,search_result.get('stop_reason') or
                      ('training_agent_stop' if control_action=='stop' else 'completed'))
    return json.loads((root/'progress.json').read_text(encoding='utf-8'))


def predict(request, root):
    from .sdk import load_module
    from .inputs import read_table
    from .persistence import tree_hash
    if tree_hash(Path(request['code_dir'])) != request['code_hash']: raise ValueError('Candidate hash mismatch')
    module = load_module(request['code_dir'])
    model = module.load_candidate(request['model_artifact'])
    X = read_table(request['features_path'])
    return {'status':'completed','predictions':write_predictions(model,X,request['kinds'],root)}


def main():
    request_path = Path(sys.argv[1]).resolve()
    root = request_path.parent
    request = json.loads(request_path.read_text(encoding='utf-8'))
    # Imports can fail before main (notably in package __init__). The launcher's
    # controller snapshot still exists in that case, alongside stderr.log.
    runtime_path = root/'runtime.json'
    runtime = json.loads(runtime_path.read_text(encoding='utf-8')) if runtime_path.exists() else {}
    runtime['worker'] = dict(executable=sys.executable, version=sys.version,
        prefix=sys.prefix, base_prefix=sys.base_prefix, sys_path=sys.path)
    from .persistence import atomic_json
    atomic_json(runtime_path, runtime)
    import faulthandler
    faulthandler.enable()
    if request.get('timeouts_enabled',True):
        faulthandler.dump_traceback_later(max(.1,request['worker_seconds']*.95))
    threading.Thread(target=watchdog,args=(request,),daemon=True).start()
    try:
        action = request['action']
        if action == 'runtime_check':
            import importlib
            from .resources import RUNTIME_MODULES, runtime_versions, worker_runtime
            for name in RUNTIME_MODULES:
                importlib.import_module(name)
            result = dict(status='completed', runtime=worker_runtime(Path(__file__).resolve().parents[1]),
                          versions=runtime_versions())
        elif action in {'smoke','probe'}: result = smoke(request,root)
        elif action=='train': result = train(request,root)
        elif action=='predict': result = predict(request,root)
        elif action=='csv':
            from .csv_tools import csv_worker
            result = csv_worker(request,root)
        elif action=='diagnose':
            from .sdk import load_module
            from .inputs import read_table,features,targets
            from .splitting import group_keys
            from .pipeline_contracts import TaskContext
            from .persistence import tree_hash
            from .diagnostics import inspect_model
            if tree_hash(Path(request['code_dir']))!=request['code_hash']:raise ValueError('Candidate hash mismatch')
            context=TaskContext.model_validate(request['context'])
            frame=read_table(request['holdout_path'])
            candidate=load_module(request['code_dir']).load_candidate(request['model_artifact'])
            result=dict(status='completed',diagnostics=inspect_model(candidate,features(frame,context),targets(frame,context),context,
                seconds=max(.1,request['worker_seconds']-3),max_rows=request.get('max_rows',1000),
                max_features=request.get('max_features',20),groups=group_keys(frame,context.task.group_columns),
                weights=frame[context.task.weight_column].to_numpy() if context.task.weight_column else None))
        elif action=='inspect':
            from .data_inspector import DataInspector
            files = request['files']
            profile = DataInspector().inspect_directory(files['root'],manifest_path=files['manifest'],train_path=files['train'],
                dictionary_paths=files['dictionaries'],task_description_path=files['task'])
            atomic_json(root/'profile.json',profile)
            result = {'status':'completed','profile_ref':str(root/'profile.json')}
        elif action=='inspect_api':
            import importlib,inspect,re
            allowed={'sklearn','catboost','lightgbm','xgboost','flaml','autogluon','lightautoml','statsmodels',
                     'torch','pandas','numpy','scipy','imblearn','feature_engine','category_encoders','optuna','polars'}
            module=request['module'];symbol=request['symbol']
            pattern=r'[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*'
            if not re.fullmatch(pattern,module) or not re.fullmatch(pattern,symbol):raise ValueError('Invalid API identifier')
            if module.split('.')[0] not in allowed and module != 'mlmom.sdk':
                raise ValueError('API module is not exposed')
            value=importlib.import_module(module)
            for part in symbol.split('.'):value=getattr(value,part)
            try:signature=str(inspect.signature(value))
            except (ValueError,TypeError):signature=None
            result={'status':'completed','module':module,'symbol':symbol,'signature':signature,'documentation':(inspect.getdoc(value) or '')[:6000]}
        else: raise ValueError(f'Unknown worker action: {action}')
        atomic_json(root/'result.json',result)
        faulthandler.cancel_dump_traceback_later()
    except Exception as exc:
        atomic_json(root/'result.json',failure_result(exc,root))
        sys.exit(1)


if __name__=='__main__': main()
