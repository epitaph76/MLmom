"""Short-lived worker entrypoint. Invoked only by SandboxManager, never by an LLM."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
import uuid
from dataclasses import replace
from pathlib import Path


def watchdog(request):
    import psutil
    until = time.monotonic()+request['worker_seconds']
    while True:
        try:
            parent = psutil.Process(request['parent_pid'])
            alive = parent.create_time() == request['parent_created']
        except psutil.Error: alive = False
        reason = 'parent_exit' if not alive else 'time_limit' if time.monotonic()>until else None
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
        validation_groups=validation_groups,validation_weight=validation_weight,work_dir=str(work_dir),task_context=context)


def write_predictions(model, X, kinds, root):
    import pandas as pd
    from .evaluation import checked_predictions
    result = {}
    for kind in dict.fromkeys(kinds):
        predictions = checked_predictions(model.predict(X,{'kind':kind}),len(X),kind)
        path = root/f'prediction-{kind}.csv'
        pd.DataFrame(predictions).to_csv(path,index=False)
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
    config = request['config']
    if request.get('code_dir'):
        if request.get('code_hash') and tree_hash(Path(request['code_dir'])) != request['code_hash']:
            raise ValueError('Candidate hash mismatch')
        module = load_module(request['code_dir'])
        model = module.create_candidate(config)
    else:
        model = LibraryCandidate(config)
        module = None
    import numpy as np
    groups = np.repeat(np.arange(12),5).astype(str) if kind == 'ranking' else None
    context = FitContext(task_type=kind,metric=request.get('metric','rmse'),direction='minimize',
        seed=42,cpus=request['worker_cpus'],memory_gb=request['worker_memory_gb'],
        time_limit_seconds=max(1,request['worker_seconds']*.5),
        classes=sorted(y.unique().tolist()) if y is not None and getattr(y,'ndim',1)==1 and ('classification' in kind) else [],
        groups=groups,work_dir=str(root/'native'))
    Path(context.work_dir).mkdir(parents=True,exist_ok=True)
    model.fit(X,y,context)
    if hasattr(model,'export_model_params'):
        frozen=model.export_model_params()
        json.dumps(frozen,allow_nan=False)
        if request.get('verify_frozen'):
            config=dict(config,model_params=frozen)
            frozen_model=module.create_candidate(config) if module else LibraryCandidate(config)
            frozen_context=replace(context,work_dir=str(root/'frozen'),started=time.monotonic(),
                                   time_limit_seconds=max(1,request['worker_seconds']*.25))
            Path(frozen_context.work_dir).mkdir(parents=True,exist_ok=True)
            frozen_model.fit(X,y,frozen_context)
            json.dumps(frozen_model.export_model_params(),allow_nan=False)
            json.dumps(frozen_model.native_history(),allow_nan=False)
            checked_predictions(frozen_model.predict(X.iloc[:5],{'kind':request.get('prediction_kind','value')}),5,
                                request.get('prediction_kind','value'))
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
    return {'status':'completed'}


def train(request, root):
    import numpy as np
    import optuna
    from .pipeline_contracts import TaskContext,HypothesisPlan,TrialResult
    from .inputs import read_table,features,targets
    from .splitting import split_indices,group_keys
    from .sdk import load_module
    from .evaluation import score_predictions,prediction_kind,better
    from .persistence import atomic_json,tree_hash,file_hash
    context = TaskContext.model_validate(request['context'])
    plan = HypothesisPlan.model_validate(request['plan'])
    if tree_hash(Path(request['code_dir'])) != request['code_hash']: raise ValueError('Candidate hash mismatch')
    module = load_module(request['code_dir'])
    frame = read_table(request['train_path'],table_format=context.table_formats.get(request['train_path']))
    X,y = features(frame,context), targets(frame,context)
    groups = group_keys(frame,context.task.group_columns)
    weights = frame[context.task.weight_column].to_numpy() if context.task.weight_column else None
    kind = prediction_kind(context.task.primary_metric,context.task.task_type)
    unsupervised = context.task.task_type in {'clustering','anomaly_detection'}
    best_score = None
    best_params = dict(plan.model_params)
    trials = []
    started = time.monotonic()
    best_model = None
    search_result = {}
    config = plan.model_dump(mode='json')
    fit_budget = request['worker_seconds']*.85
    control_action = None

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
        if not request.get('training_control'):
            return fallback(checkpoint,'Programmatic policy')
        decision=worker_checkpoint(root,checkpoint,seconds=request.get('training_control_seconds',30))
        control_action=decision.action
        return decision

    def save_progress(model, params, score, reason):
        params=model.export_model_params() if hasattr(model,'export_model_params') else params
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
            study_ref=search_result.get('study_ref'),hpo_batches=search_result.get('batches',[])))

    if request.get('fixed'):
        model = module.create_candidate(config)
        c = fit_context(context,request,work_dir=root/'fixed',groups=groups,weights=weights,seconds=fit_budget)
        model.fit(X,None if unsupervised else y,c)
        save_progress(model,best_params,None,'fixed_configuration')
    else:
        a,b,_ = split_indices(frame,context,request.get('inner_fraction',.2),request.get('seed',42)+1)
        xa,xb = X.iloc[a],X.iloc[b]
        ya = y.iloc[a] if y is not None else None
        yb = y.iloc[b] if y is not None else None
        stale = 0
        checkpoint=root/'progress.json'
        if plan.strategy=='optuna' and checkpoint.exists():
            saved=json.loads(checkpoint.read_text(encoding='utf-8'))
            best_score=saved.get('internal_score');best_params=saved['best_params']
            best_model=module.load_candidate(saved['model_artifact'])
            trials=[TrialResult.model_validate(t) for t in saved.get('trials',[])]
        def objective(sampled=None, number=None, seconds=None):
            nonlocal best_score,best_params,best_model,stale
            params = dict(plan.model_params)
            params.update(sampled or {})
            t0 = time.monotonic()
            try:
                model = module.create_candidate(dict(config,model_params=params))
                remaining = max(.1,fit_budget-(time.monotonic()-started))
                if seconds is not None: remaining=min(remaining,seconds)
                c = fit_context(context,request,work_dir=root/f'trial-{len(trials)}',X_valid=xb,
                    y_valid=None if unsupervised else yb,groups=groups[a] if groups is not None else None,
                    validation_groups=groups[b] if groups is not None else None,
                    validation_weight=weights[b] if weights is not None else None,
                    weights=weights[a] if weights is not None else None,seconds=remaining*.6 if plan.strategy!='native' else remaining)
                model.fit(xa,None if unsupervised else ya,c)
                prediction = model.predict(xb,{'kind':kind})
                score = score_predictions(yb,prediction,context,X=xb,groups=groups[b] if groups is not None else None,
                    weights=weights[b] if weights is not None else None)
                trials.append(TrialResult(number=number if number is not None else len(trials),params=params,status='completed',score=score,
                    duration_seconds=time.monotonic()-t0,kind='native_run' if plan.strategy=='native' else plan.strategy))
                if best_score is None or better(score,best_score,context.task.metric_direction):
                    selected_params=model.export_model_params() if hasattr(model,'export_model_params') else params
                    best_score,best_params,best_model,stale = score,selected_params,model,0
                    save_progress(model,selected_params,score,'best_trial_checkpoint')
                else: stale += 1
                atomic_json(root/'trials.json',trials)
                if (root/'progress.json').exists():
                    progress=json.loads((root/'progress.json').read_text(encoding='utf-8'))
                    progress['trials']=[t.model_dump(mode='json') for t in trials]
                    atomic_json(root/'progress.json',progress)
                if len(trials)==1 and plan.strategy!='optuna':
                    control('initial_fit')
                return score
            except optuna.TrialPruned: raise
            except Exception as error:
                stale += 1
                trials.append(TrialResult(number=number if number is not None else len(trials),params=params,status='failed',error=str(error)[:2000],duration_seconds=time.monotonic()-t0))
                atomic_json(root/'trials.json',trials)
                if (root/'progress.json').exists():
                    progress=json.loads((root/'progress.json').read_text(encoding='utf-8'))
                    progress['trials']=[t.model_dump(mode='json') for t in trials]
                    atomic_json(root/'progress.json',progress)
                if plan.strategy!='optuna':control('failure',error=str(error)[:2000])
                raise

        if plan.strategy=='optuna':
            from .hpo import run_search
            def search_checkpoint(details):
                return control('batch',details)
            search_result=run_search(objective,directory=root/'hpo',identity=dict(code=request['code_hash'],
                train_hash=file_hash(Path(request['train_path'])),seed=request.get('seed',42),inner_fraction=request.get('inner_fraction',.2),
                task=context.task.model_dump(mode='json'),metric_options=context.metric_options,
                base_params=plan.model_params,tool_id=plan.tool_id,preprocessing=plan.preprocessing),
                space=plan.search_space,direction=context.task.metric_direction,seed=request.get('seed',42),
                max_trials=min(plan.max_trials,request.get('hpo_max_trials',15)),batch_size=request.get('hpo_batch_size',5),
                gain_threshold=request.get('hpo_gain_threshold',.005),seconds=min(request.get('hpo_seconds',600),fit_budget*.7),
                baseline_score=request.get('baseline_score'),checkpoint=search_checkpoint if request.get('training_control') else None,
                trial_checkpoint=(lambda details:control(details['stage'],details,error=details.get('error'))) if request.get('training_control') else None)
            control_action=search_result.get('control_action',control_action)
            trials=[TrialResult(number=t['number'],params=dict(plan.model_params,**t['params']),
                status='completed' if t['status']=='complete' else 'failed',score=t['score'],
                duration_seconds=t['duration_seconds'],error=t.get('error')) for t in search_result['trials']]
            atomic_json(root/'trials.json',trials)
        else: objective()
        if best_model is None: raise ValueError('No successful internal trial')
        if plan.strategy!='native' and control_action!='stop' and fit_budget-(time.monotonic()-started) > .2:
            # Refit selected parameters on all rows allowed by this outer split.
            model = module.create_candidate(dict(config,model_params=best_params))
            model.fit(X,None if unsupervised else y,fit_context(context,request,work_dir=root/'refit-round',
                groups=groups,weights=weights,seconds=max(.1,fit_budget-(time.monotonic()-started))))
            save_progress(model,best_params,best_score,search_result.get('stop_reason') or 'completed')
        else:
            save_progress(best_model,best_params,best_score,search_result.get('stop_reason') or
                          ('native_completed' if plan.strategy=='native' else 'budget_best_trial'))
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
            if module.split('.')[0] not in allowed and module not in {'mlmom.sdk','mlmom.ensemble'}:
                raise ValueError('API module is not exposed')
            value=importlib.import_module(module)
            for part in symbol.split('.'):value=getattr(value,part)
            try:signature=str(inspect.signature(value))
            except (ValueError,TypeError):signature=None
            result={'status':'completed','module':module,'symbol':symbol,'signature':signature,'documentation':(inspect.getdoc(value) or '')[:6000]}
        else: raise ValueError(f'Unknown worker action: {action}')
        atomic_json(root/'result.json',result)
        faulthandler.cancel_dump_traceback_later()
    except Exception:
        atomic_json(root/'result.json',dict(status='failed',error=traceback.format_exc()[-8000:]))
        sys.exit(1)


if __name__=='__main__': main()
