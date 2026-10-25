"""Execute the actual reference candidates against the SDK, without an LLM."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mlmom.evaluation import prediction_kind, score_predictions
from mlmom.model_spec import frozen_plan
from mlmom.persistence import atomic_json, tree_hash
from mlmom.registry import installed_tools
from mlmom.sdk import FitContext, load_module
from tests.test_execution import execution_case


DATAFRAME_PREPROCESSOR = '''import numpy as np
import pandas as pd
MLMOM_PREPROCESSOR_V1 = True
class Preprocessor:
    def __init__(self,config):self.tool=config['tool_id']
    def fit_transform(self,X,y,context):
        self.columns=list(X);self.fills={};self.levels={}
        for c in X:
            if pd.api.types.is_numeric_dtype(X[c]):
                self.fills[c]=float(X[c].median()) if X[c].notna().any() else 0.
            else:self.levels[c]=X[c].dropna().astype(str).unique().tolist()
        return self.transform(X)
    def transform(self,X):
        X=X.loc[:,self.columns].copy()
        for c,v in self.fills.items():X[c]=X[c].replace([np.inf,-np.inf],np.nan).fillna(v)
        for c,levels in self.levels.items():
            if self.tool=='catboost':X[c]=X[c].fillna('__missing__').astype(str)
            else:X[c]=pd.Categorical(X[c].map(lambda v:str(v) if pd.notna(v) else np.nan),categories=levels)
        return X
def create_preprocessor(config):return Preprocessor(config)
'''


def reference_source(backend):
    return (ROOT/'skills/coding_agent'/backend/'REFERENCE.md').read_text(encoding='utf-8').split('```python\n',1)[1].split('```',1)[0]


PARAMETERS = {
    'sklearn':dict(n_estimators=8), 'lightgbm':dict(n_estimators=8,min_child_samples=1),
    'catboost':dict(iterations=8,depth=3), 'xgboost':dict(n_estimators=8,max_depth=3),
    'statsmodels':{},'flaml':{},'torch':dict(hidden_size=8,epochs=2,batch_size=16),
    'autogluon':dict(hyperparameters={'GBM':{'num_boost_round':8,'min_data_in_leaf':1}}),
    'lightautoml':dict(learner='lgb',num_trees=8,learner_params={'num_leaves':7,'min_data_in_leaf':1}),
}


def run_case(output, backend, kind, variant='ordinary', metric=None):
    root=output/f'{backend}-{kind}-{variant}'
    root.mkdir(parents=True,exist_ok=False)
    frame,context,plan,code=execution_case(root,kind,tool=backend)
    context.task.validation_strategy='holdout'
    single_class = kind in {'binary_classification','multiclass_classification'}
    if single_class:
        mapping={0:'Class A',1:'Class B',2:'Class C'}
        frame['t0']=frame['t0'].map(mapping)
        context.classes=sorted(frame['t0'].unique())
    context.task.primary_metric=metric or ('roc_auc' if kind=='binary_classification' else
                                 'log_loss' if kind in {'multiclass_classification','multilabel_classification'} else 'rmse')
    context.task.metric_direction='minimize' if context.task.primary_metric in {'rmse','log_loss'} else 'maximize'
    context.metric_options={}
    plan.tool_id=backend;plan.skill_ids=[backend];plan.model_params=PARAMETERS[backend].copy()
    plan.strategy='native' if backend in {'flaml','autogluon','lightautoml'} else 'single'
    plan.max_trials=2;plan.time_limit_seconds=45
    plan.stopping_policy='early_stopping' if backend in {'lightgbm','xgboost','catboost','torch'} else 'fixed'
    if variant=='log':
        frame['t0']=frame['t0'].abs()+1
        plan.target_transform=type(plan.target_transform)(name='log1p')
        plan.prediction_policy=type(plan.prediction_policy)(clip_min=0)
    if variant=='category':
        frame['category']=np.resize(['red','blue',None],len(frame))
        context.feature_columns.append('category')
        if backend=='xgboost':plan.model_params['enable_categorical']=True
    frame.to_csv(root/'train.csv',index=False)
    source=reference_source(backend)
    (code/'candidate.py').write_text(source,encoding='utf-8')
    if backend in {'autogluon','lightautoml','lightgbm','catboost','xgboost'}:
        pre=DATAFRAME_PREPROCESSOR
    else:
        pre=(ROOT/'skills/preprocessing_agent/preprocessing/REFERENCE.md').read_text(encoding='utf-8').split('```python\n',1)[1].split('```',1)[0]+'\nMLMOM_PREPROCESSOR_V1 = True\n'
    (code/'preprocessing.py').write_text(pre,encoding='utf-8')
    plan.prediction_kind=prediction_kind(context.task.primary_metric,kind)
    X=frame[context.feature_columns]
    y=frame[context.task.target_columns] if len(context.task.target_columns)>1 else frame['t0']
    a,b=train_test_split(np.arange(len(frame)),test_size=.3,random_state=42,
                         stratify=y if single_class else None)
    weighted=backend in {'flaml','lightgbm','xgboost','catboost','torch'}
    train_weight=np.linspace(1,2,len(a)) if weighted else None
    valid_weight=np.linspace(2,1,len(b)) if weighted else None
    work=root/'work';work.mkdir()
    c=FitContext(kind,context.task.primary_metric,context.task.metric_direction,42,1,3,45,
        classes=context.classes,validation_X=X.iloc[b],validation_y=y.iloc[b],
        sample_weight=train_weight,validation_weight=valid_weight,work_dir=str(work),
        task_context=context,timeouts_enabled=False)
    module=load_module(code)
    model=module.create_candidate(plan.model_dump(mode='json'))
    model.fit(X.iloc[a],y.iloc[a],c)
    before=np.asarray(model.predict(X.iloc[b],{'kind':plan.prediction_kind}))
    score=score_predictions(y.iloc[b],before,context,weights=valid_weight)
    selected=model.export_model_params()
    json.dumps(selected,allow_nan=False)
    inner=model.estimator.estimator
    stopping=getattr(inner,'stopping_info',None)
    if backend=='lightgbm' and inner.use_stopping:
        recorded=inner.model.evals_result_['valid_0']['primary']
        assert np.isclose(score,(max if context.task.metric_direction=='maximize' else min)(recorded),rtol=1e-5,atol=1e-6)
    if backend=='xgboost' and inner.use_stopping:
        assert np.isclose(score,float(inner.model.best_score),rtol=1e-5,atol=1e-6)
    if backend=='flaml':
        expected=-score if context.task.metric_direction=='maximize' else score
        assert np.isclose(selected['__native_frozen']['validation_loss'],expected,rtol=1e-5,atol=1e-6)
    artifact=root/'model';artifact.mkdir();model.save(artifact)
    relocated=root/'relocated'
    shutil.copytree(artifact,relocated)
    # Verify both model relocation and independence from the training work directory.
    for old,new in ((artifact,root/'old-model'),(work,root/'old-work')):
        assert old.resolve().is_relative_to(root.resolve()) and new.resolve().is_relative_to(root.resolve())
        old.rename(new)
    restored=module.load_candidate(relocated)
    after=np.asarray(restored.predict(X.iloc[b],{'kind':plan.prediction_kind}))
    if np.issubdtype(before.dtype,np.number):np.testing.assert_allclose(before,after,rtol=1e-6,atol=1e-6)
    else:np.testing.assert_array_equal(before,after)
    if single_class:
        probability=np.asarray(restored.predict(X.iloc[b],{'kind':'probability'}))
        labels=np.asarray(restored.predict(X.iloc[b],{'kind':'label'}))
        assert probability.shape==(len(b),len(context.classes))
        np.testing.assert_allclose(probability.sum(axis=1),1,atol=1e-6)
        ordered=np.sort(probability,axis=1);untied=(ordered[:,-1]-ordered[:,-2])>1e-8
        np.testing.assert_array_equal(np.asarray(context.classes)[probability[untied].argmax(axis=1)],labels[untied])
    elif kind=='multilabel_classification':
        probability=np.asarray(restored.predict(X.iloc[b],{'kind':'probability'}))
        labels=np.asarray(restored.predict(X.iloc[b],{'kind':'label'}))
        assert probability.shape==(len(b),len(context.task.target_columns))
        np.testing.assert_array_equal(probability>=.5,labels)
    if variant=='category':
        probe=X.iloc[:3].copy();probe['category']='unseen'
        assert np.isfinite(np.asarray(restored.predict(probe,{'kind':'value'}))).all()
    refit=root/'refit-work';refit.mkdir()
    fixed=plan.model_copy(update=dict(model_params=selected,strategy='single',stopping_policy='fixed'))
    final=module.create_candidate(fixed.model_dump(mode='json'))
    final.fit(X,y,replace(c,validation_X=None,validation_y=None,validation_weight=None,
        validation_scorer=None,sample_weight=np.linspace(1,2,len(X)) if weighted else None,work_dir=str(refit)))
    prediction=np.asarray(final.predict(X.iloc[b],{'kind':plan.prediction_kind}))
    assert len(prediction)==len(b)
    history=final.native_history()
    if backend in {'flaml','autogluon','lightautoml'}:
        assert history['new_trials']==0,history
    result=dict(backend=backend,task_type=kind,variant=variant,status='passed',score=score,
        weighted=weighted,code_hash=tree_hash(code),selected_params=selected,stopping=stopping,
        native_refit_history=history,relocated_equal=True,class_order=context.classes,refit_rows=len(X))
    atomic_json(root/'result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--backends',default=','.join(PARAMETERS))
    parser.add_argument('--tasks',help='Optional comma-separated task types for focused follow-up checks')
    parser.add_argument('--ordinary-only',action='store_true')
    parser.add_argument('--metric',help='Optional exact primary metric for focused checks')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    tools=installed_tools();results=[]
    for backend in args.backends.split(','):
        kinds=['regression'] if backend=='statsmodels' else ['regression','binary_classification','multiclass_classification']
        if args.tasks:kinds=args.tasks.split(',')
        variants=[(kind,'ordinary') for kind in kinds]
        if not args.ordinary_only:
            if backend in {'lightgbm','catboost','xgboost','torch','flaml'}:variants.append(('regression','log'))
            if backend in {'lightgbm','catboost','xgboost'}:variants.append(('regression','category'))
        for kind,variant in variants:
            started=time.monotonic()
            try:
                if not tools[backend].available:raise RuntimeError('Required backend unavailable')
                result=run_case(args.output,backend,kind,variant,args.metric)
            except Exception as error:
                result=dict(backend=backend,task_type=kind,variant=variant,status='failed',
                            error=str(error),traceback=traceback.format_exc())
            result['duration_seconds']=time.monotonic()-started
            results.append(result)
            atomic_json(args.output/'summary.json',dict(status='passed' if all(r['status']=='passed' for r in results) else 'failed',
                versions={k:t.version for k,t in tools.items()},results=results))
            print(json.dumps({k:result[k] for k in ('backend','task_type','variant','status','duration_seconds')},ensure_ascii=False),flush=True)
    return 0 if all(r['status']=='passed' for r in results) else 1


if __name__=='__main__':raise SystemExit(main())
