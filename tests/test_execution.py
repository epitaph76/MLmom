from __future__ import annotations
import json
import tempfile
import time
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from mlmom.contracts import TaskDefinition,ResultFormat
from mlmom.pipeline_contracts import TaskContext,InputFiles,Hypothesis,HypothesisPlan
from mlmom.resources import SandboxManager
from mlmom.sdk import FitContext
from mlmom.worker import synthetic
from mlmom.persistence import file_hash,tree_hash
from mlmom.evaluation import prediction_kind,evaluate
from mlmom.splitting import make_split,group_keys
from mlmom.inputs import features,targets,read_table
from tests.fakes import ADAPTER_CODE


def execution_case(root, kind, tool='sklearn'):
    X,y=synthetic(kind)
    train=X.copy();train['id']=np.arange(len(X))
    names=[]
    if y is not None:
        a=np.asarray(y)
        if a.ndim==1:a=a[:,None]
        names=[f't{i}' for i in range(a.shape[1])]
        for i,name in enumerate(names):train[name]=a[:,i]
    groups=[]
    if kind=='ranking':train['query_id']=np.repeat(np.arange(12),5);groups=['query_id']
    data=root/'train.csv';train.to_csv(data,index=False)
    metric='f1_macro' if 'classification' in kind else 'ndcg' if kind=='ranking' else 'silhouette' if kind=='clustering' else 'roc_auc' if kind=='anomaly_detection' else 'rmse'
    direction='minimize' if metric=='rmse' else 'maximize'
    outputs=names or ['cluster']
    task=TaskDefinition(objective='Synthetic integration',task_type=kind,target_columns=names,id_columns=['id'],
        group_columns=groups,excluded_feature_columns=['id']+names,primary_metric=metric,metric_direction=direction,
        validation_strategy='group_kfold' if groups else 'stratified_kfold' if 'classification' in kind else 'kfold',
        result_format=ResultFormat(file_name='submission.csv',columns=['id']+outputs,id_columns=['id'],prediction_columns=outputs))
    files=InputFiles(task=str(root/'task.md'),root=str(root),train=str(data),test=str(data),dictionaries=[])
    context=TaskContext(task_id='test',task=task,files=files,profile_ref='',feature_columns=X.columns.tolist(),
        dataset_hash=file_hash(data),classes=[0,1,2] if kind=='multiclass_classification' else [0,1] if kind in {'binary_classification','anomaly_detection'} else [])
    params={'iterations':10,'depth':3} if tool=='catboost' else {'n_clusters':3} if kind=='clustering' else {'n_estimators':8}
    plan=HypothesisPlan(hypothesis=Hypothesis(hypothesis_id='h1',description='Integration',rationale='test'),
        tool_id=tool,skill_ids=[tool],model_params=params,prediction_kind=prediction_kind(metric,kind),time_limit_seconds=20,cpus=1,memory_gb=2)
    code=root/'code';code.mkdir();(code/'candidate.py').write_text(ADAPTER_CODE,encoding='utf-8')
    return train,context,plan,code


class ExecutionTests(unittest.TestCase):
    def test_all_non_temporal_task_families_fit_evaluate_and_reload(self):
        cases=['binary_classification','multioutput_regression','multilabel_classification','clustering','anomaly_detection','ranking']
        for kind in cases:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);tool='catboost' if kind=='ranking' else 'sklearn'
                if kind=='ranking':
                    from mlmom.registry import installed_tools
                    if installed_tools()['catboost'].version is None:continue
                frame,context,plan,code=execution_case(root,kind,tool)
                split=make_split(frame,context,'r1',.25,42)
                training=root/'allowed-train.csv';frame.iloc[split.train_indices].to_csv(training,index=False)
                manager=SandboxManager();self.addCleanup(manager.close)
                request=dict(action='train',code_dir=str(code),code_hash=tree_hash(code),context=context.model_dump(mode='json'),
                    plan=plan.model_dump(mode='json'),train_path=str(training),seed=42)
                result=manager.run(request,root/'fit',seconds=25,memory_gb=2,cpus=1)
                self.assertEqual(result['status'],'completed',result)
                holdout=frame.iloc[split.holdout_indices]
                path=root/'features.csv';features(holdout,context).to_csv(path,index=False)
                prediction=manager.run(dict(action='predict',code_dir=str(code),code_hash=tree_hash(code),model_artifact=result['model_artifact'],
                    features_path=str(path),kinds=[plan.prediction_kind]),root/'inference',seconds=15,memory_gb=2,cpus=1)
                self.assertEqual(prediction['status'],'completed',prediction)
                values=read_table(prediction['predictions'][plan.prediction_kind]).to_numpy()
                if values.shape[1]==1:values=values[:,0]
                evaluation=evaluate('run',split.split_id,targets(holdout,context),values,context,X=features(holdout,context),
                    groups=group_keys(holdout,context.task.group_columns))
                self.assertTrue(np.isfinite(evaluation.score))
                self.assertEqual(len(set(split.train_indices)&set(split.holdout_indices)),0)
                if kind=='ranking':
                    self.assertFalse(set(frame.iloc[split.train_indices].query_id)&set(holdout.query_id))

    def test_optuna_trials_best_configuration_and_patience(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);frame,context,plan,code=execution_case(root,'regression')
            plan.strategy='optuna';plan.search_space={'max_depth':{'type':'int','low':1,'high':4}}
            plan.max_trials=4;plan.patience_trials=2
            manager=SandboxManager();self.addCleanup(manager.close)
            result=manager.run(dict(action='train',code_dir=str(code),code_hash=tree_hash(code),context=context.model_dump(mode='json'),
                plan=plan.model_dump(mode='json'),train_path=str(root/'train.csv'),seed=42),root/'hpo',seconds=25,memory_gb=2,cpus=1)
            self.assertEqual(result['status'],'completed',result)
            self.assertGreaterEqual(len(result['trials']),1)
            self.assertIn('max_depth',result['best_params'])
            scores=[r['score'] for r in result['trials'] if r['score'] is not None]
            self.assertAlmostEqual(result['internal_score'],min(scores))
            self.assertTrue((Path(result['model_artifact'])/'adapter.pkl').is_file())

    def test_worker_timeout_memory_and_secret_free_environment(self):
        for reason,seconds,memory in [('time_limit',2,2),('memory_limit',15,.05)]:
            with self.subTest(reason=reason),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);code=root/'code';code.mkdir()
                (code/'candidate.py').write_text('import time\ndef create_candidate(config):\n    time.sleep(60)\ndef load_candidate(path):\n    pass\n',encoding='utf-8')
                manager=SandboxManager();self.addCleanup(manager.close)
                before=time.monotonic()
                result=manager.run(dict(action='smoke',code_dir=str(code),config={'tool_id':'sklearn'}),root/'job',seconds=seconds,memory_gb=memory,cpus=1)
                self.assertEqual(result['status'],'failed')
                self.assertEqual(result['stop_reason'],reason,result)
                self.assertLess(time.monotonic()-before,seconds+5)

    def test_adapter_blend_and_oof_stacking(self):
        from mlmom.ensemble import EnsembleCandidate,load_ensemble
        for method in ['blend','stack']:
            with self.subTest(method=method),tempfile.TemporaryDirectory() as tmp:
                X,y=synthetic('binary_classification')
                candidate=EnsembleCandidate({'model_params':{'method':method},'ensemble_components':[
                    {'tool_id':'sklearn','model_params':{'n_estimators':5,'max_depth':2}},
                    {'tool_id':'sklearn','model_params':{'n_estimators':5,'max_depth':4}}]})
                context=FitContext(task_type='binary_classification',metric='f1_macro',direction='maximize',seed=42,
                    cpus=1,memory_gb=2,time_limit_seconds=20,classes=[0,1],work_dir=tmp)
                candidate.fit(X,y,context);candidate.save(tmp)
                restored=load_ensemble(tmp)
                self.assertEqual(restored.predict(X,{'kind':'probability'}).shape,(60,2))
                self.assertTrue(np.array_equal(candidate.predict(X,{'kind':'label'}),restored.predict(X,{'kind':'label'})))


if __name__=='__main__':unittest.main()
