from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from mlmom.adapters import LibraryCandidate,load_adapter
from mlmom.hpo import run_search,relative_gain
from mlmom.sdk import FitContext


class PersistentSearchTests(unittest.TestCase):
    def search(self,path,objective,**options):
        settings=dict(directory=path,identity={'code':'a','split':'one'},
            space={'x':dict(type='int',low=1,high=10)},direction='minimize',seed=42,
            max_trials=15,batch_size=5,gain_threshold=.005,seconds=30,baseline_score=100.)
        settings.update(options)
        return run_search(objective,**settings)

    def test_small_improvement_is_preserved_but_does_not_continue(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=self.search(Path(tmp),lambda p,n,s:99.8)
            self.assertEqual(len(result['trials']),5)
            self.assertEqual(result['stop_reason'],'hpo_plateau')
            self.assertAlmostEqual(result['batches'][0]['after'],99.8)
            self.assertAlmostEqual(result['batches'][0]['relative_gain'],.002)

    def test_continues_in_same_study_without_new_objective_definition(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls=[]
            def objective(p,n,s):calls.append(n);return 100.-n
            result=self.search(Path(tmp),objective)
            self.assertEqual(calls,list(range(15)))
            self.assertEqual(len(result['batches']),3)
            self.assertEqual(result['stop_reason'],'max_trials')
            again=self.search(Path(tmp),lambda *args:self.fail('Completed search reran'))
            self.assertEqual(again['study_name'],result['study_name'])
            self.assertEqual(again['trials'],result['trials'])

    def test_resume_interrupted_trial_and_isolate_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            def interrupted(p,n,s):
                if n==2:raise KeyboardInterrupt()
                return 99.8
            with self.assertRaises(KeyboardInterrupt):self.search(path,interrupted)
            from mlmom.hpo import interrupt_search
            states=interrupt_search(path,'time_limit')
            self.assertEqual(states[0]['trials'][2]['status'],'fail')
            self.assertEqual(states[0]['trials'][2]['error'],'time_limit')
            result=self.search(path,lambda p,n,s:99.8)
            self.assertEqual(len(result['trials']),5)
            self.assertEqual(result['trials'][2]['status'],'fail')
            other=self.search(path,lambda p,n,s:99.8,identity={'code':'b','split':'one'})
            self.assertNotEqual(result['study_name'],other['study_name'])
            changed_space=self.search(path,lambda p,n,s:99.8,space={'x':dict(type='int',low=1,high=20)})
            self.assertNotEqual(result['study_name'],changed_space['study_name'])

    def test_deadline_failure_and_metric_direction(self):
        self.assertAlmostEqual(relative_gain(.5,.505,'maximize'),.01)
        self.assertAlmostEqual(relative_gain(100,99,'minimize'),.01)
        self.assertTrue(np.isfinite(relative_gain(0,.001,'maximize')))
        with tempfile.TemporaryDirectory() as tmp:
            def failing(*args):raise ValueError('invalid trial')
            result=self.search(Path(tmp),failing,max_trials=5)
            self.assertEqual(result['stop_reason'],'no_successful_trials')
            self.assertTrue(all(t['error']=='invalid trial' for t in result['trials']))
        with tempfile.TemporaryDirectory() as tmp:
            def slow(*args):time.sleep(.04);return 1.
            result=self.search(Path(tmp),slow,seconds=.02)
            self.assertEqual(result['stop_reason'],'hpo_deadline')
            self.assertLess(len(result['trials']),5)


class DataPathTests(unittest.TestCase):
    def test_lightgbm_string_classes_exact_stopping_metric_during_fit(self):
        from tests.test_execution import execution_case
        from mlmom.inputs import features,targets
        from mlmom.evaluation import score_predictions
        from sklearn.model_selection import train_test_split
        for task in ['binary_classification','multiclass_classification']:
            with self.subTest(task=task),tempfile.TemporaryDirectory() as tmp:
                frame,context,_,_=execution_case(Path(tmp),task)
                X=features(frame,context)
                y=targets(frame,context).map({0:'обычный',1:'редкий',2:'третий'})
                context.classes=sorted(y.unique().tolist())
                a,b=train_test_split(np.arange(len(X)),test_size=.25,stratify=y,random_state=42)
                fit=FitContext(task_type=task,metric='f1_macro',direction='maximize',seed=42,cpus=1,memory_gb=2,
                    time_limit_seconds=20,classes=context.classes,validation_X=X.iloc[b],validation_y=y.iloc[b],task_context=context)
                candidate=LibraryCandidate(dict(tool_id='lightgbm',model_params={'n_estimators':80,'min_child_samples':2}))
                candidate.fit(X.iloc[a],y.iloc[a],fit)
                predicted=candidate.predict(X.iloc[b],{'kind':'label'})
                expected=score_predictions(y.iloc[b],predicted,context,X=X.iloc[b])
                self.assertAlmostEqual(candidate.model.best_score_['valid_0']['f1_macro'],expected)
                self.assertGreater(candidate.export_model_params()['n_estimators'],0)
                candidate.save(tmp)
                np.testing.assert_array_equal(load_adapter(tmp).predict(X.iloc[b],{'kind':'label'}),predicted)

    def test_three_feedback_cycles_and_real_optuna_worker(self):
        from unittest.mock import patch
        from tests.test_pipeline import dataset,only_sklearn
        from tests.fakes import OfflineLLM
        from mlmom.pipeline import Pipeline
        from mlmom.pipeline_contracts import RunConfig
        from mlmom.persistence import ExperimentStore
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);task=dataset(root/'inputs')
            # This tests feedback cycles, independently of host import/startup latency.
            config=RunConfig(timeouts_enabled=False,total_seconds=600,final_reserve_seconds=60,max_rounds=4,target_completed_rounds=3,fixed_outer_split=True,
                hpo_max_trials=5,hpo_seconds=45,experiment_seconds=60,initial_experiment_seconds=45,
                diagnostic_seconds=5,smoke_seconds=20,prediction_seconds=20,reserve_cpus=0,reserve_memory_gb=0,max_workers=1)
            class OneBrokenReview(OfflineLLM):
                broken=False
                def generate(self,role,instruction,payload,schema,tools=None):
                    if role=='plan_validator' and payload.get('round_number')==2 and not self.broken:
                        self.broken=True
                        from mlmom.agents import StructuredOutputError
                        raise StructuredOutputError('Injected malformed review; no semantic approval')
                    return super().generate(role,instruction,payload,schema,tools)
            client=OneBrokenReview(seconds=45)
            with patch('mlmom.pipeline.installed_tools',only_sklearn):
                result=Pipeline(task,config,run_dir=root/'run',client=client).run()
            self.assertEqual(result['status'],'completed',result)
            self.assertTrue(result['process_acceptance']['passed'],result)
            self.assertEqual(result['process_acceptance']['completed_rounds'],3)
            self.assertEqual(result['stop_reason'],'target_completed_rounds')
            self.assertTrue(any(f['kind']=='planning_failed' for f in result['failure_causes']))
            store=ExperimentStore(root/'run')
            try:
                splits=store.list('split')
                self.assertEqual(len({s['split_id'] for s in splits}),1)
                trials=[t for r in store.list('experiment') if not r['incumbent'] for t in r['trials'] if t['kind']=='optuna']
                self.assertGreaterEqual(sum(t['status']=='completed' for t in trials),5)
                self.assertEqual(sum(role=='coding_agent' for role,p in client.calls),3, [(p.get('runtime_feedback'),p.get('feedback'),p.get('errors')) for role,p in client.calls if role=='coding_agent'])
                later=[p for role,p in client.calls if role=='hypothesis_generator' and p['round_number']>1]
                self.assertTrue(all(p['champion']['evaluation']['diagnostics'] for p in later))
                self.assertTrue(any(p.get('planning_attempt')==1 and p['round_number']==2
                                    for role,p in client.calls if role=='plan_validator'))
                first=store.get('milestone','first_submission')
                self.assertIsNotNone(first)
            finally:store.close()

    def test_unicode_sparse_mixed_missing_unseen_and_save_load(self):
        X=pd.DataFrame({'случайное_поле':[f'Длинное описание документа номер {i} с полезными русскими словами и текстом' for i in range(50)],
            'число':np.arange(50,dtype=float),'тип':['a','b']*25})
        X.loc[0,'случайное_поле']=None;X.loc[1,'число']=np.nan
        for task in ['regression','multiclass_classification']:
            context=FitContext(task_type=task,metric='rmse' if task=='regression' else 'f1_macro',
                direction='minimize' if task=='regression' else 'maximize',seed=42,cpus=1,memory_gb=2,time_limit_seconds=20,
                classes=['a','b'] if task!='regression' else [])
            y=pd.Series(np.arange(50,dtype=float) if task=='regression' else ['a','b']*25)
            model=LibraryCandidate(dict(tool_id='sklearn',model_params={'estimator':'linear'},preprocessing=['tfidf']))
            model.fit(X,y,context)
            self.assertTrue(sparse.issparse(model.preprocessor.transform(model._clean(X))))
            test=X.head(3).copy();test['тип']='новая категория';test['случайное_поле']=['',None,'未知のテキスト']
            kind='value' if task=='regression' else 'label'
            before=model.predict(test,{'kind':kind})
            with tempfile.TemporaryDirectory() as tmp:
                model.save(tmp);after=load_adapter(tmp).predict(test,{'kind':kind})
            np.testing.assert_array_equal(before,after)

    def test_native_categories_early_stopping_and_frozen_refit(self):
        from tests.test_execution import execution_case
        from mlmom.inputs import read_table,features,targets
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            frame,context,plan,code=execution_case(root,'regression')
            X=features(frame,context);y=targets(frame,context)
            X['renamed_category']=['a','b']*(len(X)//2)
            for tool in ['lightgbm','catboost']:
                params={'n_estimators':100} if tool=='lightgbm' else {'iterations':100}
                fit=FitContext(task_type='regression',metric='rmse',direction='minimize',seed=42,cpus=1,memory_gb=2,
                    time_limit_seconds=20,validation_X=X.iloc[40:],validation_y=y.iloc[40:],task_context=context)
                model=LibraryCandidate(dict(tool_id=tool,model_params=params))
                model.fit(X.iloc[:40],y.iloc[:40],fit)
                frozen=model.export_model_params()
                key='n_estimators' if tool=='lightgbm' else 'iterations'
                self.assertGreater(frozen[key],0);self.assertLessEqual(frozen[key],100)
                refit=LibraryCandidate(dict(tool_id=tool,model_params=frozen))
                fixed=FitContext(task_type='regression',metric='rmse',direction='minimize',seed=42,cpus=1,memory_gb=2,time_limit_seconds=20)
                refit.fit(X,y,fixed)
                test=X.head(3).copy();test['renamed_category']='unseen'
                self.assertTrue(np.isfinite(refit.predict(test,{'kind':'value'})).all())
                with tempfile.TemporaryDirectory() as saved:
                    refit.save(saved)
                    np.testing.assert_array_equal(refit.predict(test,{'kind':'value'}),load_adapter(saved).predict(test,{'kind':'value'}))


if __name__=='__main__':unittest.main()
