"""The confirmed validation strategy is executed strictly inside outer train."""
import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from mlmom.splitting import inner_protocol
from mlmom.persistence import tree_hash
from mlmom.sdk import load_module
from mlmom.worker import train
from tests.test_execution import execution_case


class InnerProtocolTests(unittest.TestCase):
    def test_all_strategies_are_reproducible_and_keep_outer_indices(self):
        for strategy, kind in [('holdout', 'regression'), ('kfold', 'regression'),
                               ('stratified_kfold', 'binary_classification'), ('group_kfold', 'regression')]:
            with self.subTest(strategy=strategy), tempfile.TemporaryDirectory() as tmp:
                frame, context, _, _ = execution_case(Path(tmp), kind)
                context.task = type(context.task).model_validate(dict(context.task.model_dump(), validation_strategy=strategy))
                if strategy == 'group_kfold':
                    frame['group'] = np.repeat(np.arange(12), 5)
                    context.task.group_columns = ['group']
                source = np.arange(len(frame)) * 3 + 100
                protocol = inner_protocol(frame, context, source_indices=source, outer_split_id='r2-current')
                self.assertEqual(protocol, inner_protocol(frame, context, source_indices=source, outer_split_id='r2-current'))
                self.assertEqual(protocol['n_splits'], 1 if strategy == 'holdout' else 5)
                seen = []
                for fold in protocol['folds']:
                    a, b = fold['train_indices'], fold['validation_indices']
                    self.assertFalse(set(a) & set(b))
                    self.assertEqual(fold['source_validation_indices'], source[b].tolist())
                    if strategy == 'group_kfold':
                        self.assertFalse(set(frame.iloc[a]['group']) & set(frame.iloc[b]['group']))
                    seen.extend(b)
                if strategy != 'holdout':
                    self.assertEqual(sorted(seen), list(range(len(frame))))

    def test_group_holdout_and_strict_infeasibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame, context, _, _ = execution_case(Path(tmp), 'regression')
            frame['group'] = np.repeat([0, 1], 30)
            context.task.group_columns = ['group']
            task = context.task.model_dump()
            for strategy in ['holdout', 'group_kfold', 'kfold']:
                context.task = type(context.task).model_validate(dict(task, validation_strategy=strategy))
                if strategy == 'holdout':
                    self.assertEqual(inner_protocol(frame, context)['implementation'], 'group_shuffle')
                else:
                    with self.assertRaisesRegex(ValueError, 'groups|Grouped'):
                        inner_protocol(frame, context)
            with self.assertRaises(ValidationError):
                type(context.task).model_validate(dict(task, validation_folds=1))

    def test_rare_class_and_missing_class_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame, context, _, _ = execution_case(Path(tmp), 'binary_classification')
            frame['t0'] = [1] * 3 + [0] * (len(frame) - 3)
            with self.assertRaisesRegex(ValueError, 'rare class'):
                inner_protocol(frame, context)
            frame['group'] = frame['t0']
            context.task = type(context.task).model_validate(dict(context.task.model_dump(),
                validation_strategy='group_kfold', validation_folds=2, group_columns=['group']))
            with self.assertRaisesRegex(ValueError, 'every class'):
                inner_protocol(frame, context)


RECORDING_CODE = '''from pathlib import Path
import json
import cloudpickle
import numpy as np
from sklearn.dummy import DummyRegressor
class Candidate:
    def __init__(self,config): self.params=config.get('model_params',{})
    def fit(self,X,y,context):
        assert context.validation_X is None
        self.size=len(X)
        self.value=float(self.params.get('constant',np.mean(y)))
        self.model=DummyRegressor(strategy='constant',constant=self.value).fit(X,y)
        Path(context.work_dir,'fit.json').write_text(json.dumps(dict(rows=len(X),groups=None if context.groups is None else context.groups.tolist())))
        return self
    def predict(self,X,spec): return np.full(len(X),self.value)
    def export_model_params(self): return self.params
    def save(self,path): Path(path,'candidate.pkl').write_bytes(cloudpickle.dumps(self))
def create_candidate(config): return Candidate(config)
def load_candidate(path): return cloudpickle.loads(Path(path,'candidate.pkl').read_bytes())
'''

PREPROCESSOR_CODE = '''import numpy as np
MLMOM_PREPROCESSOR_V1 = True
class Preprocessor:
    def fit_transform(self,X,y,context):
        assert context.validation_y is None
        assert not hasattr(self,'seen')
        self.seen=set(X['x'])
        self.mean=float(np.mean(y))
        return X
    def transform(self,X):
        assert not (self.seen & set(X['x']))
        return X
def create_preprocessor(config): return Preprocessor()
'''


class InnerExecutionTests(unittest.TestCase):
    def request(self, root):
        frame,context,plan,code=execution_case(root,'regression')
        frame['x']=np.arange(len(frame),dtype=float)
        plan.model_params={}
        frame.to_csv(root/'train.csv',index=False)
        (code/'candidate.py').write_text(RECORDING_CODE,encoding='utf-8')
        (code/'preprocessing.py').write_text(PREPROCESSOR_CODE,encoding='utf-8')
        job=root/'job';job.mkdir()
        return frame,context,code,job,dict(code_dir=str(code),code_hash=tree_hash(code),
            context=context.model_dump(mode='json'),plan=plan.model_dump(mode='json'),
            train_path=str(root/'train.csv'),worker_seconds=30,worker_cpus=1,worker_memory_gb=2,
            outer_split_id='r3-outer',outer_train_indices=list(range(100,160)))

    def test_fold_isolation_oof_aggregation_and_mandatory_full_refit(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame,context,code,job,request=self.request(Path(tmp))
            result=train(request,job)
            self.assertTrue(result['full_refit'])
            self.assertEqual(len(result['fold_results']),5)
            self.assertAlmostEqual(result['internal_score'],np.mean([f['score'] for f in result['fold_results']]))
            self.assertNotAlmostEqual(result['internal_score'],result['oof_score'],places=7)
            oof=json.loads(Path(result['oof_ref']).read_text())
            self.assertEqual(oof['source_indices'],list(range(100,160)))
            restored=load_module(code).load_candidate(result['model_artifact'])
            self.assertEqual(restored.estimator.size,len(frame))
            fitted=[json.loads(p.read_text())['rows'] for p in job.rglob('fit.json')]
            self.assertEqual(sorted(fitted),[48]*5+[60])

    def test_feature_duplicate_constraints_reach_preprocessing_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame,context,code,job,request=self.request(Path(tmp))
            frame.loc[1,context.feature_columns]=frame.loc[0,context.feature_columns].to_numpy()
            frame.to_csv(request['train_path'],index=False)
            context.task.duplicate_policy='group_features'
            request['context']=context.model_dump(mode='json')
            result=train(request,job)
            self.assertTrue(result['full_refit'])
            records=[json.loads(path.read_text()) for path in job.rglob('fit.json')]
            self.assertTrue(all(record['groups'] is not None and len(record['groups'])==record['rows'] for record in records))
            full=next(record for record in records if record['rows']==60)
            self.assertEqual(full['groups'][0],full['groups'][1])

    def test_incomplete_trial_never_creates_publishable_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,_,code,job,request=self.request(Path(tmp))
            source=RECORDING_CODE.replace('self.size=len(X)',"raise ValueError('fold failure')")
            (code/'candidate.py').write_text(source,encoding='utf-8');request['code_hash']=tree_hash(code)
            with self.assertRaisesRegex(ValueError,'fold failure'): train(request,job)
            self.assertFalse((job/'progress.json').exists())
            trial=json.loads((job/'trials.json').read_text())[0]
            self.assertEqual(trial['status'],'failed');self.assertIsNone(trial['score'])

    def test_stop_still_refits_and_resume_preserves_deadline(self):
        from unittest.mock import patch
        from mlmom.pipeline_contracts import TrainingDecision
        with tempfile.TemporaryDirectory() as tmp:
            frame,_,code,job,request=self.request(Path(tmp))
            request['training_control']=True
            with patch('mlmom.training_control.worker_checkpoint',return_value=TrainingDecision(action='stop',reason='done')):
                result=train(request,job)
            self.assertEqual(result['stop_reason'],'training_agent_stop')
            self.assertEqual(load_module(code).load_candidate(result['model_artifact']).estimator.size,len(frame))
            expired=dict(deadline=time.time()-1)
            (job/'training-window.json').write_text(json.dumps(expired))
            request['worker_seconds']=10000
            with self.assertRaises(TimeoutError): train(request,job)
            self.assertEqual(json.loads((job/'training-window.json').read_text()),expired)

    def test_resume_reuses_completed_single_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,_,_,job,request=self.request(Path(tmp))
            train(request,job)
            result=train(request,job)
            self.assertEqual(len(result['trials']),1)
            sizes=[json.loads(p.read_text())['rows'] for p in job.rglob('fit.json')]
            self.assertEqual(sorted(sizes),[48]*5+[60]*2)

    def test_failed_study_trial_cannot_supply_selected_configuration(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            _,_,_,job,request=self.request(Path(tmp))
            request['plan'].update(strategy='optuna',search_space={'constant':dict(type='float',low=0,high=1000)})
            def interrupted_study(objective,**kwargs):
                objective({'constant':0},0,10)
                score=objective({'constant':1000},1,10)
                # The first objective finished but its study transaction was
                # interrupted before tell(); resume marks it FAIL.
                return dict(trials=[dict(number=0,params={'constant':0},status='fail',score=None,duration_seconds=1,error='interrupted'),
                    dict(number=1,params={'constant':1000},status='complete',score=score,duration_seconds=1,error=None)])
            with patch('mlmom.hpo.run_search',side_effect=interrupted_study):result=train(request,job)
            self.assertEqual(result['best_params']['constant'],1000)
            self.assertEqual(result['internal_score'],result['trials'][1]['score'])


if __name__ == '__main__':
    unittest.main()
