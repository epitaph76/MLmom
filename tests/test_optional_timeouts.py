"""Untimed execution still validates artifacts, folds and resource ownership."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from openai import OpenAI

from mlmom.agents import RoleClient
from mlmom.llm_client import LLMSettings
from mlmom.pipeline_contracts import RunConfig
from mlmom.resources import Budget, SandboxManager
from mlmom.worker import train
from tests import test_inner_validation as inner_validation
from mlmom.persistence import tree_hash
from mlmom.sdk import load_module
from mlmom.inputs import features
import numpy as np


class OptionalTimeoutTests(unittest.TestCase):
    def test_standard_budget_has_no_elapsed_time_cutoff(self):
        config=RunConfig(total_seconds=.01,final_reserve_seconds=1000)
        self.assertFalse(config.timeouts_enabled)
        budget=Budget(config,time.time()-100000)
        self.assertGreater(budget.snapshot().research_seconds,1e10)
        self.assertFalse(budget.snapshot().timeouts_enabled)
        self.assertEqual(budget.reserve,0)

    def test_api_request_disables_http_timeout_and_ignores_expired_role_window(self):
        requests=[]
        def transport(request):
            requests.append(request)
            return httpx.Response(200,json=dict(id='test',object='chat.completion',created=0,model='test',
                choices=[dict(index=0,finish_reason='stop',message=dict(role='assistant',content='{}'))]))
        client=RoleClient(LLMSettings(api_key='offline-test'),deadline=time.time()-100,
                          timeouts_enabled=False)
        client.client.close()
        client.client=OpenAI(api_key='offline-test',base_url='https://mock.invalid/v1',max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(transport)))
        self.addCleanup(client.client.close)
        with client.budget_scope('planner','expired',.001):
            self.assertEqual(client.complete([dict(role='user',content='{}')]),'{}')
        self.assertEqual(len(requests),1)
        self.assertTrue(all(value is None for value in requests[0].extensions['timeout'].values()))
        self.assertEqual(client.role_windows,{})

    def test_subprocess_imports_complete_despite_tiny_requested_window(self):
        config=RunConfig(total_seconds=.001,final_reserve_seconds=0,reserve_cpus=0,reserve_memory_gb=0)
        budget=Budget(config,time.time())
        with tempfile.TemporaryDirectory() as tmp,SandboxManager(budget=budget,max_workers=1) as sandbox:
            result=sandbox.run(dict(action='runtime_check'),Path(tmp),seconds=.001,cpus=1,memory_gb=min(2,budget.memory_gb))
            self.assertEqual(result['status'],'completed',result)
            request=json.loads((Path(tmp)/'request.json').read_text())
            self.assertFalse(request['timeouts_enabled'])
            self.assertGreater(result['duration_seconds'],.001)
            self.assertFalse((Path(tmp)/'limit.json').exists())

    def test_all_folds_and_full_refit_ignore_tiny_fit_window_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,_,_,job,request=inner_validation.InnerExecutionTests().request(Path(tmp))
            request.update(timeouts_enabled=False,worker_seconds=.001,hpo_seconds=.001)
            result=train(request,job)
            self.assertTrue(result['full_refit'])
            self.assertEqual(len(result['fold_results']),5)

    def test_direct_voting_candidate_runs_all_folds_refits_and_reloads(self):
        source='''from pathlib import Path
import cloudpickle
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
class Candidate:
    def fitted_components(self):return dict(zip(['linear','ridge'],self.model.estimators_))
    def fit(self,X,y,context):
        self.model=VotingRegressor([('linear',LinearRegression()),('ridge',Ridge(alpha=1.))],n_jobs=1)
        self.model.fit(X,y)
        return self
    def predict(self,X,spec):return self.model.predict(X)
    def save(self,path):Path(path,'estimator.pkl').write_bytes(cloudpickle.dumps(self.model))
def create_candidate(config):return Candidate()
def load_candidate(path):
    candidate=Candidate()
    candidate.model=cloudpickle.loads(Path(path,'estimator.pkl').read_bytes())
    return candidate
'''
        with tempfile.TemporaryDirectory() as tmp:
            frame,context,code,job,request=inner_validation.InnerExecutionTests().request(Path(tmp))
            (code/'candidate.py').write_text(source,encoding='utf-8')
            (code/'preprocessing.py').unlink()
            request.update(timeouts_enabled=False,worker_seconds=.001,code_hash=tree_hash(code))
            request['plan'].update(skill_ids=['sklearn','ensembles'],ensemble_components=[{'name':'linear','tool_id':'sklearn'},{'name':'ridge','tool_id':'sklearn'}])
            result=train(request,job)
            self.assertTrue(result['full_refit'])
            self.assertEqual(len(result['fold_results']),5)
            model=load_module(code).load_candidate(result['model_artifact'])
            prediction=model.predict(features(frame,context),{'kind':'value'})
            self.assertEqual(len(prediction),len(frame))
            self.assertTrue(np.isfinite(prediction).all())

    def test_untimed_pipeline_ignores_tiny_role_csv_smoke_and_fit_windows(self):
        from mlmom.pipeline import Pipeline
        from tests.test_pipeline import dataset,only_sklearn
        from tests.fakes import OfflineLLM
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);task=dataset(root/'input')
            config=RunConfig(total_seconds=.001,final_reserve_seconds=1000,max_rounds=1,max_candidates=1,
                max_workers=1,reserve_cpus=0,reserve_memory_gb=0,hpo_enabled=False,
                coding_seconds=.001,smoke_seconds=.001,initial_experiment_seconds=.001,
                prediction_seconds=.001,reconnaissance_query_seconds=.001,result_validation_seconds=.001)
            with patch('mlmom.pipeline.installed_tools',side_effect=only_sklearn):
                pipeline=Pipeline(task,config,run_dir=root/'run',client=OfflineLLM())
                result=pipeline.run()
            self.assertEqual(result['status'],'completed',result)
            self.assertEqual(result['submission']['model_role'],'submission_refit')

    def test_untimed_search_stops_at_trial_count_and_preserves_resume_policy(self):
        from mlmom.hpo import run_search
        with tempfile.TemporaryDirectory() as tmp:
            calls=[]
            def objective(params,number,seconds):
                calls.append(number);time.sleep(.01)
                return params['alpha']
            args=dict(directory=Path(tmp),identity={'test':'untimed'},space={'alpha':dict(type='float',low=.1,high=1.)},
                direction='minimize',seed=42,max_trials=2,batch_size=2,seconds=.001,timeouts_enabled=False)
            result=run_search(objective,**args)
            self.assertEqual(len(calls),2)
            self.assertEqual(result['stop_reason'],'max_trials')
            self.assertIsNone(result['deadline'])
            run_search(objective,**args)
            self.assertEqual(len(calls),2)
            with self.assertRaisesRegex(ValueError,'same timeout policy'):
                run_search(objective,**(args|dict(timeouts_enabled=True)))


if __name__=='__main__':unittest.main()
