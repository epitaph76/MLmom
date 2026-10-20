from __future__ import annotations
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from mlmom.coding import CodeBundle
from mlmom.pipeline import Pipeline
from mlmom.pipeline_contracts import RunConfig,ValidationReport,BudgetSnapshot,ExperimentRun,CandidateVersion
from mlmom.persistence import ExperimentStore,file_hash
from mlmom.planning import validate_plan
from mlmom.registry import SkillRegistry,installed_tools
from mlmom.resources import SandboxManager
from mlmom.submission import publish_submission
from tests.fakes import OfflineLLM,ADAPTER_CODE
from tests.test_pipeline import dataset,only_sklearn
from tests.test_execution import execution_case

import mlmom
ROOT=Path(mlmom.__file__).resolve().parent.parent


def broken_code(threshold):
    return f'''from mlmom.adapters import LibraryCandidate, load_adapter
class Model(LibraryCandidate):
    def fit(self,X,y,context):
        if len(X)>{threshold}:raise ValueError("intentional real-data fit failure")
        return super().fit(X,y,context)
def create_candidate(config):return Model(config)
def load_candidate(artifact_dir):return load_adapter(artifact_dir)
'''


class RepairLLM(OfflineLLM):
    def __init__(self,*,refit_failure=False):
        super().__init__();self.code_calls=0;self.refit_failure=refit_failure
    def generate(self,role,instruction,payload,schema,tools=None):
        if role=='coding_agent':
            self.code_calls+=1
            code=broken_code(150) if self.refit_failure else broken_code(60) if self.code_calls==1 else ADAPTER_CODE
            return CodeBundle(files={'candidate.py':code})
        return super().generate(role,instruction,payload,schema,tools)


class InvariantTests(unittest.TestCase):
    def test_tool_written_candidate_survives_api_or_step_limit_then_is_verified(self):
        from mlmom.agents import LLMUnavailable
        for interruption in [ValueError('coding_agent exceeded 12 tool steps'),LLMUnavailable('API unavailable')]:
            class InterruptedCoder(OfflineLLM):
                def generate(self,role,instruction,payload,schema,tools=None):
                    if role=='coding_agent':
                        result=tools.invoke('write_file',{'path':'candidate.py','content':ADAPTER_CODE})
                        if result.status!='ok':raise AssertionError(result)
                        raise interruption
                    return super().generate(role,instruction,payload,schema,tools)
            with self.subTest(error=type(interruption).__name__),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);task=dataset(root/'input')
                config=RunConfig(total_seconds=90,final_reserve_seconds=15,max_rounds=1,max_candidates=1,
                                 smoke_seconds=10,experiment_seconds=15,prediction_seconds=10)
                with patch('mlmom.pipeline.installed_tools',only_sklearn):
                    result=Pipeline(task,config,run_dir=root/'run',client=InterruptedCoder()).run()
                self.assertEqual(result['status'],'completed',result)
                store=ExperimentStore(root/'run')
                try:
                    candidates=store.list('candidate')
                    self.assertTrue(candidates[-1]['verification']['valid'])
                    self.assertTrue(candidates[-1]['verification']['warnings'])
                finally:store.close()

    def test_explicit_probability_column_uses_the_requested_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);frame,context,plan,code=execution_case(root,'binary_classification')
            context.submission_kind='probability'
            context.task.result_format.prediction_kind='probability'
            context.task.result_format.class_order=[0]
            candidate=CandidateVersion(candidate_id='c',hypothesis_id='h',version=1,code_dir=str(code),code_hash='hash',plan_ref='',verification=ValidationReport(valid=True))
            run=ExperimentRun(experiment_id='e',round_id='r',candidate_id='c',split_id='s',status='completed')
            values=np.tile([.8,.2],(len(frame),1))
            artifact=publish_submission(context,frame,values,run,candidate)
            from mlmom.inputs import read_table
            self.assertTrue(np.allclose(read_table(artifact.path)['t0'],.8))
    def test_api_inspection_is_bounded_and_does_not_expose_os(self):
        with tempfile.TemporaryDirectory() as tmp,SandboxManager() as manager:
            root=Path(tmp)
            result=manager.run(dict(action='inspect_api',module='sklearn.ensemble',symbol='ExtraTreesRegressor.fit'),root/'good',seconds=15,memory_gb=2,cpus=1)
            self.assertEqual(result['status'],'completed',result)
            self.assertIn('sample_weight',result['signature'])
            blocked=manager.run(dict(action='inspect_api',module='os',symbol='environ'),root/'bad',seconds=15,memory_gb=2,cpus=1)
            self.assertEqual(blocked['status'],'failed')
    def test_explicit_time_axis_is_deferred_before_experiments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);task=dataset(root/'input')
            path=task.parent/'dataset_manifest.json'
            manifest=json.loads(path.read_text(encoding='utf-8'));manifest['time_column']='x'
            path.write_text(json.dumps(manifest),encoding='utf-8')
            result=Pipeline(task,RunConfig(total_seconds=60,final_reserve_seconds=10),run_dir=root/'run',client=OfflineLLM()).run()
            self.assertEqual(result['status'],'deferred',result)
            self.assertEqual(result['stop_reason'],'deferred_time_series_support')
            self.assertFalse((task.parent/'submission.csv').exists())

    def test_real_training_failure_repaired_and_versioned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);task=dataset(root/'input')
            client=RepairLLM()
            cfg=RunConfig(total_seconds=180,final_reserve_seconds=20,prediction_seconds=10,smoke_seconds=15,max_rounds=1,runtime_repairs=1)
            with patch('mlmom.pipeline.installed_tools',only_sklearn):result=Pipeline(task,cfg,run_dir=root/'run',client=client).run()
            self.assertEqual(result['status'],'completed',result)
            store=ExperimentStore(root/'run')
            self.assertEqual(len(store.list('candidate')),2)
            self.assertTrue(any(r['status']=='failed' for r in store.list('experiment')))
            self.assertTrue(any(r['status']=='completed' and r['role']=='evaluation' for r in store.list('experiment')))
            self.assertEqual(client.code_calls,2);store.close()

    def test_failed_full_refit_preserves_valid_split_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);task=dataset(root/'input')
            cfg=RunConfig(total_seconds=150,final_reserve_seconds=20,prediction_seconds=10,smoke_seconds=15,max_rounds=1)
            with patch('mlmom.pipeline.installed_tools',only_sklearn):result=Pipeline(task,cfg,run_dir=root/'run',client=RepairLLM(refit_failure=True)).run()
            self.assertEqual(result['status'],'completed',result)
            self.assertEqual(result['submission']['model_role'],'evaluation')
            self.assertEqual(file_hash(Path(result['submission']['path'])),result['submission']['sha256'])
            store=ExperimentStore(root/'run');self.assertEqual(store.get('experiment','final-refit')['status'],'failed');store.close()

    def test_cpu_affinity_is_disjoint_and_worker_keys_are_removed(self):
        with tempfile.TemporaryDirectory() as tmp,SandboxManager() as manager:
            root=Path(tmp);code=root/'code';code.mkdir()
            source='import os\nassert "OPENAI_API_KEY" not in os.environ\nassert "ROUTERAI_API_KEY" not in os.environ\n'+ADAPTER_CODE
            (code/'candidate.py').write_text(source,encoding='utf-8')
            request=dict(action='smoke',code_dir=str(code),config={'tool_id':'sklearn','model_params':{'n_estimators':5}})
            with patch.dict('os.environ',{'OPENAI_API_KEY':'unit-test-only','ROUTERAI_API_KEY':'unit-test-only'}):
                first=manager.start(request,root/'a',seconds=20,memory_gb=2,cpus=1)
                second=manager.start(request,root/'b',seconds=20,memory_gb=2,cpus=1)
            if first.cpu_ids and second.cpu_ids:self.assertFalse(set(first.cpu_ids)&set(second.cpu_ids))
            for job in [first,second]:
                while job.poll() is None:time.sleep(.05)
                result=job.finish();self.assertEqual(result['status'],'completed',result)

    def test_plan_rejects_unknown_skills_and_resource_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame,context,plan,code=execution_case(Path(tmp),'regression')
            tools=installed_tools();tools['sklearn'].available=True
            skills=SkillRegistry(ROOT/'skills')
            budget=BudgetSnapshot(remaining_seconds=200,research_seconds=100,available_cpus=4,available_memory_gb=8,spent_seconds=0)
            self.assertTrue(validate_plan(plan,context,tools,skills,budget).valid)
            plan.skill_ids=['does_not_exist'];plan.model_params={'n_jobs':999};plan.cpus=99
            report=validate_plan(plan,context,tools,skills,budget)
            self.assertFalse(report.valid)
            self.assertTrue(any('resources' in e for e in report.errors))
            self.assertTrue(any('override' in e for e in report.errors))


if __name__=='__main__':unittest.main()
