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
from tests.fakes import OfflineLLM,MODEL_CODE as ADAPTER_CODE
from tests.test_pipeline import dataset,only_sklearn
from tests.test_execution import execution_case

import mlmom
ROOT=Path(mlmom.__file__).resolve().parent.parent


def broken_code(threshold):
    return ADAPTER_CODE.replace('        self.kind = context.task_type',
        f'        if len(X)>{threshold}:raise ValueError("intentional real-data fit failure")\n        self.kind = context.task_type')



class RepairLLM(OfflineLLM):
    def __init__(self,*,refit_failure=False):
        super().__init__();self.code_calls=0;self.refit_failure=refit_failure
    def generate(self,role,instruction,payload,schema,tools=None):
        if role=='coding_agent':
            self.code_calls+=1
            code=broken_code(150) if self.refit_failure else broken_code(60) if self.code_calls==1 else ADAPTER_CODE
            if schema.__name__=='CodePatch':
                source=payload['current_files']['candidate.py']
                defective=next(line for line in source.splitlines() if 'intentional real-data fit failure' in line)
                return schema(patches=[dict(path='candidate.py',sha256=payload['file_hashes']['candidate.py'],old_text=defective+'\n',new_text='')])
            return CodeBundle(files={'candidate.py':code})
        return super().generate(role,instruction,payload,schema,tools)


class CustomCandidatePipeline(Pipeline):
    """Exercise generated-candidate repair including the first baseline."""
    def _prepare(self,plan,round_id,**kwargs):
        plan.execution_mode='custom'
        return super()._prepare(plan,round_id,**kwargs)


class InvariantTests(unittest.TestCase):
    def test_lightautoml_task_objective_checked_before_planning_and_fit(self):
        from mlmom.adapters import LibraryCandidate
        from mlmom.sdk import FitContext
        from mlmom.evaluation import prediction_kind
        pairs=[('regression','rmse','roc_auc'),('binary_classification','roc_auc','log_loss'),
               ('multiclass_classification','log_loss','roc_auc')]
        for kind,accepted,rejected in pairs:
            with self.subTest(task=kind),tempfile.TemporaryDirectory() as temp:
                frame,context,plan,code=execution_case(Path(temp),kind)
                tools=installed_tools();tools['lightautoml'].available=True
                plan.tool_id='lightautoml';plan.skill_ids=['lightautoml'];plan.strategy='native';plan.patience_trials=None
                skills=SkillRegistry(ROOT/'skills')
                budget=BudgetSnapshot(remaining_seconds=200,research_seconds=100,available_cpus=4,available_memory_gb=8,spent_seconds=0)
                with patch('mlmom.registry.importlib.metadata.version',return_value='test'):
                    # The compatibility adapter's objective mapping is distinct
                    # from admission of unverified generated native protocols.
                    for metric,valid in [(accepted,True),(rejected,True)]:
                        context.task.primary_metric=metric;plan.prediction_kind=prediction_kind(metric,kind)
                        report=validate_plan(plan,context,tools,skills,budget)
                        self.assertEqual(report.valid,valid,report)
                ctx=FitContext(task_type=kind,metric=rejected,direction='minimize',seed=42,cpus=1,memory_gb=2,time_limit_seconds=1)
                with self.assertRaisesRegex(ValueError,'native objective'):
                    LibraryCandidate({'tool_id':'lightautoml'}).fit(frame[context.feature_columns],frame[context.task.target_columns],ctx)

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
                config=RunConfig(timeouts_enabled=True,coding_direct_first=False,hpo_enabled=False,diagnostic_seconds=0,total_seconds=180,final_reserve_seconds=30,max_rounds=1,max_candidates=1,
                                 smoke_seconds=20,experiment_seconds=30,prediction_seconds=20)
                with patch('mlmom.pipeline.installed_tools',only_sklearn):
                    result=CustomCandidatePipeline(task,config,run_dir=root/'run',client=InterruptedCoder(seconds=30)).run()
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
        with tempfile.TemporaryDirectory() as tmp,SandboxManager(timeouts_enabled=True) as manager:
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
            result=Pipeline(task,RunConfig(timeouts_enabled=True,hpo_enabled=False,diagnostic_seconds=0,total_seconds=60,final_reserve_seconds=10),run_dir=root/'run',client=OfflineLLM()).run()
            self.assertEqual(result['status'],'deferred',result)
            self.assertEqual(result['stop_reason'],'deferred_time_series_support')
            self.assertFalse((task.parent/'submission.csv').exists())

    def test_real_training_failure_repaired_and_versioned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);task=dataset(root/'input')
            client=RepairLLM()
            cfg=RunConfig(timeouts_enabled=True,hpo_enabled=False,diagnostic_seconds=0,total_seconds=180,final_reserve_seconds=20,prediction_seconds=10,smoke_seconds=15,max_rounds=1,runtime_repairs=1)
            with patch('mlmom.pipeline.installed_tools',only_sklearn):result=CustomCandidatePipeline(task,cfg,run_dir=root/'run',client=client).run()
            self.assertEqual(result['status'],'completed',result)
            store=ExperimentStore(root/'run')
            self.assertEqual(len(store.list('candidate')),2)
            self.assertTrue(any(r['status']=='failed' for r in store.list('experiment')))
            self.assertTrue(any(r['status']=='completed' and r['role']=='evaluation' for r in store.list('experiment')))
            self.assertEqual(client.code_calls,2);store.close()

    def test_failed_full_refit_retains_evaluation_evidence_without_publishing_it_as_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);task=dataset(root/'input')
            cfg=RunConfig(timeouts_enabled=True,hpo_enabled=False,diagnostic_seconds=0,total_seconds=150,final_reserve_seconds=20,prediction_seconds=10,smoke_seconds=15,max_rounds=1)
            with patch('mlmom.pipeline.installed_tools',only_sklearn):result=CustomCandidatePipeline(task,cfg,run_dir=root/'run',client=RepairLLM(refit_failure=True)).run()
            self.assertEqual(result['status'],'failed',result)
            self.assertIsNone(result['submission'])
            self.assertFalse(result['process_acceptance']['passed'])
            self.assertFalse(result['process_acceptance']['full_data_submission'])
            self.assertFalse((task.parent/'submission.csv').exists())
            store=ExperimentStore(root/'run')
            self.assertTrue(store.list('evaluation'))
            self.assertEqual(store.get('experiment','final-refit')['status'],'failed');store.close()

    def test_cpu_affinity_is_disjoint_and_worker_keys_are_removed(self):
        from mlmom.resources import Budget
        config=RunConfig(timeouts_enabled=True,total_seconds=60,final_reserve_seconds=0,reserve_cpus=0,reserve_memory_gb=0,max_workers=2)
        with tempfile.TemporaryDirectory() as tmp,SandboxManager(timeouts_enabled=True,budget=Budget(config,time.time()),max_workers=2) as manager:
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


if __name__=='__main__':unittest.main()
