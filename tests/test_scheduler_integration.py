"""Admission and startup behavior across the actual controller boundaries."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mlmom.pipeline import Pipeline
from mlmom.pipeline_contracts import RunConfig, HypothesisBatch
from mlmom.resources import SandboxManager, ResourceUnavailable, Job
from tests.fakes import OfflineLLM
from tests.test_pipeline import dataset, only_sklearn


def budget(now=0):
    return SimpleNamespace(cpus=2,memory_gb=4,deadline=100,reserve=20,now=lambda:now)


class AlternativeFailure(OfflineLLM):
    def __init__(self, output):
        super().__init__()
        self.output=output
        self.alternative_checked=False

    def generate(self,role,instruction,payload,schema,tools=None):
        if role=='hypothesis_generator':
            self.calls.append((role,payload))
            return HypothesisBatch(hypotheses=[
                dict(hypothesis_id='baseline',description='Small tree',rationale='Fast',baseline=True),
                dict(hypothesis_id='alternative',description='Another tree',rationale='Compare')])
        if role=='planner' and payload['hypotheses']['hypotheses'][0]['hypothesis_id']=='alternative':
            if not self.output.is_file():
                raise AssertionError('Alternative planned before baseline publication')
            self.alternative_checked=True
            raise ValueError('Intentionally invalid alternative')
        return super().generate(role,instruction,payload,schema,tools)


class SchedulerIntegrationTests(unittest.TestCase):
    def test_all_worker_actions_share_cpu_memory_and_slot_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager=SandboxManager(budget=budget(),max_workers=3)
            manager.jobs=[SimpleNamespace(released=False,cpus=1,memory_gb=3,poll=lambda:None)]
            for action in ['inspect','probe','smoke','train','predict','inspect_api']:
                with self.subTest(action=action),patch('mlmom.resources.subprocess.Popen') as popen:
                    with self.assertRaises(ResourceUnavailable):
                        manager.start({'action':action},Path(tmp)/action,seconds=10,memory_gb=2,cpus=1)
                    popen.assert_not_called()
                    self.assertFalse((Path(tmp)/action).exists())
            manager.jobs=[]
            manager.max_workers=1
            manager.jobs=[SimpleNamespace(released=False,cpus=1,memory_gb=1,poll=lambda:None)]
            self.assertFalse(manager.can_start(cpus=1,memory_gb=1))

    def test_expired_research_still_protects_final_reserve(self):
        manager=SandboxManager(budget=budget(now=85))
        self.assertEqual(manager.time_left({'action':'train','fixed':True}),0)
        self.assertEqual(manager.time_left({'action':'train','phase':'final'}),15)
        self.assertEqual(manager.time_left({'action':'predict'}),15)

    def test_late_poll_does_not_reclassify_success_as_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            (directory/'result.json').write_text(json.dumps({'status':'completed'}))
            job=Job(Mock(poll=Mock(return_value=0),returncode=0),directory,time.monotonic()-100,
                    1,1,1,Mock(),Mock(),descendants=set())
            self.assertEqual(job.poll(),0)
            result=job.finish()
            self.assertEqual(result['status'],'completed')
            self.assertIs(job.finish(),result)
            self.assertTrue(job.released)

    def test_corrupt_result_cannot_prevent_shutdown_of_other_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);first=root/'first';second=root/'second'
            first.mkdir();second.mkdir()
            (first/'result.json').write_text('{broken')
            exited=Mock(poll=Mock(return_value=0),returncode=0)
            active=Mock(poll=Mock(return_value=None),returncode=-1,pid=123456789)
            manager=SandboxManager(budget=budget())
            manager.jobs=[Job(exited,first,time.monotonic(),10,1,1,Mock(),Mock()),
                          Job(active,second,time.monotonic(),10,1,1,Mock(),Mock())]
            with patch('mlmom.resources.terminate_tree') as stop:
                manager.close()
                stop.assert_called_once_with(active.pid)
            self.assertTrue(all(job.released for job in manager.jobs))
            self.assertEqual(manager.jobs[0].result['stop_reason'],'invalid_worker_artifact')

    def test_advanced_gate_requires_second_round_submission_and_budget(self):
        pipeline=object.__new__(Pipeline)
        pipeline.config=RunConfig(hpo_enabled=False,diagnostic_seconds=0,coding_seconds=100,prediction_seconds=10)
        pipeline.restore=lambda state:Mock()
        pipeline.preflight_tools=lambda **kwargs:None
        pipeline.client=OfflineLLM()
        from mlmom.tools import ToolSet
        pipeline.tools={};pipeline.skills=Mock();pipeline.role_tools=ToolSet();pipeline.store=Mock()
        pipeline.store.list.return_value=[]
        pipeline.store.get.return_value=None
        pipeline.planning_payload=lambda state:{'history':{'evaluation':[]}}
        from mlmom.pipeline_contracts import BudgetSnapshot
        for round_number,submission,seconds,light in [(1,True,500,True),(2,False,500,True),
                                                       (2,True,50,True),(2,True,500,False)]:
            pipeline.last_submission={'path':'submission.csv'} if submission else None
            snapshot=BudgetSnapshot(remaining_seconds=seconds+100,research_seconds=seconds,
                                    available_cpus=2,available_memory_gb=4,spent_seconds=0)
            pipeline.budget=SimpleNamespace(snapshot=lambda:snapshot)
            with patch('mlmom.pipeline.build_planning_graph') as build:
                build.return_value.invoke.return_value={'valid':True,'plans':[]}
                pipeline.planning_call({'round_number':round_number})
                self.assertEqual(build.call_args.kwargs['lightweight'],light)
                self.assertEqual(build.call_args.args[4].research_seconds,seconds if light else seconds-110)

    def test_resume_reuses_valid_alternative_plan_and_preserves_dynamic_reserve(self):
        from mlmom.persistence import ExperimentStore
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);task=dataset(root/'input')
            config=RunConfig(hpo_enabled=False,diagnostic_seconds=0,total_seconds=150,final_reserve_seconds=20)
            original=Pipeline(task,config,run_dir=root/'run',client=OfflineLLM())
            original.store.put('budget','current',{'final_reserve_seconds':55})
            cached={'valid':True,'plans':[{'marker':'exact saved parameters'}]}
            original.store.put('planning','r1-alternatives',cached)
            original.sandbox.close();original.checkpointer.close();original.store.close();original.trace.close()
            restored=Pipeline(task,config,run_dir=root/'run',client=OfflineLLM(fail=True),resume=True)
            try:
                self.assertEqual(restored.budget.reserve,55)
                restored.restore=lambda state:Mock()
                with patch('mlmom.pipeline.build_planning_graph') as build:
                    self.assertEqual(restored.planning_call({'round_number':1},mode='alternatives'),cached)
                    build.assert_not_called()
            finally:
                restored.sandbox.close();restored.checkpointer.close();restored.store.close();restored.trace.close()

    def test_invalid_alternative_cannot_delay_or_destroy_first_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);task=dataset(root/'input')
            client=AlternativeFailure(task.parent/'submission.csv')
            config=RunConfig(hpo_enabled=False,diagnostic_seconds=0,total_seconds=150,final_reserve_seconds=20,max_rounds=1,max_attempts=1,
                             prediction_seconds=10,smoke_seconds=15,first_submission_seconds=.001)
            with patch('mlmom.pipeline.installed_tools',only_sklearn):
                result=Pipeline(task,config,run_dir=root/'run',client=client).run()
            self.assertEqual(result['status'],'completed',result)
            self.assertTrue(client.alternative_checked)
            self.assertFalse(result['first_submission']['target_met'])
            self.assertEqual(sum(role=='coding_agent' for role,_ in client.calls),1)
            events=[json.loads(line) for line in (root/'run'/'trace.jsonl').read_text().splitlines()]
            published=[e for e in events if e['kind']=='submission.published']
            self.assertTrue(published)
            self.assertTrue((root/'run'/'trace.md').exists())
            saved=list((root/'run'/'submissions').glob('*.csv'))
            self.assertGreaterEqual(len(saved),2)
            for path in saved:
                metadata=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
                from mlmom.persistence import file_hash
                self.assertEqual(file_hash(path),metadata['artifact']['sha256'])
                self.assertTrue(Path(metadata['model_artifact']).is_dir())
            index=json.loads((root/'run'/'artifact_index.json').read_text(encoding='utf-8'))
            self.assertTrue(any(row['path'].startswith('submissions/') for row in index['files']))


if __name__=='__main__':unittest.main()
