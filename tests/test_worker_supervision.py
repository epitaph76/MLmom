"""Stable parent liveness, orphan cleanup and terminal infrastructure failures."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import psutil

from mlmom.process_identity import ProcessIdentity, process_identity, process_matches, recorded_process_matches
from mlmom.resources import Job, SandboxManager, WorkerInfrastructureError, terminate_tree
from mlmom.tools import ToolSet
from mlmom.worker import watchdog


class ProcessIdentityTests(unittest.TestCase):
    def test_linux_stat_handles_parentheses_and_uses_kernel_ticks(self):
        fields = ['S'] + ['0'] * 18 + ['12345'] + ['0'] * 3
        stat = '99 (name with ) and ( spaces) ' + ' '.join(fields)
        with patch('mlmom.process_identity.sys.platform', 'linux'), \
             patch.object(Path, 'read_text', side_effect=[stat, 'boot-a\n']):
            identity = process_identity(99)
        self.assertEqual(identity, ProcessIdentity(99, 'linux_start_ticks', '12345', 'boot-a'))

    def test_reused_pid_or_different_boot_is_not_the_original_process(self):
        identity = ProcessIdentity(99, 'linux_start_ticks', '12345', 'boot-a')
        for current in [ProcessIdentity(99, 'linux_start_ticks', '12346', 'boot-a'),
                        ProcessIdentity(99, 'linux_start_ticks', '12345', 'boot-b')]:
            with self.subTest(current=current), patch('mlmom.process_identity.process_identity', return_value=current):
                self.assertFalse(process_matches(identity))

    def test_dead_or_unreadable_process_has_distinct_behavior(self):
        identity = ProcessIdentity(99, 'linux_start_ticks', '12345')
        for error, alive in [(psutil.NoSuchProcess(99), False), (psutil.AccessDenied(99), None)]:
            with patch('mlmom.process_identity.process_identity', side_effect=error):
                self.assertEqual(process_matches(identity), alive)

    def test_unverified_identity_cannot_authorize_cleanup(self):
        identity = ProcessIdentity(99, 'linux_start_ticks', '12345', 'boot-a')
        with tempfile.TemporaryDirectory() as tmp, \
             patch('mlmom.resources.process_matches', return_value=None), \
             patch('mlmom.resources.terminate_tree') as stop:
            root = Path(tmp); (root/'result.json').write_text('{"status":"completed"}')
            process = Mock(poll=Mock(return_value=0), returncode=0)
            job = Job(process, root, time.monotonic(), 1, 1, 1, Mock(), Mock(), descendants={identity})
            job.finish()
            stop.assert_not_called()
        record = dict(pid=98, identity=identity.to_dict())
        self.assertFalse(recorded_process_matches(record))

    def test_native_identity_and_legacy_checkpoint_support(self):
        process = Mock(create_time=Mock(return_value=123.5), status=Mock(return_value=psutil.STATUS_RUNNING))
        with patch('mlmom.process_identity.sys.platform', 'win32'), \
             patch('mlmom.process_identity.psutil.Process', return_value=process):
            identity = process_identity(99)
            self.assertTrue(process_matches(identity.to_dict()))
            self.assertTrue(recorded_process_matches(dict(pid=99, created=123.5)))
            self.assertFalse(recorded_process_matches(dict(pid=99, created=124.5)))
            process.status.return_value = psutil.STATUS_ZOMBIE
            self.assertFalse(process_matches(identity))

    @unittest.skipUnless(sys.platform == 'linux', 'Linux boot-time regression')
    def test_actual_worker_survives_one_second_psutil_clock_disagreement(self):
        original = psutil.Process().create_time()
        with tempfile.TemporaryDirectory() as tmp, SandboxManager(timeouts_enabled=False) as sandbox, \
             patch('psutil.Process.create_time', return_value=original + 1):
            result = sandbox.run(dict(action='runtime_check'), Path(tmp), seconds=.001, cpus=1, memory_gb=2)
            self.assertEqual(result['status'], 'completed', result)
            request = json.loads((Path(tmp)/'request.json').read_text())
            self.assertEqual(request['parent_identity']['source'], 'linux_start_ticks')
            self.assertNotIn('parent_created', request)
            self.assertFalse((Path(tmp)/'limit.json').exists())

    def test_actual_worker_accepts_controller_identity_through_python_launcher(self):
        # In a Windows Store Python venv, sys.executable can be python.exe while
        # psutil reports python3.12.exe; neither spelling defines process identity.
        with tempfile.TemporaryDirectory() as tmp, SandboxManager(timeouts_enabled=False) as sandbox:
            root = Path(tmp)
            result = sandbox.run(dict(action='runtime_check'), root, seconds=.001, cpus=1, memory_gb=2)
            self.assertEqual(result['status'], 'completed', result)
            request = json.loads((root/'request.json').read_text())
            self.assertEqual(request['parent_identity']['pid'], os.getpid())
            self.assertTrue(process_matches(request['parent_identity']))
            self.assertNotIn('parent_executable', request)
            self.assertFalse((root/'limit.json').exists())


class WatchdogTests(unittest.TestCase):
    def request(self):
        return dict(parent_pid=123, parent_identity=ProcessIdentity(123, 'linux_start_ticks', '1').to_dict(),
                    worker_seconds=.001, worker_memory_gb=1, timeouts_enabled=False)

    def test_untimed_worker_is_alive_but_memory_guard_still_stops_it(self):
        for rss, reason in [(0, None), (2 * 1024**3, 'memory_limit')]:
            with self.subTest(rss=rss), tempfile.TemporaryDirectory() as tmp:
                process = Mock(children=Mock(return_value=[]), memory_info=Mock(return_value=NS(rss=rss)))
                with patch('mlmom.process_identity.process_matches', return_value=True), \
                     patch('psutil.Process', return_value=process), \
                     patch('mlmom.worker.Path', side_effect=lambda name: Path(tmp)/name), \
                     patch('mlmom.worker.time.sleep', side_effect=InterruptedError), \
                     patch('mlmom.resources.terminate_tree') as stop:
                    if reason: watchdog(self.request())
                    else:
                        with self.assertRaises(InterruptedError): watchdog(self.request())
                limit = Path(tmp)/'limit.json'
                self.assertEqual(limit.exists(), reason is not None)
                if reason:
                    self.assertEqual(json.loads(limit.read_text())['reason'], reason)
                    stop.assert_called_once_with(os.getpid())
                else: stop.assert_not_called()

    def test_reused_parent_is_stopped_and_not_mislabelled_as_model_failure(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch('mlmom.process_identity.process_matches', return_value=False), \
             patch('mlmom.worker.Path', side_effect=lambda name: Path(tmp)/name), \
             patch('mlmom.resources.terminate_tree') as stop:
            watchdog(self.request())
            self.assertEqual(json.loads((Path(tmp)/'limit.json').read_text())['reason'], 'parent_exit')
            stop.assert_called_once_with(os.getpid())

    def test_parent_death_reaps_worker_and_its_child(self):
        # A real controller disappears while its detached worker and grandchild
        # are running. No LLM, training data or elapsed-time watchdog is involved.
        worker_code = '''import json,subprocess,sys,threading,time
from pathlib import Path
from mlmom.worker import watchdog
request=json.loads(Path(sys.argv[1]).read_text())
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
Path('ready.json').write_text(json.dumps(dict(child=child.pid)))
watchdog(request)
'''
        controller_code = '''import json,os,subprocess,sys
from pathlib import Path
from mlmom.process_identity import process_identity
request=Path('request.json')
request.write_text(json.dumps(dict(parent_pid=os.getpid(),parent_identity=process_identity().to_dict(),
    worker_seconds=.001,worker_memory_gb=1,timeouts_enabled=False)))
worker=subprocess.Popen([sys.executable,'-c',sys.argv[1],str(request)])
Path('worker.json').write_text(json.dumps(dict(pid=worker.pid)))
worker.wait()
'''
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONPATH=str(root))
        with tempfile.TemporaryDirectory() as tmp:
            options = dict(creationflags=subprocess.CREATE_NO_WINDOW) if os.name == 'nt' else {}
            controller = subprocess.Popen([sys.executable, '-c', controller_code, worker_code], cwd=tmp,
                env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **options)
            worker_pid = child_pid = None
            try:
                until = time.monotonic() + 15
                while not (Path(tmp)/'ready.json').exists() and time.monotonic() < until:
                    time.sleep(.05)
                self.assertTrue((Path(tmp)/'ready.json').exists())
                worker_pid = json.loads((Path(tmp)/'worker.json').read_text())['pid']
                child_pid = json.loads((Path(tmp)/'ready.json').read_text())['child']
                time.sleep(.3)
                self.assertTrue(psutil.pid_exists(worker_pid))
                self.assertFalse((Path(tmp)/'limit.json').exists())
                controller.terminate(); controller.wait(timeout=10)
                until = time.monotonic() + 10
                while time.monotonic() < until:
                    alive = []
                    for pid in (worker_pid, child_pid):
                        try:
                            if psutil.Process(pid).status() != psutil.STATUS_ZOMBIE: alive.append(pid)
                        except psutil.NoSuchProcess: pass
                    if not alive: break
                    time.sleep(.05)
                self.assertEqual(alive, [])
                self.assertEqual(json.loads((Path(tmp)/'limit.json').read_text())['reason'], 'parent_exit')
            finally:
                if controller.poll() is None: controller.kill()
                controller.wait(timeout=10)
                for pid in (worker_pid, child_pid):
                    if pid is not None: terminate_tree(pid)


class InfrastructureRoutingTests(unittest.TestCase):
    def test_role_tool_failure_is_not_wrapped_as_invalid_llm_output(self):
        from mlmom.agents import RoleClient
        from mlmom.coding import CodeBundle
        from mlmom.llm_client import LLMSettings
        def unavailable():
            raise WorkerInfrastructureError(dict(stop_reason='parent_exit'), '/worker/check')
        tools = ToolSet().register('read_evidence', unavailable, 'Read evidence')
        messages = [
            ('planner', dict(role='assistant', content=json.dumps(dict(context_requests=[
                dict(name='read_evidence', arguments={}, reason='Check the saved evidence')])))),
            ('coding_agent', dict(role='assistant', tool_calls=[dict(id='call-1', type='function',
                function=dict(name='read_evidence', arguments='{}'))])),
        ]
        for role, message in messages:
            with self.subTest(role=role), tempfile.TemporaryDirectory() as tmp:
                client = RoleClient(LLMSettings(api_key='offline-test'), deadline=time.time()+10)
                client.trace_root = Path(tmp)/'trace'
                try:
                    with patch.object(client, '_request', return_value=message) as request:
                        with self.assertRaises(WorkerInfrastructureError):
                            client.generate(role, 'Test infrastructure routing', {}, CodeBundle, tools)
                        self.assertEqual(request.call_count, 1)
                finally: client.client.close()

    def test_feature_failure_is_terminal_and_keeps_pending_hypothesis_unjudged(self):
        from mlmom.feature_search import run_feature_session
        from tests.test_feature_session import SessionAccountingTests
        with tempfile.TemporaryDirectory() as tmp:
            p, base, split, path = SessionAccountingTests().harness(Path(tmp))
            original = p.sandbox.run
            def fail(req, directory, **limits):
                if req['action'] == 'train' and Path(directory).name != 'reference':
                    return dict(status='failed', stop_reason='parent_exit')
                return original(req, directory, **limits)
            try:
                with patch.object(p.sandbox, 'run', side_effect=fail):
                    with self.assertRaises(WorkerInfrastructureError):
                        run_feature_session(p, base, split, path)
                session = p.store.get('feature_session', 'initial')
                self.assertEqual(session['attempts'], 1)
                self.assertIsNotNone(session['pending'])
                self.assertEqual(p.store.list('feature_evidence'), [])
                self.assertEqual(p.client.calls, 1)
            finally: p.store.close()

    def test_scheduler_and_tools_propagate_terminal_failure_without_retry(self):
        result = dict(status='failed', stop_reason='parent_exit')
        with tempfile.TemporaryDirectory() as tmp, SandboxManager() as sandbox:
            job = Mock(poll=Mock(return_value=1), finish=Mock(return_value=result))
            with patch.object(sandbox, 'start', return_value=job) as start:
                with self.assertRaises(WorkerInfrastructureError):
                    sandbox.run(dict(action='smoke'), Path(tmp), seconds=10, cpus=1, memory_gb=.25)
                self.assertEqual(start.call_count, 1)
            def broken(): raise WorkerInfrastructureError(result, tmp)
            tools = ToolSet().register('broken', broken, 'Injected supervisor failure')
            with self.assertRaises(WorkerInfrastructureError): tools.invoke('broken', {})

    def test_requested_smoke_propagates_infrastructure_failure_without_regeneration(self):
        from mlmom.coding import build_coding_graph
        from mlmom.pipeline_contracts import RunConfig
        from tests.test_planning_limits import case, Store, ClockBudget, Coder, ADAPTER_CODE
        class RequestingCoder(Coder):
            def generate(self,*args):
                if args[0]=='coding_agent':
                    self.calls+=1
                    args[4].invoke('write_file',dict(path='candidate.py',content=ADAPTER_CODE))
                    args[4].invoke('run_smoke_check',{})
                return super().generate(*args)
        context, tools, _, plan = case()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);budget=ClockBudget();client=RequestingCoder(budget,0)
            sandbox=Mock(run=Mock(side_effect=WorkerInfrastructureError(dict(stop_reason='parent_exit'),root/'check')))
            graph=build_coding_graph(client,workspace=root/'workspace',store=Store(root),plan=plan,
                plan_ref=root/'plan.json',skill_data=[],tools=tools,context=context,sandbox=sandbox,
                budget=budget,config=RunConfig(max_attempts=3,coding_direct_first=False),round_id='r1')
            with self.assertRaises(WorkerInfrastructureError):graph.invoke({'payload':{}})
            self.assertEqual(client.calls,1)
            self.assertEqual(sandbox.run.call_count,1)

    def test_pipeline_aborts_smoke_train_prediction_publication_diagnostics_and_final_refit(self):
        from mlmom.pipeline import Pipeline
        from mlmom.pipeline_contracts import RunConfig
        from tests.fakes import OfflineLLM
        from tests.test_pipeline import dataset, only_sklearn
        for stage in ('train', 'predict', 'publish', 'diagnose', 'final'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); client = OfflineLLM()
                with patch('mlmom.pipeline.installed_tools', only_sklearn):
                    pipeline = Pipeline(dataset(root/'input'), RunConfig(max_rounds=3, max_attempts=3,
                        runtime_repairs=2, hpo_enabled=False, feature_search_enabled=False,
                        diagnostic_seconds=1 if stage == 'diagnose' else 0,
                        reserve_cpus=0, reserve_memory_gb=0), run_dir=root/'run', client=client)
                start = pipeline.sandbox.start
                predictions = []
                def launch(request, directory, **limits):
                    action = request['action']
                    if action == 'predict': predictions.append(directory)
                    injected = ((stage == 'train' and action == 'train') or
                        (stage == 'final' and request.get('phase') == 'final') or
                        (stage == 'publish' and action == 'predict' and len(predictions) == 2) or
                        (stage in {'smoke', 'predict', 'diagnose'} and action == stage))
                    if injected:
                        directory.mkdir(parents=True, exist_ok=True)
                        (directory/'limit.json').write_text(json.dumps(dict(reason='parent_exit')))
                        # Real Job.finish exercises synchronous and async paths.
                        process = Mock(poll=Mock(return_value=1), returncode=1, pid=999999999)
                        return Job(process, directory, time.monotonic(), 10, 1, 1, Mock(), Mock())
                    return start(request, directory, **limits)
                if stage == 'final':
                    pipeline.config.max_rounds = 1
                    pipeline.config.cycle_full_refit = False
                with patch.object(pipeline.sandbox, 'start', side_effect=launch):
                    output = pipeline.run()
                self.assertEqual(output['status'], 'failed', output)
                self.assertEqual(output['stop_reason'], 'worker_infrastructure_error', output)
                self.assertFalse(output['process_acceptance']['passed'])
                self.assertEqual(sum(role == 'coding_agent' for role, _ in client.calls), 1)
                self.assertEqual(sum(role == 'hypothesis_generator' for role, _ in client.calls), 1)
                if stage == 'final':
                    self.assertIsNotNone(output['submission'])
                    self.assertTrue(Path(output['submission']['path']).is_file())
                failure = json.loads((root/'run'/'worker-infrastructure-failure.json').read_text())
                self.assertEqual(failure['reason'], 'parent_exit')


if __name__ == '__main__': unittest.main()
