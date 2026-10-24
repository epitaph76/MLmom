"""Submission-only full-data models never enter comparison or replace their source."""
import json
import tempfile
import unittest
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mlmom.persistence import ExperimentStore, file_hash
from mlmom.pipeline import Pipeline
from mlmom.pipeline_contracts import Champion, RunConfig
from mlmom.search_groups import component_pool, internal_winner
from tests.fakes import OfflineLLM
from tests.test_pipeline import dataset, only_sklearn


class SubmissionIsolationTests(unittest.TestCase):
    def test_next_cycle_provisional_prediction_cannot_replace_full_data_answer(self):
        pipeline = object.__new__(Pipeline)
        pipeline.config = RunConfig()
        pipeline.last_submission = dict(model_role='submission_refit', experiment_id='r1-submission-refit')
        pipeline.publish_model = Mock(side_effect=AssertionError('Canonical full-data CSV must remain unchanged'))
        pipeline.publish(SimpleNamespace(run=SimpleNamespace(role='evaluation')))
        pipeline.publish_model.assert_not_called()

    def test_even_a_forged_score_does_not_make_a_submission_refit_eligible(self):
        full = dict(run=dict(status='completed', role='submission_refit', internal_score=-100))
        valid = dict(run=dict(status='completed', role='evaluation', internal_score=10))
        self.assertIs(internal_winner([full, valid], 'minimize'), valid)
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperimentStore(Path(tmp))
            try:
                store.put('cycle', 'r1', dict(round_id='r1', completed=True))
                store.put('component', 'wrong', dict(experiment_id='wrong', round_id='r1',
                    candidate=dict(code_hash='hash'), configuration_hash='hash', run=full['run']))
                self.assertEqual(component_pool(store, 2), [])
            finally:
                store.close()

    def test_external_stop_keeps_the_previous_full_data_cycle_answer(self):
        from solution import external_stop
        class StopAtSecondCycle(OfflineLLM):
            def generate(self, role, instruction, payload, schema, tools=None):
                if role == 'hypothesis_generator' and payload['round_number'] == 2:
                    signal.raise_signal(signal.SIGTERM)
                return super().generate(role, instruction, payload, schema, tools)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = dataset(root/'input')
            config = RunConfig(hpo_enabled=False, feature_search_enabled=False, diagnostic_seconds=0,
                max_workers=1, reserve_cpus=0, reserve_memory_gb=0)
            original_handler = signal.signal(signal.SIGTERM, external_stop)
            try:
                with patch('mlmom.pipeline.installed_tools', only_sklearn):
                    result = Pipeline(task, config, run_dir=root/'run', client=StopAtSecondCycle()).run()
            finally:
                signal.signal(signal.SIGTERM, original_handler)
            self.assertEqual(result['status'], 'completed', result)
            self.assertEqual(result['stop_reason'], 'external_stop')
            self.assertEqual(result['submission']['model_role'], 'submission_refit')
            self.assertEqual(result['submission']['experiment_id'], 'r1-submission-refit')
            self.assertEqual(file_hash(task.parent/'submission.csv'), file_hash(root/'run/cycles/r1/submission.csv'))
            store = ExperimentStore(root/'run')
            try:
                full = store.get('submission_refit', 'r1-submission-refit')
                self.assertEqual(full['status'], 'completed')
                self.assertEqual(len(store.list('submission_refit')), 1)
                self.assertTrue(store.get('cycle', 'r1')['completed'])
                self.assertEqual(store.get('champion', 'current')['run']['role'], 'evaluation')
                self.assertEqual({r['run']['role'] for r in component_pool(store, 2)}, {'evaluation'})
                champ = dict(result['champion'], run=full)
                with self.assertRaisesRegex(ValueError, 'submission-only'):
                    Champion.model_validate(champ)
                request = json.loads((root/'run/runs/r1-submission-refit/request.json').read_text(encoding='utf-8'))
                self.assertEqual(request['train_path'], str(task.parent/'train.csv'))
                self.assertEqual(request['phase'], 'submission')
            finally:
                store.close()

    def test_failed_later_refit_preserves_last_full_answer_and_its_metric_source(self):
        from tests.fakes import MODEL_CODE
        class StopAtThirdCycle(OfflineLLM):
            def generate(self, role, instruction, payload, schema, tools=None):
                if role == 'hypothesis_generator' and payload['round_number'] == 3:
                    raise KeyboardInterrupt
                if role == 'coding_agent':
                    source = MODEL_CODE.replace('        self.kind = context.task_type',
                        "        if context.stage == 'final_full_data_refit' and 'r2-submission-refit' in context.work_dir:\n"
                        "            raise ValueError('intentional cycle-2 full-data failure')\n"
                        '        self.kind = context.task_type')
                    return schema(files={'candidate.py': source})
                return super().generate(role, instruction, payload, schema, tools)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = dataset(root/'input')
            config = RunConfig(hpo_enabled=False, feature_search_enabled=False, diagnostic_seconds=0,
                max_workers=1, reserve_cpus=0, reserve_memory_gb=0)
            with patch('mlmom.pipeline.installed_tools', only_sklearn):
                result = Pipeline(task, config, run_dir=root/'run', client=StopAtThirdCycle()).run()
            self.assertEqual(result['stop_reason'], 'external_stop', result)
            self.assertEqual(result['submission']['experiment_id'], 'r1-submission-refit')
            self.assertEqual(file_hash(task.parent/'submission.csv'), file_hash(root/'run/cycles/r1/submission.csv'))
            store = ExperimentStore(root/'run')
            try:
                self.assertFalse(store.get('cycle', 'r2')['completed'])
                self.assertTrue(store.get('cycle', 'r2')['evaluation_completed'])
                self.assertTrue(any(row['round_id'] == 'r2' for row in component_pool(store, 3)))
                self.assertEqual(store.get('submission_refit_status', 'r2-submission-refit')['status'], 'failed')
                self.assertEqual(result['champion']['run']['experiment_id'],
                    store.get('submission_refit_source', 'r1-submission-refit')['champion_experiment_id'])
            finally:
                store.close()

    def test_old_session_preserves_its_previous_limits_and_refit_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = dataset(root/'input')
            config = RunConfig(max_rounds=100, plateau_rounds=5, cycle_full_refit=False)
            def close(pipeline):
                pipeline.sandbox.close()
                pipeline.checkpointer.close()
                pipeline.store.close()
                pipeline.trace.close()
            with patch('mlmom.pipeline.installed_tools', only_sklearn):
                initial = Pipeline(task, config, run_dir=root/'run', client=OfflineLLM())
                close(initial)
                session_path = root/'run/session.json'
                session = json.loads(session_path.read_text(encoding='utf-8'))
                session['config'].pop('cycle_full_refit')
                session_path.write_text(json.dumps(session), encoding='utf-8')
                resumed = Pipeline(task, RunConfig(), run_dir=root/'run', client=OfflineLLM(), resume=True)
                try:
                    self.assertFalse(resumed.config.cycle_full_refit)
                    self.assertEqual(resumed.config.max_rounds, 100)
                    self.assertEqual(resumed.config.plateau_rounds, 5)
                finally:
                    close(resumed)


class ExternalQueueStopTests(unittest.TestCase):
    def test_queue_requests_graceful_stop_before_forced_cleanup(self):
        from scripts.run_docker_queue import stop_pipeline
        process = Mock(pid=123)
        with patch('scripts.run_docker_queue.terminate_tree') as forced:
            stop_pipeline(process, grace_seconds=7)
            process.terminate.assert_called_once()
            process.wait.assert_called_once_with(timeout=7)
            forced.assert_not_called()

    def test_queue_forces_cleanup_only_if_the_grace_period_expires(self):
        from scripts.run_docker_queue import stop_pipeline
        process = Mock(pid=123)
        process.wait.side_effect = [subprocess.TimeoutExpired('pipeline', 7), 0]
        with patch('scripts.run_docker_queue.terminate_tree') as forced:
            stop_pipeline(process, grace_seconds=7)
            forced.assert_called_once_with(123)


if __name__ == '__main__':
    unittest.main()
