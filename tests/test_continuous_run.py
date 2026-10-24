"""Continuous outer cycles resume checkpoints without a cumulative step cap."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mlmom.pipeline import Pipeline
from mlmom.pipeline_contracts import RunConfig
from tests.fakes import OfflineLLM
from tests.test_pipeline import dataset, only_sklearn


class ContinuousRunTests(unittest.TestCase):
    def test_continuous_cycles_cross_langgraph_default_limit_and_keep_resume(self):
        for resume in (False, True):
            with self.subTest(resume=resume), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                with patch('mlmom.pipeline.installed_tools', only_sklearn):
                    pipeline = Pipeline(dataset(root/'input'), RunConfig(max_rounds=None, plateau_rounds=None),
                        run_dir=root/'run', client=OfflineLLM())
                rounds = []
                pipeline.preflight_runtime = lambda: None
                pipeline.inspect = lambda state: dict(round_number=1)
                pipeline.plan = lambda state: dict(plans=[{}])
                pipeline.execute = lambda state: {}
                def analyze(state):
                    rounds.append(state['round_number'])
                    return dict(round_number=state['round_number']+1,
                        stop_reason='test_manual_stop' if state['round_number'] == 120 else None)
                pipeline.analyze = analyze
                def finalize(state):
                    (root/'run'/'summary.json').write_text(json.dumps(dict(status='completed',
                        submission=None, rounds=len(rounds))))
                    return {}
                pipeline.finalize = finalize
                with contextlib.redirect_stdout(io.StringIO()):
                    if resume:
                        graph = pipeline.graph()
                        config = {'configurable': {'thread_id': 'run'}}
                        graph.invoke({}, config)
                        self.assertEqual(graph.get_state(config).next, ('execute',))
                    result = pipeline.run(resume=resume)
                self.assertEqual(result['status'], 'completed', result)
                self.assertEqual(rounds, list(range(1, 121)))

    def test_unbounded_rounds_do_not_stop_on_plateau(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from mlmom.persistence import ExperimentStore
        with tempfile.TemporaryDirectory() as temp:
            pipeline = object.__new__(Pipeline)
            pipeline.last_submission = None
            pipeline.config = RunConfig(max_rounds=None, plateau_rounds=None, cycle_full_refit=False)
            pipeline.store = ExperimentStore(Path(temp))
            pipeline.review_results = Mock()
            pipeline.restore = lambda state: SimpleNamespace(task=SimpleNamespace(metric_direction='minimize'))
            pipeline.budget = SimpleNamespace(snapshot=lambda: SimpleNamespace(research_seconds=1e12))
            state = dict(round_number=1001, incumbent_succeeded=True, stale=1000, promoted=False,
                champion=dict(candidate=dict(candidate_id='old'), run=dict(experiment_id='old')),
                round_results=[])
            try:
                self.assertIsNone(pipeline.analyze(state)['stop_reason'])
                self.assertTrue(pipeline.can_retry_planning(state))
                self.assertFalse(pipeline.can_retry_planning(dict(state, planning_attempt=1000)))
                self.assertTrue(pipeline.can_retry_round(state))
                pipeline.store.put('reconnaissance', 'r1000', {'status':'completed', 'observations':[]})
                pipeline.context = None
                evidence = pipeline.read_planning_evidence('reconnaissance:r1000')
                self.assertIn('completed', json.dumps(evidence))
            finally:
                pipeline.store.close()

    def test_default_profile_has_no_outer_time_round_or_plateau_cutoff(self):
        config = RunConfig()
        self.assertTrue(config.continuous)
        self.assertTrue(config.cycle_full_refit)
        self.assertFalse(config.timeouts_enabled)
        self.assertIsNone(config.max_rounds)
        self.assertIsNone(config.plateau_rounds)

    def test_many_planning_retries_do_not_hit_the_graph_step_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch('mlmom.pipeline.installed_tools', only_sklearn):
                pipeline = Pipeline(dataset(root/'input'), RunConfig(), run_dir=root/'run', client=OfflineLLM())
            calls = []
            pipeline.preflight_runtime = lambda: None
            pipeline.inspect = lambda state: dict(round_number=1)
            def plan(state):
                calls.append(1)
                return dict(plans=[{}] if len(calls) == 150 else [], stop_reason='planning_retry')
            pipeline.plan = plan
            pipeline.execute = lambda state: {}
            pipeline.analyze = lambda state: dict(stop_reason='test_manual_stop')
            def finalize(state):
                (root/'run/summary.json').write_text(json.dumps(dict(status='completed', submission=None)))
                return {}
            pipeline.finalize = finalize
            with contextlib.redirect_stdout(io.StringIO()):
                result = pipeline.run()
            self.assertEqual(result['status'], 'completed', result)
            self.assertEqual(len(calls), 150)
