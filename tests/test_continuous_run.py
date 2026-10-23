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
                        self.assertEqual(graph.get_state(config).next, ('plan',))
                    result = pipeline.run(resume=resume)
                self.assertEqual(result['status'], 'completed', result)
                self.assertEqual(rounds, list(range(1, 121)))

    def test_unbounded_rounds_do_not_stop_on_plateau(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from mlmom.persistence import ExperimentStore
        with tempfile.TemporaryDirectory() as temp:
            pipeline = object.__new__(Pipeline)
            pipeline.config = RunConfig(max_rounds=None, plateau_rounds=None)
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
                pipeline.store.put('reconnaissance', 'r1000', {'status':'completed', 'observations':[]})
                pipeline.context = None
                evidence = pipeline.read_planning_evidence('reconnaissance:r1000')
                self.assertIn('completed', json.dumps(evidence))
            finally:
                pipeline.store.close()
