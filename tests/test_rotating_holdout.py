"""A new candidate cannot complete a cycle without a current-split reference."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from mlmom.pipeline import Pipeline
from mlmom.pipeline_contracts import RunConfig
from mlmom.persistence import ExperimentStore


class RotatingHoldoutTests(unittest.TestCase):
    def test_failed_incumbent_recheck_does_not_complete_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline=object.__new__(Pipeline)
            pipeline.config=RunConfig(max_rounds=6,target_completed_rounds=3)
            pipeline.store=ExperimentStore(Path(tmp))
            pipeline.review_results=lambda *args:None
            pipeline.scout=lambda *args:None
            pipeline.restore=lambda state:SimpleNamespace(task=SimpleNamespace(metric_direction='minimize'))
            pipeline.budget=SimpleNamespace(snapshot=lambda:SimpleNamespace(research_seconds=100))
            state=dict(round_number=2,incumbent_succeeded=False,stale=0,
                champion=dict(candidate=dict(candidate_id='old'),run=dict(experiment_id='r1-old')),
                round_results=[dict(candidate=dict(candidate_id='new'),plan={},
                    run=dict(experiment_id='r2-new',incumbent=False,trials=[],duration_seconds=1,
                        stop_reason='completed',best_params={},internal_score=1),
                    evaluation=dict(score=1,diagnostics=dict(mean_residual=0)))])
            try:
                pipeline.analyze(state)
                self.assertFalse(pipeline.store.get('cycle','r2')['completed'])
                decision=pipeline.store.get('analysis','r2')['decisions'][0]
                self.assertFalse(decision['accepted'])
                self.assertEqual(decision['reason'],'incumbent_recheck_failed')
                self.assertIsNone(decision['reference_score'])
            finally:
                pipeline.store.close()
