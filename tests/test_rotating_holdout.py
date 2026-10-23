"""A new candidate cannot complete a cycle without a current-split reference."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from mlmom.pipeline import Pipeline
from mlmom.pipeline_contracts import RunConfig
from mlmom.persistence import ExperimentStore


class RotatingHoldoutTests(unittest.TestCase):
    def test_failed_incumbent_recheck_does_not_complete_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline=object.__new__(Pipeline)
            pipeline.config=RunConfig(timeouts_enabled=True,max_rounds=6,target_completed_rounds=3)
            pipeline.store=ExperimentStore(Path(tmp))
            pipeline.review_results=Mock()
            pipeline.scout=Mock()
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
                pipeline.review_results.assert_called_once()
                pipeline.scout.assert_not_called()
            finally:
                pipeline.store.close()

    def test_terminal_cycle_skips_future_round_advisors(self):
        cases=[('max_rounds',dict(max_rounds=1),{},100),
               ('target_completed_rounds',dict(target_completed_rounds=1),{},100),
               ('plateau',dict(plateau_rounds=1),dict(round_number=2,promoted=False),100),
               ('research_deadline',{}, {}, .5),
               ('existing_stop',{},dict(stop_reason='existing_stop'),100)]
        for reason,config,state_changes,seconds in cases:
            with self.subTest(reason=reason),tempfile.TemporaryDirectory() as tmp:
                pipeline=object.__new__(Pipeline)
                pipeline.config=RunConfig(timeouts_enabled=True,**dict({'max_rounds':6},**config))
                pipeline.store=ExperimentStore(Path(tmp))
                pipeline.review_results=Mock()
                pipeline.scout=Mock()
                pipeline.restore=lambda state:SimpleNamespace(task=SimpleNamespace(metric_direction='minimize'))
                pipeline.budget=SimpleNamespace(snapshot=lambda:SimpleNamespace(research_seconds=seconds))
                state=dict(round_number=1,incumbent_succeeded=True,promoted=True,stale=0,
                    champion=dict(candidate=dict(candidate_id='new'),run=dict(experiment_id='r1-new')),
                    round_results=[dict(candidate=dict(candidate_id='new'),plan={},
                        run=dict(experiment_id='r1-new',incumbent=False,trials=[],duration_seconds=1,
                            stop_reason='completed',best_params={},internal_score=1),
                        evaluation=dict(score=1,diagnostics=dict(mean_residual=0)))])
                state.update(state_changes)
                try:
                    result=pipeline.analyze(state)
                    self.assertEqual(result['stop_reason'],reason)
                    self.assertTrue(pipeline.store.get('cycle',f"r{state['round_number']}")['completed'])
                    pipeline.review_results.assert_not_called()
                    pipeline.scout.assert_not_called()
                finally:
                    pipeline.store.close()
