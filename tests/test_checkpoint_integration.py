"""Actual worker/control exchange, bounded Optuna, and advisory mismatch review."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlmom.pipeline import Pipeline
from mlmom.pipeline_contracts import RunConfig
from mlmom.persistence import ExperimentStore
from tests.fakes import OfflineLLM
from tests.test_pipeline import dataset,only_sklearn


class ControlledLLM(OfflineLLM):
    def generate(self,role,instruction,payload,schema,tools=None):
        if role=='training_agent':
            self.calls.append((role,payload))
            cp=payload['checkpoint']
            if cp['stage']=='batch':return schema(action='stop',reason='Integration: keep best trial')
            if cp['remaining_trials']:
                return schema(action='continue',reason='Integration: queued trial',next_params=[{'n_estimators':11}])
            return schema(action='refit',reason='Initial single fit')
        if role=='result_validator':
            self.calls.append((role,payload))
            return schema(implementation='mismatch',effect='not_supported',explanation='Advisory mismatch for testing',
                          evidence_refs=payload['evidence_refs'])
        return super().generate(role,instruction,payload,schema,tools)


class CheckpointIntegrationTests(unittest.TestCase):
    def test_worker_consumes_decisions_before_next_batch_and_review_cannot_block_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);task=dataset(root/'input')
            config=RunConfig(timeouts_enabled=True,total_seconds=240,final_reserve_seconds=30,max_rounds=1,max_candidates=1,
                hpo_batch_size=2,hpo_max_trials=4,hpo_seconds=20,experiment_seconds=30,
                initial_experiment_seconds=15,diagnostic_seconds=0,smoke_seconds=20,prediction_seconds=15,
                reserve_cpus=0,reserve_memory_gb=0,max_workers=1)
            client=ControlledLLM(seconds=15)
            with patch('mlmom.pipeline.installed_tools',only_sklearn):
                result=Pipeline(task,config,run_dir=root/'run',client=client).run()
            self.assertEqual(result['status'],'completed',result)
            self.assertIsNotNone(result['submission'])
            store=ExperimentStore(root/'run')
            try:
                search=next(r for r in store.list('experiment') if r['experiment_id'].endswith('-hpo'))
                self.assertEqual(len(search['trials']),2)
                self.assertEqual(search['trials'][1]['params']['n_estimators'],11)
                self.assertEqual(search['stop_reason'],'training_agent_stop')
                checkpoints=store.list('training_checkpoint')
                self.assertEqual([c['stage'] for c in checkpoints],['initial_fit','initial_fit','batch'])
                self.assertTrue(all('validation_y' not in c and 'holdout' not in c for c in checkpoints))
                self.assertTrue(all(r['implementation']=='mismatch' for r in store.list('result_review')))
                self.assertEqual(len(store.list('reconnaissance')),2)
                self.assertTrue(store.get('milestone','first_submission'))
                candidate=store.list('candidate')[-1]
                self.assertTrue(candidate['preprocessing_hash'])
                self.assertEqual(sum(role=='coding_agent' for role,_ in client.calls),1)
                self.assertEqual(sum(role=='preprocessing_agent' for role,_ in client.calls),1)
            finally:store.close()


if __name__=='__main__':unittest.main()
