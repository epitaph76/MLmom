"""Prediction transport cleanup must not discard completed predictions."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from mlmom.pipeline import Pipeline


class CleanupTests(unittest.TestCase):
    def pipeline(self, root, status='completed'):
        pipeline = object.__new__(Pipeline)
        pipeline.run_dir = root
        pipeline.config = SimpleNamespace(prediction_seconds=10)
        pipeline.budget = SimpleNamespace(memory_gb=2, cpus=1,
            snapshot=lambda: SimpleNamespace(remaining_seconds=30))

        def run(request, directory, **kwargs):
            if status != 'completed':
                return {'status': status, 'error': 'synthetic failure'}
            output = directory / 'value.parquet'
            pd.DataFrame({'prediction': [1., 2.]}).to_parquet(output, index=False)
            return {'status': 'completed', 'predictions': {'value': str(output)}}

        pipeline.sandbox = SimpleNamespace(run=run)
        return pipeline

    def predict(self, pipeline):
        candidate = SimpleNamespace(candidate_id='c1', code_dir='code', code_hash='hash')
        run = SimpleNamespace(experiment_id='e1', model_artifact='model', resource_memory_gb=2, resource_cpus=1)
        return pipeline._prediction(candidate, run, pd.DataFrame({'x': [1, 2]}), ['value'])

    def test_cleanup_error_keeps_valid_predictions(self):
        original = Path.unlink
        def locked(path, *args, **kwargs):
            if path.name == 'features.csv':
                raise PermissionError('synthetic Windows file lock')
            return original(path, *args, **kwargs)
        with tempfile.TemporaryDirectory() as temp:
            pipeline = self.pipeline(Path(temp))
            with patch.object(Path, 'unlink', locked):
                result = self.predict(pipeline)
            self.assertEqual(result['value'].ravel().tolist(), [1., 2.])
            self.assertEqual(len(list(Path(temp).glob('predictions/*/features.csv'))), 1)

    def test_failed_prediction_keeps_transport_for_diagnosis(self):
        with tempfile.TemporaryDirectory() as temp:
            pipeline = self.pipeline(Path(temp), 'failed')
            with self.assertRaisesRegex(ValueError, 'synthetic failure'):
                self.predict(pipeline)
            self.assertEqual(len(list(Path(temp).glob('predictions/*/features.csv'))), 1)

    def test_unreadable_predictions_keep_transport_for_diagnosis(self):
        with tempfile.TemporaryDirectory() as temp:
            pipeline = self.pipeline(Path(temp))
            with patch('mlmom.pipeline.read_table', side_effect=OSError('synthetic corrupt output')):
                with self.assertRaisesRegex(OSError, 'synthetic corrupt output'):
                    self.predict(pipeline)
            self.assertEqual(len(list(Path(temp).glob('predictions/*/features.csv'))), 1)
