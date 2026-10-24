"""Cycle answers survive later publication and reject damaged organizer CSVs."""
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mlmom.persistence import file_hash
from mlmom.pipeline_contracts import SubmissionArtifact
from mlmom.submission import snapshot_cycle_submission, validate_submission_file
from tests.test_execution import execution_case


class CycleSubmissionTests(unittest.TestCase):
    def fixture(self, root, kind='regression'):
        frame, context, _, _ = execution_case(root, kind)
        context.submission_kind = 'label' if 'classification' in kind else 'value'
        target = context.task.target_columns[0]
        if context.submission_kind == 'label':
            context.classes = ['Normal', 'Worms', 'Analysis']
            frame[target] = [context.classes[i % 3] for i in range(len(frame))]
        source = root / 'submission.csv'
        frame[['id', target]].to_csv(source, index=False)
        artifact = SubmissionArtifact(path=str(source), experiment_id='e1', candidate_id='c1',
            model_role='evaluation', rows=len(frame), sha256=file_hash(source))
        return frame, context, artifact

    def test_unchanged_champion_is_saved_each_cycle_and_resume_is_idempotent(self):
        for kind in ['regression', 'multiclass_classification']:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                frame, context, artifact = self.fixture(root, kind)
                first = snapshot_cycle_submission(context, frame, artifact, root / 'run', 'r1')
                second = snapshot_cycle_submission(context, frame, artifact, root / 'run', 'r2')
                self.assertNotEqual(first['path'], second['path'])
                self.assertEqual(first['verification']['sha256'], second['verification']['sha256'])
                self.assertEqual(snapshot_cycle_submission(context, frame, artifact, root / 'run', 'r2'), second)
                original = Path(first['path']).read_bytes()
                Path(artifact.path).write_text('later publication', encoding='utf-8')
                self.assertEqual(Path(first['path']).read_bytes(), original)
                self.assertEqual(json.loads(Path(first['path']).with_suffix('.json').read_text())['verification']['status'], 'passed')

    def test_invalid_values_schema_ids_and_checksum_are_rejected(self):
        for problem in ['nan', 'infinity', 'order', 'duplicates', 'extra_column', 'unknown_class', 'checksum']:
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                frame, context, artifact = self.fixture(root,
                    'multiclass_classification' if problem == 'unknown_class' else 'regression')
                output = pd.read_csv(artifact.path)
                target = context.task.target_columns[0]
                if problem == 'nan': output.loc[0, target] = float('nan')
                if problem == 'infinity': output.loc[0, target] = float('inf')
                if problem == 'order': output = output.iloc[::-1]
                if problem == 'duplicates': output.loc[0, 'id'] = output.loc[1, 'id']
                if problem == 'extra_column': output['unexpected'] = 0
                if problem == 'unknown_class': output.loc[0, target] = 'not-a-class'
                if problem == 'checksum': output.loc[0, target] += 1
                output.to_csv(artifact.path, index=False)
                with self.assertRaises(ValueError):
                    validate_submission_file(context, frame, Path(artifact.path),
                        expected_hash=artifact.sha256 if problem == 'checksum' else None)
                with self.assertRaises(ValueError):
                    snapshot_cycle_submission(context, frame, artifact, root / 'run', 'r1')
                self.assertFalse((root / 'run' / 'cycles' / 'r1' / 'submission.csv').exists())

    def test_saved_cycle_cannot_be_replaced_or_silently_corrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame, context, artifact = self.fixture(root)
            report = snapshot_cycle_submission(context, frame, artifact, root / 'run', 'r1')
            with self.assertRaisesRegex(ValueError, 'replace'):
                snapshot_cycle_submission(context, frame, artifact.model_copy(update={'experiment_id': 'e2'}), root / 'run', 'r1')
            Path(report['path']).write_text('damaged', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                snapshot_cycle_submission(context, frame, artifact, root / 'run', 'r1')
