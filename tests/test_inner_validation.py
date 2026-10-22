"""The confirmed validation strategy is executed strictly inside outer train."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from mlmom.splitting import inner_protocol
from tests.test_execution import execution_case


class InnerProtocolTests(unittest.TestCase):
    def test_all_strategies_are_reproducible_and_keep_outer_indices(self):
        for strategy, kind in [('holdout', 'regression'), ('kfold', 'regression'),
                               ('stratified_kfold', 'binary_classification'), ('group_kfold', 'regression')]:
            with self.subTest(strategy=strategy), tempfile.TemporaryDirectory() as tmp:
                frame, context, _, _ = execution_case(Path(tmp), kind)
                context.task.validation_strategy = strategy
                # Assignment is not validated by pydantic; normalize as real request parsing does.
                context = type(context).model_validate(context.model_dump())
                if strategy == 'group_kfold':
                    frame['group'] = np.repeat(np.arange(12), 5)
                    context.task.group_columns = ['group']
                source = np.arange(len(frame)) * 3 + 100
                protocol = inner_protocol(frame, context, source_indices=source, outer_split_id='r2-current')
                self.assertEqual(protocol, inner_protocol(frame, context, source_indices=source, outer_split_id='r2-current'))
                self.assertEqual(protocol['n_splits'], 1 if strategy == 'holdout' else 5)
                seen = []
                for fold in protocol['folds']:
                    a, b = fold['train_indices'], fold['validation_indices']
                    self.assertFalse(set(a) & set(b))
                    self.assertEqual(fold['source_validation_indices'], source[b].tolist())
                    if strategy == 'group_kfold':
                        self.assertFalse(set(frame.iloc[a]['group']) & set(frame.iloc[b]['group']))
                    seen.extend(b)
                if strategy != 'holdout':
                    self.assertEqual(sorted(seen), list(range(len(frame))))

    def test_group_holdout_and_strict_infeasibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame, context, _, _ = execution_case(Path(tmp), 'regression')
            frame['group'] = np.repeat([0, 1], 30)
            context.task.group_columns = ['group']
            task = context.task.model_dump()
            for strategy in ['holdout', 'group_kfold', 'kfold']:
                context.task = type(context.task).model_validate(dict(task, validation_strategy=strategy))
                if strategy == 'holdout':
                    self.assertEqual(inner_protocol(frame, context)['implementation'], 'group_shuffle')
                else:
                    with self.assertRaisesRegex(ValueError, 'groups|Grouped'):
                        inner_protocol(frame, context)
            with self.assertRaises(ValidationError):
                type(context.task).model_validate(dict(task, validation_folds=1))

    def test_rare_class_and_missing_class_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame, context, _, _ = execution_case(Path(tmp), 'binary_classification')
            frame['t0'] = [1] * 3 + [0] * (len(frame) - 3)
            with self.assertRaisesRegex(ValueError, 'rare class'):
                inner_protocol(frame, context)
            frame['group'] = frame['t0']
            context.task = type(context.task).model_validate(dict(context.task.model_dump(),
                validation_strategy='group_kfold', validation_folds=2, group_columns=['group']))
            with self.assertRaisesRegex(ValueError, 'every class'):
                inner_protocol(frame, context)


if __name__ == '__main__':
    unittest.main()
