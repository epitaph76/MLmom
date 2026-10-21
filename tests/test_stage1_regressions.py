"""Regressions for task evidence and consistent original-table ingestion."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mlmom.contracts import DatasetProfile, ResultFormat, TaskDefinition
from mlmom.data_inspector import DataInspector
from mlmom.inputs import read_table
from mlmom.task_validator import TaskValidator, _normalise_metric


class StageOneRegressions(unittest.TestCase):
    def test_target_wording_does_not_collect_nearby_features_or_substrings(self):
        inspector = DataInspector()
        frame = pd.DataFrame({'id': [1, 2], 'age': [20, 30], 'price': [100., 200.], 'rice': [1, 2]})
        self.assertEqual(inspector._infer_targets(frame, {}, [],
            'Predict price from age; id is the row identifier. Metric RMSE.'), ['price'])
        self.assertEqual(inspector._infer_targets(frame, {}, [],
            'Target columns: `price`, `age`. ID is id.'), ['price', 'age'])
        self.assertEqual(inspector._infer_targets(frame, {}, [],
            'Target columns: `age`, `price`. ID is id.'), ['age', 'price'])
        self.assertEqual(inspector._infer_targets(frame, {}, [],
            'The price and age columns are features near the word target.'), [])

    def test_specific_metric_aliases_win_and_match_validator_names(self):
        cases = {
            'balanced_accuracy': 'balanced_accuracy',
            'mean absolute percentage error': 'mape',
            'symmetric mean absolute percentage error': 'smape',
            'root mean squared logarithmic error': 'rmsle',
            'rmsle': 'rmsle',
            'average precision': 'average_precision',
            'mean average precision': 'map',
            'f1_macro': 'macro_f1',
            'mean_squared_error': 'mse',
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(DataInspector._metric_from_text('Metric: ' + text), expected)
                self.assertEqual(_normalise_metric(text), expected)
        self.assertIsNone(DataInspector._metric_from_text('The mapping feature is a showcase.'))

    def test_output_filename_ignores_input_mentions(self):
        fmt, _ = DataInspector._infer_result_format(
            'Use train file train.csv and test file test.csv. Save submission.csv with columns id,price.',
            ['id'], ['price'])
        self.assertEqual(fmt.file_name, 'submission.csv')
        fmt, _ = DataInspector._infer_result_format(
            'Input file train.csv. Write to `answer.csv`.', ['id'], ['price'])
        self.assertEqual(fmt.file_name, 'answer.csv')
        fmt, _ = DataInspector._infer_result_format(
            'Input file train.csv and sample_submission.csv.', ['id'], ['price'])
        self.assertIsNone(fmt.file_name)

    def test_organizer_primary_macro_f1_wins_over_auxiliary_weighted_f1(self):
        wording = ('Классы несбалансированы. Поэтому основной\n'
                   'метрикой выбран Macro F1, а не Accuracy.\n'
                   'Основная метрика — **Macro F1**:\n'
                   'Дополнительно рассчитываются **Accuracy** и **Weighted F1**.\n')
        self.assertEqual(DataInspector._metric_from_text(wording), 'macro_f1')
        self.assertEqual(DataInspector._metric_from_text(
            'Primary metric: RMSE. Also report mean absolute percentage error.'), 'rmse')
        self.assertEqual(DataInspector._metric_from_text(
            'Primary metric: mean absolute percentage error. Report MAE as well.'), 'mape')

    def test_datetime_feature_is_not_a_time_axis_but_declared_time_is(self):
        frame = pd.DataFrame({'event_date': ['2024-01-01', '2024-01-02'], 'target': [0, 1]})
        inspector = DataInspector()
        self.assertIsNone(inspector._infer_time_column(frame, {}, None, ['target'], [], 'Binary classification.'))
        self.assertEqual(inspector._infer_time_column(frame, {}, None, ['target'], [],
            'Time series forecasting.'), 'event_date')
        self.assertEqual(inspector._infer_time_column(frame, {}, 'event_date', ['target'], [],
            'Binary classification.'), 'event_date')

    def test_heuristic_type_and_default_metric_can_be_corrected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pd.DataFrame({'x': range(120), 'target': [0, 1, 2] * 40}).to_csv(root / 'train.csv', index=False)
            profile = DataInspector().inspect_directory(root)
            self.assertEqual(profile.task.field_sources['task_type'], 'heuristic')
            self.assertEqual(profile.task.field_sources['primary_metric'], 'default')
            task = TaskDefinition(objective='Predict a rounded continuous target.', task_type='regression',
                target_columns=['target'], excluded_feature_columns=['target'], primary_metric='rmse',
                metric_direction='minimize', validation_strategy='kfold', result_format=ResultFormat(
                    file_name='submission.csv', columns=['target'], prediction_columns=['target']))
            report = TaskValidator().validate(profile, task)
            self.assertTrue(report.valid, report.model_dump())
            self.assertTrue(all(finding.severity == 'warning' for finding in report.findings))
            # Legacy profiles remain readable and preserve their former strict validation.
            raw = profile.model_dump(mode='json')
            raw['task'].pop('field_sources')
            raw.pop('table_formats')
            legacy = DatasetProfile.model_validate(raw)
            self.assertFalse(TaskValidator().validate(legacy, task).valid)

    def test_explicit_metric_still_rejects_wrong_llm_metric(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pd.DataFrame({'x': range(20), 'target': range(20)}).to_csv(root / 'train.csv', index=False)
            (root / 'task.md').write_text('Regression. Predict target using MAPE.', encoding='utf-8')
            profile = DataInspector().inspect_directory(root)
            self.assertEqual(profile.task.primary_metric, 'mape')
            self.assertEqual(profile.task.field_sources['primary_metric'], 'explicit')
            task = TaskDefinition(objective='Predict target.', task_type='regression', target_columns=['target'],
                excluded_feature_columns=['target'], primary_metric='mae', metric_direction='minimize',
                validation_strategy='kfold', result_format=ResultFormat(
                    file_name='submission.csv', columns=['target'], prediction_columns=['target']))
            report = TaskValidator().validate(profile, task)
            self.assertFalse(report.valid)
            self.assertTrue(any(finding.code == 'metric_mismatch' and finding.severity == 'error'
                                for finding in report.findings))

    def test_inspector_and_execution_share_persisted_encoding_and_delimiter(self):
        for delimiter in [';', '\t', '|', ',']:
            with self.subTest(delimiter=delimiter), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path = root / 'train.csv'
                expected = pd.DataFrame({'category': ['Москва', 'Казань'] * 10, 'target': range(20)})
                expected.to_csv(path, sep=delimiter, encoding='cp1251', index=False)
                profile = DataInspector().inspect_directory(root)
                persisted = DatasetProfile.model_validate_json(profile.model_dump_json())
                options = persisted.table_formats[str(path.resolve())]
                self.assertEqual(options.encoding, 'cp1251')
                self.assertEqual(options.delimiter, delimiter)
                pd.testing.assert_frame_equal(read_table(path), expected)
                pd.testing.assert_frame_equal(read_table(path, table_format=options), expected)
                self.assertEqual(read_table(path, table_format=options, nrows=0).columns.tolist(), expected.columns.tolist())


if __name__ == '__main__':
    unittest.main()
