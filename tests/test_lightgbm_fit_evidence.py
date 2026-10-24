"""LightGBM no-op rounds and fit-time categories retain independent execution proof."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from mlmom.execution import audit_fit, FidelityError
from mlmom.fit_provenance import capture
from mlmom.sdk import FitContext


class Candidate:
    def __init__(self, model, params):
        self.model, self.params = model, params

    def export_model_params(self):
        return copy.deepcopy(self.params)


class LightGBMFitEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(42)
        self.X = pd.DataFrame(rng.uniform(1, 10, (60, 4)), columns=['x', 'proto', 'service', 'state'])
        for c in ['proto', 'service', 'state']:
            self.X[c] = np.resize([0, 1, 2], 60)
        self.y = pd.Series(np.arange(60) % 10)

    def fit(self, requested=100, actual=None, categories=None, **extra):
        params = dict(objective='multiclass', num_class=10, n_estimators=requested,
            learning_rate=.1, num_leaves=31, min_child_samples=20,
            subsample=.8, subsample_freq=1, colsample_bytree=.8,
            reg_lambda=1., random_state=42, verbosity=-1, n_jobs=1)
        if categories is not None:
            params['categorical_feature'] = categories
        config = dict(contract_version=2, tool_id='lightgbm', model_params=params,
                      stopping_policy='fixed', _code_hash='test-source')
        backend = {k:v for k,v in params.items() if k != 'categorical_feature'}
        if actual is not None:
            backend['n_estimators'] = actual
        model = LGBMClassifier(**backend)
        context = FitContext('multiclass_classification', 'f1_macro', 'maximize', 42, 1, 2, 30)
        with capture(context, config, self.y, None):
            model.fit(self.X, self.y, **extra)
        return Candidate(model, params), config, context

    def audit(self, candidate, config, context):
        return audit_fit(candidate, config['model_params'], config, context,
                         self.root/'report.json', stage='smoke')

    def test_completed_backend_calls_with_noop_round_pass_without_changing_cap(self):
        candidate, config, context = self.fit(categories=['proto', 'service', 'state'],
            categorical_feature=['proto', 'service', 'state'])
        report = self.audit(candidate, config, context)
        actual = report['components']['model']
        self.assertEqual(actual['completed_iterations'], 100)
        self.assertEqual(actual['boosting_updates']['calls'], 100)
        self.assertGreater(actual['boosting_updates']['noop_updates'], 0)
        self.assertLess(actual['retained_iterations'], 100)
        self.assertEqual(actual['stopping_reason'], 'iteration_limit_with_noop_updates')
        self.assertEqual(report['exported_params']['n_estimators'], 100)
        self.assertEqual(actual['actual_params']['categorical_feature'], ['proto', 'service', 'state'])

    def test_short_fit_cannot_be_hidden_by_postfit_attribute_changes(self):
        candidate, config, context = self.fit(actual=10)
        candidate.model.n_estimators = 100
        with self.assertRaisesRegex(FidelityError, 'requested 100 iterations, completed 10'):
            self.audit(candidate, config, context)

    def test_fit_over_cap_is_still_rejected(self):
        candidate, config, context = self.fit(requested=5, actual=10)
        candidate.model.n_estimators = 5
        with self.assertRaisesRegex(FidelityError, 'iteration limit exceeded'):
            self.audit(candidate, config, context)

    def test_wrong_fit_time_categories_are_rejected(self):
        candidate, config, context = self.fit(categories=['proto', 'service'], categorical_feature=['proto'])
        with self.assertRaisesRegex(FidelityError, 'categorical_feature'):
            self.audit(candidate, config, context)

    def test_declared_categories_without_fit_evidence_are_still_rejected(self):
        candidate, config, context = self.fit(categories=['proto'])
        with self.assertRaisesRegex(FidelityError, 'categorical_feature'):
            self.audit(candidate, config, context)

    def test_permitted_early_stopping_preserves_selected_round(self):
        from lightgbm import early_stopping
        params = dict(objective='multiclass', n_estimators=100, min_child_samples=20,
                      n_jobs=1, verbosity=-1, random_state=42)
        config = dict(contract_version=2, tool_id='lightgbm', model_params=params,
                      stopping_policy='early_stopping', _code_hash='test-source')
        context = FitContext('multiclass_classification', 'log_loss', 'minimize', 42, 1, 2, 30,
                             validation_X=self.X.iloc[40:], validation_y=self.y.iloc[40:])
        model = LGBMClassifier(**params)
        with capture(context, config, self.y.iloc[:40], None):
            model.fit(self.X.iloc[:40], self.y.iloc[:40], eval_set=[(context.validation_X, context.validation_y)],
                      callbacks=[early_stopping(5, verbose=False)])
        frozen = dict(params, n_estimators=model.best_iteration_)
        report = self.audit(Candidate(model, frozen), config, context)
        self.assertEqual(report['components']['model']['stopping_reason'], 'early_stopping')
        self.assertEqual(report['exported_params']['n_estimators'], model.best_iteration_)
