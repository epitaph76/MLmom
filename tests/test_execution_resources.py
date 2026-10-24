"""Real fixed refits can scale CPU threads without changing frozen model behavior."""
import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mlmom.execution import audit_fit, complete_export, config_hash, FidelityError
from mlmom.execution_resources import execution_config, model_parameters
from mlmom.model_spec import ModelSpec, model_spec, with_spec
from mlmom.persistence import tree_hash
from mlmom.sdk import FitContext, load_module
from mlmom.worker import train
from tests.test_execution import execution_case


SOURCE = '''import copy
from pathlib import Path
import cloudpickle

class Candidate:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
    def fit(self, X, y, context):
        from lightgbm import LGBMRegressor
        from xgboost import XGBRegressor
        from catboost import CatBoostRegressor
        backend = {'lightgbm': LGBMRegressor, 'xgboost': XGBRegressor,
                   'catboost': CatBoostRegressor}[self.config['tool_id']]
        self.model = backend(**self.config['model_params'])
        self.model.fit(X, y)
        return self
    def export_model_params(self):
        return copy.deepcopy(self.config['model_params'])
    def predict(self, X, prediction_spec):
        return self.model.predict(X)
    def save(self, directory):
        Path(directory, 'candidate.pkl').write_bytes(cloudpickle.dumps(self))

def create_candidate(config): return Candidate(config)
def load_candidate(directory):
    return cloudpickle.loads(Path(directory, 'candidate.pkl').read_bytes())
'''


PARAMETERS = {
    'lightgbm': dict(n_estimators=8, num_leaves=7, min_child_samples=1,
                     learning_rate=.1, random_state=42, verbosity=-1, n_jobs=2),
    'xgboost': dict(n_estimators=8, max_depth=2, learning_rate=.1,
                    tree_method='hist', random_state=42, n_jobs=2),
    'catboost': dict(iterations=8, depth=2, learning_rate=.1, random_seed=42,
                     verbose=False, allow_writing_files=False, thread_count=2),
}


class ExecutionResourceTests(unittest.TestCase):
    def test_real_fixed_refits_use_two_then_six_cpus_with_one_model_identity(self):
        for backend, params in PARAMETERS.items():
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                frame, context, plan, _ = execution_case(root, 'regression', backend)
                code = root / 'resource-code'
                code.mkdir()
                (code / 'candidate.py').write_text(SOURCE, encoding='utf-8')
                plan = plan.model_copy(update=dict(contract_version=2, model_params=params))
                reports = []
                for cpus in [2, 6]:
                    output = root / f'fit-{cpus}'
                    output.mkdir()
                    result = train(dict(context=context.model_dump(mode='json'),
                        plan=plan.model_dump(mode='json'), code_dir=str(code), code_hash=tree_hash(code),
                        train_path=str(root / 'train.csv'), worker_seconds=60, worker_cpus=cpus,
                        worker_memory_gb=2, timeouts_enabled=False, fixed=True, phase='submission'), output)
                    report = result['execution_report']
                    reports.append(report)
                    self.assertEqual(result['status'], 'completed')
                    self.assertEqual(report['status'], 'passed')
                    key = 'thread_count' if backend == 'catboost' else 'n_jobs'
                    self.assertEqual(report['execution_resources']['declared_threads'], {key: 2})
                    self.assertEqual(report['execution_resources']['exported_threads'], {key: cpus})
                    actual = report['components']['model']
                    self.assertEqual(actual['fit_entry']['params'][key], cpus)
                    self.assertEqual(actual['fit_entry']['rows'], len(frame))
                    self.assertNotIn(key, report['model_spec']['backend_params'])
                    loaded = load_module(code).load_candidate(result['model_artifact'])
                    prediction = loaded.predict(frame[context.feature_columns], {'kind': 'value'})
                    self.assertEqual(len(prediction), len(frame))
                    self.assertTrue(np.isfinite(prediction).all())
                self.assertEqual(reports[0]['configuration_hash'], reports[1]['configuration_hash'])
                self.assertEqual(reports[0]['model_spec'], reports[1]['model_spec'])

    def test_thread_exemption_does_not_hide_learning_changes_or_cpu_overruns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame, task, _, _ = execution_case(root, 'regression', 'xgboost')
            code = root / 'resource-code'
            code.mkdir()
            (code / 'candidate.py').write_text(SOURCE, encoding='utf-8')
            config = dict(contract_version=2, tool_id='xgboost', model_params=PARAMETERS['xgboost'])
            context = FitContext('regression', 'rmse', 'minimize', 42, 6, 2, 60,
                                 timeouts_enabled=False, fit_rows=len(frame))
            model = load_module(code).create_candidate(execution_config(config, 6))
            model.fit(frame[task.feature_columns], frame[task.task.target_columns[0]], context)
            for key, value in [('max_depth', 3), ('learning_rate', .2), ('random_state', 43),
                               ('n_estimators', 12)]:
                requested = dict(config['model_params'], **{key: value})
                with self.subTest(key=key), self.assertRaisesRegex(FidelityError, key):
                    audit_fit(model, requested, config, context, root / f'{key}.json', stage='full_refit')
            with self.assertRaisesRegex(FidelityError, 'CPU allocation'):
                audit_fit(model, config['model_params'], config, replace(context, cpus=2),
                          root / 'over-budget.json', stage='full_refit')
            # Fit-entry evidence still wins over a post-fit attempt to disguise the thread count.
            model.estimator.model.set_params(n_jobs=2)
            model.estimator.config['model_params']['n_jobs'] = 2
            with self.assertRaisesRegex(FidelityError, 'CPU allocation'):
                audit_fit(model, config['model_params'], config, replace(context, cpus=2),
                          root / 'spoofed-budget.json', stage='full_refit')

    def test_nested_aliases_rebind_without_changing_device_seed_or_model_parameters(self):
        params = dict(left=dict(n_jobs=2, nthread=2, max_depth=4),
                      right=dict(num_threads=2, thread_count=2, iterations=8),
                      layers=[dict(num_cpus=2, depth=3)],
                      forest__n_jobs=2, random_state=42, device='cpu', weights=[.6, .4])
        config = dict(tool_id='sklearn', model_params=params)
        original = copy.deepcopy(config)
        rebound = execution_config(config, 6)
        self.assertEqual(config, original)
        self.assertEqual(rebound['model_params']['left']['nthread'], 6)
        self.assertEqual(rebound['model_params']['right']['thread_count'], 6)
        self.assertEqual(rebound['model_params']['forest__n_jobs'], 6)
        self.assertEqual(rebound['model_params']['layers'], [dict(num_cpus=6, depth=3)])
        first, second = model_spec(config), model_spec(rebound)
        self.assertEqual(first, second)
        self.assertEqual(first['backend_params']['device'], 'cpu')
        for key, value in [('random_state', 43), ('device', 'cuda')]:
            changed = model_spec(dict(config, model_params=dict(params, **{key: value})))
            self.assertNotEqual(config_hash(first), config_hash(changed))
        restored = with_spec(config, second)
        self.assertEqual(restored['model_params'], params)
        composite = dict(model_params={}, ensemble_components=[dict(model_params=params)],
                         ensemble_sources=[dict(config=config)])
        rebound_composite = execution_config(composite, 6)
        self.assertEqual(rebound_composite['ensemble_components'][0]['model_params'],
                         rebound['model_params'])
        self.assertEqual(rebound_composite['ensemble_sources'][0]['config'], rebound)
        self.assertEqual(composite['ensemble_components'][0]['model_params'], params)
        self.assertEqual(ModelSpec.model_validate(dict(first, backend_params=params)).model_dump(), first)
        for value in [-1, 0, 7, True]:
            export = SimpleNamespace(export_model_params=lambda: dict(n_jobs=value))
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'CPU allocation'):
                complete_export(export, {'n_jobs': 2}, cpus=6)
        self.assertEqual(model_parameters(params), first['backend_params'])


if __name__ == '__main__':
    unittest.main()
