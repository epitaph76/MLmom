"""Agent-owned decisions with real execution and the remaining technical boundaries."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from mlmom.coding import CodeBundle
from mlmom.execution import export_params
from mlmom.pipeline import Pipeline
from mlmom.pipeline_contracts import RunConfig, HypothesisPlan, ResultReview
from mlmom.persistence import ExperimentStore
from mlmom.research import review_result
from tests.fakes import OfflineLLM
from tests.test_pipeline import dataset, only_sklearn


MINIMAL_CANDIDATE = '''from pathlib import Path
import cloudpickle
from sklearn.linear_model import Ridge

class Candidate:
    def fit(self, X, y, context):
        raise ValueError('repair this real fit error')
        self._estimator = Ridge(alpha=1.)
        self._estimator.fit(X, y, sample_weight=context.sample_weight)
        return self
    def predict(self, X, prediction_spec):
        return self._estimator.predict(X)
    def save(self, directory):
        Path(directory, 'candidate.pkl').write_bytes(cloudpickle.dumps(self))

def create_candidate(config): return Candidate()
def load_candidate(directory):
    return cloudpickle.loads(Path(directory, 'candidate.pkl').read_bytes())
'''


class MinimalChecksTests(unittest.TestCase):
    def test_export_is_optional_and_partial_json_is_not_a_fidelity_failure(self):
        self.assertEqual(export_params(object(), {'depth': 4}), {'depth': 4})
        model = SimpleNamespace(export_model_params=lambda: {'selected': 3})
        self.assertEqual(export_params(model, {'depth': 4}), {'selected': 3})
        for invalid in [[], {'bad': float('nan')}, {'opaque': object()}]:
            with self.subTest(invalid=type(invalid).__name__), self.assertRaises((ValueError, TypeError)):
                export_params(SimpleNamespace(export_model_params=lambda: invalid), {})

    def test_explanations_are_retained_without_becoming_model_parameters(self):
        plan = HypothesisPlan(hypothesis=dict(hypothesis_id='h', description='Test', rationale='Try'),
                              tool_id='sklearn', skill_ids=[], extra_explanation='Keep this')
        self.assertEqual(plan.agent_notes, {'extra_explanation': 'Keep this'})
        self.assertEqual(plan.model_params, {})

    def test_result_review_runs_without_machine_provenance(self):
        review = ResultReview(implementation='matches', effect='supported', explanation='Agent assessment')
        client = SimpleNamespace(generate=lambda *args: review)
        result = review_result(client, {}, [], 'no-provenance', 1)
        self.assertEqual(result, review)

    def test_real_fit_repair_csv_and_resume_without_audits_or_mandatory_smoke(self):
        class RepairClient(OfflineLLM):
            def generate(self, role, instruction, payload, schema, tools=None):
                if role == 'coding_agent':
                    self.calls.append((role, payload))
                    if schema.__name__ == 'CodePatch':
                        return schema(patches=[dict(path='candidate.py',
                            sha256=payload['file_hashes']['candidate.py'],
                            old_text="        raise ValueError('repair this real fit error')",
                            new_text='        # repaired from the worker traceback')])
                    return CodeBundle(files={'candidate.py': MINIMAL_CANDIDATE})
                return super().generate(role, instruction, payload, schema, tools)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = dataset(root/'input')
            config = RunConfig(timeouts_enabled=True, initial_boosting_sweep=False,
                feature_search_enabled=False, hpo_enabled=False, diagnostic_seconds=0,
                total_seconds=240, final_reserve_seconds=20, max_rounds=1,
                experiment_seconds=30, prediction_seconds=10,
                reserve_cpus=0, reserve_memory_gb=0)
            client = RepairClient(seconds=30)
            with patch('mlmom.pipeline.installed_tools', only_sklearn):
                result = Pipeline(task, config, run_dir=root/'run', client=client).run()
            self.assertEqual(result['status'], 'completed', result)
            output = pd.read_csv(task.parent/'submission.csv')
            test = pd.read_csv(task.parent/'test.csv')
            self.assertEqual(output['id'].tolist(), test['id'].tolist())
            self.assertTrue(np.isfinite(output['target']).all())
            store = ExperimentStore(root/'run')
            candidates = store.list('candidate')
            successful = [r for r in store.list('experiment') if r['status']=='completed']
            self.assertTrue(successful)
            self.assertTrue(all(c['execution_report'] is None for c in candidates))
            self.assertTrue(all(r['execution_report'] is None for r in successful))
            self.assertTrue(any(p.get('runtime_feedback') for role,p in client.calls if role=='coding_agent'))
            self.assertTrue(list((root/'run').rglob('traceback.log')))
            actions = [json.loads(path.read_text(encoding='utf-8')).get('action')
                       for path in (root/'run').rglob('request.json')]
            self.assertNotIn('smoke', actions)
            self.assertTrue(store.list('submission_refit'))
            store.close()
            with patch('mlmom.pipeline.installed_tools', only_sklearn):
                resumed = Pipeline(task, config, run_dir=root/'run', client=OfflineLLM(fail=True),
                                   resume=True).run(resume=True)
            self.assertEqual(resumed['submission'], result['submission'])


if __name__ == '__main__':
    unittest.main()
