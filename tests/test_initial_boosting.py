"""Required starter diversity, shared admission, and recoverable ensemble output."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mlmom.agents import StructuredOutputError
from mlmom.persistence import ExperimentStore
from mlmom.pipeline_contracts import Hypothesis, HypothesisBatch, PlanBatch, RunConfig
from mlmom.planning import build_planning_graph
from mlmom.search_groups import BOOSTERS, component_pool, initial_boosting_resources, collect_groups
from tests.fakes import OfflineLLM
from tests.test_planning_limits import case, Planner, Skills


class StarterPlanner(Planner):
    def generate(self, role, instruction, payload, schema, tools=None):
        if role == 'hypothesis_generator':
            return HypothesisBatch(hypotheses=[dict(hypothesis_id='starter', baseline=True,
                description='One LightGBM model', rationale='A cheap baseline', kind='single')])
        if role == 'planner':
            self.calls.append((role, payload))
            _, _, _, plan = case()
            plan.hypothesis = Hypothesis.model_validate(payload['hypotheses']['hypotheses'][0])
            plan.tool_id = payload['required_tool_id']
            plan.skill_ids = [plan.tool_id]
            plan.cpus, plan.memory_gb = 4, 4
            plan.model_params = dict(n_jobs=4) if plan.tool_id != 'catboost' else dict(thread_count=4)
            return PlanBatch(plans=[plan])
        return super().generate(role, instruction, payload, schema, tools)


class InitialPlanningTests(unittest.TestCase):
    def test_single_baseline_becomes_three_fixed_resource_sharing_starters(self):
        context, tools, budget, _ = case()
        client = StarterPlanner()
        result = build_planning_graph(client, context, tools, Skills(), budget,
            planning_mode='initial').invoke({'payload': {'round_number': 1}})
        self.assertTrue(result['valid'], result)
        self.assertEqual([p['tool_id'] for p in result['plans']], list(BOOSTERS))
        self.assertEqual(len(result['groups']), 1)
        self.assertEqual(result['groups'][0]['kind'], 'family')
        self.assertEqual(sum(p['hypothesis']['baseline'] for p in result['plans']), 1)
        self.assertTrue(all(p['strategy'] == 'single' and p['max_trials'] == 1 for p in result['plans']))
        self.assertLessEqual(sum(p['cpus'] for p in result['plans']), budget.available_cpus)
        self.assertLessEqual(sum(p['memory_gb'] for p in result['plans']), budget.available_memory_gb)
        self.assertTrue(all(not p['model_params'] for p in result['plans']))
        self.assertEqual({payload['required_tool_id'] for role, payload in client.calls if role == 'planner'}, set(BOOSTERS))

    def test_missing_backend_is_reported_and_low_capacity_reduces_concurrency(self):
        context, tools, budget, _ = case()
        tools['catboost'].available = False
        tools['catboost'].reason = 'distribution not installed'
        result = build_planning_graph(StarterPlanner(), context, tools, Skills(), budget,
            planning_mode='initial', initial_max_workers=1).invoke({'payload': {'round_number': 1}})
        self.assertTrue(result['valid'], result)
        self.assertEqual([p['tool_id'] for p in result['plans']], ['lightgbm', 'xgboost'])
        self.assertEqual(result['skipped'][0]['tool_id'], 'catboost')
        self.assertEqual(result['plans'][0]['cpus'], budget.available_cpus)
        small = budget.model_copy(update=dict(available_cpus=2, available_memory_gb=2))
        self.assertEqual(initial_boosting_resources(small, 3, 3), dict(workers=2, cpus=1, memory_gb=1))

    def test_failed_first_backend_does_not_discard_other_starters(self):
        class BrokenFirst(StarterPlanner):
            def generate(self, role, instruction, payload, schema, tools=None):
                result = super().generate(role, instruction, payload, schema, tools)
                if role == 'planner' and payload['required_tool_id'] == 'lightgbm':
                    result.plans[0].tool_id = 'sklearn'
                return result
        context, tools, budget, _ = case()
        result = build_planning_graph(BrokenFirst(), context, tools, Skills(), budget,
            planning_mode='initial', max_attempts=1).invoke({'payload': {'round_number': 1}})
        self.assertTrue(result['valid'], result)
        self.assertEqual([p['tool_id'] for p in result['plans']], ['catboost', 'xgboost'])
        self.assertTrue(result['plans'][0]['hypothesis']['baseline'])
        self.assertEqual(len(result['errors']), 1)

    def test_lineage_does_not_replace_a_single_hypothesis_group_identity(self):
        _, _, _, plan = case()
        plan.parent_hypothesis = plan.hypothesis.model_copy(update={'hypothesis_id': 'old-parent'})
        group = collect_groups([plan])[0]
        self.assertEqual(group.hypothesis.hypothesis_id, plan.hypothesis.hypothesis_id)

    def test_ensemble_schema_failure_is_repaired_within_the_same_proposal_window(self):
        from mlmom.ensemble_planning import plan_ensemble
        from mlmom.pipeline_contracts import EnsembleProposal
        context, tools, budget, _ = case()
        context.task.model_dump = lambda **kwargs: dict(task_type='regression')
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperimentStore(Path(tmp))
            try:
                client = SimpleNamespace(generate=Mock(side_effect=[StructuredOutputError('Extra field skip_reason'),
                    EnsembleProposal(skip=True, rationale='No defensible composition')]))
                pipeline = SimpleNamespace(context=context, config=RunConfig(), tools=tools, client=client,
                    store=store, skills=Skills(), budget=SimpleNamespace(snapshot=lambda: budget), admission_cost=lambda: 1)
                pool = [dict(experiment_id=str(i), plan=dict(tool_id='sklearn')) for i in range(2)]
                with patch('mlmom.ensemble_planning.component_pool', return_value=pool), \
                     patch('mlmom.ensemble_planning.component_brief', side_effect=lambda row: row):
                    result = plan_ensemble(pipeline, {'round_number': 2})
                    self.assertEqual(result['reason'], 'No defensible composition')
                    self.assertEqual(client.generate.call_count, 2)
                    self.assertIn('skip_reason', client.generate.call_args_list[1].args[2]['validation_feedback'])
                    self.assertEqual(plan_ensemble(pipeline, {'round_number': 2}), result)
                    self.assertEqual(client.generate.call_count, 2)
            finally:
                store.close()


def native_source(backend):
    imports = {
        'lightgbm': 'from lightgbm import LGBMRegressor as Regressor, LGBMClassifier as Classifier',
        'catboost': 'from catboost import CatBoostRegressor as Regressor, CatBoostClassifier as Classifier',
        'xgboost': 'from xgboost import XGBRegressor as Regressor, XGBClassifier as Classifier',
    }
    threads = "dict(thread_count=context.cpus, verbose=False, allow_writing_files=False)" if backend == 'catboost' else 'dict(n_jobs=context.cpus)'
    return imports[backend] + '''
from pathlib import Path
import cloudpickle
import numpy as np
class Candidate:
    def __init__(self, config):
        self.params = dict(config['model_params'])
    def fit(self, X, y, context):
        self.classification = 'classification' in context.task_type
        self.classes = list(context.classes)
        values = np.asarray(y)
        if self.classification:
            values = np.asarray([self.classes.index(v) for v in values])
        params = dict(self.params, **THREADS)
        cls = Classifier if self.classification else Regressor
        self.model = cls(**params)
        kwargs = {} if context.sample_weight is None else dict(sample_weight=context.sample_weight)
        self.model.fit(X, values, **kwargs)
        return self
    def predict(self, X, prediction_spec):
        if prediction_spec.get('kind') == 'probability':
            return self.model.predict_proba(X)
        values = np.asarray(self.model.predict(X)).reshape(-1)
        return np.asarray(self.classes)[values.astype(int)] if self.classification else values
    def export_model_params(self):
        return dict(self.params)
    def save(self, artifact_dir):
        (Path(artifact_dir) / 'candidate.pkl').write_bytes(cloudpickle.dumps(self))
def create_candidate(config):
    return Candidate(config)
def load_candidate(artifact_dir):
    return cloudpickle.loads((Path(artifact_dir) / 'candidate.pkl').read_bytes())
'''.replace('THREADS', threads)


class NativeStarterClient(OfflineLLM):
    def complete(self, messages):
        task = json.loads(super().complete(messages))
        task['validation_folds'] = 2
        return json.dumps(task)

    def generate(self, role, instruction, payload, schema, tools=None):
        result = None
        if role == 'coding_agent':
            self.calls.append((role, payload))
            return schema(files={'candidate.py': native_source(payload['plan']['tool_id'])})
        result = super().generate(role, instruction, payload, schema, tools)
        if role == 'planner':
            for plan in result.plans:
                plan.tool_id = payload['required_tool_id']
                plan.skill_ids = [plan.tool_id]
                parameter = 'iterations' if plan.tool_id == 'catboost' else 'n_estimators'
                plan.search_space = {parameter: dict(type='int', low=4, high=6)}
                plan.model_params = dict(iterations=8, depth=3) if plan.tool_id == 'catboost' else dict(n_estimators=8, max_depth=3)
                if plan.tool_id == 'lightgbm':
                    plan.model_params.update(num_leaves=7, min_child_samples=2, verbosity=-1)
        return result


class InitialBoostingIntegrationTests(unittest.TestCase):
    def test_three_native_backends_overlap_and_remain_ensemble_sources(self):
        from mlmom.pipeline import Pipeline
        from mlmom.registry import installed_tools
        from tests.test_pipeline import dataset
        registry = installed_tools()
        if any(registry[key].version is None for key in BOOSTERS):
            self.skipTest('Run inside the ML Docker image with all three native libraries')
        for kind in ['regression', 'multiclass_classification']:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = RunConfig(max_rounds=1, hpo_enabled=kind == 'regression', feature_search_enabled=True,
                    hpo_max_trials=2, hpo_batch_size=2, diagnostic_seconds=0,
                    max_workers=3, reserve_cpus=0, reserve_memory_gb=0)
                pipeline = Pipeline(dataset(root/'input', kind), config, run_dir=root/'run', client=NativeStarterClient())
                result = pipeline.run()
                self.assertEqual(result['status'], 'completed', result)
                store = ExperimentStore(root/'run')
                try:
                    runs = [row for row in store.list('experiment') if row['round_id'] == 'r1']
                    self.assertEqual(len(runs), 4 if config.hpo_enabled else 3, result)
                    self.assertTrue(all(row['status'] == 'completed' for row in runs), runs)
                    self.assertEqual(len({row['split_id'] for row in runs}), 1)
                    self.assertTrue(all(len(row['trials'][0]['fold_results']) == 2 for row in runs))
                    pool = component_pool(store, 2)
                    self.assertEqual({row['plan']['tool_id'] for row in pool}, set(BOOSTERS))
                    self.assertTrue((root/'run/cycles/r1/submission.csv').is_file())
                    group = store.list('search_group_result')[0]
                    self.assertEqual({row['tool_id'] for row in group['variants']}, set(BOOSTERS))
                finally:
                    store.close()
                active, peak, starter_releases, hpo_admissions, feature_starts = {}, 0, [], [], []
                for line in (root/'run/trace.jsonl').read_text(encoding='utf-8').splitlines():
                    event = json.loads(line)
                    if event['kind'] == 'feature.started':
                        feature_starts.append(event['seq'])
                    if event['kind'] not in {'scheduler.admitted', 'scheduler.released'}:
                        continue
                    payload = json.loads((root/'run'/event['payload']['path']).read_text(encoding='utf-8'))
                    directory = payload['directory']
                    if '/runs/r1-' not in directory.replace('\\', '/'):
                        continue
                    if directory.endswith('-submission-refit'):
                        continue
                    if event['kind'] == 'scheduler.admitted' and payload.get('action') == 'train':
                        active[directory] = payload
                        if directory.endswith('-hpo'):
                            hpo_admissions.append(event['seq'])
                    elif event['kind'] == 'scheduler.released':
                        active.pop(directory, None)
                        if not directory.endswith('-hpo'):
                            starter_releases.append(event['seq'])
                    peak = max(peak, len(active))
                    self.assertLessEqual(sum(row['cpus'] for row in active.values()), pipeline.budget.cpus)
                    self.assertLessEqual(sum(row['memory_gb'] for row in active.values()), pipeline.budget.memory_gb)
                self.assertEqual(peak, 3, 'All three training processes must actually overlap')
                self.assertEqual(len(starter_releases), 3)
                self.assertTrue(feature_starts)
                self.assertGreater(min(feature_starts), max(starter_releases))
                if config.hpo_enabled:
                    self.assertEqual(len(hpo_admissions), 1)
                    self.assertGreater(hpo_admissions[0], max(starter_releases))


if __name__ == '__main__':
    unittest.main()
