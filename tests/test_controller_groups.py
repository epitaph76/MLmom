"""Regressions for controller-owned group membership and independent recipes."""
import json
import tempfile
import unittest
from contextlib import ExitStack
from concurrent.futures import CancelledError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mlmom.ensemble_planning import plan_ensemble
from mlmom.feature_contracts import FeatureSet, FeatureSpec
from mlmom.persistence import ExperimentStore
from mlmom.pipeline_contracts import (EnsembleProposal, ExperimentRun, Hypothesis,
    HypothesisPlan, PlanBatch, RunConfig, ValidationReport)
from mlmom.planning import build_planning_graph
from tests.test_planning_limits import case, Skills
from tests.test_search_groups import composite_case
from mlmom.sdk import load_module


RUNTIME_FIELDS = {'search_group_id', 'search_group_kind', 'parent_hypothesis'}


class GroupClient:
    def __init__(self):
        self.calls = []
        self.reviews = []

    def generate(self, role, instruction, payload, schema, tools=None):
        self.calls.append(role)
        if role == 'ensemble_agent':
            return EnsembleProposal(component_experiment_ids=['e0', 'e1'],
                rationale='Compare explicit frozen recipes', methods=['mean', 'median', 'weighted'],
                oof_folds=2, weight_trials=2)
        if role == 'planner':
            plan = case()[3].model_dump(mode='json')
            plan['hypothesis'] = payload['hypotheses']['hypotheses'][0]
            # These legacy fields caused the real run to fail after approval.
            plan.update(search_group_id='r5-ensemble-lgb-cat-xgb-oof',
                        search_group_kind='ensemble', parent_hypothesis=None)
            return schema.model_validate({'plans': [plan]})
        if role == 'plan_validator':
            self.reviews.append(payload)
            return ValidationReport(valid=True)
        raise AssertionError(role)


def graph_input():
    hypothesis = Hypothesis(hypothesis_id='ensemble-lgb-cat-xgb-oof-r5', kind='ensemble',
        description='Compare frozen models', rationale='Measured complementary errors')
    ordinary = Hypothesis(hypothesis_id='independent-imputation',
        description='Independent imputation experiment', rationale='Missing observations')
    return dict(payload={'round_number': 5},
                hypotheses={'hypotheses': [hypothesis.model_dump(), ordinary.model_dump()]})


class ControllerGroupTests(unittest.TestCase):
    def test_old_runtime_session_is_preserved_and_requires_a_new_run(self):
        from mlmom.pipeline import Pipeline
        from tests.test_pipeline import dataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = dataset(root / 'task')
            run = root / 'old-run'
            run.mkdir()
            session = run / 'session.json'
            original = json.dumps({'runtime_contract_version': 3}).encode()
            session.write_bytes(original)
            with self.assertRaisesRegex(ValueError, 'Start a new run'):
                Pipeline(task, RunConfig(), run_dir=run, client=GroupClient(), resume=True)
            self.assertEqual(session.read_bytes(), original)

    def test_planner_schema_contains_no_membership_assignments(self):
        schema = PlanBatch.model_json_schema()
        plan = schema['$defs']['PlannerPlan']
        self.assertTrue(RUNTIME_FIELDS.isdisjoint(plan['properties']))
        self.assertNotIn('HypothesisPlan', schema['$defs'])

    def test_obsolete_membership_hints_cannot_refuse_or_change_an_experiment(self):
        raw = case()[3].model_dump(mode='json')
        raw.update(search_group_id={'invalid': 'old ID'}, search_group_kind=123,
                   parent_hypothesis='obsolete narrative')
        decoded = PlanBatch.model_validate({'plans': [raw]}).plans[0]
        self.assertTrue(RUNTIME_FIELDS.isdisjoint(decoded.model_dump()))
        self.assertEqual(decoded.model_params, raw['model_params'])
        self.assertEqual(decoded.hypothesis.hypothesis_id, 'base')

    def test_real_failure_labels_keep_both_ensemble_and_independent_plan(self):
        context, tools, budget, _ = case()
        client = GroupClient()
        result = build_planning_graph(client, context, tools, Skills(), budget).invoke(graph_input())
        self.assertTrue(result['valid'], result)
        self.assertEqual(len(result['plans']), 2)
        self.assertEqual({group['hypothesis']['hypothesis_id'] for group in result['groups']},
                         {'ensemble-lgb-cat-xgb-oof-r5', 'independent-imputation'})
        self.assertEqual(client.calls.count('planner'), 2)
        self.assertEqual(client.calls.count('plan_validator'), 2)

    def test_controller_parent_is_bound_before_review_and_cache_reuse(self):
        context, tools, budget, _ = case()
        parent = Hypothesis(hypothesis_id='controller-ensemble', kind='ensemble',
            description='Frozen recipe group', rationale='Explicit controller assignment')
        state = graph_input()
        state['hypotheses']['hypotheses'] = state['hypotheses']['hypotheses'][:1]
        saved = {}
        def save(progress):
            saved.update(json.loads(json.dumps(progress)))
        client = GroupClient()
        first = build_planning_graph(client, context, tools, Skills(), budget,
            group_parent=parent, save_progress=save).invoke(state)
        reviewed = client.reviews[0]['plans'][0]
        self.assertEqual(reviewed['search_group_id'], parent.hypothesis_id)
        self.assertEqual(reviewed['parent_hypothesis']['hypothesis_id'], parent.hypothesis_id)
        for phase in ['proposed', 'accepted']:
            plan = saved[phase]['ensemble-lgb-cat-xgb-oof-r5']
            plan.update(search_group_id='stale-LLM-label', parent_hypothesis=None)
        resumed_client = GroupClient()
        resumed = build_planning_graph(resumed_client, context, tools, Skills(), budget,
            group_parent=parent, saved_progress=saved).invoke(state)
        self.assertEqual(resumed['plans'], first['plans'])
        self.assertEqual(resumed['groups'], first['groups'])
        self.assertEqual(resumed_client.calls, [])


class IndependentRecipeTests(unittest.TestCase):
    def make_pipeline(self, root):
        code, context, plan, X, y, fit = composite_case(root / 'sources')
        _, tools, budget, _ = case()
        store = ExperimentStore(root / 'store')
        store.put('cycle', 'r1', dict(round_id='r1', completed=True))
        for i, source in enumerate(plan.ensemble_sources):
            base = HypothesisPlan.model_validate(source['config'])
            if i == 0:
                base.feature_set = FeatureSet(feature_set_id='registered-dsl', features=[
                    FeatureSpec(name='missing_x', op='missing', columns=[X.columns[0]])])
                base.feature_set_id = base.feature_set.feature_set_id
            run = ExperimentRun(experiment_id=f'e{i}', round_id='r1', candidate_id=f'c{i}',
                split_id='r1-split', status='completed', internal_score=1.0+i,
                best_params=base.model_params, duration_seconds=1.0)
            store.put('component', f'e{i}', dict(round_id='r1', experiment_id=f'e{i}',
                candidate={'candidate_id': f'c{i}', 'code_hash': source['code_hash']},
                configuration_hash=f'config{i}', plan=base.model_dump(mode='json'),
                run=run.model_dump(mode='json')))
        pipeline = SimpleNamespace(store=store, context=context, tools=tools, skills=Skills(),
            client=GroupClient(), config=RunConfig(),
            budget=SimpleNamespace(snapshot=lambda: budget),
            planning_payload=lambda state: dict(round_number=state['round_number']))
        return pipeline, code, X, y, fit

    def test_recipe_internal_error_preserves_preceding_and_following_recipes(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as cleanup:
            pipeline, _, _, _, _ = self.make_pipeline(Path(tmp))
            cleanup.callback(pipeline.store.close)
            real_builder = build_planning_graph
            def builder(*args, **kwargs):
                real = real_builder(*args, **kwargs)
                def invoke(state, config):
                    if state['payload']['ensemble_recipe']['method'] == 'median':
                        raise TypeError('Internal recipe diagnostic')
                    return real.invoke(state, config)
                return SimpleNamespace(invoke=invoke)
            with patch('mlmom.ensemble_planning.build_planning_graph', side_effect=builder):
                result = plan_ensemble(pipeline, {'round_number': 2})
            self.assertEqual([p['model_params']['method'] for p in result['plans']], ['mean', 'weighted'])
            self.assertEqual(result['errors'][0]['method'], 'median')
            self.assertEqual(result['errors'][0]['error_type'], 'TypeError')
            calls = len(pipeline.client.calls)
            self.assertEqual(plan_ensemble(pipeline, {'round_number': 2}), result)
            self.assertEqual(len(pipeline.client.calls), calls)

    def test_cancellation_is_not_recorded_as_a_recipe_failure(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as cleanup:
            pipeline, _, _, _, _ = self.make_pipeline(Path(tmp))
            cleanup.callback(pipeline.store.close)
            graph = SimpleNamespace(invoke=lambda *args: (_ for _ in ()).throw(CancelledError()))
            with patch('mlmom.ensemble_planning.build_planning_graph', return_value=graph):
                with self.assertRaises(CancelledError):
                    plan_ensemble(pipeline, {'round_number': 2})
            self.assertIsNone(pipeline.store.get('ensemble_planning', 'r2'))

    def test_semantic_refusal_of_one_recipe_keeps_other_recipes(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as cleanup:
            pipeline, _, _, _, _ = self.make_pipeline(Path(tmp))
            cleanup.callback(pipeline.store.close)
            original = pipeline.client.generate
            def generate(role, instruction, payload, schema, tools=None):
                if role == 'plan_validator' and payload['plans'][0]['model_params']['method'] == 'median':
                    return ValidationReport(valid=False, errors=['Specific methodological rejection'])
                return original(role, instruction, payload, schema, tools)
            pipeline.client.generate = generate
            result = plan_ensemble(pipeline, {'round_number': 2})
            self.assertEqual([p['model_params']['method'] for p in result['plans']], ['mean', 'weighted'])
            self.assertTrue(any(e['method'] == 'median' for e in result['errors']))

    def test_bound_recipes_reach_fit_and_keep_frozen_dsl(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as cleanup:
            root = Path(tmp)
            pipeline, code, X, y, fit = self.make_pipeline(root)
            cleanup.callback(pipeline.store.close)
            result = plan_ensemble(pipeline, {'round_number': 2})
            self.assertEqual(len(result['plans']), 3, result)
            for plan in result['plans']:
                self.assertEqual(plan['search_group_id'], 'r2-ensemble')
                sources = plan['ensemble_sources']
                self.assertTrue(any(s['config']['feature_set_id'] == 'registered-dsl' for s in sources))
                # Fit the actual SDK composition with its unchanged source assignments.
                fit.task_context.feature_columns = list(X.columns)
                model = load_module(code).create_candidate(plan)
                model.fit(X, y, fit)
                self.assertTrue(np.isfinite(model.predict(X, {'kind': 'value'})).all())
                if plan['model_params']['method'] in {'mean', 'median'}:
                    self.assertNotIn('oof_folds', plan['model_params'])
                    self.assertNotIn('weight_trials', plan['model_params'])


if __name__ == '__main__':
    unittest.main()
