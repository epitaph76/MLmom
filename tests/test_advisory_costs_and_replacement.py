"""Historical costs cannot veto fits; feature swaps are one resumable experiment."""
import copy
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from mlmom.coding import build_coding_graph
from mlmom.feature_contracts import FeatureProposal, FeatureSet, FeatureSpec
from mlmom.feature_search import run_feature_session
from mlmom.persistence import ExperimentStore, atomic_json
from mlmom.pipeline import Pipeline
from mlmom.pipeline_contracts import RunConfig
from tests.fakes import OfflineLLM
from tests import test_controller_groups as controller_cases
from tests import test_feature_session as feature_cases
from tests.test_pipeline import dataset, only_sklearn
from tests.test_planning_limits import case, ClockBudget, Coder, Sandbox, Store
from tests.test_search_groups import COMPOSITION_CODE


class AdvisoryCostTests(unittest.TestCase):
    def test_positive_time_reaches_ensemble_agent_without_a_cost_threshold(self):
        from mlmom.ensemble_planning import plan_ensemble
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as cleanup:
            p, *_ = controller_cases.IndependentRecipeTests().make_pipeline(Path(tmp))
            cleanup.callback(p.store.close)
            snapshot = case()[2].model_copy(update=dict(timeouts_enabled=True,
                remaining_seconds=20, research_seconds=1))
            p.budget.snapshot = lambda: snapshot
            result = plan_ensemble(p, {'round_number': 2})
            self.assertEqual(len(result['plans']), 3, result)
            self.assertIn('ensemble_agent', p.client.calls)

    def test_long_plan_can_be_coded_inside_actual_remaining_deadline(self):
        context, tools, _, plan = case()
        plan.time_limit_seconds = 10000
        budget = ClockBudget()
        budget.elapsed = 870  # 30 research seconds, despite a large requested fit.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store, client = Store(root), Coder(budget, 5)
            answer = build_coding_graph(client, workspace=root/'code', store=store,
                plan=plan, plan_ref=root/'plan.json', skill_data=[], tools=tools,
                context=context, sandbox=Sandbox(budget), budget=budget,
                config=RunConfig(timeouts_enabled=True), round_id='r2').invoke({'payload': {}})
            self.assertTrue(answer['valid'], answer)
            self.assertEqual(client.calls, 1)
            self.assertEqual(store.get('coding_budget', 'r2-base')['deadline'], 1900)

    def test_retries_use_actual_deadline_without_forecasted_complete_fit(self):
        p = object.__new__(Pipeline)
        p.config = RunConfig(target_completed_rounds=3, max_rounds=3)
        remaining = 1
        p.budget = SimpleNamespace(snapshot=lambda: SimpleNamespace(research_seconds=remaining))
        self.assertTrue(p.can_retry_round({'round_number': 2}))
        self.assertTrue(p.can_retry_planning({'round_number': 2}))
        remaining = 0
        self.assertFalse(p.can_retry_round({'round_number': 2}))
        self.assertFalse(p.can_retry_planning({'round_number': 2}))

    def test_large_source_study_durations_do_not_skip_actual_ensemble_fits(self):
        class Client(OfflineLLM):
            pipeline = None

            def generate(self, role, instruction, payload, schema, tools=None):
                if role == 'hypothesis_generator':
                    number = payload['round_number']
                    return schema(hypotheses=[dict(hypothesis_id=f'h{number}-{i}',
                        description='Small independent tree', rationale='Measured comparison',
                        baseline=i == 0) for i in range(2 if number == 1 else 1)])
                if role == 'planner' and schema.__name__ == 'PlanBatch':
                    result = super().generate(role, instruction, payload, schema, tools)
                    for plan in result.plans:
                        plan.model_params['n_estimators'] = 3 if plan.hypothesis.hypothesis_id.endswith('-0') else 4
                    return result
                if role == 'ensemble_agent':
                    self.calls.append((role, payload))
                    self.assert_duration_scope = all('not selected-fit' in c['duration_scope'] for c in payload['components'])
                    chosen = [c['experiment_id'] for c in payload['components'][:2]]
                    for experiment_id in chosen:
                        run = self.pipeline.store.get('experiment', experiment_id)
                        run['duration_seconds'] = 1000000  # Whole-study history, not a frozen fit estimate.
                        self.pipeline.store.put('experiment', experiment_id, run)
                    return schema(component_experiment_ids=chosen, methods=['mean', 'weighted'],
                        rationale='Measure both recipes', oof_folds=2, weight_trials=2)
                if role == 'coding_agent' and payload['plan'].get('ensemble_sources'):
                    self.calls.append((role, payload))
                    return schema(files={'candidate.py': COMPOSITION_CODE})
                return super().generate(role, instruction, payload, schema, tools)

        with tempfile.TemporaryDirectory() as tmp, ExitStack() as cleanup:
            root = Path(tmp)
            client = Client(seconds=60)
            config = RunConfig(timeouts_enabled=True, total_seconds=900, final_reserve_seconds=30,
                max_rounds=2, max_workers=3, preparation_workers=2, reserve_cpus=0,
                reserve_memory_gb=0, hpo_enabled=False, feature_search_enabled=False,
                prediction_seconds=5, diagnostic_seconds=0)
            with patch('mlmom.pipeline.installed_tools', only_sklearn):
                p = Pipeline(dataset(root/'input'), config, run_dir=root/'run', client=client)
                client.pipeline = p
                result = p.run()
            self.assertEqual(result['status'], 'completed', result)
            self.assertTrue(client.assert_duration_scope)
            store = ExperimentStore(root/'run')
            cleanup.callback(store.close)
            group = next(g for g in store.list('search_group_result') if g['kind'] == 'ensemble')
            self.assertEqual(group['status'], 'completed', group)
            self.assertEqual({v['method'] for v in group['variants']}, {'mean', 'weighted'})
            self.assertTrue(all(store.get('experiment', v['experiment_id'])['status'] == 'completed'
                                for v in group['variants']), group)
            self.assertFalse(any(s.get('reason') == 'insufficient_complete_recipe_budget' for s in group['skipped']))
            self.assertIsNotNone(result['submission'])


class AtomicReplacementTests(unittest.TestCase):
    def make_session(self, root, *, legacy=False, interrupt=False, unknown=False, gain=True):
        p, baseline, split, path = feature_cases.SessionAccountingTests().harness(root, fail=False)
        specs = [FeatureSpec(name='old'+str(i), op='product', columns=['x0', 'x1']) for i in range(16)]
        definition = FeatureSet(feature_set_id='old-set', features=specs)
        baseline.plan.feature_set_id, baseline.plan.feature_set = definition.feature_set_id, definition
        p.config.feature_max_configurations = 1
        p.config.feature_plateau_limit = 1
        proposals, assignments, fits = [], [], []
        reference = copy.deepcopy(baseline.run.model_dump(mode='json'))
        p.client.generate = self.generate(proposals, assignments, legacy, unknown)

        def run(request, directory, **limits):
            if request['action'] == 'feature_profile':
                return dict(status='completed', profile=dict(rows=len(p.frame), columns=[], residual_slices=[]))
            current = request['plan']['feature_set']
            if Path(directory).name == 'reference':
                self.assertEqual(current['feature_set_id'], 'old-set')
                return copy.deepcopy(reference)
            names = [f['name'] for f in current['features']]
            self.assertEqual(len(names), 16)  # Never measure a deletion-only intermediate set.
            self.assertNotIn('old0', names)
            self.assertIn('new_ratio', names)
            fits.append(request)
            output = copy.deepcopy(reference)
            output['internal_score'] += -1 if gain else 1
            for fold in output['fold_results']:
                fold['score'] += -1 if gain else 1
            if interrupt:
                atomic_json(Path(directory)/'result.json', output)
                raise KeyboardInterrupt('Stopped after the atomic result was saved')
            return output

        p.sandbox.run = run
        return p, baseline, split, path, proposals, assignments, fits

    def generate(self, proposals, assignments, legacy, unknown):
        def generate(role, instruction, payload, schema, tools=None):
            if role == 'planner':
                assignments.append(payload)
                return schema(feature_set_id=payload['feature_set']['feature_set_id'],
                    reference_feature_set_id=payload['reference_feature_set_id'], rationale='One replacement')
            proposals.append(payload)
            return schema(action='add' if legacy else 'replace', rationale='Replace the weakest transformation',
                remove=['absent' if unknown else 'old0'],
                features=[dict(name='new_ratio', op='ratio', columns=['x0', 'x1'])])
        return generate

    def test_replace_and_legacy_add_are_single_measured_interventions_at_capacity(self):
        for legacy in [False, True]:
            with self.subTest(legacy=legacy), tempfile.TemporaryDirectory() as tmp, ExitStack() as cleanup:
                p, base, split, path, calls, plans, fits = self.make_session(Path(tmp), legacy=legacy)
                cleanup.callback(p.store.close)
                winner = run_feature_session(p, base, split, path)
                self.assertIsNotNone(winner)
                session = p.store.get('feature_session', 'initial')
                self.assertEqual(session['attempts'], 1)
                self.assertEqual(len(fits), 1)
                self.assertEqual(len(plans), 1)
                self.assertEqual(plans[0]['proposal']['action'], 'replace')
                self.assertEqual(plans[0]['reference_feature_set']['feature_set_id'], 'old-set')
                evidence = p.store.list('feature_evidence')
                self.assertEqual(len(evidence), 1)
                self.assertEqual(evidence[0]['proposal']['action'], 'replace')
                self.assertEqual(evidence[0]['status'], 'accepted')
                self.assertEqual(evidence[0]['parent_feature_set_id'], 'old-set')

    def test_resume_reuses_saved_atomic_result_without_another_proposal_or_fit(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as cleanup:
            p, base, split, path, calls, plans, fits = self.make_session(Path(tmp), interrupt=True)
            cleanup.callback(p.store.close)
            with self.assertRaises(KeyboardInterrupt):
                run_feature_session(p, base, split, path)
            self.assertEqual(p.store.get('feature_session', 'initial')['attempts'], 1)
            winner = run_feature_session(p, base, split, path)
            self.assertIsNotNone(winner)
            self.assertEqual((len(calls), len(plans), len(fits)), (1, 1, 1))
            self.assertEqual(len(p.store.list('feature_evidence')), 1)

    def test_non_improving_final_replacement_preserves_the_control(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as cleanup:
            p, base, split, path, _, _, fits = self.make_session(Path(tmp), gain=False)
            cleanup.callback(p.store.close)
            self.assertIsNone(run_feature_session(p, base, split, path))
            self.assertEqual(len(fits), 1)
            self.assertEqual(p.store.get('feature_session', 'initial')['best_feature_set_id'], 'old-set')
            self.assertEqual(p.store.list('feature_evidence')[0]['status'], 'rejected')

    def test_unknown_removal_returns_a_concrete_error_without_training(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as cleanup:
            p, base, split, path, _, _, fits = self.make_session(Path(tmp), unknown=True)
            cleanup.callback(p.store.close)
            self.assertIsNone(run_feature_session(p, base, split, path))
            self.assertEqual(fits, [])
            self.assertTrue(p.store.list('feature_proposal_error'))
            self.assertTrue(all('unknown features: absent' in e['reason'] for e in p.store.list('feature_proposal_error')))

    def test_replace_needs_both_sides_and_can_reuse_a_removed_name(self):
        with self.assertRaises(ValidationError):
            FeatureProposal(action='replace', rationale='Incomplete', remove=['old'])
        value = FeatureProposal(action='replace', rationale='Same name, different expression', remove=['ratio'],
            features=[dict(name='ratio', op='ratio', columns=['x', 'z'])])
        self.assertEqual(value.action, 'replace')


if __name__ == '__main__':
    unittest.main()
