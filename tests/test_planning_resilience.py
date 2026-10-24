"""Historical planner failures recover without weakening executable contracts."""
import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError
from mlmom.agents import RoleClient
from mlmom.coding import reuse_assignment
from mlmom.feature_contracts import FeatureExperimentPlan, FeatureProposal
from mlmom.llm_client import LLMSettings
from mlmom.persistence import ExperimentStore
from mlmom.pipeline import Pipeline
from mlmom.pipeline_contracts import (Hypothesis, HypothesisBatch, HypothesisPlan,
    PlanBatch, PlannerContextDraft, RunConfig, ValidationReport)
from mlmom.planning import build_planning_graph, validate_plan
from mlmom.planning_retries import PlanningRejections
from tests.test_planning_limits import Skills, case


class PlanClient:
    def __init__(self, plan):
        self.plan = plan
        self.calls = []

    def generate(self, role, instruction, payload, schema, tools=None):
        self.calls.append((role, payload))
        if role == 'hypothesis_generator':
            return HypothesisBatch(hypotheses=[self.plan.hypothesis])
        if role == 'planner':
            return PlanBatch(plans=[self.plan.model_copy(deep=True)])
        if role == 'plan_validator':
            return ValidationReport(valid=True)
        raise AssertionError(role)


class NarrativeTests(unittest.TestCase):
    def test_plan_explanations_and_unambiguous_shapes_need_one_request(self):
        raw = case()[3].model_dump()
        raw.update(change_kind='single_configuration', rationale='Extra useful explanation',
                   preprocessing='Fit medians on the current training partition.',
                   intervention='Keep the preprocessing fold-local.')
        client = RoleClient(LLMSettings(api_key='offline-test'), deadline=time.time()+60)
        self.addCleanup(client.client.close)
        with patch.object(client, '_request', return_value=dict(role='assistant', content=json.dumps(raw))) as request:
            result = client.generate('planner', 'Plan', {}, PlanBatch)
        request.assert_called_once()
        plan = result.plans[0]
        self.assertEqual(plan.change_kind, 'experiment')
        self.assertEqual(plan.agent_notes['rationale'], raw['rationale'])
        self.assertEqual(plan.agent_notes['original_change_kind'], 'single_configuration')
        self.assertEqual(plan.preprocessing, [raw['preprocessing']])
        self.assertEqual(plan.intervention.operations, [raw['intervention']])
        self.assertEqual(HypothesisPlan.model_validate(plan.model_dump()).agent_notes, plan.agent_notes)

    def test_feature_planner_extra_prose_and_length_do_not_erase_assignment(self):
        text = 'x' * 5001
        plan = FeatureExperimentPlan.model_validate(dict(feature_set_id='new', reference_feature_set_id='old',
            rationale=text, success_conditions=['Improve paired CV'], rejection_conditions=['No gain'],
            intervention='Add the requested feature', change_kind='single_change_ablation'))
        self.assertEqual(plan.feature_set_id, 'new')
        self.assertEqual(len(plan.rationale), 5000)
        self.assertEqual(plan.agent_notes['rationale_full'], text)
        self.assertEqual(plan.agent_notes['success_conditions'], ['Improve paired CV'])
        self.assertIn('intervention', plan.agent_notes)
        with self.assertRaises(ValidationError):
            FeatureProposal(action='add', rationale='A proposal', features=[])


    def test_execution_errors_are_still_rejected(self):
        raw = case()[3].model_dump()
        for update in [dict(cpus=0), dict(cpus=True), dict(memory_gb=-1), dict(memory_gb=float('inf')),
                       dict(strategy='whatever'), dict(change_kind='invented')]:
            with self.subTest(update=update), self.assertRaises(ValidationError):
                HypothesisPlan.model_validate(dict(raw, **update))




class RetryTests(unittest.TestCase):
    def test_renaming_an_identical_hypothesis_does_not_reset_its_rejections(self):
        hypothesis = case()[3].hypothesis
        ledger = PlanningRejections({}, lambda: None, 3)
        self.assertFalse(ledger.reject(hypothesis, 'Same invalid setting'))
        self.assertTrue(ledger.reject(hypothesis, 'Same invalid setting'))
        renamed = hypothesis.model_copy(update=dict(hypothesis_id='new-name', rationale='Different prose'))
        self.assertTrue(ledger.blocked(renamed))

    def test_failed_family_member_does_not_blacklist_other_backends(self):
        class FamilyClient(PlanClient):
            def generate(self, role, instruction, payload, schema, tools=None):
                if role == 'planner':
                    self.calls.append((role, payload))
                    plan = case()[3]
                    plan.hypothesis = Hypothesis.model_validate(payload['hypotheses']['hypotheses'][0])
                    plan.tool_id = payload['required_tool_id']
                    plan.skill_ids = [plan.tool_id]
                    if plan.tool_id == 'lightgbm':plan.cpus = 99
                    return PlanBatch(plans=[plan])
                return super().generate(role, instruction, payload, schema, tools)
        context, tools, budget, plan = case()
        plan.hypothesis.kind, plan.hypothesis.model_family = 'family', 'gradient_boosting'
        output = build_planning_graph(FamilyClient(plan), context, tools, Skills(), budget,
            lightweight=False).invoke({'payload': {'round_number': 2}})
        self.assertTrue(output['valid'], output)
        self.assertEqual({p['tool_id'] for p in output['plans']}, {'catboost', 'xgboost'})

    def pipeline(self, store, client):
        context, tools, budget, _ = case()
        p = object.__new__(Pipeline)
        p.config = RunConfig(ensemble_search_enabled=False)
        p.store = store
        p.budget = SimpleNamespace(snapshot=lambda: budget)
        p.last_submission = {'path': 'already-verified.csv', 'model_role': 'submission_refit'}
        def planning_call(state, **kwargs):
            progress = p.store.get('planning_progress', 'r2-all') or {}
            graph = build_planning_graph(client, context, tools, Skills(), budget, lightweight=False,
                saved_progress=progress, save_progress=lambda value: p.store.put('planning_progress', 'r2-all', value))
            initial = {'payload': dict(round_number=2)}
            if progress.get('hypotheses'):
                initial['hypotheses'] = progress['hypotheses']
            return graph.invoke(initial)
        p.planning_call = planning_call
        return p

    def test_repeated_rejection_survives_resume_and_next_attempt_uses_new_idea(self):
        class Recovering(PlanClient):
            generations = 0
            def generate(self, role, instruction, payload, schema, tools=None):
                if role == 'hypothesis_generator':
                    self.generations += 1
                    self.plan.hypothesis = Hypothesis(hypothesis_id=f'idea{self.generations}',
                        description='Impossible request' if self.generations == 1 else 'A different feasible model', rationale='Evidence')
                    self.plan.cpus = 99 if self.generations == 1 else 2
                return super().generate(role, instruction, payload, schema, tools)
        with tempfile.TemporaryDirectory() as temp:
            client = Recovering(case()[3])
            store = ExperimentStore(Path(temp))
            try:
                p = self.pipeline(store, client)
                state = dict(round_number=2)
                state.update(p.plan(state))
                self.assertEqual(state['stop_reason'], 'planning_retry')
                saved = store.get('planning_progress', 'r2-all')
                self.assertIsNone(saved['hypotheses'])
                self.assertEqual(next(iter(saved['rejections'].values()))['attempts'], 2)
                store.close()
                store = ExperimentStore(Path(temp))
                p = self.pipeline(store, client)
                result = p.plan(state)
                self.assertTrue(result['plans'], result)
                self.assertEqual(result['plans'][0]['hypothesis']['hypothesis_id'], 'idea2')
                self.assertEqual(p.last_submission['path'], 'already-verified.csv')
                self.assertEqual(sum(role == 'planner' for role, _ in client.calls), 3)
                self.assertTrue(any(item.get('blocked') for item in client.calls[-3][1]['rejected_hypotheses']))
            finally:
                store.close()

    def test_continuous_mode_cannot_repeat_invalid_new_batches_forever(self):
        class AlwaysBad(PlanClient):
            generations = 0
            def generate(self, role, instruction, payload, schema, tools=None):
                if role == 'hypothesis_generator':
                    self.generations += 1
                    self.plan.hypothesis = Hypothesis(hypothesis_id=f'bad{self.generations}',
                        description=f'Unusable approach {self.generations}', rationale='Try')
                    self.plan.cpus = 99
                return super().generate(role, instruction, payload, schema, tools)
        with tempfile.TemporaryDirectory() as temp:
            store = ExperimentStore(Path(temp))
            try:
                client = AlwaysBad(case()[3])
                p = self.pipeline(store, client)
                state = dict(round_number=2)
                for _ in range(3):
                    state.update(p.plan(state))
                self.assertEqual(state['stop_reason'], 'planning_attempts_exhausted')
                self.assertEqual(client.generations, 3)
                self.assertEqual(sum(role == 'planner' for role, _ in client.calls), 6)
                self.assertEqual(p.last_submission['path'], 'already-verified.csv')
            finally:
                store.close()


if __name__ == '__main__':
    unittest.main()
