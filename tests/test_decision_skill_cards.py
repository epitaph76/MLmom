"""Decision-role skill wiring and good/error scenario behavior."""
from __future__ import annotations

import unittest
from pathlib import Path

from mlmom.pipeline_contracts import Hypothesis, HypothesisBatch, PlanBatch, ValidationReport
from mlmom.planning import build_planning_graph
from mlmom.registry import SkillRegistry
from mlmom.research import review_result
from tests.test_planning_limits import case


ROOT = Path(__file__).resolve().parents[1]


class RepairingPlanningClient:
    def __init__(self):
        self.planner_calls = 0
        self.seen_cards = {}

    def generate(self, role, instruction, payload, schema, tools=None):
        cards = payload.get('skill_instructions', [])
        self.seen_cards[role] = [card['skill_id'] for card in cards]
        if role == 'hypothesis_generator':
            return HypothesisBatch(hypotheses=[dict(
                hypothesis_id='fixed_tree_baseline',
                description='Fit one fixed small tree baseline',
                rationale=('Evidence: no prior experiment; Mechanism: establish a nonlinear reference; '
                           'Expected effect: measurable RMSE baseline; Cost: low; Risk: underfit; '
                           'Verify: record RMSE on the runner supplied split'),
                baseline=True,
            )])
        if role == 'planner':
            self.planner_calls += 1
            self.assert_feedback_after_rejection(payload)
            _, _, _, plan = case()
            plan.hypothesis = Hypothesis.model_validate(payload['hypotheses']['hypotheses'][0])
            plan.cpus = 99 if self.planner_calls == 1 else 2
            return PlanBatch(plans=[plan])
        if role == 'plan_validator':
            return ValidationReport(valid=True)
        raise AssertionError(role)

    def assert_feedback_after_rejection(self, payload):
        if self.planner_calls > 1:
            assert payload['feedback']
            assert 'worker capacity' in payload['feedback'][0]


class ResultClient:
    def __init__(self, evidence_ref='evaluation:candidate'):
        self.evidence_ref = evidence_ref
        self.seen_skill = False

    def generate(self, role, instruction, payload, schema, tools=None):
        self.seen_skill = [card['skill_id'] for card in payload['skills']] == ['result_validator']
        return schema(implementation='matches', effect='supported',
                      evidence_refs=[self.evidence_ref],
                      explanation='Implementation matches and the same-split score improves.')


class DecisionSkillCardTests(unittest.TestCase):
    def setUp(self):
        self.registry = SkillRegistry(ROOT / 'skills')

    def test_bad_resource_plan_is_repaired_with_role_cards_loaded(self):
        context, tools, budget, _ = case()
        client = RepairingPlanningClient()
        output = build_planning_graph(client, context, tools, self.registry, budget,
                                      max_candidates=1).invoke({'payload': {'round_number': 1}})
        self.assertTrue(output['valid'], output)
        self.assertEqual(client.planner_calls, 2)
        self.assertEqual(output['plans'][0]['cpus'], 2)
        self.assertEqual(client.seen_cards['hypothesis_generator'], ['hypothesis_generator'])
        self.assertEqual(client.seen_cards['planner'], ['planner'])

    def test_result_support_requires_known_evidence_and_comparable_reference(self):
        skills = self.registry.load([], 'regression', role='result_validator')
        evaluation=dict(score_kind='outer_holdout',dataset_hash='data',split_id='same-split',score=.9,
                        metric='rmse',metric_options={},direction='minimize',weighting='unweighted')
        payload = dict(evidence_refs=['evaluation:candidate'],execution_report={'status':'passed'},evaluation=evaluation,
                       comparable_reference=dict(evaluation,score=1.0))
        good = ResultClient()
        supported = review_result(good, payload, skills, 'good', 1)
        self.assertTrue(good.seen_skill)
        self.assertEqual((supported.implementation, supported.effect), ('matches', 'supported'))

        no_reference = review_result(ResultClient(), payload | {'comparable_reference': None},
                                     skills, 'no-reference', 1)
        self.assertEqual(no_reference.effect, 'inconclusive')

        unknown_evidence = review_result(ResultClient('evaluation:unknown'), payload,
                                         skills, 'bad-evidence', 1)
        self.assertEqual((unknown_evidence.implementation, unknown_evidence.effect),
                         ('unknown', 'inconclusive'))


if __name__ == '__main__':
    unittest.main()
