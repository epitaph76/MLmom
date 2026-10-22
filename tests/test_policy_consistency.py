"""Policy is enforced beyond prompt wording, and docs derive from the registry."""
import tempfile
import unittest
from pathlib import Path

from mlmom.planning import validate_plan
from mlmom.tools import verify_source
from mlmom.worker import train
from tests.test_planning_limits import case,Skills
from tests.test_execution import execution_case
from tests.fakes import MODEL_CODE


class PolicyConsistencyTests(unittest.TestCase):
    def test_ensemble_plan_is_rejected_even_after_initial_cycle(self):
        context,tools,budget,plan=case()
        plan.skill_ids.append('ensembles')
        plan.ensemble_components=[dict(tool_id='sklearn')]
        self.assertFalse(validate_plan(plan,context,tools,Skills(),budget,lightweight=False).valid)
        with tempfile.TemporaryDirectory() as tmp:
            _,context,plan,_=execution_case(Path(tmp),'regression')
            plan.ensemble_components=[dict(tool_id='sklearn')]
            with self.assertRaisesRegex(ValueError,'disabled'):
                train(dict(context=context.model_dump(mode='json'),plan=plan.model_dump(mode='json')),Path(tmp))

    def test_generated_stacking_and_native_blending_rejected_but_extratrees_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'candidate.py'
            for extra in ['from sklearn.ensemble import StackingRegressor as Hidden',
                          'from mlmom.ensemble import EnsembleCandidate',
                          'def fit_native(a): a.fit(ensemble=True)',
                          'def fit_native(a): a.fit(num_stack_levels=1)']:
                path.write_text(MODEL_CODE+'\n'+extra,encoding='utf-8')
                self.assertFalse(verify_source(Path(tmp)).valid)
            path.write_text(MODEL_CODE,encoding='utf-8')
            self.assertTrue(verify_source(Path(tmp)).valid)

    def test_registry_owned_documentation_is_current(self):
        from scripts.check_skill_docs import check
        check()
        from mlmom.registry import SkillRegistry
        registry=SkillRegistry(Path(__file__).resolve().parents[1]/'skills')
        self.assertNotIn('ensembles',{s['skill_id'] for s in registry.list(role='coding_agent')})
        with self.assertRaisesRegex(ValueError,'not available'):
            registry.load(['ensembles'],'regression',role='coding_agent')


if __name__=='__main__':unittest.main()
