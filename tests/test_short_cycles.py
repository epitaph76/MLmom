from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from mlmom.pipeline import Pipeline
from mlmom.pipeline_contracts import RunConfig
from mlmom.coding import build_coding_graph
from mlmom.planning import build_planning_graph
from tests.test_planning_limits import case, Store, ClockBudget, Sandbox, Coder, Skills, Planner


class ShortCycleTests(unittest.TestCase):
    def test_actual_remaining_time_is_not_divided_by_estimates_or_target_cycles(self):
        context,tools,_,plan=case();plan.time_limit_seconds=300
        budget=ClockBudget();budget.elapsed=780
        config=RunConfig(timeouts_enabled=True,total_seconds=180,final_reserve_seconds=20,coding_seconds=60,
            prediction_seconds=5,target_completed_rounds=3)
        pipeline=object.__new__(Pipeline);pipeline.config=config;pipeline.budget=budget
        pipeline.store=SimpleNamespace(list=lambda *a,**kw:[])
        allocation=pipeline.allocated_fit_seconds(plan,1)
        self.assertEqual(allocation,120);plan.time_limit_seconds=allocation
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);store=Store(root);sandbox=Sandbox(budget);client=Coder(budget,20)
            result=build_coding_graph(client,workspace=root/'work',store=store,plan=plan,plan_ref=root/'plan.json',
                skill_data=[],tools=tools,context=context,sandbox=sandbox,budget=budget,config=config,round_id='r1').invoke({'payload':{}})
            self.assertTrue(result['valid'],result);self.assertEqual(client.calls,1)
            window=store.get('coding_budget','r1-base')
            self.assertEqual(window['deadline']-window['started'],60)

    def test_direct_first_custom_code_and_temporary_tool_limit(self):
        context,tools,_,plan=case();plan.execution_mode='custom';budget=ClockBudget()
        class Direct(Coder):
            max_steps=12
            def generate(self,*args):
                if args[0]=='preprocessing_agent':
                    assert {'read_skills','inspect_api','check_preprocessing'} <= set(args[-1].tools)
                else:
                    assert args[-1] is None
                assert self.max_steps==3
                return super().generate(*args)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);client=Direct(budget,5)
            result=build_coding_graph(client,workspace=root/'work',store=Store(root),plan=plan,plan_ref=root/'plan.json',
                skill_data=[],tools=tools,context=context,sandbox=Sandbox(budget),budget=budget,config=RunConfig(timeouts_enabled=True),round_id='r2').invoke({'payload':{}})
            self.assertTrue(result['valid']);self.assertEqual(client.max_steps,12)

    def test_validator_is_told_min_delta_is_not_hpo_continuation(self):
        class Review(Planner):
            def generate(self,role,instruction,*args,**kwargs):
                if role=='plan_validator':
                    assert 'NOT the batched HPO continuation threshold' in instruction
                    assert 'min_delta=0 is correct' in instruction
                return super().generate(role,instruction,*args,**kwargs)
        context,tools,budget,_=case()
        result=build_planning_graph(Review(),context,tools,Skills(),budget,planning_mode='baseline').invoke({'payload':{'round_number':2}})
        self.assertTrue(result['valid'])


if __name__=='__main__':unittest.main()
