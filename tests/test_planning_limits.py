from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from pydantic import ValidationError
from mlmom.coding import CodeBundle, build_coding_graph
from mlmom.pipeline_contracts import (RunConfig, Hypothesis, HypothesisBatch, HypothesisPlan,
    PlanBatch, ValidationReport, BudgetSnapshot)
from mlmom.planning import validate_plan, build_planning_graph
from mlmom.registry import installed_tools
from mlmom.agents import LLMUnavailable
from mlmom.tracing import current_context
from tests.fakes import GENERATED_CODE as ADAPTER_CODE


class Skills:
    def load(self,*args,**kwargs): return []
    def list(self,**kwargs): return []


def case():
    context=NS(task=NS(task_type='regression',primary_metric='rmse',group_columns=[],weight_column=None,
                      target_columns=['target']),profile_ref='',feature_columns=['x'],classes=[])
    tools=installed_tools()
    for tool in tools.values(): tool.available=True
    budget=BudgetSnapshot(remaining_seconds=1000,research_seconds=900,available_cpus=4,
                          available_memory_gb=8,spent_seconds=0)
    plan=HypothesisPlan(hypothesis=Hypothesis(hypothesis_id='base',description='Small tree',rationale='Cheap',baseline=True),
        tool_id='sklearn',skill_ids=['sklearn'],max_trials=1,time_limit_seconds=60,prediction_kind='value')
    return context,tools,budget,plan


class Planner:
    def __init__(self, bad_alternative=False): self.calls=[];self.bad_alternative=bad_alternative
    def generate(self,role,instruction,payload,schema,tools=None):
        self.calls.append((role,payload))
        if role=='hypothesis_generator':
            return HypothesisBatch(hypotheses=[dict(hypothesis_id='base',description='Small tree',rationale='Cheap',baseline=True),
                dict(hypothesis_id='alt',description='Another small tree',rationale='Diversity')])
        if role=='planner':
            context,registry,budget,p=case()
            p.hypothesis=Hypothesis.model_validate(payload['hypotheses']['hypotheses'][0])
            if p.hypothesis.hypothesis_id=='alt' and self.bad_alternative: p.cpus=99
            return PlanBatch(plans=[p])
        if role=='plan_validator': return ValidationReport(valid=True)
        raise AssertionError(role)


class PlanningLimitTests(unittest.TestCase):
    def test_planner_validator_and_selected_skills_have_hypothesis_trace_context(self):
        observed=[];skill_contexts=[]
        class TracedPlanner(Planner):
            def generate(self,role,instruction,payload,schema,tools=None):
                if role in {'planner','plan_validator'}:
                    observed.append((role,current_context().get('hypothesis_id')))
                return super().generate(role,instruction,payload,schema,tools)
        class TracedSkills(Skills):
            def load(self,*args,**kwargs):
                if args[0]:skill_contexts.append(current_context().get('hypothesis_id'))
                return super().load(*args,**kwargs)
        context,tools,budget,plan=case()
        output=build_planning_graph(TracedPlanner(),context,tools,TracedSkills(),budget).invoke({'payload':{'round_number':1}})
        self.assertTrue(output['valid'])
        self.assertEqual(observed,[('planner','base'),('plan_validator','base'),('planner','alt'),('plan_validator','alt')])
        self.assertEqual(skill_contexts,['base','alt'])
        self.assertNotIn('hypothesis_id',current_context())

    def test_batch_limit_and_config_limits(self):
        self.assertEqual(RunConfig().max_candidates,2)
        for params in ({'max_candidates':3},{'coding_seconds':301},{'initial_experiment_seconds':301}):
            with self.assertRaises(ValidationError): RunConfig(**params)
        h=case()[3].hypothesis
        with self.assertRaises(ValidationError): HypothesisBatch(hypotheses=[h,h,h])
        with self.assertRaises(ValidationError): PlanBatch(plans=[case()[3]]*3)

    def test_lightweight_rules_cannot_be_bypassed_with_fast_label(self):
        context,tools,budget,plan=case()
        for change in ({'strategy':'optuna','search_space':{'n_estimators':{'type':'int','low':1,'high':2}}},
                       {'max_trials':2},{'time_limit_seconds':301},
                       {'tool_id':'torch','skill_ids':['torch'],'cost_tag':'fast'},
                       {'tool_id':'flaml','skill_ids':['flaml'],'strategy':'native','patience_trials':None},
                       {'ensemble_components':[{'tool_id':'sklearn'}],'skill_ids':['sklearn','ensembles']}):
            with self.subTest(change=change):
                changed=plan.model_copy(update=change)
                self.assertFalse(validate_plan(changed,context,tools,Skills(),budget,lightweight=True).valid)
        cat=plan.model_copy(update={'tool_id':'catboost','skill_ids':['catboost']})
        self.assertTrue(validate_plan(cat,context,tools,Skills(),budget,lightweight=True).valid)
        self.assertFalse(validate_plan(plan,context,tools,Skills(),budget,lightweight=True,initial_experiment_seconds=30).valid)

    def test_native_backend_single_loophole_closed_but_native_advanced_allowed(self):
        context,tools,budget,plan=case()
        p=plan.model_copy(update={'tool_id':'flaml','skill_ids':['flaml']})
        self.assertFalse(validate_plan(p,context,tools,Skills(),budget).valid)
        p.strategy='native';p.patience_trials=None
        self.assertTrue(validate_plan(p,context,tools,Skills(),budget).valid)

    def test_evaluation_reserve_is_protected(self):
        context,tools,budget,plan=case()
        budget.research_seconds=80
        self.assertTrue(validate_plan(plan,context,tools,Skills(),budget,evaluation_reserve_seconds=20).valid)
        self.assertFalse(validate_plan(plan,context,tools,Skills(),budget,evaluation_reserve_seconds=21).valid)

    def test_lazy_probe_only_selected_backend_and_ensemble_components(self):
        class SelectedPlanner(Planner):
            def generate(self,role,instruction,payload,schema,tools=None):
                result=super().generate(role,instruction,payload,schema,tools)
                if role=='planner':
                    result.plans[0].tool_id='torch';result.plans[0].skill_ids=['torch','ensembles','catboost']
                    result.plans[0].ensemble_components=[{'tool_id':'catboost'}]
                return result
        context,tools,budget,plan=case();called=[]
        for key in ['torch','catboost','autogluon']:
            tools[key].available=False;tools[key].reason='adapter probe required'
        def probe(key):
            called.append(key);tools[key].available=True;tools[key].reason=None
        output=build_planning_graph(SelectedPlanner(),context,tools,Skills(),budget,lightweight=False,
            planning_mode='baseline',probe_tool=probe).invoke({'payload':{'round_number':2}})
        self.assertTrue(output['valid']);self.assertEqual(called,['torch','catboost'])
        self.assertFalse(tools['autogluon'].available)

    def test_lazy_probe_does_not_run_for_forbidden_initial_or_missing_backends(self):
        for key,lightweight,reason in [('torch',True,'adapter probe required'),
                                     ('flaml',True,'adapter probe required'),
                                     ('sklearn',False,'distribution not installed'),
                                     ('sklearn',False,'version mismatch: expected 1, found 2')]:
            with self.subTest(tool=key,lightweight=lightweight,reason=reason):
                class SelectedPlanner(Planner):
                    def generate(self,role,instruction,payload,schema,tools=None):
                        result=super().generate(role,instruction,payload,schema,tools)
                        if role=='planner':
                            result.plans[0].tool_id=key;result.plans[0].skill_ids=[key]
                            if key=='flaml':result.plans[0].strategy='native';result.plans[0].patience_trials=None
                        return result
                context,tools,budget,plan=case();called=[]
                tools[key].available=False;tools[key].reason=reason
                output=build_planning_graph(SelectedPlanner(),context,tools,Skills(),budget,lightweight=lightweight,
                    planning_mode='baseline',probe_tool=called.append).invoke({'payload':{'round_number':1}})
                self.assertFalse(output['valid']);self.assertEqual(called,[])

    def test_baseline_can_publish_before_planning_bad_alternative(self):
        context,tools,budget,plan=case();client=Planner(bad_alternative=True)
        first=build_planning_graph(client,context,tools,Skills(),budget,planning_mode='baseline').invoke({'payload':{'round_number':1}})
        self.assertTrue(first['valid']);self.assertEqual([p['hypothesis']['hypothesis_id'] for p in first['plans']],['base'])
        self.assertFalse(any(c[0]=='planner' and c[1]['hypotheses']['hypotheses'][0]['hypothesis_id']=='alt' for c in client.calls))
        second=build_planning_graph(client,context,tools,Skills(),budget,planning_mode='alternatives').invoke(
            {'payload':{'round_number':1},'hypotheses':first['hypotheses']})
        self.assertFalse(second['valid']);self.assertEqual(second['plans'],[])
        self.assertEqual(sum(role=='hypothesis_generator' for role,_ in client.calls),1)
        self.assertTrue(first['valid'])

    def test_bad_alternative_does_not_discard_valid_plan(self):
        context,tools,budget,plan=case()
        out=build_planning_graph(Planner(True),context,tools,Skills(),budget).invoke({'payload':{'round_number':1}})
        self.assertTrue(out['valid']);self.assertEqual(len(out['plans']),1)
        self.assertEqual(out['errors'][0]['hypothesis_id'],'alt')

    def test_unavailable_api_for_alternative_keeps_accepted_baseline(self):
        class FailingAlternative(Planner):
            def generate(self,role,instruction,payload,schema,tools=None):
                if role=='planner' and payload['hypotheses']['hypotheses'][0]['hypothesis_id']=='alt':
                    raise LLMUnavailable('Provider unavailable')
                return super().generate(role,instruction,payload,schema,tools)
        context,tools,budget,plan=case()
        out=build_planning_graph(FailingAlternative(),context,tools,Skills(),budget).invoke({'payload':{'round_number':1}})
        self.assertTrue(out['valid']);self.assertEqual(len(out['plans']),1)
        self.assertEqual(out['errors'][0]['hypothesis_id'],'alt')


class Store:
    def __init__(self,root): self.root=root;self.data={};self.events=[]
    def get(self,kind,key): return self.data.get((kind,key))
    def put(self,kind,key,value,**kwargs): self.data[kind,key]=value
    def event(self,*args): self.events.append(args)


class ClockBudget:
    cpus=4;memory_gb=8
    def __init__(self): self.elapsed=0;self.extra_reserve=0
    def now(self): return 1000+self.elapsed
    def snapshot(self):
        return BudgetSnapshot(remaining_seconds=1000-self.elapsed,research_seconds=max(0,900-self.elapsed-self.extra_reserve),
                              available_cpus=4,available_memory_gb=8,spent_seconds=self.elapsed)


class Sandbox:
    def __init__(self,budget): self.budget=budget;self.calls=[]
    def run(self,request,path,**limits):
        self.calls.append(limits);self.budget.elapsed+=limits['seconds']
        return {'status':'completed'}


class Coder:
    deadline=100000000000
    def __init__(self,budget,seconds): self.budget=budget;self.seconds=seconds;self.calls=0
    def generate(self,*args):
        if args[0]=='preprocessing_agent':
            from tests.fakes import PREPROCESSING_CODE
            return args[3](source=PREPROCESSING_CODE,spec=dict(description='Test transforms',output='dense'))
        self.calls+=1;self.budget.elapsed+=self.seconds
        return CodeBundle(files={'candidate.py':ADAPTER_CODE})


class CodingBudgetTests(unittest.TestCase):
    def test_runtime_repairs_share_original_deadline_and_smoke_is_clamped(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);store=Store(root);budget=ClockBudget();sandbox=Sandbox(budget)
            context,tools,_,plan=case();client=Coder(budget,250);original=client.deadline
            kwargs=dict(workspace=root/'workspace',store=store,plan=plan,plan_ref=root/'plan.json',skill_data=[],
                        tools=tools,context=context,sandbox=sandbox,budget=budget,config=RunConfig(),round_id='r1')
            first=build_coding_graph(client,**kwargs).invoke({'payload':{}})
            self.assertTrue(first['valid']);self.assertEqual(budget.elapsed,280)
            deadline=store.get('coding_budget','r1-base')['deadline']
            client.seconds=10
            second=build_coding_graph(client,**kwargs,runtime_feedback='Runtime failure').invoke({'payload':{}})
            self.assertTrue(second['valid']);self.assertEqual(sandbox.calls[-1]['seconds'],20)
            self.assertEqual(budget.elapsed,300);self.assertEqual(client.deadline,original)
            third=build_coding_graph(client,**kwargs,runtime_feedback='Another failure').invoke({'payload':{}})
            self.assertFalse(third['valid']);self.assertEqual(client.calls,1)
            self.assertEqual(store.get('coding_budget','r1-base')['deadline'],deadline)
            self.assertEqual(len(sandbox.calls),2)

    def test_research_deadline_is_tighter_than_coding_window(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);store=Store(root);budget=ClockBudget();budget.elapsed=875
            sandbox=Sandbox(budget);context,tools,_,plan=case();plan.time_limit_seconds=5;client=Coder(budget,5)
            result=build_coding_graph(client,workspace=root/'workspace',store=store,plan=plan,plan_ref=root/'plan.json',
                skill_data=[],tools=tools,context=context,sandbox=sandbox,budget=budget,
                config=RunConfig(prediction_seconds=5),round_id='r1').invoke({'payload':{}})
            self.assertTrue(result['valid']);self.assertEqual(sandbox.calls[-1]['seconds'],5)
            self.assertEqual(store.get('coding_budget','r1-base')['deadline'],1885)

    def test_increased_final_reserve_protects_fit_and_prediction_during_repair(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);store=Store(root);budget=ClockBudget();sandbox=Sandbox(budget)
            context,tools,_,plan=case();client=Coder(budget,250)
            kwargs=dict(workspace=root/'workspace',store=store,plan=plan,plan_ref=root/'plan.json',skill_data=[],
                        tools=tools,context=context,sandbox=sandbox,budget=budget,config=RunConfig(),round_id='r1')
            first=build_coding_graph(client,**kwargs).invoke({'payload':{}})
            self.assertTrue(first['valid']);budget.extra_reserve=400
            second=build_coding_graph(client,**kwargs,runtime_feedback='Repair').invoke({'payload':{}})
            self.assertFalse(second['valid']);self.assertEqual(client.calls,1);self.assertEqual(len(sandbox.calls),1)


if __name__=='__main__': unittest.main()
