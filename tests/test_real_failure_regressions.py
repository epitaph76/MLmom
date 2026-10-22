"""Reproduce the protocol and control-flow failures seen with real Qwen."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from openai import OpenAI

from mlmom.agents import RoleClient, LLMUnavailable, StructuredOutputError
from mlmom.coding import CodeBundle, build_coding_graph
from mlmom.llm_client import LLMSettings
from mlmom.persistence import ExperimentStore
from mlmom.pipeline_contracts import ValidationReport, RunConfig
from mlmom.planning import build_planning_graph
from mlmom.tools import ToolSet, Workspace
from mlmom.tracing import TraceSink
from tests.test_planning_limits import case, Skills, Planner, ClockBudget, Sandbox, Store


def response(content='{}', *, calls=None, finish='stop', **extensions):
    message=dict(role='assistant',content=content,**extensions)
    if calls: message['tool_calls']=calls
    return httpx.Response(200,json=dict(id='test',object='chat.completion',created=1,model='test',
        choices=[dict(index=0,finish_reason=finish,message=message)],
        usage=dict(prompt_tokens=1,completion_tokens=1,total_tokens=2)))


def tool_call(index, name='read_history', arguments='{"kind":"evaluation","limit":5}'):
    return dict(id=f'call-{index}',type='function',index=index,
                function=dict(name=name,arguments=arguments))


class RealFailureRegressions(unittest.TestCase):
    def client(self, handler, **kwargs):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        sink=TraceSink(Path(temp.name),'failure-regression');self.addCleanup(sink.close)
        client=RoleClient(LLMSettings(api_key='unit-test-only'),deadline=time.time()+60,trace=sink,**kwargs)
        client.client.close()
        client.client=OpenAI(api_key='unit-test-only',max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        self.addCleanup(client.client.close)
        return client,Path(temp.name)

    def test_repeated_empty_history_forces_one_final_json_and_cleans_messages(self):
        requests=[]
        def handler(request):
            requests.append(json.loads(request.content))
            if len(requests)<3:
                return response(None,calls=[tool_call(len(requests))],reasoning_content='private reasoning',
                                provider_specific_fields={'reasoning':'response only'})
            return response('{"valid":true}')
        client,root=self.client(handler)
        tools=ToolSet().register('read_history',lambda kind,limit:[], 'Read previous results.')
        client.planning_direct_first=False  # Exercise the optional legacy tool-call path.
        result=client.generate('planner','test',{},ValidationReport,tools)
        self.assertTrue(result.valid);self.assertEqual(len(requests),3)
        self.assertNotIn('tools',requests[-1])
        for request in requests:
            for m in request['messages']:
                self.assertNotIn('reasoning_content',m);self.assertNotIn('provider_specific_fields',m)
                for c in m.get('tool_calls',[]): self.assertNotIn('index',c)
        history=requests[-1]['messages']
        self.assertEqual([m['tool_call_id'] for m in history if m['role']=='tool'],['call-1','call-2'])
        self.assertTrue(any('private reasoning' in p.read_text(encoding='utf-8') for p in (root/'trace_payloads').glob('*')))

    def test_validator_repairs_length_null_and_invalid_json_without_tools(self):
        for first in (lambda:response(None,finish='length'),lambda:response('not JSON')):
            calls=[]
            def handler(request):
                calls.append(json.loads(request.content))
                return first() if len(calls)==1 else response('{"valid":true}')
            client,_=self.client(handler)
            payload={'plans':[{'tool_id':'catboost','model_params':{'iterations':20}}]}
            self.assertTrue(client.generate('plan_validator','Review',payload,ValidationReport).valid)
            self.assertEqual(len(calls),2)
            self.assertEqual(calls[0]['messages'][1],calls[1]['messages'][1])
            self.assertTrue(all('tools' not in c for c in calls))

    def test_502_retries_twice_and_permanent_400_does_not(self):
        seen=[]
        def handler(request):
            seen.append(request)
            return httpx.Response(502,json={'error':{'message':'upstream error'}}) if len(seen)<3 else response('{"valid":true}')
        client,_=self.client(handler)
        with patch('mlmom.agents.time.sleep') as sleep:
            self.assertTrue(client.generate('planner','test',{},ValidationReport).valid)
        self.assertEqual([c.args[0] for c in sleep.call_args_list],[1,2])
        self.assertEqual(len(seen),3)

    def test_actual_corrupted_validator_key_uses_fresh_context_without_approval_guess(self):
        malformed=json.dumps({')}{':'valid','errors':[], 'warnings':['Preserve original-scale RMSE']})
        requests=[]
        def handler(request):
            body=json.loads(request.content);requests.append(body)
            if len(requests)==1:return response(malformed)
            self.assertEqual(len(body['messages']),2)
            self.assertNotIn(malformed,json.dumps(body['messages']))
            self.assertEqual(body['messages'][1],requests[0]['messages'][1])
            self.assertIn('JSON boolean',body['messages'][0]['content'])
            return response('{"valid":false,"errors":["Wrong stopping metric"],"warnings":[]}')
        client,_=self.client(handler)
        result=client.generate('plan_validator','Review',{'plans':[{'model_params':{'n_estimators':100}}]},ValidationReport)
        self.assertFalse(result.valid)
        self.assertEqual(result.errors,['Wrong stopping metric'])
        self.assertEqual(len(requests),2)

    def test_role_deadline_persists_across_repair_and_client_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            store=ExperimentStore(Path(temp));self.addCleanup(store.close)
            client,_=self.client(lambda request:response(),budget_store=store)
            with patch('mlmom.agents.time.time',return_value=1000):
                with client.budget_scope('planner','r1-base',180): self.assertEqual(client.deadline,1180)
            with patch('mlmom.agents.time.time',return_value=1100):
                with client.budget_scope('planner','r1-base',180): self.assertEqual(client.deadline,1180)
            resumed,_=self.client(lambda request:response(),budget_store=store)
            with patch('mlmom.agents.time.time',return_value=1200):
                with resumed.budget_scope('planner','r1-base',180):
                    with self.assertRaises(LLMUnavailable): resumed.complete([{'role':'user','content':'{}'}])
            store.close()

    def test_generated_source_rejects_adapter_import_but_allows_own_preprocessing(self):
        from mlmom.tools import verify_source
        from tests.fakes import ADAPTER_CODE,GENERATED_CODE
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'candidate.py'
            source.write_text(ADAPTER_CODE,encoding='utf-8')
            self.assertFalse(verify_source(root).valid)
            source.write_text(GENERATED_CODE,encoding='utf-8')
            self.assertTrue(verify_source(root).valid)
            (root/'extra.py').write_text('x=1',encoding='utf-8')
            self.assertTrue(verify_source(root).valid)

    def test_workspace_keeps_every_write_and_patch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);workspace=Workspace(root/'work',revision_dir=root/'revisions')
            workspace.write_file('candidate.py','x=1\n')
            workspace.apply_patch('candidate.py','x=1','x=2')
            values=[p.read_text(encoding='utf-8') for p in (root/'revisions').glob('*/candidate.py')]
            self.assertCountEqual(values,['x=1\n','x=2\n'])

    def test_timeout_without_candidate_preserves_failure_and_smoke_reserve(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);store=Store(root);budget=ClockBudget();sandbox=Sandbox(budget)
            context,tools,_,plan=case()
            class TimedOut:
                deadline=time.time()+1000
                def generate(self,*args):
                    self.received_deadline=self.deadline
                    budget.elapsed+=240
                    raise LLMUnavailable('APITimeoutError')
            client=TimedOut();started=time.time()
            output=build_coding_graph(client,workspace=root/'work',store=store,plan=plan,plan_ref=root/'plan.json',
                skill_data=[],tools=tools,context=context,sandbox=sandbox,budget=budget,
                config=RunConfig(timeouts_enabled=True),round_id='r1').invoke({'payload':{}})
            self.assertFalse(output['valid']);self.assertEqual(sandbox.calls,[])
            self.assertIn('Missing candidate.py',output['errors']);self.assertIn('APITimeoutError',output['errors'])
            self.assertLessEqual(client.received_deadline-started,241)
            self.assertTrue(list((root/'candidates').rglob('metadata.json')))

    def test_legacy_hypothesis_reference_resolves_only_verified_candidates(self):
        class Legacy(Planner):
            def generate(self,role,*args,**kwargs):
                result=super().generate(role,*args,**kwargs)
                if role=='planner': result.plans[0].hypothesis.based_on='base'
                return result
        for known,mapping,expected in [(set(),{},None),({'r1-base-v2'},{'r1-base-v2':'base'},'r1-base-v2')]:
            context,tools,budget,plan=case()
            output=build_planning_graph(Legacy(),context,tools,Skills(),budget,planning_mode='baseline',
                known_candidates=known,candidate_hypotheses=mapping).invoke({'payload':{'round_number':1}})
            self.assertTrue(output['valid'])
            hypothesis=output['plans'][0]['hypothesis']
            self.assertEqual(hypothesis['based_on'],expected);self.assertEqual(hypothesis['based_on_hypothesis_id'],'base')

    def test_validator_format_failure_never_reinvokes_planner(self):
        class InvalidValidator(Planner):
            def generate(self,role,*args,**kwargs):
                if role=='plan_validator': raise StructuredOutputError('Invalid JSON after format repair')
                return super().generate(role,*args,**kwargs)
        context,tools,budget,plan=case();client=InvalidValidator()
        output=build_planning_graph(client,context,tools,Skills(),budget,planning_mode='baseline').invoke({'payload':{'round_number':1}})
        self.assertFalse(output['valid'])
        self.assertEqual(sum(role=='planner' for role,_ in client.calls),1)

    def test_validator_timeout_never_bypasses_review_for_generated_plans(self):
        class UnavailableValidator(Planner):
            def generate(self,role,*args,**kwargs):
                if role=='plan_validator': raise LLMUnavailable('APITimeoutError')
                return super().generate(role,*args,**kwargs)
        for lightweight,expected in [(True,False),(False,False)]:
            context,tools,budget,plan=case()
            output=build_planning_graph(UnavailableValidator(),context,tools,Skills(),budget,
                lightweight=lightweight,planning_mode='baseline').invoke({'payload':{'round_number':1}})
            self.assertEqual(output['valid'],expected)
            self.assertEqual(bool(output['warnings']),expected)

    def test_semantic_rejection_never_uses_timeout_fallback(self):
        class RejectingValidator(Planner):
            def generate(self,role,*args,**kwargs):
                if role=='plan_validator': return ValidationReport(valid=False,errors=['Unsafe data use'])
                return super().generate(role,*args,**kwargs)
        context,tools,budget,plan=case()
        output=build_planning_graph(RejectingValidator(),context,tools,Skills(),budget,
            max_attempts=1,planning_mode='baseline').invoke({'payload':{'round_number':1}})
        self.assertFalse(output['valid']);self.assertFalse(output['warnings'])

    def test_matching_cpu_parameter_is_normalized_but_override_is_rejected(self):
        for cpus,expected in [(2,True),(-1,False)]:
            class CpuPlanner(Planner):
                def generate(self,role,*args,**kwargs):
                    result=super().generate(role,*args,**kwargs)
                    if role=='planner': result.plans[0].model_params['n_jobs']=cpus
                    return result
            context,tools,budget,plan=case()
            output=build_planning_graph(CpuPlanner(),context,tools,Skills(),budget,
                max_attempts=1,planning_mode='baseline').invoke({'payload':{'round_number':1}})
            self.assertEqual(output['valid'],expected)
            if expected:self.assertNotIn('n_jobs',output['plans'][0]['model_params'])


if __name__=='__main__': unittest.main()
