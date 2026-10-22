"""Task meaning, evidence ranking, strict requirements, recovery and planner wiring."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mlmom.agents import RoleClient
from mlmom.context_agent import prepare_planner_context
from mlmom.contracts import DataIssue, TaskDefinition
from mlmom.data_inspector import DataInspector
from mlmom.inputs import resolve_inputs
from mlmom.llm_client import LLMSettings
from mlmom.persistence import ExperimentStore, atomic_json, file_hash
from mlmom.pipeline import Pipeline
from mlmom.pipeline_contracts import RunConfig, TaskContext, HypothesisBatch
from mlmom.registry import SkillRegistry
from mlmom.tools import ToolSet
from tests.fakes import OfflineLLM
from tests.test_pipeline import dataset, only_sklearn


class ContextAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.path=dataset(self.root/'input')
        self.profile=DataInspector().inspect_directory(self.path.parent)
        self.store=ExperimentStore(self.root/'run');self.addCleanup(self.store.close)
        profile_path=self.store.root/'profile.json';atomic_json(profile_path,self.profile)
        task=TaskDefinition.model_validate_json(OfflineLLM().complete([{'content':json.dumps(
            {'dataset_inspection':{'task':self.profile.task.model_dump(mode='json')}})}]))
        self.context=TaskContext(task_id='task',task=task,profile_ref=str(profile_path),files=resolve_inputs(self.path),
            feature_columns=['x','z','v'],dataset_hash=file_hash(Path(resolve_inputs(self.path).train)),
            metric_options={'multioutput':'uniform_average'},submission_kind='value')
        self.clock=[time.time()]
        self.budget=SimpleNamespace(now=lambda:self.clock[0],snapshot=lambda:SimpleNamespace(research_seconds=1000))
        self.config=RunConfig(timeouts_enabled=True)
        self.reports={'initial':dict(status='completed',observations=[dict(observation_id='tail',topic='target_tail',
            description='Heavy upper tail',rows=180,numbers={'q99':12.5},source_ref='csv:measured')],proposals=[])}

    def prepare(self,client=None,**kwargs):
        return prepare_planner_context(client=client or OfflineLLM(),context=self.context,profile=self.profile,
            reports=self.reports,store=self.store,budget=self.budget,config=self.config,round_number=1,skills=[],**kwargs)

    def test_agent_receives_complete_task_and_preserves_semantics_with_quotes(self):
        text=self.path.read_text(encoding='utf-8')+'\n'+('domain notes '*1800)+'\nWeather is already known at prediction time.'
        self.path.write_text(text,encoding='utf-8')
        parent=self
        class Client(OfflineLLM):
            def generate(self,role,instruction,payload,schema,tools=None):
                parent.assertEqual(role,'context_agent');parent.assertEqual(payload['task_description'],text)
                answer=super().generate(role,instruction,payload,schema,tools)
                return schema.model_validate(dict(answer.model_dump(),**{'task_summary':'Predict the requested target using known information.',
                    'task_details':[dict(meaning='Use contemporaneous weather, not a forecast of it.',
                        source_quote='Weather is already known at prediction time.')]}))
        result=self.prepare(Client())
        self.assertEqual(result['status'],'completed',result)
        self.assertEqual(result['requirements']['task'],self.context.task.model_dump(mode='json'))
        self.assertEqual(result['requirements']['metric_options'],self.context.metric_options)
        fact=next(r['fact'] for r in result['ranked_findings'] if r['fact']['kind']=='observation')
        self.assertEqual(fact['evidence']['numbers'],{'q99':12.5})
        self.assertIn('already known',result['task_details'][0]['source_quote'])

    def test_all_critical_findings_survive_selection_limit_and_rank_before_other_facts(self):
        self.profile.issues=[DataIssue(code=f'note{i}',severity='info',message='Minor') for i in range(20)]
        self.profile.issues += [DataIssue(code=f'leak{i}',severity='error',message='Possible target leakage') for i in range(30)]
        result=self.prepare()
        self.assertEqual(result['status'],'completed')
        self.assertEqual(len(result['mandatory_findings']),30)
        self.assertTrue(all(r['priority']=='critical' for r in result['ranked_findings']))

    def test_bad_fact_ids_are_repaired_in_same_deadline(self):
        class Client(OfflineLLM):
            def generate(self,*args,**kwargs):
                answer=super().generate(*args,**kwargs)
                if len(self.calls)==1:
                    return dict(task_summary='Bad reference',ranked_findings=[dict(finding_id='invented',priority='high',rationale='x',planner_action='x')])
                return answer
        client=Client();result=self.prepare(client)
        self.assertEqual(result['status'],'completed')
        self.assertEqual(len(client.calls),2)
        self.assertIn('supplied finding IDs',client.calls[-1][1]['validation_feedback'])
        self.assertEqual(len(self.store.list('context_budget')),1)

    def test_fabricated_quotes_or_rewritten_task_fail_closed_to_original_text(self):
        class Client(OfflineLLM):
            def generate(self,role,instruction,payload,schema,tools=None):
                return dict(task_summary='Changed metric',task_details=[dict(meaning='False',source_quote='Invented condition')])
        result=self.prepare(Client())
        self.assertEqual(result['status'],'fallback')
        self.assertEqual(result['original_task_description'],self.path.read_text(encoding='utf-8'))
        self.assertEqual(result['requirements']['task']['primary_metric'],'rmse')
        self.assertIn('quote',result['reason'])

    def test_cache_reuses_brief_for_retries_and_invalidates_on_new_evidence(self):
        client=OfflineLLM();first=self.prepare(client);again=self.prepare(client)
        self.assertEqual(first,again);self.assertEqual(len(client.calls),1)
        self.reports['initial']['observations'][0]['numbers']['q99']=15.
        changed=self.prepare(client)
        self.assertNotEqual(first['input_hash'],changed['input_hash']);self.assertEqual(len(client.calls),2)

    def test_timeout_and_interruption_do_not_grant_another_window(self):
        class Client(OfflineLLM):
            def generate(self,*args,**kwargs):raise KeyboardInterrupt('interrupted')
        with self.assertRaises(KeyboardInterrupt):self.prepare(Client())
        deadline=self.store.list('context_budget')[0]['deadline'];self.clock[0]=deadline+1
        client=OfflineLLM();result=self.prepare(client)
        self.assertEqual(result['status'],'fallback');self.assertEqual(client.calls,[])
        self.assertEqual(self.store.list('context_budget')[0]['deadline'],deadline)

    def test_no_budget_skips_agent_and_delivers_original_condition(self):
        client=OfflineLLM();result=self.prepare(client,seconds=0)
        self.assertEqual(client.calls,[]);self.assertEqual(result['status'],'fallback')
        self.assertEqual(result['original_task_description'],self.path.read_text(encoding='utf-8'))

    def test_next_round_receives_new_findings_prior_brief_and_partial_status(self):
        client=OfflineLLM();self.prepare(client)
        self.reports['r1']=dict(status='partial',reason='Time budget',unfinished_checks=['Inspect duplicates'],
                               observations=[],proposals=[])
        result=prepare_planner_context(client=client,context=self.context,profile=self.profile,reports=self.reports,
            store=self.store,budget=self.budget,config=self.config,round_number=2,skills=[])
        self.assertEqual(result['status'],'completed')
        incoming=client.calls[-1][1]
        self.assertEqual(incoming['previous_brief']['requirements']['task'],self.context.task.model_dump(mode='json'))
        self.assertEqual(result['reconnaissance_status']['r1']['unfinished_checks'],['Inspect duplicates'])
        self.assertEqual(incoming['dataset_profile']['dataset'],self.profile.dataset_name)

    def test_late_response_does_not_replace_safe_fallback(self):
        parent=self
        class Client(OfflineLLM):
            def generate(self,*args,**kwargs):
                answer=super().generate(*args,**kwargs);parent.clock[0]+=121
                return answer
        client=Client();result=self.prepare(client)
        self.assertEqual(len(client.calls),1);self.assertEqual(result['status'],'fallback')
        self.assertIn('original_task_description',result)

    def test_planning_and_hypotheses_share_one_prepared_brief(self):
        config=RunConfig(timeouts_enabled=True,total_seconds=1800,final_reserve_seconds=0,prediction_seconds=1)
        client=OfflineLLM()
        with patch('mlmom.pipeline.installed_tools',only_sklearn):
            pipeline=Pipeline(self.path,config,run_dir=self.root/'pipeline',client=client)
        try:
            state=pipeline.inspect({});result=pipeline.planning_call(state)
            self.assertTrue(result['valid'],result)
            roles=[role for role,_ in client.calls]
            self.assertLess(roles.index('reconnaissance_agent'),roles.index('context_agent'))
            self.assertLess(roles.index('context_agent'),roles.index('hypothesis_generator'))
            for role,payload in client.calls:
                if role in {'hypothesis_generator','planner','plan_validator'}:
                    self.assertEqual(payload['planning_brief']['status'],'completed')
                    self.assertEqual(payload['planning_brief']['requirements']['task'],payload['task'])
            pipeline.planning_payload(state)
            self.assertEqual(sum(role=='context_agent' for role,_ in client.calls),1)
            self.assertIn('Predict `target`',pipeline.role_tools.invoke('read_task',{}).data['text'])
        finally:
            pipeline.sandbox.close();pipeline.store.close();pipeline.checkpointer.close();pipeline.trace.close()

    def test_direct_planner_can_request_original_task(self):
        client=RoleClient(LLMSettings(api_key='offline-test'),deadline=time.time()+100)
        self.addCleanup(client.client.close)
        tools=ToolSet()
        def read_task(offset: int=0,chars: int=8000) -> dict:return {'text':'Original meaning'}
        tools.register('read_task',read_task,'Read original task')
        responses=[dict(role='assistant',content=json.dumps({'context_requests':[dict(name='read_task',arguments={},reason='Check target units')]})),
            dict(role='assistant',content=json.dumps({'hypotheses':[dict(hypothesis_id='h',description='Baseline',rationale='Fast',baseline=True)]}))]
        with patch.object(client,'_request',side_effect=responses):
            result=client.generate('hypothesis_generator','Use task meaning',{},HypothesisBatch,tools)
        self.assertEqual(len(result.hypotheses),1)


if __name__=='__main__':unittest.main()
