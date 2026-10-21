"""Regression checks for the slow, tool-heavy Qwen planning failures."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch

from mlmom.agents import RoleClient,LLMUnavailable,StructuredOutputError
from mlmom.llm_client import LLMSettings
from mlmom.pipeline import Pipeline
from mlmom.pipeline_contracts import RunConfig,HypothesisBatch
from mlmom.planning import build_planning_graph
from mlmom.tools import ToolSet
from tests.test_planning_limits import Planner,Skills,case
from mlmom.planning_context import profile_summary,experiment_summary


class PlanningRecoveryTests(unittest.TestCase):
    def client(self,**kwargs):
        client=RoleClient(LLMSettings(api_key='unit-test-only'),deadline=time.time()+600,**kwargs)
        self.addCleanup(client.client.close)
        return client

    def final(self):
        return dict(role='assistant',content=json.dumps(dict(hypotheses=[case()[3].hypothesis.model_dump()])))

    def test_first_request_has_no_tools_and_can_finish_immediately(self):
        client=self.client();tools=ToolSet();read=Mock(return_value=[])
        def history():return read()
        tools.register('read_history',history,'Read earlier evidence')
        with patch.object(client,'_request',return_value=self.final()) as request:
            result=client.generate('hypothesis_generator','Use existing evidence',{},HypothesisBatch,tools)
        self.assertEqual(len(result.hypotheses),1)
        self.assertIsNone(request.call_args.kwargs['tools']);request.assert_called_once();read.assert_not_called()

    def test_compact_views_preserve_schema_target_and_failed_trial_counts(self):
        from mlmom.data_inspector import DataInspector,build_llm_context
        from tests.test_pipeline import dataset
        with tempfile.TemporaryDirectory() as tmp:
            task=dataset(Path(tmp)/'input','multiclass_classification')
            profile=DataInspector().inspect_directory(task.parent)
            compact=profile_summary(profile)
        self.assertEqual({c['name'] for c in compact['columns']},{c.name for c in profile.columns})
        self.assertEqual(compact['checks']['duplicate_train_feature_rows'],profile.checks.duplicate_train_feature_rows)
        self.assertTrue(compact['targets']['target']['distribution'])
        self.assertLess(len(json.dumps(compact)),len(json.dumps(build_llm_context(profile))))
        summary=experiment_summary(dict(experiment_id='run',trials=[dict(status='completed')]*15+[dict(status='failed')]))
        self.assertEqual(summary['trials'],dict(total=16,completed=15,failed=1))

    def test_context_is_read_only_when_explicitly_requested(self):
        client=self.client();tools=ToolSet();calls=[]
        def read_skills(skill_ids:list[str]):
            calls.append(skill_ids);return ['Backend contract']
        tools.register('read_skills',read_skills,'Read a missing backend contract')
        extra=dict(role='assistant',content=json.dumps(dict(context_requests=[dict(name='read_skills',
            arguments=dict(skill_ids=['sklearn']),reason='Need the linear estimator parameter contract')])))
        with patch.object(client,'_request',side_effect=[extra,self.final()]) as request:
            result=client.generate('planner','Plan',{},HypothesisBatch,tools)
        self.assertEqual(calls,[['sklearn']]);self.assertEqual(len(result.hypotheses),1)
        self.assertTrue(all(c.kwargs['tools'] is None for c in request.call_args_list))
        self.assertEqual(request.call_count,2)

    def test_unjustified_or_repeated_context_reads_are_rejected(self):
        for reason in ('','Specific missing contract'):
            with self.subTest(reason=reason):
                client=self.client();tools=ToolSet();calls=[]
                def read_skills():calls.append(1);return []
                tools.register('read_skills',read_skills,'Contract')
                extra=dict(role='assistant',content=json.dumps(dict(context_requests=[dict(name='read_skills',
                    arguments={},reason=reason)])))
                with patch.object(client,'_request',side_effect=[extra,extra]):
                    with self.assertRaises(StructuredOutputError):
                        client.generate('planner','Plan',{},HypothesisBatch,tools)
                self.assertEqual(len(calls),int(bool(reason)))

    def test_three_and_half_minute_role_window_still_obeys_research_deadline(self):
        config=RunConfig()
        self.assertEqual((config.hypothesis_seconds,config.planner_seconds,config.validator_seconds),(210,210,210))
        client=self.client();client.deadline=1300
        with patch('mlmom.agents.time.time',return_value=1000):
            with client.budget_scope('planner','2:p0',config.planner_seconds):self.assertEqual(client.deadline,1210)
        client.deadline=1100
        with patch('mlmom.agents.time.time',return_value=1000):
            with client.budget_scope('planner','2:p1',config.planner_seconds):self.assertEqual(client.deadline,1100)

    def test_planning_retry_stays_in_current_cycle_and_is_bounded(self):
        pipeline=object.__new__(Pipeline)
        pipeline.config=RunConfig(max_rounds=2,max_attempts=3)
        pipeline.store=SimpleNamespace(event=Mock(),list=lambda *a,**kw:[])
        pipeline.budget=SimpleNamespace(snapshot=lambda:SimpleNamespace(research_seconds=2000))
        pipeline.last_submission={'path':'existing.csv'}
        pipeline.planning_call=Mock(side_effect=LLMUnavailable('APITimeoutError'))
        state=dict(round_number=2)
        for expected in (1,2):
            result=pipeline.plan(state)
            self.assertNotIn('round_number',result)
            self.assertEqual(result['planning_attempt'],expected)
            self.assertEqual(result['stop_reason'],'planning_retry');state.update(result)
        result=pipeline.plan(state)
        self.assertEqual(result['stop_reason'],'planning_attempts_exhausted')
        self.assertEqual(state['round_number'],2)

    def test_validation_retry_reuses_hypothesis_and_proposed_code_plan(self):
        class ReviewTimeout(Planner):
            fail=True
            def generate(self,role,instruction,payload,schema,tools=None):
                if role=='plan_validator' and self.fail:
                    self.fail=False;self.calls.append((role,payload));raise LLMUnavailable('APITimeoutError')
                result=super().generate(role,instruction,payload,schema,tools)
                if role=='planner':result.plans[0].execution_mode='custom'
                return result
        context,tools,budget,_=case();client=ReviewTimeout();saved={}
        def save(progress):saved.update(json.loads(json.dumps(progress)))
        first=build_planning_graph(client,context,tools,Skills(),budget,planning_mode='baseline',
            lightweight=False,save_progress=save).invoke({'payload':{'round_number':2}})
        self.assertFalse(first['valid']);self.assertIn('base',saved['proposed'])
        second=build_planning_graph(client,context,tools,Skills(),budget,planning_mode='baseline',
            lightweight=False,saved_progress=saved,save_progress=save).invoke(
                {'payload':{'round_number':2,'planning_attempt':1},'hypotheses':saved['hypotheses']})
        self.assertTrue(second['valid'])
        self.assertEqual(sum(role=='hypothesis_generator' for role,_ in client.calls),1)
        self.assertEqual(sum(role=='planner' for role,_ in client.calls),1)
        self.assertEqual(sum(role=='plan_validator' for role,_ in client.calls),2)
