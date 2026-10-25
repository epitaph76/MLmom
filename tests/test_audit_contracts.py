"""Regressions for evidence loss, incomplete exports and false experiment acceptance."""
import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import httpx
import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.ensemble import ExtraTreesRegressor

from mlmom.agents import RoleClient, GlobalAPIError
from mlmom.context_agent import prepare_planner_context
from mlmom.evaluation import error_metrics, comparable_scores, relative_gain, compact_diagnostics
from mlmom.execution import config_hash
from mlmom.llm_client import LLMSettings
from mlmom.pipeline_contracts import Observation, ReconnaissanceReport, ResultReview
from mlmom.research import review_result
from mlmom.sdk import FitContext
from mlmom.splitting import inner_protocol, make_split, validation_groups
from mlmom.tools import verify_source
from tests.fakes import OfflineLLM
from tests.test_execution import execution_case
from tests.test_planning_limits import case, Skills, Planner
from tests import test_reconnaissance as scouting_fixture


class EvidenceContractTests(unittest.TestCase):
    setUp = scouting_fixture.ScoutingTests.setUp
    session = scouting_fixture.ScoutingTests.session

    def test_worker_error_is_compact_and_preserves_full_traceback(self):
        from mlmom.worker import failure_result
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'preprocessing.py'
            code=compile("raise ValueError('specific defect ' + 'x' * 9000)",str(source),'exec')
            try: exec(code,{})
            except ValueError as error: result=failure_result(error,tmp)
            self.assertLess(len(result['error']),1800)
            self.assertNotIn('Traceback (most recent call last)',result['error'])
            self.assertEqual(result['error_details']['source']['file'],str(source))
            self.assertEqual(result['error_details']['source']['line'],1)
            self.assertIn('apply_patch',result['error_details']['hint'])
            self.assertIn('x'*9000,Path(result['traceback_ref']).read_text(encoding='utf-8'))

    def test_documented_polars_templates_execute_on_pinned_engine(self):
        session=self.session()
        binned=session.csv.query_csv("SELECT bin, COUNT(*) AS n, AVG(target) AS average FROM (SELECT CASE WHEN x < 0 THEN 'low' ELSE 'high' END AS bin, target FROM train) AS b GROUP BY bin ORDER BY bin")
        self.assertEqual(sum(row['n'] for row in binned['rows']),180)
        missing=session.csv.query_csv('SELECT SUM(CASE WHEN x IS NULL THEN 1 ELSE 0 END) AS missing, COUNT(*) AS n FROM train')
        self.assertEqual(missing['rows'],[dict(missing=0,n=180)])
        combined=session.csv.query_csv("SELECT 'train' AS source, COUNT(*) AS n FROM train UNION ALL SELECT 'test' AS source, COUNT(*) AS n FROM test")
        self.assertEqual({row['source']:row['n'] for row in combined['rows']},{'train':180,'test':20})

    def test_structurally_invalid_final_item_does_not_erase_valid_items(self):
        good=dict(observation_id='local',topic='size',description='Count',numbers={'n':180},source_ref='csv:valid')
        report=ReconnaissanceReport.model_validate(dict(observations=[good,dict(good,rows=-1)]))
        self.assertEqual(report.status,'partial');self.assertEqual(len(report.observations),1)
        self.assertTrue(report.unfinished_checks)

    def test_observation_numbers_and_source_attribution_are_retained(self):
        session=self.session()
        result=session.csv.query_csv('SELECT AVG(x) AS average FROM train')
        saved=session.save_observation('average','Rounded assessment',numbers={'average':0.123},rows=180,source_ref=result['source_ref'])
        self.assertEqual(saved['rows'],180)
        self.assertEqual(saved['numbers'],{'average':0.123})
        self.assertEqual(saved['source_ref'],result['source_ref'])


    def test_final_observations_keep_proposals_with_original_unchecked_ids(self):
        session=self.session();result=session.csv.query_csv('SELECT COUNT(*) AS n FROM train')
        draft=Observation(observation_id='local-count',topic='size',description='Measured training count',numbers={'n':180},source_ref=result['source_ref'])
        valid=dict(description='Check size',evidence_ids=['local-count'],expected_effect='unknown',verification='same split')
        answer=session.reconcile(ReconnaissanceReport(observations=[draft],proposals=[valid,dict(valid,evidence_ids=['invented'])]))
        self.assertEqual(answer.status,'partial');self.assertEqual(len(answer.observations),1)
        self.assertEqual(len(answer.proposals),2)
        self.assertEqual([p.evidence_ids for p in answer.proposals],[['local-count'],['invented']])
        self.assertIsNone(answer.reason)
        self.assertEqual(len(self.session().observations),1)

    def test_invalid_observation_does_not_discard_proposals(self):
        session=self.session();result=session.csv.query_csv('SELECT COUNT(*) AS n FROM train')
        a=Observation(observation_id='same',topic='size',description='Count',numbers={'n':180},source_ref=result['source_ref'])
        b=a.model_copy(update={'numbers':{'n':999}})
        report=session.reconcile(ReconnaissanceReport(observations=[a,b],proposals=[dict(description='bad alias',evidence_ids=['same'],expected_effect='gain',verification='test')]))
        self.assertEqual(len(report.observations),2)
        self.assertEqual(len(report.proposals),1)
        self.assertEqual(report.proposals[0].evidence_ids,['same'])
        self.assertEqual(report.status,'partial')  # Pending EDA does not discard advisory proposals.

    def test_reworded_saved_observation_does_not_discard_its_proposal(self):
        session=self.session();result=session.csv.query_csv('SELECT COUNT(*) AS n FROM train')
        saved=Observation.model_validate(session.save_observation('size','Training count',numbers={'n':180},source_ref=result['source_ref']))
        proposal=dict(description='Check size',evidence_ids=[saved.observation_id],expected_effect='unknown',verification='same split')
        report=session.reconcile(ReconnaissanceReport(observations=[saved.model_copy(update={'description':'Reworded training count'})],proposals=[proposal]))
        self.assertEqual(report.status,'partial')
        self.assertEqual(report.proposals[0].model_dump(exclude={'agent_notes'}),proposal)
        self.assertIsNone(report.reason)

    def test_proposal_references_can_be_missing_or_empty(self):
        session=self.session()
        proposal=dict(description='Try a different model',expected_effect='unknown',verification='same split')
        report=session.reconcile(ReconnaissanceReport(proposals=[proposal,dict(proposal,evidence_ids=[])]))
        self.assertEqual(report.status,'partial')
        self.assertEqual(len(report.proposals),2)
        self.assertEqual([p.evidence_ids for p in report.proposals],[[],[]])

    def test_saved_observation_reaches_context_and_hypothesis_roles(self):
        parent=self
        class Client(OfflineLLM):
            def generate(self,role,instruction,payload,schema,tools=None):
                if role=='reconnaissance_agent':
                    super().generate(role,instruction,payload,schema,tools)
                    evidence=tools.invoke('query_csv',dict(sql='SELECT COUNT(*) AS n FROM train')).data
                    saved=tools.invoke('save_observation',dict(topic='size',description='Training count',numbers={'n':180},source_ref=evidence['source_ref'])).data
                    return schema(observations=[saved])
                return super().generate(role,instruction,payload,schema,tools)
        client=Client();session=self.session(client);report=session.run()
        # A real pipeline payload exercises scout -> context_agent -> hypothesis_generator.
        from mlmom.pipeline import Pipeline
        from mlmom.pipeline_contracts import RunConfig
        from tests.test_pipeline import only_sklearn
        with patch('mlmom.pipeline.installed_tools',only_sklearn):
            pipe=Pipeline(self.task,RunConfig(max_rounds=1,max_workers=1,reserve_cpus=0,reserve_memory_gb=0),run_dir=self.root/'handoff',client=client)
            try:
                state=pipe.inspect({})
                pipe.store.put('reconnaissance','initial',report)
                payload=pipe.planning_payload(state)
                facts=next(p for role,p in reversed(client.calls) if role=='context_agent')['dataset_facts']
                self.assertTrue(any(f['kind']=='observation' and f['evidence']['numbers']=={'n':180} for f in facts))
                from mlmom.planning import build_planning_graph
                graph=build_planning_graph(client,pipe.context,pipe.tools,pipe.skills,pipe.budget.snapshot(),planning_mode='baseline')
                graph.invoke({'payload':payload})
                received=next(p for role,p in reversed(client.calls) if role=='hypothesis_generator')
                self.assertTrue(received['observation_index'])
                self.assertTrue(all(item.get('source_ref') for item in received['observation_index']))
                self.assertEqual(received['reconnaissance']['evidence_available_via'],'read_evidence')
            finally:
                pipe.sandbox.close();pipe.checkpointer.close();pipe.store.close();pipe.trace.close()








class MetricsAndSplitTests(unittest.TestCase):
    def test_slice_absolute_error_and_signed_residual_are_distinct(self):
        values=error_metrics([10,20],[9,23],'regression')
        self.assertEqual(values['mae'],2);self.assertEqual(values['mean_residual'],-1)
        self.assertAlmostEqual(values['rmse'],np.sqrt(5))
        self.assertEqual(values['positive_residual_means'],'underprediction')
        self.assertIn('hamming_loss',error_metrics([[0,1]],[[1,1]],'multilabel_classification'))
        legacy=compact_diagnostics(dict(mean_residual=1,safe_slices=[dict(rows=20,mean_error=2)]))
        self.assertNotIn('mean_error',legacy['worst_slices'][0]);self.assertEqual(legacy['worst_slices'][0]['mae'],2)

    def test_relative_gain_requires_matching_metric_level_options_and_split(self):
        a=dict(score_kind='outer_holdout',dataset_hash='data',split_id='r1',metric='rmse',metric_options={},direction='minimize',weighting='unweighted',score=9)
        b=dict(a,score=10)
        self.assertAlmostEqual(relative_gain(a,b),.1)
        for change in [dict(score_kind='inner_cv_mean'),dict(split_id='r2'),dict(metric_options={'multioutput':'raw_values'}),dict(weighting='sample_weight')]:
            self.assertFalse(comparable_scores(a,dict(b,**change)));self.assertIsNone(relative_gain(a,dict(b,**change)))

    def test_duplicate_groups_and_explicit_groups_form_connected_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame,context,_,_=execution_case(Path(tmp),'regression')
            frame.loc[1,context.feature_columns]=frame.loc[0,context.feature_columns].to_numpy()
            frame['group']=np.arange(len(frame))
            frame.loc[2,'group']=frame.loc[1,'group']
            context.task.group_columns=['group'];context.task.validation_strategy='group_kfold';context.task.validation_folds=3
            context.task.duplicate_policy='group_features'
            labels=validation_groups(frame,context)
            self.assertEqual(labels[0],labels[2])
            protocol=inner_protocol(frame,context)
            for fold in protocol['folds']:
                self.assertFalse(set(labels[fold['train_indices']]) & set(labels[fold['validation_indices']]))
            outer=make_split(frame,context,'r1',.2,42)
            self.assertFalse(set(labels[outer.train_indices]) & set(labels[outer.holdout_indices]))

    def test_duplicate_classification_protocol_and_infeasible_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame,context,_,_=execution_case(Path(tmp),'binary_classification')
            context.task.duplicate_policy='group_features';context.task.validation_folds=3
            protocol=inner_protocol(frame,context)
            self.assertEqual(protocol['implementation'],'stratified_group_kfold')
            frame[context.feature_columns]=0
            with self.assertRaisesRegex(ValueError,'groups'):inner_protocol(frame,context)

    def test_methodological_claims_are_left_to_the_agent(self):
        from mlmom.planning import validate_plan
        context,tools,budget,plan=case()
        plan.hypothesis.description='duplicate-aware evaluation'
        self.assertTrue(validate_plan(plan,context,tools,Skills(),budget).valid)



class ProviderFailureTests(unittest.TestCase):
    def test_billing_failure_is_latched_across_roles_without_another_request(self):
        calls=[]
        def handler(request):
            calls.append(request)
            return httpx.Response(402,json={'error':{'message':'Insufficient balance'}})
        client=RoleClient(LLMSettings(api_key='offline-test'),deadline=time.time()+20)
        client.client.close()
        client.client=OpenAI(api_key='offline-test',max_retries=0,http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        self.addCleanup(client.client.close)
        with tempfile.TemporaryDirectory() as tmp:
            from mlmom.tracing import TraceSink
            sink=TraceSink(Path(tmp),'billing');client.trace=sink;self.addCleanup(sink.close)
            for role in ['reconnaissance_agent','context_agent']:
                with self.assertRaises(GlobalAPIError):client._request([{'role':'user','content':'test'}],role=role)
            sink.close()
        self.assertEqual(len(calls),1)


if __name__=='__main__':unittest.main()
