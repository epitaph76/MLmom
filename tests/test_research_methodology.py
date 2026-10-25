"""EDA coverage is observable; importance diagnostics preserve correlated families."""
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np
import pandas as pd

from mlmom.eda import EDA_TOPICS, eda_matrix
from mlmom.pipeline_contracts import ReconnaissanceReport
from tests import test_reconnaissance as recon_fixtures


ROOT=Path(__file__).resolve().parents[1]


class EDAMethodologyTests(unittest.TestCase):
    def setUp(self):
        self.fixture=recon_fixtures.ScoutingTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_unreported_screening_is_partial_without_blocking_confirmed_requirements(self):
        session=self.fixture.session()
        report=session.run()
        self.assertEqual(report.status,'partial')
        self.assertIsNotNone(session.store.get('confirmed_task','task'))
        self.assertEqual(set(report.agent_notes['eda_matrix']),set(EDA_TOPICS))
        self.assertEqual(report.agent_notes['eda_coverage'],dict(reviewed=0,total=10))
        self.assertEqual(len(report.unfinished_checks),10)

    def test_topic_progress_resumes_without_resetting_budget_and_can_complete(self):
        session=self.fixture.session()
        result=session.tools.invoke('record_eda_status',dict(topic='missingness',status='screened',
            details=dict(scope='current train',reviewed_columns=['x'],summary='Missing rates screened')))
        self.assertEqual(result.status,'ok',result)
        again=session.tools.invoke('record_eda_status',dict(topic='missingness',status='screened',
            details=dict(topic='copied metadata',scope='current train',summary='Scoped screen')))
        self.assertEqual(again.status,'ok',again)
        self.assertEqual(again.data['topic'],'missingness')
        resumed=self.fixture.session()
        self.assertEqual(resumed.deadline,session.deadline)
        self.assertEqual(resumed.eda_matrix['missingness']['status'],'screened')
        all_done={topic:dict(status='screened',scope='current train',summary='Synthetic report declaration')
                  for topic in EDA_TOPICS}
        report=resumed.reconcile(ReconnaissanceReport(agent_notes=dict(eda_matrix=all_done),
            unfinished_checks=[f'EDA screening pending: {topic}' for topic in EDA_TOPICS]))
        self.assertEqual(report.status,'completed')
        self.assertEqual(report.agent_notes['eda_coverage'],dict(reviewed=10,total=10))
        self.assertEqual(report.unfinished_checks,[])
        resumed.unfinished=['Old manual follow-up']
        resolved=resumed.reconcile(ReconnaissanceReport(agent_notes=dict(eda_matrix=all_done),unfinished_checks=[]))
        self.assertEqual(resolved.status,'completed')
        self.assertEqual(resolved.unfinished_checks,[])

    def test_not_applicable_needs_reason_and_invalid_states_stay_pending(self):
        session=self.fixture.session()
        invalid=session.tools.invoke('record_eda_status',dict(topic='temporal_patterns',status='not_applicable'))
        self.assertEqual(invalid.status,'error')
        valid=session.tools.invoke('record_eda_status',dict(topic='temporal_patterns',status='not_applicable',
            details=dict(reason='No observable time/order fields in permitted inputs')))
        self.assertEqual(valid.status,'ok')
        matrix=eda_matrix(update=dict(target=dict(status='fully_verified'),leakage=dict(status='not_applicable')))
        self.assertEqual(matrix['target']['status'],'pending')
        self.assertEqual(matrix['leakage']['status'],'pending')
        self.assertEqual(eda_matrix(update=dict(target=dict(status='screened')))['target']['status'],'pending')
        invalid=session.tools.invoke('record_eda_status',dict(topic='target',status='screened'))
        self.assertEqual(invalid.status,'error')

    def test_effect_support_and_performed_alternatives_survive_tool_and_report_persistence(self):
        session=self.fixture.session()
        result=session.csv.query_csv('SELECT COUNT(*) AS n FROM train')
        controls=[dict(explanation='Possible duplicate composition',status='tested',
            source_ref=result['source_ref'],result='Synthetic control result; not a population claim')]
        saved=session.tools.invoke('save_observation',dict(topic='shift',description='Supported segment effect',
            rows=40,numbers=dict(population_rows=180,effect_delta=.2),source_ref=result['source_ref'],
            agent_notes=dict(comparator='rest of current train',alternative_checks=controls)))
        self.assertEqual(saved.status,'ok',saved)
        self.assertEqual(saved.data['rows'],40)
        self.assertEqual(saved.data['result_rows'],1)
        resumed=self.fixture.session()
        found=next(iter(resumed.observations.values()))
        self.assertEqual(found.agent_notes['alternative_checks'],controls)
        report=resumed.reconcile(ReconnaissanceReport(observations=[found]))
        self.assertEqual(report.observations[0].numbers['population_rows'],180)
        self.assertEqual(report.observations[0].agent_notes['alternative_checks'],controls)

    def test_live_payload_contains_matrix_and_uses_registered_round_train_hash(self):
        from tests.fakes import OfflineLLM
        seen=[]
        class Client(OfflineLLM):
            def generate(self,role,instruction,payload,schema,tools=None):
                seen.append(payload['eda_research_plan'])
                assert 'record_eda_status' in tools.tools
                return super().generate(role,instruction,payload,schema,tools)
        session=self.fixture.session(Client());session.run()
        self.assertEqual(set(seen[0]['topics']),set(EDA_TOPICS))
        self.assertEqual(seen[0]['train_hash'],session.csv.tables['train']['hash'])


class ImportanceMethodologyTests(unittest.TestCase):
    def helper(self):
        path=ROOT/'skills/preprocessing_agent/feature_selection/REFERENCE.md'
        source=re.search(r'```python\n(.*?)\n```',path.read_text(encoding='utf-8'),re.S).group(1)
        namespace={};exec(source,namespace)
        return namespace['repeated_permutation']

    def test_repeated_joint_permutation_detects_information_masked_in_individual_scores(self):
        rng=np.random.default_rng(19);signal=rng.normal(size=256)
        X=pd.DataFrame(dict(a=signal,b=signal,c=signal,noise=rng.normal(size=256)),index=np.arange(256)+1000)
        original=X.copy();weights=np.linspace(1.,3.,len(X))
        predict=lambda frame:np.median(frame[['a','b','c']].to_numpy(),axis=1)
        score=lambda prediction:float(np.sqrt(np.average((signal-prediction)**2,weights=weights)))
        groups=dict(a=['a'],b=['b'],c=['c'],noise=['noise'],family=['a','b','c'])
        helper=self.helper();result=helper(X,predict,score,groups,n_repeats=7,seed=25)
        self.assertEqual(result['baseline'],0.)
        for name in ['a','b','c','noise']:
            self.assertEqual(result['groups'][name]['values'],[0.]*7)
        self.assertGreater(result['groups']['family']['mean'],.5)
        self.assertEqual(result['groups']['family']['repeats'],7)
        self.assertEqual(result,helper(X,predict,score,groups,n_repeats=7,seed=25))
        maximizing=helper(X,predict,lambda p:-score(p),groups,n_repeats=7,seed=25,direction='maximize')
        np.testing.assert_allclose(maximizing['groups']['family']['values'],result['groups']['family']['values'])
        pd.testing.assert_frame_equal(X,original)

    def test_joint_shuffle_keeps_internal_relations_and_row_axis(self):
        X=pd.DataFrame(dict(a=np.arange(30.),b=np.arange(30.),c=np.arange(30.)),index=np.arange(30)+500)
        def predict(frame):
            self.assertTrue(frame.index.equals(X.index))
            np.testing.assert_array_equal(frame.a,frame.b)
            np.testing.assert_array_equal(frame.a,frame.c)
            return frame.a.to_numpy()
        score=lambda p:float(np.mean((X.a.to_numpy()-p)**2))
        result=self.helper()(X,predict,score,dict(family=['a','b','c']),n_repeats=3)
        self.assertEqual(len(result['groups']['family']['values']),3)
        self.assertGreater(result['groups']['family']['mean'],0)


class MethodCardDeliveryTests(unittest.TestCase):
    def test_preprocessing_receives_both_method_references_and_records_versions(self):
        from mlmom.coding import build_coding_graph
        from mlmom.registry import SkillRegistry
        from mlmom.pipeline_contracts import RunConfig
        from tests import test_planning_limits as fixtures
        registry=SkillRegistry(ROOT/'skills')
        cards=registry.load(['feature_engineering','feature_selection'],'regression',
            role='preprocessing_agent',include_references=True,max_chars=48000)
        captured=[]
        class Client(fixtures.Coder):
            def generate(self,role,instruction,payload,schema,tools=None):
                if role=='preprocessing_agent':captured.extend(payload['skills'])
                return super().generate(role,instruction,payload,schema,tools)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);context,tools,_,plan=fixtures.case();budget=fixtures.ClockBudget()
            result=build_coding_graph(Client(budget,1),workspace=root/'code',store=fixtures.Store(root),
                plan=plan,plan_ref=root/'plan.json',skill_data=[],tools=tools,context=context,
                sandbox=fixtures.Sandbox(budget),budget=budget,config=RunConfig(),round_id='methods',
                preprocessing_skills=cards,skill_registry=registry).invoke({'payload':{}})
            self.assertTrue(result['valid'],result)
            for key in ['feature_engineering','feature_selection']:
                card=next(c for c in captured if c['skill_id']==key)
                self.assertTrue(card['content'].endswith((registry.paths[key]/'REFERENCE.md').read_text(encoding='utf-8')))
                self.assertEqual(result['candidate']['skill_versions'][key]['sha256'],card['sha256'])
                self.assertEqual(result['candidate']['skill_versions'][key]['version'],'1.1.0')


if __name__=='__main__':unittest.main()
