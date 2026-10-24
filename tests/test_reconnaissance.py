"""Real pinned CSV engine, partial scouting recovery, and unchanged task validation."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import polars as pl

from mlmom.csv_tools import csv_worker, validate_sql
from mlmom.data_inspector import DataInspector
from mlmom.inputs import resolve_inputs
from mlmom.persistence import ExperimentStore, atomic_json
from mlmom.pipeline_contracts import RunConfig
from mlmom.research import ReconnaissanceSession
from mlmom.resources import Budget, SandboxManager
from mlmom.pipeline import Pipeline
from mlmom.agents import RoleClient
from mlmom.llm_client import LLMSettings
from tests.fakes import OfflineLLM
from tests.test_pipeline import dataset


class CSVEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        pd.DataFrame({'x':range(250),'y':[i*2. for i in range(250)],
                      'город':['Москва' if i%2 else 'Тула' for i in range(250)]}).to_csv(self.root/'train.csv',index=False)
        pd.DataFrame({'id':[1],'y':[999]}).to_csv(self.root/'sample.csv',index=False)
        self.tables = {'train':dict(path=str(self.root/'train.csv'),role='training'),
                       'sample_submission':dict(path=str(self.root/'sample.csv'),role='template',schema_only=True)}

    def run_csv(self, operation, **args):
        directory = Path(tempfile.mkdtemp(dir=self.root))
        return csv_worker(dict(operation=operation,tables=self.tables,cache_dir=str(self.root/'cache'),**args),directory)

    def test_pinned_engine_aggregates_all_rows_and_bounds_raw_outputs(self):
        self.assertEqual(pl.__version__,'1.26.0')
        result=self.run_csv('query',sql='SELECT COUNT(*) AS n, AVG(y) AS avg_y FROM train')
        self.assertEqual(result['rows'],[{'n':250,'avg_y':249.}])
        result=self.run_csv('query',sql='SELECT * FROM train')
        self.assertTrue(result['truncated']);self.assertEqual(len(result['rows']),100)
        self.assertEqual(pl.read_parquet(result['artifact_ref']).height,250)
        self.assertLessEqual(len(json.dumps(result,ensure_ascii=False)),32000)

    def test_unicode_grouping_preview_and_pairwise_correlation(self):
        result=self.run_csv('query',sql='SELECT "город", COUNT(*) AS n FROM train GROUP BY "город"')
        self.assertEqual({r['город']:r['n'] for r in result['rows']},{'Москва':125,'Тула':125})
        self.assertEqual(self.run_csv('preview',table='train',columns=['x'],offset=20,rows=2)['rows'],[{'x':20},{'x':21}])
        result=self.run_csv('correlate',table='train',columns=['x'],target='y')
        self.assertAlmostEqual(result['rows'][0]['correlation'],1.)
        self.assertEqual(result['rows'][0]['pairs'],250)

    def test_cp1251_chunks_and_invalid_csv_not_silently_skipped(self):
        pd.DataFrame({'x':[1,2,3],'город':['Москва','Тула',None]}).to_csv(self.root/'cp.csv',index=False,encoding='cp1251',sep=';')
        self.tables['train']=dict(path=str(self.root/'cp.csv'),role='training',format=dict(encoding='cp1251',delimiter=';'))
        result=self.run_csv('query',sql='SELECT AVG(x) AS avg_x, COUNT(*) AS n FROM train')
        self.assertEqual(result['rows'],[{'avg_x':2.,'n':3}])
        self.assertEqual(len(list((self.root/'cache').glob('*.parquet'))),1)
        (self.root/'cp.csv').write_text('x;город\n1;Тула\n2;Москва;extra\n',encoding='cp1251')
        with self.assertRaises(pd.errors.ParserError):self.run_csv('query',sql='SELECT * FROM train')

    def test_nulls_long_cells_and_wide_tables(self):
        frame=pd.DataFrame({f'c{i}':[None,'x'*40000] for i in range(25)})
        frame.to_csv(self.root/'train.csv',index=False)
        result=self.run_csv('query',sql='SELECT * FROM train')
        self.assertTrue(result['truncated']);self.assertEqual(len(result['columns']),20)
        self.assertLessEqual(len(json.dumps(result,ensure_ascii=False)),32000)
        self.assertEqual(pl.read_parquet(result['artifact_ref']).shape,(2,25))
        result=self.run_csv('query',sql='SELECT c0 FROM train')
        self.assertTrue(result['truncated']);self.assertTrue(result['cells_truncated'])

    def test_sql_access_is_limited_to_registered_read_only_tables(self):
        bad=['DELETE FROM train','SELECT * FROM train; SELECT * FROM train',
             "SELECT * FROM read_csv('outside.csv')", "WITH t AS (SELECT * FROM READ_PARQUET('x')) SELECT * FROM t",
             '''SELECT * FROM "read_csv"('x')''', "SELECT * FROM read_/*comment*/csv('x')",
             'WITH t AS (DELETE FROM train RETURNING *) SELECT * FROM t']
        for sql in bad:
            with self.subTest(sql=sql),self.assertRaises(ValueError):validate_sql(sql)
        for sql in ['SELECT * FROM holdout','SELECT * FROM sample_submission','SELECT absent FROM train']:
            with self.subTest(sql=sql),self.assertRaises(pl.exceptions.PolarsError):self.run_csv('query',sql=sql)
        self.assertIn('total_columns',self.run_csv('schema',table='sample_submission')['tables'][0])
        self.assertEqual(self.run_csv('query',sql="SELECT 'read_csv' AS text FROM train LIMIT 1")['rows'][0]['text'],'read_csv')

    def test_csv_worker_uses_real_scheduler_and_obeys_deadline(self):
        config=RunConfig(timeouts_enabled=True,total_seconds=60,final_reserve_seconds=0,reserve_cpus=0,reserve_memory_gb=0)
        with SandboxManager(timeouts_enabled=True,budget=Budget(config,time.time()),max_workers=1) as sandbox:
            req=dict(action='csv',operation='query',tables=self.tables,cache_dir=str(self.root/'cache'),sql='SELECT COUNT(*) AS n FROM train')
            result=sandbox.run(req,self.root/'worker',seconds=15,memory_gb=2,cpus=1)
            self.assertEqual(result['status'],'completed',result);self.assertEqual(result['rows'],[{'n':250}])
            result=sandbox.run(req,self.root/'expired',seconds=.01,memory_gb=2,cpus=1)
            self.assertEqual(result['status'],'failed');self.assertEqual(result['stop_reason'],'time_limit')


class ScoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.task=dataset(self.root/'input')
        self.profile=DataInspector().inspect_directory(self.task.parent)
        self.store=ExperimentStore(self.root/'run');self.addCleanup(self.store.close)
        atomic_json(self.store.root/'inspection'/'profile.json',self.profile)
        self.config=RunConfig(timeouts_enabled=True,total_seconds=1000,final_reserve_seconds=0)
        self.clock=[time.time()]
        self.budget=SimpleNamespace(now=lambda:self.clock[0],cpus=2,memory_gb=4,
                                    snapshot=lambda:SimpleNamespace(research_seconds=1000))
        class DirectSandbox:
            calls=0
            def run(inner,req,directory,**limits):
                inner.calls+=1;directory.mkdir(parents=True,exist_ok=True)
                return csv_worker(req,directory)
        self.sandbox=DirectSandbox()

    def session(self,client=None,**args):
        return ReconnaissanceSession(client=client or OfflineLLM(),profile=self.profile,files=resolve_inputs(self.task),
            sandbox=self.sandbox,budget=self.budget,store=self.store,config=self.config,key='initial',history={},skills=[],**args)

    def test_confirmation_corrects_exclusions_and_validator_allows_repair(self):
        session=self.session()
        task=json.loads(OfflineLLM().complete([{'content':json.dumps({'dataset_inspection':{'task':self.profile.task.model_dump(mode='json')}})}]))
        invalid=task|dict(target_columns=['invented'])
        self.assertFalse(session.tools.invoke('confirm_task',{'task':invalid}).data['valid'])
        task['excluded_feature_columns']=[]
        self.assertTrue(session.tools.invoke('confirm_task',{'task':task}).data['valid'])
        self.assertEqual(self.store.get('confirmed_task','task')['excluded_feature_columns'],['target','id'])
        self.assertEqual(session.tools.invoke('confirm_task',{'task':task|dict(primary_metric='mae')}).status,'error')

    def test_real_role_client_repairs_requirements_through_tool_feedback(self):
        task=json.loads(OfflineLLM().complete([{'content':json.dumps({'dataset_inspection':{'task':self.profile.task.model_dump(mode='json')}})}]))
        client=RoleClient(LLMSettings(api_key='offline-test'),deadline=time.time()+500,max_steps=2,budget_store=self.store)
        def call(task_value,number):
            return dict(role='assistant',tool_calls=[dict(id=str(number),type='function',function=dict(name='confirm_task',
                arguments=json.dumps(dict(task=task_value))))])
        replies=[call(task|dict(target_columns=['missing_target']),1),call(task,2),
                 dict(role='assistant',content=json.dumps(dict(observations=[],proposals=[],unfinished_checks=[])))]
        with patch.object(client,'_request',side_effect=replies) as request:
            report=self.session(client).run()
        self.assertEqual(report.status,'completed',report)
        self.assertEqual(request.call_count,3)
        messages=request.call_args_list[1].args[0]
        # API requests share one conversation; a failed tool reports concrete validator feedback.
        feedback=[json.loads(m['content']) for m in messages if m['role']=='tool']
        self.assertTrue(any(item.get('data',{}).get('valid') is False for item in feedback))
        self.assertEqual(client.max_steps,2)

    def test_reconfirming_task_with_other_explanations_does_not_change_execution(self):
        session=self.session()
        task=json.loads(OfflineLLM().complete([{'content':json.dumps({
            'dataset_inspection':{'task':self.profile.task.model_dump(mode='json')}})}]))
        first=session.confirm_task(task | {'notes':'Initial explanation'})
        again=session.confirm_task(task | {'notes':'Another explanation'})
        self.assertTrue(first['valid'])
        self.assertEqual(first,again)
        self.assertEqual(again['task']['agent_notes'],{'notes':'Initial explanation'})

    def test_confirmation_accepts_json_string_with_unchanged_validation_and_deadline(self):
        session=self.session();deadline=session.deadline
        task=json.loads(OfflineLLM().complete([{'content':json.dumps({'dataset_inspection':{'task':self.profile.task.model_dump(mode='json')}})}]))
        for invalid in ['{',json.dumps(task|dict(validation_folds=1))]:
            self.assertEqual(session.tools.invoke('confirm_task',{'task':invalid}).status,'error')
            self.assertIsNone(self.store.get('confirmed_task','task'))
        invalid=json.dumps(task|dict(target_columns=['invented']))
        self.assertFalse(session.tools.invoke('confirm_task',{'task':invalid}).data['valid'])
        self.assertIsNone(self.store.get('confirmed_task','task'))
        result=session.tools.invoke('confirm_task',{'task':json.dumps(task)})
        self.assertEqual(result.status,'ok',result)
        self.assertTrue(result.data['valid'])
        self.assertEqual(result.data,session.tools.invoke('confirm_task',{'task':task}).data)
        self.assertEqual(session.tools.invoke('confirm_task',{'task':json.dumps(task|dict(primary_metric='mae'))}).status,'error')
        self.assertEqual(session.deadline,deadline)

    def test_timeout_after_confirmation_keeps_findings_and_query_cache(self):
        parent=self
        class Client(OfflineLLM):
            def generate(self,role,instruction,payload,schema,tools=None):
                super().generate(role,instruction,payload,schema,tools)
                result=tools.invoke('query_csv',{'sql':'SELECT COUNT(*) AS n FROM train'}).data
                finding=tools.invoke('save_observation',dict(topic='size',description='Training rows',rows=180,
                    numbers={'n':180},source_ref=result['source_ref']))
                assert finding.status=='ok',finding
                parent.clock[0]+=301
                raise TimeoutError('API timeout')
        session=self.session(Client());report=session.run()
        self.assertEqual(report.status,'partial');self.assertEqual(len(report.observations),1)
        self.assertIsNotNone(self.store.get('confirmed_task','task'))
        deadline=session.deadline;calls=self.sandbox.calls
        resumed=self.session();self.assertEqual(resumed.deadline,deadline)
        self.assertEqual(resumed.csv.query_csv('SELECT COUNT(*) AS n FROM train')['rows'],[{'n':180}])
        self.assertEqual(self.sandbox.calls,calls)
        with self.assertRaisesRegex(ValueError,'deadline'):resumed.read_task()

    def test_timeout_without_confirmed_requirements_cannot_start_training(self):
        class Client(OfflineLLM):
            def generate(self,*args):raise TimeoutError('timeout before confirmation')
        with self.assertRaisesRegex(ValueError,'No validated task'):self.session(Client()).run()
        self.assertIsNone(self.store.get('confirmed_task','task'))

    def test_interrupted_scout_resumes_confirmed_task_and_original_window(self):
        class Client(OfflineLLM):
            def generate(self,*args):
                super().generate(*args)
                raise KeyboardInterrupt('Interrupted after confirmation')
        session=self.session(Client())
        with self.assertRaises(KeyboardInterrupt):session.run()
        self.clock[0]+=10
        resumed=self.session()
        self.assertEqual(resumed.deadline,session.deadline)
        self.assertEqual(resumed.run().status,'completed')

    def test_measured_numbers_are_checked_but_proposal_references_are_not(self):
        session=self.session()
        result=session.csv.query_csv('SELECT COUNT(*) AS n FROM train')
        saved=session.save_observation('size','Agent assessment',180,{'n':999},result['source_ref'])
        self.assertEqual(saved['numbers'],{'n':999})
        class Client(OfflineLLM):
            def generate(self,*args):
                super().generate(*args)
                return args[3](proposals=[dict(description='invented',evidence_ids=['missing'],expected_effect='gain',verification='metric')])
        report=self.session(Client()).run()
        self.assertEqual(report.status,'completed')
        self.assertEqual(len(report.proposals),1)
        self.assertEqual(report.proposals[0].evidence_ids,['missing'])
        self.assertIsNone(report.reason)

    def test_test_target_is_hidden_and_template_is_metadata_only(self):
        test=pd.read_csv(self.task.parent/'test.csv');test['target']=123
        test.to_csv(self.task.parent/'test.csv',index=False)
        session=self.session()
        with self.assertRaises(pl.exceptions.ColumnNotFoundError):session.csv.query_csv('SELECT target FROM test')
        with self.assertRaises(pl.exceptions.PolarsError):session.csv.query_csv('SELECT * FROM sample_submission')

    def test_follow_up_exposes_only_the_completed_round_training_rows(self):
        parent=self
        skill_payloads={}
        class Client(OfflineLLM):
            def generate(self,role,instruction,payload,schema,tools=None):
                if role=='reconnaissance_agent':
                    cards=payload['skills']
                    skill_payloads[payload['initial']]=[card['skill_id'] for card in cards]
                    parent.assertEqual(skill_payloads[payload['initial']],['reconnaissance_agent','polars_sql'])
                    parent.assertTrue(all(card['content'] and card['sha256'] for card in cards))
                result=super().generate(role,instruction,payload,schema,tools)
                if role=='reconnaissance_agent' and not payload['initial']:
                    query=tools.invoke('query_csv',{'sql':'SELECT COUNT(*) AS n FROM train'})
                    parent.assertEqual(query.status,'ok',query)
                    parent.assertEqual(query.data['rows'],[{'n':4}])
                    parent.assertNotIn('holdout',tools.invoke('list_tables',{}).data)
                return result
        pipeline=Pipeline(self.task,self.config,run_dir=self.root/'pipeline',client=Client())
        try:
            pipeline.inspect({})
            pipeline.store.put('split','r1',dict(train_indices=[0,1,2,3],holdout_indices=list(range(4,180))))
            train_path=pipeline.run_dir/'splits'/'r1'/'train.csv'
            train_path.parent.mkdir(parents=True)
            pipeline.frame.iloc[:4].to_csv(train_path,index=False)
            saved_train=train_path.read_bytes()
            with patch.object(pd.DataFrame,'to_csv',side_effect=AssertionError('Existing train split must be reused')):
                report=pipeline.scout('r1',{})
            self.assertEqual(report['status'],'completed',report)
            self.assertEqual(set(skill_payloads),{True,False})
            self.assertTrue((pipeline.run_dir/'splits'/'r1'/'train.csv').is_file())
            self.assertEqual(train_path.read_bytes(),saved_train)
            self.assertFalse((pipeline.run_dir/'reconnaissance'/'r1'/'train.csv').exists())
            follow=next(r for r in pipeline.store.list('csv_result') if 'rows' in r)
            self.assertEqual(follow['rows'],[{'n':4}])
        finally:
            pipeline.sandbox.close();pipeline.store.close();pipeline.checkpointer.close();pipeline.trace.close()


if __name__=='__main__':unittest.main()
