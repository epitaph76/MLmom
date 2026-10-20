from __future__ import annotations
import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from langgraph.graph import StateGraph,START,END
from mlmom.checkpoint import SQLiteCheckpointer
from mlmom.pipeline import Pipeline
from mlmom.pipeline_contracts import RunConfig,TaskContext,HypothesisPlan,Hypothesis
from mlmom.registry import installed_tools,SkillRegistry
from mlmom.inputs import resolve_inputs
from mlmom.splitting import make_split
from mlmom.evaluation import score_predictions,better
from mlmom.persistence import file_hash,ExperimentStore
from mlmom.tools import Workspace,verify_source
from tests.fakes import OfflineLLM

import mlmom
ROOT=Path(mlmom.__file__).resolve().parent.parent


def dataset(root,kind='regression'):
    root.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(42)
    X=rng.normal(size=(180,3))
    y=X[:,0]*3+X[:,1] if kind=='regression' else np.asarray(['a','b','c'])[np.arange(180)%3]
    train=pd.DataFrame(X,columns=['x','z','v']);train['id']=range(180);train['target']=y
    train.to_csv(root/'train.csv',index=False)
    test=train.drop(columns='target').iloc[:20].copy();test['id']+=1000;test.to_csv(root/'test.csv',index=False)
    test[['id']].assign(target=0 if kind=='regression' else 'a').to_csv(root/'sample_submission.csv',index=False)
    pd.DataFrame({'feature':train.columns,'role':['feature']*3+['id','target'],
                  'dtype':['numeric']*4+['numeric' if kind=='regression' else 'category']}).to_csv(root/'data_dictionary.csv',index=False)
    metric='rmse' if kind=='regression' else 'macro_f1'
    (root/'dataset_manifest.json').write_text(json.dumps(dict(train='train.csv',dictionaries=['data_dictionary.csv'],
        task_description='task.md',task_type=kind,target_columns=['target'],id_columns=['id'],primary_metric=metric)),encoding='utf-8')
    (root/'task.md').write_text(f'Predict `target` using {metric}.\n'
        'Paths: `missing/participants/train.csv`, `missing/participants/test.csv`.\n'
        'File `submission.csv`: CSV in UTF-8, comma delimiter, without index.\n'
        '```csv\nid,target\n```\nOne finite prediction per ID. No missing predictions or duplicate IDs. '
        'Row order must match test.csv.',encoding='utf-8')
    return root/'task.md'


def only_sklearn():return {'sklearn':installed_tools()['sklearn']}


class PipelineIntegrationTests(unittest.TestCase):
    def test_two_rounds_regression_and_multiclass_and_resume(self):
        for kind in ['regression','multiclass_classification']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temp:
                root=Path(temp); task=dataset(root/'input',kind)
                config=RunConfig(total_seconds=180,final_reserve_seconds=20,max_rounds=2,smoke_seconds=15,
                    prediction_seconds=10,experiment_seconds=20,reserve_cpus=0,reserve_memory_gb=0)
                client=OfflineLLM()
                with patch('mlmom.pipeline.installed_tools',only_sklearn):
                    result=Pipeline(task,config,run_dir=root/'run',client=client).run()
                self.assertEqual(result['status'],'completed',result)
                self.assertEqual(result['submission']['model_role'],'submission_refit',result)
                output=pd.read_csv(root/'input'/'submission.csv')
                test=pd.read_csv(root/'input'/'test.csv')
                self.assertEqual(output.id.tolist(),test.id.tolist())
                self.assertEqual(output.columns.tolist(),['id','target'])
                if kind=='regression':self.assertTrue(np.isfinite(output.target).all())
                else:self.assertLessEqual(set(output.target),{'a','b','c'})
                store=ExperimentStore(root/'run')
                splits=store.list('split');runs=store.list('experiment');evaluations=store.list('evaluation')
                self.assertEqual(len(splits),2)
                self.assertNotEqual(splits[0]['holdout_indices'],splits[1]['holdout_indices'])
                incumbent=[r for r in runs if r['incumbent']]
                self.assertEqual(len(incumbent),1)
                self.assertEqual(incumbent[0]['split_id'],splits[1]['split_id'])
                self.assertTrue(all(r['split_id']!='all-train' for r in evaluations))
                self.assertTrue(all(s['status']=='ready' for c in store.list('candidate') for s in c['skill_versions'].values()))
                self.assertTrue(all({'candidate_sdk','validation','metrics','preprocessing','sklearn'} <= set(c['skill_versions'])
                                    for c in store.list('candidate')))
                count=len(runs);store.close()
                with patch('mlmom.pipeline.installed_tools',only_sklearn):
                    again=Pipeline(task,config,run_dir=root/'run',client=OfflineLLM(fail=True),resume=True).run(resume=True)
                self.assertEqual(again['submission'],result['submission'])
                store=ExperimentStore(root/'run');self.assertEqual(len(store.list('experiment')),count);store.close()

    def test_failure_without_llm_does_not_create_fallback_submission(self):
        with tempfile.TemporaryDirectory() as temp:
            task=dataset(Path(temp)/'input')
            result=Pipeline(task,RunConfig(total_seconds=60,final_reserve_seconds=10),run_dir=Path(temp)/'run',client=OfflineLLM(fail=True)).run()
            self.assertEqual(result['status'],'failed')
            self.assertFalse((task.parent/'submission.csv').exists())


class ComponentTests(unittest.TestCase):
    def test_workspace_escape_patch_and_forbidden_import(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace=Workspace(Path(temp))
            with self.assertRaises(ValueError):workspace.write_file('../escape.py','x=1')
            workspace.write_file('candidate.py','import os\n')
            self.assertFalse(verify_source(workspace.root).valid)
            workspace.write_file('candidate.py','x=1\n')
            workspace.apply_patch('candidate.py','x=1','x=2')
            self.assertEqual(workspace.read_file('candidate.py'),'x=2\n')
            with self.assertRaises(ValueError):workspace.apply_patch('candidate.py','missing','x')

    def test_ready_skills_and_registry_no_false_availability(self):
        registry=SkillRegistry(ROOT/'skills')
        loaded=registry.load(['sklearn','metrics'],'regression')
        self.assertTrue(all(s['status']=='ready' and len(s['sha256'])==64 for s in loaded))
        self.assertFalse(any(t.available for t in installed_tools().values()))

    def test_checkpoint_sync_async_resume_and_pending_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'checkpoints.db'
            saver=SQLiteCheckpointer(str(path))
            graph=StateGraph(dict);graph.add_node('first',lambda s:dict(n=s.get('n',0)+1));graph.add_node('second',lambda s:dict(n=s['n']+1))
            graph.add_edge(START,'first');graph.add_edge('first','second');graph.add_edge('second',END)
            compiled=graph.compile(checkpointer=saver,interrupt_after=['first'])
            config={'configurable':{'thread_id':'test'}}
            self.assertEqual(compiled.invoke({'n':0},config)['n'],1)
            saver.close();saver=SQLiteCheckpointer(str(path))
            resumed=graph.compile(checkpointer=saver)
            self.assertEqual(resumed.invoke(None,config)['n'],2)
            self.assertGreaterEqual(len(list(saver.list(config))),3)
            self.assertIsNotNone(asyncio.run(saver.aget_tuple(config)))
            saver.delete_thread('test');self.assertIsNone(saver.get_tuple(config));saver.close()


if __name__=='__main__':unittest.main()
