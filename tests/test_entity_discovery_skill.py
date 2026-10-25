"""Executable linkage diagnostics and delivery of the research procedure."""
from contextlib import ExitStack
from itertools import combinations
from pathlib import Path
import re
import tempfile
import time
import unittest

import pandas as pd

from mlmom.csv_tools import csv_worker
from mlmom.feature_search import prepare_initial_proposal, run_feature_session
from mlmom.resources import Budget
from tests import test_feature_session as session_fixtures


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/'skills/feature_agent/feature_agent/REFERENCE.md'


def observations():
    return pd.DataFrame(dict(
        anchor_id=['A','A','A','B','B','C','C','D',None,'E'],
        region=['N','N','N','N','N','S','S','N','N',None],
        category=['c','c','c','c','c','d','d','c','c','c'],
        size_bucket=[10,10,11,10,10,7,7,12,10,11],
        period=[1,2,3,1,2,1,2,3,4,1],
        measure=[10.,10.1,10.2,20.,20.,7.,7.05,25.,11.,3.]))


def run_query(sql, frame, root):
    root.mkdir(parents=True)
    frame.to_csv(root/'train.csv',index=False)
    return csv_worker(dict(operation='query',sql=sql,cache_dir=str(root/'cache'),
        tables={'train':dict(path=str(root/'train.csv'),role='feature_training',columns=list(frame))}),root)


def pair_truth(frame):
    """Independent tiny oracle; query implementation uses aggregated counts instead."""
    truth, accepted = set(), set()
    for a,b in combinations(frame.index,2):
        left,right=frame.loc[a],frame.loc[b]
        if pd.isna(left.anchor_id) or pd.isna(right.anchor_id):continue
        if left.anchor_id==right.anchor_id:truth.add((a,b))
        key=['region','category','size_bucket']
        if left[key].notna().all() and right[key].notna().all() and all(left[k]==right[k] for k in key):
            accepted.add((a,b))
    return truth, accepted


class EntityDiscoverySkillTests(unittest.TestCase):
    def test_documented_sql_measures_invariance_collision_cost_and_link_errors(self):
        queries=re.findall(r'```sql\n(.*?)\n```',REFERENCE.read_text(encoding='utf-8'),re.S)
        self.assertEqual(len(queries),3,'Every executable diagnostic needs a result check')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);frame=observations()
            results=[run_query(q,frame,root/str(i)) for i,q in enumerate(queries)]
            for result in results:
                self.assertEqual(result['status'],'completed')
                self.assertFalse(result['truncated'])
            self.assertEqual([r['anchor_id'] for r in results[0]['rows']],['A','B','C'])
            self.assertEqual([r['category_levels'] for r in results[0]['rows']],[1,1,1])
            self.assertEqual([r['observations'] for r in results[0]['rows']],[3,2,2])
            self.assertAlmostEqual(results[0]['rows'][0]['max_measure']-results[0]['rows'][0]['min_measure'],.2)
            self.assertEqual(results[1]['rows'],[dict(key_groups=4,covered_rows=9,repeated_rows=7,
                cross_period_rows=7,conflict_rows=5,largest_block=5,candidate_pairs=11.)])
            quality=results[2]['rows'][0]
            truth,accepted=pair_truth(frame)
            self.assertEqual(quality,dict(proposed_pairs=len(accepted),
                true_positive_pairs=len(truth & accepted),true_pairs=len(truth)))
            self.assertAlmostEqual(quality['true_positive_pairs']/quality['proposed_pairs'],3/7)
            self.assertAlmostEqual(quality['true_positive_pairs']/quality['true_pairs'],.6)
            # A perfectly stable category still merges independently known A and B.
            self.assertEqual(len(accepted-truth),4)
            self.assertEqual(len(truth-accepted),2)

    def test_missing_keys_stay_in_recall_denominator_and_absent_truth_is_not_accuracy(self):
        query=re.findall(r'```sql\n(.*?)\n```',REFERENCE.read_text(encoding='utf-8'),re.S)[2]
        frame=observations();frame.loc[0,'region']=None
        with tempfile.TemporaryDirectory() as tmp:
            result=run_query(query,frame,Path(tmp)/'missing')['rows'][0]
            truth,accepted=pair_truth(frame)
            self.assertEqual(result['true_pairs'],5)
            self.assertEqual(result['true_positive_pairs'],len(truth & accepted))
            self.assertEqual(result['proposed_pairs'],len(accepted))
            frame['anchor_id']=None
            result=run_query(query,frame,Path(tmp)/'unknown')['rows'][0]
            self.assertEqual(result,dict(proposed_pairs=0,true_positive_pairs=0,true_pairs=0))
            self.assertNotIn('precision',result)
            self.assertNotIn('recall',result)

    def test_both_production_proposal_phases_receive_the_full_reference_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as cleanup:
            p,base,split,path=session_fixtures.SessionAccountingTests().harness(Path(tmp),fail=True)
            cleanup.callback(p.store.close)
            p.config.feature_max_configurations=2
            p.budget=Budget(p.config,time.time())
            p.store.put('split','r1',split,immutable=True)
            expected=p.skills.load([],'regression',role='feature_agent',include_references=True)
            captured=[]
            class Client:
                def generate(self,role,instruction,payload,schema,tools=None):
                    if role=='planner':
                        return schema(feature_set_id=payload['feature_set']['feature_set_id'],
                            reference_feature_set_id=payload['reference_feature_set_id'],rationale='Paired diagnostic')
                    captured.append(payload['skill_instructions'])
                    if len(captured)==1:
                        return schema(action='add',rationale='Measure a supported feature family',
                            features=[dict(name='product',op='product',columns=['x0','x1'])])
                    return schema(action='stop',rationale='Diagnostic complete')
            p.client=Client()
            prepare_initial_proposal(p,{})
            self.assertIsNone(p.store.get('feature_proposal_error','early'))
            run_feature_session(p,base,split,path)
            self.assertEqual(len(captured),2)
            for cards in captured:
                self.assertEqual(cards,expected)
                self.assertTrue(cards[0]['content'].endswith(REFERENCE.read_text(encoding='utf-8')))
                self.assertLessEqual(sum(len(c['content']) for c in cards),24000)


if __name__=='__main__':unittest.main()
