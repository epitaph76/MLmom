"""Role boundaries, fold-local transforms, checkpoint control and advisory evidence."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd

from mlmom.hpo import run_search
from mlmom.pipeline_contracts import TrainingCheckpoint,TrainingDecision,ResultReview
from mlmom.registry import SkillRegistry,ROLE_DEFAULTS
from mlmom.sdk import FitContext,load_module
from mlmom.tools import Workspace
from mlmom.training_control import decide,validate_decision,worker_checkpoint
from mlmom.research import data_observations,error_slices,review_result
from tests.fakes import PREPROCESSING_CODE,MODEL_CODE

ROOT=Path(__file__).resolve().parents[1]


def checkpoint(**updates):
    values=dict(checkpoint_id='batch-5',stage='batch',plan=dict(search_space={
        'depth':dict(type='int',low=1,high=4),
        'rate':dict(type='float',low=.01,high=.5),
        'kind':dict(type='categorical',choices=['a','b'])}),
        remaining_seconds=10,remaining_trials=3,default_action='continue')
    return TrainingCheckpoint(**(values|updates))


class RoleSkillTests(unittest.TestCase):
    def test_review_distinguishes_comparable_gain_missing_evidence_and_unknown(self):
        registry=SkillRegistry(ROOT/'skills')
        cards=registry.load([], 'regression',role='result_validator')
        class Reviewer:
            def generate(self,role,instruction,payload,schema,tools=None):
                if payload.get('invent_reference'):
                    return schema(implementation='matches',effect='supported',explanation='gain',evidence_refs=['invented'])
                return schema(implementation='matches',effect='supported',explanation='gain',evidence_refs=['r2-score'])
        for comparable,expected in [(True,'supported'),(False,'supported')]:
            evaluation=dict(score_kind='outer_holdout',dataset_hash='data',split_id='r2',metric='rmse',direction='minimize',metric_options={},weighting='unweighted',score=.9)
            reference=dict(evaluation,score=1) if comparable else None
            report=review_result(Reviewer(),dict(evaluation=evaluation,execution_report={'status':'passed'},comparable_reference=reference,evidence_refs=['r2-score']),cards,'review',1)
            self.assertEqual(report.effect,expected)
            self.assertEqual(report.implementation,'matches')
        report=review_result(Reviewer(),dict(invent_reference=True,evidence_refs=['r2-score']),cards,'review',1)
        self.assertEqual((report.implementation,report.effect),('matches','supported'))

    def test_each_role_cannot_read_another_role_cards(self):
        registry=SkillRegistry(ROOT/'skills')
        for role in ROLE_DEFAULTS:
            visible={s['skill_id'] for s in registry.list(role=role)}
            for key,owner in registry.owners.items():
                if owner not in {'shared',role}:
                    self.assertNotIn(key,visible)
                    with self.assertRaisesRegex(ValueError,'not available'):
                        registry.load([key],'regression',role=role)
        placeholders=[s for s in registry.list() if s['status']=='placeholder']
        self.assertEqual(placeholders, [])
        for role in ['planner','hypothesis_generator','result_validator']:
            loaded=registry.load([], 'regression',role=role)
            self.assertIn(role,[card['skill_id'] for card in loaded])
            self.assertTrue(all(card['status']=='ready' for card in loaded))

    def test_writes_are_checked_after_path_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            model=Workspace(Path(tmp),writable=lambda name:name!='preprocessing.py')
            for name in ['preprocessing.py','./preprocessing.py']:
                with self.assertRaisesRegex(ValueError,'another coding role'):model.write_file(name,'x=1')
            pre=Workspace(Path(tmp),writable=lambda name:name=='preprocessing.py')
            with self.assertRaises(ValueError):pre.write_file('candidate.py','x=1')


class SeparatePreprocessingTests(unittest.TestCase):
    def test_preprocessing_repair_reuses_model_when_contract_is_unchanged(self):
        from mlmom.coding import build_coding_graph
        from mlmom.pipeline_contracts import RunConfig
        from tests.test_planning_limits import case,Store,ClockBudget,Sandbox,Coder
        class Client(Coder):
            pre_calls=0
            def generate(self,*args):
                if args[0]=='preprocessing_agent':
                    self.pre_calls+=1
                    if args[3].__name__=='CodePatch':
                        previous=args[2]['previous']
                        return args[3](patches=[dict(path='preprocessing.py',sha256=previous['sha256'],
                            old_text='# revision 1',new_text='# revision 2')])
                    return args[3](source=PREPROCESSING_CODE+'\n# revision 1\n',
                        spec=dict(description='Fixed interface',output='dense'))
                return super().generate(*args)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);budget=ClockBudget();client=Client(budget,1);sandbox=Sandbox(budget)
            context,tools,_,plan=case()
            args=dict(workspace=root/'work',store=Store(root),plan=plan,plan_ref=root/'plan.json',
                      skill_data=[],tools=tools,context=context,sandbox=sandbox,budget=budget,
                      config=RunConfig(timeouts_enabled=True),round_id='r1')
            first=build_coding_graph(client,**args).invoke({'payload':{}})
            self.assertTrue(first['valid'],first)
            result=build_coding_graph(client,**args,runtime_feedback='preprocessing.py: repair a real failure').invoke({'payload':{}})
            self.assertTrue(result['valid'],result)
            self.assertEqual(client.pre_calls,2)
            self.assertEqual(client.calls,1)
            self.assertEqual(sandbox.calls,[])

    def test_fresh_state_train_only_unseen_categories_empty_columns_and_portability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);code=root/'code';code.mkdir()
            (code/'preprocessing.py').write_text(PREPROCESSING_CODE,encoding='utf-8')
            (code/'candidate.py').write_text(MODEL_CODE,encoding='utf-8')
            module=load_module(code)
            X=pd.DataFrame(dict(n=[1.,np.nan,3.,4.],empty=[np.nan]*4,cat=['a','b','a',None]))
            valid=pd.DataFrame(dict(n=[10000.],empty=[100.],cat=['validation_only']))
            ctx=FitContext('regression','rmse','minimize',42,1,2,20,validation_X=valid,validation_y=pd.Series([5.]))
            obj=module.create_candidate(dict(model_params={'n_estimators':3}))
            obj.fit(X,pd.Series([1.,2.,3.,4.]),ctx)
            first=obj.preprocessor
            self.assertEqual(first.transformer.transformers_[0][1].statistics_[0],3.)
            categories=first.transformer.transformers_[1][1].named_steps['encode'].categories_[0]
            self.assertNotIn('validation_only',categories)
            predictions=obj.predict(valid[['cat','empty','n']],{'kind':'value'})
            artifact=root/'artifact';artifact.mkdir();obj.save(artifact)
            moved=root/'moved';shutil.copytree(artifact,moved)
            restored=module.load_candidate(moved)
            np.testing.assert_array_equal(predictions,restored.predict(valid,{'kind':'value'}))
            obj.fit(X.assign(n=100.),pd.Series([1.,2.,3.,4.]),ctx)
            self.assertIsNot(obj.preprocessor,first)
            self.assertEqual(first.transformer.transformers_[0][1].statistics_[0],3.)
            self.assertEqual(obj.preprocessor.transformer.transformers_[0][1].statistics_[0],100.)

    def test_groups_pass_to_preprocessor_without_validation_labels(self):
        pre='''MLMOM_PREPROCESSOR_V1 = True
class Pre:
    def fit_transform(self,X,y,context):
        assert context.validation_X is None and context.validation_y is None
        assert list(context.groups)==[1,1,2,2]
        self.groups=list(context.groups)
        return X
    def transform(self,X):return X
def create_preprocessor(config):return Pre()
'''
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'candidate.py').write_text(MODEL_CODE,encoding='utf-8')
            (root/'preprocessing.py').write_text(pre,encoding='utf-8')
            ctx=FitContext('regression','rmse','minimize',1,1,2,10,groups=np.array([1,1,2,2]),
                validation_X=pd.DataFrame({'x':[9.]}),validation_y=np.array([99.]))
            obj=load_module(root).create_candidate({'model_params':{'n_estimators':2}})
            obj.fit(pd.DataFrame({'x':[1.,2.,3.,4.]}),pd.Series([1.,2.,3.,4.]),ctx)
            self.assertEqual(obj.preprocessor.groups,[1,1,2,2])

    def test_candidate_sibling_imports_do_not_cross_contaminate(self):
        with tempfile.TemporaryDirectory() as tmp:
            modules=[]
            for index in [1,2]:
                root=Path(tmp)/str(index);root.mkdir()
                (root/'helper.py').write_text(f'VALUE={index}\n',encoding='utf-8')
                (root/'candidate.py').write_text('from helper import VALUE\ndef create_candidate(config):return VALUE\ndef load_candidate(path):return VALUE\n',encoding='utf-8')
                modules.append(load_module(root))
            self.assertEqual([m.create_candidate({}) for m in modules],[1,2])


class TrainingControlTests(unittest.TestCase):
    def test_suggestions_obey_types_bounds_and_remaining_trials(self):
        cp=checkpoint()
        validate_decision(TrainingDecision(action='continue',reason='test',next_params=[dict(depth=2,rate=.1,kind='a')]),cp)
        for params in [dict(depth=True),dict(depth=2.),dict(depth=99),dict(rate=float('nan')),dict(kind='c'),dict(n_jobs=5)]:
            with self.subTest(params=params),self.assertRaises(ValueError):
                validate_decision(TrainingDecision(action='continue',reason='test',next_params=[params]),cp)
        with self.assertRaises(ValueError):validate_decision(TrainingDecision(action='continue',reason='test'),checkpoint(remaining_trials=0))

    def test_failure_uses_default_and_worker_timeout_is_replayed(self):
        class Broken:
            def generate(self,*args):raise RuntimeError('offline')
        cp=checkpoint(default_action='refit')
        self.assertEqual(decide(Broken(),cp,[]).action,'refit')
        with tempfile.TemporaryDirectory() as tmp:
            first=worker_checkpoint(tmp,cp,seconds=0)
            second=worker_checkpoint(tmp,cp,seconds=1)
            self.assertEqual(first,second)
            self.assertEqual(first.action,'refit')

    def test_control_at_initial_fit_and_batches_enqueues_params_and_resumes_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls=[];values=[]
            def objective(params,number,seconds):
                values.append(params['depth']);return float(params['depth'])
            def initial(data):
                calls.append(('initial',len(data['trials'])))
                return TrainingDecision(action='continue',reason='try depth four',next_params=[dict(depth=4)])
            def batch(data):
                calls.append(('batch',len(data['trials'])))
                return TrainingDecision(action=data['default_action'],reason='programmatic')
            kwargs=dict(directory=tmp,identity={'code':'fixed'},space={'depth':dict(type='int',low=1,high=4)},
                direction='maximize',seed=42,max_trials=4,batch_size=2,seconds=20,gain_threshold=0,
                checkpoint=batch,trial_checkpoint=initial)
            state=run_search(objective,**kwargs)
            self.assertEqual(calls,[('initial',1),('batch',2),('batch',4)])
            self.assertEqual(values[1],4)
            deadline=state['deadline']
            again=run_search(objective,**kwargs)
            self.assertEqual(len(values),4)
            self.assertEqual(again['deadline'],deadline)

    def test_stop_after_first_fit_prevents_remaining_trials(self):
        with tempfile.TemporaryDirectory() as tmp:
            state=run_search(lambda *args:1.,directory=tmp,identity={},space={'depth':dict(type='int',low=1,high=4)},
                direction='maximize',seed=1,max_trials=10,batch_size=5,seconds=10,
                trial_checkpoint=lambda data:TrainingDecision(action='stop',reason='enough'))
            self.assertEqual(len(state['trials']),1)
            self.assertEqual(state['stop_reason'],'training_agent_stop')

    def test_pending_checkpoint_survives_interruption_and_keeps_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            def interrupted(data):raise KeyboardInterrupt('Controller interrupted')
            kwargs=dict(directory=tmp,identity={'case':'resume'},space={'depth':dict(type='int',low=1,high=4)},
                direction='maximize',seed=1,max_trials=3,batch_size=2,seconds=20)
            with self.assertRaises(KeyboardInterrupt):run_search(lambda *args:1.,trial_checkpoint=interrupted,**kwargs)
            saved=json.loads(next(Path(tmp).glob('*.json')).read_text(encoding='utf-8'))
            self.assertEqual(saved['pending_control']['kind'],'trial')
            seen=[]
            def resumed(data):
                seen.append(len(data['trials']))
                return TrainingDecision(action='stop',reason='Recovered checkpoint')
            result=run_search(lambda *args:self.fail('Must not run another trial'),trial_checkpoint=resumed,**kwargs)
            self.assertEqual(seen,[1])
            self.assertEqual(result['deadline'],saved['deadline'])
            self.assertEqual(len(result['trials']),1)

    def test_failure_checkpoint_occurs_after_failed_trial_is_committed(self):
        with tempfile.TemporaryDirectory() as tmp:
            seen=[]
            def objective(params,number,seconds):
                if number==1:raise ValueError('training failure')
                return 1.
            def control(data):
                seen.append(data['stage'])
                if data['stage']=='failure':
                    self.assertEqual(data['trials'][-1]['status'],'fail')
                    return TrainingDecision(action='stop',reason='Stop failed run')
                return TrainingDecision(action='continue',reason='Proceed')
            result=run_search(objective,directory=tmp,identity={},space={'depth':dict(type='int',low=1,high=4)},
                direction='maximize',seed=1,max_trials=5,batch_size=5,seconds=20,trial_checkpoint=control)
            self.assertEqual(seen,['initial_fit','failure'])
            self.assertEqual(len(result['trials']),2)


class ResearchTests(unittest.TestCase):
    def test_evidence_numbers_and_tail_boundaries_use_training(self):
        train=pd.DataFrame({'x':[1.,1.,np.nan,4.],'cat':['a','a','b','c'],'target':[1.,2.,3.,4.]})
        context=NS(feature_columns=['x','cat'],task=NS(target_columns=['target'],task_type='regression'))
        observations=data_observations(train,context,'profile.json')
        self.assertEqual(observations[0].numbers['duplicate_rows'],1)
        self.assertEqual(observations[0].numbers['conflicting_groups'],1)
        y=pd.Series([100.,200.]);X=pd.DataFrame({'x':[1.,1000.],'cat':['a','new']})
        slices=error_slices(y,np.array([99.,190.]),X,train[['x','cat']],train.target,'regression')
        tail=next(s for s in slices if s['topic']=='target_tail' and '>=' in s['slice'])
        self.assertEqual(tail['rows'],2)
        self.assertIn('3.97',tail['slice'])

    def test_result_failure_is_advisory_inconclusive(self):
        class Broken:
            def generate(self,*args):raise ValueError('no response')
        result=review_result(Broken(),{},[],'review',1)
        self.assertEqual(result.effect,'inconclusive')
        self.assertEqual(result.implementation,'unknown')


if __name__=='__main__':unittest.main()
