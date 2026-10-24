"""Feature search sequencing, failure accounting, resume and one outer selection."""
import copy
from contextlib import ExitStack
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import pandas as pd

from mlmom.pipeline import Pipeline
from mlmom.pipeline_contracts import RunConfig, Champion, CandidateVersion, ExperimentRun, EvaluationResult, SplitSpec
from mlmom.feature_contracts import FeatureSet, feature_hash
from mlmom.feature_contracts import FeatureSession
from mlmom.feature_search import (run_feature_session, complete_feature_session, paired_evidence,
                                 select_feature_baseline)
from mlmom.persistence import ExperimentStore, atomic_json
from mlmom.registry import SkillRegistry
from tests.fakes import OfflineLLM
from tests.test_pipeline import dataset, only_sklearn
from tests import test_feature_engineering as feature_fixture
from tests.test_feature_engineering import LINEAR_CODE
from mlmom.worker import train
from mlmom.splitting import inner_protocol, feature_context


class FeatureClient(OfflineLLM):
    def complete(self,messages):
        task=json.loads(super().complete(messages))
        task['evidence']=['SENTINEL_FULL_TRAIN_FACT']
        return json.dumps(task)
    def generate(self,role,instruction,payload,schema,tools=None):
        if role=='feature_agent':
            self.calls.append((role,copy.deepcopy(payload)))
            if payload['history']:
                return schema(action='stop',rationale='One useful semantic ratio is enough')
            return schema(action='add',rationale='Normalize x by z',features=[dict(name='ratio',op='ratio',columns=['x','z'])])
        result=super().generate(role,instruction,payload,schema,tools)
        if role=='planner' and schema.__name__=='PlanBatch':
            for plan in result.plans:
                plan.model_params={'fit_intercept':True}
                plan.search_space={'fit_intercept':{'type':'categorical','choices':[True,False]}}
        if role=='coding_agent':return schema(files={'candidate.py':LINEAR_CODE})
        return result


class FeaturePipelineTests(unittest.TestCase):
    def test_feature_agent_runs_again_on_rotated_split_and_retains_accepted_features(self):
        class ContinuingClient(FeatureClient):
            def generate(self,role,instruction,payload,schema,tools=None):
                if role=='feature_agent' and payload['current_feature_set']['features']:
                    self.calls.append((role,copy.deepcopy(payload)))
                    return schema(action='stop',rationale='Keep the verified ratio in this cycle')
                return super().generate(role,instruction,payload,schema,tools)
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as cleanup:
            root=Path(tmp);task=dataset(root/'input')
            frame=pd.read_csv(task.parent/'train.csv');rng=np.random.default_rng(21)
            frame['x']=rng.uniform(1,10,len(frame));frame['z']=rng.uniform(1,10,len(frame))
            frame['target']=frame.x/frame.z
            frame.to_csv(task.parent/'train.csv',index=False)
            test=frame.drop(columns='target').iloc[:20].copy();test.id+=1000
            test.to_csv(task.parent/'test.csv',index=False)
            config=RunConfig(max_rounds=2,hpo_enabled=False,diagnostic_seconds=0,
                reserve_cpus=0,reserve_memory_gb=0,max_workers=1)
            client=ContinuingClient()
            with patch('mlmom.pipeline.installed_tools',only_sklearn):
                result=Pipeline(task,config,run_dir=root/'run',client=client).run()
            self.assertEqual(result['status'],'completed',result)
            store=ExperimentStore(root/'run');cleanup.callback(store.close)
            first=store.get('feature_session','initial');second=store.get('feature_session','r2')
            self.assertEqual(first['reason'],'outer_improvement',first)
            self.assertEqual(second['status'],'completed',second)
            self.assertEqual(second['baseline_feature_set_id'],first['best_feature_set_id'])
            self.assertEqual(second['best_feature_set_id'],first['best_feature_set_id'])
            self.assertNotEqual(first['train_hash'],second['train_hash'])
            self.assertNotEqual(first['protocol_hash'],second['protocol_hash'])
            self.assertEqual(len(result['feature_search_sessions']),2)
            self.assertEqual(result['feature_search']['round_id'],'r2')
            # Once after r1's accepted proposal and again on r2's new folds.
            retained=[p for role,p in client.calls if role=='feature_agent' and p['current_feature_set']['features']]
            self.assertEqual(len(retained),2)
            self.assertEqual(retained[-1]['history'],[])
            before=len(client.calls)
            with patch('mlmom.pipeline.installed_tools',only_sklearn):
                Pipeline(task,config,run_dir=root/'run',client=client,resume=True).run(resume=True)
            self.assertEqual(len(client.calls),before)

    def test_first_csv_features_hpo_final_refit_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as cleanup:
            root=Path(tmp);task=dataset(root/'input')
            frame=pd.read_csv(task.parent/'train.csv')
            rng=np.random.default_rng(21)
            frame['x']=rng.uniform(1,10,len(frame));frame['z']=rng.uniform(1,10,len(frame))
            frame['target']=frame.x/frame.z
            frame.to_csv(task.parent/'train.csv',index=False)
            test=frame.drop(columns='target').iloc[:20].copy();test.id+=1000
            test.to_csv(task.parent/'test.csv',index=False)
            config=RunConfig(max_rounds=1,hpo_enabled=True,hpo_max_trials=2,diagnostic_seconds=0,
                reserve_cpus=0,reserve_memory_gb=0,max_workers=1)
            client=FeatureClient()
            with patch('mlmom.pipeline.installed_tools',only_sklearn):
                pipeline=Pipeline(task,config,run_dir=root/'run',client=client)
                result=pipeline.run()
            self.assertEqual(result['status'],'completed',result)
            store=ExperimentStore(root/'run');cleanup.callback(store.close)
            session=store.get('feature_session','initial')
            self.assertEqual(session['status'],'completed',session)
            self.assertEqual(session['reason'],'outer_improvement',session)
            self.assertEqual(session['attempts'],1)
            self.assertEqual(len(store.list('feature_registry')),1)
            feature_id=session['best_feature_set_id']
            outer=store.get('evaluation',session['outer_experiment_id'])
            self.assertEqual(outer['score_kind'],'outer_holdout')
            self.assertLess(outer['score'],1e-10)
            self.assertEqual(len([r for r in store.list('evaluation') if r['experiment_id']==session['outer_experiment_id']]),1)
            hpo=[r for r in store.list('experiment') if r['study_ref']]
            self.assertTrue(hpo)
            self.assertTrue(all(r['model_spec']['feature_set_hash'] for r in hpo))
            features=[p for role,p in client.calls if role=='feature_agent']
            self.assertEqual(len(features),2)
            for payload in features:
                serialized=json.dumps(payload)
                self.assertNotIn('outer_holdout',serialized)
                self.assertNotIn('model_inspection',serialized)
                self.assertNotIn('SENTINEL_FULL_TRAIN_FACT',serialized)
                self.assertNotIn('evaluation',payload)
                self.assertEqual(payload['train_profile']['rows'],144)
            self.assertEqual(result['submission']['model_role'],'submission_refit')
            predictions=pd.read_csv(task.parent/'submission.csv')
            np.testing.assert_allclose(predictions.target,test.x/test.z,atol=1e-10)
            events=[r[0] for r in store.db.execute('SELECT kind FROM events ORDER BY seq')]
            self.assertEqual(len(store.list('feature_evidence')),1)
            first_calls=len(client.calls)
            with patch('mlmom.pipeline.installed_tools',only_sklearn):
                resumed=Pipeline(task,config,run_dir=root/'run',client=client,resume=True)
                again=resumed.run(resume=True)
            self.assertEqual(again['status'],'completed')
            self.assertEqual(first_calls,len(client.calls))

    def test_old_session_disables_new_branch_and_rejects_explicit_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);task=dataset(root/'input')
            config=RunConfig(max_rounds=1,hpo_enabled=False,diagnostic_seconds=0,feature_search_enabled=False)
            with patch('mlmom.pipeline.installed_tools',only_sklearn):
                initial=Pipeline(task,config,run_dir=root/'run',client=OfflineLLM())
                initial.sandbox.close();initial.checkpointer.close();initial.store.close();initial.trace.close()
            session_path=root/'run/session.json'
            session=json.loads(session_path.read_text());session['config'].pop('feature_search_enabled')
            atomic_json(session_path,session)
            supplied=RunConfig(max_rounds=1,hpo_enabled=False,diagnostic_seconds=0)
            with patch('mlmom.pipeline.installed_tools',only_sklearn):
                resumed=Pipeline(task,supplied,run_dir=root/'run',client=OfflineLLM(),resume=True)
                self.assertFalse(resumed.config.feature_search_enabled)
                resumed.sandbox.close();resumed.checkpointer.close();resumed.store.close();resumed.trace.close()
                with self.assertRaisesRegex(ValueError,'same task and configuration'):
                    Pipeline(task,supplied.model_copy(update={'feature_search_enabled':True}),
                             run_dir=root/'run',client=OfflineLLM(),resume=True)


class SessionAccountingTests(unittest.TestCase):
    def harness(self, root, *, fail=True, interrupt=False):
        frame,context,plan,code,_,request=feature_fixture.FeatureWorkerTests().request(root)
        request['outer_train_indices']=list(range(len(frame)))
        protocol=inner_protocol(frame,feature_context(context),seed=43,outer_split_id='r1',
            source_indices=request['outer_train_indices'])
        request['feature_protocol']=protocol
        original=root/'original';original.mkdir()
        result=train(request,original)
        plan_ref=root/'baseline-plan.json';atomic_json(plan_ref,plan)
        Path(context.files.task).write_text('Test safe features',encoding='utf-8')
        candidate=CandidateVersion(candidate_id='baseline',hypothesis_id='h1',version=1,
            code_dir=str(code),code_hash=request['code_hash'],plan_ref=str(plan_ref),verification=dict(valid=True))
        run=ExperimentRun.model_validate(dict({k:v for k,v in result.items() if k in ExperimentRun.model_fields},
            experiment_id='baseline',round_id='r1',candidate_id='baseline',split_id='r1'))
        baseline=Champion(candidate=candidate,plan=plan,run=run,
            evaluation=EvaluationResult(experiment_id='baseline',split_id='r1',metric='rmse',direction='minimize',score=999))
        store=ExperimentStore(root/'run')
        store.put('preprocessing',plan_ref.stem,dict(spec=dict(description='Base',output='dense')))
        class Client:
            calls=0
            def generate(self,role,instruction,payload,schema,tools=None):
                if role=='planner':return schema(feature_set_id=payload['feature_set']['feature_set_id'],
                    reference_feature_set_id=payload['reference_feature_set_id'],rationale='Fixed paired test')
                specs=[('product',['x0','x1']),('ratio',['x0','x1']),('difference',['x0','x1']),
                       ('log',['x0']),('log',['x1']),('ratio',['x1','x0'])]
                op,columns=specs[self.calls];self.calls+=1
                return schema(action='add',rationale='New related transformation',
                    features=[dict(name='f'+str(self.calls),op=op,columns=columns)])
        class Sandbox:
            fits=0
            def run(self,req,directory,**limits):
                if req['action']=='feature_profile':
                    return dict(status='completed',profile=dict(rows=len(frame),columns=[],residual_slices=[]))
                if Path(directory).name=='reference':return copy.deepcopy(result)
                self.fits+=1
                output=dict(status='failed',error='Injected backend failure') if fail else copy.deepcopy(result)
                if interrupt and self.fits==6:
                    atomic_json(Path(directory)/'result.json',output)
                    raise KeyboardInterrupt('Controller stopped after the last completed worker')
                return output
        pipeline=NS(config=RunConfig(),context=context,frame=frame,store=store,run_dir=store.root,
            files=context.files,client=Client(),sandbox=Sandbox(),
            skills=SkillRegistry(Path(__file__).resolve().parents[1]/'skills'),
            budget=NS(snapshot=lambda:NS(research_seconds=10000)),
            sanitized_context=lambda path:context.model_dump(mode='json'))
        split=SplitSpec(split_id='r1',round_id='r1',dataset_hash=context.dataset_hash,seed=42,
            strategy='holdout',train_indices=list(range(len(frame))),holdout_indices=[])
        return pipeline,baseline,split,Path(request['train_path'])

    def test_same_formula_is_retested_in_new_cycle_without_reusing_old_results(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as cleanup:
            p,base,split,path=self.harness(Path(tmp))
            cleanup.callback(p.store.close)
            p.config.feature_max_configurations=1
            original=p.sandbox.run
            references=[]
            def run(request,directory,**limits):
                if Path(directory).name=='reference':
                    references.append(request['outer_split_id'])
                    return train(dict(request,worker_seconds=60,worker_cpus=1,
                                      worker_memory_gb=2,timeouts_enabled=False),Path(directory))
                return original(request,directory,**limits)
            p.sandbox.run=run
            run_feature_session(p,base,split,path)
            first=p.store.list('feature_evidence')[0]
            first_definition=p.store.get('feature_set',first['feature_set_id'])
            p.client.calls=0
            generate=p.client.generate
            def renamed(role,instruction,payload,schema,tools=None):
                proposal=generate(role,instruction,payload,schema,tools)
                if role=='feature_agent':proposal.features[0].name='renamed'
                return proposal
            p.client.generate=renamed
            second_split=split.model_copy(update=dict(round_id='r2',split_id='r2-new',seed=97))
            run_feature_session(p,base,second_split,path)
            evidence=p.store.list('feature_evidence')
            self.assertEqual(references,['r1','r2-new'])
            self.assertEqual(len(evidence),2,evidence)
            self.assertEqual(evidence[0]['feature_set_id'],evidence[1]['feature_set_id'])
            self.assertEqual(p.store.get('feature_set',first['feature_set_id']),first_definition)
            self.assertNotEqual(evidence[0]['protocol_hash'],evidence[1]['protocol_hash'])
            self.assertNotEqual(evidence[0]['result_ref'],evidence[1]['result_ref'])
            self.assertEqual(p.store.get('feature_session','r2')['attempts'],1)
            self.assertIsNotNone(p.store.get('feature_proposal','r2:0'))
            self.assertIsNotNone(p.store.get('feature_experiment_plan','r2:'+first['feature_set_id']))
            fits=p.sandbox.fits
            run_feature_session(p,base,second_split,path)
            self.assertEqual(p.sandbox.fits,fits)

    def test_ensemble_champion_uses_verified_single_source_and_refreshes_reference(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as cleanup:
            p,base,split,path=self.harness(Path(tmp))
            cleanup.callback(p.store.close)
            ensemble=base.model_copy(deep=True)
            ensemble.plan.ensemble_sources=[dict(experiment_id=base.run.experiment_id)]
            row=base.model_dump(mode='json')
            chosen=select_feature_baseline(p,ensemble,{'baseline':row},2)
            self.assertEqual(chosen.candidate.candidate_id,base.candidate.candidate_id)
            self.assertFalse(chosen.plan.ensemble_sources)
            self.assertFalse(hasattr(chosen,'evaluation'))
            p.store.put('cycle','r1',dict(round_id='r1',completed=True))
            p.store.put('component','baseline',dict(row,round_id='r1',experiment_id='baseline',configuration_hash='fixed'))
            historical=select_feature_baseline(p,ensemble,{},2)
            self.assertEqual(historical.candidate.candidate_id,base.candidate.candidate_id)
            self.assertFalse(historical.plan.ensemble_sources)

    def test_failures_consume_six_and_resume_finishes_reserved_last_attempt_once(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as cleanup:
            p,base,split,path=self.harness(Path(tmp),interrupt=True)
            cleanup.callback(p.store.close)
            with self.assertRaises(KeyboardInterrupt):run_feature_session(p,base,split,path)
            pending=p.store.get('feature_session','initial')
            self.assertEqual(pending['attempts'],6)
            self.assertIsNotNone(pending['pending'])
            self.assertEqual(len(p.store.list('feature_evidence')),5)
            result=run_feature_session(p,base,split,path)
            self.assertIsNone(result)
            session=p.store.get('feature_session','initial')
            self.assertEqual(session['status'],'completed')
            self.assertEqual(session['attempts'],6)
            self.assertEqual(len(p.store.list('feature_evidence')),6)
            self.assertEqual(p.sandbox.fits,6)
            self.assertTrue(all(e['status']=='failed' for e in p.store.list('feature_evidence')))
            self.assertEqual(p.client.calls,6)
            self.assertIsNone(run_feature_session(p,base,split,path))
            self.assertEqual(p.sandbox.fits,6)

    def test_three_completed_no_gain_tests_stop_and_never_request_outer(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as cleanup:
            p,base,split,path=self.harness(Path(tmp),fail=False)
            cleanup.callback(p.store.close)
            self.assertIsNone(run_feature_session(p,base,split,path))
            session=p.store.get('feature_session','initial')
            self.assertEqual(session['attempts'],3)
            self.assertEqual(session['stagnant'],3)
            self.assertEqual(p.sandbox.fits,3)
            self.assertEqual(len(p.store.list('feature_evidence')),3)
            self.assertTrue(all(e['status']=='rejected' for e in p.store.list('feature_evidence')))

    def test_positive_mean_is_accepted_without_stability_gate(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as cleanup:
            p,base,split,path=self.harness(Path(tmp),fail=False)
            cleanup.callback(p.store.close)
            reference=base.run.model_dump()
            protocol=json.loads(Path(reference['validation_protocol_ref']).read_text())
            result=copy.deepcopy(reference)
            # Three folds, positive mean but only one positive gain.
            gains=[10.,-1.,-1.]
            for row,gain in zip(result['fold_results'],gains):row['score']-=gain
            result['internal_score']=np.mean([f['score'] for f in result['fold_results']])
            from mlmom.feature_search import fixed_model_hash
            status,mean,deltas=paired_evidence(result,reference,protocol,fixed_model_hash(reference['model_spec']),'minimize')
            self.assertEqual(status,'accepted')
            self.assertGreater(mean,0)


class FeaturePlanningTests(unittest.TestCase):
    def test_repeated_outer_acceptance_preserves_registry_and_records_each_cycle(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as cleanup:
            store=ExperimentStore(Path(tmp));cleanup.callback(store.close)
            pipeline=NS(store=store,run_dir=Path(tmp))
            definition=FeatureSet(feature_set_id='accepted',features=[dict(name='ln',op='log',columns=['x'])])
            original=None
            for session_id,round_id in [('initial','r1'),('r2','r2')]:
                experiment_id=round_id+'-accepted'
                winner=dict(plan=dict(feature_set=definition.model_dump()),
                            run=dict(round_id=round_id,experiment_id=experiment_id))
                store.put('feature_session',session_id,FeatureSession(session_id=session_id,round_id=round_id,
                    status='selected',baseline_candidate_id='base',model_hash='fixed',train_hash=round_id,
                    protocol_hash=round_id,protocol_ref='folds.json',winner=winner))
                key='accepted' if session_id=='initial' else session_id+':accepted'
                store.put('feature_evidence',key,dict(session_id=session_id,protocol_hash=round_id))
                complete_feature_session(pipeline,winner,True)
                registry=store.get('feature_registry','accepted')
                if original is None:original=registry
                self.assertEqual(registry,original)
                self.assertEqual(store.get('feature_registry_acceptance',key)['outer_experiment_id'],experiment_id)
                complete_feature_session(pipeline,winner,True)
            self.assertEqual(len(store.list('feature_registry')),1)
            self.assertEqual(len(store.list('feature_registry_acceptance')),2)

    def test_planner_selects_only_registered_immutable_definitions(self):
        from mlmom.planning import build_planning_graph
        from tests.test_planning_limits import case, Planner, Skills
        context,tools,budget,_=case()
        definition=FeatureSet(feature_set_id='accepted',features=[dict(name='ln',op='log',columns=['x'])])
        class Selector(Planner):
            selected='accepted'
            def generate(self,role,instruction,payload,schema,tools=None):
                result=super().generate(role,instruction,payload,schema,tools)
                if role=='planner':result.plans[0].feature_set_id=self.selected
                return result
        payload=dict(round_number=2,feature_sets=[dict(definition=definition.model_dump(),evidence={})])
        hypotheses=dict(hypotheses=[dict(hypothesis_id='reuse',description='Use a registered representation',rationale='Prior paired evidence')])
        client=Selector()
        graph=build_planning_graph(client,context,tools,Skills(),budget,max_attempts=1,max_candidates=1)
        output=graph.invoke(dict(payload=payload,hypotheses=hypotheses))
        self.assertTrue(output['valid'],output)
        self.assertEqual(output['plans'][0]['feature_set'],definition.model_dump())
        client=Selector();client.selected='unknown'
        graph=build_planning_graph(client,context,tools,Skills(),budget,max_attempts=1,max_candidates=1)
        self.assertFalse(graph.invoke(dict(payload=payload,hypotheses=hypotheses))['valid'])

    def test_outer_rejection_completes_without_registering_or_reopening_search(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as cleanup:
            store=ExperimentStore(Path(tmp));cleanup.callback(store.close)
            definition=FeatureSet(feature_set_id='candidate',features=[dict(name='ln',op='log',columns=['x'])])
            winner=dict(plan=dict(feature_set=definition.model_dump()),run=dict(experiment_id='winner'))
            session=FeatureSession(status='selected',baseline_candidate_id='base',model_hash='fixed',
                train_hash='train',protocol_hash='protocol',protocol_ref='folds.json',winner=winner,
                best_feature_set_id=definition.feature_set_id,attempts=3)
            store.put('feature_session','initial',session)
            pipeline=NS(store=store,run_dir=Path(tmp),config=RunConfig())
            (pipeline.run_dir/'feature_search').mkdir()
            complete_feature_session(pipeline,winner,False)
            self.assertEqual(store.get('feature_session','initial')['reason'],'no_outer_improvement')
            self.assertEqual(store.list('feature_registry'),[])
            # Completed-session replay needs no context, agent or worker; only cached selection.
            self.assertEqual(run_feature_session(pipeline,None,None,None),winner)
            self.assertEqual(store.get('feature_session','initial')['attempts'],3)


if __name__=='__main__':unittest.main()
