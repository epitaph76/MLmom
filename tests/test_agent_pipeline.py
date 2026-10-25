"""Regression scenarios from the agent-first run; no paid API calls."""
import copy
import json
import tempfile
import threading
import unittest
from contextvars import ContextVar
from pathlib import Path
from unittest.mock import patch

from mlmom.concurrency import PreparationPool
from mlmom.execution_contract import execution_contract
from mlmom.hpo import run_search
from mlmom.hpo_baseline import importable_baseline
from mlmom.persistence import atomic_json, file_hash
from mlmom.pipeline_contracts import TrainingCheckpoint, TrainingDecision, RunConfig
from mlmom.search_groups import configure_composite, ensemble_recipes
from mlmom.time_policy import agent_payload
from mlmom.training_control import validate_decision
from tests.test_planning_limits import case


class SamplerOwnershipTests(unittest.TestCase):
    def arguments(self, root, **updates):
        return dict(directory=root, identity={'implementation':'frozen'},
            space={'depth':dict(type='int',low=1,high=4)},direction='maximize',seed=42,
            max_trials=2,batch_size=2,gain_threshold=0,timeouts_enabled=False,**updates)

    def test_baseline_import_uses_real_evidence_without_consuming_new_trials_or_reimporting(self):
        with tempfile.TemporaryDirectory() as tmp:
            selection=Path(tmp)/'baseline-selection.json'
            selection.write_text('{}')
            seen=[]
            args=self.arguments(tmp,baseline_trial=dict(params={'depth':4},score=99.,selection_ref=str(selection)))
            with patch('optuna.study.Study.enqueue_trial',side_effect=AssertionError('LLM must not queue parameters')):
                state=run_search(lambda p,n,s:seen.append(n) or 1.,**args)
            self.assertEqual(seen,[1,2])
            self.assertEqual(state['new_trials'],2)
            self.assertEqual(state['trials'][0]['origin'],'baseline')
            self.assertEqual(state['trials'][0]['selection_ref'],str(selection))
            again=run_search(lambda *a:self.fail('Resume must reuse finished work'),**args)
            self.assertEqual(len(again['trials']),3)

    def test_agent_renews_budget_once_and_explicit_user_maximum_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls=[]
            def control(data):
                calls.append(data)
                self.assertIsNone(data['remaining_seconds'])
                return TrainingDecision(action='continue',additional_trials=3,reason='More evidence')
            state=run_search(lambda *a:1.,checkpoint=control,**self.arguments(tmp,hard_limit=4))
            self.assertEqual(state['new_trials'],4)
            self.assertEqual(state['trial_budget'],4)
            self.assertEqual(len(calls),2)
            self.assertEqual(len(state['applied_controls']),2)
            run_search(lambda *a:self.fail('No duplicate trials'),checkpoint=control,**self.arguments(tmp,hard_limit=4))
            self.assertEqual(len(calls),2)

    def test_checkpoint_can_override_plateau_without_changing_sampler_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.arguments(tmp)
            args.update(max_trials=6,gain_threshold=.005,baseline_score=1.)
            calls=[]
            def control(data):
                calls.append(data['default_action'])
                return TrainingDecision(action='continue' if len(calls)==1 else 'refit',reason='Measured decision')
            state=run_search(lambda *a:1.,checkpoint=control,**args)
            self.assertEqual(state['new_trials'],4)
            self.assertEqual(calls,['refit','refit'])

    def test_extension_survives_interrupted_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            def stopped(data): raise KeyboardInterrupt()
            args=self.arguments(tmp)
            with self.assertRaises(KeyboardInterrupt):
                run_search(lambda *a:1.,checkpoint=stopped,**args)
            decisions=[]
            def resumed(data):
                decisions.append(len(data['trials']))
                return TrainingDecision(action='continue' if len(decisions)==1 else 'refit',
                    additional_trials=2 if len(decisions)==1 else 0,reason='Resume decision')
            state=run_search(lambda *a:1.,checkpoint=resumed,**args)
            self.assertEqual(state['new_trials'],4)
            self.assertEqual(decisions,[2,4])


class ExecutablePlanTests(unittest.TestCase):
    def test_rejected_effective_plan_and_field_ownership_reach_the_planner(self):
        from mlmom.planning import build_planning_graph
        from tests.test_planning_limits import Planner,Skills
        context,tools,budget,plan=case()
        class Reviewing(Planner):
            def __init__(self):
                super().__init__();self.repairs=[];self.reviews=0
            def generate(self,role,instruction,payload,schema,role_tools=None):
                if role=='planner': self.repairs.append(payload.get('repair_context'))
                if role=='plan_validator':
                    self.reviews+=1
                    self.assert_sources=payload['field_sources']
                    return schema(valid=self.reviews>1,errors=['Clarify the effective SDK assignment'] if self.reviews==1 else [])
                return super().generate(role,instruction,payload,schema,role_tools)
        client=Reviewing()
        result=build_planning_graph(client,context,tools,Skills(),budget,lightweight=False).invoke(
            dict(payload={'round_number':1},hypotheses={'hypotheses':[plan.hypothesis.model_dump(mode='json')]}))
        self.assertTrue(result['valid'],result)
        self.assertEqual(client.repairs[1]['effective_plan']['tool_id'],'sklearn')
        self.assertEqual(client.repairs[1]['execution_contract']['feature_owner'],'SDK FeatureAugmentingPreprocessor')
        self.assertIn('model_params',client.assert_sources)

    def test_new_source_repairs_continue_but_unchanged_failure_is_not_repeated_forever(self):
        from types import SimpleNamespace
        from mlmom.recovery import repair_allowed
        from mlmom.persistence import ExperimentStore
        _,_,_,plan=case()
        with tempfile.TemporaryDirectory() as tmp:
            store=ExperimentStore(Path(tmp))
            try:
                for attempt in range(4):
                    candidate=SimpleNamespace(code_hash=str(attempt))
                    self.assertTrue(repair_allowed(store,'experiment',candidate,plan,'ValueError: invalid fit',attempt))
                self.assertFalse(repair_allowed(store,'experiment',candidate,plan,'ValueError: invalid fit',4))
                self.assertFalse(repair_allowed(store,'explicit',candidate,plan,'error',2,maximum=2))
            finally: store.close()

    def test_warm_start_rejects_changed_data_protocol_and_preprocessing_without_blocking_search(self):
        context,_,_,plan=case()
        plan.model_params={'n_estimators':10}
        plan.search_space={'n_estimators':dict(type='int',low=1,high=100)}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'train.csv';data.write_text('x,y\n1,2\n')
            original=dict(code_hash='code',train_path=str(data),plan=plan.model_dump(mode='json'),
                          context=dict(task={'task_type':'regression'},metric_options={}))
            selection=dict(protocol={'folds':[1]},params=plan.model_params,score=1.,
                execution_identity=dict(code_hash='code',train_hash=file_hash(data)))
            atomic_json(root/'request.json',original);atomic_json(root/'selection.json',selection)
            request=dict(original,baseline_evidence=dict(request_ref=str(root/'request.json'),selection_ref=str(root/'selection.json')))
            self.assertIsNotNone(importable_baseline(request,selection['protocol']))
            self.assertIsNone(importable_baseline(request,{'folds':[2]}))
            changed=copy.deepcopy(request);changed['plan']['preprocessing']=['different']
            self.assertIsNone(importable_baseline(changed,selection['protocol']))
            changed=copy.deepcopy(request);changed['plan']['search_space']['n_estimators']['high']=200
            self.assertIsNone(importable_baseline(changed,selection['protocol']))
            data.write_text('x,y\n1,9\n')
            self.assertIsNone(importable_baseline(request,selection['protocol']))

    def test_baseline_assignment_identity_ignores_threads_and_accepts_measured_export_defaults(self):
        from mlmom.hpo_baseline import same_fixed_settings
        original=dict(iterations=300,depth=4,thread_count=2)
        current=dict(iterations=300,depth=4)
        exported=dict(original,task_type='CPU')
        space={'depth':dict(type='int',low=2,high=5)}
        self.assertTrue(same_fixed_settings(original,current,exported,space))
        self.assertTrue(same_fixed_settings(original,dict(current,task_type='CPU'),exported,space))
        self.assertFalse(same_fixed_settings(original,dict(current,iterations=500),exported,space))
        self.assertFalse(same_fixed_settings(original,dict(depth=4),exported,space))
        self.assertFalse(same_fixed_settings(original,dict(current,task_type='GPU'),exported,space))

    def test_mean_correction_is_idempotent_and_retains_component_feature_set(self):
        _,_,_,plan=case()
        source=dict(name='base_0',config=plan.model_dump(mode='json'),code_hash='unchanged')
        source['config']['feature_set_id']='measured-features'
        second=copy.deepcopy(source);second['name']='base_1'
        plan.model_params=dict(method='mean',weights=[.25,.75],oof_folds=3,weight_trials=15)
        prepared=configure_composite(plan,[source,second],15)
        self.assertNotIn('oof_folds',prepared.model_params)
        self.assertNotIn('weight_trials',prepared.model_params)
        before=prepared.model_dump(mode='json')
        self.assertEqual(configure_composite(prepared,[source,second],15).model_dump(mode='json'),before)
        self.assertEqual(prepared.model_params['weights'],[.25,.75])
        contract=execution_contract(prepared)
        self.assertEqual(contract['frozen_components'][0]['effective_configuration']['feature_set_id'],'measured-features')
        self.assertEqual(len(ensemble_recipes(prepared,'regression').variants),1)

    def test_no_time_budget_is_explicit_and_numeric_llm_suggestions_are_inert(self):
        payload=agent_payload({'budget':{'remaining_seconds':1e12,'research_seconds':1e12},
            'improvement_policy':{'remaining_cycle_seconds':1e12}},timed=False)
        self.assertIsNone(payload['budget']['remaining_seconds'])
        self.assertIsNone(payload['improvement_policy']['remaining_cycle_seconds'])
        cp=TrainingCheckpoint(checkpoint_id='renew',stage='batch',plan={},remaining_trials=0,
            trial_limit=4,remaining_seconds=None,timeouts_enabled=False,default_action='refit',
            trials=[{'origin':'trained'},{'origin':'trained'}])
        decision=TrainingDecision(action='continue',reason='Extend',additional_trials=2,next_params=[{'depth':999}])
        self.assertEqual(validate_decision(decision,cp).additional_trials,2)
        self.assertNotIn('next_params',TrainingDecision.model_json_schema()['properties'])
        with self.assertRaises(ValueError):
            validate_decision(TrainingDecision(action='continue',reason='Exceeds cap',additional_trials=3),cp)
        self.assertFalse(RunConfig().initial_boosting_sweep)
        self.assertIsNone(RunConfig().hpo_max_trials)


class PreparationConcurrencyTests(unittest.TestCase):
    def test_real_role_settings_are_effective_and_isolated_between_parallel_assignments(self):
        import time
        from mlmom.agents import RoleClient
        from mlmom.concurrency import AssignmentClient
        from mlmom.llm_client import LLMSettings
        client=RoleClient(LLMSettings(api_key='offline-test'),deadline=time.time()+100,max_steps=12)
        self.addCleanup(client.client.close)
        pool=PreparationPool(2)
        barrier=threading.Barrier(2)
        original_deadline=client.deadline
        def inspect(*args):
            barrier.wait(timeout=3)
            return client.max_steps,client.deadline
        def assigned(steps,deadline):
            assignment=AssignmentClient(client,threading.Event())
            assignment.max_steps,assignment.deadline=steps,deadline
            result=assignment.generate('coding_agent','',{},TrainingDecision)
            return result,client.max_steps,client.deadline
        try:
            with patch.object(client,'_generate',side_effect=inspect):
                first=pool.submit(assigned,3,10.)
                second=pool.submit(assigned,7,20.)
                self.assertEqual(first.result(timeout=4),((3,10.),12,original_deadline))
                self.assertEqual(second.result(timeout=4),((7,20.),12,original_deadline))
            self.assertEqual(client.max_steps,12)
            self.assertEqual(client.deadline,original_deadline)
        finally: pool.close()

    def test_cancelled_role_response_cannot_be_applied_after_stop(self):
        from mlmom.concurrency import AssignmentClient
        from concurrent.futures import CancelledError
        cancelled=threading.Event()
        class Client:
            def generate(self):
                cancelled.set()
                return 'source changes'
        with self.assertRaises(CancelledError): AssignmentClient(Client(),cancelled).generate()

    def test_independent_tasks_overlap_and_keep_context_local(self):
        context=ContextVar('assignment',default='controller')
        barrier=threading.Barrier(2)
        pool=PreparationPool(2)
        try:
            def prepare():
                value=context.get()
                context.set('private-'+value)
                barrier.wait(timeout=3)
                return context.get()
            context.set('a');first=pool.submit(prepare)
            context.set('b');second=pool.submit(prepare)
            self.assertEqual(first.result(timeout=4),'private-a')
            self.assertEqual(second.result(timeout=4),'private-b')
            self.assertEqual(context.get(),'b')
        finally: pool.close()

    def test_stop_cancels_queued_work_without_waiting_for_active_task(self):
        pool=PreparationPool(1)
        started=threading.Event();release=threading.Event()
        def active():
            started.set();release.wait(3)
        first=pool.submit(active)
        self.assertTrue(started.wait(2))
        queued=pool.submit(lambda:self.fail('Cancelled task ran'))
        pool.close()
        self.assertTrue(queued.cancelled())
        self.assertFalse(first.done())
        release.set()
        first.result(timeout=2)


class PipelineFlowTests(unittest.TestCase):
    def test_early_features_parallel_preparation_and_full_submission_precede_hpo(self):
        from mlmom.pipeline import Pipeline
        from mlmom.pipeline_contracts import HypothesisBatch
        from mlmom.persistence import ExperimentStore
        from tests.fakes import OfflineLLM
        from tests.test_pipeline import dataset,only_sklearn
        barrier=threading.Barrier(2)
        class Portfolio(OfflineLLM):
            def generate(self,role,instruction,payload,schema,tools=None):
                if role=='hypothesis_generator':
                    self.calls.append((role,payload))
                    return HypothesisBatch(hypotheses=[dict(hypothesis_id='a',description='Small control',rationale='Control',baseline=True),
                        dict(hypothesis_id='b',description='Another fixed configuration',rationale='Alternative')])
                if role=='feature_agent':
                    self.calls.append((role,payload))
                    return schema(action='add',rationale='Measure missingness',features=[dict(name='x_missing',op='missing',columns=['x'])])
                if role=='preprocessing_agent': barrier.wait(timeout=5)
                result=super().generate(role,instruction,payload,schema,tools)
                if role=='planner' and schema.__name__=='PlanBatch':
                    result.plans[0].hpo_trial_budget=2
                return result
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);client=Portfolio(params={'n_estimators':4})
            with patch('mlmom.pipeline.installed_tools',only_sklearn):
                output=Pipeline(dataset(root/'input'),RunConfig(max_rounds=1,hpo_max_trials=2,hpo_batch_size=2,
                    feature_max_configurations=1,diagnostic_seconds=0,reserve_memory_gb=0,reserve_cpus=0),
                    run_dir=root/'run',client=client).run()
            self.assertEqual(output['status'],'completed',output)
            self.assertEqual(output['submission']['model_role'],'submission_refit')
            roles=[role for role,_ in client.calls]
            self.assertLess(roles.index('context_agent'),roles.index('feature_agent'))
            self.assertLess(roles.index('feature_agent'),roles.index('coding_agent'))
            events=[json.loads(line) for line in (root/'run/trace.jsonl').read_text().splitlines()]
            def first(kind): return next(e['seq'] for e in events if e['kind']==kind)
            hpo=[e['seq'] for e in events if e['kind']=='scheduler.admitted' and
                json.loads((root/'run'/e['payload']['path']).read_text()).get('directory','').endswith('-hpo')]
            self.assertTrue(hpo)
            self.assertLess(first('submission.first'),min(hpo))
            self.assertLess(first('feature.evaluated'),min(hpo))
            store=ExperimentStore(root/'run')
            try:
                self.assertEqual(len(store.list('feature_evidence')),1)
                self.assertEqual(len(store.list('candidate')),2)
            finally: store.close()

    def test_stop_during_fit_or_later_refit_preserves_only_completed_full_result(self):
        from mlmom.pipeline import Pipeline
        from mlmom.resources import SandboxManager
        from mlmom.persistence import ExperimentStore
        from tests.fakes import OfflineLLM
        from tests.test_pipeline import dataset,only_sklearn
        for stage in ('fit','later_refit'):
            with self.subTest(stage=stage),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);task=dataset(root/'input');original=SandboxManager.start
                historical=root/'run/runs/old-completed/hpo'
                historical.mkdir(parents=True)
                (historical/'studies.sqlite3').write_text('Historical artifact must not be opened during stop')
                refits=[];published=[]
                def start(manager,request,directory,**limits):
                    job=original(manager,request,directory,**limits)
                    if request.get('phase')=='submission': refits.append(directory)
                    if (stage=='fit' and request.get('action')=='train' and not request.get('fixed')) or (
                            stage=='later_refit' and len(refits)==2 and request.get('phase')=='submission'):
                        if (task.parent/'submission.csv').exists(): published.append((task.parent/'submission.csv').read_bytes())
                        raise KeyboardInterrupt()
                    return job
                with patch('mlmom.pipeline.installed_tools',only_sklearn),patch.object(SandboxManager,'start',start):
                    output=Pipeline(task,RunConfig(max_rounds=1,hpo_enabled=False,feature_search_enabled=False,
                        diagnostic_seconds=0,reserve_cpus=0,reserve_memory_gb=0),run_dir=root/'run',client=OfflineLLM()).run()
                self.assertEqual(output['status'],'interrupted',output)
                self.assertEqual(output['stop_reason'],'external_stop')
                self.assertEqual(output['result_available'],stage=='later_refit')
                if published: self.assertEqual((task.parent/'submission.csv').read_bytes(),published[0])
                store=ExperimentStore(root/'run')
                try:
                    self.assertFalse(any(row['status']=='running' for kind in ('experiment','submission_refit') for row in store.list(kind)))
                finally: store.close()
