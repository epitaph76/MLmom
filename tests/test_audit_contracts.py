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
from mlmom.execution import audit_fit, complete_export, FidelityError, require_admitted, config_hash
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

    def test_aggregate_support_is_optional_and_metadata_is_not_evidence(self):
        session=self.session()
        result=session.csv.query_csv('SELECT AVG(x) AS average FROM train')
        saved=session.save_observation('average','Measured average',numbers={'average':result['rows'][0]['average']},source_ref=result['source_ref'])
        self.assertIsNone(saved['rows']);self.assertEqual(saved['result_rows'],1)
        bad=session.tools.invoke('save_observation',dict(topic='bad',description='Metadata is not data',
            numbers={'invented_measurement':1},source_ref=result['source_ref']))
        self.assertEqual(bad.error_code,'unverified_measurement')
        self.assertEqual(bad.field_errors[0]['field'],'numbers.invented_measurement')
        self.assertIn('Omit rows',bad.hint)

    def test_rounding_and_wrong_support_get_specific_feedback_and_bounded_repeats(self):
        session=self.session();result=session.csv.query_csv('SELECT AVG(x) AS average FROM train')
        exact=result['rows'][0]['average']
        args=dict(topic='average',description='Rounded value',numbers={'average':round(exact,2)},rows=180,source_ref=result['source_ref'])
        for _ in range(2):
            feedback=session.tools.invoke('save_observation',args)
            self.assertEqual({e['field'] for e in feedback.field_errors},{'numbers.average','rows'})
        resumed=self.session()
        self.assertEqual(resumed.tools.invoke('save_observation',args).error_code,'repeated_observation_error')
        self.assertEqual(len(resumed.unfinished),1)

    def test_final_observations_keep_proposals_with_original_unchecked_ids(self):
        session=self.session();result=session.csv.query_csv('SELECT COUNT(*) AS n FROM train')
        draft=Observation(observation_id='local-count',topic='size',description='Measured training count',numbers={'n':180},source_ref=result['source_ref'])
        valid=dict(description='Check size',evidence_ids=['local-count'],expected_effect='unknown',verification='same split')
        answer=session.reconcile(ReconnaissanceReport(observations=[draft],proposals=[valid,dict(valid,evidence_ids=['invented'])]))
        self.assertEqual(answer.status,'completed');self.assertEqual(len(answer.observations),1)
        self.assertEqual(len(answer.proposals),2)
        self.assertEqual([p.evidence_ids for p in answer.proposals],[['local-count'],['invented']])
        self.assertIsNone(answer.reason)
        self.assertEqual(len(self.session().observations),1)

    def test_invalid_observation_does_not_discard_proposals(self):
        session=self.session();result=session.csv.query_csv('SELECT COUNT(*) AS n FROM train')
        a=Observation(observation_id='same',topic='size',description='Count',numbers={'n':180},source_ref=result['source_ref'])
        b=a.model_copy(update={'numbers':{'n':999}})
        report=session.reconcile(ReconnaissanceReport(observations=[a,b],proposals=[dict(description='bad alias',evidence_ids=['same'],expected_effect='gain',verification='test')]))
        self.assertEqual(len(report.observations),1)
        self.assertEqual(len(report.proposals),1)
        self.assertEqual(report.proposals[0].evidence_ids,['same'])
        self.assertEqual(report.status,'partial')

    def test_reworded_saved_observation_does_not_discard_its_proposal(self):
        session=self.session();result=session.csv.query_csv('SELECT COUNT(*) AS n FROM train')
        saved=Observation.model_validate(session.save_observation('size','Training count',numbers={'n':180},source_ref=result['source_ref']))
        proposal=dict(description='Check size',evidence_ids=[saved.observation_id],expected_effect='unknown',verification='same split')
        report=session.reconcile(ReconnaissanceReport(observations=[saved.model_copy(update={'description':'Reworded training count'})],proposals=[proposal]))
        self.assertEqual(report.status,'completed')
        self.assertEqual(report.proposals[0].model_dump(),proposal)
        self.assertIsNone(report.reason)

    def test_proposal_references_can_be_missing_or_empty(self):
        session=self.session()
        proposal=dict(description='Try a different model',expected_effect='unknown',verification='same split')
        report=session.reconcile(ReconnaissanceReport(proposals=[proposal,dict(proposal,evidence_ids=[])]))
        self.assertEqual(report.status,'completed')
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


class FittedCandidate:
    def __init__(self, params, model):self.params=copy.deepcopy(params);self.model=model
    def export_model_params(self):return copy.deepcopy(self.params)


class RoutingEnsemble:
    def __init__(self,config):self.params=copy.deepcopy(config['model_params'])
    def fit(self,X,y,context):
        from sklearn.ensemble import RandomForestRegressor
        a=dict(self.params['trees'],max_depth=self.params['max_depth'])
        self.models={'trees':ExtraTreesRegressor(**a,random_state=context.seed,n_jobs=1).fit(X,y),
                     'forest':RandomForestRegressor(**self.params['forest'],random_state=context.seed,n_jobs=1).fit(X,y)}
        return self
    def fitted_components(self):return self.models
    def export_model_params(self):return copy.deepcopy(self.params)
    def predict(self,X,spec):return sum(w*m.predict(X) for w,m in zip(self.params['weights'],self.models.values()))
    def save(self,path):
        import cloudpickle
        Path(path,'model.pkl').write_bytes(cloudpickle.dumps(self))


class ExecutionContractTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.X=pd.DataFrame(np.random.default_rng(4).normal(size=(60,3)))
        self.y=self.X[0]*2+self.X[1]
        self.context=FitContext('regression','rmse','minimize',42,1,2,30,timeouts_enabled=False)

    def test_incomplete_nested_export_and_flat_hpo_are_rejected(self):
        requested=dict(lightgbm=dict(n_estimators=8,learning_rate=.06,num_leaves=93),catboost=dict(iterations=8,depth=4),
                       blend=dict(weights=[.7,.3]),learning_rate=.08)
        candidate=NS(export_model_params=lambda:dict(lightgbm=dict(n_estimators=8),catboost=dict(iterations=8)))
        with self.assertRaisesRegex(ValueError,'learning_rate'):
            complete_export(candidate,requested)
        candidate.export_model_params=lambda:copy.deepcopy(requested)
        self.assertEqual(complete_export(candidate,requested),requested)

    def test_actual_backend_parameters_and_iteration_counts_are_verified(self):
        params=dict(n_estimators=8,max_depth=3)
        model=ExtraTreesRegressor(**params,random_state=42,n_jobs=1).fit(self.X,self.y)
        candidate=FittedCandidate(params,model)
        report=audit_fit(candidate,params,{},self.context,self.root/'report.json',stage='test')
        self.assertEqual(report['components']['model']['completed_iterations'],8)
        self.assertEqual(report['configuration_hash'],config_hash(params))
        with self.assertRaisesRegex(FidelityError,'max_depth'):
            audit_fit(candidate,dict(params,max_depth=7),{},self.context,self.root/'bad.json',stage='test')

    def test_sparse_catboost_metric_history_cannot_undercount_fitted_trees(self):
        from catboost import CatBoostRegressor
        params=dict(iterations=8,depth=2,metric_period=3)
        model=CatBoostRegressor(**params,verbose=False,allow_writing_files=False,thread_count=1).fit(self.X,self.y)
        self.assertEqual(len(model.get_evals_result()['learn']['RMSE']),4)
        report=audit_fit(FittedCandidate(params,model),params,dict(tool_id='catboost'),self.context,
                         self.root/'sparse-history.json',stage='full_refit')
        actual=report['components']['model']
        self.assertEqual(actual['completed_iterations'],8)
        self.assertEqual(actual['retained_trees'],8)
        self.assertEqual(actual['stopping_reason'],'iteration_limit')

    def test_actual_iterations_above_requested_cap_cannot_pass_either_stopping_policy(self):
        params=dict(n_estimators=4,max_depth=3)
        model=ExtraTreesRegressor(**dict(params,n_estimators=8),random_state=42,n_jobs=1).fit(self.X,self.y)
        for policy in ('fixed','early_stopping'):
            with self.subTest(policy=policy):
                path=self.root/f'over-cap-{policy}.json'
                with self.assertRaisesRegex(FidelityError,'requested at most 4 iterations, completed 8'):
                    audit_fit(FittedCandidate(params,model),params,dict(stopping_policy=policy),self.context,path,stage='test')
                actual=json.loads(path.read_text())['components']['model']
                self.assertEqual(actual['stopping_reason'],'iteration_limit_exceeded')

    def test_nested_ensemble_and_routed_flat_hpo_roundtrip_preserve_predictions(self):
        import inspect
        from mlmom.worker import smoke
        source='import copy\nimport numpy as np\nfrom pathlib import Path\nfrom sklearn.ensemble import ExtraTreesRegressor\n'+inspect.getsource(RoutingEnsemble)
        source+='\ndef create_candidate(config):return RoutingEnsemble(config)\ndef load_candidate(path):\n    import cloudpickle\n    return cloudpickle.loads(Path(path,"model.pkl").read_bytes())\n'
        code=self.root/'code';code.mkdir();(code/'candidate.py').write_text(source)
        params=dict(trees=dict(n_estimators=4),forest=dict(n_estimators=4,max_depth=2),weights=[.7,.3],max_depth=4)
        config=dict(tool_id='sklearn',model_params=params,parameter_routes={'max_depth':'trees'},
            ensemble_components=[dict(name='trees',tool_id='sklearn'),dict(name='forest',tool_id='sklearn')])
        job=self.root/'smoke';job.mkdir()
        result=smoke(dict(code_dir=str(code),config=config,task_type='regression',worker_seconds=30,worker_cpus=1,worker_memory_gb=2),job)
        self.assertEqual(result['execution_report']['roundtrip'],'passed')
        self.assertEqual(result['execution_report']['exported_params'],params)
        components=result['execution_report']['components']
        self.assertEqual(components['trees']['actual_params']['max_depth'],4)
        self.assertEqual(components['forest']['actual_params']['max_depth'],2)

    def test_lightgbm_metric_alias_representation_is_not_a_false_mismatch(self):
        from lightgbm import LGBMRegressor
        params=dict(n_estimators=8,metric='rmse',min_child_samples=4)
        model=LGBMRegressor(**params,verbosity=-1,n_jobs=1).fit(self.X,self.y)
        report=audit_fit(FittedCandidate(params,model),params,{'tool_id':'lightgbm'},self.context,self.root/'lgb.json',stage='test')
        self.assertEqual(report['status'],'passed')

    def test_none_actual_parameter_cannot_hide_a_dropped_depth(self):
        params=dict(n_estimators=8,max_depth=3)
        model=ExtraTreesRegressor(n_estimators=8,max_depth=None,n_jobs=1).fit(self.X,self.y)
        with self.assertRaisesRegex(FidelityError,'max_depth'):
            audit_fit(FittedCandidate(params,model),params,{},self.context,self.root/'none-depth.json',stage='test')

    def test_torch_epochs_are_derived_from_actual_optimizer_steps(self):
        import importlib.util
        if importlib.util.find_spec('torch') is None:
            from mlmom.registry import installed_tools
            self.assertIsNone(installed_tools()['torch'].version)
            return  # This backend is exercised in the provided-dependencies Linux image.
        import torch
        from dataclasses import replace
        torch.set_num_threads(1);torch.manual_seed(42)
        model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.ReLU(),torch.nn.Linear(4,1))
        optimizer=torch.optim.Adam(model.parameters(),lr=.01)
        self.context.fit_rows=len(self.X)
        handle=replace(self.context).track_optimizer(optimizer,20)
        X=torch.tensor(self.X.to_numpy(),dtype=torch.float32);y=torch.tensor(self.y.to_numpy()[:,None],dtype=torch.float32)
        for _ in range(3):
            for start in range(0,len(X),20):
                optimizer.zero_grad();loss=torch.square(model(X[start:start+20])-y[start:start+20]).mean();loss.backward();optimizer.step()
        params=dict(epochs=3,hidden_size=4,lr=.01,batch_size=20)
        report=audit_fit(FittedCandidate(params,model),params,{'tool_id':'torch'},self.context,self.root/'torch.json',stage='test')
        self.assertEqual(report['components']['model']['completed_iterations'],3)
        self.assertEqual(report['components']['model']['optimizer_steps'],9)
        with self.assertRaises(FidelityError):
            audit_fit(FittedCandidate(dict(params,epochs=5),model),dict(params,epochs=5),{},self.context,self.root/'short-torch.json',stage='test')

        # A partial additional epoch must not disappear through integer division.
        optimizer.zero_grad();torch.square(model(X[:20])-y[:20]).mean().backward();optimizer.step()
        handle.remove()
        with self.assertRaisesRegex(FidelityError,'iteration limit exceeded'):
            audit_fit(FittedCandidate(params,model),params,{'tool_id':'torch'},self.context,self.root/'extra-step-torch.json',stage='test')
        actual=json.loads((self.root/'extra-step-torch.json').read_text())['components']['model']
        self.assertEqual(actual['optimizer_steps'],10)
        self.assertAlmostEqual(actual['completed_iterations'],10/3)

    def test_historical_catboost_callback_cannot_pass_gate(self):
        from catboost import CatBoostRegressor
        class ReversedCallback:
            def after_iteration(self,info):return time.time() >= time.time()+60
        params=dict(iterations=8,depth=2,learning_rate=.1)
        model=CatBoostRegressor(**params,verbose=False,allow_writing_files=False,thread_count=1)
        model.fit(self.X,self.y,callbacks=[ReversedCallback()])
        self.assertEqual(model.tree_count_,1)
        with self.assertRaisesRegex(FidelityError,'completed 1'):
            audit_fit(FittedCandidate(params,model),params,{},self.context,self.root/'bad.json',stage='test')
        self.assertEqual(json.loads((self.root/'bad.json').read_text())['status'],'failed')

    def test_permitted_catboost_early_stopping_has_measured_reason(self):
        from catboost import CatBoostRegressor
        params=dict(iterations=40,depth=2,learning_rate=.2)
        model=CatBoostRegressor(**params,verbose=False,allow_writing_files=False,thread_count=1)
        model.fit(self.X,self.y,eval_set=(self.X,-self.y),early_stopping_rounds=3,use_best_model=True)
        self.context.validation_y=-self.y
        frozen=FittedCandidate(dict(params,iterations=model.tree_count_),model)
        report=audit_fit(frozen,params,{'stopping_policy':'early_stopping'},self.context,self.root/'early.json',stage='test')
        self.assertEqual(report['components']['model']['stopping_reason'],'early_stopping')
        with self.assertRaisesRegex(FidelityError,'must freeze'):
            audit_fit(FittedCandidate(params,model),params,{'stopping_policy':'early_stopping'},self.context,self.root/'unfrozen.json',stage='test')
        with self.assertRaises(FidelityError):
            audit_fit(FittedCandidate(params,model),params,{},self.context,self.root/'fixed.json',stage='test')

    def test_xgboost_early_stopping_freezes_prediction_round_not_patience_trees(self):
        from xgboost import XGBRegressor
        params=dict(n_estimators=40,max_depth=2,learning_rate=.2)
        model=XGBRegressor(**params,n_jobs=1,early_stopping_rounds=3)
        model.fit(self.X,self.y,eval_set=[(self.X,-self.y)],verbose=False)
        self.context.validation_y=-self.y
        selected=model.best_iteration+1
        self.assertLess(selected,model.get_booster().num_boosted_rounds())
        frozen=FittedCandidate(dict(params,n_estimators=selected),model)
        report=audit_fit(frozen,params,dict(stopping_policy='early_stopping',tool_id='xgboost'),self.context,self.root/'xgb-early.json',stage='test')
        self.assertEqual(report['components']['model']['retained_iterations'],selected)

    def test_admission_and_review_cannot_override_missing_or_failed_provenance(self):
        run=NS(execution_report=None,full_refit=True,best_params={})
        with self.assertRaisesRegex(ValueError,'missing'):
            require_admitted(run,NS(code_hash='code'))
        client=NS(generate=lambda *args:ResultReview(implementation='matches',effect='not_supported',explanation='wrong'))
        report=review_result(client,dict(execution_report=dict(status='failed',errors=['1 tree instead of 800'])),[],'bad',1)
        self.assertEqual((report.implementation,report.effect),('mismatch','inconclusive'))

    def test_future_annotations_is_allowed_without_allowing_infrastructure(self):
        source="from __future__ import annotations\ndef create_candidate(config): return None\ndef load_candidate(path): return None\n"
        path=self.root/'candidate.py';path.write_text(source)
        self.assertTrue(verify_source(self.root).valid)
        path.write_text(source+'import os\n')
        self.assertFalse(verify_source(self.root).valid)


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

    def test_narrative_cannot_claim_unimplemented_duplicate_protocol(self):
        from mlmom.planning import validate_plan
        context,tools,budget,plan=case()
        plan.hypothesis.description='duplicate-aware evaluation'
        self.assertFalse(validate_plan(plan,context,tools,Skills(),budget).valid)

    def test_single_change_ablation_inherits_authoritative_champion(self):
        from mlmom.planning import build_planning_graph
        context,tools,budget,_=case()
        class AblationPlanner(Planner):
            def generate(self,role,instruction,payload,schema,tools=None):
                result=super().generate(role,instruction,payload,schema,tools)
                if role=='planner':
                    plan=result.plans[0];plan.hypothesis.based_on='current';plan.change_kind='single_change_ablation'
                    plan.model_params={'n_estimators':1};plan.strategy='optuna';plan.search_space={'n_estimators':dict(type='int',low=1,high=3)}
                return result
        params={'n_estimators':8,'max_depth':7}
        payload=dict(round_number=2,champion=dict(candidate_id='current',configuration_hash=config_hash(params),model_params=params,plan={'tool_id':'sklearn'}))
        graph=build_planning_graph(AblationPlanner(),context,tools,Skills(),budget,planning_mode='baseline',known_candidates=['current'],lightweight=False)
        result=graph.invoke({'payload':payload})
        self.assertTrue(result['valid'],result)
        self.assertEqual(result['plans'][0]['model_params'],params)
        self.assertEqual(result['plans'][0]['strategy'],'single')
        self.assertEqual(result['plans'][0]['base_config_hash'],config_hash(params))


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
