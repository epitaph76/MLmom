"""Reproduced skill/runtime mismatches: complete facts and original-scale scoring."""
import copy
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from mlmom.evaluation import score_predictions, prediction_kind
from mlmom.execution_facts import execution_context, RUNTIME_CONTRACT_VERSION
from mlmom.metric_contracts import metric_catalog, metric_errors
from mlmom.model_spec import apply_overrides
from mlmom.pipeline_contracts import TrainingCheckpoint, TrainingDecision
from mlmom.sdk import FitContext, PolicyCandidate, PreparedCandidate
from mlmom.task_validator import TaskValidator
from mlmom.training_control import decide
from tests.test_execution import execution_case


class MetricContractV8Tests(unittest.TestCase):
    def test_mcc_matches_sklearn_with_original_labels_and_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, context, _, _ = execution_case(Path(tmp), 'multiclass_classification')
            context.classes = ['one','two','three']
            truth = pd.Series(['one','two','three','one','two','three'])
            prediction = np.array(['two','one','three','two','one','one'])
            weights = np.array([1.,2.,3.,4.,1.,2.])
            for name in ['mcc','Matthews correlation coefficient','matthews_corrcoef']:
                context.task.primary_metric = name
                self.assertEqual(prediction_kind(name, context.task.task_type),'label')
                for weight in (None, weights):
                    self.assertAlmostEqual(score_predictions(truth,prediction,context,weights=weight),
                        matthews_corrcoef(truth,prediction,sample_weight=weight))

    def test_mcc_negative_and_constant_predictions_are_finite(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, context, _, _ = execution_case(Path(tmp), 'binary_classification')
            context.task.primary_metric = 'mcc'
            self.assertEqual(score_predictions(pd.Series([0,1,0,1]),[1,0,1,0],context),-1.)
            self.assertEqual(score_predictions(pd.Series([0,1,0,1]),[0,0,0,0],context),0.)

    def test_catalog_and_task_confirmation_reject_unexecutable_metric_contracts(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame, context, _, _ = execution_case(Path(tmp), 'binary_classification')
            profile = SimpleNamespace(columns=[SimpleNamespace(name=c) for c in frame])
            context.task.primary_metric = 'mcc'
            context.task.metric_direction = 'maximize'
            self.assertTrue(TaskValidator().validate(profile,context.task).valid)
            for changes in [dict(primary_metric='made_up'), dict(task_type='multilabel_classification'),
                            dict(metric_options={'average':'macro'}), dict(metric_direction='minimize')]:
                task = context.task.model_copy(update=changes)
                self.assertFalse(TaskValidator().validate(profile,task).valid)
            self.assertIn('mcc',{row['name'] for row in metric_catalog()})
            self.assertFalse(metric_errors('root mean squared error','regression',{}))

    def test_metric_catalog_rejects_unsupported_task_and_option_combinations(self):
        for name, task, options in [
            ('balanced_accuracy','multilabel_classification',{}),
            ('f1','multiclass_classification',{}),
            ('precision','binary_classification',{'average':'samples'}),
            ('recall','multilabel_classification',{'average':None}),
            ('roc_auc','multiclass_classification',{'max_fpr':.5}),
        ]:
            with self.subTest(name=name,task=task):
                self.assertTrue(metric_errors(name,task,options))
        self.assertFalse(metric_errors('f1','multiclass_classification',{'average':'weighted'}))
        self.assertFalse(metric_errors('f1','multilabel_classification',{'average':'samples'}))


class CheckpointConfigurationTests(unittest.TestCase):
    def test_checkpoint_view_merges_partial_overrides_and_routes_policies(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, plan, _ = execution_case(Path(tmp),'regression')
            plan.model_params = dict(n_estimators=300,max_depth=6,min_samples_leaf=2)
            sampled = {'max_depth':8,'policy_params.weight_policy.name':'tail_aware'}
            class Client:
                def generate(self,role,instruction,payload,schema,tools=None):
                    self.payload=payload
                    return schema(action='refit',reason='Complete checkpoint inspected')
            client=Client()
            checkpoint=TrainingCheckpoint(checkpoint_id='one',stage='batch',plan=plan.model_dump(mode='json'),
                best_params=sampled,remaining_trials=0,default_action='refit')
            decide(client,checkpoint,[])
            view=client.payload['execution_context']['current_configuration']
            self.assertEqual(view['model_params'],dict(n_estimators=300,max_depth=8,min_samples_leaf=2))
            self.assertEqual(view['weight_policy']['name'],'tail_aware')
            self.assertEqual(plan.model_params['max_depth'],6)

    def test_saved_selection_takes_precedence_over_sampled_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, plan, _ = execution_case(Path(tmp),'regression')
            plan.model_params = dict(n_estimators=300,max_depth=6)
            effective=apply_overrides(plan.model_dump(mode='json'),{'max_depth':8})
            effective['model_params']['n_estimators']=120
            view=execution_context(plan,dict(best_params={'max_depth':8},
                sampled_params={'max_depth':8},effective_configuration=effective))
            self.assertEqual(view['current_configuration']['model_params'],dict(n_estimators=120,max_depth=8))
            self.assertEqual(RUNTIME_CONTRACT_VERSION,8)

    def test_worker_checkpoint_uses_selected_export_without_additional_fits(self):
        from mlmom.worker import train
        from mlmom.persistence import tree_hash
        from tests.fakes import MODEL_CODE, PREPROCESSING_CODE
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            _,context,plan,code=execution_case(root,'regression')
            plan.strategy='optuna';plan.max_trials=1
            plan.model_params=dict(n_estimators=8,min_samples_leaf=2)
            plan.search_space={'max_depth':dict(type='categorical',choices=[3]),
                'policy_params.weight_policy.name':dict(type='categorical',choices=['tail_aware'])}
            source=MODEL_CODE.replace('self.kind = context.task_type',
                "self.params['n_estimators'] = 3\n        self.kind = context.task_type")
            (code/'candidate.py').write_text(source,encoding='utf-8')
            (code/'preprocessing.py').write_text(PREPROCESSING_CODE,encoding='utf-8')
            work=root/'work';work.mkdir();checkpoints=[]
            def control(directory,checkpoint,**kwargs):
                checkpoints.append(checkpoint)
                return TrainingDecision(action='refit',reason='Use the selected trial')
            request=dict(code_dir=str(code),code_hash=tree_hash(code),plan=plan.model_dump(mode='json'),
                context=context.model_dump(mode='json'),train_path=str(root/'train.csv'),
                worker_seconds=30,worker_cpus=1,worker_memory_gb=2,timeouts_enabled=False,training_control=True)
            with patch('mlmom.training_control.worker_checkpoint',side_effect=control):
                result=train(request,work)
            selected=next(c for c in checkpoints if c.best_score is not None)
            self.assertEqual(selected.sampled_params['max_depth'],3)
            self.assertEqual(selected.best_params['n_estimators'],3)
            self.assertEqual(selected.best_params['min_samples_leaf'],2)
            self.assertEqual(selected.effective_configuration['weight_policy']['name'],'tail_aware')
            self.assertEqual(len(result['execution_facts']),context.task.validation_folds+1)


class ValidationScorerTests(unittest.TestCase):
    def test_optional_plain_callable_scorer_does_not_need_metadata_attributes(self):
        from mlmom.sdk import PolicyValidationScorer
        class Policy:
            def inverse(self,values,kind):return np.exp(values)
        score=PolicyValidationScorer(lambda predictions,kind:float(np.mean(predictions)),Policy())
        self.assertAlmostEqual(score(np.log([1.,3.]),'value'),2.)

    def test_original_scale_options_weights_and_policies_are_applied_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,context,_,_=execution_case(Path(tmp),'regression')
            X=pd.DataFrame({'x':[0.,1.]});y=pd.Series([1.,100.]);weights=np.array([1.,4.])
            class Estimator:
                def fit(self,X,y,context):
                    self.context=context
                    self.scorer=context.validation_scorer
                    return self
            estimator=Estimator()
            candidate=PolicyCandidate(estimator,dict(tool_id='sklearn',
                target_transform={'name':'log1p'},prediction_policy={'clip_min':0}))
            candidate.fit(X,y,FitContext('regression','rmse','minimize',42,1,2,30,
                validation_X=X,validation_y=y,validation_weight=weights,task_context=context))
            prediction=np.array([10.,100.])
            score=estimator.scorer(np.log1p(prediction),'value')
            self.assertAlmostEqual(score,np.sqrt(np.average((y-prediction)**2,weights=weights)))
            np.testing.assert_allclose(estimator.context.validation_y,np.log1p(y))
            self.assertLess(estimator.scorer(np.log1p([10.,100.]),'value'),
                            estimator.scorer(np.log1p([1.,80.]),'value'))
            # A proxy on the transformed target chooses the opposite model.
            log_error=lambda p:np.sqrt(np.average((np.log1p(y)-np.log1p(p))**2,weights=weights))
            self.assertGreater(log_error([10.,100.]),log_error([1.,80.]))

    def test_preprocessing_never_receives_the_validation_scorer(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,context,plan,_=execution_case(Path(tmp),'regression')
            X=pd.DataFrame({'x':[0.,1.,2.]});y=pd.Series([1.,2.,3.]);seen=[]
            class Pre:
                def fit_transform(self,X,y,c):
                    seen.append(c);return X
                def transform(self,X):return X
            class Model:
                def fit(self,X,y,c):self.context=c
            model=PreparedCandidate(plan.model_dump(mode='json'),lambda config:Model(),lambda config:Pre())
            model.fit(X,y,FitContext('regression','rmse','minimize',42,1,2,30,
                validation_X=X,validation_y=y,task_context=context))
            self.assertIsNone(seen[0].validation_y)
            self.assertIsNone(seen[0].validation_scorer)
            self.assertIsNotNone(model.estimator.estimator.context.validation_scorer)


class NativeWeightContractTests(unittest.TestCase):
    def test_flaml_passes_current_train_validation_and_frozen_refit_weights(self):
        import re
        from mlmom.registry import installed_tools
        if installed_tools()['flaml'].version is None:
            self.skipTest('FLAML optional dependency not installed')
        root=Path(__file__).resolve().parents[1]
        text=(root/'skills/coding_agent/flaml/REFERENCE.md').read_text(encoding='utf-8')
        namespace={}
        exec(re.search(r'```python\n(.*?)\n```',text,re.S).group(1),namespace)
        calls=[]
        class AutoML:
            def fit(self,**kwargs):
                calls.append(('search',kwargs))
                self.model=SimpleNamespace(estimator=object())
                self.best_estimator='lgbm';self.best_config={'n_estimators':2};self.best_loss=1.
            def retrain_from_log(self,*args,**kwargs):
                calls.append(('refit',kwargs));self.model=SimpleNamespace(estimator=object())
        namespace['AutoML']=AutoML
        with tempfile.TemporaryDirectory() as tmp:
            train=np.array([[1.],[2.],[3.]])
            train_weight=np.array([1.,2.,5.]);valid_weight=np.array([3.,7.])
            context=FitContext('regression','rmse','minimize',42,1,2,30,
                sample_weight=train_weight,validation_X=train[:2],validation_y=np.array([1.,2.]),
                validation_weight=valid_weight,validation_scorer=object(),work_dir=tmp)
            model=namespace['create_candidate']({'max_trials':2})
            model.fit(train,np.array([1.,2.,3.]),context)
            np.testing.assert_array_equal(calls[0][1]['sample_weight'],train_weight)
            np.testing.assert_array_equal(calls[0][1]['sample_weight_val'],valid_weight)
            full_weight=np.array([1.,2.,3.,4.,5.])
            fixed=namespace['create_candidate']({'model_params':model.export_model_params()})
            fixed.fit(np.arange(5.).reshape(-1,1),np.arange(5.),replace(context,
                sample_weight=full_weight,validation_X=None,validation_y=None,
                validation_weight=None,validation_scorer=None))
            np.testing.assert_array_equal(calls[1][1]['sample_weight'],full_weight)
            self.assertEqual(len(calls),2)


class RoleToolTests(unittest.TestCase):
    def test_feature_tools_only_register_current_training_rows_and_page_saved_evidence(self):
        from mlmom.pipeline_contracts import RunConfig
        from mlmom.persistence import ExperimentStore
        from mlmom.resources import Budget, SandboxManager
        from mlmom.role_tools import feature_data_tools
        import time
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);frame,context,_,_=execution_case(root,'regression')
            allowed=root/'allowed.csv';frame.iloc[:12].to_csv(allowed,index=False)
            config=RunConfig(reserve_cpus=0,reserve_memory_gb=0)
            budget=Budget(config,time.time());store=ExperimentStore(root/'store')
            try:
                with SandboxManager(budget=budget,max_workers=1) as sandbox:
                    p=SimpleNamespace(context=context,config=config,budget=budget,store=store,sandbox=sandbox)
                    tools=feature_data_tools(p,allowed,'initial')
                    measured=tools.invoke('query_csv',{'sql':'SELECT COUNT(*) AS n FROM train'})
                    self.assertEqual(measured.status,'ok',measured)
                    self.assertEqual(measured.data['rows'],[{'n':12}])
                    ref=measured.data['source_ref']
                    self.assertEqual(tools.invoke('read_evidence',{'reference':ref,'chars':15}).data['offset'],0)
                    resumed=feature_data_tools(p,allowed,'initial')
                    self.assertIn(ref,resumed.evidence_refs)
                    self.assertEqual(resumed.invoke('read_evidence',{'reference':ref}).status,'ok')
                    self.assertEqual(tools.invoke('query_csv',{'sql':'SELECT * FROM test'}).status,'error')
                    self.assertEqual(tools.invoke('read_evidence',{'reference':'evaluation:outer'}).status,'error')
                    table=tools.invoke('query_csv',{'sql':'SELECT * FROM train'})
                    table_ref=table.data['source_ref']
                    page=resumed.invoke('read_query_result',{'reference':ref})
                    self.assertEqual(page.status,'ok',page)
                    page=tools.invoke('read_query_result',{'reference':table_ref,'row_offset':10,'rows':1})
                    self.assertEqual(page.status,'ok',page)
                    self.assertEqual(len(page.data['rows']),1)
                    self.assertEqual(page.data['next_row_offset'],11)
                    self.assertEqual(tools.invoke('read_query_result',{'reference':'evaluation:outer'}).status,'error')
            finally:store.close()

    def test_wide_profile_marks_omitted_columns(self):
        from mlmom.feature_search import feature_profile
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,context,_,_=execution_case(root,'regression')
            context.feature_columns=[f'f{i}' for i in range(101)]
            frame=pd.DataFrame({name:np.arange(12.) for name in context.feature_columns})
            frame['t0']=np.arange(12.);frame.to_csv(root/'wide.csv',index=False)
            profile=feature_profile(dict(context=context.model_dump(mode='json'),train_path=str(root/'wide.csv')))['profile']
            self.assertEqual(profile['total_columns'],101)
            self.assertTrue(profile['columns_truncated'])
            self.assertIn('list_tables',profile['remaining_columns_via'])
            from mlmom.pipeline_contracts import RunConfig
            from mlmom.persistence import ExperimentStore
            from mlmom.resources import Budget, SandboxManager
            from mlmom.role_tools import feature_data_tools
            import time
            config=RunConfig(reserve_cpus=0,reserve_memory_gb=0);budget=Budget(config,time.time())
            store=ExperimentStore(root/'store')
            try:
                with SandboxManager(budget=budget,max_workers=1) as sandbox:
                    tools=feature_data_tools(SimpleNamespace(context=context,config=config,budget=budget,
                        store=store,sandbox=sandbox),root/'wide.csv','wide')
                    schema=tools.invoke('list_tables',{'table':'train','offset':100})
                    self.assertIn('f100',[c['name'] for c in schema.data['tables'][0]['columns']])
                    table=tools.invoke('query_csv',{'sql':'SELECT * FROM train'})
                    page=tools.invoke('read_query_result',{'reference':table.data['source_ref'],'column_offset':100})
                    self.assertEqual(page.status,'ok',page)
                    self.assertIn('f100',page.data['columns'])
                    self.assertEqual(page.data['rows'][0]['f100'],0.)
            finally:store.close()

    def test_preprocessing_reads_owned_cards_and_records_actual_versions(self):
        from mlmom.coding import build_coding_graph
        from mlmom.pipeline_contracts import RunConfig
        from mlmom.registry import SkillRegistry
        from tests.test_planning_limits import case, Store, ClockBudget, Sandbox, Coder
        from tests.fakes import PREPROCESSING_CODE
        registry=SkillRegistry(Path(__file__).resolve().parents[1]/'skills')
        class Client(Coder):
            def generate(self,role,instruction,payload,schema,tools=None):
                if role=='preprocessing_agent':
                    self.pre_tools=tools
                    self.loaded=tools.invoke('read_skills',{'skill_ids':['categorical_encoding']})
                    self.read_again=tools.invoke('read_skills',{'skill_ids':['categorical_encoding']})
                    self.foreign=tools.invoke('read_skills',{'skill_ids':['lightgbm']})
                    return schema(source=PREPROCESSING_CODE,spec=dict(description='Dense',output='dense',categorical_features=[]))
                return super().generate(role,instruction,payload,schema,tools)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);context,tools,_,plan=case();budget=ClockBudget();client=Client(budget,1)
            graph=build_coding_graph(client,workspace=root/'code',store=Store(root),plan=plan,
                plan_ref=root/'plan.json',skill_data=[],tools=tools,context=context,sandbox=Sandbox(budget),
                budget=budget,config=RunConfig(),round_id='r1',skill_registry=registry,
                preprocessing_skills=registry.load([],'regression',role='preprocessing_agent'))
            result=graph.invoke({'payload':{}})
            self.assertTrue(result['valid'],result)
            self.assertEqual(client.loaded.status,'ok')
            self.assertEqual(client.foreign.status,'error')
            self.assertEqual(client.read_again.data[0]['skill_id'],'categorical_encoding')
            self.assertEqual(result['candidate']['skill_versions']['categorical_encoding']['sha256'],
                             client.read_again.data[0]['sha256'])
            upgraded=registry.load([],'regression',role='preprocessing_agent',include_references=True,max_chars=48000)
            self.assertEqual(result['candidate']['skill_versions']['candidate_sdk']['sha256'],
                             next(c['sha256'] for c in upgraded if c['skill_id']=='candidate_sdk'))
            self.assertTrue({'inspect_api','read_skills','check_preprocessing'}<=set(client.pre_tools.tools))
            self.assertEqual(result['candidate']['skill_versions']['categorical_encoding']['version'],
                             registry.manifests['categorical_encoding'].version)

    def test_optional_preprocessing_probe_has_no_estimator_or_real_data(self):
        from mlmom.worker import check_preprocessing
        from tests.fakes import PREPROCESSING_CODE
        with tempfile.TemporaryDirectory() as tmp:
            result=check_preprocessing(dict(source=PREPROCESSING_CODE,config={},task_type='regression',
                worker_cpus=1,worker_memory_gb=2,worker_seconds=20,timeouts_enabled=False,
                feature_schema=[dict(name='number',semantic_type='numeric'),dict(name='category',semantic_type='categorical')]),Path(tmp))
            self.assertTrue(result['portable_equal'])
            self.assertTrue(result['shuffled_columns_checked'])


if __name__=='__main__':
    unittest.main()
