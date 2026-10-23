"""Regression checks for the lifecycle audit, using local fits and fake transport."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch, Mock

import numpy as np
import pandas as pd

from mlmom.model_spec import model_spec, normalize_protocol, constant_space, apply_overrides, hpo_child
from mlmom.sdk import PolicyCandidate, FitContext, _load_policy
from mlmom.execution import audit_fit, FidelityError, config_hash
from mlmom.llm_policy import NetworkCircuit, thinking_parameters
from mlmom.tools import Workspace
from mlmom.context_views import compact_planning_payload, previous_brief_summary, paged_evidence


class Estimator:
    def __init__(self, config, *, ignore_weights=False, spoof=None, hidden_log=False):
        self.config=copy.deepcopy(config)
        self.ignore_weights=ignore_weights
        self.spoof=spoof
        self.hidden_log=hidden_log

    def fit(self,X,y,context):
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        cls=RandomForestClassifier if 'classification' in context.task_type else RandomForestRegressor
        self.model=cls(**self.config['model_params'],random_state=42,n_jobs=1)
        self.model.fit(X,np.log1p(y) if self.hidden_log else y,sample_weight=None if self.ignore_weights else context.sample_weight)
        if self.spoof:
            for key,value in self.spoof.items():setattr(self.model,key,value)
        return self

    def predict(self,X,spec):return self.model.predict(X)
    def export_model_params(self):return self.config['model_params']
    def save(self,directory):
        import cloudpickle
        Path(directory).mkdir(parents=True,exist_ok=True)
        (Path(directory)/'estimator.pkl').write_bytes(cloudpickle.dumps(self))


class ModelContractV2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.X=pd.DataFrame({'x':np.arange(40),'z':np.arange(40)%3})

    def run_fit(self,config,kind='regression',**kwargs):
        config=dict(contract_version=2,tool_id='sklearn',model_params={'n_estimators':5},_code_hash='source',**config)
        y=pd.Series(np.exp(np.arange(40)/10)) if kind=='regression' else pd.Series(['a']*30+['b']*10)
        context=FitContext(kind,'rmse','minimize',42,1,2,30)
        model=PolicyCandidate(Estimator(config,**kwargs),config)
        model.fit(self.X,y,context)
        return model,context,config

    def test_full_spec_roundtrip_policies_hash_and_portable_predictions(self):
        model,context,config=self.run_fit(dict(target_transform={'name':'log1p'},weight_policy={'name':'tail_aware'},prediction_policy={'clip_min':0}))
        report=audit_fit(model,config['model_params'],config,context,self.root/'report.json',stage='full_refit')
        self.assertEqual(report['version'],2)
        self.assertEqual(report['configuration_hash'],config_hash(report['model_spec']))
        self.assertEqual(report['model_spec']['target_transform']['name'],'log1p')
        self.assertNotEqual(report['configuration_hash'],config_hash(config['model_params']))
        model.save(self.root/'saved')
        import cloudpickle
        estimator=cloudpickle.loads((self.root/'saved/estimator.pkl').read_bytes())
        restored=_load_policy(estimator,self.root/'saved')
        np.testing.assert_array_equal(model.predict(self.X,{'kind':'value'}),restored.predict(self.X,{'kind':'value'}))

    def test_custom_weight_policy_has_no_native_name_collision(self):
        model,context,config=self.run_fit({'weight_policy':{'name':'sqrt_balanced'}},'multiclass_classification')
        report=audit_fit(model,config['model_params'],config,context,self.root/'report.json',stage='full_refit')
        entry=report['components']['model']['fit_entry']
        self.assertIsNone(entry['params']['class_weight'])
        self.assertEqual(entry['weight_fingerprint'],entry['expected_weight_fingerprint'])

    def test_nested_components_do_not_apply_runtime_policies_twice(self):
        config=dict(target_transform={'name':'log1p'},weight_policy={'name':'tail_aware'})
        single,context,full=self.run_fit(config)
        nested=PolicyCandidate(PolicyCandidate(Estimator(full),full),full)
        nested.fit(self.X,pd.Series(np.exp(np.arange(40)/10)),FitContext('regression','rmse','minimize',42,1,2,30))
        np.testing.assert_array_equal(single.predict(self.X,{'kind':'value'}),nested.predict(self.X,{'kind':'value'}))

    def test_wrong_weights_cannot_be_hidden_by_post_fit_attribute(self):
        model,context,config=self.run_fit({'weight_policy':{'name':'sqrt_balanced'}},'multiclass_classification',
            ignore_weights=True,spoof={'class_weight':'sqrt_balanced'})
        with self.assertRaisesRegex(FidelityError,'sample_weight differs'):
            audit_fit(model,config['model_params'],config,context,self.root/'failed.json',stage='full_refit')
        report=json.loads((self.root/'failed.json').read_text())
        self.assertIsNone(report['components']['model']['actual_params']['class_weight'])

    def test_hidden_target_transform_rejected_at_backend_entry(self):
        model,context,config=self.run_fit({},hidden_log=True)
        with self.assertRaisesRegex(FidelityError,'target differs'):
            audit_fit(model,config['model_params'],config,context,self.root/'failed.json',stage='full_refit')

    def test_metadata_does_not_change_validation_protocol(self):
        expected=dict(strategy='kfold',folds=5,duplicate_policy='group_features')
        actual,meta=normalize_protocol(dict(expected,metric='rmse',refit=True),expected)
        self.assertEqual(actual,expected);self.assertEqual(meta,dict(metric='rmse',refit=True))
        with self.assertRaisesRegex(ValueError,'validation_protocol.folds'):
            normalize_protocol(dict(expected,folds=3),expected)

    def test_policy_hpo_routes_are_separate_from_backend_parameters(self):
        config=apply_overrides(dict(tool_id='sklearn',model_params={'n_estimators':5}),
            {'policy_params.weight_policy.name':'sqrt_balanced','n_estimators':8})
        self.assertEqual(config['model_params'],{'n_estimators':8})
        self.assertEqual(config['weight_policy'],{'name':'sqrt_balanced'})
        self.assertEqual(constant_space({'n_estimators':dict(type='int',low=8,high=8)}),{'n_estimators':8})

    def test_hpo_has_separate_intervention_and_parent(self):
        from mlmom.pipeline_contracts import HypothesisPlan
        plan=HypothesisPlan(hypothesis=dict(hypothesis_id='baseline',description='fixed baseline',rationale='control',baseline=True),
                            tool_id='sklearn',skill_ids=['sklearn'],search_space={'max_depth':dict(type='int',low=2,high=4)})
        child=hpo_child(plan,'baseline-fit')
        self.assertNotEqual(child.hypothesis.hypothesis_id,plan.hypothesis.hypothesis_id)
        self.assertFalse(child.hypothesis.baseline)
        self.assertEqual(child.intervention.parent_experiment_id,'baseline-fit')
        self.assertTrue(plan.hypothesis.baseline)


class NetworkPolicyTests(unittest.TestCase):
    def test_http_probe_has_no_internal_retry_and_is_shared_by_roles(self):
        import httpx
        from openai import OpenAI
        from mlmom.agents import RoleClient, LLMUnavailable
        from mlmom.llm_client import LLMSettings
        now=[0.];calls=[]
        circuit=NetworkCircuit(clock=lambda:now[0])
        def handler(request):
            calls.append(json.loads(request.content))
            if len(calls)<=4:return httpx.Response(502,json={'error':'upstream'})
            return httpx.Response(200,json=dict(id='ok',object='chat.completion',created=1,model='test',
                choices=[dict(index=0,finish_reason='stop',message=dict(role='assistant',content='{}'))]))
        client=RoleClient(LLMSettings(api_key='test'),deadline=time.time()+60,network_circuit=circuit)
        client.client.close()
        client.client=OpenAI(api_key='test',max_retries=0,http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        self.addCleanup(client.client.close)
        message=[dict(role='user',content='json')]
        with patch('mlmom.agents.time.sleep'):
            with self.assertRaises(LLMUnavailable):client._request_impl(message,role='planner')
            self.assertEqual(len(calls),3)
            now[0]=30
            with self.assertRaisesRegex(LLMUnavailable,'probe failed'):client._request_impl(message,role='coding_agent')
            self.assertEqual(len(calls),4);self.assertEqual(circuit.open_until,60)
            now[0]=60
            client._request_impl(message,role='training_agent')
            self.assertEqual(len(calls),5);self.assertIsNone(circuit.open_until)

    def test_unsupported_thinking_is_removed_once_without_changing_model(self):
        import httpx
        from openai import OpenAI
        from mlmom.agents import RoleClient
        from mlmom.llm_client import LLMSettings
        calls=[]
        def handler(request):
            body=json.loads(request.content);calls.append(body)
            if 'enable_thinking' in body:
                return httpx.Response(400,json={'error':{'message':'unsupported enable_thinking','type':'invalid_request_error'}})
            return httpx.Response(200,json=dict(id='ok',object='chat.completion',created=1,model='test',
                choices=[dict(index=0,finish_reason='stop',message=dict(role='assistant',content='{}'))]))
        client=RoleClient(LLMSettings(api_key='test',model='qwen/qwen3.8-27b'),deadline=time.time()+60)
        client.client.close();client.client=OpenAI(api_key='test',max_retries=0,http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        self.addCleanup(client.client.close)
        for role in ('context_agent','training_agent'):
            client._request_impl([dict(role='user',content='json')],role=role)
        self.assertEqual(len(calls),3)
        self.assertEqual(sum('enable_thinking' in body for body in calls),1)
        self.assertTrue(all(body['model']=='qwen/qwen3.8-27b' for body in calls))

    def test_three_errors_pause_one_probe_reopen_then_recover(self):
        now=[0.]
        circuit=NetworkCircuit(clock=lambda:now[0])
        for _ in range(3):circuit.failure(circuit.acquire())
        self.assertEqual(circuit.open_until,30)
        now[0]=30
        probe=circuit.acquire();self.assertTrue(probe[0])
        with self.assertRaises(TimeoutError):circuit.acquire(remaining=0)
        circuit.failure(probe);self.assertEqual(circuit.open_until,60)
        now[0]=60
        circuit.success(circuit.acquire());self.assertIsNone(circuit.open_until)
        self.assertFalse(circuit.acquire()[0])

    def test_only_one_waiting_thread_gets_probe(self):
        circuit=NetworkCircuit(threshold=1,cooldown=.02)
        circuit.failure(circuit.acquire());received=[];release=threading.Event()
        def call():
            token=circuit.acquire(remaining=1)
            received.append(token)
            if token[0]:release.wait(1);circuit.success(token)
        threads=[threading.Thread(target=call) for _ in range(3)]
        for thread in threads:thread.start()
        time.sleep(.06)
        self.assertEqual(len(received),1)
        release.set()
        for thread in threads:thread.join(1)
        self.assertEqual(sum(token[0] for token in received),1)

    def test_thinking_only_off_roles_have_parameters_without_caps(self):
        from mlmom.agents import RoleClient
        from mlmom.llm_client import LLMSettings
        client=RoleClient(LLMSettings(api_key='test',model='qwen/qwen3.8-27b'),deadline=time.time()+10)
        self.addCleanup(client.client.close)
        response=NS(choices=[NS(message=NS(model_dump=lambda **kw:dict(role='assistant',content='{}')),finish_reason='stop')],usage=None)
        response.model_dump=lambda **kw:dict(choices=[],usage=None)
        with patch.object(client.client,'with_options') as options:
            options.return_value.chat.completions.create.return_value=response
            client._request_impl([dict(role='user',content='json')],role='context_agent')
            first=options.return_value.chat.completions.create.call_args.kwargs
            self.assertEqual(first['extra_body'],{'enable_thinking':False})
            client._request_impl([dict(role='user',content='json')],role='planner')
            second=options.return_value.chat.completions.create.call_args.kwargs
            self.assertNotIn('extra_body',second)
            for key in ('max_tokens','max_completion_tokens','thinking_budget'):
                self.assertNotIn(key,first);self.assertNotIn(key,second)
        self.assertEqual(thinking_parameters('deepseek/deepseek-v4.1-flash-1017','off'),{'thinking':{'type':'disabled'}})


class ContextAndPatchTests(unittest.TestCase):
    def test_unchanged_preprocessing_and_parameter_only_change_reuse_source(self):
        from mlmom.coding import build_coding_graph
        from mlmom.persistence import ExperimentStore
        from mlmom.pipeline_contracts import RunConfig
        from tests.test_planning_limits import case, ClockBudget
        from tests.fakes import PREPROCESSING_CODE, MODEL_CODE
        context,tools,_,plan=case()
        old=plan.model_dump(mode='json');old['model_params']={'n_estimators':4}
        plan.model_params={'n_estimators':5}
        previous=dict(plan=old,dependencies_match=True,preprocessing_spec=dict(module='preprocessing.py',factory='create_preprocessor',
            description='unchanged',output='dense',constraints=[]),files={'candidate.py':MODEL_CODE,'preprocessing.py':PREPROCESSING_CODE})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);store=ExperimentStore(root)
            client=NS(generate=Mock(side_effect=AssertionError('Source reuse must not call an LLM')))
            sandbox=NS(run=lambda *args,**kwargs:dict(status='completed'))
            graph=build_coding_graph(client,workspace=root/'workspace',store=store,plan=plan,plan_ref=root/'plan.json',
                skill_data=[],tools=tools,context=context,sandbox=sandbox,budget=ClockBudget(),config=RunConfig(timeouts_enabled=True),
                round_id='r2',previous=previous)
            try:
                result=graph.invoke({'payload':{},'attempts':0,'errors':[]})
            finally:
                store.close()
            self.assertTrue(result['valid'],result)
            client.generate.assert_not_called()
            self.assertEqual((root/'workspace/candidate.py').read_text(),MODEL_CODE)

    def test_false_lightgbm_rejection_is_an_advisory_concern_with_local_probe(self):
        from mlmom.planning import ground_semantic_report,categorical_probe
        from mlmom.pipeline_contracts import ValidationReport
        from tests.test_planning_limits import case
        context,_,_,plan=case()
        report=ground_semantic_report(ValidationReport(valid=False,errors=['unseen categorical NaN is unsafe']),plan,context)
        self.assertTrue(report.valid);self.assertTrue(report.warnings)
        self.assertEqual(categorical_probe()['status'],'passed')

    def test_patches_are_atomic_scoped_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace=Workspace(Path(tmp),writable=lambda path:path!='preprocessing.py')
            original='x = 1\ny = 2\n';workspace.write_file('candidate.py',original)
            digest=hashlib.sha256(original.encode()).hexdigest()
            repair=workspace.tools(repair=True)
            self.assertNotIn('write_file',repair.tools)
            with self.assertRaisesRegex(ValueError,'entire'):
                workspace.apply_patches([dict(path='candidate.py',sha256=digest,old_text=original,new_text='x=3')])
            with self.assertRaisesRegex(ValueError,'Stale'):
                workspace.apply_patches([dict(path='candidate.py',sha256='0'*64,old_text='x = 1',new_text='x = 3')])
            workspace.apply_patches([dict(path='candidate.py',sha256=digest,old_text='x = 1',new_text='x = 3')])
            self.assertEqual(workspace.read_file('candidate.py'),'x = 3\ny = 2\n')

    def test_compact_context_preserves_requirements_all_ids_and_critical_evidence(self):
        observations=[dict(observation_id=f'obs-{i}',source_ref=f'csv:{i}',description='large '*200,numbers={'n':i}) for i in range(40)]
        critical={'finding_id':'critical','evidence':{'risk':'leakage'}}
        payload=dict(task={'target':'y'},reconnaissance={'status':'completed','observations':observations},
            dataset_profile={'large':'profile '*500},planning_brief={'requirements':{'metric':'rmse'},
            'mandatory_findings':[critical],'ranked_findings':[dict(finding_id='critical',fact=critical)]})
        small=compact_planning_payload(payload)
        self.assertLess(len(json.dumps(small)),len(json.dumps(payload)))
        self.assertEqual(small['planning_brief']['requirements'],payload['planning_brief']['requirements'])
        self.assertEqual(small['planning_brief']['mandatory_findings'],[critical])
        self.assertEqual(len(small['observation_index']),40)
        self.assertNotIn('fact',small['planning_brief']['ranked_findings'][0])
        tools=paged_evidence({'allowed':{'full':observations}})
        self.assertEqual(tools.invoke('read_evidence',{'reference':'secret'}).status,'error')


if __name__=='__main__':unittest.main()
