from __future__ import annotations
import json
import tempfile
import time
import unittest
import httpx
from openai import APIStatusError
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from openai import APITimeoutError
from mlmom.agents import RoleClient,LLMUnavailable
from mlmom.llm_client import LLMSettings
from mlmom.coding import CodeBundle
from mlmom.tools import Workspace


class AgentProtocolTests(unittest.TestCase):
    def test_function_call_protocol_writes_workspace_then_validates_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace=Workspace(Path(tmp));tools=workspace.tools()
            client=RoleClient(LLMSettings(api_key='unit-test-only'),deadline=time.time()+10)
            self.addCleanup(client.client.close)
            responses=[
                {'role':'assistant','tool_calls':[{'id':'call-1','type':'function','function':{
                    'name':'write_file','arguments':json.dumps({'path':'candidate.py','content':'x=1\n'})}}]},
                {'role':'assistant','content':json.dumps({'files':{'candidate.py':'x=1\n'}})},
            ]
            with patch.object(client,'_request',side_effect=responses) as request:
                bundle=client.generate('coding_agent','Test role',{},CodeBundle,tools)
                history=request.call_args_list[-1].args[0]
                messages=[m for m in history if m['role']=='tool']
                self.assertEqual(messages[0]['tool_call_id'],'call-1')
                self.assertEqual(json.loads(messages[0]['content'])['status'],'ok')
            self.assertEqual(bundle.files['candidate.py'],workspace.read_file('candidate.py'))
            self.assertEqual(tools.invoke('write_file',{'path':'../escape.py','content':'x=1'}).status,'error')
            self.assertEqual(tools.invoke('read_file',{'path':'candidate.py','unexpected':True}).status,'error')

    def test_expired_deadline_rejects_before_api_request(self):
        client=RoleClient(LLMSettings(api_key='unit-test-only'),deadline=time.time()-1)
        self.addCleanup(client.client.close)
        with self.assertRaises(LLMUnavailable):client.complete([{'role':'user','content':'{}'}])

    def test_standard_environment_overrides_legacy_provider(self):
        env={'OPENAI_API_KEY':'standard-test-key','OPENAI_MODEL':'provided-model','OPENAI_BASE_URL':'https://example.invalid/v1',
             'ROUTERAI_API_KEY':'legacy-test-key','ROUTERAI_MODEL':'legacy-model','ROUTERAI_BASE_URL':'https://legacy.invalid/v1',
             'ROUTERAI_MAX_TOKENS':'4000'}
        with tempfile.TemporaryDirectory() as tmp,patch.dict('os.environ',env,clear=True):
            settings=LLMSettings.from_env(str(Path(tmp)/'absent.env'))
        self.assertEqual(settings.api_key,'standard-test-key')
        self.assertEqual(settings.model,'provided-model')
        self.assertEqual(settings.base_url,'https://example.invalid/v1')
        self.assertIsNone(settings.max_tokens)

    def test_platform_model_optional_and_never_uses_local_credentials(self):
        with patch.dict('os.environ',{'OPENAI_API_KEY':'test-only','OPENAI_BASE_URL':'https://proxy.invalid/v1',
                                    'ROUTERAI_MODEL':'legacy-model'},clear=True):
            settings=LLMSettings.from_platform_env()
            self.assertEqual(settings.model,'qwen/qwen3.8-27b')
            self.assertEqual(settings.base_url,'https://proxy.invalid/v1')
        with patch.dict('os.environ',{'ROUTERAI_API_KEY':'legacy-only'},clear=True):
            with self.assertRaisesRegex(ValueError,'OPENAI_BASE_URL'):
                LLMSettings.from_platform_env()

    def test_local_settings_require_explicit_endpoint_and_default_to_organizer_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            absent=str(Path(tmp)/'absent.env')
            with patch.dict('os.environ',{'OPENAI_API_KEY':'test-only'},clear=True):
                with self.assertRaisesRegex(ValueError,'OPENAI_BASE_URL'):
                    LLMSettings.from_env(absent)
            with patch.dict('os.environ',{'OPENAI_API_KEY':'test-only',
                                        'OPENAI_BASE_URL':'https://proxy.invalid/v1'},clear=True):
                settings=LLMSettings.from_env(absent)
                self.assertEqual(settings.base_url,'https://proxy.invalid/v1')
                self.assertEqual(settings.model,'qwen/qwen3.8-27b')

    def test_platform_accepts_supplied_endpoint_and_model_without_dotenv(self):
        for url in ['https://changellenge.up.railway.app/v1','https://api.alfahack-gateway26.ru/v1']:
            with self.subTest(url=url),patch.dict('os.environ',{
                    'OPENAI_API_KEY':'platform-test-only','OPENAI_BASE_URL':url,
                    'OPENAI_MODEL':'proxy-task-model','ROUTERAI_API_KEY':'private-test-only',
                    'ROUTERAI_BASE_URL':'https://private.invalid/v1'},clear=True),\
                    patch('mlmom.llm_client.load_dotenv') as dotenv:
                settings=LLMSettings.from_platform_env()
                self.assertEqual(settings.base_url,url)
                self.assertEqual(settings.api_key,'platform-test-only')
                self.assertEqual(settings.model,'proxy-task-model')
                dotenv.assert_not_called()

    def test_daily_budget_error_is_actionable_and_does_not_echo_provider_body(self):
        client=RoleClient(LLMSettings(api_key='test-only'),deadline=time.time()+10)
        self.addCleanup(client.client.close)
        error=APIStatusError('request failed',response=httpx.Response(402,request=httpx.Request('POST','https://proxy.invalid/v1')),
                             body={'error':{'message':'Budget has been exceeded; sensitive provider details'}})
        with patch.object(client.client,'with_options',side_effect=error) as request:
            with self.assertRaisesRegex(LLMUnavailable,'resets at 03:00 MSK') as result:
                client.complete([{'role':'user','content':'{}'}])
        request.assert_called_once()
        self.assertNotIn('sensitive',str(result.exception))

    def test_retry_observability_and_timeout_remain_within_global_deadline(self):
        events=[];usage=[]
        client=RoleClient(LLMSettings(api_key='test-only',timeout_seconds=300),deadline=time.time()+10,
                          request_callback=events.append,usage_callback=usage.append)
        self.addCleanup(client.client.close)
        message={'role':'assistant','content':json.dumps({'files':{'candidate.py':'x=1'}})}
        response=SimpleNamespace(usage=SimpleNamespace(model_dump=lambda:{'total_tokens':7}),
                                 choices=[SimpleNamespace(message=SimpleNamespace(model_dump=lambda **kwargs:message))])
        with patch.object(client.client,'with_options') as options,patch('mlmom.agents.time.sleep'):
            options.return_value.chat.completions.create.side_effect=[
                APITimeoutError(request=httpx.Request('POST','https://proxy.invalid/v1')),response]
            result=client.generate('coding_agent','Minimal test',{},CodeBundle)
        self.assertEqual(result.files,{'candidate.py':'x=1'})
        self.assertEqual([event['status'] for event in events],['started','retry','started'])
        self.assertTrue(all(event['role']=='coding_agent' for event in events))
        self.assertEqual(usage[0]['total_tokens'],7)
        self.assertEqual(usage[0]['role'],'coding_agent')
        self.assertTrue(all(0<call.kwargs['timeout']<=10 for call in options.call_args_list))
        self.assertNotIn('test-only',json.dumps(events+usage))

    def test_provider_truncation_has_explicit_diagnostic_without_client_limit(self):
        usage=[]
        client=RoleClient(LLMSettings(api_key='test-only'),deadline=time.time()+10,usage_callback=usage.append)
        self.addCleanup(client.client.close)
        response=SimpleNamespace(usage=SimpleNamespace(model_dump=lambda:{'completion_tokens':4000}),
            choices=[SimpleNamespace(finish_reason='length',message=SimpleNamespace(model_dump=lambda **kwargs:{'role':'assistant'}))])
        with patch.object(client.client,'with_options') as options:
            options.return_value.chat.completions.create.return_value=response
            with self.assertRaisesRegex(ValueError,'the client did not request an output token limit'):
                client.generate('coding_agent','Test',{},CodeBundle)
        self.assertEqual(usage[0]['finish_reason'],'length')


if __name__=='__main__':unittest.main()
