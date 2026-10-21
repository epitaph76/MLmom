from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI

from mlmom.agents import RoleClient, LLMUnavailable
from mlmom.coding import CodeBundle
from mlmom.llm_client import LLMSettings, RouterAIChatClient
from mlmom.registry import SkillRegistry
from mlmom.tools import Workspace
from mlmom.tracing import TraceSink, bind, emit, span, current_context, current_sink


ROOT = Path(__file__).resolve().parents[1]
SECRET = 'trace-test-secret-12345'


def completion(content='{}', finish='stop', **extra):
    body=dict(id='provider-response', object='chat.completion', created=1,
        model='test-model', choices=[{'index':0,'finish_reason':finish,
            'message':{'role':'assistant','content':content}}],
        usage={'prompt_tokens':11,'completion_tokens':7,'total_tokens':18})
    body.update(extra)
    return httpx.Response(200, json=body,
        headers={'x-request-id':'provider-request','set-cookie':SECRET})


class TraceTests(unittest.TestCase):
    @contextmanager
    def directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.sinks=[]
            try:
                yield tmp
            finally:
                for sink in self.sinks:
                    sink.close()

    def sink(self, path):
        sink = TraceSink(path, 'run-one', 'task-one', secrets=[SECRET])
        self.sinks.append(sink)
        return sink

    def client(self, sink, handler):
        settings = LLMSettings(api_key=SECRET,base_url='https://test.invalid/v1',timeout_seconds=2,max_tokens=12000)
        client = RoleClient(settings,deadline=time.time()+60,trace=sink)
        client.client.close()
        client.client = OpenAI(api_key=SECRET,base_url=settings.base_url,max_retries=0,
                               http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        self.addCleanup(client.client.close)
        return client

    def records(self, path):
        return [json.loads(line) for line in (path/'trace.jsonl').read_text(encoding='utf-8').splitlines()]

    def payload(self, path, record):
        return json.loads((path/record['payload']['path']).read_text(encoding='utf-8'))

    def test_physical_attempts_retry_fallback_and_full_replay(self):
        with self.directory() as tmp:
            root=Path(tmp);sink=self.sink(root);calls=[]
            def handler(request):
                calls.append(json.loads(request.content))
                if len(calls)==1:
                    raise httpx.ReadTimeout('timeout '+SECRET,request=request)
                if len(calls)==2:
                    return httpx.Response(400,json={'error':{'message':'unsupported response_format '+SECRET}})
                return completion('{"files":{"candidate.py":"x=1"}}', metadata={SECRET:'secret key name',
                    'api_key':SECRET,'headers':{'Authorization':SECRET}})
            client=self.client(sink,handler)
            with sink.activate(stage='planning',hypothesis_id='h1'),patch('mlmom.agents.time.sleep'):
                result=client.generate('coder','implement',{'input':'secret '+SECRET},CodeBundle)
            self.assertEqual(result.files['candidate.py'],'x=1')
            self.assertEqual(len(calls),3)
            self.assertIn('response_format',calls[1]);self.assertNotIn('response_format',calls[2])
            rows=self.records(root)
            starts=[r for r in rows if r['kind']=='llm.attempt.started']
            ends=[r for r in rows if r['kind'] in {'llm.attempt.failed','llm.attempt.completed'}]
            self.assertEqual(len(starts),3);self.assertEqual(len(ends),3)
            self.assertEqual({r['span_id'] for r in starts},{r['span_id'] for r in ends})
            self.assertEqual([r['attempt'] for r in starts],[1,2,3])
            self.assertTrue(all(r['task_id']=='task-one' and r['hypothesis_id']=='h1' for r in starts))
            params=self.payload(root,starts[0]);self.assertIn('messages',params)
            for params in [params,*calls]:
                self.assertTrue({'max_tokens','max_completion_tokens','max_output_tokens'}.isdisjoint(params))
            response=next(r for r in rows if r['kind']=='llm.response')
            self.assertEqual(response['request_id'],'provider-request')
            self.assertEqual(self.payload(root,response)['id'],'provider-response')
            self.assertEqual(self.payload(root,response)['usage']['total_tokens'],18)
            sink.report()
            for file in [root/'trace.jsonl',root/'trace.md',*root.glob('trace_payloads/*')]:
                text=file.read_text(encoding='utf-8')
                self.assertNotIn(SECRET,text)
                self.assertNotIn('Authorization',text)
            for (body,) in sink.db.execute('SELECT body FROM trace_events'):
                self.assertNotIn(SECRET,body)

    def test_invalid_json_and_length_preserve_responses_before_failure(self):
        for content,finish in [('not JSON','stop'),('', 'length')]:
            with self.subTest(finish=finish),self.directory() as tmp:
                root=Path(tmp);sink=self.sink(root)
                client=self.client(sink,lambda request:completion(content,finish))
                with self.assertRaises(ValueError):
                    client.generate('planner','plan',{},CodeBundle)
                rows=self.records(root)
                raw=self.payload(root,next(r for r in rows if r['kind']=='llm.response'))
                self.assertEqual(raw['choices'][0]['message']['content'],content)
                self.assertEqual(rows[-1]['kind'],'llm.role.failed')

    def test_terminal_api_error_and_no_usage_are_traced(self):
        with self.directory() as tmp:
            root=Path(tmp);sink=self.sink(root)
            client=self.client(sink,lambda request:httpx.Response(402,json={'error':{'message':'Budget has been exceeded '+SECRET}}))
            with self.assertRaises(LLMUnavailable):client.complete([{'role':'user','content':'{}'}])
            rows=self.records(root)
            self.assertEqual(len([r for r in rows if r['kind']=='llm.attempt.started']),1)
            self.assertEqual(rows[-1]['kind'],'llm.completion.failed')
            client=self.client(sink,lambda request:completion(usage=None))
            self.assertEqual(client.complete([{'role':'user','content':'{}'}]),'{}')
            response=next(r for r in self.records(root) if r['kind']=='llm.response')
            self.assertIsNone(self.payload(root,response)['usage'])

    def test_tool_step_limit_keeps_provider_calls_and_results(self):
        with self.directory() as tmp:
            root=Path(tmp);sink=self.sink(root)
            choices=[{'index':0,'finish_reason':'tool_calls','message':{'role':'assistant','content':None,
                'tool_calls':[{'id':'tool-one','type':'function','function':{'name':'list_files','arguments':'{}'}}]}}]
            client=self.client(sink,lambda request:completion(choices=choices))
            client.max_steps=1
            with self.assertRaisesRegex(ValueError,'exceeded 1 tool steps'):
                client.generate('coding','test',{},CodeBundle,Workspace(root/'workspace').tools())
            rows=self.records(root)
            self.assertTrue(any(r['kind']=='tool.result' and r['tool_call_id']=='tool-one' for r in rows))
            self.assertEqual(rows[-1]['kind'],'llm.role.failed')

    def test_parallel_context_isolation_and_resume_sequence(self):
        with self.directory() as tmp:
            root=Path(tmp);sink=self.sink(root);barrier=threading.Barrier(2)
            def worker(task):
                with sink.activate(task_id=task),span('stage',stage='inspect'):
                    captured=current_context()
                    barrier.wait()
                    with bind(hypothesis_id='same-id'):
                        emit('observation',{'task':task})
                    self.assertEqual(current_context(),captured)
            with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(worker,['a','b']))
            self.assertIsNone(current_sink())
            rows=self.records(root)
            observations=[r for r in rows if r['kind']=='observation']
            self.assertEqual({r['task_id'] for r in observations},{'a','b'})
            self.assertTrue(all(self.payload(root,r)['task']==r['task_id'] for r in observations))
            sink.close()
            resumed=self.sink(root)
            with resumed.activate():emit('resume')
            allrows=self.records(root)
            self.assertEqual([r['seq'] for r in allrows],list(range(1,len(allrows)+1)))

    def test_interrupted_payload_write_is_atomic_and_reports_damaged_history(self):
        with self.directory() as tmp:
            root=Path(tmp);sink=self.sink(root)
            with patch('mlmom.tracing.os.replace',side_effect=OSError('simulated interruption')):
                with self.assertRaises(OSError):sink.emit('interrupted',{'value':'incomplete'})
            self.assertEqual(list((root/'trace_payloads').glob('*')),[])
            missing=sink.emit('llm.response',{'id':'missing','usage':{'total_tokens':5}})
            damaged=sink.emit('llm.response',{'id':'damaged','usage':{'total_tokens':7}})
            (root/missing['payload']['path']).unlink()
            (root/damaged['payload']['path']).write_text('{broken',encoding='utf-8')
            report=sink.report().read_text(encoding='utf-8')
            self.assertIn('Trace integrity warnings',report)
            self.assertIn('FileNotFoundError',report)
            self.assertIn('SHA256 mismatch',report)
            self.assertIn('unknown',report)
            # A repeated snapshot repairs an old incomplete hash-named file.
            repaired=sink.emit('llm.response',{'id':'damaged','usage':{'total_tokens':7}})
            self.assertEqual(repaired['payload']['path'],damaged['payload']['path'])
            self.assertEqual(self.payload(root,repaired)['id'],'damaged')
            sink.close()
            self.assertTrue((root/'trace.md').is_file())

    def test_skill_snapshots_and_tool_arguments_results(self):
        with self.directory() as tmp:
            root=Path(tmp);sink=self.sink(root)
            with sink.activate(stage='coding'):
                loaded=SkillRegistry(ROOT/'skills').load(['sklearn'],'regression')
                tools=Workspace(root/'workspace').tools()
                with bind(tool_call_id='provider-tool-1'):
                    result=tools.invoke('write_file',{'path':'candidate.py','content':'x=1'})
                failure=tools.invoke('read_file',{'path':'../outside.py'})
            self.assertEqual(result.status,'ok');self.assertEqual(failure.status,'error')
            rows=self.records(root)
            skills=self.payload(root,next(r for r in rows if r['kind']=='skills.loaded'))
            self.assertEqual(skills,loaded)
            self.assertTrue(all(item['content'] and item['sha256'] for item in skills))
            writes=[r for r in rows if r.get('tool_call_id')=='provider-tool-1']
            self.assertTrue(writes)
            results=[self.payload(root,r) for r in rows if r['kind']=='tool.result']
            self.assertEqual([r['status'] for r in results],['ok','error'])

    def test_legacy_client_shares_traced_sdk_without_hidden_retries(self):
        with self.directory() as tmp:
            root=Path(tmp);sink=self.sink(root);calls=[]
            client=RouterAIChatClient(LLMSettings(api_key=SECRET,max_tokens=4000),trace=sink)
            self.addCleanup(client.close)
            client._role.client.close()
            def handler(request):
                calls.append(request)
                return completion()
            client._role.client=OpenAI(api_key=SECRET,max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(handler)))
            client._client=client._role.client
            self.assertEqual(client.complete([{'role':'user','content':'{}'}]),'{}')
            self.assertEqual(len(calls),1)
            self.assertTrue({'max_tokens','max_completion_tokens','max_output_tokens'}.isdisjoint(json.loads(calls[0].content)))
            self.assertEqual(len([r for r in self.records(root) if r['kind']=='llm.response']),1)


if __name__=='__main__':unittest.main()
