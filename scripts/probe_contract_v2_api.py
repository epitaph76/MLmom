"""Four tiny diagnostic calls on the explicitly authorized personal RouterAI account."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--key-stdin',action='store_true')
    parser.add_argument('--complete-remaining',action='store_true')
    args=parser.parse_args()
    # Deliberately do not load .env or organizer credentials.
    key=sys.stdin.readline().strip() if args.key_stdin else os.environ.get('MLMOM_TEST_API_KEY','').strip()
    if not key:raise ValueError('Supply the personal diagnostic key through stdin or MLMOM_TEST_API_KEY')
    from mlmom.agents import RoleClient
    from mlmom.llm_client import LLMSettings
    from mlmom.pipeline_contracts import PlannerContextDraft, TrainingDecision, PlanBatch
    from mlmom.coding import CodePatch
    from mlmom.tools import Workspace
    from mlmom.tracing import TraceSink
    if args.complete_remaining:
        from pydantic import BaseModel
        previous=json.loads((args.output/'report.json').read_text(encoding='utf-8'))
        if previous['http_attempts']!=3 or len(previous['checks'])!=3:
            raise ValueError('The continuation is only for the one remaining request of this diagnostic')
        if (args.output/'fourth-call/report.json').exists():
            raise ValueError('The final diagnostic request has already been used')
        class Both(BaseModel):
            plan: PlanBatch
            patch: CodePatch
        trace=TraceSink(args.output/'fourth-call','contract-v2-fourth-request',secrets=[key])
        requests=[];usage=[]
        def last_request(event):
            if event.get('status')=='started':
                if requests:raise RuntimeError('Four-request diagnostic allowance exhausted')
                requests.append(event['role'])
        client=RoleClient(LLMSettings(api_key=key,base_url='https://routerai.ru/api/v1',
            model='deepseek/deepseek-v4.1-flash-1017',timeout_seconds=90),deadline=time.time()+100,
            max_steps=0,request_callback=last_request,usage_callback=usage.append,trace=trace)
        source='def value():\n    return 1\n\n# Keep this comment unchanged.\n'
        digest=hashlib.sha256(source.encode()).hexdigest()
        status='failed';error_type=None
        try:
            result=client.generate('coding_agent',
                'Return both a fixed single-fit sklearn PlanBatch and a CodePatch changing only return 1 to return 2. '
                'The backend/skill are explicitly available; no missing context. Preserve the exact supplied hypothesis and source hash. '
                'Set plan model_params n_estimators=5, strategy=single, skill_ids=[sklearn], no policies or search.',
                dict(hypothesis=dict(hypothesis_id='tiny-baseline',description='Small tree baseline',rationale='Fast check',baseline=True),
                     task=dict(task_type='regression',primary_metric='rmse',metric_direction='minimize',target_columns=['y'],validation_strategy='kfold',validation_folds=5),
                     ml_tools=[dict(tool_id='sklearn',available=True,version='1.5.2',skill_ids=['sklearn'],task_types=['regression'])],
                     skills=[dict(skill_id='sklearn',status='ready')],current_files={'candidate.py':source},file_hashes={'candidate.py':digest}),Both)
            if len(result.plan.plans)!=1 or result.plan.plans[0].strategy!='single':raise ValueError('Wrong plan')
            with tempfile.TemporaryDirectory() as tmp:
                workspace=Workspace(Path(tmp));workspace.write_file('candidate.py',source)
                workspace.apply_patches([p.model_dump() for p in result.patch.patches])
                if workspace.read_file('candidate.py')!=source.replace('return 1','return 2'):raise ValueError('Wrong patch')
            status='passed'
        except Exception as error:error_type=type(error).__name__
        finally:client.client.close();trace.close()
        report=dict(status=status,diagnostic_only=True,http_attempts=len(requests),cumulative_http_attempts=3+len(requests),
                    checks=['planner_with_backend_metadata','scoped_code_patch'],error_type=error_type,usage=usage,
                    prior_fixture_issue='The third request correctly asked for backend metadata omitted by the tiny diagnostic fixture.',
                    organizer_gateway_verified=False)
        (args.output/'fourth-call/report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(report),flush=True)
        return 0 if status=='passed' else 1
    args.output.mkdir(parents=True,exist_ok=True)
    attempts=[];usage=[];checks=[]
    def request(event):
        if event.get('status')=='started':
            if len(attempts)>=4:raise RuntimeError('Diagnostic HTTP allowance exhausted')
            attempts.append(dict(role=event['role'],attempt=event['attempt']))
    trace=TraceSink(args.output,'contract-v2-personal-api',secrets=[key])
    client=RoleClient(LLMSettings(api_key=key,base_url='https://routerai.ru/api/v1',
        model='deepseek/deepseek-v4.1-flash-1017',timeout_seconds=90),deadline=time.time()+360,
        max_steps=0,request_callback=request,usage_callback=usage.append,trace=trace)
    source='def value():\n    return 1\n\n# Keep this comment unchanged.\n'
    digest=hashlib.sha256(source.encode()).hexdigest()
    cases=[
        ('context_agent','Summarize the task in one sentence and rank the supplied finding using its exact ID.',
         dict(task='Predict y from numeric x using RMSE.',findings=[dict(finding_id='finding-1',description='x has missing values')]),PlannerContextDraft),
        ('training_agent','Return action=refit and a short reason. No search trials remain.',
         dict(remaining_trials=0,default_action='refit'),TrainingDecision),
        ('planner','Return one fixed single-fit sklearn plan for the supplied hypothesis. Use n_estimators=5, no search, no transforms or custom weights.',
         dict(hypotheses=[dict(hypothesis_id='tiny-baseline',description='Small tree baseline',rationale='Fast check',baseline=True)],
              task_type='regression',metric='rmse'),PlanBatch),
        ('coding_agent','Return a CodePatch that changes only return 1 to return 2. Preserve the function and comment; use the supplied hash.',
         dict(current_files={'candidate.py':source},file_hashes={'candidate.py':digest}),CodePatch)]
    try:
        for role,instruction,payload,schema in cases:
            try:
                answer=client.generate(role,instruction,payload,schema)
                if role=='context_agent' and any(f.finding_id!='finding-1' for f in answer.ranked_findings):
                    raise ValueError('Unknown finding ID')
                if role=='training_agent' and answer.action!='refit':raise ValueError('Wrong action')
                if role=='planner' and (len(answer.plans)!=1 or answer.plans[0].strategy!='single'):
                    raise ValueError('Wrong plan')
                if role=='coding_agent':
                    with tempfile.TemporaryDirectory() as tmp:
                        workspace=Workspace(Path(tmp));workspace.write_file('candidate.py',source)
                        workspace.apply_patches([p.model_dump() for p in answer.patches])
                        if workspace.read_file('candidate.py')!=source.replace('return 1','return 2'):
                            raise ValueError('Patch did not preserve unrelated source')
                checks.append(dict(role=role,status='passed'))
            except Exception as error:
                checks.append(dict(role=role,status='failed',error_type=type(error).__name__,status_code=getattr(error,'status_code',None)))
                break  # No paid repetition after rejection or invalid output.
    finally:
        client.client.close();trace.close()
    result=dict(status='passed' if len(checks)==4 and all(c['status']=='passed' for c in checks) else 'failed',
        diagnostic_only=True,organizer_gateway_verified=False,model=client.model,endpoint='https://routerai.ru/api/v1',
        http_attempts=len(attempts),checks=checks,usage=usage,
        thinking_unsupported=client.thinking_unsupported,
        limitations=['This is a small protocol/format check, not a full ML acceptance run or an organizer-Qwen test.'])
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],http_attempts=len(attempts),checks=checks)),flush=True)
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
