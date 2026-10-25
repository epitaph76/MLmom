"""Exercise real role payloads/tools with a persistent 20-attempt API ceiling.

Development diagnostic only. Unrelated roles use deterministic reference sources;
this never starts a full competition run. Repeated invocations share --ledger.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlmom.agents import RoleClient
from mlmom.coding import build_coding_graph
from mlmom.feature_contracts import FeatureProposal, FeatureSet
from mlmom.feature_search import FEATURE_PROMPT, feature_profile, feature_runtime_info
from mlmom.llm_client import LLMSettings
from mlmom.model_spec import apply_overrides
from mlmom.persistence import ExperimentStore, atomic_json, tree_hash
from mlmom.pipeline_contracts import (HypothesisBatch, PlanBatch, RunConfig,
    TrainingCheckpoint, ValidationReport)
from mlmom.planning import build_planning_graph
from mlmom.registry import SkillRegistry, installed_tools
from mlmom.resources import Budget, SandboxManager
from mlmom.role_tools import feature_data_tools
from mlmom.training_control import decide
from mlmom.tracing import TraceSink
from tests.test_execution import execution_case


MODEL = 'deepseek/deepseek-v4.1-flash-1017'


def reference(path):
    text = (ROOT / path).read_text(encoding='utf-8')
    return re.search(r'```python\n(.*?)\n```', text, re.S).group(1)


class AttemptLimit:
    def __init__(self, path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else dict(
            limit=20, model=MODEL, attempts=[])
        if self.data['limit'] != 20 or self.data['model'] != MODEL:
            raise ValueError('Unexpected diagnostic ledger')
        atomic_json(path, self.data)

    def __call__(self, event):
        if event['status'] != 'started':
            return
        if len(self.data['attempts']) >= 20:
            raise RuntimeError('Diagnostic limit reached: no more than 20 HTTP attempts')
        # Commit before sending; process interruptions and retries still count.
        self.data['attempts'].append(dict(event, number=len(self.data['attempts'])+1,
                                          started_at=time.time()))
        atomic_json(self.path, self.data)


def setup_case(root, kind='regression'):
    root.mkdir(parents=True)
    frame, context, plan, code = execution_case(root, kind)
    plan.cpus, plan.memory_gb, plan.time_limit_seconds = 1, 2, 60
    plan.max_trials, plan.strategy = 1, 'single'
    plan.preprocessing = ['Median numeric imputation; train-fitted categorical one-hot encoding']
    frame.iloc[:42].to_csv(root/'allowed-train.csv', index=False)
    return frame, context, plan, code


def check_generated(sandbox, context, plan, code, root):
    result = sandbox.run(dict(action='train', code_dir=str(code), code_hash=tree_hash(code),
        plan=plan.model_dump(mode='json'), context=context.model_dump(mode='json'),
        train_path=str(root/'allowed-train.csv'), seed=42, training_control=False),
        root/'fit', seconds=60, cpus=1, memory_gb=2)
    atomic_json(root/'fit-result.json', result)
    if result.get('status') != 'completed':
        raise ValueError('Generated implementation fit failed: '+str(result.get('error')))
    from mlmom.inputs import features, read_table
    heldout = features(read_table(root/'train.csv').iloc[-8:], context)
    heldout_path = root/'inference.csv'
    heldout.to_csv(heldout_path, index=False)
    prediction = sandbox.run(dict(action='predict', code_dir=str(code), code_hash=tree_hash(code),
        model_artifact=result['model_artifact'], features_path=str(heldout_path),
        kinds=[plan.prediction_kind]), root/'reload', seconds=30, cpus=1, memory_gb=2)
    atomic_json(root/'reload-result.json', prediction)
    if prediction.get('status') != 'completed':
        raise ValueError('Generated implementation reload failed: '+str(prediction.get('error')))
    return dict(fit='passed', reload='passed', score=result['internal_score'])


def feature(client, registry, root, config, budget, sandbox, store):
    frame, context, plan, code = setup_case(root)
    pipe = SimpleNamespace(context=context, config=config, budget=budget, sandbox=sandbox, store=store)
    tools = feature_data_tools(pipe, root/'allowed-train.csv', 'live-v8')
    profile = feature_profile(dict(context=context.model_dump(mode='json'),
                                   train_path=str(root/'allowed-train.csv')))['profile']
    payload = dict(task=context.task.model_dump(mode='json'), train_profile=profile,
        current_feature_set=FeatureSet(feature_set_id='empty').model_dump(mode='json'), history=[],
        feature_runtime=feature_runtime_info(), remaining_configurations=1,
        phase='proposal_before_baseline', train_evidence_refs=tools.evidence_refs,
        skill_instructions=registry.load([],context.task.task_type,role='feature_agent'),
        diagnostic='Use a train query to investigate a concrete feature hypothesis, then propose one '
            'small executable Python feature program. This diagnostic checks tool delivery and '
            'fold-fitted state, not whether it improves a benchmark. Do not stop before proposing.')
    proposal = client.generate('feature_agent', FEATURE_PROMPT, payload, FeatureProposal, tools)
    atomic_json(root/'proposal.json', proposal)
    if not proposal.programs:
        raise ValueError('Diagnostic requested a Python feature program')
    if not store.get('feature_query_refs','live-v8'):
        raise ValueError('No train evidence query was exercised')
    plan.feature_set = FeatureSet(feature_set_id='live-v8',features=proposal.features,programs=proposal.programs)
    plan.feature_set_id = 'live-v8'
    (code/'candidate.py').write_text(reference('skills/coding_agent/sklearn/REFERENCE.md'),encoding='utf-8')
    (code/'preprocessing.py').write_text(reference('skills/preprocessing_agent/preprocessing/REFERENCE.md'),encoding='utf-8')
    return check_generated(sandbox,context,plan,code,root)


def preprocessing(client, registry, root, config, budget, sandbox, store):
    _, context, plan, code = setup_case(root)
    plan.skill_ids = ['sklearn','missing_values']
    class Proxy:
        def __getattr__(self,key): return getattr(client,key)
        def __setattr__(self,key,value): setattr(client,key,value)
        def generate(self,role,instruction,payload,schema,tools=None):
            if role != 'preprocessing_agent':
                return schema(files={'candidate.py':reference('skills/coding_agent/sklearn/REFERENCE.md')})
            payload = dict(payload, diagnostic='For this tool-delivery diagnostic, read an additional '
                'relevant preprocessing card with read_skills, inspect the installed API you will use, '
                'then implement the assigned transform. You may use check_preprocessing for feedback.')
            return client.generate(role,instruction,payload,schema,tools)
    # Existing skeleton belongs to the fixture, not to the new coder assignment.
    (code/'candidate.py').unlink()
    plan_ref=root/'plan.json'; atomic_json(plan_ref,plan)
    graph = build_coding_graph(Proxy(),workspace=code,store=store,plan=plan,plan_ref=plan_ref,
        skill_data=registry.load(['sklearn'],context.task.task_type,role='coding_agent'),
        tools=installed_tools(),context=context,sandbox=sandbox,budget=budget,config=config,round_id='live',
        preprocessing_skills=registry.load(['missing_values'],context.task.task_type,role='preprocessing_agent'),
        skill_registry=registry)
    result=graph.invoke({'payload':{}})
    atomic_json(root/'coding-result.json',result)
    if not result['valid']:raise ValueError('Preprocessing graph: '+str(result['errors']))
    usage = store.get('preprocessing_skill_usage','live-'+plan.hypothesis.hypothesis_id)
    if not usage:raise ValueError('Dynamic card read was not exercised')
    versions=result['candidate']['skill_versions']
    if not all(card['skill_id'] in versions for card in usage['cards']):
        raise ValueError('Dynamic cards absent from candidate versions')
    return dict(check_generated(sandbox,context,plan,Path(result['candidate']['code_dir']),root),
                dynamically_read_cards=[c['skill_id'] for c in usage['cards']])


def training(client, registry, root, config, budget, sandbox, store):
    _, context, plan, _ = setup_case(root)
    plan.model_params = dict(n_estimators=300,max_depth=6,min_samples_leaf=2)
    sampled={'max_depth':8,'policy_params.weight_policy.name':'tail_aware'}
    effective=apply_overrides(plan.model_dump(mode='json'),sampled)
    effective['model_params']['n_estimators']=120
    checkpoint=TrainingCheckpoint(checkpoint_id='live-selected',stage='batch',
        plan=plan.model_dump(mode='json'),sampled_params=sampled,best_params=effective['model_params'],
        effective_configuration=effective,best_score=1.2,remaining_seconds=None,remaining_trials=0,
        trial_limit=1,default_action='refit',trials=[dict(number=0,score=1.2,params=sampled)],
        timeouts_enabled=False)
    answer=decide(client,checkpoint,registry.load([],context.task.task_type,role='training_agent'),seconds=180)
    atomic_json(root/'checkpoint.json',checkpoint);atomic_json(root/'decision.json',answer)
    if answer.action=='continue' or answer.reason.startswith('Programmatic policy:'):
        raise ValueError('No trials remain; expected a genuine selected-configuration decision')
    return dict(action=answer.action,reason=answer.reason)


def validator(client, registry, root, config, budget, sandbox, store):
    _, context, plan, _ = setup_case(root,'binary_classification')
    context.task.primary_metric='mcc';context.task.metric_direction='maximize'
    context.task.validation_strategy='holdout';plan.prediction_kind='label'
    plan.stopping_policy='fixed';plan.search_space={}
    plan.hypothesis.baseline=True
    class Proxy:
        def __getattr__(self,key):return getattr(client,key)
        def generate(self,role,instruction,payload,schema,tools=None):
            if role=='hypothesis_generator':return HypothesisBatch(hypotheses=[plan.hypothesis])
            if role=='planner':return PlanBatch(plans=[plan])
            return client.generate(role,instruction,payload,schema,tools)
    result=build_planning_graph(Proxy(),context,installed_tools(),registry,budget.snapshot(),
        max_attempts=1,max_candidates=1,validator_seconds=180).invoke({'payload':{'round_number':1}})
    atomic_json(root/'validation-result.json',result)
    if not result['valid']:raise ValueError('Known feasible MCC plan rejected: '+str(result['errors']))
    return dict(valid=True,warnings=result.get('warnings',[]))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ledger',type=Path,required=True)
    parser.add_argument('--roles',default='feature,preprocessing,training,validator')
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    settings=replace(LLMSettings.from_env(),timeout_seconds=180)
    if settings.model!=MODEL:raise ValueError('Configured model does not match the approved diagnostic model')
    limit=AttemptLimit(args.ledger.resolve())
    trace=TraceSink(root,'skill-contract-v8-live',secrets=[settings.api_key])
    config=RunConfig(timeouts_enabled=True,total_seconds=2400,final_reserve_seconds=0,
        reserve_cpus=0,reserve_memory_gb=0,coding_seconds=420,coding_tool_steps=6,max_attempts=1)
    budget=Budget(config,time.time());store=ExperimentStore(root/'store')
    client=RoleClient(settings,deadline=budget.deadline,max_steps=6,trace=trace,
                      request_callback=limit,thinking_mode='off')
    results=[]
    try:
        with trace.activate(),SandboxManager(budget=budget,max_workers=1) as sandbox:
            for name in args.roles.split(','):
                started=len(limit.data['attempts'])
                try:
                    detail=globals()[name](client,SkillRegistry(ROOT/'skills'),root/name,
                                           config,budget,sandbox,store)
                    result=dict(role=name,status='passed',detail=detail)
                except Exception as error:
                    result=dict(role=name,status='failed',error=trace.scrub(str(error)))
                result['http_attempts']=len(limit.data['attempts'])-started
                results.append(result);atomic_json(root/'results.json',results)
                print(json.dumps(result,ensure_ascii=True),flush=True)
    finally:
        client.client.close();store.close();trace.close()
    events=[json.loads(line) for line in (root/'trace.jsonl').read_text().splitlines()]
    usage=[]
    for event in events:
        if event['kind']=='llm.response':
            usage.append(json.loads((root/event['payload']['path']).read_text()).get('usage') or {})
    report=dict(status='passed' if all(r['status']=='passed' for r in results) else 'failed',
        model=settings.model,results=results,total_attempts=len(limit.data['attempts']),limit=20,
        tokens={key:sum(u.get(key) or 0 for u in usage) for key in ('prompt_tokens','completion_tokens','total_tokens')},
        tools=dict(Counter(e.get('tool_name') for e in events if e['kind']=='tool.started')))
    atomic_json(root/'summary.json',report)
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
