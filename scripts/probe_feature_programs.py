"""Two isolated live Feature Agent calls on recorded cycle contexts, then real CV.

Run in Docker with read-only historical sources and a fresh /work directory.
No Planner/coder/HPO or complete pipeline run is started.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mlmom.agents import RoleClient
from mlmom.feature_contracts import FeatureProposal,FeatureSet
from mlmom.feature_search import FEATURE_PROMPT,feature_runtime_info,feature_task_view,proposed_feature_set
from mlmom.inputs import read_table,features
from mlmom.llm_client import LLMSettings
from mlmom.model_spec import frozen_plan
from mlmom.persistence import atomic_json,file_hash,tree_hash
from mlmom.pipeline_contracts import HypothesisPlan,ExperimentRun,TaskContext,RunConfig
from mlmom.registry import SkillRegistry
from mlmom.resources import Budget,SandboxManager
from mlmom.splitting import inner_protocol
from mlmom.tracing import TraceSink,emit


def recorded_context(source):
    run=source/'run'
    events=[json.loads(line) for line in (run/'trace.jsonl').read_text().splitlines()]
    request=next(e for e in reversed(events) if e.get('role')=='feature_agent' and e['kind']=='llm.attempt.started')
    raw=json.loads((run/request['payload']['path']).read_text())
    payload=next(json.loads(m['content']) for m in reversed(raw['messages']) if m['role']=='user')
    with sqlite3.connect('file:'+str(run/'experiments.sqlite3')+'?mode=ro',uri=True) as db:
        records={(k,i):json.loads(body) for k,i,body in db.execute('SELECT kind,id,body FROM objects')}
    round_id=request.get('round_id') or 'r1'
    session=records['feature_session','initial' if round_id=='r1' else round_id]
    candidate=records['candidate',session['baseline_candidate_id']]
    measured=[v for (kind,key),v in records.items() if kind=='experiment' and v['candidate_id']==candidate['candidate_id'] and v['status']=='completed' and v.get('role','evaluation')=='evaluation' and v['round_id']==round_id]
    measured.sort(key=lambda r:bool(r.get('study_ref')))
    result=measured[-1]
    context=TaskContext.model_validate(records['context','task'])
    old_case=str(Path(context.files.task).parent.parent)
    relative=lambda path:source/Path(path).relative_to(old_case)
    plan=HypothesisPlan.model_validate_json(relative(candidate['plan_ref']).read_text())
    plan=frozen_plan(plan,ExperimentRun.model_validate(result))
    return payload,context,plan,candidate,relative,records['split',round_id],request['seq']


def probe(name,source,out,rows):
    out.mkdir(parents=True)
    payload,context,plan,candidate,relative,split,seq=recorded_context(source)
    parent=FeatureSet.model_validate(payload['current_feature_set'])
    plan.feature_set_id,plan.feature_set=parent.feature_set_id,parent
    plan.strategy,plan.search_space,plan.max_trials='single',{},1
    plan.stopping_policy,plan.component_stopping_policies='fixed',{}
    plan.cpus,plan.memory_gb,plan.time_limit_seconds=2,2,180
    code=out/'control-code'
    shutil.copytree(relative(candidate['code_dir']),code,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    assert tree_hash(code)==candidate['code_hash']
    frame=read_table(relative(context.files.train),table_format=context.table_formats.get(context.files.train)).iloc[split['train_indices']]
    if len(frame)>rows:
        if 'classification' in str(context.task.task_type) and len(context.task.target_columns)==1:
            column=context.task.target_columns[0]
            pieces=[part.sample(n=min(len(part),max(8,round(rows*len(part)/len(frame)))),random_state=42)
                    for _,part in frame.groupby(column,sort=False)]
            import pandas as pd
            frame=pd.concat(pieces).sample(frac=1,random_state=42)
        else:frame=frame.sample(n=rows,random_state=42)
    frame=frame.reset_index(drop=True)
    train_path=out/'train.csv';frame.to_csv(train_path,index=False)
    context=context.model_copy(deep=True)
    context.task.validation_folds=3
    context.files=context.files.model_copy(update=dict(root=str(out),train=str(train_path),test='',task='',dictionaries=[],sample_submission=None,manifest=None))
    context.profile_ref='';context.table_formats={}
    protocol=inner_protocol(frame,context,fraction=.2,seed=43,outer_split_id='isolated-'+name,source_indices=list(range(len(frame))))
    payload={k:v for k,v in payload.items() if k not in {'output_schema','skill_instructions'}}
    payload.update(feature_runtime=feature_runtime_info(),frozen_model=plan.model_dump(mode='json'),
        skill_instructions=SkillRegistry(ROOT/'skills').load([],context.task.task_type,role='feature_agent',include_references=True),
        isolated_probe=dict(source_request_seq=seq,request='Propose one concrete executable Python feature experiment beyond previously measured proposals. Include at least one program; do not stop in this isolated probe. No particular algorithm or entity key is prescribed.',measurement_rows=len(frame),measurement_folds=len(protocol['folds']),scope='Exploratory subset of outer train only; scores are not a new external benchmark.'))
    atomic_json(out/'agent-input.json',payload)
    atomic_json(out/'control-plan.json',plan)
    settings=replace(LLMSettings.from_platform_env(),timeout_seconds=300)
    trace=TraceSink(out,'feature-probe-'+name,secrets=[settings.api_key])
    budget=Budget(RunConfig(timeouts_enabled=True,total_seconds=1800,final_reserve_seconds=0,reserve_cpus=0,reserve_memory_gb=0),time.time())
    client=RoleClient(settings,deadline=budget.deadline,max_steps=4,timeouts_enabled=True,trace=trace)
    report=dict(case=name,status='running',source_request_seq=seq,rows=len(frame),folds=len(protocol['folds']),
        model=settings.model,metric=str(context.task.primary_metric),direction=str(context.task.metric_direction),
        split_id='isolated-'+name,score_kind='feature_inner_cv_mean',role='evaluation',
        reference_feature_set_id=parent.feature_set_id,attempts=[])
    request=dict(action='train',code_dir=str(code),code_hash=candidate['code_hash'],plan=plan.model_dump(mode='json'),
        context=context.model_dump(mode='json'),train_path=str(train_path),seed=42,inner_fraction=.2,
        outer_split_id='isolated-'+name,outer_train_indices=list(range(len(frame))),training_control=False,
        validation_scope='feature',feature_protocol=protocol,feature_holdout_folds=3)
    try:
        with trace.activate(),SandboxManager(budget=budget,max_workers=1) as sandbox:
            baseline=sandbox.run(request,out/'reference',seconds=180,cpus=2,memory_gb=2)
            if baseline.get('status')!='completed':raise ValueError('Control failed: '+str(baseline.get('error') or baseline.get('stop_reason')))
            report['baseline_score']=baseline['internal_score']
            feedback=None
            for attempt in range(2):
                active_payload=dict(payload,runtime_feedback=feedback)
                proposal=client.generate('feature_agent',FEATURE_PROMPT,active_payload,FeatureProposal)
                atomic_json(out/f'proposal-{attempt}.json',proposal)
                definition,fingerprint=proposed_feature_set(parent,proposal,context.feature_columns)
                atomic_json(out/f'feature-set-{attempt}.json',definition)
                for program in proposal.programs:
                    (out/f'{attempt}-{program.name}.py').write_text(program.source,encoding='utf-8')
                modified=copy.deepcopy(request)
                modified['plan'].update(feature_set_id=definition.feature_set_id,feature_set=definition.model_dump(mode='json'))
                measured=sandbox.run(modified,out/f'candidate-{attempt}',seconds=180,cpus=2,memory_gb=2)
                item=dict(number=attempt,status=measured.get('status'),programs=[p.name for p in proposal.programs],
                    rationale=proposal.rationale,score=measured.get('internal_score'),error=measured.get('error'),stop_reason=measured.get('stop_reason'),fingerprint=fingerprint)
                report['attempts'].append(item)
                atomic_json(out/'report.json',report)
                if measured.get('status')=='completed':
                    # Independent loading/inference in another ML worker exercises serialized program state.
                    path=out/'predict.csv';features(frame.iloc[:128],context).to_csv(path,index=False)
                    predicted=sandbox.run(dict(action='predict',code_dir=str(code),code_hash=candidate['code_hash'],
                        model_artifact=measured['model_artifact'],features_path=str(path),kinds=[context.submission_kind]),out/'saved-model-replay',seconds=60,cpus=2,memory_gb=2)
                    if predicted.get('status')!='completed':raise ValueError('Saved model replay failed: '+str(predicted.get('error')))
                    sign=1 if str(context.task.metric_direction)=='maximize' else -1
                    report.update(status='passed',candidate_score=measured['internal_score'],
                        signed_gain=sign*(measured['internal_score']-baseline['internal_score']),
                        program_count=len(proposal.programs),saved_model_replay='passed',
                        output_rows=128,source_hash=tree_hash(code),control_source_unchanged=tree_hash(code)==candidate['code_hash'])
                    break
                feedback=dict(error=measured.get('error'),stop_reason=measured.get('stop_reason'),previous_proposal=proposal.model_dump(mode='json'),
                    instruction='Correct this concrete execution failure. Keep the control and input context unchanged.')
            else:report['status']='failed'
    except Exception as error:
        report.update(status='failed',error=str(error))
    finally:
        client.client.close()
        trace.close()
    events=[json.loads(line) for line in (out/'trace.jsonl').read_text().splitlines()]
    usages=[]
    for event in events:
        if event['kind']=='llm.response':
            raw=json.loads((out/event['payload']['path']).read_text());usages.append(raw.get('usage') or {})
    report.update(api_calls=sum(e['kind']=='llm.call.completed' for e in events),
        api_attempts=sum(e['kind']=='llm.attempt.started' for e in events),
        tokens={key:sum(u.get(key) or 0 for u in usages) for key in ['prompt_tokens','completion_tokens','total_tokens']},
        provider_cost=sum(u.get('cost') or 0 for u in usages),cost_currency=None)
    atomic_json(out/'report.json',report)
    print(json.dumps(dict(case=name,status=report['status'],baseline_score=report.get('baseline_score'),candidate_score=report.get('candidate_score'),programs=[a['programs'] for a in report['attempts']],api_calls=report['api_calls'],error=report.get('error'))),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--rows',type=int,default=6000)
    parser.add_argument('--output',type=Path,default=Path('/work/probes'))
    args=parser.parse_args()
    results=[probe(name,Path(source),args.output/name,args.rows) for name,source in
        [('regression','/source-reg'),('multiclass','/source-multi')]]
    atomic_json(args.output/'summary.json',dict(status='passed' if all(r['status']=='passed' for r in results) else 'failed',cases=results))
    return 0 if all(r['status']=='passed' for r in results) else 1


if __name__=='__main__':raise SystemExit(main())
