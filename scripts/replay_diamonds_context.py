"""Replay the completed diamonds r1 boundary, optionally calling live roles; never fit a model."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from mlmom.evaluation import compact_diagnostics
from mlmom.execution_facts import FitExecutionFacts, RUNTIME_CONTRACT_VERSION
from mlmom.persistence import ExperimentStore, atomic_json
from mlmom.pipeline_contracts import Champion, HypothesisPlan, RunConfig, TaskContext


def capture_boundary(source_case):
    source_case=Path(source_case)
    with sqlite3.connect((source_case/'run/experiments.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as db:
        records={(kind,key):json.loads(body) for kind,key,body in db.execute('SELECT kind,id,body FROM objects')}
    if not records.get(('cycle','r1'),{}).get('completed'):
        raise ValueError('Replay requires a completed first cycle')
    result=records['round_result','r1']
    definitions={}
    def compact_intervention(value,identity):
        value=copy.deepcopy(value)
        if value and len(json.dumps(value.get('operations',[])))>3000:
            value['operations']=['Apply SDK feature definition '+str(identity)+' (source/config retained in feature_sets)']
        return value
    def compact(row):
        row=copy.deepcopy(row)
        definition=row['plan'].pop('feature_set',None)
        if definition:definitions[definition['feature_set_id']]=definition
        row['plan']['intervention']=compact_intervention(row['plan'].get('intervention'),row['plan'].get('feature_set_id'))
        if row['run'].get('intervention'):
            row['run']['intervention']=compact_intervention(row['run']['intervention'],row['plan'].get('feature_set_id'))
        row['run']['trials']=[{key:trial[key] for key in ('number','params','status','score','duration_seconds',
            'origin','kind','score_kind','outer_split_id','validation_protocol_ref') if key in trial}
            for trial in row['run'].get('trials',[])]
        if row.get('evaluation'):
            row['evaluation']['diagnostics']=compact_diagnostics(row['evaluation'].get('diagnostics',{}))
        return row
    champion=compact(result['champion'])
    rows=[compact(row) for row in result['round_results']]
    components=[compact(row) for (kind,_),row in records.items() if kind=='component' and row['round_id']=='r1']
    context=records['context','task']
    profile=json.loads((source_case/'run/inspection/profile.json').read_text(encoding='utf-8'))
    # Full column summaries/distributions are not needed to reproduce this handoff.
    for column in profile.get('columns',[]):
        for partition in ('train','test'):
            if column.get(partition):column[partition]['distribution']=[]
    task=(source_case/'task'/Path(context['files']['task']).name).read_text(encoding='utf-8-sig')
    source_dir=source_case/'run'/champion['candidate']['code_dir'].split('/run/',1)[1]
    sources={p.relative_to(source_dir).as_posix():p.read_text(encoding='utf-8') for p in source_dir.rglob('*.py')}
    observation_path=next((source_case.parent/'monitoring').glob('diamonds-r1-model-manifest-*.json'))
    observation=json.loads(observation_path.read_text(encoding='utf-8'))
    if observation['candidate_id']!=champion['candidate']['candidate_id']:
        raise ValueError('Saved manifest does not describe the r1 champion')
    manifest=observation['manifest']
    programs=definitions[champion['plan']['feature_set_id']]['programs']
    facts=FitExecutionFacts(fit_id='imported-r1-submission-refit',recorded_at=0,
        stage='submission_refit',split_id='all-train',status='observed',estimator='Ridge',
        parameters=champion['run']['best_params'],parameter_source='saved_submission_configuration',
        base_columns=manifest['base_width'],derived_columns=manifest['output_width']-manifest['base_width'],
        total_columns=manifest['output_width'],feature_set_hash=manifest['feature_set_hash'],
        feature_names=manifest['names'],feature_execution_owner='SDK',
        feature_programs=[dict(name=p['name'],config=p['config'],source_hash=hashlib.sha256(p['source'].encode()).hexdigest()) for p in programs],
        reason='Imported saved-model manifest observation: monitoring/'+observation_path.name)
    analysis=copy.deepcopy(records['analysis','r1'])
    analysis['selected']=[]
    analysis['search_groups']=[]
    for decision in analysis.get('decisions',[]):
        decision['diagnostics']=compact_diagnostics(decision.get('diagnostics',{}))
    refit=copy.deepcopy(records['submission_refit','r1-submission-refit'])
    refit['intervention']=compact_intervention(refit.get('intervention'),champion['plan'].get('feature_set_id'))
    return dict(version=1,origin=source_case.parent.name+'/'+source_case.name,
        boundary='after completed r1; before planning r2',context=context,profile=profile,task_description=task,
        champion=champion,rows=rows,components=components,feature_sets=definitions,sources=sources,
        reports={key:records['reconnaissance',key] for key in ('initial','r1')},analysis=analysis,
        submission=result['submission'],submission_refit=refit,
        submission_source=records['submission_refit_source','r1-submission-refit'],submission_observation=facts.model_dump(mode='json'))


def hydrate(fixture):
    fixture=copy.deepcopy(fixture)
    for row in [fixture['champion'],*fixture['rows'],*fixture['components']]:
        identity=row['plan'].get('feature_set_id')
        if identity:row['plan']['feature_set']=fixture['feature_sets'][identity]
    return fixture


def seed_pipeline(fixture, output, client):
    """Only the planning/review surface exists: no sandbox or training entrypoint."""
    from mlmom.pipeline import Pipeline
    from mlmom.registry import SkillRegistry, installed_tools
    from mlmom.resources import Budget
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    data=hydrate(fixture)
    atomic_json(output/'profile.json',data['profile'])
    (output/'task.md').write_text(data['task_description'],encoding='utf-8')
    raw=copy.deepcopy(data['context'])
    raw['profile_ref']=str((output/'profile.json').resolve())
    raw['files']['task']=str((output/'task.md').resolve())
    pipeline=object.__new__(Pipeline)
    pipeline.run_dir=output.resolve()
    pipeline.context=TaskContext.model_validate(raw)
    pipeline.files=pipeline.context.files
    pipeline.config=RunConfig(max_rounds=2,reserve_cpus=0,reserve_memory_gb=0)
    pipeline.budget=Budget(pipeline.config,time.time())
    pipeline.client=client
    pipeline.skills=SkillRegistry(ROOT/'skills')
    pipeline.tools=installed_tools()
    pipeline.store=ExperimentStore(output)
    pipeline.last_submission=data['submission']
    pipeline.scout=lambda key,history:data['reports'][key]
    for row in [data['champion'],*data['rows'],*data['components']]:
        pipeline.store.put('candidate',row['candidate']['candidate_id'],row['candidate'])
        pipeline.store.put('experiment',row['run']['experiment_id'],row['run'])
        if row.get('evaluation'):pipeline.store.put('evaluation',row['run']['experiment_id'],row['evaluation'])
    for row in data['components']:pipeline.store.put('component',row['experiment_id'],row)
    for key,value in data['reports'].items():pipeline.store.put('reconnaissance',key,value)
    for key,value in data['feature_sets'].items():
        pipeline.store.put('feature_set',key,value)
        pipeline.store.put('feature_registry',key,dict(definition=value,evidence=dict(round_id='r1',status='historical')))
    pipeline.store.put('analysis','r1',data['analysis'])
    pipeline.store.put('champion','current',data['champion'])
    pipeline.store.put('cycle','r1',dict(round_id='r1',completed=True,evaluation_completed=True))
    refit=dict(data['submission_refit'],execution_facts=[data['submission_observation']])
    pipeline.store.put('submission_refit',refit['experiment_id'],refit)
    pipeline.store.put('submission_refit_source',refit['experiment_id'],data['submission_source'])
    state=dict(context=raw,round_number=2,champion=data['champion'],selected=data['analysis'])
    return pipeline,state


class RecordedClient:
    def __init__(self,client,output):
        self.client=client
        self.output=Path(output)
        self.calls=[]
        self.reads=[]

    def __getattr__(self,name):return getattr(self.client,name)

    def generate(self,role,instruction,payload,schema,tools=None):
        from mlmom.tools import ToolSet
        number=len(self.calls)
        record=dict(role=role,instruction=instruction,input=payload,status='running')
        self.calls.append(record)
        path=self.output/'role-calls'/f'{number:02d}-{role}.json'
        atomic_json(path,record)
        if tools and 'read_evidence' in tools.tools:
            scoped=ToolSet();scoped.tools=dict(tools.tools)
            reader,_,description=tools.tools['read_evidence']
            def read_evidence(reference: str,offset: int=0,chars: int=8000):
                answer=reader(reference,offset,chars)
                self.reads.append(dict(role=role,reference=reference,offset=offset,chars=chars,total_chars=answer['total_chars']))
                atomic_json(self.output/'evidence-reads.json',self.reads)
                return answer
            scoped.register('read_evidence',read_evidence,description)
            tools=scoped
        try:
            answer=self.client.generate(role,instruction,payload,schema,tools)
            record.update(status='completed',output=answer.model_dump(mode='json') if hasattr(answer,'model_dump') else answer)
            return answer
        except Exception as error:
            record.update(status='failed',error=str(error))
            raise
        finally:atomic_json(path,record)


def replay(fixture,output,live=False):
    from mlmom.planning import build_planning_graph
    from mlmom.pipeline_contracts import EnsembleProposal
    from mlmom.planning_context import champion_snapshot
    from mlmom.research import review_result
    from mlmom.review_context import review_payload
    from mlmom.search_groups import component_brief
    from mlmom.tools import ToolSet
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',live=live,training_started=False,runtime_contract_version=RUNTIME_CONTRACT_VERSION,
        boundary=fixture['boundary'],origin=fixture['origin'],roles=[],evidence_checks=[])
    pipeline=None
    base_client=None
    try:
        if live:
            from dataclasses import replace
            from mlmom.agents import RoleClient
            from mlmom.llm_client import LLMSettings
            settings=replace(LLMSettings.from_env(str(ROOT/'.env')),timeout_seconds=210)
            base_client=RoleClient(settings,deadline=time.time()+1800,max_steps=8)
            report['model']=settings.model
        else:
            from tests.fakes import OfflineLLM
            base_client=OfflineLLM()
        client=RecordedClient(base_client,output)
        pipeline,state=seed_pipeline(fixture,output/'replay',client)
        data=hydrate(fixture)
        row=data['champion']
        payload=review_payload(row,data['rows'],sources=data['sources'],store=pipeline.store)
        review=review_result(client,payload,pipeline.skills.load([],'regression',role='result_validator'),'diamonds-r1',210)
        atomic_json(output/'review.json',review)
        pipeline.store.put('result_review',row['run']['experiment_id'],review)
        planning=pipeline.planning_payload(state)
        atomic_json(output/'planning-input.json',planning)
        for reference in planning['evidence_refs']:
            answer=pipeline.read_planning_evidence(reference,chars=73)
            pipeline.read_planning_evidence(reference,offset=73,chars=73)
            report['evidence_checks'].append(dict(reference=reference,total_chars=answer['total_chars']))
        tools=ToolSet()
        tools.register('read_evidence',pipeline.read_planning_evidence,'Read supplied evidence in bounded pages.')
        tools.register('read_task',pipeline.read_task,'Read original task.')
        tools.register('read_history',pipeline.read_history,'Read saved first-cycle evidence.')
        graph=build_planning_graph(client,pipeline.context,pipeline.tools,pipeline.skills,pipeline.budget.snapshot(),
            max_attempts=2,max_candidates=2,role_tools=tools,lightweight=False,
            known_candidates={r['candidate']['candidate_id'] for r in data['components']})
        planned=graph.invoke(dict(payload=planning),{'recursion_limit':100})
        atomic_json(output/'planning-output.json',planned)
        components=[component_brief(row) for row in data['components']]
        if live:
            proposal=client.generate('ensemble_agent',
                'Select complementary frozen single experiments or skip. Read the current SDK execution facts. '
                'Historical cross-split scores are not comparable. Do not train or change component configurations.',
                dict(task=pipeline.context.task.model_dump(mode='json'),components=components,
                    champion=champion_snapshot(Champion.model_validate(state['champion']),pipeline.store),
                    skill_instructions=pipeline.skills.load([],'regression',role='ensemble_agent')),
                EnsembleProposal,tools)
            atomic_json(output/'ensemble-proposal.json',proposal)
        report.update(status='passed' if planned.get('valid') else 'incomplete',roles=client.calls,
            planning_valid=planned.get('valid'),live_evidence_reads=client.reads)
    except Exception as error:
        report.update(status='incomplete',error=str(error),roles=client.calls if 'client' in locals() else [])
    finally:
        if 'client' in locals():atomic_json(output/'evidence-reads.json',client.reads)
        if pipeline is not None:pipeline.store.close()
        if live and base_client is not None:base_client.client.close()
        atomic_json(output/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-case',type=Path)
    parser.add_argument('--fixture',type=Path,default=ROOT/'tests/fixtures/diamonds_r1_boundary.json')
    parser.add_argument('--capture',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--live',action='store_true')
    args=parser.parse_args()
    if args.capture:
        if not args.source_case:parser.error('--capture requires --source-case')
        if args.fixture.exists():parser.error('Capture never overwrites a fixture')
        atomic_json(args.fixture,capture_boundary(args.source_case))
    if args.output:
        report=replay(json.loads(args.fixture.read_text(encoding='utf-8')),args.output,args.live)
        print(json.dumps({key:report.get(key) for key in ('status','training_started','model','error')},ensure_ascii=False))


if __name__=='__main__':main()
