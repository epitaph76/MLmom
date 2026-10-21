"""Observe real local runs, retain every attempt and verify saved-model replay.

No mocked LLM, patched model selection, or runtime fallback is used here.
Repeated invocations of a case share its persisted wall-time allowance.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mlmom.persistence import atomic_json, file_hash
from mlmom.pipeline_contracts import RunConfig, TaskContext
from mlmom.resources import capacity, terminate_tree
from mlmom.llm_client import LLMSettings


def validate(run_dir, expected_rows, seconds):
    """Replay the final saved submission model in a supervised predict worker."""
    import numpy as np
    import pandas as pd
    from mlmom.inputs import read_table, features
    from mlmom.resources import Budget, SandboxManager
    summary=json.loads((run_dir/'summary.json').read_text(encoding='utf-8'))
    if summary['status']!='completed':
        return dict(status='failed',reason='Pipeline did not produce a submission')
    with sqlite3.connect(run_dir/'experiments.sqlite3') as db:
        context=TaskContext.model_validate(json.loads(db.execute(
            "SELECT body FROM objects WHERE kind='context' AND id='task'").fetchone()[0]))
    test=read_table(context.files.test,table_format=context.table_formats.get(context.files.test))
    output=read_table(summary['submission']['path'])
    fmt=context.task.result_format
    if len(output)!=expected_rows or len(test)!=expected_rows or output.columns.tolist()!=fmt.columns:
        raise ValueError('Submission shape/columns mismatch')
    if output.isna().any().any(): raise ValueError('Missing submission values')
    for column in context.task.id_columns:
        if output[column].tolist()!=test[column].tolist() or output[column].duplicated().any():
            raise ValueError('Submission IDs/order mismatch')
    values=output[fmt.prediction_columns].to_numpy()
    if 'classification' in context.task.task_type:
        if not set(values.ravel())<=set(context.classes): raise ValueError('Unknown predicted class')
    elif not np.isfinite(values.astype(float)).all(): raise ValueError('Nonfinite regression predictions')
    matching=[p for p in (run_dir/'submissions').glob('*.json')
              if json.loads(p.read_text(encoding='utf-8'))['artifact']['sha256']==summary['submission']['sha256']]
    metadata=json.loads(max(matching,key=lambda p:p.stat().st_mtime_ns).read_text(encoding='utf-8'))
    replay=run_dir/'verification';replay.mkdir(exist_ok=True)
    x=features(test,context).head(128);path=replay/'features.csv';x.to_csv(path,index=False)
    config=RunConfig(total_seconds=max(1,seconds),final_reserve_seconds=0,reserve_cpus=6,
                     reserve_memory_gb=max(0,capacity(RunConfig(reserve_memory_gb=0))[1]-3),max_workers=1)
    budget=Budget(config,time.time());manager=SandboxManager(budget=budget,max_workers=1)
    try:
        result=manager.run(dict(action='predict',code_dir=metadata['code_dir'],code_hash=metadata['code_hash'],
            model_artifact=metadata['model_artifact'],features_path=str(path),kinds=[context.submission_kind]),
            replay/'worker',seconds=max(.1,seconds),memory_gb=3,cpus=2)
    finally:
        manager.close()
    if result['status']!='completed': raise ValueError('Saved model replay failed: '+str(result))
    predictions=pd.read_csv(result['predictions'][context.submission_kind]).to_numpy()
    reference=values[:len(x)]
    equal=(np.array_equal(predictions,reference) if 'classification' in context.task.task_type else
           np.allclose(predictions.astype(float),reference.astype(float),rtol=1e-9,atol=1e-9))
    if not equal: raise ValueError('Reloaded model predictions differ from submission')
    report=dict(status='passed',rows=len(output),replay_rows=len(x),replay_equal=True,
        primary_metric=context.task.primary_metric,holdout_score=summary['champion']['evaluation']['score'],
        submission=str(Path(summary['submission']['path'])),saved_submission=metadata['saved_path'],
        model_artifact=metadata['model_artifact'],candidate_id=summary['submission']['candidate_id'],
        model_role=summary['submission']['model_role'])
    atomic_json(replay/'report.json',report)
    return report


def index(run_dir):
    atomic_json(run_dir/'artifact_index.json',dict(run_dir=str(run_dir),files=[
        dict(path=p.relative_to(run_dir).as_posix(),bytes=p.stat().st_size,sha256=file_hash(p))
        for p in sorted(run_dir.rglob('*')) if p.is_file() and p.name!='artifact_index.json' and p.suffix!='.pyc']))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case',choices=['all','regression','multiclass'],default='all')
    parser.add_argument('--seconds',type=float,default=3600)
    parser.add_argument('--quiet',action='store_true',help='Skip periodic log reads and progress printing; keep resource/deadline guards.')
    parser.add_argument('--expected-model',default='qwen/qwen3.8-27b',
                        help='Explicit local test model; defaults to the organizer Qwen. Credentials remain in .env.')
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env',override=True)  # Explicit local diagnostics credentials.
    settings=LLMSettings.from_env(str(ROOT/'.env'))
    if settings.model!=args.expected_model: raise ValueError('Configured model differs from --expected-model; check the local environment')
    reports_path=root/'reports.json'
    reports=json.loads(reports_path.read_text(encoding='utf-8')) if reports_path.exists() else []
    specs=[('regression','Регрессия-20261020T130119Z-1-001.zip',22449),
           ('multiclass','Многоклассовая классификация-20261020T130139Z-1-001.zip',41166)]
    for name,filename,rows in specs:
        if args.case not in {'all',name}: continue
        case=root/name;case.mkdir(exist_ok=True)
        ledger_path=case/'budget.json'
        ledger=json.loads(ledger_path.read_text(encoding='utf-8')) if ledger_path.exists() else dict(spent_seconds=0,attempts=0)
        if ledger.get('active_started_at'):
            if ledger.get('pid') and psutil.pid_exists(ledger['pid']):
                raise RuntimeError(f'{name} already has an active observed run (pid {ledger["pid"]})')
            ledger['spent_seconds']+=max(0,time.time()-ledger.pop('active_started_at'))
            ledger.pop('pid',None)
            atomic_json(ledger_path,ledger)
        remaining=args.seconds-ledger['spent_seconds']
        if remaining<900:
            print(json.dumps(dict(case=name,status='not_started',reason='Insufficient complete hypothesis budget',remaining=remaining)),flush=True)
            continue
        archive=Path.home()/'Downloads'/filename
        inputs=root/'inputs'/name
        if not inputs.exists():
            inputs.mkdir(parents=True)
            with zipfile.ZipFile(archive) as source:
                for member in source.infolist():
                    if not (inputs/member.filename).resolve().is_relative_to(inputs): raise ValueError('Unsafe archive path')
                source.extractall(inputs)
        tasks=list(inputs.rglob('task.md'))
        if len(tasks)!=1: raise ValueError('Expected exactly one task.md')
        ledger['attempts']+=1
        run_dir=case/f"attempt-{ledger['attempts']}";run_dir.mkdir()
        from scripts.package_solution import package
        package(ROOT,run_dir/'runtime-solution.zip')
        cpus,memory=capacity(RunConfig(reserve_cpus=0,reserve_memory_gb=0))
        # Leave a minute of the task allowance for independent reload verification.
        config=RunConfig(profile='dev',total_seconds=remaining-60,final_reserve_seconds=600,
            first_submission_seconds=600,experiment_seconds=810,initial_experiment_seconds=120,
            coding_seconds=300,prediction_seconds=45,smoke_seconds=30,max_rounds=6,target_completed_rounds=3,
            fixed_outer_split=False,max_workers=1,reserve_cpus=cpus-4,reserve_memory_gb=memory-4,llm_timeout_seconds=300,
            hypothesis_seconds=210,planner_seconds=210,validator_seconds=210)
        config_path=run_dir/'config.json';atomic_json(config_path,config)
        atomic_json(run_dir/'input_manifest.json',dict(archive=str(archive),sha256=file_hash(archive),
            task=str(tasks[0]),model=settings.model,base_url=settings.base_url,
            runtime_archive_sha256=file_hash(run_dir/'runtime-solution.zip'),
            started_utc=datetime.now(timezone.utc).isoformat(),source_hashes={
                p.relative_to(ROOT).as_posix():file_hash(p) for p in (ROOT/'mlmom').glob('*.py')}))
        started=time.monotonic();peak=0;guard=None
        ledger['active_started_at']=time.time()
        atomic_json(ledger_path,ledger)
        with (run_dir/'controller.stdout.log').open('w',encoding='utf-8') as out, (run_dir/'controller.stderr.log').open('w',encoding='utf-8') as err:
            process=subprocess.Popen([sys.executable,'-u',str(ROOT/'solution.py'),str(tasks[0]),
                '--config',str(config_path),'--run-dir',str(run_dir)],cwd=ROOT,stdout=out,stderr=err)
            ledger['pid']=process.pid;atomic_json(ledger_path,ledger)
            print(json.dumps(dict(case=name,pid=process.pid,run_dir=str(run_dir),budget_seconds=remaining)),flush=True)
            tick=0
            while process.poll() is None:
                elapsed=time.monotonic()-started
                try:
                    parent=psutil.Process(process.pid);tree=[parent]+parent.children(recursive=True)
                    rss=sum(p.memory_info().rss for p in tree if p.is_running())/1024**3
                    peak=max(peak,rss)
                    with (run_dir/'resources.jsonl').open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps(dict(elapsed_seconds=elapsed,rss_gb=rss,pids=[p.pid for p in tree]))+'\n')
                except psutil.Error: pass
                if elapsed>=remaining:
                    guard='task_wall_time_limit';terminate_tree(process.pid);break
                if not args.quiet and tick%6==0:
                    tail=(run_dir/'controller.stdout.log').read_text(encoding='utf-8',errors='replace').splitlines()[-1:]
                    print(json.dumps(dict(case=name,elapsed_seconds=round(elapsed,1),peak_rss_gb=round(peak,3),latest=tail),ensure_ascii=False),flush=True)
                tick+=1;time.sleep(min(5,max(.1,remaining-elapsed)))
            process.wait()
        report=dict(case=name,attempt=ledger['attempts'],run_dir=str(run_dir),exit_code=process.returncode,guard=guard,peak_rss_gb=peak)
        summary=run_dir/'summary.json'
        if summary.exists(): report['result']=json.loads(summary.read_text(encoding='utf-8'))
        try:
            available=remaining-(time.monotonic()-started)
            report['verification']=validate(run_dir,rows,min(45,max(.1,available))) if available>1 else dict(status='not_run',reason='Task budget exhausted')
        except Exception as error:
            report['verification']=dict(status='failed',error=str(error))
        duration=time.monotonic()-started
        ledger['spent_seconds']+=duration
        ledger.pop('active_started_at',None);ledger.pop('pid',None)
        atomic_json(ledger_path,ledger)
        report['duration_seconds']=duration
        reports.append(report);atomic_json(reports_path,reports)
        atomic_json(run_dir/'observed_report.json',report);index(run_dir)
        print(json.dumps({k:v for k,v in report.items() if k!='result'},ensure_ascii=False),flush=True)
    selected=[r for r in reports if args.case=='all' or r['case']==args.case]
    required=2 if args.case=='all' else 1
    return 0 if len(selected)>=required and all(r.get('verification',{}).get('status')=='passed' and
        r.get('result',{}).get('process_acceptance',{}).get('passed',False) for r in selected[-required:]) else 2


if __name__=='__main__': raise SystemExit(main())
