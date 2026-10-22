"""Bounded exact-entrypoint acceptance against one immutable flat ZIP.

Offline runs inject an external test client, never a production fallback. Real
runs use the configured Qwen provider without injection. Neither edits the ZIP.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_run(run_dir, expected_rows, expected_metric):
    """Runs in a new interpreter bound to the package, with relocated source/model."""
    import numpy as np
    import pandas as pd
    from mlmom.pipeline_contracts import TaskContext
    from mlmom.inputs import read_table,features
    from mlmom.resources import SandboxManager,Budget
    from mlmom.pipeline_contracts import RunConfig
    from mlmom.persistence import tree_hash
    from mlmom.evaluation import metric_name
    import sqlite3
    summary=json.loads((run_dir/'summary.json').read_text(encoding='utf-8'))
    if summary['status']!='completed':raise ValueError('No completed submission')
    with sqlite3.connect(run_dir/'experiments.sqlite3') as db:
        context=TaskContext.model_validate(json.loads(db.execute("SELECT body FROM objects WHERE kind='context' AND id='task'").fetchone()[0]))
        cycles=[json.loads(r[0]) for r in db.execute("SELECT body FROM objects WHERE kind='cycle'")]
        experiments=[json.loads(r[0]) for r in db.execute("SELECT body FROM objects WHERE kind='experiment'")]
    if metric_name(context.task.primary_metric)!=metric_name(expected_metric):raise ValueError('Wrong primary metric')
    output=read_table(summary['submission']['path']);test=read_table(context.files.test)
    fmt=context.task.result_format
    if len(output)!=expected_rows or len(test)!=expected_rows or list(output)!=fmt.columns:raise ValueError('Submission shape/columns mismatch')
    if output.isna().any().any():raise ValueError('Missing submission values')
    for name in context.task.id_columns:
        if output[name].tolist()!=test[name].tolist() or output[name].duplicated().any():raise ValueError('Submission ID/order mismatch')
    values=output[fmt.prediction_columns].to_numpy()
    if 'classification' in context.task.task_type:
        if not set(values.ravel())<=set(context.classes):raise ValueError('Unknown submission label')
    elif not np.isfinite(values.astype(float)).all():raise ValueError('Nonfinite predictions')
    matches=[json.loads(p.read_text(encoding='utf-8')) for p in (run_dir/'submissions').glob('*.json')
             if json.loads(p.read_text(encoding='utf-8'))['artifact']['sha256']==summary['submission']['sha256']]
    if not matches:raise ValueError('Missing saved submission metadata')
    metadata=matches[-1]
    replay=run_dir/'relocation-check';replay.mkdir(exist_ok=True)
    code=replay/'code';model=replay/'model'
    if not code.exists():shutil.copytree(metadata['code_dir'],code)
    if not model.exists():shutil.copytree(metadata['model_artifact'],model)
    x=features(test,context).head(128);path=replay/'features.csv';x.to_csv(path,index=False)
    replay_config=RunConfig(total_seconds=40,final_reserve_seconds=0,reserve_cpus=0,reserve_memory_gb=0,max_workers=1)
    with SandboxManager(budget=Budget(replay_config,time.time()),max_workers=1) as manager:
        result=manager.run(dict(action='predict',code_dir=str(code),code_hash=tree_hash(code),
            model_artifact=str(model),features_path=str(path),kinds=[context.submission_kind]),
            replay/'worker',seconds=30,memory_gb=3,cpus=2)
    if result['status']!='completed':raise ValueError('Relocated model failed: '+str(result))
    prediction=pd.read_csv(result['predictions'][context.submission_kind]).to_numpy()
    reference=values[:len(x)]
    equal=np.array_equal(prediction,reference) if 'classification' in context.task.task_type else np.allclose(prediction.astype(float),reference.astype(float),rtol=1e-9,atol=1e-9)
    if not equal:raise ValueError('Relocated predictions differ')
    completed=sum(c['completed'] for c in cycles)
    split_ids={r['split_id'] for r in experiments if r['role']=='evaluation' and r['status']=='completed'}
    incumbent_rounds={r['round_id'] for r in experiments if r['incumbent'] and r['status']=='completed' and r.get('full_refit')}
    generated=all((Path(metadata['code_dir'])/name).is_file() for name in ['preprocessing.py','candidate.py'])
    if completed<3 or len(split_ids)<3 or len(incumbent_rounds)<2:raise ValueError('Three rotating cycles with refitted incumbents not completed')
    if not generated:raise ValueError('Separate generated preprocessing/model path missing')
    report=dict(status='passed',rows=len(output),primary_metric=expected_metric,completed_cycles=completed,
        outer_splits=len(split_ids),incumbent_rechecks=len(incumbent_rounds),generated_path=True,
        relocated_model_equal=True,submission_sha256=digest(Path(summary['submission']['path'])))
    write(replay/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['offline','real'],default='offline')
    parser.add_argument('--image',help='Docker image with the provided Python 3.12 dependencies')
    parser.add_argument('--model',choices=['qwen/qwen3.8-27b'],help='Explicit per-run model selection; never edits .env')
    parser.add_argument('--seconds',type=float,default=1800)
    parser.add_argument('--case',choices=['all','regression','multiclass'],default='all')
    parser.add_argument('--regression-task',type=Path,default=ROOT.parent/'data/Регрессия/task.md')
    parser.add_argument('--multiclass-task',type=Path,default=ROOT.parent/'data/Многоклассовая классификация/task.md')
    parser.add_argument('--verify-run',type=Path)
    parser.add_argument('--expected-rows',type=int)
    parser.add_argument('--expected-metric')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.verify_run:
        sys.path.insert(0,str(Path.cwd()))
        import solution
        report=verify_run(args.verify_run,args.expected_rows,args.expected_metric)
        write(out/'verification.json',report);print(json.dumps(report));return 0
    if not args.archive:parser.error('--archive is required')
    if args.seconds<=60 or (args.mode=='real' and args.seconds>1800):parser.error('Budget must exceed 60 seconds and real cases may use at most 1800 seconds each')
    sha=digest(args.archive)
    frozen_archive=out/'runtime-solution.zip'
    if frozen_archive.exists():
        if digest(frozen_archive)!=sha:raise ValueError('Output directory belongs to a different ZIP')
    else:shutil.copyfile(args.archive,frozen_archive)
    secrets={}
    if args.mode=='real':
        from mlmom.llm_client import LLMSettings,PLATFORM_MODEL
        settings=LLMSettings.from_env(str(ROOT/'.env'))
        if args.model:
            from dataclasses import replace
            settings=replace(settings,model=args.model)
        if settings.model!=PLATFORM_MODEL:raise ValueError('Configured model differs from organizer Qwen; no substitution allowed')
        secrets=dict(OPENAI_API_KEY=settings.api_key,OPENAI_BASE_URL=settings.base_url,OPENAI_MODEL=settings.model)
    specs=[('regression',args.regression_task,22449,'rmse'),('multiclass',args.multiclass_task,41166,'f1_macro')]
    reports_path=out/'reports.json'
    reports=json.loads(reports_path.read_text(encoding='utf-8')) if reports_path.exists() else []
    for name,task,rows,metric in specs:
        if args.case not in {'all',name}:continue
        case=out/name;case.mkdir(exist_ok=True)
        ledger_path=case/'budget.json'
        ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(archive_sha256=sha,deadline=time.time()+args.seconds,attempts=0)
        if ledger['archive_sha256']!=sha:raise ValueError('Use a new output directory for a different ZIP')
        remaining=min(args.seconds,ledger['deadline']-time.time())
        if remaining<=60:
            reports.append(dict(case=name,status='not_verified',reason='Original case budget exhausted',archive_sha256=sha));write(reports_path,reports);continue
        ledger['attempts']+=1;write(ledger_path,ledger)
        attempt=case/f"attempt-{ledger['attempts']}";attempt.mkdir()
        platform_dir=attempt/'task';platform_dir.mkdir()
        from mlmom.inputs import resolve_inputs
        files=resolve_inputs(task.resolve())
        for source in [files.task,files.train,files.test,*files.dictionaries,files.sample_submission,files.manifest]:
            if not source:continue
            source=Path(source);target=platform_dir/source.relative_to(files.root)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        with zipfile.ZipFile(frozen_archive) as package:
            if set(package.namelist())!={'solution.py','requirements.txt','mlmom_bundle.py'}:raise ValueError('Unexpected ZIP members')
            package.extractall(platform_dir)
        control=attempt/'control';control.mkdir()
        if args.mode=='offline':shutil.copyfile(ROOT/'tests/fixtures/release_sitecustomize.py',control/'sitecustomize.py')
        config=dict(profile='platform',total_seconds=max(1,remaining-50),final_reserve_seconds=120 if args.mode=='offline' else 300,
            max_workers=1,max_rounds=6,target_completed_rounds=3,reserve_cpus=0,reserve_memory_gb=0,
            experiment_seconds=120 if args.mode=='offline' else 420,initial_experiment_seconds=120,
            prediction_seconds=30,smoke_seconds=30,diagnostic_seconds=0,hpo_batch_size=2,hpo_max_trials=4,hpo_seconds=120,
            fixed_outer_split=False)
        write(control/'config.json',config)
        name_tag='mlmom-release-'+uuid.uuid4().hex[:10]
        if args.image:
            command=['docker','run','--rm','--name',name_tag,'--cpus','4','--memory','6g',
                '-v',f'{platform_dir}:/case','-v',f'{control}:/control','-v',f'{ROOT}:/fixtures:ro',
                '-w','/case','-e','MLMOM_CONFIG=/control/config.json','-e','MLMOM_RUN_DIR=/case/run']
            if args.mode=='offline':command+=['-e','MLMOM_RELEASE_OFFLINE=1','-e','PYTHONPATH=/control:/fixtures']
            if secrets:
                (control/'api.env').write_text('\n'.join(k+'='+v for k,v in secrets.items())+'\n',encoding='utf-8')
                command+=['--env-file',str(control/'api.env')]
            command+=['--entrypoint','python3',args.image,'solution.py','task.md']
            verify=['docker','run','--rm','--cpus','2','--memory','4g','-v',f'{platform_dir}:/case','-v',f'{ROOT}:/fixtures:ro',
                    '-w','/case','--entrypoint','python3',args.image,'/fixtures/scripts/release_cases.py',
                    '--verify-run','/case/run','--output','/case','--expected-rows',str(rows),'--expected-metric',metric]
            env=os.environ.copy()
        else:
            env=dict(os.environ,**secrets,MLMOM_CONFIG=str(control/'config.json'),MLMOM_RUN_DIR=str(platform_dir/'run'))
            if args.mode=='offline':env.update(MLMOM_RELEASE_OFFLINE='1',PYTHONPATH=os.pathsep.join([str(control),str(ROOT)]))
            command=[sys.executable,'solution.py','task.md']
            verify=[sys.executable,str(Path(__file__).resolve()),'--verify-run',str(platform_dir/'run'),
                    '--output',str(platform_dir),'--expected-rows',str(rows),'--expected-metric',metric]
        report=dict(case=name,attempt=ledger['attempts'],archive_sha256=sha,mode=args.mode,
            environment='linux/docker' if args.image else sys.platform,command=['python3' if args.image else sys.executable,'solution.py','task.md'],
            run_dir=str(platform_dir/'run'),status='failed')
        if args.mode=='real':report['model']=settings.model
        started=time.monotonic()
        try:
            with (attempt/'run.log').open('w',encoding='utf-8') as log:
                process=subprocess.Popen(command,cwd=platform_dir,env=env,stdout=log,stderr=subprocess.STDOUT)
                try:report['exit_code']=process.wait(timeout=max(1,ledger['deadline']-time.time()-45))
                except subprocess.TimeoutExpired:
                    if args.image:subprocess.run(['docker','rm','-f',name_tag],capture_output=True,timeout=20)
                    from mlmom.resources import terminate_tree
                    terminate_tree(process.pid);process.wait();report['reason']='Original case deadline reached'
                available=ledger['deadline']-time.time()
                if available>1:
                    checked=subprocess.run(verify,cwd=platform_dir,env={k:v for k,v in env.items() if k not in {'MLMOM_RELEASE_OFFLINE','PYTHONPATH'}},
                        stdout=log,stderr=subprocess.STDOUT,timeout=available)
                    if checked.returncode==0 and report.get('exit_code')==0:
                        report.update(status='passed',verification=json.loads((platform_dir/'verification.json').read_text(encoding='utf-8')))
                    else:report.setdefault('reason','Independent acceptance failed; see run.log')
        except Exception as error:report['reason']=str(error)
        finally:
            if (control/'api.env').exists():(control/'api.env').unlink()
        report['duration_seconds']=time.monotonic()-started
        summary_path=platform_dir/'run/summary.json'
        if summary_path.exists():
            summary=json.loads(summary_path.read_text(encoding='utf-8'))
            report.update(pipeline_stop_reason=summary.get('stop_reason'),process_acceptance=summary.get('process_acceptance'))
        reports.append(report);write(reports_path,reports);write(attempt/'report.json',report)
        print(json.dumps(report,ensure_ascii=False),flush=True)
    latest={r['case']:r for r in reports}
    required=['regression','multiclass'] if args.case=='all' else [args.case]
    return 0 if all(latest.get(n,{}).get('status')=='passed' for n in required) else 2


if __name__=='__main__':raise SystemExit(main())
