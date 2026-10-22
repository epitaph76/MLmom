"""Fail-closed release gate: every executed check must identify the same ZIP."""
from __future__ import annotations
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def evidence(report, sha, *, linux=False):
    if report is None:return dict(status='not_verified',reason='No report supplied')
    if report.get('archive_sha256')!=sha:return dict(status='failed',reason='ZIP SHA256 mismatch',report=report)
    valid=report.get('status')=='passed'
    if linux:valid=valid and report.get('os')=='Linux' and report.get('python','').startswith('3.12.')
    if report.get('failures',0) or report.get('errors',0):valid=False
    return dict(status='passed' if valid else 'failed',report=report)


def cases(reports, sha, mode):
    latest={r['case']:r for r in (reports or [])}
    checks={}
    for name in ['regression','multiclass']:
        check=evidence(latest.get(name),sha)
        if check['status']=='passed':
            r=latest[name];v=r.get('verification',{})
            if (r.get('mode')!=mode or r.get('environment')!='linux/docker' or v.get('status')!='passed'
                    or v.get('completed_cycles',0)<3 or not v.get('generated_path') or not v.get('relocated_model_equal')):
                check=dict(status='failed',reason='Missing Linux/generated/three-cycle/relocation evidence',report=r)
            if mode=='real' and r.get('model')!='qwen/qwen3.8-27b':
                check=dict(status='failed',reason='Required Qwen model not verified',report=r)
        checks[name]=check
    status='passed' if all(c['status']=='passed' for c in checks.values()) else 'failed' if any(c['status']=='failed' for c in checks.values()) else 'not_verified'
    return dict(status=status,cases=checks)


def gate(sha, source=None, packaged=None, linux=None, offline=None, real=None):
    checks=dict(source_tests=evidence(source,sha),packaged_tests=evidence(packaged,sha),
        linux_tests=evidence(linux,sha,linux=True),offline=cases(offline,sha,'offline'),real_api=cases(real,sha,'real'),
        platform=dict(status='not_verified',reason='Organizer platform has not been exercised'))
    required=[v for k,v in checks.items() if k!='platform']
    status='passed' if all(c['status']=='passed' for c in required) else 'failed' if any(c['status']=='failed' for c in required) else 'not_verified'
    return dict(status=status,checks=checks)


def package_info(path):
    with zipfile.ZipFile(path) as archive:
        if set(archive.namelist())!={'solution.py','requirements.txt','mlmom_bundle.py'}:raise ValueError('Unexpected ZIP members')
        entries={name:archive.read(name) for name in archive.namelist()}
    assignments={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(entries['mlmom_bundle.py'].decode('utf-8')).body
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in {'FILES','DIGEST'}}
    sources={p.relative_to(ROOT).as_posix():p.read_text(encoding='utf-8') for folder in ['mlmom','skills','profiles']
             for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in {'.py','.json','.md'} and '__pycache__' not in p.parts}
    if assignments['FILES']!=sources:raise ValueError('Bundled sources differ from checkout')
    for name in ['solution.py','requirements.txt']:
        if entries[name].decode('utf-8')!=(ROOT/name).read_text(encoding='utf-8'):raise ValueError('Root entry differs from checkout: '+name)
    from dotenv import dotenv_values
    known=dict(dotenv_values(ROOT/'.env'));known.update(os.environ)
    secrets=[str(v).encode() for k,v in known.items() if v and len(str(v))>=8 and
             (k.upper().endswith('_TOKEN') or any(s in k.upper() for s in ['API_KEY','APIKEY','PASSWORD','SECRET']))]
    if any(s in content for s in secrets for content in entries.values()):raise ValueError('Configured secret detected in ZIP')
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,
        root_files=sorted(entries),source_bundle_digest=assignments['DIGEST'],bundled_source_matches_checkout=True,secret_scan='passed',
        requirements_sha256=hashlib.sha256(entries['requirements.txt']).hexdigest())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path,default=ROOT/'submission/mlmom-solution.zip')
    for name in ['source-tests','tests','linux-tests','offline-reports','real-reports']:parser.add_argument('--'+name,type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'submission/validation.json')
    parser.add_argument('--environment',type=Path)
    args=parser.parse_args()
    read=lambda p:json.loads(p.read_text(encoding='utf-8')) if p and p.exists() else None
    info=package_info(args.archive)
    report=gate(info['sha256'],read(args.source_tests),read(args.tests),read(args.linux_tests),read(args.offline_reports),read(args.real_reports))
    environment=read(args.environment)
    report['checks']['environment']=dict(status='not_verified' if environment is None else 'passed' if
        environment.get('status')=='passed' and environment.get('requirements_sha256')==info['requirements_sha256'] else 'failed',report=environment)
    if report['checks']['environment']['status']!='passed':
        report['status']='failed' if report['status']=='failed' or report['checks']['environment']['status']=='failed' else 'not_verified'
    report.update(created_utc=datetime.now(timezone.utc).isoformat(),base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),package=info,
        environment_limits=['Docker image approximates provided dependencies; organizer infrastructure is not verified.',
                            'Offline LLM is an external test fixture; production runtime has no offline fallback.',
                            'Real API allowance is at most 1800 seconds per task, including all attempts and verification.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],sha256=info['sha256'],checks={k:v['status'] for k,v in report['checks'].items()})))
    return 0 if report['status']=='passed' else 2


if __name__=='__main__':raise SystemExit(main())
