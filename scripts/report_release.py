"""Record reproducible package checks and offline validation evidence."""
from __future__ import annotations

import ast
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import zipfile
from dotenv import dotenv_values


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser()
    parser.add_argument('--tests',type=Path,default=root/'artifacts/repaired-release-final-v2/tests.json')
    parser.add_argument('--real-reports',type=Path)
    args=parser.parse_args()
    tests_path=args.tests.resolve()
    tests=json.loads(tests_path.read_text(encoding='utf-8'))
    if tests['status']!='passed':raise ValueError('Packaged test suite did not pass')
    reports=json.loads(args.real_reports.read_text(encoding='utf-8')) if args.real_reports else []
    archive=root/'submission/mlmom-solution.zip'
    with zipfile.ZipFile(archive) as package:
        names=package.namelist()
        if set(names)!={'solution.py','requirements.txt','mlmom_bundle.py'}:raise ValueError('Unexpected package members')
        entries={name:package.read(name) for name in names}
    assignments={node.targets[0].id:ast.literal_eval(node.value)
                 for node in ast.parse(entries['mlmom_bundle.py'].decode('utf-8')).body
                 if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)
                 and node.targets[0].id in {'FILES','DIGEST'}}
    sources=assignments['FILES']
    tested_runtime=Path(tests['runtime'])
    if tested_runtime.name!='.mlmom-runtime-'+assignments['DIGEST'][:16]:
        raise ValueError('Tested runtime digest differs from package')
    if any((tested_runtime/name).read_text(encoding='utf-8')!=text for name,text in sources.items()):
        raise ValueError('Tested runtime files differ from package')
    expected={path.relative_to(root).as_posix():path.read_text(encoding='utf-8')
              for directory in ['mlmom','skills','profiles'] for path in (root/directory).rglob('*')
              if path.is_file() and path.suffix in {'.py','.json','.md'} and '__pycache__' not in path.parts}
    if sources!=expected:raise ValueError('Bundled sources differ from current checkout')
    for name in ['solution.py','requirements.txt']:
        if entries[name].decode('utf-8')!=(root/name).read_text(encoding='utf-8'):
            raise ValueError('Root entry differs from checkout: '+name)
    known=dict(dotenv_values(root/'.env'));known.update(os.environ)
    secrets=[str(value).encode() for key,value in known.items() if value and len(str(value))>=8 and
             (key.upper().endswith('_TOKEN') or any(part in key.upper() for part in ('API_KEY','APIKEY','PASSWORD','SECRET')))]
    if any(secret in content for secret in secrets for content in entries.values()):
        raise ValueError('Configured secret detected in package')
    tools={}
    sys.path.insert(0,str(root))
    from mlmom.registry import installed_tools
    for name,spec in installed_tools().items():
        tools[name]=dict(version=spec.version,installed=spec.reason=='adapter probe required')
    attempts=[]
    release_hash=hashlib.sha256(archive.read_bytes()).hexdigest()
    for report in reports:
        manifest=json.loads((Path(report['run_dir'])/'input_manifest.json').read_text(encoding='utf-8'))
        attempts.append(dict(case=report['case'],attempt=report['attempt'],duration_seconds=report['duration_seconds'],
            exit_code=report['exit_code'],verification=report.get('verification'),run_dir=report['run_dir'],
            runtime_matches_release=manifest['runtime_archive_sha256']==release_hash,
            process_acceptance=report.get('result',{}).get('process_acceptance',{})))
    passed_cases={r['case'] for r in attempts if r['runtime_matches_release'] and r.get('verification',{}).get('status')=='passed'}
    cycle_cases={r['case'] for r in attempts if r['runtime_matches_release'] and r.get('verification',{}).get('status')=='passed' and r['process_acceptance'].get('passed')}
    result=dict(date='2026-10-21',base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        working_tree_changes=True,package=dict(path='submission/mlmom-solution.zip',root_files=names,
            bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            source_bundle_digest=assignments['DIGEST'],bundled_source_matches_checkout=True,
            secret_scan='passed: configured keys absent'),
        tests=dict(tests,source='extracted final ZIP',python=platform.python_version(),os=platform.system(),llm='mock',
                   report=str(tests_path.with_name('tests.log'))),
        real_api=dict(model='qwen/qwen3.8-27b',report=str(args.real_reports.resolve()) if args.real_reports else None,
            attempts=attempts,both_real_tasks_passed=passed_cases=={'regression','multiclass'},
            both_three_cycle_tasks_passed=cycle_cases=={'regression','multiclass'}),
        adapters=tools,limits=dict(current_real_api_run=bool(passed_cases),three_hour_stress_run=False,
                                  linux_run=False,platform_score_verified=False))
    (root/'submission/validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='passed',bytes=result['package']['bytes'],sha256=result['package']['sha256'],tests=tests['count'])))


if __name__=='__main__':main()
