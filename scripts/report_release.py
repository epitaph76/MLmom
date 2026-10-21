"""Record reproducible package checks and offline validation evidence."""
from __future__ import annotations

import ast
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
    tests=json.loads((root/'artifacts/reliable-start-release/tests.json').read_text(encoding='utf-8'))
    if tests['status']!='passed':raise ValueError('Packaged test suite did not pass')
    reports_path=root/'artifacts/reliable-start-release/organizer/reports.json'
    if not reports_path.exists(): reports_path=root/'artifacts/reliable-start-organizer/reports.json'
    reports=json.loads(reports_path.read_text(encoding='utf-8'))
    if len(reports)!=2 or any(report['status']!='completed' for report in reports):
        raise ValueError('Both organizer tasks must complete')
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
    result=dict(date='2026-10-21',base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        working_tree_changes=True,package=dict(path='submission/mlmom-solution.zip',root_files=names,
            bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            source_bundle_digest=assignments['DIGEST'],bundled_source_matches_checkout=True,
            secret_scan='passed: configured keys absent'),
        tests=dict(tests,source='extracted final ZIP',python=platform.python_version(),os=platform.system(),llm='mock',
                   report='artifacts/reliable-start-release/tests.log'),
        organizer_smoke=dict(llm='mock',tool='catboost',rounds_per_task=2,
            report=reports_path.relative_to(root).as_posix(),
            source='extracted final ZIP' if 'reliable-start-release' in reports_path.parts else 'checkout',
            tasks=[dict(metric=r['primary_metric'],status=r['status'],rows=r['rows'],
                model_role=r['submission']['model_role'],first_submission_seconds=r['first_submission']['elapsed_seconds']) for r in reports]),
        adapters=tools,limits=dict(current_real_api_run=False,three_hour_stress_run=False,
                                  linux_run=False,platform_score_verified=False))
    (root/'submission/validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='passed',bytes=result['package']['bytes'],sha256=result['package']['sha256'],tests=tests['count'])))


if __name__=='__main__':main()
