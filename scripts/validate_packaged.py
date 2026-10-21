"""Run offline tests against bundled runtime code, with repository tests as fixtures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import zipfile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('archive',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    platform=output/'platform';platform.mkdir(exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        if set(archive.namelist())!={'solution.py','requirements.txt','mlmom_bundle.py'}:
            raise ValueError('Unexpected archive members')
        archive.extractall(platform)
    code='''import json,sys,time,unittest
from pathlib import Path
platform,repository,report=map(Path,sys.argv[1:])
sys.path.insert(0,str(platform))
import solution,mlmom
runtime=Path(mlmom.__file__).resolve().parents[1]
assert runtime.parent==platform.resolve(),runtime
# Tests are external fixtures; imported production modules stay bound to the ZIP.
sys.path.append(str(repository))
suite=unittest.defaultTestLoader.discover(str(repository/'tests'),top_level_dir=str(repository))
started=time.monotonic()
result=unittest.TextTestRunner(verbosity=2).run(suite)
report.write_text(json.dumps(dict(status='passed' if result.wasSuccessful() else 'failed',
    count=result.testsRun,failures=len(result.failures),errors=len(result.errors),
    skipped=len(result.skipped),duration_seconds=time.monotonic()-started,runtime=str(runtime)),indent=2),encoding='utf-8')
raise SystemExit(0 if result.wasSuccessful() else 1)
'''
    with (output/'tests.log').open('w',encoding='utf-8') as log:
        completed=subprocess.run([sys.executable,'-I','-c',code,str(platform),str(root),str(output/'tests.json')],
            cwd=output,stdout=log,stderr=subprocess.STDOUT,text=True)
    if (output/'tests.json').exists():print((output/'tests.json').read_text(encoding='utf-8'))
    return completed.returncode


if __name__=='__main__':raise SystemExit(main())
