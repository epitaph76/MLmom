"""Write a machine-readable source-suite report and retain the complete log."""
import argparse
import contextlib
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(ROOT))
    from scripts.package_solution import package
    def archive_hash():
        with tempfile.TemporaryDirectory() as tmp:
            archive=Path(tmp)/'solution.zip';package(ROOT,archive)
            return hashlib.sha256(archive.read_bytes()).hexdigest()
    source_hash=archive_hash()
    started=time.monotonic()
    with (args.output/'tests.log').open('w',encoding='utf-8') as log, contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),top_level_dir=str(ROOT))
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    report=dict(status='passed' if result.wasSuccessful() else 'failed',count=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        duration_seconds=time.monotonic()-started,python=platform.python_version(),os=platform.system(),
        source='checkout',runtime=str(ROOT))
    report['archive_sha256']=source_hash
    if source_hash!=archive_hash():report.update(status='failed',reason='Production sources changed while tests were running')
    (args.output/'tests.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
