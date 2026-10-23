"""Retest the affected fixture modules; preserve the original full-suite result."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--prior-report',type=Path,required=True)
    parser.add_argument('--platform',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    sha=hashlib.sha256(args.archive.read_bytes()).hexdigest()
    previous=json.loads(args.prior_report.read_text(encoding='utf-8'))
    if previous.get('archive_sha256')!=sha or previous.get('errors')!=1 or previous.get('failures')!=0:
        raise ValueError('This verification only resolves the one identified fixture error on the identical ZIP')
    old_log=args.prior_report.with_name('tests.log').read_text(encoding='utf-8')
    error_names=[line for line in old_log.splitlines() if line.startswith(('ERROR:','FAIL:'))]
    expected='test_failed_study_trial_cannot_supply_selected_configuration'
    if len(error_names)!=1 or expected not in error_names[0]:raise ValueError('Unexpected full-suite failure')
    if args.platform:
        sys.path.insert(0,str(args.platform.resolve()))
        import solution,mlmom
        if Path(mlmom.__file__).resolve().parents[2]!=args.platform.resolve():raise ValueError('Wrong extracted runtime')
    sys.path.append(str(ROOT))
    if not args.platform:
        from scripts.report_release import package_info
        if package_info(args.archive)['sha256']!=sha:raise ValueError('Runtime archive mismatch')
    import mlmom
    suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_inner_validation','tests.test_target_encoding_boundary'])
    def names(suite):
        for item in suite:
            if isinstance(item,unittest.TestSuite):yield from names(item)
            else:yield item.id()
    selected=list(names(suite))
    if not any(expected in name for name in selected):raise ValueError('Affected regression test omitted')
    started=time.monotonic()
    with (args.output/'tests.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    report=dict(status='passed' if result.wasSuccessful() else 'failed',archive_sha256=sha,count=previous['count'],
        failures=len(result.failures),errors=len(result.errors),skipped=previous.get('skipped',0),
        runtime=str(Path(mlmom.__file__).resolve().parents[1]),os=platform.system(),python=platform.python_version(),
        source='full suite plus affected-module rerun',initial_full_suite=previous,
        initial_full_suite_report=str(args.prior_report),
        verification_method='The runtime ZIP is unchanged. One obsolete fixture passed a custom offset as a backend parameter. It now uses native DummyRegressor.constant. Both affected modules were rerun; the original failed report/log remain intact.',
        fixture_sha256=hashlib.sha256((ROOT/'tests/test_inner_validation.py').read_bytes()).hexdigest(),
        rerun=dict(count=result.testsRun,test_ids=selected,status='passed' if result.wasSuccessful() else 'failed',
                   duration_seconds=time.monotonic()-started))
    (args.output/'tests.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],count=report['count'],rerun_count=result.testsRun,os=report['os'])))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
