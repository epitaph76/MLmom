"""No stale ZIP, missing platform evidence or unfinished run can pass release."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.report_release import gate


class ReleaseGateTests(unittest.TestCase):
    def test_missing_or_mismatched_evidence_never_passes(self):
        self.assertEqual(gate('current')['status'],'not_verified')
        passed=dict(status='passed',archive_sha256='old')
        self.assertEqual(gate('current',packaged=passed)['status'],'failed')
        failed=dict(status='failed',archive_sha256='current',failures=1)
        self.assertEqual(gate('current',packaged=failed)['status'],'failed')

    def test_complete_local_acceptance_keeps_platform_separate(self):
        test=dict(status='passed',archive_sha256='zip',python='3.12.15',os='Linux')
        def cases(mode):
            return [dict(case=name,status='passed',archive_sha256='zip',mode=mode,environment='linux/docker',model='qwen/qwen3.8-27b',
                verification=dict(status='passed',completed_cycles=3,generated_path=True,relocated_model_equal=True))
                for name in ['regression','multiclass']]
        report=gate('zip',test,test,test,cases('offline'),cases('real'))
        self.assertEqual(report['status'],'passed')
        self.assertEqual(report['checks']['platform']['status'],'not_verified')
        partial=cases('real');partial[1]['verification']['completed_cycles']=1
        self.assertEqual(gate('zip',test,test,test,cases('offline'),partial)['status'],'failed')

    def test_exact_entrypoint_environment_config_and_cli_precedence(self):
        import solution
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'config.json';path.write_text(json.dumps(dict(total_seconds=1800,final_reserve_seconds=300)))
            with patch.dict('os.environ',dict(MLMOM_CONFIG=str(path),MLMOM_RUN_DIR=str(Path(tmp)/'run'))),patch('mlmom.pipeline.Pipeline') as pipeline,patch('sys.argv',['solution.py','task.md']):
                pipeline.return_value.run.return_value=dict(status='completed')
                self.assertEqual(solution.main(),0)
                self.assertEqual(pipeline.call_args.args[1].total_seconds,1800)
                self.assertEqual(pipeline.call_args.kwargs['run_dir'],Path(tmp)/'run')


if __name__=='__main__':unittest.main()
