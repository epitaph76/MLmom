"""Workers must retain the controller's effective Python package environment."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import venv

import mlmom
from mlmom.resources import worker_runtime


ROOT = Path(mlmom.__file__).resolve().parents[1]


class WorkerRuntimeTests(unittest.TestCase):
    def test_paths_are_absolute_deduplicated_and_executable_is_not_resolved(self):
        with patch.object(sys, 'path', ['', 'relative-packages', 'relative-packages']), \
             patch.object(sys, 'executable', '/virtualenv/bin/python'):
            runtime = worker_runtime(ROOT)
        self.assertEqual(runtime['executable'], '/virtualenv/bin/python')
        self.assertEqual(runtime['sys_path'][0], str(ROOT.resolve()))
        self.assertEqual(len(runtime['sys_path']), len(set(runtime['sys_path'])))
        self.assertTrue(all(os.path.isabs(path) for path in runtime['sys_path']))

    def test_worker_keeps_packages_supplied_only_by_parent_import_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            environment = root/'isolated-venv'
            venv.EnvBuilder(with_pip=False).create(environment)
            executable = environment/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
            # This venv has no third-party packages. The controller gets its
            # dependencies from external import paths, as some runners do.
            env = dict(os.environ, PYTHONPATH=os.pathsep.join(worker_runtime(ROOT)['sys_path']),
                       PYTHONIOENCODING='utf-8', OPENAI_API_KEY='runtime-test-secret')
            code = '''import json,sys,os
from pathlib import Path
from mlmom.resources import SandboxManager
directory=Path(sys.argv[1])
with SandboxManager() as manager:
    check=manager.run(dict(action='runtime_check'),directory/'check',seconds=30,memory_gb=2,cpus=1)
    assert check['status']=='completed',check
    assert {'numpy','pandas','sklearn','pydantic','psutil','polars','pyarrow'}<=set(check['versions'])
    result=manager.run(dict(action='probe',config={'tool_id':'sklearn','model_params':{}},
        task_type='regression'),directory,seconds=30,memory_gb=2,cpus=1)
assert result['status']=='completed',result
runtime=json.loads((directory/'runtime.json').read_text(encoding='utf-8'))
assert runtime['controller']['prefix']==runtime['worker']['prefix']==sys.prefix,runtime
assert runtime['controller']['executable']==runtime['worker']['executable']==sys.executable,runtime
assert 'runtime-test-secret' not in json.dumps(runtime)
print(json.dumps(result))
'''
            result = subprocess.run([str(executable), '-c', code, str(root/'job')],
                cwd=root, env=env, capture_output=True, text=True, encoding='utf-8', timeout=45)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertEqual(json.loads(result.stdout)['status'], 'completed')

    def test_preflight_stops_before_llm_on_import_prefix_or_version_failure(self):
        from mlmom.pipeline import Pipeline
        from mlmom.pipeline_contracts import RunConfig
        from mlmom.resources import runtime_versions
        from tests.fakes import OfflineLLM
        from tests.test_pipeline import dataset
        runtime = worker_runtime(ROOT)
        versions = runtime_versions()
        cases = [dict(status='failed', error="ModuleNotFoundError: No module named 'pydantic'"),
                 dict(status='completed', runtime=dict(runtime, prefix='/wrong-venv'), versions=versions),
                 dict(status='completed', runtime=runtime, versions=dict(versions, numpy='wrong-version'))]
        for response in cases:
            with self.subTest(response=response), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                client = OfflineLLM()
                pipeline = Pipeline(dataset(root/'input'),
                    RunConfig(total_seconds=60, final_reserve_seconds=10),
                    run_dir=root/'run', client=client)
                with patch.object(pipeline.sandbox, 'run', return_value=response) as launch, \
                     patch.object(pipeline, 'graph') as graph:
                    result = pipeline.run()
                self.assertEqual(result['stop_reason'], 'worker_runtime_error', result)
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(client.calls, [])
                graph.assert_not_called()
                self.assertEqual(launch.call_count, 1)
                self.assertEqual(launch.call_args.args[0]['action'], 'runtime_check')
                self.assertEqual(result['failure_causes'][0]['kind'], 'worker_runtime_failed')
