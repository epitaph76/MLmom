"""Real installed-library tests; full Docker environment exercises every adapter."""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from mlmom.registry import installed_tools
from mlmom.resources import SandboxManager


class AdapterTests(unittest.TestCase):
    def test_every_installed_library_fit_save_load_and_frozen_native_config(self):
        for name,spec in installed_tools().items():
            if spec.reason!='adapter probe required':continue
            if spec.native_hpo and not spec.validation_strategies:continue
            with self.subTest(tool=name),tempfile.TemporaryDirectory() as tmp, SandboxManager(timeouts_enabled=True) as manager:
                result=manager.run(dict(action='probe',config={'tool_id':name,'model_params':{}},task_type='regression',verify_frozen=spec.native_hpo),
                    Path(tmp)/'probe',seconds=60,memory_gb=3,cpus=1)
                self.assertEqual(result['status'],'completed',result)

    def test_native_multiclass_probabilities(self):
        specs=installed_tools()
        for name in ['flaml','autogluon','lightautoml']:
            if specs[name].reason!='adapter probe required':continue
            if not specs[name].validation_strategies:continue
            with self.subTest(tool=name),tempfile.TemporaryDirectory() as tmp, SandboxManager(timeouts_enabled=True) as manager:
                result=manager.run(dict(action='probe',config={'tool_id':name,'model_params':{}},task_type='multiclass_classification',
                    prediction_kind='probability',metric='log_loss',verify_frozen=True),
                    Path(tmp)/'probe',seconds=60,memory_gb=3,cpus=1)
                self.assertEqual(result['status'],'completed',result)


if __name__=='__main__':unittest.main()
