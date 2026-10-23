"""Exercise the exact root-file copying behavior of the organizer platform."""
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from scripts.package_solution import package


class FlatPackageTests(unittest.TestCase):
    def test_flat_archive_bootstraps_skills_and_worker_without_repository(self):
        root=Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            archive=directory/'solution.zip'
            package(root,archive)
            second=directory/'second.zip'
            package(root,second)
            self.assertEqual(archive.read_bytes(),second.read_bytes())
            with zipfile.ZipFile(archive) as source:
                self.assertEqual(set(source.namelist()),{'solution.py','requirements.txt','mlmom_bundle.py'})
                self.assertEqual(source.read('requirements.txt'),(root/'requirements.txt').read_text(encoding='utf-8').encode('utf-8'))
                source.extractall(directory/'platform')
            script='''import sys,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import solution,mlmom
from mlmom.registry import SkillRegistry
from mlmom.resources import SandboxManager
runtime=Path(mlmom.__file__).resolve().parent.parent
assert runtime.parent==Path(sys.argv[1]).resolve()
registry=SkillRegistry(runtime/'skills')
assert len(registry.list())==32
catalog={s['skill_id'] for s in registry.technique_catalog('multiclass_classification')}
assert {'categorical_encoding','missing_values','preprocessing_core','imbalance'}<=catalog
assert registry.owners['imbalance']=='coding_agent'
assert registry.owners['ensemble_agent']=='ensemble_agent'
assert {card['skill_id'] for card in registry.load([], 'regression', role='ensemble_agent')}=={'ensemble_agent'}
from mlmom.sdk import FrozenEnsemble
from mlmom.pipeline_contracts import SearchGroupPlan
assert FrozenEnsemble.__module__=='mlmom.composite'
loaded=registry.load(['sklearn'],'regression')
assert {s['skill_id'] for s in loaded}=={'candidate_sdk','validation','metrics','preprocessing','sklearn'}
assert all(s['status']=='ready' for s in loaded)
assert loaded[0]['content'].startswith((runtime/'skills'/'shared'/'COMMON.md').read_text(encoding='utf-8'))
with SandboxManager() as sandbox:
    result=sandbox.run({'action':'probe','config':{'tool_id':'sklearn','model_params':{}},'task_type':'regression'},
        Path(sys.argv[1])/'worker',seconds=30,memory_gb=2,cpus=1)
assert result['status']=='completed',result
print(json.dumps({'status':'ok'}))
'''
            result=subprocess.run([sys.executable,'-I','-c',script,str(directory/'platform')],
                                  cwd=directory,capture_output=True,text=True,timeout=50)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertEqual(json.loads(result.stdout)['status'],'ok')
