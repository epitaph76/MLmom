"""Native CV is never inferred from an installed AutoML library."""
import json
import numpy as np
import pandas as pd
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlmom.pipeline_contracts import BudgetSnapshot
from mlmom.planning import validate_plan
from mlmom.registry import installed_tools, SkillRegistry
from mlmom.resources import SandboxManager
from mlmom.persistence import tree_hash
from tests.fakes import PREPROCESSING_CODE
from tests.test_execution import execution_case

ROOT=Path(__file__).resolve().parents[1]


class NativeValidationTests(unittest.TestCase):
    def test_native_cv_and_unverified_backends_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,context,plan,_=execution_case(Path(tmp),'regression')
            tools=installed_tools()
            skills=SkillRegistry(ROOT/'skills')
            budget=BudgetSnapshot(remaining_seconds=200,research_seconds=100,available_cpus=4,available_memory_gb=8,spent_seconds=0)
            for name in ['flaml','autogluon','lightautoml']:
                tools[name].available=True
                plan.tool_id=name;plan.skill_ids=[name];plan.strategy='native';plan.patience_trials=None
                with patch('mlmom.registry.importlib.metadata.version',return_value='test'):
                    result=validate_plan(plan,context,tools,skills,budget)
                self.assertFalse(result.valid)
                self.assertTrue(any('verified' in e for e in result.errors))

    def test_generated_flaml_holdout_freezes_refits_and_relocates(self):
        if installed_tools()['flaml'].version is None:
            self.skipTest('FLAML not installed; this environment does not verify native execution')
        for kind in ['regression','binary_classification','multiclass_classification']:
            with self.subTest(kind=kind):self.check_generated_holdout(kind)

    def check_generated_holdout(self,kind):
        with tempfile.TemporaryDirectory() as tmp, SandboxManager() as manager:
            root=Path(tmp)
            frame,context,plan,code=execution_case(root,kind)
            if 'classification' in kind:
                mapping={0:'Alpha',1:'Beta',2:'Gamma'}
                frame['t0']=frame['t0'].map(mapping);frame.to_csv(root/'train.csv',index=False)
                context.classes=list(reversed(sorted(frame['t0'].unique())))
            context.task=type(context.task).model_validate(dict(context.task.model_dump(),validation_strategy='holdout'))
            plan.tool_id='flaml';plan.skill_ids=['flaml'];plan.strategy='native';plan.patience_trials=None
            plan.model_params={};plan.max_trials=2
            source=(ROOT/'skills/coding_agent/flaml/REFERENCE.md').read_text(encoding='utf-8').split('```python\n',1)[1].split('```',1)[0]
            (code/'candidate.py').write_text(source,encoding='utf-8')
            (code/'preprocessing.py').write_text(PREPROCESSING_CODE,encoding='utf-8')
            result=manager.run(dict(action='train',code_dir=str(code),code_hash=tree_hash(code),
                context=context.model_dump(mode='json'),plan=plan.model_dump(mode='json'),train_path=str(root/'train.csv')),
                root/'train',seconds=40,memory_gb=3,cpus=1)
            self.assertEqual(result['status'],'completed',result)
            self.assertTrue(result['full_refit'])
            self.assertIn('__native_frozen',result['best_params'])
            history=json.loads(Path(result['native_history_ref']).read_text())
            self.assertEqual(history['new_trials'],0)
            moved=root/'moved';Path(result['model_artifact']).rename(moved)
            from mlmom.sdk import load_module
            restored=load_module(code).load_candidate(moved)
            X=pd.read_csv(root/'train.csv')[context.feature_columns]
            prediction=restored.predict(X,{'kind':plan.prediction_kind})
            self.assertEqual(len(prediction),60)
            if 'classification' in kind:
                self.assertTrue(set(prediction)<=set(context.classes))
                probability=restored.predict(X,{'kind':'probability'})
                self.assertEqual(probability.shape,(60,len(context.classes)))
                np.testing.assert_allclose(probability.sum(axis=1),1)
                sorted_probability=np.sort(probability,axis=1)
                unique=(sorted_probability[:,-1]-sorted_probability[:,-2])>1e-10
                np.testing.assert_array_equal(np.asarray(context.classes)[probability[unique].argmax(axis=1)],np.asarray(prediction)[unique])


if __name__=='__main__':unittest.main()
