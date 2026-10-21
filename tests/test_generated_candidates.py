"""Generated implementations receive references, own transforms and preserve HPO."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from pydantic import ValidationError
from mlmom.coding import build_coding_graph
from mlmom.pipeline_contracts import RunConfig,HypothesisPlan
from mlmom.registry import SkillRegistry
from mlmom.sdk import FitContext,load_module
from mlmom.tools import verify_source
from tests.fakes import GENERATED_CODE,ADAPTER_CODE
from tests.test_planning_limits import case,Store,ClockBudget,Sandbox,Coder,Planner,Skills
from mlmom.planning import build_planning_graph

ROOT=Path(__file__).resolve().parents[1]


class GeneratedCandidateTests(unittest.TestCase):
    def test_selected_reference_is_coder_only_versioned_and_budgeted(self):
        registry=SkillRegistry(ROOT/'skills')
        brief=registry.load(['sklearn'],'regression')
        coding=registry.load(['sklearn'],'regression',include_references=True,max_chars=48000)
        a=next(s for s in brief if s['skill_id']=='sklearn')
        b=next(s for s in coding if s['skill_id']=='sklearn')
        self.assertNotIn('class Candidate:',a['content'])
        self.assertIn('class Candidate:',b['content'])
        self.assertNotEqual(a['sha256'],b['sha256'])
        self.assertNotIn('XGBRegressor', ''.join(s['content'] for s in coding))
        with self.assertRaisesRegex(ValueError,'context budget'):
            registry.load(['sklearn'],'regression',include_references=True,max_chars=100)

    def test_old_mode_and_forwarding_source_cannot_enter_new_candidates(self):
        data=case()[3].model_dump();data['execution_mode']='adapter_only'
        with self.assertRaises(ValidationError):HypothesisPlan.model_validate(data)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'candidate.py'
            source.write_text(ADAPTER_CODE,encoding='utf-8')
            self.assertFalse(verify_source(root).valid)
            source.write_text(GENERATED_CODE,encoding='utf-8')
            self.assertTrue(verify_source(root).valid)

    def test_first_cycle_allows_real_preprocessing_and_keeps_declared_hpo_space(self):
        class TransformPlanner(Planner):
            def generate(self,role,instruction,*args,**kwargs):
                result=super().generate(role,instruction,*args,**kwargs)
                if role=='planner':
                    for p in result.plans:
                        p.preprocessing=['cast Year_Factor to categorical']
                        p.search_space={'n_estimators':dict(type='int',low=4,high=12)}
                return result
        context,tools,budget,_=case()
        result=build_planning_graph(TransformPlanner(),context,tools,Skills(),budget,
            planning_mode='baseline').invoke({'payload':{'round_number':1}})
        self.assertTrue(result['valid'],result)
        self.assertEqual(result['plans'][0]['execution_mode'],'custom')
        self.assertEqual(result['plans'][0]['strategy'],'single')
        self.assertTrue(result['plans'][0]['preprocessing'])
        self.assertTrue(result['plans'][0]['search_space'])

    def test_generated_cast_and_clipping_execute_and_reload(self):
        code=GENERATED_CODE.replace(
            'return X.loc[:, self.columns].replace([np.inf, -np.inf], np.nan)',
            "result = X.loc[:, self.columns].replace([np.inf, -np.inf], np.nan)\n        result['Year_Factor'] = result['Year_Factor'].fillna(-1).astype(str)\n        return result")
        code=code.replace('self.kind = context.task_type','self.bounds = (float(np.min(y)),float(np.max(y)))\n        self.kind = context.task_type')
        code=code.replace('return self.model.predict(values)','return np.clip(self.model.predict(values), *self.bounds)')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'candidate.py';source.write_text(code,encoding='utf-8')
            self.assertTrue(verify_source(root).valid)
            module=load_module(root)
            X=pd.DataFrame({'Year_Factor':[1,2,1,3,2,3]*10,'площадь / м²':np.arange(60.)})
            y=pd.Series(np.arange(60.)+1)
            model=module.create_candidate({'model_params':{'n_estimators':4}})
            model.fit(X,y,FitContext(task_type='regression',metric='rmse',direction='minimize',seed=42,
                cpus=1,memory_gb=2,time_limit_seconds=20))
            self.assertIn('Year_Factor',model.preprocessor.transformers_[1][2])
            probe=X.iloc[:8].copy();probe['Year_Factor']=99
            predictions=model.predict(probe,{'kind':'value'})
            self.assertTrue(np.isfinite(predictions).all())
            artifact=root/'artifact';artifact.mkdir();model.save(str(artifact))
            restored=module.load_candidate(str(artifact))
            np.testing.assert_array_equal(predictions,restored.predict(probe,{'kind':'value'}))
            self.assertFalse(hasattr(restored,'context'))


if __name__=='__main__':unittest.main()
