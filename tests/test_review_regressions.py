"""Regression coverage for the cross-module review findings."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from mlmom.evaluation import score_predictions, checked_predictions
from mlmom.inputs import read_table
from mlmom.sdk import PreparedCandidate, FitContext, load_module
from mlmom.worker import write_predictions
from mlmom.planning import validate_plan
from mlmom.pipeline import Pipeline
from mlmom.pipeline_contracts import RunConfig
from tests.test_pipeline import dataset, only_sklearn
from tests.test_planning_limits import case, Skills
from tests.fakes import OfflineLLM

class MeanPreprocessor:
    def fit_transform(self,X,y,context):
        self.mean=float(np.mean(y))
        return pd.DataFrame({'mean': np.full(len(X),self.mean)},index=X.index)
    def transform(self,X):return pd.DataFrame({'mean': np.full(len(X),self.mean)},index=X.index)
class Base:
    def fit(self,X,y,context):
        self.model=Ridge().fit(X,y)
        self.seen=X.to_numpy().copy()
    def predict(self,X,spec):return self.model.predict(X)
class Stack:
    def fitted_components(self):return {'base':self.base,'meta':self.meta}
    def fit_stack(self,X,y,context,create_preprocessor):
        self.oof=np.empty(len(X));self.audits=[]
        for a,b in KFold(3,shuffle=True,random_state=43).split(X):
            base=PreparedCandidate({},lambda config:Base(),create_preprocessor)
            base.fit(X.iloc[a],y.iloc[a],context)
            self.audits.append(base.estimator.seen.copy())
            self.oof[b]=base.predict(X.iloc[b],{})
        self.meta=Ridge().fit(self.oof[:,None],y)
        self.base=PreparedCandidate({},lambda config:Base(),create_preprocessor)
        self.base.fit(X,y,context)
    def predict(self,X,spec):return self.meta.predict(self.base.predict(X,spec)[:,None])
    def save(self,path):
        import cloudpickle
        Path(path,'stack.pkl').write_bytes(cloudpickle.dumps(self))

class ReviewRegressionTests(unittest.TestCase):
    def test_multilabel_binary_loss_and_weights(self):
        y=np.array([[1,1],[0,0],[1,0],[0,1]])
        p=np.full(y.shape,.999)
        ctx=NS(task=NS(primary_metric='log_loss',task_type='multilabel_classification'),classes=[],metric_options={})
        expected=-(y*np.log(p)+(1-y)*np.log1p(-p)).mean(axis=1)
        self.assertAlmostEqual(score_predictions(y,p,ctx),expected.mean())
        self.assertAlmostEqual(score_predictions(y,p,ctx,weights=[1,3,2,1]),np.average(expected,weights=[1,3,2,1]))
        self.assertTrue(np.isfinite(score_predictions(y,np.ones_like(p),ctx)))
    def test_string_labels_roundtrip_and_two_dimensional_guard(self):
        model=NS(predict=lambda X,spec:np.array(['001']*len(X)))
        with tempfile.TemporaryDirectory() as tmp:
            paths=write_predictions(model,pd.DataFrame({'x':[1,2]}),['label'],Path(tmp))
            values=read_table(paths['label']).to_numpy()
            self.assertEqual(values[:,0].tolist(),['001','001'])
            checked_predictions(values,2,'label',classes=['001','A'])
            with self.assertRaisesRegex(ValueError,'unknown class'):
                checked_predictions(np.array([[1],[1]]),2,'label',classes=['001','A'])
    def test_native_and_slow_components_rejected(self):
        context,tools,budget,plan=case()
        plan.skill_ids+=['ensembles','flaml','torch']
        for groups in [[],['g']]:
            context.task.group_columns=groups
            plan.ensemble_components=[{'tool_id':'sklearn'},{'tool_id':'flaml'}]
            self.assertTrue(validate_plan(plan,context,tools,Skills(),budget).valid)
        plan.ensemble_components[1]['tool_id']='torch'
        self.assertTrue(validate_plan(plan,context,tools,Skills(),budget,lightweight=True).valid)
    def test_stack_fold_local_preprocessing_and_persistence(self):
        X=pd.DataFrame({'x':range(30)});y=pd.Series(np.arange(30,dtype=float))
        context=FitContext('regression','rmse','minimize',42,1,2,30)
        config={'ensemble_components':[{},{}],'model_params':{'method':'stack'}}
        models=[]
        _,b=next(KFold(3,shuffle=True,random_state=43).split(X))
        for delta in [0,1000]:
            changed=y.copy();changed.iloc[b]+=delta
            model=PreparedCandidate(config,lambda config:Stack(),lambda config:MeanPreprocessor())
            model.fit(X,changed,context);models.append(model)
        np.testing.assert_array_equal(models[0].estimator.audits[0],models[1].estimator.audits[0])
        with tempfile.TemporaryDirectory() as tmp:
            import cloudpickle
            root=Path(tmp);models[0].save(root)
            moved=root/'estimator'/'stack.pkl'
            restored=cloudpickle.loads(moved.read_bytes())
            np.testing.assert_array_equal(models[0].predict(X,{}),restored.predict(X,{}))
        with self.assertRaisesRegex(ValueError,'fit_stack'):
            PreparedCandidate(config,lambda config:Base(),lambda config:MeanPreprocessor()).fit(X,y,context)
    def test_generated_stack_smoke_reloads_after_relocation(self):
        import inspect
        from mlmom.worker import smoke
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);code=root/'code';code.mkdir()
            (code/'preprocessing.py').write_text('import numpy as np\nimport pandas as pd\n'+inspect.getsource(MeanPreprocessor)+'\nMLMOM_PREPROCESSOR_V1=True\ndef create_preprocessor(config):return MeanPreprocessor()\n')
            source='from pathlib import Path\nimport numpy as np\nfrom sklearn.linear_model import Ridge\nfrom sklearn.model_selection import KFold\nfrom mlmom.sdk import PreparedCandidate\n'
            source+=inspect.getsource(Base)+inspect.getsource(Stack)
            source+='\ndef create_candidate(config):return Stack()\ndef load_candidate(path):\n    import cloudpickle\n    return cloudpickle.loads(Path(path,"stack.pkl").read_bytes())\n'
            (code/'candidate.py').write_text(source)
            from mlmom.tools import verify_source
            self.assertTrue(verify_source(code).valid)
            job=root/'job';job.mkdir()
            result=smoke(dict(code_dir=str(code),task_type='regression',config={'ensemble_components':[{},{}],'model_params':{'method':'stack'}},worker_seconds=30,worker_cpus=1,worker_memory_gb=2),job)
            self.assertEqual(result['status'],'completed')
    def test_preprocessing_sibling_import_is_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'candidate.py').write_text('def create_candidate(config): return object()\ndef load_candidate(path): return object()\n')
            (root/'helper.py').write_text('class Helper: pass\n')
            (root/'preprocessing.py').write_text('from helper import Helper\nMLMOM_PREPROCESSOR_V1=True\ndef create_preprocessor(config):return Helper()\n')
            module=load_module(root)
            self.assertEqual(type(module.create_candidate({}).create_preprocessor({})).__name__,'Helper')
    def test_legacy_resume_preserves_timed_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);task=dataset(root/'input');config=RunConfig(max_rounds=1)
            def close(p):
                p.sandbox.close();p.checkpointer.close();p.store.close();p.trace.close()
            with patch('mlmom.pipeline.installed_tools',only_sklearn):
                close(Pipeline(task,config,run_dir=root/'run',client=OfflineLLM()))
                path=root/'run'/'session.json';saved=json.loads(path.read_text());saved['config'].pop('timeouts_enabled');path.write_text(json.dumps(saved))
                resumed=Pipeline(task,config,run_dir=root/'run',client=OfflineLLM(),resume=True)
                try:self.assertTrue(resumed.config.timeouts_enabled)
                finally:close(resumed)
                self.assertTrue(json.loads(path.read_text())['config']['timeouts_enabled'])
                with self.assertRaisesRegex(ValueError,'same task and configuration'):
                    Pipeline(task,RunConfig(max_rounds=1,timeouts_enabled=False),run_dir=root/'run',client=OfflineLLM(),resume=True)

if __name__=='__main__':unittest.main()
