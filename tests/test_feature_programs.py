"""Executable Python features: label boundaries, real folds and portable state."""
import copy
import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import cloudpickle
import numpy as np
import pandas as pd
from scipy import sparse

from mlmom.feature_contracts import FeatureProgram, FeatureProposal, FeatureSet, FeatureSpec, feature_hash
from mlmom.feature_engineering import FeatureAugmentingPreprocessor
from mlmom.feature_search import proposed_feature_set, run_feature_session
from mlmom.sdk import FitContext, load_module
from mlmom.worker import train
from tests import test_feature_engineering as fixtures
from tests import test_feature_session as sessions


TARGET_PROGRAM = '''import numpy as np
import pandas as pd
class Features:
    def fit_transform(self, X, y, context):
        assert context.validation_X is None and context.validation_y is None
        data = pd.DataFrame({'key': X['entity'].to_numpy(), 'y': np.asarray(y)})
        table = data.groupby('key')['y'].agg(['sum','count'])
        self.means = (table['sum']/table['count']).to_dict()
        self.fallback = float(np.mean(y))
        sums = data['key'].map(table['sum']).to_numpy() - np.asarray(y)
        counts = data['key'].map(table['count']).to_numpy() - 1
        fallback = (float(np.sum(y))-np.asarray(y))/max(1,len(y)-1)
        values = np.divide(sums,counts,out=fallback.copy(),where=counts>0)
        return pd.DataFrame({'target_mean':values},index=X.index)
    def transform(self, X):
        return pd.DataFrame({'target_mean':X['entity'].map(self.means).fillna(self.fallback)},index=X.index)
def create_features(config):
    return Features()
'''

RATIO_PROGRAM = '''import pandas as pd
class Features:
    def fit_transform(self, X, y, context):
        assert context.validation_y is None
        return self.transform(X)
    def transform(self, X):
        return pd.DataFrame({'quotient':X['x0']/X['x1']},index=X.index)
def create_features(config):
    return Features()
'''


class FeatureProgramTests(unittest.TestCase):
    def test_target_program_excludes_own_label_and_never_receives_validation(self):
        class Base:
            def fit_transform(self, X, y, context):
                return self.transform(X)
            def transform(self, X):
                return X[['x']]
        X = pd.DataFrame(dict(x=range(6), entity=['a']*3+['b']*3))
        y = pd.Series([10.,20.,30.,100.,200.,300.])
        definition = FeatureSet(feature_set_id='python', programs=[FeatureProgram(name='history', source=TARGET_PROGRAM)])
        context = FitContext('regression','rmse','minimize',42,1,2,60,
            validation_X=pd.DataFrame({'sentinel':[1]}), validation_y=np.array([99999.]),
            task_context=SimpleNamespace(feature_columns=['x','entity']))
        wrapped = FeatureAugmentingPreprocessor(Base(), definition)
        values = wrapped.fit_transform(X, y, context)
        np.testing.assert_allclose(values['__fe_history_target_mean'],[25,20,15,250,200,150])
        self.assertEqual(values.shape,(6,2))
        loaded = cloudpickle.loads(cloudpickle.dumps(wrapped))
        future = pd.DataFrame(dict(x=[9,10],entity=['a','unknown']))
        np.testing.assert_allclose(loaded.transform(future)['__fe_history_target_mean'],[20,110])
        self.assertEqual(wrapped.manifest()['output_width'],2)

    def test_program_replacement_and_source_identity_without_feature_count_cap(self):
        parent = FeatureSet(feature_set_id='old', programs=[FeatureProgram(name='history', source=TARGET_PROGRAM)])
        proposal = FeatureProposal(action='replace', rationale='Change transformation', remove=['history'],
            programs=[FeatureProgram(name='ratio', source=RATIO_PROGRAM)])
        definition, _ = proposed_feature_set(parent, proposal, ['x0','x1'])
        self.assertEqual([p.name for p in definition.programs],['ratio'])
        self.assertNotEqual(feature_hash(parent),feature_hash(definition))
        changed = definition.model_copy(deep=True)
        changed.programs[0].source += '\n# source revision\n'
        self.assertNotEqual(feature_hash(definition), feature_hash(changed))
        many = FeatureSet(feature_set_id='many', features=[FeatureSpec(name='m'+str(i),op='missing',columns=['x']) for i in range(40)])
        self.assertEqual(len(many.features),40)
        expected=hashlib.sha256(json.dumps([f.model_dump(mode='json') for f in many.features],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.assertEqual(feature_hash(many),expected)  # Existing DSL inference identities stay valid.
        legacy=copy.deepcopy(many)
        del legacy.__dict__['programs']  # State of a FeatureSet pickled before Python programs existed.
        self.assertEqual(feature_hash(legacy),expected)
        from mlmom.feature_engineering import FeatureTransformer
        transformer=FeatureTransformer(legacy)
        transformer.fit_transform(pd.DataFrame({'x':[1.,np.nan]}))
        self.assertEqual(transformer.transform(pd.DataFrame({'x':[2.]})).shape,(1,40))

    def test_sparse_program_outputs_survive_saved_model_inference(self):
        class Base:
            def fit_transform(self, X, y, context):
                return X[['x']]
            def transform(self, X):
                return X[['x']]
        source = '''from scipy import sparse
class Features:
    def fit_transform(self, X, y, context):
        return self.transform(X)
    def transform(self, X):
        return sparse.csr_matrix((X[['x']].to_numpy() ** 2))
def create_features(config):
    return Features()
'''
        definition = FeatureSet(feature_set_id='sparse', programs=[FeatureProgram(name='square', source=source)])
        context = FitContext('regression','rmse','minimize',42,1,2,60,
            task_context=SimpleNamespace(feature_columns=['x']))
        wrapped = FeatureAugmentingPreprocessor(Base(), definition)
        values = wrapped.fit_transform(pd.DataFrame({'x':[2.,3.]}),np.array([1.,2.]),context)
        self.assertTrue(sparse.issparse(values))
        np.testing.assert_allclose(values.toarray(),[[2.,4.],[3.,9.]])
        loaded = cloudpickle.loads(cloudpickle.dumps(wrapped))
        np.testing.assert_allclose(loaded.transform(pd.DataFrame({'x':[4.]})).toarray(),[[4.,16.]])

    def test_real_worker_folds_fit_python_features_and_reload_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            frame,context,plan,code,protocol,request=fixtures.FeatureWorkerTests().request(root)
            ref=root/'control';ref.mkdir()
            reference=train(request,ref)
            definition=FeatureSet(feature_set_id='python-ratio',programs=[FeatureProgram(name='ratio',source=RATIO_PROGRAM)])
            request=copy.deepcopy(request)
            request['plan'].update(feature_set_id=definition.feature_set_id,feature_set=definition.model_dump(mode='json'))
            work=root/'program';work.mkdir()
            result=train(request,work)
            self.assertEqual(result['status'],'completed',result)
            self.assertEqual(len(result['fold_results']),len(protocol['folds']))
            self.assertLess(result['internal_score'],reference['internal_score'])
            model=load_module(code).load_candidate(result['model_artifact'])
            np.testing.assert_allclose(model.predict(frame[context.feature_columns],{'kind':'value'}),frame.t0,atol=1e-10)

    def test_feature_session_registers_program_once_and_reuses_the_measured_result(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as cleanup:
            p,base,split,path=sessions.SessionAccountingTests().harness(Path(tmp),fail=False)
            cleanup.callback(p.store.close)
            p.config.feature_max_configurations=1
            original=p.client.generate
            def generate(role,instruction,payload,schema,tools=None):
                if role=='feature_agent':
                    return schema(action='add',rationale='Measure Python ratio',programs=[dict(name='ratio',source=RATIO_PROGRAM)])
                return original(role,instruction,payload,schema,tools)
            p.client.generate=generate
            old=p.sandbox.run
            def run(request,directory,**limits):
                if request['action']=='feature_profile':return old(request,directory,**limits)
                return train(dict(request,worker_seconds=60,worker_cpus=1,worker_memory_gb=2,timeouts_enabled=False),Path(directory))
            p.sandbox.run=run
            winner=run_feature_session(p,base,split,path)
            self.assertIsNotNone(winner)
            self.assertEqual(winner['plan']['feature_set']['programs'][0]['name'],'ratio')
            self.assertEqual(p.store.get('feature_session','initial')['attempts'],1)
            self.assertEqual(len(p.store.list('feature_evidence')),1)
            self.assertEqual(run_feature_session(p,base,split,path),winner)


if __name__=='__main__':
    unittest.main()
