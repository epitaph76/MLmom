"""Executable evidence for fold-local DSL features and scoped, paired evaluation."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import cloudpickle
import numpy as np
import pandas as pd
from scipy import sparse

from mlmom.feature_contracts import FeatureSpec, FeatureSet, feature_hash
from mlmom.feature_engineering import FeatureTransformer, FeatureAugmentingPreprocessor, validate_features
from mlmom.feature_search import paired_evidence, fixed_model_hash, feature_profile, dictionary_metadata
from mlmom.contracts import ValidationStrategy
from mlmom.model_spec import model_spec, with_spec, apply_overrides
from mlmom.sdk import FitContext, load_module
from mlmom.splitting import feature_context, inner_protocol
from mlmom.worker import train, smoke
from mlmom.persistence import tree_hash
from tests.test_execution import execution_case
from tests.fakes import MODEL_CODE, PREPROCESSING_CODE


def definition(*features):
    return FeatureSet(feature_set_id='test', features=[FeatureSpec(**f) for f in features])


class Base:
    def __init__(self, representation):
        self.representation = representation
    def fit_transform(self, X, y, context):
        assert context.validation_y is None
        return self.transform(X)
    def transform(self, X):
        values = X[['a']].to_numpy()
        return sparse.csr_matrix(values) if self.representation == 'sparse' else (
            X[['a']].copy() if self.representation == 'dataframe' else values)


class FeatureDSLTests(unittest.TestCase):
    def test_only_dictionary_meanings_are_reused_and_temporal_validation_is_deferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'profile.json'
            path.write_text(json.dumps(dict(columns=[dict(name='a',description='Annual income',unit='RUB',
                declared_dtype='float',train=dict(mean='SENTINEL_GLOBAL_STATS')),dict(name='target',description='Label')])) )
            metadata=dictionary_metadata(NS(profile_ref=str(path),feature_columns=['a']))
            self.assertEqual(metadata,[dict(name='a',description='Annual income',unit='RUB',declared_dtype='float')])
            self.assertNotIn('SENTINEL_GLOBAL_STATS',json.dumps(metadata))
        with self.assertRaisesRegex(ValueError,'deferred_time_series'):
            feature_context(NS(task=NS(time_column='t',task_type='regression')))

    def test_closed_grammar_and_column_boundary(self):
        for invalid in [dict(name='x',op='python',columns=['a']),
                        dict(name='x',op='ratio',columns=['a']),
                        dict(name='x',op='log',columns=['a'],by='g'),
                        dict(name='x',op='date_part',columns=['a'])]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                FeatureSpec(**invalid)
        for column in ['target','id','fold']:
            with self.assertRaisesRegex(ValueError,'forbidden'):
                validate_features(definition(dict(name='bad',op='log',columns=[column])), ['a'])
        self.assertEqual(len(definition(dict(name='ab',op='product',columns=['a','b']),dict(name='ba',op='product',columns=['b','a'])).features), 2)

    def test_safe_arithmetic_dates_and_unseen_groups_use_saved_state(self):
        X=pd.DataFrame(dict(a=[1.,4.,-1.,np.inf],b=[1.,0.,2.,1.],g=['a','a','b',None],
            date=['2026-01-01','2026-02-01','bad',None]),index=[9,9,1,7])
        spec=definition(dict(name='ln',op='log',columns=['a']),dict(name='ratio',op='ratio',columns=['a','b']),
            dict(name='missing',op='missing',columns=['date']),dict(name='month',op='date_part',columns=['date'],part='month'),
            dict(name='freq',op='frequency',columns=['g']),dict(name='count',op='group_count',by='g'),
            *[dict(name='g_'+stat,op='group_'+stat,columns=['a'],by='g') for stat in ['mean','std','min','max']])
        engine=FeatureTransformer(spec)
        result=engine.fit_transform(X)
        self.assertTrue(np.isfinite(result).all())
        self.assertAlmostEqual(result[1,1],.25)  # invalid division is filled from finite training ratios
        incoming=pd.DataFrame(dict(a=[100.,200.],b=[2.,4.],g=['new','a'],date=['2027-03-01',None]))
        before=cloudpickle.dumps(engine)
        together=engine.transform(incoming)
        separately=np.vstack([engine.transform(incoming.iloc[[i]]) for i in range(2)])
        np.testing.assert_allclose(together,separately)
        self.assertEqual(together[0,4],0.)
        self.assertEqual(together[1,4],.5)
        self.assertEqual(together[0,5],0.)
        self.assertEqual(together[1,6],2.5)
        self.assertEqual(before,cloudpickle.dumps(engine))

    def test_representation_order_and_pickle(self):
        X=pd.DataFrame(dict(a=[1.,2.,3.],b=[4.,5.,6.]),index=[8,8,2])
        spec=definition(dict(name='ratio',op='ratio',columns=['a','b']))
        context=FitContext('regression','rmse','minimize',42,1,2,20)
        for representation in ['dense','sparse','dataframe']:
            with self.subTest(representation=representation):
                pre=FeatureAugmentingPreprocessor(Base(representation),spec)
                values=pre.fit_transform(X,pd.Series([1.,2.,3.]),context)
                self.assertEqual(values.shape,(3,2))
                if representation=='sparse':self.assertTrue(sparse.isspmatrix_csr(values))
                elif representation=='dataframe':
                    self.assertEqual(list(values.columns),['a','__fe_ratio'])
                    self.assertEqual(values.index.tolist(),[8,8,2])
                restored=cloudpickle.loads(cloudpickle.dumps(pre))
                after=restored.transform(X[['b','a']])
                np.testing.assert_allclose(values.toarray() if sparse.issparse(values) else values,
                    after.toarray() if sparse.issparse(after) else after)

    def test_constant_and_redundant_features_are_executable(self):
        X=pd.DataFrame(dict(a=[1.,2.,3.,4.],b=[2.,4.,6.,8.],g=['x']*4))
        for spec in [definition(dict(name='zero',op='difference',columns=['a','a'])),
                     definition(dict(name='freq',op='frequency',columns=['g']))]:
            values=FeatureTransformer(spec).fit_transform(X)
            self.assertEqual(values.shape, (4,1))

    def test_feature_hash_is_in_model_identity_and_frozen_spec(self):
        a=definition(dict(name='ab',op='product',columns=['a','b']))
        b=definition(dict(name='ab',op='ratio',columns=['a','b']))
        config=dict(tool_id='sklearn',feature_set=a.model_dump(),model_params={'n_estimators':4})
        spec=model_spec(config)
        self.assertEqual(spec['feature_set_hash'],feature_hash(a))
        self.assertEqual(apply_overrides(config,{'n_estimators':8})['feature_set'],config['feature_set'])
        updated=with_spec(dict(config,feature_set=b.model_dump()),spec)
        self.assertEqual(updated['feature_set'],b.model_dump())


class FeatureWorkerTests(unittest.TestCase):
    def request(self, root):
        frame,context,plan,code=execution_case(root,'regression')
        # An exactly learnable ratio gives a reliable feature-only experiment.
        rng=np.random.default_rng(5)
        frame['x0']=rng.uniform(1,9,len(frame));frame['x1']=rng.uniform(1,9,len(frame))
        frame['t0']=frame['x0']/frame['x1']
        context.feature_columns=['x0','x1']
        frame.to_csv(root/'train.csv',index=False)
        (code/'candidate.py').write_text(LINEAR_CODE,encoding='utf-8')
        (code/'preprocessing.py').write_text(PREPROCESSING_CODE,encoding='utf-8')
        plan.model_params={};plan.max_trials=1;plan.stopping_policy='fixed'
        context.task.validation_strategy=ValidationStrategy.HOLDOUT
        source=list(range(100,100+len(frame)))
        protocol=inner_protocol(frame,feature_context(context),seed=43,outer_split_id='r1',source_indices=source)
        request=dict(context=context.model_dump(mode='json'),plan=plan.model_dump(mode='json'),
            code_dir=str(code),code_hash=tree_hash(code),train_path=str(root/'train.csv'),seed=42,
            worker_seconds=60,worker_cpus=1,worker_memory_gb=2,timeouts_enabled=False,training_control=False,
            outer_train_indices=source,outer_split_id='r1',validation_scope='feature',feature_protocol=protocol)
        return frame,context,plan,code,protocol,request

    def test_useful_feature_scoped_cv_portability_and_frozen_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);frame,context,plan,code,protocol,request=self.request(root)
            ref_dir=root/'reference';ref_dir.mkdir()
            reference=train(request,ref_dir)
            self.assertEqual(context.task.validation_strategy,'holdout')
            spec=definition(dict(name='ratio',op='ratio',columns=['x0','x1']))
            request['plan'].update(feature_set_id=spec.feature_set_id,feature_set=spec.model_dump())
            cand_dir=root/'candidate';cand_dir.mkdir()
            result=train(request,cand_dir)
            status,mean,deltas=paired_evidence(result,reference,protocol,fixed_model_hash(reference['model_spec']),'minimize')
            self.assertEqual(status,'accepted')
            self.assertGreater(mean,0)
            self.assertTrue(all(d>0 for d in deltas))
            self.assertEqual(result['score_kind'],'feature_inner_cv_mean')
            self.assertEqual(result['oof_score_kind'],'feature_inner_oof')
            loaded=load_module(code).load_candidate(result['model_artifact'])
            np.testing.assert_allclose(loaded.predict(frame[context.feature_columns],{'kind':'value'}),frame.t0,atol=1e-10)
            self.assertIsNone(result.get('execution_report'))
            altered=copy.deepcopy(result)
            altered['model_spec']['backend_params']={'fit_intercept':False}
            self.assertEqual(paired_evidence(altered,reference,protocol,'unused','minimize')[0], 'accepted')

    def test_group_folds_and_protocol_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);frame,context,plan,code,_,request=self.request(root)
            frame['group']=np.repeat(np.arange(12),5)
            context.task.group_columns=['group'];context.task.excluded_feature_columns+=['group']
            frame.to_csv(request['train_path'],index=False)
            protocol=inner_protocol(frame,feature_context(context),seed=43,outer_split_id='r1',
                source_indices=request['outer_train_indices'])
            for fold in protocol['folds']:
                self.assertFalse(set(frame.iloc[fold['train_indices']]['group']) & set(frame.iloc[fold['validation_indices']]['group']))
            request.update(context=context.model_dump(),feature_protocol=protocol)
            work=root/'ok';work.mkdir();self.assertEqual(train(request,work)['status'],'completed')
            request['feature_protocol']=copy.deepcopy(protocol)
            request['feature_protocol']['folds'][0]['validation_indices'].reverse()
            work=root/'bad';work.mkdir()
            self.assertEqual(train(request,work)['status'],'completed')

    def test_profile_ignores_outer_values_and_retains_train_only_residuals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);frame,context,plan,code,_,request=self.request(root)
            work=root/'ref';work.mkdir();result=train(request,work)
            report=feature_profile(dict(request,oof_ref=result['oof_ref']))['profile']
            self.assertEqual(report['rows'],len(frame))
            self.assertEqual({c['name'] for c in report['columns']},set(context.feature_columns))
            self.assertTrue(report['residual_slices'])
            frame['flag']=np.arange(len(frame))%2==0
            context.feature_columns+=['flag']
            frame.to_csv(request['train_path'],index=False)
            request['context']=context.model_dump(mode='json')
            oof=json.loads(Path(result['oof_ref']).read_text())
            oof['predictions']=[[p] for p in oof['predictions']]
            Path(result['oof_ref']).write_text(json.dumps(oof))
            report=feature_profile(dict(request,oof_ref=result['oof_ref']))['profile']
            self.assertIn('flag',{c['name'] for c in report['columns']})
            self.assertTrue(all(np.isfinite(row['mean_error']) for row in report['residual_slices']))


LINEAR_CODE='''from pathlib import Path
import cloudpickle
from sklearn.linear_model import LinearRegression
class Candidate:
    def __init__(self,config):self.params=dict(config.get('model_params',{}))
    def fit(self,X,y,context):
        self.model=LinearRegression(**self.params)
        self.model.fit(X,y,sample_weight=context.sample_weight)
        return self
    def predict(self,X,spec):return self.model.predict(X)
    def export_model_params(self):return self.params
    def save(self,path):Path(path,'model.pkl').write_bytes(cloudpickle.dumps(self))
def create_candidate(config):return Candidate(config)
def load_candidate(path):return cloudpickle.loads(Path(path,'model.pkl').read_bytes())
'''


if __name__=='__main__':unittest.main()
