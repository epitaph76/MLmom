"""CPU adapters usable by generated candidates. No pretrained weights/downloads."""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from .evaluation import metric_name
from .sdk import FitContext
from .registry import LIGHTAUTOML_OBJECTIVES


def classification(kind):
    return kind in {'binary_classification', 'multiclass_classification', 'multilabel_classification'}


def preprocessor(X):
    numeric = X.select_dtypes(include=['number', 'bool']).columns.tolist()
    category = [c for c in X.columns if c not in numeric]
    return ColumnTransformer([
        ('number', Pipeline([('fill', SimpleImputer(strategy='median', keep_empty_features=True)),
                             ('scale', StandardScaler())]), numeric),
        ('category', Pipeline([('fill', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
                               ('encode', OneHotEncoder(handle_unknown='ignore', min_frequency=2, max_categories=100, sparse_output=False))]), category),
    ], remainder='drop')


class LibraryCandidate:
    def __init__(self, config):
        self.config = config
        self.tool = config['tool_id']
        self.params = dict(config.get('model_params', {}))
        self.preprocessor = None
        self.encoder = None
        self.context = None

    def _clean(self, X):
        X = X.copy().replace([np.inf, -np.inf], np.nan)
        for column in X:
            if not pd.api.types.is_numeric_dtype(X[column]):
                X[column] = X[column].map(lambda v: str(v) if not pd.isna(v) else np.nan)
        return X

    def fit(self, X, y, context: FitContext):
        self.context = context
        self.columns = X.columns.tolist()
        self.classes = list(context.classes)
        X = self._clean(X)
        kind = context.task_type
        p = dict(self.params)
        if self.tool in {'autogluon', 'lightautoml', 'flaml'}:
            return self._fit_automl(X, y, context, p)
        if self.tool == 'catboost':
            from catboost import CatBoostClassifier, CatBoostRegressor, CatBoostRanker
            self.category = [c for c in X if not pd.api.types.is_numeric_dtype(X[c])]
            for c in self.category: X[c] = X[c].fillna('__missing__').astype(str)
            cls = CatBoostClassifier if classification(kind) else CatBoostRanker if kind == 'ranking' else CatBoostRegressor
            defaults = dict(iterations=200, verbose=False, random_seed=context.seed, thread_count=context.cpus,
                            allow_writing_files=False)
            if kind == 'multioutput_regression': defaults['loss_function'] = 'MultiRMSE'
            if kind == 'multilabel_classification': defaults['loss_function'] = 'MultiLogloss'
            self.model = cls(**(defaults | p))
            kwargs = {'cat_features': self.category}
            if kind == 'ranking':
                if context.groups is None: raise ValueError('Ranking requires query groups')
                order = np.argsort(context.groups, kind='stable')
                kwargs['group_id'] = np.asarray(context.groups)[order]
                X, y = X.iloc[order], np.asarray(y)[order]
            if context.sample_weight is not None:
                kwargs['sample_weight'] = np.asarray(context.sample_weight)[order] if kind=='ranking' else context.sample_weight
            class TimeLimit:
                def after_iteration(self,info):return context.remaining_seconds>1.5
            kwargs['callbacks']=[TimeLimit()]
            self.model.fit(X, y, **kwargs)
            return self
        self.preprocessor = preprocessor(X)
        values = self.preprocessor.fit_transform(X)
        self.transformed_columns = values.shape[1]
        fit_params = {}
        if context.sample_weight is not None: fit_params['sample_weight'] = context.sample_weight
        if self.tool == 'sklearn':
            if kind == 'clustering':
                from sklearn.cluster import KMeans
                self.model = KMeans(**(dict(n_clusters=3, random_state=context.seed, n_init=3) | p))
            elif kind == 'anomaly_detection':
                from sklearn.ensemble import IsolationForest
                self.model = IsolationForest(**(dict(n_estimators=100, random_state=context.seed, n_jobs=context.cpus) | p))
            elif classification(kind):
                from sklearn.ensemble import ExtraTreesClassifier
                self.model = ExtraTreesClassifier(**(dict(n_estimators=100, random_state=context.seed, n_jobs=context.cpus) | p))
            else:
                from sklearn.ensemble import ExtraTreesRegressor
                self.model = ExtraTreesRegressor(**(dict(n_estimators=100, random_state=context.seed, n_jobs=context.cpus) | p))
        elif self.tool == 'lightgbm':
            from lightgbm import LGBMClassifier, LGBMRegressor, LGBMRanker
            cls = LGBMClassifier if classification(kind) else LGBMRanker if kind == 'ranking' else LGBMRegressor
            self.model = cls(**(dict(n_estimators=200, random_state=context.seed, n_jobs=context.cpus, verbosity=-1) | p))
            if kind == 'ranking':
                order = np.argsort(context.groups, kind='stable')
                values, y = values[order], np.asarray(y)[order]
                fit_params['group'] = np.unique(np.asarray(context.groups)[order], return_counts=True)[1].tolist()
                if 'sample_weight' in fit_params: fit_params['sample_weight']=np.asarray(fit_params['sample_weight'])[order]
        elif self.tool == 'xgboost':
            from xgboost import XGBClassifier, XGBRegressor, XGBRanker
            cls = XGBClassifier if classification(kind) else XGBRanker if kind == 'ranking' else XGBRegressor
            self.model = cls(**(dict(n_estimators=200, random_state=context.seed, n_jobs=context.cpus, tree_method='hist', device='cpu') | p))
            if kind == 'ranking':
                order = np.argsort(context.groups, kind='stable')
                values, y = values[order], np.asarray(y)[order]
                fit_params['qid'] = pd.factorize(np.asarray(context.groups)[order], sort=True)[0]
                if 'sample_weight' in fit_params: fit_params['sample_weight']=np.asarray(fit_params['sample_weight'])[order]
            if classification(kind) and np.asarray(y).ndim == 1:
                self.encoder = LabelEncoder().fit(y)
                y = self.encoder.transform(y)
        elif self.tool == 'statsmodels':
            import statsmodels.api as sm
            self.model = sm.OLS(np.asarray(y, dtype=float), sm.add_constant(values, has_constant='add')).fit(**p)
            return self
        elif self.tool == 'torch':
            return self._fit_torch(values, y, context, p)
        else:
            raise ValueError(f'Unknown adapter: {self.tool}')
        if kind == 'multioutput_regression' and self.tool in {'lightgbm', 'xgboost'}:
            self.model = MultiOutputRegressor(self.model, n_jobs=1)
        if kind == 'multilabel_classification' and self.tool in {'lightgbm', 'xgboost'}:
            self.model = MultiOutputClassifier(self.model, n_jobs=1)
        unsupervised = kind in {'clustering', 'anomaly_detection'}
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=context.cpus):
            self.model.fit(values, None if unsupervised else y, **fit_params)
        return self

    def _fit_automl(self, X, y, context, p):
        frozen=p.pop('__native_frozen',None)
        if self.tool == 'flaml':
            from flaml import AutoML
            self.model = AutoML()
            metric = {'rmse':'rmse', 'mae':'mae', 'f1_macro':'macro_f1', 'roc_auc':'roc_auc', 'accuracy':'accuracy', 'log_loss':'log_loss'}.get(metric_name(context.metric))
            if metric is None: raise ValueError('FLAML adapter does not support this metric')
            settings=dict(estimator_list=['lgbm','rf','extra_tree'],max_iter=self.config.get('max_trials',20),
                          log_file_name=str(Path(context.work_dir)/'flaml-trials.jsonl'),log_type='all')
            Path(context.work_dir).mkdir(parents=True,exist_ok=True)
            if frozen:
                # retrain_from_log reproduces a selected configuration on all rows,
                # avoiding another search (and FLAML's one-trial RF budget bug).
                from flaml.automl.training_log import training_log_writer
                log=Path(context.work_dir)/'fixed-configuration.jsonl'
                with training_log_writer(str(log)) as writer:
                    writer.append(0,None,0.,0.,frozen['validation_loss'],frozen['config'],frozen['estimator'],len(X))
                self.model.retrain_from_log(str(log),X_train=X,y_train=y,record_id=0,train_full=True,
                    task='classification' if classification(context.task_type) else 'regression',
                    n_jobs=context.cpus,time_budget=max(1.,context.remaining_seconds))
                return self
            self.model.fit(X_train=X, y_train=y, task='classification' if classification(context.task_type) else 'regression',
                metric=metric, time_budget=max(1.,context.remaining_seconds), n_jobs=context.cpus,
                seed=context.seed, verbose=0, **settings,
                X_val=self._clean(context.validation_X) if context.validation_X is not None else None,
                y_val=context.validation_y, **p)
        elif self.tool == 'autogluon':
            from autogluon.tabular import TabularPredictor
            label = '__mlmom_target__'
            if label in X: raise ValueError('Reserved target column exists')
            training = X.assign(**{label: np.asarray(y)})
            validation = self._clean(context.validation_X).assign(**{label: np.asarray(context.validation_y)}) if context.validation_X is not None else None
            metric = {'rmse':'root_mean_squared_error','mae':'mean_absolute_error', 'f1_macro':'f1_macro',
                      'accuracy':'accuracy','roc_auc':'roc_auc','log_loss':'log_loss'}.get(metric_name(context.metric))
            if metric is None: raise ValueError('AutoGluon adapter does not support this metric')
            self.model = TabularPredictor(label=label, path=str(Path(context.work_dir)/'autogluon'), eval_metric=metric, verbosity=0)
            # Explicit families exclude foundation models and neural pretrained backbones.
            hyperparameters={frozen['family']:frozen['params']} if frozen else {'GBM':{}, 'CAT':{}, 'XGB':{}, 'RF':{}, 'XT':{}}
            self.model.fit(training, tuning_data=validation, time_limit=max(1.,context.remaining_seconds*(.5 if frozen else 1.)),
                hyperparameters=hyperparameters,fit_weighted_ensemble=False,
                num_cpus=context.cpus, num_gpus=0, **p)
            if frozen:
                self.model.refit_full(model='best',set_best_to_refit_full=True,num_cpus=context.cpus,num_gpus=0,fit_strategy='sequential')
        else:
            metric = metric_name(context.metric)
            if metric != LIGHTAUTOML_OBJECTIVES.get(context.task_type):
                raise ValueError('LightAutoML native objective does not match the requested task metric')
            from lightautoml.automl.presets.tabular_presets import TabularAutoML
            from lightautoml.tasks import Task
            # The preset's default task metric is allowed only when it matches the objective.
            metric = metric_name(context.metric)
            task_name = {'regression':'reg','binary_classification':'binary','multiclass_classification':'multiclass'}[context.task_type]
            if classification(context.task_type):
                self.encoder = LabelEncoder().fit(y)
                y = self.encoder.transform(y)
            self.model = TabularAutoML(task=Task(task_name), timeout=max(1,int(context.remaining_seconds)),gpu_ids='',
                cpu_limit=context.cpus, memory_limit=max(1,int(context.memory_gb)),
                general_params={'use_algos':[['lgb','cb']]}, reader_params={'cv':3,'random_state':context.seed}, **p)
            training = X.assign(__mlmom_target__=np.asarray(y))
            validation = None
            if context.validation_X is not None:
                vy = self.encoder.transform(context.validation_y) if self.encoder else context.validation_y
                validation = self._clean(context.validation_X).assign(__mlmom_target__=np.asarray(vy))
            elif frozen:
                # Fixed untuned preset refit: fit every base learner on all allowed
                # rows. In-sample native reports are never used as evaluation scores.
                validation=training
            self.model.fit_predict(training, roles={'target':'__mlmom_target__'}, valid_data=validation, verbose=0)
        return self

    def export_model_params(self):
        params=dict(self.params)
        if self.tool=='catboost':params['iterations']=int(self.model.tree_count_)
        if self.tool=='flaml':
            if '__native_frozen' not in params:
                params['__native_frozen']={'estimator':self.model.best_estimator,'config':self.model.best_config,
                                         'validation_loss':float(self.model.best_loss)}
        elif self.tool=='autogluon':
            info=self.model.model_info(self.model.model_best)
            families={'LGBModel':'GBM','LightGBMModel':'GBM','CatBoostModel':'CAT','XGBoostModel':'XGB','RFModel':'RF','XTModel':'XT'}
            family=families.get(str(info['model_type']))
            if family is None:raise ValueError('Cannot freeze this AutoGluon model family')
            params['__native_frozen']={'family':family,'params':info['hyperparameters']}
        elif self.tool=='lightautoml':
            # This adapter uses only untuned lgb/cb presets; no *_tuned algorithms.
            params['__native_frozen']={'fixed_preset':True}
        return params

    def native_history(self):
        if self.tool=='flaml':
            if '__native_frozen' in self.params:
                return {'backend':'flaml','mode':'fixed_configuration','configuration':self.params['__native_frozen'],
                        'validation_loss_is_historical':True,'new_trials':0}
            path=Path(self.context.work_dir)/'flaml-trials.jsonl'
            return {'backend':'flaml','trial_log':path.read_text(encoding='utf-8') if path.exists() else '',
                    'best_estimator':self.model.best_estimator,'best_config':self.model.best_config}
        if self.tool=='autogluon':
            return {'backend':'autogluon','models':self.model.leaderboard(silent=True).to_json(orient='records')}
        if self.tool=='lightautoml':
            return {'backend':'lightautoml','models':self.model.create_model_str_desc(),'trial_callbacks_available':False}
        return None

    def _fit_torch(self, values, y, context, p):
        import torch
        torch.set_num_threads(context.cpus)
        torch.manual_seed(context.seed)
        array = np.asarray(y)
        cls = classification(context.task_type)
        if cls and array.ndim == 1:
            self.encoder = LabelEncoder().fit(array)
            target = torch.tensor(self.encoder.transform(array), dtype=torch.long)
            outputs = len(self.encoder.classes_)
            criterion = torch.nn.CrossEntropyLoss()
        else:
            target = torch.tensor(array, dtype=torch.float32)
            if target.ndim == 1: target = target[:,None]
            outputs = target.shape[1]
            criterion = torch.nn.BCEWithLogitsLoss() if cls else torch.nn.MSELoss()
        hidden = int(p.pop('hidden_size', 64))
        epochs = int(p.pop('epochs', 20))
        lr = float(p.pop('lr', .001))
        batch = int(p.pop('batch_size', 256))
        if p: raise ValueError(f'Unknown torch parameters: {list(p)}')
        self.torch_shape = (values.shape[1],hidden,outputs)
        self.model = torch.nn.Sequential(torch.nn.Linear(values.shape[1],hidden), torch.nn.ReLU(),torch.nn.Linear(hidden,outputs))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        tensor = torch.tensor(values, dtype=torch.float32)
        for _ in range(epochs):
            if context.remaining_seconds < .2: break
            for ids in torch.randperm(len(tensor)).split(batch):
                optimizer.zero_grad()
                loss = criterion(self.model(tensor[ids]), target[ids])
                loss.backward()
                optimizer.step()
                if context.remaining_seconds < .2: break
        self.model.eval()
        return self

    def predict(self, X, prediction_spec):
        kind = prediction_spec.get('kind', 'value')
        X = self._clean(X[self.columns])
        if self.tool == 'catboost':
            for c in self.category: X[c] = X[c].fillna('__missing__').astype(str)
        if self.preprocessor is not None: X = self.preprocessor.transform(X)
        if self.tool == 'torch':
            import torch
            with torch.no_grad(): values = self.model(torch.tensor(X, dtype=torch.float32)).numpy()
            if classification(self.context.task_type):
                if self.encoder:
                    values = np.exp(values-values.max(axis=1,keepdims=True)); values /= values.sum(axis=1,keepdims=True)
                    return values if kind == 'probability' else self.encoder.inverse_transform(values.argmax(axis=1))
                values = 1/(1+np.exp(-values))
                return values if kind == 'probability' else (values >= .5).astype(int)
            return values.ravel() if values.shape[1] == 1 else values
        if self.tool == 'statsmodels':
            import statsmodels.api as sm
            return self.model.predict(sm.add_constant(X, has_constant='add'))
        if self.tool == 'lightautoml':
            values = self.model.predict(X).data
            if classification(self.context.task_type):
                if values.shape[1] == 1: values = np.column_stack([1-values[:,0],values[:,0]])
                return values if kind == 'probability' else self.encoder.inverse_transform(values.argmax(axis=1))
            return values.ravel()
        if kind == 'anomaly_score':
            return -np.asarray(self.model.score_samples(X))
        if self.context.task_type == 'anomaly_detection' and kind == 'label':
            return (np.asarray(self.model.predict(X)) == -1).astype(int)
        if kind == 'probability':
            values = self.model.predict_proba(X)
            if isinstance(values, list):
                return np.column_stack([v[:,1] if v.shape[1] == 2 else np.zeros(len(v)) for v in values])
            if isinstance(values, pd.DataFrame):
                return values.loc[:, self.classes].to_numpy()
            actual = getattr(self.model, 'classes_', self.classes)
            if self.encoder: actual = self.encoder.inverse_transform(actual)
            values = np.asarray(values)
            if self.classes and values.ndim == 2:
                order = [list(actual).index(c) for c in self.classes]
                values = values[:,order]
            return values
        result = np.asarray(self.model.predict(X))
        if self.encoder: result = self.encoder.inverse_transform(result.astype(int).ravel())
        return result.ravel() if result.ndim == 2 and result.shape[1] == 1 else result

    def save(self, artifact_dir):
        import cloudpickle
        path = Path(artifact_dir)
        path.mkdir(parents=True, exist_ok=True)
        # FitContext may retain fold data; do not persist those records with the model.
        context = self.context
        original_model=self.model
        from dataclasses import replace
        self.context = replace(context, groups=None, sample_weight=None, validation_X=None,
                               validation_y=None, validation_groups=None, task_context=None)
        try:
            if self.tool=='autogluon':
                # Predictor pickles refer to external model files: bundle them in
                # the artifact, then rebase on load so checkpoints can be moved.
                self.model.clone(str(path/'autogluon-model'))
                self.model=None
            (path/'adapter.pkl').write_bytes(cloudpickle.dumps(self))
        finally:
            self.context = context
            self.model=original_model


def load_adapter(artifact_dir):
    import cloudpickle
    candidate=cloudpickle.loads((Path(artifact_dir)/'adapter.pkl').read_bytes())
    if candidate.tool=='autogluon':
        from autogluon.tabular import TabularPredictor
        candidate.model=TabularPredictor.load(str(Path(artifact_dir)/'autogluon-model'),verbosity=0)
    return candidate
