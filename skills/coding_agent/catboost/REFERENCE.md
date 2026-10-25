# Estimator-only catboost reference

Runnable for single-target regression and binary/multiclass classification. Preprocessing supplies native string categories and numeric NaN; the estimator does not clean X.
The SDK owns preprocessing and target/weight/output policies. Save inference state only.

```python
from pathlib import Path
import copy
import cloudpickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor, CatBoostClassifier, Pool

class Candidate:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.params = dict(config.get('model_params', {}))

    def fit(self, X, y, context):
        self.classification = context.task_type in {'binary_classification','multiclass_classification'}
        if context.task_type not in {'regression','binary_classification','multiclass_classification'}:
            raise ValueError('Adapt this single-target reference to the assigned task')
        self.classes = list(context.classes)
        self.encoder = LabelEncoder().fit(context.classes) if self.classification else None
        target = self.encoder.transform(np.asarray(y)) if self.encoder is not None else y
        categories = [c for c in X if not pd.api.types.is_numeric_dtype(X[c])] if isinstance(X,pd.DataFrame) else []
        self.params = dict(iterations=300, verbose=False, random_seed=context.seed) | self.params
        params = dict(self.params, thread_count=context.cpus, task_type='CPU', allow_writing_files=False)
        metric = {'rmse':'RMSE','mae':'MAE','accuracy':'Accuracy',
                  'log_loss':'MultiClass' if len(self.classes)>2 else 'Logloss',
                  'macro_f1':'TotalF1:average=Macro','f1_macro':'TotalF1:average=Macro'}.get(context.metric)
        identity = (self.config.get('target_transform',{}).get('name','identity') == 'identity'
                    and self.config.get('target_transform',{}).get('clip_min') is None
                    and self.config.get('target_transform',{}).get('nonfinite','error') == 'error'
                    and self.config.get('prediction_policy',{}).get('clip_min') is None
                    and self.config.get('prediction_policy',{}).get('nonfinite','error') == 'error')
        options = context.task_context.metric_options if context.task_context is not None else {}
        self.use_stopping = (self.config.get('stopping_policy') == 'early_stopping'
            and context.validation_X is not None and context.validation_scorer is not None
            and identity and metric is not None and not options)
        self.stopping_info = dict(mode='primary_metric' if self.use_stopping else 'fixed',
            reason='Exact native metric with identity policies' if self.use_stopping else
                   'Fixed fit: exact native callback unavailable, transformed/output policy, or fixed request',
            metric=context.metric)
        kwargs = {}
        if context.timeouts_enabled:
            class TimeLimit:
                def after_iteration(self, info):
                    return context.remaining_seconds > 0
            kwargs['callbacks'] = [TimeLimit()]
        if self.use_stopping:
            suffix = ';' if ':' in metric else ':'
            params['eval_metric'] = metric+suffix+'use_weights='+('true' if context.validation_weight is not None else 'false')
            labels = self.encoder.transform(np.asarray(context.validation_y)) if self.encoder is not None else context.validation_y
            kwargs.update(eval_set=Pool(context.validation_X, label=labels, cat_features=categories,
                                        weight=context.validation_weight), early_stopping_rounds=50, use_best_model=True)
        else:
            params['use_best_model'] = False
        cls = CatBoostClassifier if self.classification else CatBoostRegressor
        self.model = cls(**params)
        self.model.fit(Pool(X, label=target, cat_features=categories, weight=context.sample_weight), **kwargs)
        return self

    def predict(self, X, prediction_spec):
        if self.classification and prediction_spec['kind'] == 'probability':
            return self.model.predict_proba(X)[:, self.encoder.transform(self.classes)]
        values = np.asarray(self.model.predict(X)).ravel()
        return self.encoder.inverse_transform(values.astype(int)) if self.classification else values

    def export_model_params(self):
        params = copy.deepcopy(self.params)
        params['iterations'] = int(self.model.tree_count_)
        return params

    def save(self, artifact_dir):
        self.model.save_model(str(Path(artifact_dir)/'model.cbm'))
        state = {k:v for k,v in vars(self).items() if k != 'model'}
        (Path(artifact_dir)/'state.pkl').write_bytes(cloudpickle.dumps(state))

def create_candidate(config):
    return Candidate(config)

def load_candidate(artifact_dir):
    candidate = Candidate({})
    vars(candidate).update(cloudpickle.loads((Path(artifact_dir)/'state.pkl').read_bytes()))
    cls = CatBoostClassifier if candidate.classification else CatBoostRegressor
    candidate.model = cls()
    candidate.model.load_model(str(Path(artifact_dir)/'model.cbm'))
    return candidate
```

For target/output policies or unsupported metric options this reference deliberately fits fixed iterations and records the reason. A custom exact callback may extend it. CatBoost time callbacks return True to continue. Multioutput, multilabel and ranking require the task-specific path in SKILL.md.
