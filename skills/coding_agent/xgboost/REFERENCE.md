# Estimator-only xgboost reference

Runnable XGBoost 2.1.4 single-target regression and binary/multiclass classification; native categories are already stable pandas category dtypes from preprocessing.
The SDK owns preprocessing and target/weight/output policies. Save inference state only.

```python
from pathlib import Path
import copy
import cloudpickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor, XGBClassifier
from xgboost.callback import EarlyStopping

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
        self.params = dict(n_estimators=300, tree_method='hist', random_state=context.seed) | self.params
        params = dict(self.params, n_jobs=context.cpus, device='cpu')
        params.pop('early_stopping_rounds',None)
        params.pop('callbacks',None)
        scorer = context.validation_scorer
        self.use_stopping = (self.config.get('stopping_policy') == 'early_stopping'
                             and context.validation_X is not None and scorer is not None)
        self.stopping_info = dict(mode='primary_metric' if self.use_stopping else 'fixed',
            reason='SDK original-scale primary metric' if self.use_stopping else
                   'Fixed fit: requested fixed policy or no exact holdout scorer', metric=context.metric)
        kwargs = {'sample_weight':context.sample_weight} if context.sample_weight is not None else {}
        if self.use_stopping:
            def primary(truth, prediction, sample_weight=None):
                return scorer(self.output(prediction, scorer.kind), scorer.kind)
            params.update(eval_metric=primary, disable_default_eval_metric=1,
                callbacks=[EarlyStopping(rounds=50, metric_name='primary', data_name='validation_0',
                    maximize=context.direction == 'maximize', save_best=True)])
            labels = self.encoder.transform(np.asarray(context.validation_y)) if self.encoder is not None else context.validation_y
            kwargs.update(eval_set=[(context.validation_X, labels)], verbose=False)
            if context.validation_weight is not None:
                kwargs['sample_weight_eval_set'] = [context.validation_weight]
        cls = XGBClassifier if self.classification else XGBRegressor
        self.model = cls(**params)
        self.model.fit(X, target, **kwargs)
        return self

    def output(self, values, kind):
        values = np.asarray(values)
        if not self.classification:
            return values.ravel()
        if kind == 'label':
            codes = (values >= .5).astype(int) if values.ndim == 1 else values.argmax(axis=1)
            return self.encoder.inverse_transform(codes)
        if values.ndim == 1:
            values = np.column_stack([1-values, values])
        return values[:, self.encoder.transform(self.classes)]

    def predict(self, X, prediction_spec):
        if self.classification:
            return self.output(self.model.predict_proba(X), prediction_spec['kind'])
        return self.model.predict(X)

    def export_model_params(self):
        params = copy.deepcopy(self.params)
        if self.use_stopping:
            params['n_estimators'] = int(self.model.best_iteration)+1
        return params

    def save(self, artifact_dir):
        self.model.save_model(str(Path(artifact_dir)/'model.json'))
        state = {k:v for k,v in vars(self).items() if k != 'model'}
        (Path(artifact_dir)/'state.pkl').write_bytes(cloudpickle.dumps(state))

def create_candidate(config):
    return Candidate(config)

def load_candidate(artifact_dir):
    candidate = Candidate({})
    vars(candidate).update(cloudpickle.loads((Path(artifact_dir)/'state.pkl').read_bytes()))
    cls = XGBClassifier if candidate.classification else XGBRegressor
    candidate.model = cls()
    candidate.model.load_model(str(Path(artifact_dir)/'model.json'))
    return candidate
```

Constructor callbacks use the correct maximize/minimize direction. JSON preserves categorical models and inference state without serializing holdout callbacks. Ranking needs sorted queries and query-weight semantics; adapt explicitly rather than forwarding row weights.
