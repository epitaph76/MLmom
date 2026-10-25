# Estimator-only lightgbm reference

Runnable for single-target regression and binary/multiclass classification; numeric matrices or preprocessed DataFrames with stable categorical dtypes.
The SDK owns preprocessing and target/weight/output policies. Save inference state only.

```python
from pathlib import Path
import copy
import cloudpickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor, LGBMClassifier, early_stopping

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
        self.params = dict(n_estimators=300, verbosity=-1, random_state=context.seed) | self.params
        params = dict(self.params, n_jobs=context.cpus, device_type='cpu')
        scorer = context.validation_scorer
        self.use_stopping = (self.config.get('stopping_policy') == 'early_stopping'
                             and context.validation_X is not None and scorer is not None)
        self.stopping_info = dict(mode='primary_metric' if self.use_stopping else 'fixed',
            reason='SDK original-scale primary metric' if self.use_stopping else
                   'Fixed fit: requested fixed policy or no exact holdout scorer', metric=context.metric)
        kwargs = {'sample_weight':context.sample_weight} if context.sample_weight is not None else {}
        if self.use_stopping:
            params['metric'] = 'None'
            labels = self.encoder.transform(np.asarray(context.validation_y)) if self.encoder is not None else context.validation_y
            def primary(truth, prediction):
                return 'primary', scorer(self.output(prediction, scorer.kind), scorer.kind), context.direction == 'maximize'
            kwargs.update(eval_set=[(context.validation_X, labels)], eval_metric=primary,
                          callbacks=[early_stopping(50, first_metric_only=True, verbose=False)])
            if context.validation_weight is not None:
                kwargs['eval_sample_weight'] = [context.validation_weight]
        cls = LGBMClassifier if self.classification else LGBMRegressor
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
        kind = prediction_spec['kind']
        if self.classification:
            return self.output(self.model.predict_proba(X), kind)
        return self.model.predict(X)

    def export_model_params(self):
        params = copy.deepcopy(self.params)
        if self.use_stopping and self.model.best_iteration_ > 0:
            params['n_estimators'] = int(self.model.best_iteration_)
        else:
            params['n_estimators'] = int(self.model.n_estimators_)
        return params

    def save(self, artifact_dir):
        (Path(artifact_dir)/'candidate.pkl').write_bytes(cloudpickle.dumps(self))

def create_candidate(config):
    return Candidate(config)

def load_candidate(artifact_dir):
    return cloudpickle.loads((Path(artifact_dir)/'candidate.pkl').read_bytes())
```

The fit callback is transient; never retain the scorer or FitContext. CV/refit uses fixed counts. For ranking and multioutput, implement the separate group/per-target path described in SKILL.md. Row sampling needs subsample_freq > 0.
