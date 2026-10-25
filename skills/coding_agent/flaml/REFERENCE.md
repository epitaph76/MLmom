# Estimator-only flaml reference

Runnable native holdout search and frozen refit for regression/binary/multiclass tasks, including supplied training and validation weights.
The SDK owns preprocessing and target/weight/output policies. Save inference state only.

```python
from pathlib import Path
import copy
import cloudpickle
import numpy as np
from flaml import AutoML
from sklearn.preprocessing import LabelEncoder

class Candidate:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.params = dict(config.get('model_params', {}))

    def fit(self, X, y, context):
        self.classes = list(context.classes)
        params = dict(self.params)
        frozen = params.pop('__native_frozen', None)
        automl = AutoML()
        task = 'classification' if 'classification' in context.task_type else 'regression'
        self.encoder = LabelEncoder().fit(context.classes) if task == 'classification' else None
        labels = self.encoder.transform(np.asarray(y)) if self.encoder is not None else y
        weight_args = {'sample_weight':context.sample_weight} if context.sample_weight is not None else {}
        allowance = context.remaining_seconds if context.timeouts_enabled else context.time_limit_seconds
        if not np.isfinite(allowance):
            raise ValueError('Native search requires a finite library budget')
        if frozen:
            from flaml.automl.training_log import training_log_writer
            log = Path(context.work_dir)/'fixed.jsonl'
            with training_log_writer(str(log)) as writer:
                writer.append(0, None, 0., 0., frozen['validation_loss'], frozen['config'], frozen['estimator'], len(X))
            automl.retrain_from_log(str(log), X_train=X, y_train=labels, record_id=0,
                train_full=True, task=task, n_jobs=context.cpus, time_budget=max(.1,allowance), **weight_args)
            self.frozen = frozen
            self.history = {'mode':'fixed_configuration','new_trials':0}
        else:
            scorer = context.validation_scorer
            if context.validation_X is None or scorer is None:
                raise ValueError('Native reference search requires explicit holdout and the exact SDK scorer')
            def primary(X_val, y_val, estimator, labels, X_train, y_train,
                        weight_val=None, weight_train=None, *args, **kwargs):
                kind = scorer.kind
                if self.encoder is not None and kind == 'probability':
                    values = estimator.predict_proba(X_val)
                    values = np.asarray(values)[:, self.encoder.transform(self.classes)]
                elif self.encoder is not None:
                    values = self.encoder.inverse_transform(np.asarray(estimator.predict(X_val),dtype=int))
                else:
                    values = estimator.predict(X_val)
                score = scorer(values, kind)
                return (-score if context.direction == 'maximize' else score), {'primary_metric':score}
            validation_y = self.encoder.transform(np.asarray(context.validation_y)) if self.encoder is not None else context.validation_y
            automl.fit(X_train=X, y_train=labels, X_val=context.validation_X, y_val=validation_y,
                sample_weight_val=context.validation_weight, task=task, metric=primary,
                time_budget=max(.1,allowance), n_jobs=context.cpus, seed=context.seed,
                estimator_list=['lgbm','extra_tree'], max_iter=self.config.get('max_trials',15),
                eval_method='holdout', ensemble=False, retrain_full=False, verbose=0,
                **weight_args, **params)
            self.frozen = {'estimator':automl.best_estimator,'config':automl.best_config,
                           'validation_loss':float(automl.best_loss)}
            self.history = {'mode':'native_holdout','best_configuration':self.frozen}
        self.model = automl.model.estimator
        self.stopping_info = dict(mode='fixed' if frozen else 'primary_metric',
            reason='Frozen native configuration' if frozen else 'SDK original-scale primary objective',
            metric=context.metric)
        return self

    def predict(self, X, prediction_spec):
        if prediction_spec['kind'] == 'probability':
            p = self.model.predict_proba(X)
            encoded = self.encoder.transform(self.classes)
            return p[:, [list(self.model.classes_).index(c) for c in encoded]]
        prediction = self.model.predict(X)
        return self.encoder.inverse_transform(np.asarray(prediction,dtype=int)) if self.encoder is not None else prediction

    def export_model_params(self):
        return dict(self.params, __native_frozen=self.frozen)

    def native_history(self):
        return self.history

    def save(self, artifact_dir):
        (Path(artifact_dir)/'candidate.pkl').write_bytes(cloudpickle.dumps(self))

def create_candidate(config):
    return Candidate(config)

def load_candidate(artifact_dir):
    return cloudpickle.loads((Path(artifact_dir)/'candidate.pkl').read_bytes())
```

The transient custom objective uses the SDK scorer, including target/output policies and metric options. Keep only the final underlying estimator, not AutoML or its callback. max_trials bounds native iterations. Native CV and grouped-native selection are not verified by this reference.
