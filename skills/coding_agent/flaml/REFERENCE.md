# Direct FLAML holdout reference

The SDK supplies transformed features. Only holdout is verified; no outer Optuna,
native CV, blending or stacking. Keep preprocessing outside this estimator.
The runner refits the exported frozen configuration before external evaluation.

```python
from pathlib import Path
import cloudpickle
import numpy as np
from flaml import AutoML
from sklearn.preprocessing import LabelEncoder

class Candidate:
    def __init__(self, config):
        self.config = config
        self.params = dict(config.get('model_params', {}))

    def fit(self, X, y, context):
        self.classes = list(context.classes)
        params = dict(self.params)
        frozen = params.pop('__native_frozen', None)
        automl = AutoML()
        task = 'classification' if 'classification' in context.task_type else 'regression'
        self.encoder = LabelEncoder().fit(context.classes) if task == 'classification' else None
        labels = self.encoder.transform(np.asarray(y)) if self.encoder is not None else y
        if frozen:
            from flaml.automl.training_log import training_log_writer
            log = Path(context.work_dir) / 'fixed.jsonl'
            with training_log_writer(str(log)) as writer:
                writer.append(0, None, 0., 0., frozen['validation_loss'], frozen['config'], frozen['estimator'], len(X))
            automl.retrain_from_log(str(log), X_train=X, y_train=labels, record_id=0,
                train_full=True, task=task, n_jobs=context.cpus,
                time_budget=max(.1, (context.remaining_seconds if context.timeouts_enabled else context.time_limit_seconds)))
            self.frozen = frozen
            self.history = {'mode':'fixed_configuration', 'new_trials':0}
        else:
            if context.validation_X is None:
                raise ValueError('Native search requires an explicit internal holdout')
            metric = {'f1_macro':'macro_f1', 'macro_f1':'macro_f1'}.get(context.metric, context.metric)
            validation_y = self.encoder.transform(np.asarray(context.validation_y)) if self.encoder is not None else context.validation_y
            automl.fit(X_train=X, y_train=labels, X_val=context.validation_X, y_val=validation_y,
                task=task, metric=metric, time_budget=max(.1, (context.remaining_seconds if context.timeouts_enabled else context.time_limit_seconds)),
                n_jobs=context.cpus, seed=context.seed, estimator_list=['lgbm','extra_tree'],
                max_iter=self.config.get('max_trials',15), eval_method='holdout',
                ensemble=False, retrain_full=False, verbose=0, **params)
            self.frozen = {'estimator':automl.best_estimator, 'config':automl.best_config,
                           'validation_loss':float(automl.best_loss)}
            self.history = {'mode':'native_holdout', 'best_configuration':self.frozen}
        self.model = automl.model.estimator
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
        (Path(artifact_dir) / 'candidate.pkl').write_bytes(cloudpickle.dumps(self))

def create_candidate(config):
    return Candidate(config)

def load_candidate(artifact_dir):
    return cloudpickle.loads((Path(artifact_dir) / 'candidate.pkl').read_bytes())
```

Adapt metric aliases to the confirmed primary metric; never substitute an objective.
