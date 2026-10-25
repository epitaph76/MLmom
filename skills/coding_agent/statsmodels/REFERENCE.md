# Estimator-only statsmodels reference

Runnable statsmodels 0.14.0 OLS for finite, modest-width numeric matrices and single-output regression.
The SDK supplies transformed X and owns target/weight/output policies.

```python
from pathlib import Path
import copy
import cloudpickle
import numpy as np
import statsmodels.api as sm

class Candidate:
    def __init__(self, config):
        self.params = dict(config.get('model_params', {}))

    def fit(self, X, y, context):
        if context.task_type != 'regression':
            raise ValueError('OLS reference requires single-output regression')
        if context.sample_weight is not None:
            raise ValueError('OLS does not support sample weights; implement explicitly planned WLS')
        values = np.asarray(X, dtype=float)
        self.model = sm.OLS(np.asarray(y,dtype=float),
                            sm.add_constant(values,has_constant='add')).fit(**self.params)
        self.stopping_info = dict(mode='fixed',reason='OLS direct fit')
        return self

    def predict(self, X, prediction_spec):
        return np.asarray(self.model.predict(sm.add_constant(np.asarray(X,dtype=float),has_constant='add')))

    def export_model_params(self):
        return dict(self.params)

    def save(self, artifact_dir):
        portable = copy.copy(self)
        portable.model = copy.deepcopy(self.model)
        portable.model.remove_data()
        (Path(artifact_dir)/'candidate.pkl').write_bytes(cloudpickle.dumps(portable))

def create_candidate(config):
    return Candidate(config)

def load_candidate(artifact_dir):
    return cloudpickle.loads((Path(artifact_dir)/'candidate.pkl').read_bytes())
```

model_params are OLS.fit options. The model adds only its intercept; it does not refit an imputer/encoder. Saved results remove training data while retaining prediction coefficients. Weighted tasks need a separately planned WLS implementation.
