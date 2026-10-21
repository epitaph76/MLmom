# Direct sklearn baseline example

Adapt the estimator, transforms and parameter contract to the hypothesis. For text use sparse TF-IDF instead of this ordinary-table example. No implicit estimator selector exists: implement any selector you declare.

```python
from pathlib import Path
import cloudpickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, IsolationForest
from sklearn.cluster import KMeans

class Candidate:
    def __init__(self, config):
        self.params = dict(config.get('model_params', {}))

    def clean(self, X):
        return X.loc[:, self.columns].replace([np.inf, -np.inf], np.nan)

    def fit(self, X, y, context):
        self.columns = X.columns.tolist()
        self.kind = context.task_type
        self.classes = list(context.classes)
        X = self.clean(X)
        numeric = X.select_dtypes(include=['number', 'bool']).columns.tolist()
        categorical = [c for c in self.columns if c not in numeric]
        self.preprocessor = ColumnTransformer([
            ('number', SimpleImputer(strategy='median', keep_empty_features=True), numeric),
            ('category', Pipeline([
                ('fill', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
                ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical)
        ])
        values = self.preprocessor.fit_transform(X)
        params = dict(self.params)
        if self.kind == 'clustering':
            self.model = KMeans(**(dict(n_clusters=3, n_init=3, random_state=context.seed) | params))
        else:
            cls = IsolationForest if self.kind == 'anomaly_detection' else ExtraTreesClassifier if 'classification' in self.kind else ExtraTreesRegressor
            self.model = cls(**(dict(n_estimators=100, random_state=context.seed) | params | dict(n_jobs=context.cpus)))
        kwargs = {'sample_weight': context.sample_weight} if context.sample_weight is not None else {}
        self.model.fit(values, None if self.kind in {'clustering', 'anomaly_detection'} else y, **kwargs)
        return self

    def predict(self, X, prediction_spec):
        values = self.preprocessor.transform(self.clean(X))
        kind = prediction_spec.get('kind', 'value')
        if kind == 'anomaly_score':
            return -self.model.score_samples(values)
        if self.kind == 'anomaly_detection' and kind == 'label':
            return (self.model.predict(values) == -1).astype(int)
        if kind == 'probability':
            p = self.model.predict_proba(values)
            if isinstance(p, list):
                return np.column_stack([v[:, list(c).index(1)] if 1 in c else np.zeros(len(v)) for v,c in zip(p,self.model.classes_)])
            return p[:, [list(self.model.classes_).index(c) for c in self.classes]] if self.classes else p
        return self.model.predict(values)

    def export_model_params(self):
        return dict(self.params)

    def save(self, artifact_dir):
        (Path(artifact_dir) / 'candidate.pkl').write_bytes(cloudpickle.dumps(self))

def create_candidate(config):
    return Candidate(config)

def load_candidate(artifact_dir):
    return cloudpickle.loads((Path(artifact_dir) / 'candidate.pkl').read_bytes())
```
