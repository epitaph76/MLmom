# Generated candidate persistence

Implement your own Candidate with fit/predict/save and module factories:
```python
from pathlib import Path
import cloudpickle

def create_candidate(config):
    return Candidate(config)

def load_candidate(artifact_dir):
    return cloudpickle.loads((Path(artifact_dir) / 'candidate.pkl').read_bytes())

# Candidate.save writes cloudpickle.dumps(self) to artifact_dir/candidate.pkl.
```
Store only inference state, not FitContext or X/y. Native predictors referencing external model files must copy files into artifact_dir and rebase on load. Trace every promised operation to code: categorical casts with column names, missing policy, class order and parameter consumption. Declared target inverses, weights and output clipping are executed once by the SDK. Preserve column order and fit transforms on allowed training rows only. No project adapter imports.

Search-group identity is controller state. PlannerPlan has no membership fields;
HypothesisPlan adds a derived read-only execution view. FitContext.groups contains
row-level validation groups and must not be confused with search_group_id.

Duration history is advisory. The scheduler admits actual CPU/RAM leases and uses
explicit deadlines; it does not sum source HPO wall times to refuse an ensemble.
FeatureProposal(action="replace", rationale="Measure the replacement", remove=["old_ratio"], features=[
    {"name": "new_ratio", "op": "ratio", "columns": ["x", "z"]}
]) describes one final-set experiment, without a removal-only fit. FeatureProposal
is a controller contract from mlmom.feature_contracts, not a candidate implementation API.

FeatureSet may include programs=[{name, source, config}]. Source defines
create_features(config); the returned object implements fit_transform(X,y,context)
and transform(X). SDK fits it on raw permitted X/current-fold y and appends its
DataFrame/2D-array/sparse columns to the base representation. It is cloudpickled
with the model; it must preserve rows, output schema and inference state. DSL and
Python programs can coexist; there is no fixed feature-count limit. Supervised
statistics are allowed with training cross-fitting/exclusion and train-only maps.

```python
import pandas as pd
class RatioFeatures:
    def fit_transform(self, X, y, context):
        return self.transform(X)
    def transform(self, X):
        values = X['x'] / X['z'].replace(0, float('nan'))
        return pd.DataFrame({'ratio': values.fillna(0)}, index=X.index)
def create_features(config):
    return RatioFeatures()
```
