# Generated candidate persistence

Runtime v8 preserves facts without an additional fit. Example of an observed submission:
```python
from mlmom.execution_facts import FitExecutionFacts

facts = FitExecutionFacts(
    fit_id="submission-example", stage="submission_refit", split_id="all-train",
    status="observed", estimator="Ridge", base_columns=28,
    derived_columns=282, total_columns=310, feature_execution_owner="SDK",
    feature_programs=[{"name": name, "config": {}} for name in (
        "diamond_domain_features", "carat_spline_quality_interactions",
        "rbf_kernel_diamond", "gbm_oof_representation")],
)
assert facts.total_columns == facts.base_columns + facts.derived_columns
```
This illustrates the schema; production values come from the executing SDK, not the coder.
feature_programs=None means unknown; [] means observed absence. Source hashes and the exact
program configs accompany real observations. Candidate fit/predict/save remain unchanged.

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
training-row grouping IDs and must not be confused with search_group_id.

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

Validation callbacks receive context.validation_scorer; its kind is the evaluator's required
representation. Pass raw estimator predictions to scorer(predictions, kind); the SDK handles
target/output policies once. Never serialize that callable or FitContext. In Python untimed
remaining_seconds is infinity; JSON checkpoint budgets use null. TrainingCheckpoint separates
sampled_params from effective_configuration and full exported best_params.
