# Diagnosing contribution before selecting columns

Selection is a measured intervention, not a threshold on one importance ranking. Keep a
prediction-availability review separate from statistical usefulness. Fit selectors, groups,
thresholds and reductions inside training folds; where diagnostics choose a mask, create a
training-internal validation/nested procedure. Outer evaluation is not selection training.
Preprocessing receives no outer validation scorer and must not fit the predictive estimator;
use an assigned train-only selector or record a diagnostic request for the experiment owner.

## Repeated permutation importance

For a fixed fitted model, evaluate baseline primary metric on permitted internal held-out
rows. Repeatedly permute one feature and recompute the same metric/options/weights; use paired
rows and declared seeds. For maximize, importance is baseline-permuted; for minimize it is
permuted-baseline. Keep repeat values, mean/spread, support and fold/segment identity. Start
with a justified shortlist when expensive and report partial coverage. A training-set ranking
does not establish generalization. A weak/misspecified model can give all features low scores.

An estimate near zero can reflect sampling noise, insufficient power, correlated substitutes,
interactions not used by this model, or a feature helping a rare segment. Negative estimates
can also be noisy. Neither justifies automatic removal. A single-repeat runtime summary is
preliminary evidence; it does not become repeated importance by averaging across columns.
See [scikit-learn permutation importance](https://scikit-learn.org/1.5/modules/permutation_importance.html).

## Joint permutation for correlated feature families

Define groups from domain relationships, training-only correlation structure or the train-fitted
mapping from raw columns to encoded/derived columns. Permute all group columns with the SAME
row permutation to preserve their internal relationships. Compare individual and joint results;
correlated substitutes may each look unimportant while their family is essential. Do not
independently shuffle every member and call it joint permutation. A whole-group shuffle can
still break relationships to other groups and is not conditional/causal importance.
See the [official multicollinearity example](https://scikit-learn.org/1.5/auto_examples/inspection/plot_permutation_importance_multicollinear.html).

Respect the unit of exchangeability. The helper below is an IID-row diagnostic for a bounded
DataFrame and a fixed predictor; it is not preprocessing.py and is not automatically invoked
by the runtime. `score(predictions)` must use the true original-scale primary metric/options/
weights on exactly these held-out rows. Grouped/time/trajectory data need a separately justified
restricted or block permutation; do not pass FitContext.groups as column groups or blindly
shuffle time across entities. The helper returns evidence and never selects/removes columns.

```python
import numpy as np

def repeated_permutation(X, predict, score, feature_groups, *, n_repeats=5,
                         seed=42, direction='minimize'):
    if n_repeats < 2 or direction not in {'minimize','maximize'}:
        raise ValueError('Choose repeated sampling and an explicit metric direction')
    rng = np.random.default_rng(seed)
    baseline = float(score(predict(X.copy())))
    results = {}
    for name, columns in feature_groups.items():
        if not columns or not set(columns) <= set(X.columns):
            raise ValueError('Feature groups must name existing columns')
        values = []
        for _ in range(n_repeats):
            order = rng.permutation(len(X))
            shuffled = X.copy()
            for column in columns:
                shuffled[column] = X[column].iloc[order].set_axis(X.index)
            permuted = float(score(predict(shuffled)))
            values.append(baseline-permuted if direction == 'maximize' else permuted-baseline)
        if not np.isfinite([baseline, *values]).all():
            raise ValueError('Importance scores must be finite')
        results[name] = dict(columns=list(columns), values=values,
            mean=float(np.mean(values)), std=float(np.std(values,ddof=1)), repeats=len(values))
    return dict(baseline=baseline, groups=results, seed=seed, permutation_unit='iid_rows')
```

## Drop-column and family ablation

Retrain the full fold-local pipeline on the SAME internal training/validation rows without
one candidate feature or family. Refit learned transforms on that fold's train; do not merely
zero a column in a previously fitted model and call it drop-column. Preserve model parameters,
policies, metric, weights, seeds and comparable budgets for the controlled comparison. If
retuning after selection is a separate question, tune both alternatives with matched budgets
and identify the extra search. Report baseline/ablated scores and paired fold deltas.

Single-feature drops can look harmless when another feature substitutes for it. Combine that
evidence with group ablation/permutation and the domain meaning before eliminating a whole
family. A jointly required replacement is a final-set comparison; do not require a removal-only
gain before testing it. Dropping a feature can reduce cost or improve robustness without a
large metric change; state that objective and the uncertainty rather than inventing a gain.

## Stability across folds and repeats

Record retained sets and importance distributions per fold/seed, selection frequency for each
eligible feature, overlap/Jaccard of masks, and group-level retention. Align names to the
train-fitted encoding map: a category absent in one fold was not eligible there. Report that
denominator. Equivalent correlated proxies can alternate between folds while the information
family remains stable; low individual stability is not proof of uselessness. Fold samples
are dependent, so repeated values are not automatically independent confidence observations.

Inspect meaningful segments when support permits, including rare classes, new entities and
missingness regimes; mark undefined metrics. Do not keep searching thresholds/segments on the
same holdout and claim independent confirmation. A reproducible noise/constant control can
help interpret the diagnostic scale, but importance is not evidence of causal effect.

## Decision and persistence

State retain/drop/needs-more-evidence with the model/data scope, repeated/grouped importance,
refitted ablation, stability, semantic reason and cost. If a required diagnostic cannot be
run, leave it unmeasured; do not manufacture its result from a one-shot summary. Selection
may be disabled when evidence is inconclusive. Persist selected names/indices, fitted groups
and mappings without fitting on prediction batches; preserve sparse inputs. Missing required
raw columns need an explicit policy, while extra/reordered columns and unseen encoded levels
follow the saved schema. These recommendations add no provenance or mandatory smoke gate.
