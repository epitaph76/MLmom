---
name: feature-engineering
description: Design and implement semantically justified feature transformations with isolated baseline comparisons.
---
# Feature engineering

Use [REFERENCE.md](REFERENCE.md) for six research families: arithmetic invariants, relative
quantities, group deviations, interactions, repeated events and relational aggregation.
For each transformation state input meaning/units, inference availability, mechanism,
domain/fallback and an individual comparison with the unchanged baseline. Do not justify a
large bundle by its combined gain alone. Coherent dependencies and atomic replacements
need feasible ablations, not a mandatory removal-only gain.

Implement only the assigned preprocessing. Extra experiments/comparisons belong to the
Feature/experiment owner; record unmeasured hypotheses in agent_notes. Do not fit the
predictive estimator or duplicate SDK feature_set programs. plan.preprocessing is intent,
not executable code. Prefer a meaningful supported hypothesis over arbitrary polynomial/ID crosses.

Protect denominator/log domains, units, timezones and fixed reference dates. Fit maps,
centres, scales, vocabularies and thresholds inside the current training fold; freeze them
for inference with unknown-group fallbacks. Target-derived training values/priors need
group/time-safe OOF or exclusion. Never fit from validation or prediction batches.

Compare each proposed transformation on the same folds/metric/options/weights, reporting
support and cost. Persist schema/state; check reordered/single-row and save/load behavior.
