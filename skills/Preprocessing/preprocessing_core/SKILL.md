---
name: preprocessing-core
description: Enforce fold-local, persisted and resource-aware preprocessing for every candidate.
---
# Preprocessing core

Choose the split before any learned transform. Fit imputers, encoders, scalers, selectors, target statistics and samplers only on the current training portion. Validation and inference data are transform-only. Never resample validation or inference rows.

Start with the least complex representation compatible with the estimator. Trees usually do not need scaling; linear, distance and neighbour methods usually do. A plan's preprocessing list records intent but does not implement it: custom behavior must live in a persisted Pipeline or candidate wrapper.

Exclude target, ID, group, fold and weight columns unless the task explicitly assigns them as features. Convert infinities deliberately, preserve row order, and make output columns deterministic. Fit-time decisions such as dropped columns, category vocabularies, clipping bounds, reference dates, selected features and thresholds are model state. Save them and reuse them unchanged after load and during final refit.

Before accepting a candidate, test missing values, unseen categories, shuffled input columns, non-default indices and a single-row prediction. Check transformed shape, finite-value requirements, sparse/dense memory and save/load prediction equality. Use only APIs and packages pinned by this repository; inspect an installed API instead of assuming a newer documentation example is available.

Load a specialized preprocessing card only when the dataset profile or hypothesis needs it.
