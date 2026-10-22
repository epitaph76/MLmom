---
name: preprocessing-core
description: Enforce fold-local, persisted and resource-aware preprocessing for every candidate.
---
# Preprocessing core

The orchestrator chooses the split before any learned transform; report split risks instead of changing its rows or folds. Fit imputers, encoders, scalers, selectors and target statistics only on the current training portion. Validation and inference data are transform-only. preprocessing.py must preserve row count and order for all inputs. Any training-only resampling belongs to coding_agent, which must align y, groups and weights; never resample validation or inference rows.

Start with the least complex representation compatible with the estimator. Trees usually do not need scaling; linear, distance and neighbour methods usually do. A plan's preprocessing list records intent but does not implement it: feature transforms live in preprocessing.py; estimator-side sampling, target transforms and thresholds belong to coding_agent. The SDK fits and persists the preprocessor separately.

Exclude target, ID, group, fold and weight columns unless the task explicitly assigns them as features. Convert infinities deliberately, preserve row order, and make output columns deterministic. Fit-time decisions such as dropped columns, category vocabularies, clipping bounds, reference dates, selected features and thresholds are model state. Reuse fitted transform state unchanged after load and at prediction time. Each new fit, including final refit, creates a fresh preprocessor and learns its statistics from that fit's allowed training rows. Preserve the selected recipe and settings; do not carry validation-fit statistics into a new fit. The estimator separately exports selected model parameters and thresholds for fixed refit.

Before accepting a candidate, test missing values, unseen categories, shuffled input columns, non-default indices and a single-row prediction. Check transformed shape, finite-value requirements, sparse/dense memory and save/load prediction equality. Use only APIs and packages pinned by this repository; inspect an installed API instead of assuming a newer documentation example is available.

Load a specialized preprocessing card only when the dataset profile or hypothesis needs it.
