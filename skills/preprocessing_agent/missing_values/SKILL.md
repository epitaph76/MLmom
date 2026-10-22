---
name: missing-values
description: Select and persist leakage-safe missing-value treatment from simple baselines to model-native handling.
---
# Missing values

Profile missing rates by column and, when available, by target, time or group. Distinguish true absence from parse failure and sentinel values such as empty strings or impossible numbers. Learn every fill value on the current training fold only.

Use median for numeric and most-frequent or an explicit sentinel for categorical data as the dependable baseline. Add a missing indicator when absence may carry signal, but validate the benefit. Preserve columns that are entirely missing in a fold or record a deterministic train-derived drop list so later column positions cannot shift.

Prefer native missing-value handling only when the selected pinned estimator and adapter actually support it. KNN or iterative imputation is a later, costed experiment: it is distance/model sensitive, can be expensive in wide data, and must remain inside CV. It is not a default upgrade over simple imputation.

Normalize positive and negative infinity to missing only when this matches the feature semantics. Never derive an imputation statistic from validation or inference batches. Persist fill values, indicators, retained columns and dtypes. Test all-missing columns, new missingness patterns and save/load equality.
