---
name: validation
description: Keep model selection inside supplied training data and form valid group, class and OOF partitions.
---
# Internal validation

The runner supplies an inner split through context.validation_X/y and hides its outer holdout. Reuse the inner validation for early stopping/native search. When absent during fixed refit, use selected parameters and all supplied train rows without new tuning.

Only operations needing OOF predictions should create folds inside X/y. Single-label classification: StratifiedKFold, splits no greater than minimum class count. Groups: GroupKFold or StratifiedGroupKFold, at least two groups; verify class coverage because stratification is approximate. Ranking splits whole queries. Exact duplicates need shared groups; report when the runner's split did not account for them.

Fit transforms, resampling and target statistics within each training fold. Fill OOF values at original row positions exactly once. Slice groups/weights with indices. Verify both sides support the metric; do not silently weaken validation to make a fit succeed.

Multilabel needs per-label coverage checks; single-label stratification cannot handle its matrix. Define constant-label behavior. Temporal data needs chronological splits, but current MLmom rejects time_column/forecasting tasks: report deferred support, never substitute random CV.
