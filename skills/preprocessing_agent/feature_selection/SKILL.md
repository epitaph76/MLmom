---
name: feature-selection
description: Reduce constant, redundant or high-dimensional features with selection fitted inside validation folds.
---
# Feature selection

Remove explicit leakage, targets, identifiers and confirmed post-outcome fields before statistical selection. Constant and near-constant filtering is a cheap unsupervised first step. Record the retained columns so inference uses the identical projection.

Correlation and VIF are diagnostics, not automatic universal thresholds. Dropping correlated variables matters most for coefficient stability and memory; regularized or tree models may retain them. Prefer one representative only when validation supports the choice and domain meaning is preserved.

Any target-dependent selector, mutual information, model importance, SelectFromModel or recursive elimination must be fitted inside each training fold. Never select features once on all labeled data and then report CV on that same projection. PCA and other learned reductions follow the same rule and require scaling when their geometry does.

For wide sparse data, preserve sparsity and budget transformed dimensions. Compare selection with the same folds and metric as the baseline, including fit time and fold variance. Persist selected names or indices and test reordered, missing and extra input columns.
