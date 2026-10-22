---
name: numeric-transforms
description: Handle numeric scale, skew, heavy tails and outliers without deleting valid rare signal.
---
# Numeric transforms

Use train-fold quantiles, median, IQR, skew and domain bounds to distinguish data errors from valid tails. Do not automatically remove statistical outliers: rare target cases may be the predictive signal. Isolation or local-density methods are diagnostic unless the task and validation demonstrate a safe filtering rule.

Scale for linear, SVM, kNN, PCA and neighbour-based sampling. Standard scaling is the baseline for regular distributions; robust scaling is a candidate when valid heavy tails dominate. Trees and native boosters usually need no scaling. With sparse data avoid centering and estimate dense memory before conversion.

Apply feature log1p only when its domain is valid. Target transformations belong to coding_agent, with the inverse applied to predictions and persisted in the estimator; preprocessing.py transforms X only and must not change y. Yeo-Johnson, power or quantile transforms are optional experiments, not universal cleanup; nonlinear transforms alter distances and interpretation. Learn clipping or winsorization bounds inside each fold and compare against an unclipped baseline.

Replace non-finite results deliberately, guard ratios against zero denominators, and preserve the original feature order. Evaluate transformations on the task metric and required target scale rather than on prettier histograms.
