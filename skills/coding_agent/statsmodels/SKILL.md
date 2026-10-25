---
name: statsmodels
description: Fit a modest-width statsmodels 0.14.0 single-output OLS baseline.
---
# statsmodels 0.14.0

Use direct OLS on already prepared finite numeric/OHE values; add the same intercept at fit and
predict. model_params target OLS.fit, not its constructor. OLS has no sample_weight argument:
the runnable reference rejects weights explicitly; implement WLS only when that is the assignment.
Do not clean features or fit an encoder again.

Prefer modest, approximately linear tables. Ridge/boosting are alternatives for collinearity
or nonlinear hypotheses; avoid huge dense designs and quadratic feature-memory costs.
Never use missing='drop' to silently remove rows. Compare the actual requested metric.

REFERENCE.md implements factories, fit/predict/save/load and optional full parameter export.
Portable results remove cached training data while retaining prediction coefficients.