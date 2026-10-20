---
name: statsmodels
description: Fit a small interpretable single-output OLS regression baseline with statsmodels 0.14.0.
---
# statsmodels 0.14.0

Only single-output regression. Prefer small approximately linear tables with modest numeric/OHE width. Use Ridge or boosting for collinearity/nonlinearity; avoid huge dense OHE and quadratic feature-memory costs.

LibraryCandidate fits sm.OLS(y, sm.add_constant(values, has_constant='add')).fit(**model_params) after common preprocessing. Parameters target OLS.fit, not its constructor: no missing or estimator-selector keys. This adapter has no sample_weight support; reject weighted tasks rather than discard weights.

Custom code imputes/encodes train-only and adds the same intercept at fit/predict. Keep arrays finite; missing='drop' must not silently remove rows. Persist preprocessing and fitted results. Check rank deficiency and compare the requested metric on the runner split; interpretation alone does not establish predictive gain.
