---
name: preprocessing
description: Fit fold-local tabular transforms, preserve inference state and control unknown categories and sparse/dense memory.
---
# Preprocessing

LibraryCandidate uses median imputation/StandardScaler for numeric/bool and most-frequent imputation/dense OneHotEncoder(handle_unknown='ignore', min_frequency=2, max_categories=100) otherwise. CatBoost/native AutoML use separate paths. plan.preprocessing describes intent; it does not implement transforms.

Use train-derived column lists in ColumnTransformer/Pipeline. Preserve empty columns with keep_empty_features=True or record train-only drops. Convert infinity to missing and mixed categories to stable strings/sentinels. Never fit get_dummies separately at prediction.

For wide linear models keep OHE sparse; StandardScaler on sparse input needs with_mean=False. Estimate rows * transformed_columns * dtype_size plus copies/model state before densifying. Trees usually need no scaling. Native categorical boosters need fixed train vocabularies/codes.

Parse dates consistently into selected calendar/elapsed features; learned reference dates/cutoffs are train-only. Persist columns, dtypes and all encoders/imputers/scalers. Predict only transforms. Check unseen categories, missing values, shuffled columns and non-default row indices. Invert any y transform while preserving output order.
