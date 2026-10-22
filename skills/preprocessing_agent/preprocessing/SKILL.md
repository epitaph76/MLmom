---
name: preprocessing
description: Fit fold-local tabular and sparse text transforms and preserve inference state.
---
# Preprocessing

Fallback: numeric/bool median imputation, StandardScaler; categories imputed and dense OHE with unknown=ignore, min_frequency=2, max_categories=100. Choose and implement native categories for LightGBM/CatBoost deliberately; no hidden adapter prepares data. All vocabularies and transforms fit on current training rows only.

For text or mixed tables generate a sparse ColumnTransformer with word/character TF-IDF, LogisticRegression (C) or Ridge (alpha). Implement text-column selection explicitly from the schema/hypothesis. Numeric scaling preserves sparse output and category OHE stays sparse. No pretrained models/downloads. This path supports unseen Unicode tokens, empty text and missing columns' values.

Custom preprocessing must be implemented and saved; a plan description alone does not execute it. Preserve empty columns or record train-only drops. Replace infinities with missing. Do not fit encoders separately at predict. Estimate dense memory before conversion. Do not apply target transforms, weights or duplicate removal universally.

Calendar/lag features require understood semantics and prediction-time availability; forecasting execution remains deferred. Persist columns, encoders, imputers, scales and inverse target transforms. Test unseen values, missingness, shuffled columns and row order.
