---
name: preprocessing
description: Fit fold-local tabular and sparse text transforms and preserve inference state.
---
# Preprocessing

Fallback: numeric/bool median imputation, StandardScaler; categories imputed and dense OHE with unknown=ignore, min_frequency=2, max_categories=100. LightGBM defaults to native categories; CatBoost uses native category strings. All vocabularies and transforms fit on current training rows only.

For text or mixed tables use sklearn with model_params estimator=linear: C for classification (LogisticRegression), alpha for regression (Ridge). preprocessing=['tfidf'] explicitly enables sparse word/character TF-IDF; long text is also recognized by content. Numeric scaling preserves sparse output and category OHE stays sparse. No pretrained models/downloads. This path supports unseen Unicode tokens, empty text and missing columns' values.

Custom preprocessing must be implemented and saved; a plan description alone does not execute it. Preserve empty columns or record train-only drops. Replace infinities with missing. Do not fit encoders separately at predict. Estimate dense memory before conversion. Do not apply target transforms, weights or duplicate removal universally.

Calendar/lag features require understood semantics and prediction-time availability; forecasting execution remains deferred. Persist columns, encoders, imputers, scales and inverse target transforms. Test unseen values, missingness, shuffled columns and row order.
