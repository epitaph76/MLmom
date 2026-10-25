---
name: categorical-encoding
description: Encode categorical features with controlled cardinality, unknown handling and target-leakage protection.
---
# Categorical encoding

Measure cardinality, rare-category mass and identifier-like uniqueness before choosing an encoding. Use one-hot encoding for low or moderate cardinality with explicit unknown handling and, where supported by the pinned API, bounded infrequent-category grouping. Keep the matrix sparse for wide linear models. Do not assign arbitrary ordinal numbers to nominal values for linear, distance or neighbour models.

Use ordinal encoding only for a real ordered feature or as an intermediate representation required by a compatible algorithm. For CatBoost, preprocessing returns stable string/sentinel categories. For LightGBM use train-derived pandas category levels or compatible categorical integer codes; object strings are not accepted. For XGBoost 2.1.4 use matching pandas category levels/codes and have the model enable categorical support. Unknown levels follow the saved backend-specific policy. Declare native column names/indices in PreprocessingSpec.categorical_features.

High-cardinality features require a deliberate hypothesis: rare grouping, hashing, frequency encoding or target encoding. Target encoding must produce out-of-fold values for training rows and a full-training map only for new rows. Respect group or temporal boundaries; ordinary random cross-fitting is not group-safe. Store a fallback for unseen and missing categories.

Never fit `get_dummies` separately on train and prediction data. Persist vocabulary, output order, smoothing and unknown policy. Check unseen categories, categories absent from a fold, mixed Python types, memory growth and save/load equality.
