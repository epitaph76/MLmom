---
name: sklearn
description: Build cheap CPU baselines, sparse linear models, multioutput predictors, clustering or anomaly detectors with scikit-learn 1.5.2.
---
# scikit-learn 1.5.2

Generate direct calls to the chosen estimator. REFERENCE.md shows a complete ExtraTrees/KMeans/IsolationForest candidate. Adapt it for Ridge, LogisticRegression or MiniBatchKMeans; implement any estimator selector explicitly. All declared parameters must reach their actual constructor.

For trees start 100-400 estimators, min_samples_leaf 2-5 and bounded depth on large tables. Sparse/high-dimensional inputs favor scaled Ridge/LogisticRegression with sparse OHE; the ordinary-table reference uses dense OHE; select sparse preprocessing explicitly. Check the concrete estimator's NaN/sparse/sample_weight support. LabelEncoder is for y, never X.

ExtraTrees supports multioutput directly. Custom MultiOutputClassifier/Regressor wrappers should use n_jobs=1 while the base estimator uses the allocation. Multilabel probabilities need positive-class extraction and constant-label handling.

Clustering must predict new rows; DBSCAN/AgglomerativeClustering lack this contract. KMeans needs scaled meaningful features; the evaluator uses the original numeric metric space. IsolationForest anomaly_score is -score_samples; labels map -1 to anomaly 1, +1 to normal 0.
