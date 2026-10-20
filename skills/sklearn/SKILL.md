---
name: sklearn
description: Build cheap CPU baselines, sparse linear models, multioutput predictors, clustering or anomaly detectors with scikit-learn 1.5.2.
---
# scikit-learn 1.5.2

LibraryCandidate selects ExtraTreesClassifier/Regressor, KMeans or IsolationForest from task_type. model_params go to that constructor; no estimator/solver selector exists. Ridge, LogisticRegression or MiniBatchKMeans need custom candidates.

For trees start 100-400 estimators, min_samples_leaf 2-5 and bounded depth on large tables. Sparse/high-dimensional inputs favor scaled Ridge/LogisticRegression with sparse OHE; stock preprocessing is dense. Check the concrete estimator's NaN/sparse/sample_weight support. LabelEncoder is for y, never X.

ExtraTrees supports multioutput directly. Custom MultiOutputClassifier/Regressor wrappers should use n_jobs=1 while the base estimator uses the allocation. Multilabel probabilities need positive-class extraction and constant-label handling.

Clustering must predict new rows; DBSCAN/AgglomerativeClustering lack this contract. KMeans needs scaled meaningful features; the evaluator uses the original numeric metric space. IsolationForest anomaly_score is -score_samples; labels map -1 to anomaly 1, +1 to normal 0.
