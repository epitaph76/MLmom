---
name: metrics
description: Match prediction shape, class order and score polarity to MLmom's trusted evaluator and primary metric.
---
# Prediction semantics

Obey prediction_spec['kind']. Accuracy/F1/precision/recall/MCC use label; MCC supports binary and multiclass targets, accepts sample weights and is maximized; ROC-AUC, average_precision and log_loss use probability; regression uses value; ranking uses ranking_score; clustering uses cluster. Anomaly AUC/AP uses anomaly_score, larger = more anomalous. Do not invent evaluator metric support.

Single-label predictions are original labels or probabilities (n, len(context.classes)), columns in context.classes order. Binary positive is context.classes[1]. Align model.classes_, including inverse target encoding. Missing training classes require a defined fallback or split failure, never shifted columns. Probabilities must be finite, bounded and normalized. Raw margins need conversion/calibration under this SDK even where upstream AUC accepts scores.

Multilabel: (n, n_targets), one positive probability/binary decision per target; no row normalization. Inspect each target's classes_: constant-positive yields ones, constant-negative zeros. Multioutput regression retains (n, n_outputs); ordinary regression is (n,).

Metric direction affects selection, not prediction sign. Invert normality scores only for anomaly ranking. Never argmax probability metrics. Persist deliberate clipping/inverse transforms, including nonnegative RMSLE outputs. Infrastructure computes final scores.

## Diagnostic and comparison metadata
Regression diagnostic mae is mean absolute error; rmse is root mean squared error; mean_residual = mean(y_true-y_pred), positive for underprediction. Classification diagnostic error_rate and multilabel hamming_loss have no residual sign. Diagnostic weighting is explicitly unweighted; primary-score weighting is reported separately. Never infer bias direction from absolute errors. score_kind distinguishes inner_cv_mean, inner_oof and outer_holdout; compare only matching dataset/split, metric/options/direction and weighting. supported_metrics and allowed_metric_options come from the same executable metric catalog used by evaluator and confirm_task. Unknown metric/task/option combinations return an explicit error; do not substitute a different metric.

Early stopping must use the same metric, scale, options, class order and validation weights.
Use context.validation_scorer for raw estimator predictions. A library RMSE on log1p(y) is not
original-scale RMSE. When an exact callback is unavailable, keep fixed fitting and record the
limitation. Native search objectives need the same assessment independently of outer scoring.
