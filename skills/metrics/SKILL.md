---
name: metrics
description: Match prediction shape, class order and score polarity to MLmom's trusted evaluator and primary metric.
---
# Prediction semantics

Obey prediction_spec['kind']. Accuracy/F1/precision/recall/MCC use label; ROC-AUC, average_precision and log_loss use probability; regression uses value; ranking uses ranking_score; clustering uses cluster. Anomaly AUC/AP uses anomaly_score, larger = more anomalous. Do not invent evaluator metric support.

Single-label predictions are original labels or probabilities (n, len(context.classes)), columns in context.classes order. Binary positive is context.classes[1]. Align model.classes_, including inverse target encoding. Missing training classes require a defined fallback or split failure, never shifted columns. Probabilities must be finite, bounded and normalized. Raw margins need conversion/calibration under this SDK even where upstream AUC accepts scores.

Multilabel: (n, n_targets), one positive probability/binary decision per target; no row normalization. Inspect each target's classes_: constant-positive yields ones, constant-negative zeros. Multioutput regression retains (n, n_outputs); ordinary regression is (n,).

Metric direction affects selection, not prediction sign. Invert normality scores only for anomaly ranking. Never argmax probability metrics. Persist deliberate clipping/inverse transforms, including nonnegative RMSLE outputs. Infrastructure computes final scores.
