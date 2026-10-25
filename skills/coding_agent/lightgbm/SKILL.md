---
name: lightgbm
description: Fit CPU LightGBM 4.6.0 with a prepared representation and exact primary stopping.
---
# LightGBM 4.6.0

Preprocessing owns train-derived category dtype/codes, including explicit casts for numeric-coded
categories. Unknown levels become missing; numeric NaN is supported. Do not recast or refit X
inside the estimator. REFERENCE.md provides a complete single-target candidate using the SDK
scorer, validation weights and correct metric polarity. Unsupported exact stopping uses fixed
fitting with a recorded reason. Export full parameters and best_iteration_ only when stopping ran.

Typical starting values: learning_rate .03-.08, num_leaves 31-63, min_child_samples 20-50.
Row subsampling needs subsample_freq>0. Threads come from context.cpus.
Serial multioutput wrappers and custom ranking need their own implementation. Ranking passes
contiguous query sizes summing to row count and jointly permutes X/y/groups/weights.

A successful fit can retain fewer trees than requested when no valid split exists. Preserve the
declared fixed count; do not edit counters, intercept Booster.update or raise audit exceptions.