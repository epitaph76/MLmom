---
name: temporal-group-structure
description: Preserve chronology and entity isolation when preprocessing repeated or time-dependent observations.
---
# Temporal and group structure

Identify repeated entities such as users, devices, buildings or patients before random splitting and report required group boundaries to the planner/orchestrator, which owns the split. Keep a group wholly on one side of a fold; preprocessing.py must not change folds, row count or order. For classification use stratified group splitting only when class coverage remains valid; verify every fold rather than assuming approximate stratification succeeded.

For temporal prediction, train must precede validation. Build lags, rolling aggregates and point-in-time joins only from information available before each prediction timestamp. Fit reference dates, category maps, imputers and samplers on the past training window. Never let a future row contribute to a past aggregate.

The current MLmom orchestrator defers explicit time columns, time-series validation and forecasting. Do not bypass that guard with a random split. Report the task as unsupported until the runtime provides a chronological contract.

For supported grouped tasks, any estimator-side training sampling or permutation must slice y, groups and sample weights identically. Target-derived aggregates require group-safe out-of-fold training values and train-only maps for inference. Feature-only frequency maps are fitted on the current training rows; apply stricter entity isolation when the hypothesis requires unseen-group generalization. Persist group-independent transforms; do not require known inference groups unless the task explicitly guarantees them.
