---
name: temporal-group-structure
description: Preserve chronology and entity isolation when preprocessing repeated or time-dependent observations.
---
# Temporal and group structure

Identify repeated entities such as users, devices, buildings or patients before random splitting. Keep a group wholly on one side of a fold. For classification use stratified group splitting only when class coverage remains valid; verify every fold rather than assuming approximate stratification succeeded.

For temporal prediction, train must precede validation. Build lags, rolling aggregates and point-in-time joins only from information available before each prediction timestamp. Fit reference dates, category maps, imputers and samplers on the past training window. Never let a future row contribute to a past aggregate.

The current MLmom orchestrator defers explicit time columns, time-series validation and forecasting. Do not bypass that guard with a random split. Report the task as unsupported until the runtime provides a chronological contract.

For supported grouped tasks, slice groups and sample weights with every training permutation. Target or frequency aggregates require group-safe out-of-fold values. Persist group-independent transforms; do not require known inference groups unless the task explicitly guarantees them.
