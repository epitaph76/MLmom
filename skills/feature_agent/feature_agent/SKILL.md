---
name: feature-agent
description: Write executable Python feature experiments from train-only evidence, including supervised representations and inferred entity history.
---
# Feature agent

Propose additions, removals or replacements as Python programs; DSL is optional. No fixed
feature count or operator whitelist limits your methods. Use installed dependencies and
permitted raw columns. Never use outer holdout/test labels, files, network or credentials.
Programs define create_features(config) returning fit_transform(X,y,context)/transform(X).
Return extra DataFrame/2D array/sparse columns, preserving rows and inference schema. Persist
learned state on the object; the SDK saves it with the model. Keep base preprocessing/model fixed.
Fold-local target statistics, supervised encodings, compound keys, learned representations
and entity history are allowed experiments. Prevent own-label encoding through appropriate
cross-fitting/exclusion, including priors/fallbacks; inference uses training-fitted maps
and no prediction-batch fitting.
Respect actual group/time boundaries. A proxy entity key need not be an official ID: measure
its ambiguity/coverage and benefit rather than treating uncertainty as an automatic refusal.
Explain why each feature could help the requested metric. You may stop when further work is
unlikely to help. Your first proposal is requested immediately after context, before model preparation; there is no fitted control or OOF evidence yet. Runtime later measures proposals with a frozen control and identical internal splits before long HPO. A proposal is not a measured gain.
Selection uses the measured internal mean; fold dispersion is diagnostic and is not a stability
veto. No provenance, estimator identity or mandatory synthetic smoke report is required.

Use action=replace with remove names and new programs/features for one atomic replacement.
Runtime measures the final set against the current accepted set with the same model/folds;
removal alone need not improve. Choose feature count against actual CPU/RAM. Legacy
action=add with both lists is read as replace. Stop changes nothing.
