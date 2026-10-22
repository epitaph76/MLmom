---
name: validation
description: Keep model selection inside supplied training data and form valid group, class and OOF partitions.
---
# Internal validation

The runner executes the confirmed internal holdout or full kfold/stratified_kfold/group_kfold on outer train. The outer holdout never enters tuning. Only holdout supplies context.validation_X/y for early stopping/native search. Ordinary CV withholds them: fit fixed parameters, then let the runner score every fold. Each fold creates fresh model/preprocessing state. Always reserve a fresh full-train refit; stop ends search but never permits an individual fold model to be evaluated externally. Reject plans that budget only one fold or silently replace CV with holdout.

Mean fold score selects parameters. Fold standard deviation and pooled OOF score are separate evidence; an unfinished fold set is not a completed trial. Native search needs an explicit verified validation_strategies entry and frozen refit without new search. Candidate ensembles and native blending/stacking are disabled.

Only operations needing OOF predictions should create folds inside X/y. Single-label classification: StratifiedKFold, splits no greater than minimum class count. Groups: GroupKFold or StratifiedGroupKFold, at least two groups; verify class coverage because stratification is approximate. Ranking splits whole queries. Exact duplicates need shared groups; report when the runner's split did not account for them.

Fit transforms, resampling and target statistics within each training fold. Fill OOF values at original row positions exactly once. Slice groups/weights with indices. Verify both sides support the metric; do not silently weaken validation to make a fit succeed.

Multilabel needs per-label coverage checks; single-label stratification cannot handle its matrix. Define constant-label behavior. Temporal data needs chronological splits, but current MLmom rejects time_column/forecasting tasks: report deferred support, never substitute random CV.
