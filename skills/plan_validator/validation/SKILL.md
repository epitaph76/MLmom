---
name: validation
description: Keep model selection inside supplied training data and form valid group, class and OOF partitions.
---
# Internal validation

## Review scope and execution ownership

Review exactly the single executable plan supplied in plans. review_scope identifies
its backend, family and siblings scheduled or already accepted by the runtime.
Sibling plans are intentionally absent from this payload. Never reject one member
because it does not also contain LightGBM, CatBoost or XGBoost siblings. The runtime
expands families and checks the required backend identity and group membership.

execution_contract declares trusted ownership. CPU/RAM allocation and worker admission
belong to the runtime; the estimator coder obtains threads from FitContext.cpus and
the machine execution gate verifies the fitted model. Do not require n_jobs,
num_threads or thread_count in model_params/search_space: their absence is expected.
Explicit conflicting resource overrides remain defects, checked by Python.
preprocessing_agent owns X transforms, coding_agent owns the estimator, and the SDK
owns target/weight/prediction policies. Compare executable policy fields with the
current assignment; parent-hypothesis prose is historical intent, not an override.

Review feasibility and leakage of the declared experiment, not possible redesigns.
A cheap baseline need not exhaust every feature-engineering technique. Circular
encoding, rare-category pooling and missing indicators are optional experiments
unless supplied evidence demonstrates a concrete failure of the chosen recipe.
Prefer at most three distinct, actionable warnings, tied to plan fields and evidence.
Do not limit actual blocking issues or replace missing evidence with speculation.

Machine checks own formal backend/skill/protocol compatibility. For a blocking semantic
issue return rule_id, field, reason and an exact source_quote from that plan field in issues.
Use task_contract for canonical task contradictions and holdout_training for explicit use
of external labels in fitting. Unproven library incompatibilities are warnings requiring a
local counterexample/probe, not a reason to regenerate the plan. pandas categorical unseen
values represented as NaN are supported by pinned LightGBM. Runtime executes declared
target_transform, weight_policy and prediction_policy; ownership need not be reassigned to a coder.

The runner executes the confirmed internal holdout or full kfold/stratified_kfold/group_kfold on outer train. The outer holdout never enters tuning. Only holdout supplies context.validation_X/y for early stopping/native search. Ordinary CV withholds them: fit fixed parameters, then let the runner score every fold. Each fold creates fresh model/preprocessing state. Always reserve a fresh full-train refit; stop ends search but never permits an individual fold model to be evaluated externally. Reject plans that budget only one fold or silently replace CV with holdout.

Mean fold score selects parameters. Fold standard deviation and pooled OOF score are separate evidence; an unfinished fold set is not a completed trial. Native search needs an explicit verified validation_strategies entry and frozen refit without new search. Ensembles are allowed; verify every component and require OOF predictions for stack meta learners, including fold-local preprocessing and group isolation.

Only operations needing OOF predictions should create folds inside X/y. Single-label classification: StratifiedKFold, splits no greater than minimum class count. Groups: GroupKFold or StratifiedGroupKFold, at least two groups; verify class coverage because stratification is approximate. Ranking splits whole queries. Exact duplicates need shared groups; report when the runner's split did not account for them.

Fit transforms, resampling and target statistics within each training fold. Fill OOF values at original row positions exactly once. Slice groups/weights with indices. Verify both sides support the metric; do not silently weaken validation to make a fit succeed.

Multilabel needs per-label coverage checks; single-label stratification cannot handle its matrix. Define constant-label behavior. Temporal data needs chronological splits, but current MLmom rejects time_column/forecasting tasks: report deferred support, never substitute random CV.

Reject a narrative duplicate-aware protocol unless confirmed duplicate_policy=group_features. validation_protocol must match the immutable task. Require full export/recreate and fitted component inspection; plan stopping_policy and component policies must be executable with the confirmed validation. A single-change ablation freezes Champion config/hash without additional HPO. Flat ensemble search keys need parameter_routes. Machine execution admission is independent of this advisory source review.
