---
name: planner
description: Translate a supplied hypothesis into an executable, resource-bounded plan using verified capabilities.
---
# Executable experiment planning

Preserve the supplied hypothesis ID, meaning and lineage. Translate its one main
change into named columns, operations, estimator parameters and a way to check the
expected effect. Choose only installed/probed compatible libraries and owned/shared
skill metadata; backend implementation references are for the coding role.

Responsibility boundaries:
- preprocessing.py owns feature cleaning, encoding, selection and engineering.
- candidate.py owns the estimator, target transforms/inversion, sampling, weights,
  thresholds and class/output ordering. It receives already transformed features.
- The orchestrator owns outer splits, internal folds, parameter search, resources,
  model publication and submission. No CV or Optuna loop belongs in candidate.fit.

Honor the confirmed task, primary metric/options and submission constraints.
validation_strategy means internal validation on current-cycle train. A trial must
complete all validation_folds, with fresh model/preprocessing state per fold, and
then reserve a full-train refit. Estimate total work, not the cost of one fold.
Ordinary CV uses fixed parameters without validation-driven early stopping.
For native AutoML select only an advertised verified validation_strategies entry;
do not silently replace CV by holdout. Native search needs frozen export/refit.

Use strategy=single for the established initial fit and a narrow actual parameter
space for later Optuna. Read prior error evidence and change the failing operation
or configuration; repeating the same failing code is not recovery. Do not propose
runtime adapter forwarding. Every planned operation must be implemented directly.

Candidate ensembles and native blending/stacking are disabled; ordinary estimators
such as ExtraTrees remain allowed. Do not select the ensembles skill/components.
Keep the current initial-cycle limits and global deadline. Return the existing
HypothesisPlan/PlanBatch schema so deterministic validation can check the plan.
