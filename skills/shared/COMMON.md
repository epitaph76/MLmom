# Shared contract

Priority: project > plan > pinned API > examples. Inputs: X/y/FitContext only.
Runner owns splits/search/evaluation/submission. preprocessing.py owns fold-local features;
candidate.py owns estimator logic. SDK applies target_transform/weight_policy/prediction_policy:
never repeat them. Pass context.sample_weight unchanged. ModelSpec combines complete backend
exports, policies and source hashes. Fit-entry records prove execution; never alter fitted
attributes to pass checks. Existing-file repairs use hashed CodePatch; CodeBundle creates files.

Fit current train only; fresh transforms per fold/refit, transform-only validation/inference.
Preserve rows, groups, weights and class order. OOF stacking includes preprocessing in base folds
and persists every component. No nested HPO. CPU/resources always apply; time callbacks are opt-in.
No external data/network/credentials/shell/pip/downloads. Only mlmom.sdk infrastructure imports.
Persist portable state, not records/FitContext; no import/load/predict training. Use work_dir for
scratch and artifact_dir for persistence.
