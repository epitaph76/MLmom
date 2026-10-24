# Shared contract

Use the confirmed task and Planner assignment. Inputs are X/y/FitContext; runner owns
splits, search, scoring and CSV. preprocessing.py owns transforms, candidate.py owns estimator
logic, SDK owns target/weight/output policies. Do not apply policies twice.
Fit fresh state on current train only; validation/inference use saved transforms. Preserve
row/group/weight/class order. Stacking uses fold-local OOF; no nested HPO. Use allocated CPU/RAM
and configured budgets. Do not read external labels, credentials, network data or task input
directories. Persist portable state in artifact_dir and scratch work in work_dir.
Runtime checks syntax/interfaces and real I/O. Optional export_model_params supplies JSON
parameters for refit; otherwise trial parameters are reused. There is no fitted-model audit.
