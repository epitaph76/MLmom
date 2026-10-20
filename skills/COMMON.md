# Shared AutoML contract

Priority: project contract > available adapter > pinned library API > heuristics. Installed packages do not prove adapter support; use the capability report and inspect_api.

Use only X, y and FitContext for this fit. The runner owns outer holdouts, trials, Champion and submission.csv. Never read input directories, other splits, prior full-fit artifacts or credentials. context.validation_X/y are INTERNAL validation for selection/early stopping, not ordinary training. Fixed refits may have no validation.

Python 3.12, CPU only: at most min(context.cpus, 7) CPU and min(context.memory_gb, 28) GB. Respect remaining_seconds; leave time for prediction/saving. Never infer budgets from the host. Keep wrappers/transformers/loaders serial when the estimator uses its allocation. Estimate matrix/copy costs; memory settings are hints.

Fit transforms on training folds only. Persist columns, vocabularies, class/output order, target transforms and thresholds. Preserve prediction row order. Exclude target/ID/group/fold/weight features unless explicitly allowed. Slice groups/weights with training permutations.

The runner owns Optuna; fit trains one supplied configuration. Native AutoML is an alternative search owner. Freeze selected parameters for refit; no search on load/predict. Keep a verified baseline before boosters/features/ensembles.

Generated code may import only mlmom.sdk, mlmom.adapters and mlmom.ensemble from project internals. Follow the source allowlist: no network, shell, environment access, GPU, pretrained artifacts or pip. Write native temporary files under context.work_dir and portable inference state under artifact_dir. Run check_source and run_smoke_check before returning complete source.
