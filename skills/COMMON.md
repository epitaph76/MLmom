# Shared AutoML contract

Priority: project contract > executable hypothesis > pinned API > reference examples. Generate training and preprocessing directly; examples are guidance, not callable project adapters.

Use only X/y/FitContext. The runner owns splits, Optuna, comparison and submission.csv. Internal validation is for selection/stopping, never ordinary training. Fixed refit may have no validation. No access to other splits, input folders, full-fit artifacts or credentials.

CPU only: respect context.cpus/memory_gb/remaining_seconds, at most 7 CPU/28 GB. Keep wrappers/loaders serial. Budget arrays/copies and reserve prediction/save time.

Fit transforms on train only. Persist columns, vocabularies, class/output order, inverse target transforms and thresholds. Preserve row order; exclude target/ID/group/fold/weight features. Align groups/weights with permutations.

fit trains one supplied configuration; no nested search. Native AutoML owns its search instead of Optuna. Export selected parameters for fixed refit; no training during import/load/predict.

Only mlmom.sdk and mlmom.ensemble project imports are allowed. No mlmom.adapters/LibraryCandidate, network, shell, environment, GPU, pretrained weights or pip. Use work_dir for temporary files and artifact_dir for saved inference state. Return complete source for mandatory AST/smoke checks.
