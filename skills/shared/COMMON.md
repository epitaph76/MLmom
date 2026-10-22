# Shared AutoML contract

Priority: project contract > executable planner assignment > pinned API > reference examples.
The planner selects the model and transformations. The preprocessing agent writes feature transforms;
the model coder writes estimator and target/output logic. SDK composes both. The separate training
agent controls existing execution at checkpoints; it never writes code or changes search distributions.

Use only X/y/FitContext. The runner owns splits, Optuna, comparison and submission. Preprocessing
learns only from current training rows; validation uses transform. Fixed refit uses fresh transforms
on all permitted rows with selected model parameters. Preserve rows, groups, weights and class order.

CPU only: respect context.cpus/memory_gb/remaining_seconds. No external data, network, credentials,
pretrained models, shell or pip. Only mlmom.sdk infrastructure imports are allowed.
Candidate ensembles, blending and OOF stacking are allowed. Persist every component;
fit stack meta learners on OOF base predictions with preprocessing inside each base fold.
Hard timeouts are opt-in; CPU, memory and data-isolation constraints always apply.
Use work_dir for temporary state and artifact_dir for portable inference files. Never persist training
records or FitContext. No training at import/load/predict. No nested HPO inside candidate.fit.
