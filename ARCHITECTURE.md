# MLmom runtime

The runtime extends the existing DataAnalisys package. `solution.py` is the organizer entrypoint; `mlmom.pipeline.Pipeline` owns the persistent run graph. No runtime fallback generates a model without the LLM.

Boosters copies only root Python files and requirements. The upload ZIP contains
`solution.py`, `requirements.txt`, and generated `mlmom_bundle.py`. The latter
stores this project's source, manifests and skill placeholders as Python strings
and restores them to a content-addressed local runtime directory before imports.
Workers import that restored package through their explicit PYTHONPATH. No
third-party code, packages or model weights are bundled. No installation or
network request is used for bootstrapping. Platform credentials come only from
OPENAI_API_KEY/OPENAI_BASE_URL; OPENAI_MODEL defaults to qwen/qwen3.8-27b.

## Graphs and ownership

`inspect -> plan -> execute -> analyze -> plan | finalize` is a LangGraph 0.6.7 StateGraph with a SQLite checkpointer. Planning and coding each run bounded validation/repair subgraphs. The original task-analysis graph retains its Pydantic state and explicitly disables inherited persistence; its result is persisted by the outer graph as JSON.

LLM roles propose hypotheses, plans and source. Trusted Python owns split indices, trial objectives, resource allocation, metric computation, ranking and output publication. Candidate code never executes in the controller. Tool schemas are built from typed function signatures; roles receive an explicit capability set.

## Candidate lifecycle

1. Resolve inputs and profile only train/dictionaries/task. Prefer explicit manifest paths, existing paths from task prose, siblings, then a unique recursive match. Reject ambiguity/path traversal.
2. Validate TaskDefinition against the stage-one profile. Keep TaskSpec as the existing heuristic contract; TaskContext references the canonical task and profile.
3. Probe each installed, version-matching library adapter in a bounded subprocess. Missing or failing libraries are unavailable to Planner.
4. Generate up to three plans. Apply deterministic checks, semantic validation and bounded repair.
5. Load versioned skills. Placeholder packages are explicitly marked and do not block code generation.
6. Generate source in a restricted workspace. Preserve every revision. Check AST/import policy and a synthetic fit/predict/save/load cycle in a subprocess.
7. Train only the allowed outer-train rows. Optuna optimizes ordinary candidates on an internal split; native AutoML uses its own search. Save a new immutable model directory for every best checkpoint, atomically updating its progress pointer.
8. Predict holdout features in a subprocess. The controller calculates metrics against labels it retains. Predictions, model code, data hash, split, parameters and dependency versions remain linked in the ledger.

The generated module exports `create_candidate(config)` and `load_candidate(artifact_dir)`. Its object implements `fit(X, y, FitContext)`, `predict(X, {kind: ...})`, and `save(artifact_dir)`. Models may use the real library adapters or implement custom transformations. Factories never train at import time. Local custom classes have stable module names for serialization.

Stage-one models have additive optional fields: `TaskDefinition.metric_options` and `ResultFormat.prediction_kind/class_order`. Old artifacts without them remain readable. These fields carry ranking cutoffs, averaging rules, label/probability output distinctions and explicit probability-column ordering. Evaluation retains canonical training class order; output ordering is applied only when publishing a submission.

## Rounds and evaluation

Each full round gets a new reproducible outer split. All candidates in that round share it. Classification is stratified, multilabel uses rare-label allocation, ranking/groups preserve group membership. Time-aware execution is deferred. Explicit dictionary feature roles outrank year/date name heuristics.

The incumbent is retrained using its selected parameters, without a new search. Promotion compares only current-round evaluations. Ties retain the incumbent. If incumbent evaluation fails, historical Champion remains and new candidates are provisional. Earlier scores on other splits are never treated as comparable evidence of improvement.

Top results within a relative 1% gap are passed forward, at most three, together with aggregate error diagnostics and recent failures. No raw holdout examples enter prompts. Five conclusive rounds without a challenger victory stop research. Blending and OOF stacking refit their components using only current training data.

## Processes, deadlines and recovery

Default budget is 180 minutes, with a 40-minute finalization reserve, 1 CPU and 4 GB controller reserve, at most three concurrent experiments. Admission uses declared CPU/RAM plus cost-tag priority; affinity allocations are disjoint. Workers receive thread limits and no API keys, and their process trees are monitored for RSS and elapsed time. A worker watchdog also detects parent death. This is fault containment, not a hostile-code security boundary.

The initial baseline is prepared/executed before alternatives. The ten-minute first-file target is a goal, not a guarantee when the LLM or library is slow. Research models predict test directly and atomically update a validated submission. At finalization, a separate model is refitted on all train; it is never evaluated on an earlier holdout. Failed finalization preserves the existing file.

SQLite stores domain objects/events and checkpoint state separately. Workers emit file results; the controller writes the ledger. Deterministic experiment IDs reconcile completed workers with resumed graph execution. Session hashes reject changed task inputs. API usage is recorded; dev has no separate token/currency ceiling. No state is shared between competition tasks.

## Supported libraries and limits

Adapters cover sklearn, CatBoost, LightGBM, XGBoost, statsmodels OLS, CPU torch, FLAML, AutoGluon and LightAutoML. Capabilities differ by task; the registry and runtime probes decide availability. AutoGluon explicitly selects non-pretrained tree families. LightAutoML uses untuned CPU lgb/cb presets. Native trial reporting preserves backend logs/descriptions instead of pretending all libraries expose identical callbacks.

Generic clustering evaluation currently requires finite numeric features for its metric space. Anomaly metrics require explicit evaluation labels. Unknown metrics/output contracts are rejected. Temporal forecasting and a separate untouched final audit are outside v1. Result probabilities for submission can be selected through `submission_kind`; per-metric options are explicit configuration.

All 15 SKILL.md files intentionally contain placeholders. Their manifests, loading, hashes, compatibility checks and context limits are implemented; substantive instruction content remains future work.
