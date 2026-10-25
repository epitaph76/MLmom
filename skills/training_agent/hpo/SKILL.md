---
name: hpo
description: Configure persistent batched Optuna inside a hypothesis, without recoding candidates or nested search.
---
# HPO

After first valid submission, strategy=optuna uses the shared runner. fit trains supplied model_params once. Search specs: {'type':'float','low':.01,'high':.15,'log':True}, {'type':'int','low':3,'high':8}, {'type':'categorical','choices':[...]}.

One persistent study per implementation, data split, metric and space. Defaults: batches of 5, an agent-assigned initial trial budget (default 15); time deadlines apply only when enabled. The default programmatic policy continues only after >=0.5% relative primary-metric improvement per batch. Smaller improvements still update the best checkpoint. min_delta=0 is correct; it is not the batched hpo_gain_threshold. Trials and batch decisions persist immediately; resume keeps the original deadline. HPO batches do NOT create new hypotheses or coder calls.

Use narrow spaces: LightGBM num_leaves 15-63/min_child_samples 10-80; CatBoost depth 4-8/l2_leaf_reg 1-20; learning_rate .025-.15. Search only actual constructor parameters, never CPU/GPU/deadlines/private frozen keys. Export selected tree counts for fixed refit. Stop on plateau and propose a logic/preprocessing change from diagnostics.

Native AutoML uses strategy=native, patience_trials=null; no outer Optuna. Preserve first submission, final reserve and time for later cycles.

## Checkpoint controller

You control execution; the planner selected the model and the separate coder implemented it. Review internal results after the initial trial, at batch boundaries and on failures. Return continue, stop or refit. Both stop and refit end search and require a fresh full-train fit of selected parameters before external evaluation. A trial completes every confirmed internal fold; a partial fold set is not a valid score. Budget all folds plus refit; never publish an individual fold model. Optuna alone chooses numeric configurations. Use additional_trials with continue to extend the unchanged study in measured batches; trial_limit, when supplied, is the explicit user maximum. You can override a plateau when justified, never explicit time/resource/user trial limits. No code, new distributions, outer labels or per-iteration polling. Put architectural changes in recommendations for the next cycle. Invalid/unavailable decisions use the default policy. Resume reuses committed decisions, imported baseline evidence, budget extensions and the original deadline when one exists.

A null remaining_seconds means no deadline; use actual trial/fold durations and remaining_trials. TPE starts after three completed observations by default (configurable). An equivalent already measured baseline may be imported without fitting and does not consume new-trial budget. Imported historical scores with different splits/protocols are never eligible. Architectural, feature or search-space changes require a new experiment. When plateau suggests refit, a justified continue may override it; even tiny gains remain best checkpoints.
Family baselines can be reused from their measured internal protocol before outer evaluation.
CPU thread allocations are execution settings. Source-owned defaults added by an optional
export do not by themselves change the original assignment; changed fixed learning knobs do.
