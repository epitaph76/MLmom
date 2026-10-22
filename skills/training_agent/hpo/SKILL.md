---
name: hpo
description: Configure persistent batched Optuna inside a hypothesis, without recoding candidates or nested search.
---
# HPO

After first valid submission, strategy=optuna uses the shared runner. fit trains supplied model_params once. Search specs: {'type':'float','low':.01,'high':.15,'log':True}, {'type':'int','low':3,'high':8}, {'type':'categorical','choices':[...]}.

One persistent study per implementation, data split, metric and space. Defaults: batches of 5, at most 15 trials / 600 seconds. The default programmatic policy continues only after >=0.5% relative primary-metric improvement per batch. Smaller improvements still update the best checkpoint. min_delta=0 is correct; it is not the batched hpo_gain_threshold. Trials and batch decisions persist immediately; resume keeps the original deadline. HPO batches do NOT create new hypotheses or coder calls.

Use narrow spaces: LightGBM num_leaves 15-63/min_child_samples 10-80; CatBoost depth 4-8/l2_leaf_reg 1-20; learning_rate .025-.15. Search only actual constructor parameters, never CPU/GPU/deadlines/private frozen keys. Export selected tree counts for fixed refit. Stop on plateau and propose a logic/preprocessing change from diagnostics.

Native AutoML uses strategy=native, patience_trials=null; no outer Optuna. Preserve first submission, final reserve and time for later cycles.

## Checkpoint controller

You control execution; the planner selected the model and the separate coder implemented it. Review internal results after the initial fit, at batch boundaries and on failures. Return continue, stop (retain best fitted artifact) or refit (stop search and refit best parameters). You may propose up to five next trial parameter sets within existing distributions. You can override a plateau when justified, never time/trial/resource limits. No code, new distributions, outer labels or per-iteration polling. Put architectural changes in recommendations for the next cycle. Invalid/unavailable decisions use the default policy. Resume reuses committed decisions and the original deadline.
