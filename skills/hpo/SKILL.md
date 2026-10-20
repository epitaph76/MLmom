---
name: hpo
description: Configure bounded search spaces for MLmom's shared Optuna runner without nested search or resource overrides.
---
# HPO

After a verified baseline, strategy='optuna' uses the shared runner. candidate.fit trains supplied model_params once. plan.search_space specs use {'type':'float','low':.01,'high':.15,'log':True}, {'type':'int','low':3,'high':8}, or {'type':'categorical','choices':[...]}.

Use narrow priors: LightGBM num_leaves 15-127, min_child_samples 10-100; XGBoost max_depth 3-8, min_child_weight .5-20; CatBoost depth 4-8, l2_leaf_reg 1-30. learning_rate .02-.15 is a useful log range. Try a few influential parameters and roughly 10-30 affordable trials after timing a fit.

Only search actual constructor parameters; never infrastructure CPU/GPU/deadline/family/private __*_frozen keys. Custom early stopping exports fixed tree counts; stock boosting adapters need explicitly budgeted n_estimators/iterations.

Native AutoML uses strategy='native', patience_trials=null, no outer Optuna. max_trials bounds FLAML; other native adapters rely on time limits. Keep the final training reserve. Avoid search when split noise or remaining time exceeds plausible gain.
