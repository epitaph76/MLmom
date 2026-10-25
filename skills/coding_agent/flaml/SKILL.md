---
name: flaml
description: Run FLAML 2.3.6 native holdout search and weighted frozen refit.
---
# FLAML 2.3.6

Use strategy=native and patience_trials=null for regression/binary/multiclass. The complete
REFERENCE.md uses explicit holdout, estimator_list=['lgbm','extra_tree'], max_trials as max_iter,
ensemble=False and retrain_full=False. Native CV/grouped selection is not verified by this
recipe; do not invent a holdout or reduce the confirmed folds. Empty capability metadata is
uncertainty, not an automatic experiment veto.

Pass sample_weight in search/refit and sample_weight_val in validation. The transient custom
objective calls context.validation_scorer for original-scale metrics/options/class order, including
MCC. FLAML minimizes the returned loss; maximize metrics use its negative. Keep the original
primary score in history. Library budgets must be finite.

Export __native_frozen estimator/config/loss. Fixed refit uses retrain_from_log(train_full=True)
with current weights and no new search. Save only the underlying inference estimator, never
AutoML, FitContext or callback closures. No ranking/forecasting path is supplied here.