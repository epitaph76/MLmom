---
name: xgboost
description: Use the pinned CPU XGBoost 2.1.4 API for histogram models and query rankers.
---
# XGBoost 2.1.4

Use CPU hist with preprocessing-fitted OHE or stable native pandas category levels and
enable_categorical=True. Do not assume category recoding from newer releases. Encode single-label
targets contiguously and preserve the mapping. Native categorical models use JSON/UBJSON.

REFERENCE.md contains a complete single-target candidate. Early stopping/eval_metric/callbacks
belong in the constructor in 2.1.4. Custom metrics must accept sample_weight; maximize/minimize
must match the primary direction. Use the SDK scorer or fixed fitting, and freeze best_iteration+1
only when stopping actually ran. Remove stopping callbacks for validation-free fixed refit.
Optional exports retain all parameters; do not duplicate SDK output or target policies.

Ranking needs sorted qid and joint metadata permutation. Native ranker weights are per-query;
row weights require an explicit compatible conversion or a different recipe. Multioutput wrappers
are serial. Threads use context.cpus. Typical max_depth is 3-8 and learning_rate .03-.1.