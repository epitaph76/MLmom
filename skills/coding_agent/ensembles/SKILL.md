---
name: ensembles
description: Implement blending or OOF stacking with direct candidate code and portable component artifacts.
---
# Ensembles

Use when diverse models have complementary errors. plan.ensemble_components contains at least two {'tool_id': ..., 'model_params': {...}} entries; model_params.method is 'blend' or 'stack'. Select component library skills. Implement create_candidate/load_candidate in candidate.py with installed libraries and mlmom.sdk; do not import runtime adapters.

For blending, align probabilities to context.classes before averaging. For stacking,
fit the meta learner on OOF base predictions: every base estimator and trainable
preprocessor, including target encoding, fits only its base-fold training rows.
Honor groups and sample weights. Refit bases on the current complete training data
after producing OOF predictions. Never use the outer holdout to fit components,
weights, thresholds or the meta learner. Save/load every component and meta learner;
verify relocation and class order. Each outer internal fold receives a fresh ensemble.

Blending uses aligned values or probabilities, not arbitrary class labels.
Choose fixed weights in the plan, or fit weights using only internal OOF data.
Stacking folds must have enough groups and every required class; fail with a
concrete reason rather than silently changing the protocol. Retain component
configuration, class order and any fixed iteration counts for search-free refit.

For model_params.method='stack', implement fit_stack(X,y,context,create_preprocessor).
The SDK passes raw features to this hook and raw features to predict. Compose each base with
mlmom.sdk.PreparedCandidate(base_config, create_base, create_preprocessor); base_config must
not declare the outer stacking ensemble. Use fresh wrappers in each OOF fold and full refit.
Slice groups/weights to fold training rows and clear validation metadata for each base fold.
Save/load wrapped bases and meta learner together. Do not stack on globally transformed X.
Native AutoML components are not admitted until frozen component refit is verified.

## Complete freezing and inspection
Export the full config: components/meta learner, weights, target/output options and flat HPO values. fitted_components() exposes every named base/meta estimator. Apply flat HPO keys only to plan.parameter_routes components. SDK audits actual execution and export/recreate.
