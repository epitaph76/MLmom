---
name: ensembles
description: Implement blending or OOF stacking with direct candidate code and portable component artifacts.
---
# Ensembles

With ensemble_sources, use SDK FrozenEnsemble(config, Path(__file__).resolve().parent/'components')
in create_candidate; load_candidate calls FrozenEnsemble.load(artifact_dir).
Import from mlmom.sdk and pathlib. Do not generate preprocessing.py or change bundled
components. The SDK owns raw-input composition, fold-local policies, OOF and persistence.
The remaining instructions apply to manually composed candidates without ensemble_sources.

Use when diverse models have complementary errors. plan.ensemble_components contains at least two {'tool_id': ..., 'model_params': {...}} entries; model_params.method is 'blend' or 'stack'. Select component library skills. Implement create_candidate/load_candidate in candidate.py with installed libraries and mlmom.sdk; do not import runtime adapters.

Blend values or probabilities aligned to context.classes, never class labels.
Weights are fixed or learned on internal OOF data. Meta learners use OOF predictions;
every base and trainable transform, including target encoding, fits inside its base
fold. Honor groups/weights and require enough groups and every required class; fail
explicitly instead of changing the protocol. Refit bases on all current training rows.
Outer holdout never trains components, weights, thresholds or meta learners. Every
internal evaluation fold gets a fresh ensemble. Persist class order and fixed counts.

For model_params.method='stack', implement fit_stack(X,y,context,create_preprocessor).
This hook and predict receive raw features. Wrap bases with
mlmom.sdk.PreparedCandidate(base_config,create_base,create_preprocessor), omitting the
outer ensemble declaration from base_config. Create fresh wrappers per OOF/full fit;
slice groups/weights and clear validation metadata. Never stack globally transformed X.
Save/load all wrapped bases and meta learners; verify relocation and class order.
Native AutoML components are not admitted until frozen component refit is verified.

## Complete freezing and inspection
Export the full config: components/meta learner, weights, target/output options and flat HPO values. fitted_components() is optional. Apply flat HPO keys only to plan.parameter_routes components. Runtime executes the candidate and reports real errors for repair.
