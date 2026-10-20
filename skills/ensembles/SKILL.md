---
name: ensembles
description: Blend or OOF-stack diverse supervised candidates using the project ensemble adapter and portable component artifacts.
---
# Ensembles

Use after distinct strong models exist and refit time remains. plan.ensemble_components contains at least two {'tool_id': ..., 'model_params': {...}} entries; model_params.method is 'blend' or 'stack'. Select component library skills. Factories use mlmom.ensemble.EnsembleCandidate and load_ensemble.

Stock blend is an equal mean, with no tuned-weight option. Align probability columns before blending; preserve regression target dimensions. Weighted blends need custom internal-validation/OOF selection. Never average arbitrary labels.

Stock stack fits its meta-model on three-fold OOF predictions then refits components. It uses GroupKFold for groups, StratifiedKFold for single-label classes and KFold otherwise. Verify three groups/minimum class count and rare-label coverage. Invalid folds or custom class order need a custom path. Never use in-sample base scores or prefit models from another split for meta-fit.

Budget OOF plus final component fits; prefer blend if gains are uncertain. Preserve export_model_params' frozen components for refit. Save/load each component under the artifact, checking probability alignment and reloaded predictions.
