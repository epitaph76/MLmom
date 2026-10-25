"""Shared ownership facts for executable Python features and optional DSL."""
from .feature_contracts import FeatureSpec


def plan_field_sources(plan):
    sources = dict(model_params='Planner parameters',preprocessing='Planner intent; preprocessing_agent implementation',
        feature_set='Runtime registered definition; SDK execution',ensemble_sources='Runtime frozen component registry',
        ensemble_components='Runtime references to frozen sources',cpus='Planner request and runtime allocation',
        memory_gb='Planner request and runtime allocation',target_transform='Planner declaration; SDK execution',
        weight_policy='Planner declaration; SDK execution',prediction_policy='Planner declaration; SDK execution')
    sources.update(search_group_id='Controller-owned containing group',
        search_group_kind='Controller-owned containing group',parent_hypothesis='Controller-owned containing group')
    if plan.configuration_diff:
        sources['inheritance']='Controller-resolved saved champion; reset_fields/remove_feature_names are explicit changes'
    for source in plan.ensemble_sources:
        sources['model_params.'+source['name']]='Runtime frozen configuration: '+source.get('experiment_id',source['name'])
    if plan.ensemble_sources:
        sources['model_params.method']='Explicit ensemble recipe selected by agents'
    return sources


def execution_contract(plan):
    return dict(resource_owner='runtime', cpus=plan.cpus, memory_gb=plan.memory_gb,
        thread_source='FitContext.cpus', thread_params_required_in_model_params=False,
        search_group_owner='controller', search_group_fields='Derived execution metadata, not agent assignments',
        resource_enforcement='worker_admission', preprocessing_owner='preprocessing_agent',
        time_admission='Actual remaining deadlines and explicit limits; historical durations are advisory',
        estimator_owner='coding_agent', policy_owner='runtime', validation_owner='runtime', refit_owner='runtime',
        validation_scorer='Optional internal-holdout primary scorer; original scale/options/weights; absent in preprocessing/features/OOF base fits',
        training_checkpoint_configuration='Complete selected configuration, separate from Optuna sampled_params; no additional fit',
        feature_owner='SDK FeatureAugmentingPreprocessor',
        feature_implementations='Agent Python programs with fit_transform(X,y,context)/transform(X); optional DSL shortcuts',
        feature_target_access='Current training fold y only; target-derived training columns use cross-fitting/exclusion; transform never receives labels',
        feature_input='Raw X; feature statistics fit on current training rows, alongside the base transformed representation',
        transform_order=['fold-local base preprocessing', 'SDK feature_set augmentation', 'estimator'],
        feature_operators=FeatureSpec.model_json_schema()['properties']['op']['enum'],
        feature_operator_list_is_exhaustive=False,
        effective_feature_set=plan.feature_set.model_dump(mode='json') if plan.feature_set else None,
        frozen_components=[dict(name=s['name'], effective_configuration=s['config'],
            preprocessing_prose_role='historical intent; active feature_set is applied separately by SDK')
            for s in plan.ensemble_sources],
        export_model_params='optional', fitted_components='optional', estimator_audit=False)
