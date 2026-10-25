"""Resolve full agent plans against the exact saved champion before review."""
from __future__ import annotations

import copy
from .execution_facts import configuration_diff
from .feature_contracts import FeatureSet, feature_hash


def inherit_champion(plan, champion, supplied):
    if not champion or plan.hypothesis.based_on != champion['candidate_id']:
        return plan
    base = copy.deepcopy(champion['plan'])
    result = plan.model_dump(mode='json')
    resets = set(plan.reset_fields)
    same_backend = plan.tool_id == base['tool_id']
    result['model_params'] = dict(base.get('model_params', {}) if same_backend and 'model_params' not in resets else {},
                                  **plan.model_params)
    result['preprocessing'] = (plan.preprocessing or ([] if 'preprocessing' in resets else base.get('preprocessing', [])))
    for key in ('target_transform', 'weight_policy', 'prediction_policy'):
        original = {} if key in resets else base.get(key, {})
        result[key] = dict(original, **(supplied.get(key) or {}))
    if not plan.feature_set_id:
        if 'feature_set' in resets:
            result.update(feature_set_id=None, feature_set=None)
        else:
            result.update(feature_set_id=base.get('feature_set_id'), feature_set=base.get('feature_set'))
    if plan.remove_feature_names:
        definition = FeatureSet.model_validate(result.get('feature_set') or {'feature_set_id':'empty'})
        names = {item.name for item in [*definition.features, *definition.programs]}
        unknown = set(plan.remove_feature_names) - names
        if unknown:
            raise ValueError('Removal references unknown features: ' + ', '.join(sorted(unknown)))
        definition.features = [item for item in definition.features if item.name not in plan.remove_feature_names]
        definition.programs = [item for item in definition.programs if item.name not in plan.remove_feature_names]
        fingerprint = feature_hash(definition)
        definition.feature_set_id = 'fe-' + fingerprint[:16] if fingerprint else 'empty'
        result.update(feature_set_id=definition.feature_set_id, feature_set=definition.model_dump(mode='json'))
    # Ensemble sources contain frozen per-component configurations and must survive reuse.
    for key in ('ensemble_sources', 'ensemble_components', 'parameter_routes', 'stopping_policy', 'component_stopping_policies'):
        if key not in supplied:
            result[key] = copy.deepcopy(base.get(key, result[key]))
    result['base_config_hash'] = champion['configuration_hash']
    result['configuration_diff'] = dict(base_candidate_id=champion['candidate_id'],
        base_experiment_id=champion.get('experiment_id'),base_configuration_hash=champion['configuration_hash'],
        base_code_hash=champion.get('code_hash'),base_preprocessing_hash=champion.get('preprocessing_hash'),
        base_parameters=base.get('model_params', {}), **configuration_diff(base, result))
    result['intervention']['changed_fields'] = [key for key in result['configuration_diff']
        if key in ('tool_id','model_params','preprocessing','target_transform','weight_policy','prediction_policy','ensemble_sources')]
    features = result['configuration_diff']['features']
    if any(features.values()):result['intervention']['changed_fields'].append('feature_set')
    return type(plan).model_validate(result)
