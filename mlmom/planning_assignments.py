"""Normalize experiment bookkeeping without changing execution safety checks."""
from __future__ import annotations

import copy


def normalize_champion_assignment(item, champion):
    warnings = []
    if not champion or item.hypothesis.based_on != champion['candidate_id']:
        if item.change_kind == 'single_change_ablation':
            item.change_kind = 'experiment'
            warnings.append('Ablation has no current Champion reference; treated as an ordinary experiment.')
        return warnings
    digest = champion.get('configuration_hash')
    if item.base_config_hash and item.base_config_hash != digest:
        warnings.append('Replaced narrative base_config_hash with the authoritative Champion reference.')
    item.base_config_hash = digest
    if item.change_kind != 'single_change_ablation':
        return warnings
    old = champion['plan']
    fields = {'preprocessing', 'target_transform', 'weight_policy', 'prediction_policy',
              'ensemble_components', 'stopping_policy', 'feature_set_id', 'model_params'}
    declared = {key.split('.')[0] for key in item.intervention.changed_fields}
    if item.tool_id != old['tool_id'] or declared - fields:
        item.change_kind = 'experiment'
        warnings.append('Model/backend or custom-code intervention is an ordinary experiment, without a single-change claim.')
        return warnings
    # A policy-only ablation reuses the saved preprocessor rather than asking an
    # LLM to reproduce an English description character for character.
    if declared and 'preprocessing' not in declared:
        if item.preprocessing != old.get('preprocessing', []):
            item.agent_notes['preprocessing_description'] = item.preprocessing
            warnings.append('Using saved preprocessing for the ablation; alternative wording retained in agent_notes.')
        item.preprocessing = copy.deepcopy(old.get('preprocessing', []))
    for key in ('target_transform', 'weight_policy', 'prediction_policy'):
        if key not in item.model_fields_set and getattr(item, key) == type(getattr(item, key))():
            setattr(item, key, type(getattr(item, key)).model_validate(old.get(key, {})))
    if 'feature_set_id' not in declared and 'feature_set_id' not in item.model_fields_set:
        item.feature_set_id = old.get('feature_set_id')
        from .feature_contracts import FeatureSet
        item.feature_set = FeatureSet.model_validate(old['feature_set']) if old.get('feature_set') else None
    if 'model_params' not in declared:
        item.model_params = copy.deepcopy(champion['model_params'])
    item.strategy, item.search_space, item.max_trials = 'single', {}, 1
    current = item.model_dump(mode='json')
    changed = []
    for key in sorted(fields):
        if key == 'model_params':
            from .execution_resources import model_parameters
            different = model_parameters(item.model_params) != model_parameters(champion['model_params'])
        elif key == 'preprocessing' and declared and key not in declared:
            different = False
        else:
            default = type(getattr(item, key))().model_dump() if key in {
                'target_transform', 'weight_policy', 'prediction_policy'} else current[key]
            different = current[key] != old.get(key, default)
        if different:
            changed.append(key)
    item.intervention.changed_fields = changed
    if len(changed) > 1:
        item.change_kind = 'experiment'
        warnings.append('Multiple settings changed; accepted as an ordinary experiment, without a single-change claim.')
    return warnings


def infrastructure_overrides(params, search, *, native=False):
    from .execution_resources import is_thread_parameter
    reserved = {'n_jobs', 'nthread', 'num_threads', 'num_thread', 'nthreads', 'thread_count',
        'num_cpus', 'num_gpus', 'device', 'device_type', 'task_type', 'pretrained', 'presets',
        'hyperparameters', 'general_params', 'reader_params', 'estimator_list', 'time_budget',
        'time_limit', 'timeout', 'gpu_ids', 'starting_points', '__native_frozen',
        '__ensemble_frozen', 'log_file_name', 'log_type'}
    if native:
        reserved.add('max_iter')
    fixed_cpu = {'device': 'cpu', 'device_type': 'cpu', 'task_type': 'CPU'}
    return ((set(params) & reserved) - {key for key, value in fixed_cpu.items() if params.get(key) == value}
            | {key for key in search if key in reserved or is_thread_parameter(key)})


def normalize_cpu_hints(item):
    from .execution_resources import model_parameters, thread_settings
    settings = thread_settings(item.model_params)
    item.model_params = model_parameters(item.model_params)
    for index, component in enumerate(item.ensemble_components):
        params = component.get('model_params', {})
        settings.update({f'component[{index}].{key}': value for key, value in thread_settings(params).items()})
        if 'model_params' in component:
            component['model_params'] = model_parameters(params)
    return sorted(settings)
