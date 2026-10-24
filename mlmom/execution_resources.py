"""CPU thread allocations are execution settings, separate from model behavior."""
from __future__ import annotations

import copy


THREAD_PARAMETERS = frozenset({'n_jobs', 'nthread', 'num_threads', 'num_thread',
                               'nthreads', 'thread_count', 'num_cpus'})


def is_thread_parameter(key):
    return key.rsplit('__', 1)[-1] in THREAD_PARAMETERS


def model_parameters(params):
    """Preserve all learning settings, including seeds, devices and tree counts."""
    if isinstance(params, dict):
        return {key: model_parameters(value) for key, value in params.items()
                if not is_thread_parameter(key)}
    if isinstance(params, list):
        return [model_parameters(value) for value in params]
    return copy.deepcopy(params)


def thread_settings(params, prefix=''):
    result = {}
    if isinstance(params, dict):
        for key, value in params.items():
            path = f'{prefix}.{key}' if prefix else key
            if is_thread_parameter(key):
                result[path] = value
            else:
                result.update(thread_settings(value, path))
    elif isinstance(params, list):
        for index, value in enumerate(params):
            result.update(thread_settings(value, f'{prefix}[{index}]'))
    return result


def restore_thread_settings(params, original):
    """Keep legacy resource hints outside the canonical frozen ModelSpec."""
    result = copy.deepcopy(params)
    if isinstance(result, dict) and isinstance(original, dict):
        for key, value in original.items():
            if is_thread_parameter(key):
                result[key] = copy.deepcopy(value)
            elif key in result:
                result[key] = restore_thread_settings(result[key], value)
    elif isinstance(result, list) and isinstance(original, list):
        result = [restore_thread_settings(value, original[index])
                  if index < len(original) else value for index, value in enumerate(result)]
    return result


def _allocated_params(params, cpus):
    if isinstance(params, dict):
        return {key: cpus if is_thread_parameter(key) else _allocated_params(value, cpus)
                for key, value in params.items()}
    if isinstance(params, list):
        return [_allocated_params(value, cpus) for value in params]
    return copy.deepcopy(params)


def execution_config(config, cpus):
    """Rebind existing thread hints to this worker; do not inject unsupported kwargs."""
    if type(cpus) is not int or cpus < 1:
        raise ValueError('Worker CPU allocation must be a positive integer')
    result = copy.deepcopy(config)
    if 'model_params' in result:
        result['model_params'] = _allocated_params(result['model_params'], cpus)
    for entry in result.get('ensemble_components', []):
        if 'model_params' in entry:
            entry['model_params'] = _allocated_params(entry['model_params'], cpus)
    for source in result.get('ensemble_sources', []):
        source['config'] = execution_config(source['config'], cpus)
    return result


def thread_budget_errors(params, cpus, prefix):
    errors = []
    for key, value in thread_settings(params).items():
        # None delegates to the backend default; process-level CPU limits still apply.
        if value is not None and (type(value) is not int or not 1 <= value <= cpus):
            errors.append(f'{prefix}.{key}: thread setting {value!r} exceeds or bypasses '
                          f'the worker CPU allocation 1..{cpus}')
    return errors
