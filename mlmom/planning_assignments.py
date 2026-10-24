"""Normalize experiment bookkeeping without changing execution safety checks."""
from __future__ import annotations

import copy


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
