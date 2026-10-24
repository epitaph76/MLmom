"""Serializable fitted parameters used for refit, without estimator audits."""
from __future__ import annotations

import hashlib
import json


def config_hash(params):
    return hashlib.sha256(json.dumps(params, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def export_params(model, requested):
    """Use the candidate's optional export, otherwise the executed trial parameters."""
    export = getattr(model, 'export_model_params', None)
    values = export() if callable(export) else requested
    if not isinstance(values, dict):
        raise ValueError('export_model_params() must return a JSON object')
    return json.loads(json.dumps(values, allow_nan=False))
