"""Human explanations are retained separately from executable agent assignments."""
from __future__ import annotations

from typing import Any
from pydantic import Field, model_validator
from .contracts import StrictModel


class AgentNarrative(StrictModel):
    agent_notes: dict[str, Any] = Field(default_factory=dict,
        description='Free-form explanation and annotations; never executed as configuration.')

    @model_validator(mode='before')
    @classmethod
    def retain_explanations(cls, value):
        if not isinstance(value, dict):
            return value
        value = dict(value)
        descriptive = {'analysis', 'reasoning', 'reason', 'explanation', 'notes', 'comments',
            'rationale', 'assumptions', 'recommendations', 'warnings', 'success_conditions',
            'rejection_conditions', 'intervention', 'change_kind'}
        extras = {key: value.pop(key) for key in list(value)
                  if key not in cls.model_fields and key in descriptive}
        if extras:
            notes = value.get('agent_notes', {})
            if not isinstance(notes, dict):
                return dict(value, **extras)  # Let the declared type reject malformed notes.
            value['agent_notes'] = dict(notes, **extras)
        # Length targets for prose are presentation limits, not execution gates.
        # Retain the complete explanation for downstream agents and diagnostics.
        for key in ('rationale', 'description', 'explanation', 'task_summary'):
            field = cls.model_fields.get(key)
            if field is None or not isinstance(value.get(key), str):
                continue
            limit = next((m.max_length for m in field.metadata if hasattr(m, 'max_length')), None)
            if limit is not None and len(value[key]) > limit:
                notes = value.get('agent_notes', {})
                if isinstance(notes, dict):
                    value['agent_notes'] = dict(notes, **{key + '_full': value[key]})
                    value[key] = value[key][:limit]
        return value
