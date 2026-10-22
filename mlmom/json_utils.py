"""Shared structured-response parsing, independent of any agent role."""
from __future__ import annotations

import json
from typing import Any


def extract_json(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith('```'):
        lines = candidate.splitlines()
        if lines and lines[0].strip().lower() in {'```', '```json'}:
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        candidate = '\n'.join(lines).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find('{'), candidate.rfind('}')
        if start < 0 or end <= start:
            raise
        value = json.loads(candidate[start:end+1])
    if not isinstance(value, dict):
        raise ValueError('LLM output must be one JSON object')
    return value
