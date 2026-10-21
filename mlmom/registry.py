"""Installed ML capabilities and immutable, versioned instruction packages."""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path
from .contracts import TaskType
from .pipeline_contracts import MLToolSpec, SkillManifest
from .tracing import span, emit

SUPERVISED = ['regression','binary_classification','multiclass_classification']
MULTI = SUPERVISED + ['multioutput_regression','multilabel_classification']
CORE_SKILLS = ('candidate_sdk', 'validation', 'metrics', 'preprocessing')
LIGHTAUTOML_OBJECTIVES = {'regression':'rmse','binary_classification':'roc_auc','multiclass_classification':'log_loss'}
DEFINITIONS = [
    ('sklearn','scikit-learn','sklearn','1.5.2',MULTI+['clustering','anomaly_detection'],'fast',False,True,False),
    ('catboost','catboost','catboost','1.2.8',MULTI+['ranking'],'medium',False,True,True),
    ('lightgbm','lightgbm','lightgbm','4.6.0',MULTI+['ranking'],'fast',False,True,True),
    ('xgboost','xgboost','xgboost','2.1.4',MULTI+['ranking'],'medium',False,True,True),
    ('flaml','flaml','flaml','2.3.6',SUPERVISED,'medium',True,False,False),
    ('autogluon','autogluon.tabular','autogluon.tabular','1.3.1',SUPERVISED,'slow',True,False,False),
    ('lightautoml','lightautoml','lightautoml','0.4.2',SUPERVISED,'slow',True,False,False),
    ('statsmodels','statsmodels','statsmodels','0.14.0',['regression'],'fast',False,False,False),
    ('torch','torch','torch','2.5.1',MULTI,'slow',False,False,False),
]


def installed_tools() -> dict[str, MLToolSpec]:
    result = {}
    for tool, distribution, module, expected, tasks, cost, native, weights, groups in DEFINITIONS:
        try:
            version = importlib.metadata.version(distribution)
            found = importlib.util.find_spec(module) is not None
            reason = 'adapter probe required' if found else 'import not found'
            if version.split('+')[0] != expected: reason = f'version mismatch: expected {expected}, found {version}'
        except (importlib.metadata.PackageNotFoundError, ImportError, OSError, ValueError):
            version, reason = None, 'distribution not installed'
        result[tool] = MLToolSpec(tool_id=tool, distribution=distribution, import_name=module,
            version=version, expected_version=expected, reason=reason, task_types=tasks,
            prediction_kinds=['value','label','probability'] + (['ranking_score'] if 'ranking' in tasks else [])
            + (['cluster','anomaly_score'] if tool=='sklearn' else []), skill_ids=[tool],
            cost_tag=cost, native_hpo=native, supports_weights=weights, supports_groups=groups,
            supports_patience=not native, adapter='mlmom.adapters.LibraryCandidate')
    return result


class SkillRegistry:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.manifests = {}
        for path in sorted(self.root.glob('*/manifest.json')):
            manifest = SkillManifest.model_validate_json(path.read_text(encoding='utf-8'))
            if manifest.skill_id != path.parent.name or manifest.skill_id in self.manifests:
                raise ValueError(f'Invalid/duplicate skill ID: {path}')
            if not (path.parent/'SKILL.md').is_file():
                raise ValueError(f'Missing SKILL.md: {path.parent.name}')
            self.manifests[manifest.skill_id] = manifest

    def list(self):
        return [s.model_dump(mode='json') for s in self.manifests.values()]

    def load(self, ids: list[str], task_type: str, *, max_chars=24000):
        with span('skills', {'requested_ids':ids,'automatic_ids':CORE_SKILLS,'task_type':task_type}):
            loaded = self._load(ids,task_type,max_chars=max_chars)
            emit('skills.loaded', loaded, skill_ids=[item['skill_id'] for item in loaded])
            return loaded

    def _load(self, ids: list[str], task_type: str, *, max_chars=24000):
        loaded = []
        used = 0
        # Every candidate receives the same contract, even when a planner only
        # selects a backend. Shared text is counted and versioned exactly once.
        for key in dict.fromkeys([*CORE_SKILLS, *ids]):
            if key not in self.manifests: raise ValueError(f'Unknown skill: {key}')
            manifest = self.manifests[key]
            if manifest.task_types and task_type not in manifest.task_types:
                raise ValueError(f'Skill {key} is incompatible with {task_type}')
            for distribution in manifest.dependencies:
                importlib.metadata.version(distribution)
            root = self.root/key
            text = (root/'SKILL.md').read_text(encoding='utf-8')
            if key == 'candidate_sdk':
                text = (self.root/'COMMON.md').read_text(encoding='utf-8') + '\n' + text
            used += len(text)
            if used > max_chars: raise ValueError('Selected skills exceed context budget')
            digest = hashlib.sha256((root/'manifest.json').read_bytes() + text.encode()).hexdigest()
            loaded.append(dict(manifest.model_dump(mode='json'), sha256=digest, content=text))
        return loaded
