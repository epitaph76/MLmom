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
ROLE_DEFAULTS = {
    'preprocessing_agent': ('candidate_sdk', 'preprocessing', 'preprocessing_core'),
    'coding_agent': ('candidate_sdk', 'metrics'),
    'training_agent': ('metrics', 'hpo'),
    'plan_validator': ('validation', 'metrics'),
    'reconnaissance_agent': ('reconnaissance_agent', 'polars_sql'),
    **{role: (role,) for role in ('result_validator',
                                  'context_agent', 'hypothesis_generator', 'planner', 'ensemble_agent', 'feature_agent')},
}
LIGHTAUTOML_OBJECTIVES = {'regression':'rmse','binary_classification':'roc_auc','multiclass_classification':'log_loss'}
DEFINITIONS = [
    ('sklearn','scikit-learn','sklearn','1.5.2',MULTI+['clustering','anomaly_detection'],'fast',False,True,False),
    ('catboost','catboost','catboost','1.2.8',MULTI+['ranking'],'medium',False,True,True),
    ('lightgbm','lightgbm','lightgbm','4.6.0',MULTI+['ranking'],'fast',False,True,True),
    ('xgboost','xgboost','xgboost','2.1.4',MULTI+['ranking'],'medium',False,True,True),
    ('flaml','flaml','flaml','2.3.6',SUPERVISED,'medium',True,True,False),
    ('autogluon','autogluon.tabular','autogluon.tabular','1.3.1',SUPERVISED,'slow',True,False,False),
    ('lightautoml','lightautoml','lightautoml','0.4.2',SUPERVISED,'slow',True,False,False),
    ('statsmodels','statsmodels','statsmodels','0.14.0',['regression'],'fast',False,False,False),
    ('torch','torch','torch','2.5.1',MULTI,'slow',False,True,False),
]


def installed_tools() -> dict[str, MLToolSpec]:
    result = {}
    for tool, distribution, module, expected, tasks, cost, native, weights, groups in DEFINITIONS:
        try:
            version = importlib.metadata.version(distribution)
            found = importlib.util.find_spec(module) is not None
            reason = None if found else 'import not found'
            if version.split('+')[0] != expected: reason = f'version mismatch: expected {expected}, found {version}'
        except (importlib.metadata.PackageNotFoundError, ImportError, OSError, ValueError):
            found, version, reason = False, None, 'distribution not installed'
        result[tool] = MLToolSpec(tool_id=tool, distribution=distribution, import_name=module,
            version=version, expected_version=expected, available=found, reason=reason, task_types=tasks,
            prediction_kinds=['value','label','probability'] + (['ranking_score'] if 'ranking' in tasks else [])
            + (['cluster','anomaly_score'] if tool=='sklearn' else []), skill_ids=[tool],
            cost_tag=cost, native_hpo=native, supports_weights=weights, supports_groups=groups,
            supports_patience=not native, adapter='generated_candidate',
            validation_strategies=['holdout'] if native else
                ['holdout','kfold','stratified_kfold','group_kfold'])
    return result


def capability_matrix(tools, verified=()):
    """Recognition and installed execution are distinct from verified task paths."""
    rows=[]
    for task in TaskType:
        backends=[t.tool_id for t in tools.values() if task in t.task_types and t.available]
        reason=('time-series executor deferred' if task==TaskType.TIME_SERIES_FORECASTING else
                'unknown task or output contract' if task==TaskType.UNKNOWN else
                'finite numeric metric features required' if task==TaskType.CLUSTERING else
                'explicit evaluation labels required' if task==TaskType.ANOMALY_DETECTION else None)
        rows.append(dict(task_type=task.value,recognized=task!=TaskType.UNKNOWN,
            executable=bool(backends),verified=task.value in verified,backends=backends,reason=reason))
    rows.append(dict(task_type='free_text_output',recognized=False,executable=False,verified=False,
        backends=[],reason='free text output contract not implemented'))
    return rows


class SkillRegistry:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.manifests = {}
        self.paths = {}
        self.owners = {}
        for path in sorted(self.root.rglob('manifest.json')):
            manifest = SkillManifest.model_validate_json(path.read_text(encoding='utf-8'))
            if manifest.skill_id != path.parent.name or manifest.skill_id in self.manifests:
                raise ValueError(f'Invalid/duplicate skill ID: {path}')
            if not (path.parent/'SKILL.md').is_file():
                raise ValueError(f'Missing SKILL.md: {path.parent.name}')
            self.manifests[manifest.skill_id] = manifest
            self.paths[manifest.skill_id] = path.parent
            self.owners[manifest.skill_id] = path.relative_to(self.root).parts[0] if len(path.relative_to(self.root).parts)>2 else 'shared'

    def allowed(self, key, role):
        return role is None or self.owners[key] in {'shared', role}

    def list(self, *, role=None):
        if role is not None and role not in ROLE_DEFAULTS:raise ValueError(f'Unknown role: {role}')
        return [dict(s.model_dump(mode='json'), owner=self.owners[s.skill_id])
                for s in self.manifests.values() if self.allowed(s.skill_id, role)]

    def technique_catalog(self, task_type):
        """Planning metadata only; implementation instructions remain role-scoped."""
        return [dict(skill_id=s.skill_id, title=s.title, owner=self.owners[s.skill_id],
                     task_types=list(s.task_types))
                for s in self.manifests.values()
                if (self.owners[s.skill_id] == 'preprocessing_agent' or s.skill_id in {'imbalance','ensembles'})
                and s.status == 'ready' and (not s.task_types or task_type in s.task_types)]

    def validate_selection(self, ids: list[str], task_type: str, *, role=None):
        """Check metadata/dependencies without building any role's prompt."""
        if role is not None and role not in ROLE_DEFAULTS:raise ValueError(f'Unknown role: {role}')
        selected = []
        for key in dict.fromkeys(ids):
            if key not in self.manifests:raise ValueError(f'Unknown skill: {key}')
            if not self.allowed(key,role):raise ValueError(f'Skill {key} is not available to {role}')
            manifest = self.manifests[key]
            selected.append(manifest)
        return selected

    def load(self, ids: list[str], task_type: str, *, role=None, max_chars=24000,include_references=False):
        if role is not None and role not in ROLE_DEFAULTS:raise ValueError(f'Unknown role: {role}')
        automatic = ROLE_DEFAULTS.get(role, ()) if role else CORE_SKILLS
        with span('skills', {'requested_ids':ids,'automatic_ids':automatic,'task_type':task_type,'role':role}):
            loaded = self._load(ids,task_type,role=role,automatic=automatic,max_chars=max_chars,include_references=include_references)
            emit('skills.loaded', loaded, skill_ids=[item['skill_id'] for item in loaded])
            return loaded

    def _load(self, ids: list[str], task_type: str, *, role=None,automatic=CORE_SKILLS,max_chars=24000,include_references=False):
        loaded = []
        used = 0
        # Every candidate receives the same contract, even when a planner only
        # selects a backend. Shared text is counted and versioned exactly once.
        for manifest in self.validate_selection([*automatic, *ids],task_type,role=role):
            key = manifest.skill_id
            root = self.paths[key]
            text = (root/'SKILL.md').read_text(encoding='utf-8')
            if key == 'candidate_sdk':
                common=self.root/'shared'/'COMMON.md'
                if not common.exists():common=self.root/'COMMON.md'
                text = common.read_text(encoding='utf-8') + '\n' + text
            reference=root/'REFERENCE.md'
            if include_references and reference.is_file():
                text+='\n\n'+reference.read_text(encoding='utf-8')
            used += len(text)
            if used > max_chars: raise ValueError('Selected skills exceed context budget')
            digest = hashlib.sha256((root/'manifest.json').read_bytes() + text.encode()).hexdigest()
            loaded.append(dict(manifest.model_dump(mode='json'), sha256=digest, content=text))
        return loaded
