"""Runtime instruction loading, provenance, compatibility and context budget."""
from __future__ import annotations

import importlib.metadata
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from mlmom.registry import CORE_SKILLS, DEFINITIONS, SkillRegistry


ROOT = Path(__file__).resolve().parent.parent / 'skills'


class SkillTests(unittest.TestCase):
    def test_core_contract_is_automatic_ordered_and_deduplicated(self):
        registry = SkillRegistry(ROOT)
        loaded = registry.load(['sklearn', 'metrics', 'sklearn', 'candidate_sdk'], 'regression')
        self.assertEqual([s['skill_id'] for s in loaded], [*CORE_SKILLS, 'sklearn'])
        common = (ROOT / 'COMMON.md').read_text(encoding='utf-8')
        sdk = (ROOT / 'candidate_sdk' / 'SKILL.md').read_text(encoding='utf-8')
        self.assertEqual(loaded[0]['content'], common + '\n' + sdk)
        self.assertEqual(registry.load(['sklearn'], 'regression'), loaded)

    def test_budget_counts_automatic_and_shared_instructions(self):
        registry = SkillRegistry(ROOT)
        loaded = registry.load(['sklearn'], 'regression')
        used = sum(len(s['content']) for s in loaded)
        self.assertEqual(registry.load(['sklearn'], 'regression', max_chars=used), loaded)
        with self.assertRaisesRegex(ValueError, 'context budget'):
            registry.load(['sklearn'], 'regression', max_chars=used - 1)
        # Even the entire corpus fits, before task-specific filtering. No
        # optional library has to be installed to check this instruction cap.
        total = sum(len(p.read_text(encoding='utf-8')) for p in ROOT.glob('*/SKILL.md'))
        total += len((ROOT / 'COMMON.md').read_text(encoding='utf-8')) + 1
        self.assertLessEqual(total, 24000)

    def test_shared_contract_and_manifest_changes_affect_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'skills'
            shutil.copytree(ROOT, root)
            registry = SkillRegistry(root)
            before = {s['skill_id']: s['sha256'] for s in registry.load([], 'regression')}
            with (root / 'COMMON.md').open('a', encoding='utf-8') as stream:
                stream.write('\nAdditional shared constraint.\n')
            after = {s['skill_id']: s['sha256'] for s in registry.load([], 'regression')}
            self.assertNotEqual(before['candidate_sdk'], after['candidate_sdk'])
            self.assertTrue(all(before[k] == after[k] for k in CORE_SKILLS[1:]))
            manifest = root / 'validation' / 'manifest.json'
            manifest.write_text(manifest.read_text(encoding='utf-8').replace('1.0.0', '1.0.1'), encoding='utf-8')
            changed = {s['skill_id']: s['sha256'] for s in registry.load([], 'regression')}
            self.assertNotEqual(after['validation'], changed['validation'])

    def test_backend_manifests_match_adapter_tasks_and_dependencies(self):
        registry = SkillRegistry(ROOT)
        self.assertEqual(len(registry.list()), 17)
        for tool, distribution, _, _, tasks, *_ in DEFINITIONS:
            with self.subTest(tool=tool):
                manifest = registry.manifests[tool]
                self.assertEqual(set(manifest.task_types), set(tasks))
                self.assertEqual(manifest.dependencies, [distribution])
                self.assertEqual(manifest.ml_tools, [tool])
        for key, manifest in registry.manifests.items():
            with self.subTest(skill=key):
                self.assertEqual(manifest.status, 'ready')
                self.assertRegex(manifest.version, r'^\d+\.\d+\.\d+$')
                text = (ROOT / key / 'SKILL.md').read_text(encoding='utf-8')
                self.assertTrue(text.isascii(), 'Runtime instructions must remain compact English text')
                self.assertTrue(text.startswith('---\n'))
                metadata = yaml.safe_load(text.split('---\n', 2)[1])
                self.assertEqual(metadata['name'], key.replace('_', '-'))
                self.assertTrue(metadata['description'])

    def test_incompatible_unknown_and_missing_dependency_rejected(self):
        registry = SkillRegistry(ROOT)
        for ids, task in [(['statsmodels'], 'multiclass_classification'),
                          (['imbalance'], 'regression'), (['ensembles'], 'ranking')]:
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, 'incompatible'):
                registry.load(ids, task)
        with self.assertRaisesRegex(ValueError, 'Unknown skill'):
            registry.load(['missing_skill'], 'regression')
        with patch('mlmom.registry.importlib.metadata.version',
                   side_effect=importlib.metadata.PackageNotFoundError('flaml')):
            with self.assertRaises(importlib.metadata.PackageNotFoundError):
                registry.load(['flaml'], 'regression')

    def test_each_task_loads_all_compatible_cards_within_budget(self):
        registry = SkillRegistry(ROOT)
        tasks = {t for definition in DEFINITIONS for t in definition[4]}
        # This verifies instruction loading, not optional backend availability.
        with patch('mlmom.registry.importlib.metadata.version', return_value='test'):
            for task in sorted(tasks):
                ids = [s.skill_id for s in registry.manifests.values()
                       if not s.task_types or task in s.task_types]
                with self.subTest(task=task):
                    loaded = registry.load(ids, task)
                    self.assertEqual({s['skill_id'] for s in loaded}, set(ids))
                    self.assertLessEqual(sum(len(s['content']) for s in loaded), 24000)


if __name__ == '__main__':
    unittest.main()
