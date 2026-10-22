"""Render/check registry-owned documentation without maintaining duplicate counts."""
import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mlmom.registry import SkillRegistry,ROLE_DEFAULTS

START='<!-- skill-registry:start -->'
END='<!-- skill-registry:end -->'


def render():
    registry=SkillRegistry(ROOT/'skills')
    counts=Counter(s.status for s in registry.manifests.values())
    lines=[START,f"Всего карточек: {len(registry.manifests)}; ready: {counts['ready']}; placeholder: {counts['placeholder']}.",
           '', '| Владелец | Карточки | Автоматический набор |', '|---|---|---|']
    for role in sorted(set(registry.owners.values())):
        keys=sorted(k for k,v in registry.owners.items() if v==role)
        lines.append(f"| {role} | {', '.join(keys)} | {', '.join(ROLE_DEFAULTS.get(role,())) or '-'} |")
    return '\n'.join([*lines,END])


def check(write=False):
    path=ROOT/'docs/SKILLS.md'
    text=path.read_text(encoding='utf-8')
    start=text.index(START);end=text.index(END)+len(END)
    expected=text[:start]+render()+text[end:]
    if write:path.write_text(expected,encoding='utf-8')
    elif text!=expected:raise ValueError('Skill documentation differs from registry; run scripts/check_skill_docs.py --write')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true')
    check(parser.parse_args().write)
