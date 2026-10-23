"""Lossless requirement/ID indexes with evidence supplied once, on demand."""
from __future__ import annotations

import copy
import json


def previous_brief_summary(brief):
    if not brief:
        return None
    return dict(task_summary=brief.get('task_summary'), input_hash=brief.get('input_hash'),
        requirements=brief.get('requirements'),
        assumptions=brief.get('assumptions', []),
        selected_findings=[dict(finding_id=item['finding_id'], priority=item['priority'])
                           for item in brief.get('ranked_findings', [])],
        input_ref=brief.get('input_ref'))


def observation_index(reports):
    index={}
    for key, report in reports.items():
        for item in report.get('observations', []):
            index[item['observation_id']]=dict(observation_id=item['observation_id'],
                source_ref=item.get('source_ref'), report_ref='reconnaissance:'+key)
    return list(index.values())


def compact_planning_payload(payload):
    value=copy.deepcopy(payload)
    recon=value.get('reconnaissance')
    if recon:
        value['observation_index']=observation_index({'current':recon})
        value['reconnaissance']={key:recon.get(key) for key in ('status','reason','unfinished_checks')}
        value['reconnaissance']['evidence_available_via']='read_evidence'
    brief=value.get('planning_brief')
    if brief:
        critical={item['finding_id'] for item in brief.get('mandatory_findings', [])}
        selected=[]
        for item in brief.get('ranked_findings', [])[:24]:
            if item['finding_id'] in critical:
                item.pop('fact', None)  # already attached, without truncation, in mandatory_findings
            selected.append(item)
        brief['ranked_findings']=selected
        value['dataset_profile']={'available_via':'read_evidence', 'evidence_ref':'profile'}
    return value


def paged_evidence(values):
    from .tools import ToolSet
    def read_evidence(reference: str, offset: int=0, chars: int=8000):
        if reference not in values:
            raise ValueError('Unknown evidence reference; choose a supplied evidence_refs value')
        if offset<0 or not 1<=chars<=8000:
            raise ValueError('Use a non-negative offset and 1..8000 characters')
        value=values[reference]
        text=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,allow_nan=False)
        return dict(reference=reference,text=text[offset:offset+chars],offset=offset,total_chars=len(text),
                    truncated=offset+chars<len(text))
    tools=ToolSet()
    tools.register('read_evidence',read_evidence,'Read a page of an explicitly supplied source/evidence reference.')
    return tools
