"""Bounded coding graph; every source revision is preserved before verification."""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import TypedDict
from pydantic import Field
from langgraph.graph import StateGraph, START, END
from .contracts import StrictModel
from .pipeline_contracts import CandidateVersion,ValidationReport
from .tools import Workspace,verify_source
from .persistence import atomic_json,tree_hash
from .contracts import DatasetProfile
from .agents import LLMUnavailable


SDK_PROMPT = '''Implement candidate.py with create_candidate(config) and load_candidate(artifact_dir).
The returned object implements fit(X, y, context), predict(X, prediction_spec), save(artifact_dir).
fit returns self. predict returns a numpy-like array in input row order, never a file.
context contains task_type, metric, direction, seed, cpus, memory_gb, time_limit_seconds,
remaining_seconds, classes, groups, sample_weight, validation_X, validation_y, work_dir.
context is a mlmom.sdk.FitContext dataclass: read attributes, not dictionary keys.
The infrastructure creates artifact_dir before save. Once checks pass, return the final JSON promptly.
All preprocessing must be fitted within fit. No training happens at import time.
Do not implement another HPO loop in fit: the shared runner supplies each trial's parameters.
Prediction kinds: value, label, probability (columns in context.classes order), cluster,
ranking_score, anomaly_score (larger means more anomalous).
Use CPU only, no external data, pretrained models, pip, network, shell or credentials.
No read access to input directories: use X/y/context only. Save to artifact_dir only.
Use helpers from mlmom.sdk, mlmom.adapters or mlmom.ensemble when useful.
An existing functional adapter is available, so a minimal correct candidate can be:
from mlmom.adapters import LibraryCandidate, load_adapter
def create_candidate(config): return LibraryCandidate(config)
def load_candidate(artifact_dir): return load_adapter(artifact_dir)
For ensemble plans use EnsembleCandidate and load_ensemble from mlmom.ensemble.
These are real library adapters, not a substitute for implementing data-specific hypotheses.
Custom feature engineering must be implemented and persisted with the model.
Return {"files":{"candidate.py":"...", ...}}. Files must contain the final complete source.
'''


class CodeBundle(StrictModel):
    files: dict[str,str] = Field(min_length=1,max_length=20)


class CodingState(TypedDict, total=False):
    payload: dict
    attempts: int
    errors: list[str]
    valid: bool
    candidate: dict | None
    warnings: list[str]


def build_coding_graph(client, *, workspace, store, plan, plan_ref, skill_data, tools, context,
                       sandbox, budget, config, round_id, previous=None, runtime_feedback=None):
    workspace = Workspace(workspace)
    budget_key=f'{round_id}-{plan.hypothesis.hypothesis_id}'
    def coding_capacity():
        # Preserve a complete selected fit and prediction window, including
        # when previous experiments increase the final-training reserve.
        return max(0.,budget.snapshot().research_seconds-plan.time_limit_seconds-config.prediction_seconds)

    window=store.get('coding_budget',budget_key)
    if window is None:
        started=budget.now()
        window=dict(started=started,deadline=started+min(300,config.coding_seconds,coding_capacity()))
        store.put('coding_budget',budget_key,window,immutable=True)

    def remaining():
        return max(0.,min(window['deadline']-budget.now(),coding_capacity()))
    # Synthetic checks must keep the task's column names and types: generated
    # feature engineering may legitimately refer to those names explicitly.
    schema=[]
    if Path(context.profile_ref).is_file():
        profile=DatasetProfile.model_validate_json(Path(context.profile_ref).read_text(encoding='utf-8'))
        columns={c.name:c for c in profile.columns}
        schema=[{'name':name,'semantic_type':str(columns[name].semantic_type)}
                for name in context.feature_columns if name in columns]
    if not schema:schema=[{'name':name,'semantic_type':'numeric'} for name in context.feature_columns]
    smoke_schema=dict(feature_schema=schema,classes=context.classes,target_columns=context.task.target_columns)
    role_tools = workspace.tools()
    role_tools.register('list_ml_tools',lambda:[s.model_dump(mode='json') for s in tools.values()], 'Read installed ML capabilities.')
    role_tools.register('read_previous',lambda:previous or {}, 'Read the explicitly selected prior implementation.')
    role_tools.register('list_skills',lambda:skill_data,'Read selected skill manifests and contents.')

    def check_source() -> dict:
        return verify_source(workspace.root).model_dump()
    role_tools.register('check_source',check_source,'Check syntax and allowed imports without running code.')

    def smoke_check() -> dict:
        report = verify_source(workspace.root)
        if not report.valid: return report.model_dump()
        seconds = min(config.smoke_seconds,remaining())
        if seconds < .5: return {'status':'failed','error':'No verification time remains'}
        directory = store.root/'checks'/f'{round_id}-{plan.hypothesis.hypothesis_id}-{time.time_ns()}'
        return sandbox.run(dict(action='smoke',code_dir=str(workspace.root),config=plan.model_dump(mode='json'),
             task_type=context.task.task_type,metric=context.task.primary_metric,prediction_kind=plan.prediction_kind,**smoke_schema),
             directory,seconds=seconds,memory_gb=min(plan.memory_gb,budget.memory_gb),cpus=min(plan.cpus,budget.cpus))
    role_tools.register('run_smoke_check',smoke_check,'Run a bounded synthetic fit/predict/save/load check. No real data.')

    def inspect_api(module: str,symbol: str) -> dict:
        seconds=min(15.,remaining())
        if seconds<.5:raise ValueError('No API inspection budget remains')
        return sandbox.run(dict(action='inspect_api',module=module,symbol=symbol),
            store.root/'checks'/f'api-{time.time_ns()}',seconds=seconds,memory_gb=plan.memory_gb,cpus=plan.cpus)
    role_tools.register('inspect_api',inspect_api,'Inspect an installed library class/function signature and documentation in a bounded subprocess.')

    def generate(state):
        attempts = state.get('attempts',0)+1
        original_deadline=getattr(client,'deadline',None)
        try:
            if remaining()<.5: raise LLMUnavailable('Hypothesis coding deadline reached')
            if original_deadline is not None:
                # All API steps and their retries share the same persisted
                # deadline as source repairs and smoke/inspection workers.
                client.deadline=min(original_deadline,time.time()+remaining())
            bundle = client.generate('coding_agent',SDK_PROMPT,
                dict(state['payload'],feedback=state.get('errors',[]),runtime_feedback=runtime_feedback,
                     current_files={name:workspace.read_file(name) for name in workspace.list_files() if name.endswith('.py')}),
                CodeBundle,role_tools)
            if sum(map(len,bundle.files.values())) > 200000: raise ValueError('Generated source exceeds workspace budget')
            for name,content in bundle.files.items():
                if not name.endswith('.py'): raise ValueError('Final bundle must contain Python source only')
                workspace.write_file(name,content)
            return {'attempts':attempts,'errors':[],'warnings':[]}
        except (ValueError,LLMUnavailable) as error:
            # Tool calls may have already written a complete candidate before
            # the final JSON, API failure or step limit. Verify that source in
            # the usual subprocess rather than throwing completed work away.
            if verify_source(workspace.root).valid:
                store.event('generation_interrupted',round_id,{'hypothesis_id':plan.hypothesis.hypothesis_id,'error':str(error)})
                return {'attempts':attempts,'errors':[],
                        'warnings':['Using tool-written source after interrupted generation: '+str(error)]}
            return {'attempts':attempts,'errors':[str(error)]}
        finally:
            if original_deadline is not None: client.deadline=original_deadline

    def verify(state):
        report = verify_source(workspace.root)
        report.warnings.extend(state.get('warnings',[]))
        report.errors.extend(state.get('errors',[])); report.valid = not report.errors
        parent = store.root/'candidates'/round_id/plan.hypothesis.hypothesis_id
        parent.mkdir(parents=True,exist_ok=True)
        version = max([int(p.name[1:]) for p in parent.glob('v*') if p.name[1:].isdigit()] or [0])+1
        snapshot = parent/f'v{version}'
        snapshot.mkdir()
        for source in workspace.root.rglob('*.py'):
            dest = snapshot/source.relative_to(workspace.root); dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,dest)
        digest = tree_hash(snapshot)
        if report.valid:
            seconds = min(config.smoke_seconds,remaining())
            if seconds < .5:
                report.errors.append('No smoke-check budget remains')
            else:
                result = sandbox.run(dict(action='smoke',code_dir=str(snapshot),code_hash=digest,
                    config=plan.model_dump(mode='json'),task_type=context.task.task_type,
                    metric=context.task.primary_metric,prediction_kind=plan.prediction_kind,**smoke_schema),
                    store.root/'checks'/f'{round_id}-{plan.hypothesis.hypothesis_id}-v{version}',
                    seconds=seconds,memory_gb=plan.memory_gb,cpus=plan.cpus)
                if result['status']!='completed': report.errors.append(result.get('error') or result.get('stop_reason','Smoke failed'))
        report.valid = not report.errors
        candidate = CandidateVersion(candidate_id=f'{round_id}-{plan.hypothesis.hypothesis_id}-v{version}',
            hypothesis_id=plan.hypothesis.hypothesis_id,version=version,code_dir=str(snapshot),code_hash=digest,
            plan_ref=str(plan_ref),skill_versions={s['skill_id']:{'version':s['version'],'sha256':s['sha256'],'status':s['status']} for s in skill_data},
            dependencies={k:v.version for k,v in tools.items() if v.available and v.version},based_on=plan.hypothesis.based_on,
            verification=report)
        store.put('candidate',candidate.candidate_id,candidate,immutable=True)
        atomic_json(snapshot/'metadata.json',candidate)
        return {'candidate':candidate.model_dump(mode='json'),'valid':report.valid,'errors':report.errors}

    graph = StateGraph(CodingState)
    graph.add_node('generate',generate); graph.add_node('verify',verify)
    graph.add_edge(START,'generate'); graph.add_edge('generate','verify')
    graph.add_conditional_edges('verify',lambda s:'finish' if s['valid'] or s['attempts']>=config.max_attempts or remaining()<.5 else 'generate',
                                {'finish':END,'generate':'generate'})
    return graph.compile()
