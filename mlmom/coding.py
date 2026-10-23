"""Bounded coding graph; every source revision is preserved before verification."""
from __future__ import annotations

from .agents import GlobalAPIError

import shutil
import time
import ast
from pathlib import Path
from typing import TypedDict
from pydantic import Field
from langgraph.graph import StateGraph, START, END
from .contracts import StrictModel
from .pipeline_contracts import CandidateVersion,ValidationReport,PreprocessingBundle
from .tools import Workspace,verify_source
from .persistence import atomic_json,tree_hash
from .contracts import DatasetProfile
from .agents import LLMUnavailable


SDK_PROMPT = MODEL_PROMPT = """Implement estimator-only candidate.py for the planner assignment.
Export create_candidate(config) and load_candidate(artifact_dir). The returned object implements
fit(X,y,context), predict(X,prediction_spec) and save(artifact_dir). fit returns self; predict returns
an array in input row order. FitContext is a dataclass; use attributes, never dictionary access.
context has task_type, metric, direction, seed, cpus, memory_gb, time_limit_seconds, remaining_seconds,
classes, groups, sample_weight, validation_X/y, validation_groups/weight, work_dir.
preprocessing.py is read-only and owned by a separate role. The SDK supplies already transformed X
and context.validation_X in preprocessing_spec's declared dataframe/dense/sparse representation.
Do not import preprocessing or fit feature cleaning/encoding/imputation again. Backend reference
excerpts must be adapted to this separation. Implement the selected model directly, target/output
transforms and constraints, prediction/class ordering and portable estimator persistence.
Consume all config['model_params'], including declared search parameters. No Optuna loop inside fit:
the runner supplies each trial. Reuse validation only for stopping/selection, never ordinary fitting.
export_model_params() MUST return the COMPLETE reproducible model_params, not a patch: retain
every fixed parameter, nested ensemble config, blend weight, target/output option and flat HPO
override. Only explicitly permitted stopping can replace counts with measured selected counts.
The runner rejects incomplete exports. Smoke recreates, refits and compares effective configuration
and predictions. Expose self.model for a single estimator; ensembles MUST implement
fitted_components() returning every named fitted estimator (including stacking meta learners).
The runner inspects backend parameters and actual iterations after every fit and blocks mismatches.
stopping_policy and component_stopping_policies in the plan define whether early stopping is allowed.
context.timeouts_enabled controls time callbacks: when false do not add deadline callbacks.
CatBoost callback after_iteration returns True to CONTINUE, False to STOP. Never invert it.
Prediction kinds: value, label, probability in context.classes order, cluster, ranking_score,
anomaly_score (larger is more anomalous). SDK separately fits and persists preprocessing per fit.
Use mlmom.sdk and installed selected libraries directly; no LibraryCandidate or hidden adapters.
Candidate ensembles, voting and stacking are allowed when planned. Use OOF base predictions
for stacking meta learners. For model_params.method='stack', implement
fit_stack(X,y,context,create_preprocessor): the SDK supplies RAW X and validation_X to this hook.
Wrap every base estimator with mlmom.sdk.PreparedCandidate using create_preprocessor, fitting
fresh wrappers on each base-fold train; slice groups and sample_weight, clear validation metadata
for those folds. Refit fresh wrapped bases on all current train after OOF. predict receives RAW X
and uses the saved wrapped bases. save/load must persist those bases and meta learner. The SDK
skips its outer preprocessing only for this explicit stack hook. Ordinary fit remains estimator-only.
Implement the candidate in its own code using installed libraries; infrastructure imports
remain forbidden except mlmom.sdk. ExtraTrees and other single
estimators remain allowed. In CV, validation_X is absent: fit fixed parameters without early stopping.
CPU only; respect context resources and time. No network, pip, pretrained models, environment,
shell, credentials or input-directory reads. Use X/y/context; save estimator files to artifact_dir.
No training during import/load/predict. The SDK is installed; do not search for its source.
During repair use the existing apply_patch tool to fix the specific defect in current_files.
Worker errors include the failing source line and a full traceback artifact reference.
Return complete {"files":{"candidate.py":"...", ...}} immediately for AST/smoke verification.
Do not return preprocessing.py. Fix reported errors promptly using only the selected API.
"""

PREPROCESSING_PROMPT = '''Implement only preprocessing.py for the supplied planner assignment.
Return source and spec. Export create_preprocessor(config), returning a fresh object with
fit_transform(X, y, context) and transform(X). context is FitContext with training groups/weights,
never validation data. Fit all vocabularies, imputers, feature statistics on X only. For target
encoding produce group-safe OOF training encodings; transform uses train-fitted inference maps.
Preserve rows and order. Implement planned categories, missingness, text and feature operations.
The spec declares dataframe, dense or sparse output, feature semantics and backend constraints.
Use only installed allowed libraries and X/y/context; no files, network or environment access.
Planned train-only feature selectors may be fitted; do not train the predictive estimator or change y.
State must be cloudpickle serializable. No import-time training. Support unseen categories,
missing values, empty text, and stable feature order. The SDK owns fitting and persistence.
During repair retain previous_spec unless the output interface must change.
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
    preprocessing_spec: dict
    preprocessing_attempts: int
    preprocessing_valid: bool
    preprocessing_hash: str
    reuse_model: bool


def build_coding_graph(client, *, workspace, store, plan, plan_ref, skill_data, tools, context,
                       sandbox, budget, config, round_id, previous=None, runtime_feedback=None, preprocessing_skills=None):
    workspace = Workspace(workspace,revision_dir=store.root/'source_revisions'/round_id/plan.hypothesis.hypothesis_id,
                          writable=lambda name:name != 'preprocessing.py')
    pre_workspace = Workspace(workspace.root,revision_dir=workspace.revision_dir,
                              writable=lambda name:name == 'preprocessing.py')
    budget_key=f'{round_id}-{plan.hypothesis.hypothesis_id}'
    def coding_capacity():
        # Preserve a complete selected fit and prediction window, including
        # when previous experiments increase the final-training reserve.
        return max(0.,budget.snapshot().research_seconds-plan.time_limit_seconds-2*config.prediction_seconds)

    window=store.get('coding_budget',budget_key)
    if window is None:
        started=budget.now()
        from .time_policy import UNBOUNDED_SECONDS, timeouts_enabled
        window=dict(started=started,deadline=started+min(config.coding_seconds,coding_capacity()) if timeouts_enabled(config) else UNBOUNDED_SECONDS)
        store.put('coding_budget',budget_key,window,immutable=True)

    def remaining():
        from .time_policy import UNBOUNDED_SECONDS, timeouts_enabled
        if not timeouts_enabled(config):return UNBOUNDED_SECONDS
        return max(0.,min(window['deadline']-budget.now(),coding_capacity()))
    verification_reserve=min(config.coding_verification_seconds,
                             max(0,window['deadline']-window['started'])*.2)
    if not getattr(config,'timeouts_enabled',True):verification_reserve=0

    def generation_remaining():
        return max(0.,remaining()-verification_reserve)

    def preprocessing(state):
        attempts=state.get('preprocessing_attempts',0)+1
        original=getattr(client,'deadline',None)
        original_steps=getattr(client,'max_steps',None)
        if original_steps is not None:client.max_steps=min(original_steps,config.coding_tool_steps)
        try:
            saved=store.get('preprocessing',budget_key)
            from .persistence import file_hash
            path=workspace.root/'preprocessing.py'
            repair=bool(runtime_feedback and 'preprocessing.py' in str(runtime_feedback)) or any('preprocessing.py' in e for e in state.get('errors',[]))
            if saved and path.is_file() and file_hash(path)==saved['sha256'] and not repair:
                return dict(preprocessing_valid=True,preprocessing_spec=saved['spec'],preprocessing_hash=saved['sha256'])
            if generation_remaining()<.5:raise LLMUnavailable('No preprocessing generation time remains')
            if original is not None:client.deadline=min(original,time.time()+generation_remaining())
            bundle=client.generate('preprocessing_agent',PREPROCESSING_PROMPT,
                dict(plan=plan.model_dump(mode='json'),task=context.task.model_dump(mode='json') if hasattr(context.task,'model_dump') else vars(context.task),
                     feature_schema=schema,skills=preprocessing_skills or [],feedback=state.get('errors',[]),
                     runtime_feedback=runtime_feedback,previous_spec=saved['spec'] if saved else None,
                     previous=previous),PreprocessingBundle,None)
            tree=ast.parse(bundle.source)
            if 'create_preprocessor' not in {n.name for n in tree.body if isinstance(n,ast.FunctionDef)}:
                raise ValueError('preprocessing.py must define create_preprocessor')
            if bundle.spec.module!='preprocessing.py' or bundle.spec.factory!='create_preprocessor':
                raise ValueError('Unsupported preprocessing factory')
            pre_workspace.write_file('preprocessing.py',bundle.source+'\nMLMOM_PREPROCESSOR_V1 = True\n')
            errors=[e for e in verify_source(workspace.root).errors if e.startswith('preprocessing.py:')]
            if errors:raise ValueError('; '.join(errors))
            reuse=bool(repair and saved and saved['spec']==bundle.spec.model_dump(mode='json') and verify_source(workspace.root).valid)
            saved=dict(spec=bundle.spec.model_dump(mode='json'),sha256=file_hash(path))
            store.put('preprocessing',budget_key,saved)
            return dict(preprocessing_valid=True,preprocessing_spec=saved['spec'],preprocessing_hash=saved['sha256'],
                        preprocessing_attempts=attempts,errors=[],reuse_model=reuse,attempts=state.get('attempts',0))
        except GlobalAPIError:
            raise
        except (ValueError,SyntaxError,LLMUnavailable) as error:
            return dict(preprocessing_valid=False,preprocessing_attempts=attempts,errors=[str(error)])
        finally:
            if original is not None:client.deadline=original
            if original_steps is not None:client.max_steps=original_steps

    def source_report(root):
        return verify_source(root)
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
        return source_report(workspace.root).model_dump()
    role_tools.register('check_source',check_source,'Check syntax and allowed imports without running code.')

    def smoke_check() -> dict:
        report = source_report(workspace.root)
        if not report.valid: return report.model_dump()
        seconds = min(config.smoke_seconds,remaining()) if getattr(config,'timeouts_enabled',True) else remaining()
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
        original_steps=getattr(client,'max_steps',None)
        if original_steps is not None:client.max_steps=min(original_steps,config.coding_tool_steps)
        try:
            if generation_remaining()<.5: raise LLMUnavailable('Hypothesis code generation deadline reached; verification time reserved')
            if original_deadline is not None:
                # All API steps and their retries share the same persisted
                # deadline as source repairs and smoke/inspection workers.
                client.deadline=min(original_deadline,time.time()+generation_remaining())
            bundle = client.generate('coding_agent',MODEL_PROMPT,
                dict(state['payload'],feedback=state.get('errors',[]),runtime_feedback=runtime_feedback,
                     preprocessing_spec=state['preprocessing_spec'],
                     current_files={name:workspace.read_file(name) for name in workspace.list_files() if name.endswith('.py')}),
                CodeBundle,None if config.coding_direct_first and attempts==1 and not runtime_feedback else role_tools)
            if sum(map(len,bundle.files.values())) > 200000: raise ValueError('Generated source exceeds workspace budget')
            for name,content in bundle.files.items():
                if not name.endswith('.py'): raise ValueError('Final bundle must contain Python source only')
                workspace.write_file(name,content)
            from .persistence import file_hash
            if file_hash(workspace.root/'preprocessing.py')!=state['preprocessing_hash']:
                raise ValueError('Locked transform source was modified')
            return {'attempts':attempts,'errors':[],'warnings':[]}
        except GlobalAPIError:
            raise
        except (ValueError,LLMUnavailable) as error:
            # Tool calls may have already written a complete candidate before
            # the final JSON, API failure or step limit. Verify that source in
            # the usual subprocess rather than throwing completed work away.
            if source_report(workspace.root).valid:
                store.event('generation_interrupted',round_id,{'hypothesis_id':plan.hypothesis.hypothesis_id,'error':str(error)})
                return {'attempts':attempts,'errors':[],
                        'warnings':['Using tool-written source after interrupted generation: '+str(error)]}
            return {'attempts':attempts,'errors':[str(error)]}
        finally:
            if original_deadline is not None: client.deadline=original_deadline
            if original_steps is not None:client.max_steps=original_steps

    def verify(state):
        smoke_report=None
        smoke_report_ref=None
        report = source_report(workspace.root)
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
            seconds = min(config.smoke_seconds,remaining()) if getattr(config,'timeouts_enabled',True) else remaining()
            if seconds < .5:
                report.errors.append('No smoke-check budget remains')
            else:
                result = sandbox.run(dict(action='smoke',code_dir=str(snapshot),code_hash=digest,
                    config=plan.model_dump(mode='json'),task_type=context.task.task_type,
                    metric=context.task.primary_metric,prediction_kind=plan.prediction_kind,
                    verify_frozen=plan.strategy=='native',native_holdout=plan.strategy=='native',**smoke_schema),
                    store.root/'checks'/f'{round_id}-{plan.hypothesis.hypothesis_id}-v{version}',
                    seconds=seconds,memory_gb=plan.memory_gb,cpus=plan.cpus)
                if result['status']!='completed': report.errors.append(result.get('error') or result.get('stop_reason','Smoke failed'))
                smoke_report=result.get('execution_report');smoke_report_ref=result.get('execution_report_ref')
        report.valid = not report.errors
        candidate = CandidateVersion(candidate_id=f'{round_id}-{plan.hypothesis.hypothesis_id}-v{version}',
            hypothesis_id=plan.hypothesis.hypothesis_id,version=version,code_dir=str(snapshot),code_hash=digest,
            plan_ref=str(plan_ref),skill_versions={s['skill_id']:{'version':s['version'],'sha256':s['sha256'],'status':s['status']} for s in [*skill_data,*(preprocessing_skills or [])]},
            dependencies={k:v.version for k,v in tools.items() if v.available and v.version},based_on=plan.hypothesis.based_on,
            verification=report,preprocessing_hash=state.get('preprocessing_hash'),
            execution_report=smoke_report,execution_report_ref=smoke_report_ref)
        store.put('candidate',candidate.candidate_id,candidate,immutable=True)
        atomic_json(snapshot/'metadata.json',candidate)
        return {'candidate':candidate.model_dump(mode='json'),'valid':report.valid,'errors':report.errors}

    graph = StateGraph(CodingState)
    def preprocessing_failed(state):
        return verify(dict(state,attempts=config.max_attempts))
    graph.add_node('preprocessing_failed',preprocessing_failed)
    graph.add_node('preprocessing',preprocessing)
    graph.add_node('generate',generate); graph.add_node('verify',verify)
    graph.add_edge(START,'preprocessing')
    graph.add_conditional_edges('preprocessing',lambda s:('verify' if s.get('reuse_model') else 'generate') if s['preprocessing_valid'] else
        'finish' if s.get('preprocessing_attempts',0)>=config.max_attempts or generation_remaining()<.5 else 'preprocessing',
        {'finish':'preprocessing_failed','generate':'generate','preprocessing':'preprocessing','verify':'verify'})
    graph.add_edge('preprocessing_failed',END)
    graph.add_edge('generate','verify')
    graph.add_conditional_edges('verify',lambda s:'finish' if s['valid'] or s['attempts']>=config.max_attempts or generation_remaining()<.5 or
        (s.get('preprocessing_attempts',0)>=config.max_attempts and any('preprocessing.py' in e for e in s['errors'])) else
        'preprocessing' if any('preprocessing.py' in e for e in s['errors']) else 'generate',
        {'finish':END,'generate':'generate','preprocessing':'preprocessing'})
    return graph.compile()
