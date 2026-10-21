"""End-to-end LangGraph orchestration with process-isolated experiment workers."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import TypedDict
import pandas as pd
from langgraph.graph import StateGraph,START,END

from .agents import RoleClient,LLMUnavailable
from .checkpoint import SQLiteCheckpointer
from .coding import build_coding_graph
from .contracts import DatasetProfile,TaskDefinition
from .data_inspector import build_llm_context
from .evaluation import evaluate,better,prediction_kind
from .inputs import resolve_inputs,read_table,features,targets,validate_tables
from .llm_client import LLMSettings
from .persistence import ExperimentStore,atomic_json,file_hash,tree_hash
from .pipeline_contracts import (RunConfig,TaskContext,HypothesisPlan,CandidateVersion,Champion,
    ExperimentRun,EvaluationResult,SelectedPreviousResults,SplitSpec,SubmissionArtifact)
from .planning import build_planning_graph
from .registry import SkillRegistry,installed_tools
from .resources import Budget,SandboxManager,terminate_tree
from .splitting import make_split,group_keys
from .submission import publish_submission
from .task_graph import analyze_task
from .tools import ToolSet
from .tracing import TraceSink, bind, emit, span


class RunState(TypedDict, total=False):
    context: dict
    round_number: int
    plans: list[dict]
    champion: dict | None
    selected: dict
    stale: int
    stop_reason: str | None
    round_results: list[dict]
    incumbent_succeeded: bool
    promoted: bool
    submission: dict | None
    deferred_hypotheses: dict | None


class DeferredTask(RuntimeError):
    pass


class Pipeline:
    def __init__(self, task_path: Path, config: RunConfig, *, run_dir: Path | None=None, client=None, resume=False, started=None):
        self.config = config
        settings=None
        if client is None:
            from dataclasses import replace
            settings=LLMSettings.from_platform_env() if config.profile=='platform' else LLMSettings.from_env()
            settings=replace(settings,timeout_seconds=config.llm_timeout_seconds)
        self.files = resolve_inputs(task_path)
        self.run_dir = (run_dir or Path(__file__).resolve().parent.parent/'artifacts'/f'{Path(self.files.root).name}-{uuid.uuid4().hex[:10]}').resolve()
        self.run_dir.mkdir(parents=True,exist_ok=True)
        session_path = self.run_dir/'session.json'
        if session_path.exists():
            if not resume: raise ValueError('Run already exists; use --resume')
            session = json.loads(session_path.read_text(encoding='utf-8'))
            if session['task'] != self.files.task or session['config'] != config.model_dump(mode='json'):
                raise ValueError('Resume requires the same task and configuration')
            if session['input_hashes'] != self.input_hashes(): raise ValueError('Task inputs changed since this run')
        else:
            if resume: raise ValueError('Cannot resume a run without session.json')
            session = dict(run_id=uuid.uuid4().hex,task=self.files.task,started=started or time.time(),config=config.model_dump(mode='json'),input_hashes=self.input_hashes())
            atomic_json(session_path,session)
        if 'run_id' not in session:
            session['run_id']=uuid.uuid4().hex
            atomic_json(session_path,session)
        self.trace = TraceSink(self.run_dir,session['run_id'],
            task_id=hashlib.sha256(self.files.task.encode()).hexdigest()[:16],
            secrets=[settings.api_key] if settings else [])
        self.budget = Budget(config,session['started'])
        self.store = ExperimentStore(self.run_dir)
        saved_budget=self.store.get('budget','current')
        if saved_budget:
            self.budget.reserve=max(self.budget.reserve,saved_budget['final_reserve_seconds'])
        self.sandbox = SandboxManager(budget=self.budget,max_workers=config.max_workers)
        self.skills = SkillRegistry(Path(__file__).resolve().parent.parent/'skills')
        self.tools = installed_tools()
        self.context = None; self.frame = None; self.test = None
        self.last_submission = self.store.get('submission','current')
        self.checkpointer = SQLiteCheckpointer(str(self.run_dir/'checkpoints.sqlite3'))
        if client is None:
            client = RoleClient(settings,deadline=self.budget.deadline-self.budget.reserve,max_steps=config.agent_steps,
                usage_callback=lambda usage:self.store.event('llm_usage','run',usage),
                request_callback=lambda event:self.store.event('llm_request','run',event))
        self.client = client
        self.role_tools = ToolSet()
        self.role_tools.register('list_ml_tools',lambda:[s.model_dump(mode='json') for s in self.tools.values()], 'List installed, probed ML tools.')
        self.role_tools.register('list_skills',self.skills.list,'List registered instruction packages.')
        self.role_tools.register('read_skills',self.read_skills,'Read compatible skill instructions and hashes, including the automatic core contract.')
        self.role_tools.register('read_history',self.read_history,'Read bounded structured experiment history. No raw rows.')

    def read_skills(self, skill_ids: list[str]) -> list[dict]:
        if self.context is None:raise ValueError('Skills require a validated task context')
        return self.skills.load(skill_ids,self.context.task.task_type)

    def input_hashes(self):
        paths = [self.files.task,self.files.train,self.files.test,*self.files.dictionaries]
        if self.files.sample_submission: paths.append(self.files.sample_submission)
        if self.files.manifest: paths.append(self.files.manifest)
        return {path:file_hash(Path(path)) for path in paths}

    def read_history(self, kind: str='evaluation', limit: int=10) -> list[dict]:
        if kind not in {'evaluation','experiment','plan','candidate'}: raise ValueError('History kind is not exposed')
        records = self.store.list(kind,limit=min(max(limit,1),20))
        if kind=='candidate':
            records = [{k:v for k,v in r.items() if k!='code_dir'} for r in records]
        return records

    def restore(self,state):
        if self.context is None:
            self.context = TaskContext.model_validate(state['context'])
            validate_tables(self.context)
            self.frame = read_table(self.files.train,table_format=self.context.table_formats.get(self.files.train))
        return self.context

    def preflight_tools(self, *, advanced=False, requested=None):
        for name,spec in self.tools.items():
            if spec.reason != 'adapter probe required': continue
            if requested is not None and name not in requested: continue
            if not advanced and (spec.native_hpo or spec.cost_tag=='slow'): continue
            saved = self.store.get('capability',name)
            if saved:
                spec.available=saved['available'];spec.reason=saved['reason'];continue
            if self.budget.snapshot().research_seconds < 1: break
            result = self.sandbox.run(dict(action='probe',config={'tool_id':name,'model_params':{}},task_type='regression'),
                self.run_dir/'probes'/name,seconds=min(self.config.smoke_seconds,self.budget.snapshot().research_seconds),
                memory_gb=self.budget.memory_gb,cpus=min(2,self.budget.cpus))
            spec.available = result['status']=='completed'
            spec.reason = None if spec.available else (result.get('error') or result.get('stop_reason','Probe failed'))[-2000:]
            self.store.put('capability',name,spec)

    def inspect(self,state):
        existing = self.store.get('context','task')
        if existing:
            result = dict(context=existing,round_number=1,champion=None,selected={},stale=0,stop_reason=None)
            self.restore(result);return result
        result = self.sandbox.run(dict(action='inspect',files=self.files.model_dump()),self.run_dir/'inspection',
            seconds=max(.1,min(300,self.budget.snapshot().research_seconds)),memory_gb=self.budget.memory_gb,cpus=self.budget.cpus)
        if result['status']!='completed': raise ValueError('Data inspection failed: '+result.get('error','timeout'))
        profile = DatasetProfile.model_validate_json(Path(result['profile_ref']).read_text(encoding='utf-8'))
        atomic_json(self.run_dir/'llm_context.json',build_llm_context(profile))
        from .llm_report import build_llm_markdown
        (self.run_dir/'llm_report.md').write_text(build_llm_markdown(profile),encoding='utf-8')
        task_result = analyze_task(profile,self.client,max_attempts=self.config.max_attempts)
        atomic_json(self.run_dir/'task.json',task_result)
        if task_result.status!='validated': raise ValueError('Task validation failed: '+task_result.validation.model_dump_json())
        task = task_result.task
        if task.time_column or task.task_type=='time_series_forecasting': raise DeferredTask('Time-aware execution is deferred')
        if task.task_type=='unknown': raise ValueError('Unknown task type')
        if task.validation_strategy=='time_series_split': raise DeferredTask('Time-aware execution is deferred')
        if task.task_type=='ranking' and not task.group_columns: raise ValueError('Ranking requires explicit query groups')
        if task.task_type=='anomaly_detection' and not task.target_columns:
            raise ValueError('Anomaly detection needs explicit evaluation labels for the requested metric')
        fmt = task.result_format
        if not fmt.file_name or not fmt.columns or not fmt.prediction_columns:
            raise ValueError('Task must explicitly define the output filename, columns and prediction columns')
        excluded = set(task.excluded_feature_columns+task.target_columns+task.id_columns+task.group_columns)
        if task.weight_column: excluded.add(task.weight_column)
        frame = read_table(self.files.train,table_format=profile.table_formats.get(self.files.train))
        selected_features = [c for c in frame if c not in excluded]
        if not selected_features: raise ValueError('No usable features')
        classes = []
        if task.task_type in {'binary_classification','multiclass_classification','anomaly_detection'}:
            classes = sorted(frame[task.target_columns[0]].dropna().unique().tolist())
        if fmt.class_order:
            mapping={str(label):label for label in classes}
            order=list(map(str,fmt.class_order))
            single_probability=((self.config.submission_kind or fmt.prediction_kind)=='probability' and len(fmt.prediction_columns)==1 and len(order)==1)
            if len(set(order))!=len(order) or not set(order)<=set(mapping) or (not single_probability and set(order)!=set(mapping)):
                raise ValueError('Output class_order does not match observed classes')
        submission_kind = self.config.submission_kind or fmt.prediction_kind or ('label' if 'classification' in task.task_type else
            'cluster' if task.task_type=='clustering' else 'anomaly_score' if task.task_type=='anomaly_detection' else
            'ranking_score' if task.task_type=='ranking' else 'value')
        context = TaskContext(task_id=hashlib.sha256((file_hash(Path(self.files.train))+Path(self.files.task).read_text(encoding='utf-8')).encode()).hexdigest()[:16],
            task=task,profile_ref=result['profile_ref'],files=self.files,feature_columns=selected_features,classes=classes,
            dataset_hash=file_hash(Path(self.files.train)),metric_options=task.metric_options|self.config.metric_options,
            submission_kind=submission_kind,table_formats=profile.table_formats)
        validate_tables(context)
        self.store.put('context','task',context,immutable=True)
        self.context=context;self.frame=frame
        return dict(context=context.model_dump(mode='json'),round_number=1,champion=None,selected={},stale=0,stop_reason=None)

    def planning_payload(self,state):
        context = self.restore(state)
        profile = DatasetProfile.model_validate_json(Path(context.profile_ref).read_text(encoding='utf-8'))
        payload = dict(round_number=state['round_number'],task=context.task.model_dump(mode='json'),
            dataset_profile=build_llm_context(profile),ml_tools=[t.model_dump(mode='json') for t in self.tools.values()],
            skills=self.skills.list(),budget=self.budget.snapshot().model_dump(),previous=state.get('selected',{}),
            champion=state.get('champion'),defaults=dict(max_trials=self.config.max_trials,patience_trials=self.config.patience_trials,
                experiment_seconds=self.config.experiment_seconds,first_submission_seconds=self.config.first_submission_seconds))
        # Never put model artifact paths or the full run archive into a prompt.
        if payload['champion']:
            champ = Champion.model_validate(payload['champion'])
            payload['champion'] = dict(candidate_id=champ.candidate.candidate_id,plan=champ.plan.model_dump(mode='json'),evaluation=champ.evaluation.model_dump())
        return payload

    def planning_call(self,state, *, mode='all', hypotheses=None):
        context = self.restore(state)
        key=f"r{state['round_number']}-{mode}"
        cached=self.store.get('planning',key)
        if isinstance(cached,dict) and cached.get('valid'):
            emit('planning.reused',{'key':key})
            return cached
        snapshot=self.budget.snapshot()
        advanced = (state['round_number']>1 and bool(self.last_submission) and
                    snapshot.research_seconds>self.config.coding_seconds+self.config.prediction_seconds+1)
        lightweight = not advanced
        if isinstance(self.client,RoleClient):self.client.deadline=self.budget.deadline-self.budget.reserve
        self.preflight_tools()
        payload=self.planning_payload(state)
        snapshot=self.budget.snapshot()
        if advanced:
            snapshot=snapshot.model_copy(update={'research_seconds':max(0.,snapshot.research_seconds-
                self.config.coding_seconds-self.config.prediction_seconds)})
        payload['budget']=snapshot.model_dump()
        emit('planning.policy',dict(lightweight=lightweight,mode=mode,budget=payload['budget']))
        graph = build_planning_graph(self.client,context,self.tools,self.skills,snapshot,
            self.config.max_attempts,self.config.max_candidates,self.role_tools,
            known_candidates={c['candidate_id'] for c in self.store.list('candidate')},
            lightweight=lightweight,initial_experiment_seconds=self.config.initial_experiment_seconds,planning_mode=mode,
            probe_tool=self.probe_ml_tool)
        initial={'payload':payload}
        if hypotheses is not None: initial['hypotheses']=hypotheses
        with span('planning',round_id=f"r{state['round_number']}"):
            output=graph.invoke(initial,{'recursion_limit':100})
        self.store.put('planning',key,output)
        return output

    def probe_ml_tool(self, tool_id):
        self.preflight_tools(advanced=True,requested={tool_id})

    def plan(self,state):
        if self.budget.snapshot().research_seconds <= .5: return {'plans':[],'stop_reason':'research_deadline'}
        try:
            first=state['round_number']==1
            output=self.planning_call(state,mode='baseline' if first else 'all')
            if not output['valid']: return {'plans':[],'stop_reason':'planning_failed'}
            return {'plans':output['plans'],'stop_reason':None,
                    'deferred_hypotheses':output.get('hypotheses') if first else None}
        except (LLMUnavailable,ValueError) as error:
            self.store.event('llm_unavailable','planning',{'error':str(error)})
            return {'plans':[],'stop_reason':'llm_unavailable'}

    def prepare(self,plan,round_id,*,feedback=None, prior=None):
        with bind(hypothesis_id=plan.hypothesis.hypothesis_id,round_id=round_id), span('coding'):
            return self._prepare(plan,round_id,feedback=feedback,prior=prior)

    def _prepare(self,plan,round_id,*,feedback=None, prior=None):
        if isinstance(self.client,RoleClient):self.client.deadline=self.budget.deadline-self.budget.reserve
        plan_ref = self.run_dir/'plans'/f'{round_id}-{plan.hypothesis.hypothesis_id}.json'
        if not plan_ref.exists(): atomic_json(plan_ref,plan)
        self.store.put('plan',f'{round_id}-{plan.hypothesis.hypothesis_id}',plan,immutable=True)
        if feedback is None:
            ready = [c for c in self.store.list('candidate') if c['plan_ref']==str(plan_ref) and c['verification']['valid']]
            if ready:
                candidate = CandidateVersion.model_validate(ready[-1])
                if tree_hash(Path(candidate.code_dir))!=candidate.code_hash: raise ValueError('Stored candidate was modified')
                return candidate
        previous = None
        if plan.hypothesis.based_on or prior:
            previous_candidate = prior or self.store.get('candidate',plan.hypothesis.based_on)
            if previous_candidate:
                if isinstance(previous_candidate,CandidateVersion): previous_candidate=previous_candidate.model_dump()
                source = Path(previous_candidate['code_dir'])
                previous = {'candidate_id':previous_candidate['candidate_id'],'files':{p.relative_to(source).as_posix():p.read_text(encoding='utf-8') for p in source.rglob('*.py')}}
        skills = self.skills.load(plan.skill_ids,self.context.task.task_type)
        workspace = self.run_dir/'workspaces'/round_id/plan.hypothesis.hypothesis_id
        graph = build_coding_graph(self.client,workspace=workspace,store=self.store,plan=plan,plan_ref=plan_ref,
            skill_data=skills,tools=self.tools,context=self.context,sandbox=self.sandbox,budget=self.budget,
            config=self.config,round_id=round_id,previous=previous,runtime_feedback=feedback)
        output = graph.invoke({'payload':dict(plan=plan.model_dump(mode='json'),task=self.context.task.model_dump(mode='json'),
            features=self.context.feature_columns,classes=self.context.classes,skills=skills,previous=previous),
            'attempts':0,'errors':[]},{'recursion_limit':100})
        if output.get('candidate') and output['valid']: return CandidateVersion.model_validate(output['candidate'])
        self.store.event('coding_failed',plan.hypothesis.hypothesis_id,{'errors':output.get('errors',[])})
        return None

    def sanitized_context(self,train_path):
        value = self.context.model_dump(mode='json')
        value['profile_ref'] = ''
        value['table_formats']={str(train_path):self.context.table_formats[self.files.train].model_dump()} if (
            str(train_path)==self.files.train and self.files.train in self.context.table_formats) else {}
        value['files'] = dict(value['files'],root=str(Path(train_path).parent),train=str(train_path),test='',task='',dictionaries=[],sample_submission=None,manifest=None)
        return value

    def _prediction(self,candidate,run,X,kinds,*,seconds=None):
        directory = self.run_dir/'predictions'/f'{run.experiment_id}-{uuid.uuid4().hex[:8]}'
        directory.mkdir(parents=True)
        features_path = directory/'features.csv'; X.to_csv(features_path,index=False)
        remaining = self.budget.snapshot().remaining_seconds
        with bind(candidate_id=candidate.candidate_id,experiment_id=run.experiment_id):
            result = self.sandbox.run(dict(action='predict',code_dir=candidate.code_dir,code_hash=candidate.code_hash,
                model_artifact=run.model_artifact,features_path=str(features_path),kinds=kinds),directory,
                seconds=max(.1,min(seconds or self.config.prediction_seconds,remaining-1)),
                memory_gb=min(self.budget.memory_gb,run.resource_memory_gb),cpus=min(self.budget.cpus,run.resource_cpus))
        if result['status']!='completed': raise ValueError('Inference failed: '+result.get('error','deadline'))
        return {k:read_table(v).to_numpy() for k,v in result['predictions'].items()}

    def publish(self,champion):
        if self.budget.snapshot().remaining_seconds < 1: return
        if self.test is None: self.test=read_table(self.files.test,table_format=self.context.table_formats.get(self.files.test))
        try:
            p = self._prediction(champion.candidate,champion.run,features(self.test,self.context),[self.context.submission_kind])
            artifact = publish_submission(self.context,self.test,p[self.context.submission_kind],champion.run,champion.candidate)
            self.store.put('submission','current',artifact)
            self.store.put('submission',f'{artifact.experiment_id}-{artifact.sha256[:8]}',artifact,immutable=True)
            self.last_submission=artifact.model_dump()
            elapsed=self.budget.now()-self.budget.started
            first=self.store.get('milestone','first_submission')
            if first is None:
                first=dict(elapsed_seconds=elapsed,target_seconds=self.config.first_submission_seconds,
                           target_met=elapsed<=self.config.first_submission_seconds)
                self.store.put('milestone','first_submission',first,immutable=True)
                emit('submission.first',first)
            emit('submission.published',artifact.model_dump(),experiment_id=champion.run.experiment_id,
                 candidate_id=champion.candidate.candidate_id)
        except Exception as error:
            self.store.event('submission_failed',champion.run.experiment_id,{'error':str(error)})
            emit('submission.failed',{'error':str(error)},experiment_id=champion.run.experiment_id)

    def execute(self,state):
        context = self.restore(state)
        round_id = f"r{state['round_number']}"
        cached = self.store.get('round_result',round_id)
        if cached: return cached
        previous = Champion.model_validate(state['champion']) if state.get('champion') else None
        saved_split = self.store.get('split',round_id)
        split = SplitSpec.model_validate(saved_split) if saved_split else make_split(self.frame,context,round_id,self.config.holdout_fraction,self.config.seed+state['round_number'])
        self.store.put('split',round_id,split,immutable=True)
        directory = self.run_dir/'splits'/round_id; directory.mkdir(parents=True,exist_ok=True)
        atomic_json(directory/'split.json',split)
        train_path = directory/'train.csv'
        if not train_path.exists(): self.frame.iloc[split.train_indices].to_csv(train_path,index=False)
        holdout = self.frame.iloc[split.holdout_indices]
        results = []
        incumbent_ok = previous is None
        champion = previous
        promoted = False

        def finished(candidate,plan,run,result):
            nonlocal incumbent_ok,champion,promoted
            if result.get('status')!='completed':
                progress = Path(self.run_dir/'runs'/run.experiment_id/'progress.json')
                if result.get('stop_reason')=='time_limit' and progress.exists():
                    result = dict(json.loads(progress.read_text(encoding='utf-8')),stop_reason='time_limit_best_model',
                                  duration_seconds=result['duration_seconds'],peak_memory_gb=result['peak_memory_gb'])
            run = ExperimentRun.model_validate(dict(run.model_dump(),**{k:v for k,v in result.items() if k in ExperimentRun.model_fields}))
            try:
                if run.status!='completed': raise ValueError(run.error or run.stop_reason or 'Training failed')
                if run.role!='evaluation' or run.split_id!=split.split_id: raise ValueError('Incomparable evaluation artifact')
                saved_eval = self.store.get('evaluation',run.experiment_id)
                if saved_eval: evaluation = EvaluationResult.model_validate(saved_eval)
                else:
                    metric_kind = prediction_kind(context.task.primary_metric,context.task.task_type)
                    label_kind = 'label' if 'classification' in context.task.task_type else metric_kind
                    p = self._prediction(candidate,run,features(holdout,context),[metric_kind,label_kind])
                    unbox = lambda a:a[:,0] if a.shape[1]==1 else a
                    evaluation = evaluate(run.experiment_id,split.split_id,targets(holdout,context),unbox(p[metric_kind]),context,
                        X=features(holdout,context),groups=group_keys(holdout,context.task.group_columns),label_prediction=unbox(p[label_kind]),
                        weights=holdout[context.task.weight_column].to_numpy() if context.task.weight_column else None)
                    self.store.put('evaluation',run.experiment_id,evaluation,immutable=True)
                emit('evaluation.completed',evaluation.model_dump(),candidate_id=candidate.candidate_id,experiment_id=run.experiment_id)
                entry = dict(candidate=candidate.model_dump(mode='json'),plan=plan.model_dump(mode='json'),run=run.model_dump(mode='json'),evaluation=evaluation.model_dump())
                results.append(entry)
                if run.incumbent: incumbent_ok=True
                eligible = [Champion.model_validate(r) for r in results]
                if incumbent_ok:
                    current_incumbent = next((r for r in eligible if r.run.incumbent),None)
                    winner = current_incumbent or eligible[0]
                    for item in eligible:
                        if better(item.evaluation.score,winner.evaluation.score,context.task.metric_direction): winner=item
                    if previous is None or current_incumbent is not None:
                        changed = champion is None or champion.run.experiment_id != winner.run.experiment_id
                        champion=winner
                        promoted = previous is not None and not winner.run.incumbent
                        if changed:
                            emit('champion.selected',dict(score=winner.evaluation.score,incumbent=winner.run.incumbent),
                                 candidate_id=winner.candidate.candidate_id,experiment_id=winner.run.experiment_id)
                            self.publish(winner)
                old_reserve=self.budget.reserve
                self.budget.reserve=max(self.budget.reserve,min(self.config.total_seconds*.5,
                    run.duration_seconds/(1-self.config.holdout_fraction)*1.5+self.config.prediction_seconds))
                if self.budget.reserve!=old_reserve:
                    self.store.put('budget','current',{'final_reserve_seconds':self.budget.reserve})
                    emit('budget.reserve_updated',{'final_reserve_seconds':self.budget.reserve})
            except Exception as error:
                run.status='failed';run.error=str(error)
                self.store.event('experiment_failed',run.experiment_id,{'error':str(error)})
            self.store.put('experiment',run.experiment_id,run)
            return run

        def process_jobs(entries):
            pending=sorted(entries,key=lambda e:(not e[2],{'fast':0,'medium':1,'slow':2}[e[1].cost_tag])); active=[]; repair_queue=[]
            while pending or active or repair_queue:
                if not active and repair_queue:
                    for candidate,plan,error,repairs in repair_queue:
                        if self.budget.snapshot().research_seconds<=1:continue
                        try:
                            repaired=self.prepare(plan,round_id,feedback=error,prior=candidate)
                            if repaired:pending.append((repaired,plan,False,repairs+1))
                        except LLMUnavailable:pass
                    repair_queue=[]
                available=self.budget.snapshot().research_seconds
                for entry in list(pending):
                    candidate,plan,incumbent,repairs = entry
                    experiment_id=f'{round_id}-incumbent' if incumbent else candidate.candidate_id
                    existing=self.store.get('experiment',experiment_id)
                    if existing and existing['status']=='completed' and self.store.get('evaluation',experiment_id):
                        finished(candidate,plan,ExperimentRun.model_validate(existing),existing);pending.remove(entry);continue
                    if available < .5:
                        pending.remove(entry)
                        self.store.event('budget_skipped',experiment_id,{'remaining_seconds':available});continue
                    # One admission ledger covers training and synchronous checks/predictions.
                    if not self.sandbox.can_start(cpus=plan.cpus,memory_gb=plan.memory_gb): continue
                    run=ExperimentRun(experiment_id=experiment_id,round_id=round_id,candidate_id=candidate.candidate_id,
                        split_id=split.split_id,status='running',incumbent=incumbent,resource_cpus=plan.cpus,resource_memory_gb=plan.memory_gb)
                    old_worker=self.store.get('worker',experiment_id)
                    if old_worker:
                        import psutil
                        try:
                            p=psutil.Process(old_worker['pid'])
                            if p.create_time()==old_worker['created']: terminate_tree(p.pid)
                        except psutil.Error: pass
                    run_dir=self.run_dir/'runs'/experiment_id
                    # Reuse a completed worker result after an interrupted controller, before relaunching.
                    completed=run_dir/'result.json'
                    if completed.exists() and (not existing or existing['status']=='running'):
                        result=json.loads(completed.read_text(encoding='utf-8'))
                        if result.get('status')=='completed':
                            finished(candidate,plan,run,result);pending.remove(entry);continue
                    self.store.put('experiment',experiment_id,run)
                    req=dict(action='train',code_dir=candidate.code_dir,code_hash=candidate.code_hash,context=self.sanitized_context(train_path),
                        plan=plan.model_dump(mode='json'),train_path=str(train_path),seed=split.seed,inner_fraction=self.config.inner_fraction,fixed=incumbent)
                    seconds=min(plan.time_limit_seconds,self.config.experiment_seconds,available)
                    if state['round_number']==1 or not self.last_submission:
                        seconds=min(seconds,self.config.initial_experiment_seconds)
                    with bind(hypothesis_id=plan.hypothesis.hypothesis_id,candidate_id=candidate.candidate_id,experiment_id=experiment_id):
                        job=self.sandbox.start(req,run_dir,seconds=seconds,memory_gb=plan.memory_gb,cpus=plan.cpus)
                    import psutil
                    try:self.store.put('worker',experiment_id,{'pid':job.process.pid,'created':psutil.Process(job.process.pid).create_time()})
                    except psutil.Error:pass
                    active.append((entry,job,run));pending.remove(entry)
                for entry,job,run in list(active):
                    if job.poll() is None: continue
                    candidate,plan,incumbent,repairs=entry
                    result=job.finish(); completed_run=finished(candidate,plan,run,result)
                    active.remove((entry,job,run))
                    if completed_run.status=='failed' and not incumbent and repairs<self.config.runtime_repairs and self.budget.snapshot().research_seconds>1:
                        repair_queue.append((candidate,plan,completed_run.error,repairs))
                if active: time.sleep(.05)
                elif pending:
                    # All individual plans were validated against capacity; reject stale/infeasible queues.
                    impossible=[e for e in pending if e[1].cpus>self.budget.cpus or e[1].memory_gb>self.budget.memory_gb]
                    for e in impossible:
                        pending.remove(e);self.store.event('resource_skipped',e[0].candidate_id,{})

        plans=sorted([HypothesisPlan.model_validate(p) for p in state['plans']],key=lambda p:not p.hypothesis.baseline)
        entries=[]
        if previous:
            incumbent_plan=previous.plan.model_copy(update={'model_params':previous.run.best_params or previous.plan.model_params,'strategy':'single'})
            entries.append((previous.candidate,incumbent_plan,True,0))
        for number,plan in enumerate(plans):
            if self.budget.snapshot().research_seconds<.5: break
            try: candidate=self.prepare(plan,round_id)
            except LLMUnavailable:
                self.store.event('llm_unavailable','coding',{});break
            if candidate:
                entry=(candidate,plan,False,0)
                if previous is None and number==0:
                    process_jobs([entry])  # obtain a first valid file before preparing alternatives
                else: entries.append(entry)
            if number==0 and state.get('deferred_hypotheses'):
                try:
                    alternative=self.planning_call(state,mode='alternatives',hypotheses=state['deferred_hypotheses'])
                    if alternative.get('valid'):
                        plans.extend(HypothesisPlan.model_validate(p) for p in alternative['plans'])
                except (LLMUnavailable,ValueError) as error:
                    emit('planning.alternative_failed',{'error':str(error)})
        process_jobs(entries)
        output=dict(champion=champion.model_dump(mode='json') if champion else None,round_results=results,
            incumbent_succeeded=incumbent_ok,promoted=promoted,submission=self.last_submission)
        self.store.put('round_result',round_id,output,immutable=True)
        if champion:self.store.put('champion','current',champion)
        return output

    def analyze(self,state):
        context=self.restore(state); round_id=f"r{state['round_number']}"
        rows=sorted(state.get('round_results',[]),key=lambda r:r['evaluation']['score'],reverse=context.task.metric_direction=='maximize')
        selected=[]
        if rows:
            best_score=rows[0]['evaluation']['score']
            for rank,row in enumerate(rows,1):
                score=row['evaluation']['score']; gap=abs(best_score-score)
                if gap/max(abs(best_score),1e-12)>self.config.selection_threshold or len(selected)>=3: continue
                selected.append(dict(candidate_id=row['candidate']['candidate_id'],rank=rank,relative_gap=gap/max(abs(best_score),1e-12),
                    plan=row['plan'],evaluation=row['evaluation'],trials=row['run']['trials'],duration_seconds=row['run']['duration_seconds'],stop_reason=row['run']['stop_reason']))
        failures=[{'candidate_id':r['candidate_id'],'error':r.get('error'),'stop_reason':r.get('stop_reason')}
                  for r in self.store.list('experiment') if r['round_id']==round_id and r['status']=='failed'][-10:]
        previous=SelectedPreviousResults(round_id=round_id,selected=selected,failures=failures,
            champion_id=state['champion']['candidate']['candidate_id'] if state.get('champion') else None)
        self.store.put('analysis',round_id,previous,immutable=True)
        stale=state.get('stale',0)
        if state['round_number']>1 and state.get('incumbent_succeeded'):
            stale=0 if state.get('promoted') else stale+1
        stop=state.get('stop_reason')
        if stale>=self.config.plateau_rounds:stop='plateau'
        if not rows and not state.get('champion'):stop='no_successful_candidates'
        if state['round_number']>=self.config.max_rounds:stop='max_rounds'
        if self.budget.snapshot().research_seconds<1:stop='research_deadline'
        return {'selected':previous.model_dump(mode='json'),'stale':stale,'stop_reason':stop,'round_number':state['round_number']+1}

    def finalize(self,state):
        self.restore(state)
        champion=Champion.model_validate(state['champion']) if state.get('champion') else None
        if champion and self.budget.snapshot().remaining_seconds>self.config.prediction_seconds+2:
            experiment_id='final-refit'
            saved=self.store.get('experiment',experiment_id)
            plan=champion.plan.model_copy(update={'model_params':champion.run.best_params or champion.plan.model_params,'strategy':'single'})
            run=ExperimentRun(experiment_id=experiment_id,round_id='final',candidate_id=champion.candidate.candidate_id,
                split_id='all-train',role='submission_refit',status='running',resource_cpus=self.budget.cpus,resource_memory_gb=self.budget.memory_gb)
            try:
                if saved and saved['status']=='completed': run=ExperimentRun.model_validate(saved)
                else:
                    seconds=max(.1,self.budget.snapshot().remaining_seconds-self.config.prediction_seconds-2)
                    result=self.sandbox.run(dict(action='train',phase='final',code_dir=champion.candidate.code_dir,code_hash=champion.candidate.code_hash,
                        context=self.sanitized_context(self.files.train),plan=plan.model_dump(mode='json'),train_path=self.files.train,
                        seed=self.config.seed,fixed=True),self.run_dir/'runs'/experiment_id,seconds=seconds,memory_gb=self.budget.memory_gb,cpus=self.budget.cpus)
                    run=ExperimentRun.model_validate(dict(run.model_dump(),**{k:v for k,v in result.items() if k in ExperimentRun.model_fields}))
                self.store.put('experiment',experiment_id,run)
                if run.status=='completed':
                    full=champion.model_copy(update={'run':run})
                    self.publish(full)
            except Exception as error:
                self.store.event('refit_failed',experiment_id,{'error':str(error)})
        output=dict(status='completed' if self.last_submission else 'failed',stop_reason=state.get('stop_reason') or 'finished',
            submission=self.last_submission,champion=state.get('champion'),run_dir=str(self.run_dir),
            trace=str(self.run_dir/'trace.md'),first_submission=self.store.get('milestone','first_submission'))
        atomic_json(self.run_dir/'summary.json',output)
        return {'submission':self.last_submission}

    def graph(self):
        graph=StateGraph(RunState)
        for name,node in [('inspect',self.inspect),('plan',self.plan),('execute',self.execute),('analyze',self.analyze),('finalize',self.finalize)]:
            def traced_node(state, _node=node, _name=name):
                with bind(stage=_name,round_id=f"r{state.get('round_number',1)}"), span('stage',{'name':_name}):
                    print(json.dumps(dict(event='stage.started',stage=_name,round=state.get('round_number',1)),ensure_ascii=False),flush=True)
                    return _node(state)
            graph.add_node(name,traced_node)
        graph.add_edge(START,'inspect');graph.add_edge('inspect','plan')
        graph.add_conditional_edges('plan',lambda s:'execute' if s.get('plans') else 'finalize',{'execute':'execute','finalize':'finalize'})
        graph.add_edge('execute','analyze')
        graph.add_conditional_edges('analyze',lambda s:'finalize' if s.get('stop_reason') else 'plan',{'plan':'plan','finalize':'finalize'})
        graph.add_edge('finalize',END)
        return graph.compile(checkpointer=self.checkpointer)

    def run(self,*,resume=False):
        with self.trace.activate():
            return self._run(resume=resume)

    def _run(self,*,resume=False):
        try:
            emit('run.resumed' if resume else 'run.started',dict(deadline=self.budget.deadline,
                cpus=self.budget.cpus,memory_gb=self.budget.memory_gb,config=self.config.model_dump()))
            graph=self.graph()
            graph_config={'configurable':{'thread_id':'run'},'recursion_limit':max(100,self.config.max_rounds*5+10)}
            graph.invoke(None if resume and self.checkpointer.get_tuple(graph_config) else {},graph_config)
            output=json.loads((self.run_dir/'summary.json').read_text(encoding='utf-8'))
            emit('run.completed',output)
            return output
        except DeferredTask as error:
            output=dict(status='deferred',stop_reason='deferred_time_series_support',error=self.trace.scrub(str(error)),submission=None,run_dir=str(self.run_dir),trace=str(self.run_dir/'trace.md'))
            emit('run.deferred',output)
            atomic_json(self.run_dir/'summary.json',output)
            return output
        except Exception as error:
            output=dict(status='completed' if self.last_submission else 'failed',stop_reason='error',error=self.trace.scrub(str(error)),submission=self.last_submission,run_dir=str(self.run_dir),trace=str(self.run_dir/'trace.md'))
            emit('run.failed',output)
            atomic_json(self.run_dir/'summary.json',output)
            return output
        finally:
            try:
                self.sandbox.close()
            finally:
                try:
                    self.checkpointer.close()
                finally:
                    try:
                        self.store.close()
                    finally:
                        self.trace.close()
