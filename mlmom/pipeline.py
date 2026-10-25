"""End-to-end LangGraph orchestration with process-isolated experiment workers."""
from __future__ import annotations

from .agents import GlobalAPIError

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
from .evaluation import evaluate,better,prediction_kind,compact_diagnostics,relative_gain
from .inputs import resolve_inputs,read_table,features,targets,validate_tables
from .llm_client import LLMSettings
from .persistence import ExperimentStore,atomic_json,file_hash,tree_hash
from .pipeline_contracts import (RunConfig,TaskContext,HypothesisPlan,CandidateVersion,Champion,
    ExperimentRun,EvaluationResult,SelectedPreviousResults,SplitSpec,SubmissionArtifact)
from .planning import build_planning_graph
from .planning_context import profile_summary,plan_summary,experiment_summary,evaluation_summary
from .context_agent import prepare_planner_context
from .registry import SkillRegistry,installed_tools,capability_matrix
from .resources import (Budget, SandboxManager, terminate_tree, WorkerRuntimeError, WorkerInfrastructureError,
                        require_worker_infrastructure, worker_runtime, runtime_versions)
from .process_identity import worker_record, recorded_process_matches
from .splitting import make_split,group_keys
from .submission import publish_submission, snapshot_cycle_submission
from .tools import ToolSet
from .tracing import TraceSink, bind, emit, span
from .pipeline_contracts import TrainingCheckpoint,TrainingDecision,Observation
from .training_control import decide,validate_decision,commit_mailbox
from .research import data_observations,ReconnaissanceSession,review_result
from .execution_facts import RUNTIME_CONTRACT_VERSION, execution_context, related_fit_observations


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
    planning_attempt: int
    search_groups: list[dict]
    initial_boosting_sweep: bool


class DeferredTask(RuntimeError):
    pass


class Pipeline:
    def __init__(self, task_path: Path, config: RunConfig, *, run_dir: Path | None=None, client=None, resume=False, started=None):
        explicit_config_fields=set(config.model_fields_set)
        self.config = config
        self.files = resolve_inputs(task_path)
        self.run_dir = (run_dir or Path(__file__).resolve().parent.parent/'artifacts'/f'{Path(self.files.root).name}-{uuid.uuid4().hex[:10]}').resolve()
        self.run_dir.mkdir(parents=True,exist_ok=True)
        session_path = self.run_dir/'session.json'
        if session_path.exists():
            if not resume: raise ValueError('Run already exists; use --resume')
            session = json.loads(session_path.read_text(encoding='utf-8'))
            if session.get('runtime_contract_version') != RUNTIME_CONTRACT_VERSION:
                raise ValueError('This run uses an older runtime contract. Start a new run; saved models remain available for inference.')
            legacy_policy = 'timeouts_enabled' not in session['config']
            if legacy_policy:
                session['config']['timeouts_enabled'] = True
            if 'timeouts_enabled' not in explicit_config_fields:
                config = RunConfig.model_validate(dict(config.model_dump(), timeouts_enabled=session['config']['timeouts_enabled']))
                self.config = config
            # Old sessions must not silently gain an additional search branch.
            legacy_features = 'feature_search_enabled' not in session['config']
            if legacy_features:
                session['config']['feature_search_enabled'] = False
            if 'feature_search_enabled' not in explicit_config_fields:
                config = config.model_copy(update={'feature_search_enabled':session['config']['feature_search_enabled']})
                self.config = config
            legacy_boosting = 'initial_boosting_sweep' not in session['config']
            session['config'].setdefault('initial_boosting_sweep', False)
            if 'initial_boosting_sweep' not in explicit_config_fields:
                config = config.model_copy(update={'initial_boosting_sweep':session['config']['initial_boosting_sweep']})
                self.config = config
            legacy_cycle_refit = 'cycle_full_refit' not in session['config']
            session['config'].setdefault('cycle_full_refit', False)
            preserved = {key: session['config'][key] for key in ('cycle_full_refit', 'max_rounds', 'plateau_rounds')
                         if key not in explicit_config_fields and key in session['config']}
            config = config.model_copy(update=preserved)
            self.config = config
            if session['task'] != self.files.task or RunConfig.model_validate(session['config']).model_dump(mode='json') != config.model_dump(mode='json'):
                raise ValueError('Resume requires the same task and configuration')
            if session['input_hashes'] != self.input_hashes(): raise ValueError('Task inputs changed since this run')
            if legacy_policy or legacy_features or legacy_boosting or legacy_cycle_refit: atomic_json(session_path, session)
        else:
            if resume: raise ValueError('Cannot resume a run without session.json')
            session = dict(runtime_contract_version=RUNTIME_CONTRACT_VERSION,run_id=uuid.uuid4().hex,task=self.files.task,started=started or time.time(),config=config.model_dump(mode='json'),input_hashes=self.input_hashes())
            atomic_json(session_path,session)
        if 'run_id' not in session:
            session['run_id']=uuid.uuid4().hex
            atomic_json(session_path,session)
        settings=None
        if client is None:
            from dataclasses import replace
            settings=LLMSettings.from_platform_env() if config.profile=='platform' else LLMSettings.from_env()
            settings=replace(settings,timeout_seconds=config.llm_timeout_seconds if config.timeouts_enabled else None)
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
        atomic_json(self.run_dir/'capabilities.json',capability_matrix(self.tools))
        self.context = None; self.frame = None; self.test = None
        self.last_submission = self.store.get('submission','current')
        self.checkpointer = SQLiteCheckpointer(str(self.run_dir/'checkpoints.sqlite3'))
        if client is None:
            client = RoleClient(settings,deadline=self.budget.deadline-self.budget.reserve,max_steps=config.agent_steps,
                timeouts_enabled=config.timeouts_enabled,
                planning_direct_first=config.planning_direct_first,planning_context_reads=config.planning_context_reads,
                thinking_mode=config.thinking_mode,role_thinking=config.role_thinking,
                network_failure_threshold=config.network_failure_threshold,network_cooldown_seconds=config.network_cooldown_seconds,
                usage_callback=lambda usage:self.store.event('llm_usage','run',usage),
                request_callback=lambda event:self.store.event('llm_request','run',event),budget_store=self.store)
        self.client = client
        self._preparation_pool = None
        import threading
        self._cancel = threading.Event()
        self._controller_thread = threading.get_ident()
        self.sandbox.wait_callback = self.poll_training_controls
        if isinstance(client,RoleClient):
            client.timeouts_enabled=config.timeouts_enabled
        self.role_tools = ToolSet()
        self.role_tools.register('list_ml_tools',lambda:[s.model_dump(mode='json') for s in self.tools.values()], 'List installed, probed ML tools.')
        self.role_tools.register('list_skills',self.skills.list,'List registered instruction packages.')
        self.role_tools.register('read_skills',self.read_skills,'Read compatible skill instructions and hashes, including the automatic core contract.')
        self.role_tools.register('read_history',self.read_history,'Read bounded structured experiment history. No raw rows.')
        self.role_tools.register('read_task',self.read_task,'Read the complete original task in bounded pages when the briefing needs clarification.')
        self.role_tools.register('read_evidence',self.read_planning_evidence,'Read full registered reconnaissance or profile evidence omitted from compact context.')

    def read_planning_evidence(self, reference: str, offset: int=0, chars: int=8000):
        from .context_views import paged_evidence
        return paged_evidence(self.planning_evidence()).tools['read_evidence'][0](reference,offset,chars)

    def planning_evidence(self):
        from .context_views import stored_evidence
        profile=json.loads(Path(self.context.profile_ref).read_text(encoding='utf-8')) if self.context else None
        return stored_evidence(self.store,profile=profile)

    def read_task(self, offset: int=0, chars: int=8000) -> dict:
        if offset<0 or not 1<=chars<=8000:raise ValueError('Use non-negative offset and 1..8000 characters')
        text=Path(self.files.task).read_text(encoding='utf-8-sig')
        return dict(text=text[offset:offset+chars],offset=offset,total_chars=len(text),truncated=offset+chars<len(text))

    def read_skills(self, skill_ids: list[str]) -> list[dict]:
        if self.context is None:raise ValueError('Skills require a validated task context')
        return self.skills.load(skill_ids,self.context.task.task_type,role='planner')

    def read_missing_planning_skills(self, skill_ids: list[str]) -> list[dict]:
        # Core contracts are already present in the request, once.
        from .registry import CORE_SKILLS
        return [s for s in self.read_skills(skill_ids) if s['skill_id'] not in CORE_SKILLS]

    def input_hashes(self):
        paths = [self.files.task,self.files.train,self.files.test,*self.files.dictionaries]
        if self.files.sample_submission: paths.append(self.files.sample_submission)
        if self.files.manifest: paths.append(self.files.manifest)
        return {path:file_hash(Path(path)) for path in paths}

    def read_history(self, kind: str='evaluation', limit: int=10) -> list[dict]:
        if kind not in {'evaluation','experiment','plan','candidate'}: raise ValueError('History kind is not exposed')
        records = self.store.list(kind,limit=min(max(limit,1),5))
        if kind == 'experiment':
            records = [r for r in records if r.get('role', 'evaluation') == 'evaluation']
        if kind=='evaluation':
            records=[evaluation_summary(r) for r in records]
        if kind=='experiment':records=[experiment_summary(r) for r in records]
        if kind=='plan':records=[plan_summary(r) for r in records]
        if kind=='candidate':
            records = [{k:r[k] for k in ('candidate_id','hypothesis_id','version','code_hash','plan_ref')} for r in records]
        return records

    def restore(self,state):
        if self.context is None:
            self.context = TaskContext.model_validate(state['context'])
            validate_tables(self.context)
            self.frame = read_table(self.files.train,table_format=self.context.table_formats.get(self.files.train))
        return self.context


    def preflight_runtime(self):
        """Check the real child process before spending tokens or training budget."""
        seconds = min(30., self.budget.snapshot().research_seconds)
        if seconds <= 0 and self.last_submission:
            return  # Preserve an existing submission when resuming an expired run.
        expected = worker_runtime(Path(__file__).resolve().parents[1])
        report = dict(controller=expected)
        try:
            versions = runtime_versions()
            result = self.sandbox.run(dict(action='runtime_check'), self.run_dir/'runtime-check',
                seconds=max(.1, seconds), memory_gb=min(2., self.budget.memory_gb), cpus=1)
            report['result'] = result
            if result.get('status') != 'completed':
                raise WorkerRuntimeError(result.get('error') or result.get('stop_reason') or 'Worker did not start')
            actual = result['runtime']
            for key in ('executable', 'prefix', 'base_prefix'):
                if os.path.normcase(os.path.abspath(actual[key])) != os.path.normcase(os.path.abspath(expected[key])):
                    raise WorkerRuntimeError(f'{key} differs: controller={expected[key]}, worker={actual[key]}')
            if result['versions'] != versions:
                raise WorkerRuntimeError(f'Package versions differ: controller={versions}, worker={result["versions"]}')
            report['status'] = 'passed'
            self.store.put('runtime_check', 'startup', report)
            emit('worker.runtime_checked', report)
        except (GlobalAPIError, WorkerInfrastructureError):
            raise
        except Exception as error:
            report.update(status='failed', error=self.trace.scrub(str(error)))
            self.store.put('runtime_check', 'startup', report)
            self.store.event('worker_runtime_failed', 'startup', report)
            emit('worker.runtime_failed', report)
            raise WorkerRuntimeError(f'Worker environment check failed before LLM calls. '
                f'Python={expected["executable"]}; prefix={expected["prefix"]}. {error}') from error

    def inspect(self,state):
        existing = self.store.get('context','task')
        if existing:
            result = dict(context=existing,round_number=1,champion=None,selected={},stale=0,stop_reason=None)
            self.restore(result);return result
        profile_path = self.run_dir/'inspection'/'profile.json'
        result = dict(status='completed',profile_ref=str(profile_path)) if profile_path.is_file() else self.sandbox.run(
            dict(action='inspect',files=self.files.model_dump()),self.run_dir/'inspection',
            seconds=max(.1,min(300,self.budget.snapshot().research_seconds)),memory_gb=self.budget.memory_gb,cpus=self.budget.cpus)
        if result['status']!='completed': raise ValueError('Data inspection failed: '+result.get('error','timeout'))
        profile = DatasetProfile.model_validate_json(Path(result['profile_ref']).read_text(encoding='utf-8'))
        atomic_json(self.run_dir/'llm_context.json',build_llm_context(profile))
        from .llm_report import build_llm_markdown
        (self.run_dir/'llm_report.md').write_text(build_llm_markdown(profile),encoding='utf-8')
        if not self.store.get('reconnaissance','initial'):
            ReconnaissanceSession(client=self.client,profile=profile,files=self.files,sandbox=self.sandbox,
                budget=self.budget,store=self.store,config=self.config,key='initial',history={},
                skills=self.skills.load([],profile.task.task_type,role='reconnaissance_agent')).run()
        confirmed = self.store.get('confirmed_task','task')
        if not confirmed: raise ValueError('Reconnaissance did not confirm a valid task')
        task = TaskDefinition.model_validate(confirmed)
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
        previous=state.get('selected',{})
        decisions=previous.get('decisions',[])
        decision_ids={d.get('experiment_id') for d in decisions}
        champion_id=state['champion']['run']['experiment_id'] if state.get('champion') else None
        payload = dict(round_number=state['round_number'],task=context.task.model_dump(mode='json'),
            planning_attempt=state.get('planning_attempt',0),dataset_profile=profile_summary(profile),
            ml_tools=[{k:v for k,v in t.model_dump(mode='json').items() if k in
                {'tool_id','available','reason','task_types','skill_ids','native_hpo','cost_tag','supports_weights','version','validation_strategies'}}
                for t in self.tools.values()],
            skills=self.skills.list(role='planner'),
            technique_skills=self.skills.technique_catalog(context.task.task_type),
            budget=self.budget.snapshot().model_dump(),
            previous=dict(round_id=previous.get('round_id'),failures=previous.get('failures',[])[-5:],
                search_groups=previous.get('search_groups',[]),
                decisions=[{k:d[k] for k in ('candidate_id','experiment_id','accepted','reason','score','reference_score',
                    'evaluation','configuration_diff') if k in d}
                           for d in decisions[-5:] if d.get('experiment_id')!=champion_id]),
            champion=None,defaults=dict(max_trials=self.config.max_trials,patience_trials=self.config.patience_trials,
                experiment_seconds=self.config.experiment_seconds,first_submission_seconds=self.config.first_submission_seconds))
        payload['history']=dict(evaluation=[r for r in self.read_history('evaluation',limit=5)
            if r['experiment_id'] not in decision_ids|{champion_id}],
            experiment=[experiment_summary(r) for r in self.store.list('experiment',limit=10) if r['status']=='failed'][-3:])
        payload['known_candidates']=[dict(candidate_id=c['candidate_id'],hypothesis_id=c['hypothesis_id'])
            for c in self.store.list('candidate') if c['verification']['valid']]
        payload['improvement_policy']=dict(hpo_inside_hypothesis=self.config.hpo_enabled,
            batch_size=self.config.hpo_batch_size,relative_gain=self.config.hpo_gain_threshold,
            max_trials=self.config.hpo_initial_trials,explicit_trial_limit=self.config.hpo_max_trials,remaining_cycle_seconds=self.cycle_allowance(state['round_number']),
            require_evidence=state['round_number']>1,ensembles=True,timeouts_enabled=self.config.timeouts_enabled,
            outer_split_policy='fixed' if self.config.fixed_outer_split else 'reshuffle_each_cycle',
            comparison_policy='Retrain champion code with saved best parameters on this cycle train; compare only on this cycle holdout. Previous-cycle scores are historical, not a promotion threshold.',
            note='Use prior diagnostics/trials to change model logic or preprocessing; parameter trials are NOT new hypotheses. '
                 'After implementation failure, simplify the candidate code or fix the specific diagnosed operation. '
                 'Use strategy=optuna with a small compatible space after baseline. For text use sklearn estimator=linear, C for classification or alpha for regression.')
        payload['first_candidate_required']=not bool(self.last_submission)
        payload['artifact_refs']=dict(profile=context.profile_ref,history='experiments.sqlite3',
                                     previous_analysis=f"analysis:{previous.get('round_id','none')}")
        if state.get('champion'):
            champ = Champion.model_validate(state['champion'])
            from .planning_context import champion_snapshot
            search=experiment_summary(champ.run.model_dump(mode='json'))
            search={k:v for k,v in search.items() if k not in {'candidate_id','best_params','observed_execution'}}
            payload['champion'] = dict(champion_snapshot(champ,self.store),search=search)
            payload['execution_context'] = payload['champion']['execution_context']
        payload['reconnaissance']=self.scout('initial' if state['round_number']==1 else f"r{state['round_number']-1}",previous)
        reports={'initial':self.store.get('reconnaissance','initial') or payload['reconnaissance']}
        if state['round_number']>1:reports[f"r{state['round_number']-1}"]=payload['reconnaissance']
        payload['planning_brief']=prepare_planner_context(client=self.client,context=context,profile=profile,
            reports=reports,store=self.store,budget=self.budget,config=self.config,round_number=state['round_number'],
            skills=self.skills.load([],context.task.task_type,role='context_agent'),
            seconds=max(0,self.budget.snapshot().research_seconds),champion=payload['champion'])
        if state['round_number'] == 1:
            from .feature_search import prepare_initial_proposal
            prepare_initial_proposal(self,payload['planning_brief'])
        payload['result_reviews']=self.store.list('result_review',limit=4)
        payload['feature_sets']=self.store.list('feature_registry')
        payload['feature_search_history']=[dict(evidence=e,definition=self.store.get('feature_set',e['feature_set_id']),
            validation_scope='feature') for e in self.store.list('feature_evidence')]
        payload['training_recommendations']=[d.get('recommendations',[]) for d in self.store.list('training_decision',limit=4)]
        # Never put model artifact paths or the full run archive into a prompt.
        from .context_views import compact_planning_payload, observation_index
        payload=compact_planning_payload(payload)
        payload['observation_index']=observation_index(reports)
        payload['artifact_refs']['reconnaissance']=['reconnaissance:'+key for key in reports]
        payload['evidence_refs']=list(self.planning_evidence())
        return payload

    def scout(self,key,history):
        cached=self.store.get('reconnaissance',key)
        if cached:return cached
        # Raw outer-holdout labels never enter a follow-up tool registry.
        split = self.store.get('split',key)
        if not split:
            raise ValueError('Follow-up reconnaissance requires a completed round split')
        # Execute already materializes this exact training partition.
        train_path = self.run_dir/'splits'/key/'train.csv'
        train_path.parent.mkdir(parents=True,exist_ok=True)
        training = self.frame.iloc[split['train_indices']]
        if not train_path.exists(): training.to_csv(train_path,index=False)
        observations=data_observations(training,self.context,self.context.profile_ref)
        for item in history.get('decisions',[]):
            for index,(label,stats) in enumerate(item.get('diagnostics',{}).get('worst_classes',{}).items()):
                observations.append(Observation(observation_id=f"{item['experiment_id']}-class-{index}",
                    topic='rare_classes',description=f'Error metrics for class {label}',rows=int(stats.get('support',0)),
                    numbers={k:float(v) for k,v in stats.items() if type(v) in {int,float}},
                    source_ref=f"evaluation:{item['experiment_id']}"))
            for index,row in enumerate(item.get('diagnostics',{}).get('research_slices',[])):
                numbers={k:float(v) for k,v in row.items() if type(v) in {int,float} and v is not None}
                observations.append(Observation(observation_id=f"{item['experiment_id']}-slice-{index}",
                    topic=row['topic'],description=row['slice'],rows=row['rows'],numbers=numbers,
                    source_ref=f"evaluation:{item['experiment_id']}"))
        profile=DatasetProfile.model_validate_json(Path(self.context.profile_ref).read_text(encoding='utf-8'))
        report=ReconnaissanceSession(client=self.client,profile=profile,files=self.files,sandbox=self.sandbox,
            budget=self.budget,store=self.store,config=self.config,key=key,history=history,context=self.context,
            train_path=train_path,observations=observations,
            skills=self.skills.load([],self.context.task.task_type,role='reconnaissance_agent')).run()
        value=report.model_dump(mode='json')
        emit('reconnaissance.completed',value)
        return value

    def poll_training_controls(self):
        """Synchronous inference/refit must not starve another worker's mailbox."""
        import threading
        if threading.get_ident() != self._controller_thread:
            return
        for job in list(self.sandbox.jobs):
            if not job.released and (job.directory/'control').is_dir():
                run = self.store.get('experiment',job.directory.name)
                if run is not None:
                    self.training_checkpoints(job,ExperimentRun.model_validate(run))

    def training_checkpoints(self,job,run):
        for path in sorted((job.directory/'control').glob('*.request.json')):
            checkpoint=TrainingCheckpoint.model_validate_json(path.read_text(encoding='utf-8'))
            response=path.with_name(path.name.replace('.request.json','.decision.json'))
            key=f'{run.experiment_id}:{checkpoint.checkpoint_id}'
            cached=self.store.get('training_decision',key)
            if response.exists():
                if not cached:
                    cached=TrainingDecision.model_validate_json(response.read_text(encoding='utf-8')).model_dump(mode='json')
                    self.store.put('training_decision',key,cached,immutable=True)
                continue
            if cached:
                commit_mailbox(response,cached);continue
            # Poll watchdogs before any potentially blocking API request.
            self.sandbox.reap()
            remaining=max(0.,min(checkpoint.remaining_seconds if checkpoint.remaining_seconds is not None else float('inf'),
                job.seconds-(time.monotonic()-job.started),self.budget.snapshot().research_seconds))
            live=checkpoint.model_copy(update={'checkpoint_id':key,'remaining_seconds':remaining if self.config.timeouts_enabled else None,
                                               'timeouts_enabled':self.config.timeouts_enabled})
            seconds=min(self.config.training_control_seconds,remaining) if job.process.poll() is None else 0.
            decision=decide(self.client,live,self.skills.load([],self.context.task.task_type,role='training_agent'),seconds=seconds) if seconds>.5 else TrainingDecision(action=checkpoint.default_action,reason='No control budget')
            # A worker timeout may have committed its fallback during the API call.
            decision=TrainingDecision.model_validate(commit_mailbox(response,decision))
            self.store.put('training_checkpoint',key,checkpoint,immutable=True)
            self.store.put('training_decision',key,decision,immutable=True)
            emit('training.decision',dict(checkpoint=key,decision=decision.model_dump(mode='json')))

    def cycle_allowance(self,round_number):
        available=self.budget.snapshot().research_seconds
        completed=sum(c['completed'] for c in self.store.list('cycle'))
        outstanding=max(1,self.config.target_completed_rounds-completed)
        return max(0.,available/outstanding)

    def planning_call(self,state, *, mode='all', hypotheses=None):
        context = self.restore(state)
        key=f"r{state['round_number']}-{mode}"
        cached=self.store.get('planning',key)
        if isinstance(cached,dict) and cached.get('valid'):
            emit('planning.reused',{'key':key})
            return cached
        snapshot=self.budget.snapshot()
        advanced = state['round_number']>1 and bool(self.last_submission)
        lightweight = not advanced
        if isinstance(self.client,RoleClient):self.client.deadline=self.budget.deadline-self.budget.reserve
        payload=self.planning_payload(state)
        progress=self.store.get('planning_progress',key) or {}
        snapshot=self.budget.snapshot()
        payload['budget']=snapshot.agent_view()
        role_tools=ToolSet()
        role_tools.tools=dict(self.role_tools.tools)
        role_tools.register('read_skills',self.read_missing_planning_skills,'Read only missing backend/technique contracts; core instructions are already supplied.')
        if not any(payload['history'].values()):
            role_tools.tools.pop('read_history',None)
        emit('planning.policy',dict(lightweight=lightweight,mode=mode,budget=payload['budget']))
        candidate_limit=1 if mode == 'initial' else self.config.max_candidates
        graph = build_planning_graph(self.client,context,self.tools,self.skills,snapshot,
            self.config.max_attempts,candidate_limit,role_tools,
            known_candidates={c['candidate_id'] for c in self.store.list('candidate') if c['verification']['valid']},
            candidate_hypotheses={c['candidate_id']:c['hypothesis_id'] for c in self.store.list('candidate')},
            lightweight=lightweight,initial_experiment_seconds=self.config.initial_experiment_seconds,planning_mode=mode,
            evaluation_reserve_seconds=2*self.config.prediction_seconds,
            hypothesis_seconds=self.config.hypothesis_seconds,planner_seconds=self.config.planner_seconds,
            validator_seconds=self.config.validator_seconds,saved_progress=progress,
            save_progress=lambda value:self.store.put('planning_progress',key,value),
            initial_max_workers=self.config.max_workers)
        initial={'payload':payload}
        if hypotheses is not None: initial['hypotheses']=hypotheses
        elif progress.get('hypotheses'):initial['hypotheses']=progress['hypotheses']
        with span('planning',round_id=f"r{state['round_number']}"):
            started=time.monotonic()
            output=graph.invoke(initial,{'recursion_limit':100})
            self.store.put('cost',uuid.uuid4().hex,dict(stage='planning',seconds=time.monotonic()-started))
        for group in output.get('groups',[]):
            identity=f"r{state['round_number']}-"+group['hypothesis']['hypothesis_id']
            old=self.store.get('search_group',identity)
            if old:
                variants={p['hypothesis']['hypothesis_id']:p for p in [*old['variants'],*group['variants']]}
                group=dict(group,variants=list(variants.values()),skipped=[*old.get('skipped',[]),*group.get('skipped',[])])
            self.store.put('search_group',identity,group)
        for plan in output.get('plans',[]):
            if plan.get('feature_set'):
                self.store.put('feature_set',plan['feature_set_id'],plan['feature_set'],immutable=True)
        self.store.put('planning',key,output)
        if output.get('warnings'):
            self.store.put('warning',key,{'warnings':output['warnings']})
            self.store.event('validation_degraded',key,output['warnings'])
        return output


    def plan(self,state):
        if self.budget.snapshot().research_seconds <= .5: return {'plans':[],'stop_reason':'research_deadline'}
        try:
            first=not bool(self.last_submission)
            from .search_groups import SUPERVISED, compatible_boosters
            initial_sweep = bool(first and state['round_number'] == 1 and self.config.initial_boosting_sweep and
                (context := self.restore(state)).task.task_type in SUPERVISED and
                compatible_boosters(self.tools, context.task.task_type))
            output=self.planning_call(state,mode='initial' if initial_sweep else 'baseline' if first else 'all')
            from .ensemble_planning import plan_ensemble
            ensemble=plan_ensemble(self,state)
            if not output['valid']:
                self.store.event('planning_failed',f"r{state['round_number']}",output.get('errors',[]))
                if ensemble['plans']:
                    return dict(plans=ensemble['plans'],stop_reason=None,planning_attempt=0,deferred_hypotheses=None)
                if ('budget exceeded' not in json.dumps(output.get('errors',[])).lower() and self.can_retry_planning(state)):
                    previous=dict(state.get('selected',{}))
                    previous['failures']=(previous.get('failures',[])+[dict(stage='planning',
                        round_number=state['round_number'],errors=output.get('errors',[]))])[-10:]
                    return dict(plans=[],stop_reason='planning_retry',planning_attempt=state.get('planning_attempt',0)+1,selected=previous)
                return {'plans':[],'stop_reason':output.get('stop_reason') or 'planning_attempts_exhausted'}
            return {'plans':[*output['plans'],*ensemble['plans']],'stop_reason':None,
                    'planning_attempt':0,
                    'initial_boosting_sweep':initial_sweep,
                    'deferred_hypotheses':output.get('hypotheses') if first and not initial_sweep else None}
        except (GlobalAPIError, WorkerInfrastructureError):
            raise
        except (LLMUnavailable,ValueError) as error:
            if isinstance(error,LLMUnavailable):
                self.store.event('llm_unavailable','planning',{'error':str(error)})
            self.store.event('planning_failed',f"r{state['round_number']}",dict(
                attempt=state.get('planning_attempt',0)+1,error=str(error),error_type=type(error).__name__))
            from .ensemble_planning import plan_ensemble
            ensemble=plan_ensemble(self,state)
            if ensemble['plans']:
                return dict(plans=ensemble['plans'],stop_reason=None,planning_attempt=0,deferred_hypotheses=None)
            if 'budget exceeded' not in str(error).lower() and self.can_retry_planning(state):
                previous=dict(state.get('selected',{}))
                previous['failures']=(previous.get('failures',[])+[dict(stage='planning',error=str(error))])[-10:]
                return dict(plans=[],stop_reason='planning_retry',planning_attempt=state.get('planning_attempt',0)+1,selected=previous)
            return {'plans':[],'stop_reason':'llm_unavailable' if isinstance(error,LLMUnavailable)
                    and 'budget exceeded' in str(error).lower() else 'planning_attempts_exhausted'}

    def can_retry_round(self,state):
        return ((self.config.continuous or self.config.target_completed_rounds>0) and
                (self.config.max_rounds is None or state['round_number']<self.config.max_rounds) and
                self.budget.snapshot().research_seconds>.5)

    def can_retry_planning(self,state):
        return ((self.config.max_rounds is None or state['round_number']<=self.config.max_rounds) and
                state.get('planning_attempt',0)+1<self.config.max_attempts and
                self.budget.snapshot().research_seconds>.5)

    def prepare(self,plan,round_id,*,feedback=None, prior=None, coding_config=None):
        with bind(hypothesis_id=plan.hypothesis.hypothesis_id,round_id=round_id), span('coding'):
            started=time.monotonic()
            result=self._prepare(plan,round_id,feedback=feedback,prior=prior,coding_config=coding_config)
            self.store.put('cost',uuid.uuid4().hex,dict(stage='coding',seconds=time.monotonic()-started))
            return result

    def _prepare(self,plan,round_id,*,feedback=None, prior=None,coding_config=None):
        if isinstance(self.client,RoleClient):self.client.deadline=self.budget.deadline-self.budget.reserve
        plan_ref = self.run_dir/'plans'/f'{round_id}-{plan.hypothesis.hypothesis_id}.json'
        if not plan_ref.exists(): atomic_json(plan_ref,plan)
        stored_plan=self.store.get('plan',f'{round_id}-{plan.hypothesis.hypothesis_id}')
        if stored_plan:
            if HypothesisPlan.model_validate(stored_plan).model_dump(mode='json')!=plan.model_dump(mode='json'):
                raise ValueError('Stored plan differs from the requested candidate assignment')
        else:self.store.put('plan',f'{round_id}-{plan.hypothesis.hypothesis_id}',plan,immutable=True)
        if feedback is None:
            ready = [c for c in self.store.list('candidate') if c['plan_ref']==str(plan_ref) and c['verification']['valid']]
            if ready:
                candidate = CandidateVersion.model_validate(ready[-1])
                if tree_hash(Path(candidate.code_dir))!=candidate.code_hash: raise ValueError('Stored candidate was modified')
                from .tools import verify_source
                report=verify_source(Path(candidate.code_dir))
                if not report.valid:raise ValueError('Stored candidate violates current source policy; start a new run: '+ '; '.join(report.errors))
                return candidate
        previous = None
        if plan.hypothesis.based_on or prior:
            previous_candidate = prior or self.store.get('candidate',plan.hypothesis.based_on)
            if previous_candidate:
                if isinstance(previous_candidate,CandidateVersion): previous_candidate=previous_candidate.model_dump()
                source = Path(previous_candidate['code_dir'])
                old_plan=json.loads(Path(previous_candidate['plan_ref']).read_text(encoding='utf-8'))
                parent_id=plan.configuration_diff.get('base_experiment_id')
                parent_run=self.store.get('experiment',parent_id) if parent_id else None
                if parent_run:
                    from .model_spec import frozen_plan
                    old_plan=frozen_plan(HypothesisPlan.model_validate(old_plan),ExperimentRun.model_validate(parent_run)).model_dump(mode='json')
                pre=self.store.get('preprocessing',Path(previous_candidate['plan_ref']).stem) or {}
                if not pre and (source/'preprocessing.py').is_file():
                    pre=next((p for p in self.store.list('preprocessing') if p.get('sha256')==file_hash(source/'preprocessing.py')), {})
                dependencies={k:v.version for k,v in self.tools.items() if v.available and v.version}
                previous = {'candidate_id':previous_candidate['candidate_id'],'plan':old_plan,
                    'execution_context':execution_context(old_plan,parent_run,candidate=previous_candidate,
                        related_fits=related_fit_observations(self.store,parent_id)) if parent_run else execution_context(old_plan),
                    'preprocessing_spec':pre.get('spec'),
                    'dependencies_match':previous_candidate.get('dependencies',{})==dependencies,
                    'files':{p.relative_to(source).as_posix():p.read_text(encoding='utf-8') for p in source.rglob('*.py')}}
        skills = self.skills.load([k for k in plan.skill_ids if self.skills.allowed(k,'coding_agent')],
            self.context.task.task_type,role='coding_agent',include_references=True,max_chars=48000)
        preprocessing_skills=self.skills.load([k for k in plan.skill_ids if self.skills.owners[k]=='preprocessing_agent'],
            self.context.task.task_type,role='preprocessing_agent',include_references=True,max_chars=48000)
        workspace = self.run_dir/'workspaces'/round_id/plan.hypothesis.hypothesis_id
        if plan.ensemble_sources:
            from .ensemble_planning import bundle_components
            bundle_components(plan,self.store,workspace)
        from .concurrency import AssignmentClient
        graph = build_coding_graph(AssignmentClient(self.client,self._cancel),workspace=workspace,store=self.store,plan=plan,plan_ref=plan_ref,
            skill_data=skills,tools=self.tools,context=self.context,sandbox=self.sandbox,budget=self.budget,
            config=coding_config or self.config,round_id=round_id,previous=previous,runtime_feedback=feedback,preprocessing_skills=preprocessing_skills)
        output = graph.invoke({'payload':dict(plan=plan.model_dump(mode='json'),task=self.context.task.model_dump(mode='json'),
            features=self.context.feature_columns,classes=self.context.classes,skills=skills,previous=previous,
            execution_context=execution_context(plan)),
            'attempts':0,'errors':[]},{'recursion_limit':100})
        if output.get('candidate') and output['valid']: return CandidateVersion.model_validate(output['candidate'])
        self.store.event('coding_failed',plan.hypothesis.hypothesis_id,{'round_id':round_id,'errors':output.get('errors',[])})
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
        if result['status']!='completed':
            raise ValueError('Inference failed: '+(result.get('error') or result.get('stop_reason') or 'worker_error'))
        predictions = {k:read_table(v).to_numpy() for k,v in result['predictions'].items()}
        # Keep outputs and logs for audit; this input is only worker transport.
        try:
            features_path.unlink(missing_ok=True)
        except OSError as error:
            emit('prediction.input_cleanup_failed',{'path':str(features_path),'error':str(error)[:300]})
        return predictions

    def publish(self,champion):
        if self.config.cycle_full_refit and champion.run.role == 'evaluation':
            if not self.last_submission or self.last_submission.get('model_role') != 'submission_refit':
                report = self.refit_submission(champion,round_id=champion.run.round_id,
                    experiment_id=champion.run.experiment_id+'-first-submission-refit')
                return SubmissionArtifact.model_validate(report['artifact']) if report.get('status')=='passed' else None
            return None
        return self.publish_model(champion.candidate, champion.plan, champion.run)

    def publish_model(self,candidate,plan,run):
        if self.budget.snapshot().remaining_seconds < 1: return
        if self.test is None: self.test=read_table(self.files.test,table_format=self.context.table_formats.get(self.files.test))
        try:
            p = self._prediction(candidate,run,features(self.test,self.context),[self.context.submission_kind])
            artifact = publish_submission(self.context,self.test,p[self.context.submission_kind],run,candidate)
            saved=self.run_dir/'submissions'/f'{artifact.experiment_id}-{artifact.sha256}.csv'
            saved.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(artifact.path,saved)
            atomic_json(saved.with_suffix('.json'),dict(artifact=artifact.model_dump(mode='json'),
                execution_facts=run.model_dump(mode='json')['execution_facts'],configuration_diff=run.configuration_diff,
                saved_path=str(saved),model_artifact=run.model_artifact,
                code_dir=candidate.code_dir,code_hash=candidate.code_hash,
                plan=plan.model_dump(mode='json'),best_params=run.best_params))
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
            emit('submission.published',artifact.model_dump(),experiment_id=run.experiment_id,
                 candidate_id=candidate.candidate_id)
            return artifact
        except (GlobalAPIError, WorkerInfrastructureError):
            raise
        except Exception as error:
            self.store.event('submission_failed',run.experiment_id,{'error':str(error)})
            emit('submission.failed',{'error':str(error)},experiment_id=run.experiment_id)

    def refit_submission(self,champion,*,round_id,experiment_id,final=False):
        """Fit the frozen winner on all labeled rows, outside the comparison ledger."""
        from .execution import config_hash
        from .model_spec import frozen_plan
        source = dict(champion_experiment_id=champion.run.experiment_id,
            candidate_id=champion.candidate.candidate_id, code_hash=champion.candidate.code_hash,
            configuration_hash=config_hash(champion.run.model_spec or champion.run.best_params),
            dataset_hash=self.context.dataset_hash, train_path=self.files.train, train_rows=len(self.frame),
            model_role='submission_refit', split_id='all-train', used_for_comparison=False)
        self.store.put('submission_refit_source', experiment_id, source, immutable=True)
        self.store.put('submission_champion', experiment_id, champion, immutable=True)
        kind = 'experiment' if final else 'submission_refit'
        saved = self.store.get(kind, experiment_id)
        plan = frozen_plan(champion.plan, champion.run)
        run = ExperimentRun(experiment_id=experiment_id, round_id=round_id,
            candidate_id=champion.candidate.candidate_id, split_id='all-train', role='submission_refit',
            status='running', resource_cpus=self.budget.cpus, resource_memory_gb=self.budget.memory_gb)
        report = dict(source, experiment_id=experiment_id, round_id=round_id, status='failed')
        directory = self.run_dir/'runs'/experiment_id
        try:
            if saved and saved['status'] == 'completed':
                run = ExperimentRun.model_validate(saved)
            else:
                available = self.budget.snapshot().remaining_seconds if final else self.budget.snapshot().research_seconds
                seconds = available - self.config.prediction_seconds - 2
                if seconds <= .1:
                    report.update(status='budget_skipped', reason='Insufficient full-data fit and prediction budget')
                    self.store.put('submission_refit_status', experiment_id, report)
                    emit('submission.refit_skipped', report)
                    return report
                self.store.put(kind, experiment_id, run)
                emit('submission.refit_started', dict(source, experiment_id=experiment_id), round_id=round_id)
                result_path = directory/'result.json'
                result = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else None
                if not result or result.get('status') != 'completed':
                    result = self.sandbox.run(dict(action='train', phase='final' if final else 'submission',
                        code_dir=champion.candidate.code_dir, code_hash=champion.candidate.code_hash,
                        context=self.sanitized_context(self.files.train), plan=plan.model_dump(mode='json'),
                        train_path=self.files.train, seed=self.config.seed, fixed=True), directory,
                        seconds=max(.1, seconds), memory_gb=self.budget.memory_gb, cpus=self.budget.cpus)
                require_worker_infrastructure(result, directory)
                run = ExperimentRun.model_validate(dict(run.model_dump(), **{k:v for k,v in result.items() if k in ExperimentRun.model_fields}))
            if run.status != 'completed':
                raise ValueError(run.error or run.stop_reason or 'Full-data refit failed')
            if run.role != 'submission_refit' or run.split_id != 'all-train' or run.internal_score is not None or run.trials:
                raise ValueError('Full-data refit must have no comparison scores or search trials')
            self.store.put(kind, experiment_id, run)
            artifact = self.publish_model(champion.candidate, plan, run)
            if artifact is None:
                raise ValueError('Full-data submission publication failed')
            report.update(status='passed', model_artifact=run.model_artifact, artifact=artifact.model_dump(mode='json'))
            emit('submission.refit_completed', report, round_id=round_id)
        except (GlobalAPIError, WorkerInfrastructureError):
            raise
        except Exception as error:
            if run.status != 'completed':
                run.status, run.error = 'failed', str(error)
            self.store.put(kind, experiment_id, run)
            report['error'] = str(error)
            self.store.event('refit_failed', experiment_id, {'error':str(error)})
        self.store.put('submission_refit_status', experiment_id, report)
        return report

    def execute(self,state):
        context = self.restore(state)
        round_id = f"r{state['round_number']}"
        cached = self.store.get('round_result',round_id)
        if cached: return cached
        previous = Champion.model_validate(state['champion']) if state.get('champion') else None
        initial_sweep = bool(state.get('initial_boosting_sweep'))
        saved_split = self.store.get('split',round_id)
        anchor=(self.store.get('outer_split','shared') or self.store.get('split','r1')) if self.config.fixed_outer_split else None
        split = (SplitSpec.model_validate(saved_split) if saved_split else
                 SplitSpec.model_validate(anchor).model_copy(update={'round_id':round_id}) if anchor else
                 make_split(self.frame,context,round_id,self.config.holdout_fraction,
                            self.config.seed if self.config.fixed_outer_split else self.config.seed+state['round_number']-1))
        self.store.put('split',round_id,split,immutable=True)
        if self.config.fixed_outer_split and not self.store.get('outer_split','shared'):
            self.store.put('outer_split','shared',split,immutable=True)
        directory = self.run_dir/'splits'/round_id; directory.mkdir(parents=True,exist_ok=True)
        atomic_json(directory/'split.json',split)
        train_path = directory/'train.csv'
        if not train_path.exists(): self.frame.iloc[split.train_indices].to_csv(train_path,index=False)
        holdout = self.frame.iloc[split.holdout_indices]
        holdout_path=directory/'holdout.csv'
        if not holdout_path.exists():holdout.to_csv(holdout_path,index=False)
        results = []
        incumbent_ok = previous is None
        champion = previous
        promoted = False
        trained = {}
        evaluation_ready = set()
        group_skips=[]

        def group_remaining(plan):
            # Independent recipes share the actual run deadline and resource ledger.
            # No synthetic serial group window is inferred from historical costs.
            return self.budget.snapshot().research_seconds

        def finished(candidate,plan,run,result):
            nonlocal incumbent_ok,champion,promoted
            if result.get('status')!='completed' and plan.strategy=='optuna':
                from .hpo import interrupt_search
                states=interrupt_search(self.run_dir/'runs'/run.experiment_id/'hpo',result.get('stop_reason') or 'worker_interrupted')
                if states:
                    state=states[-1]
                    result['trials']=[dict(number=t['number'],params=dict(plan.model_params,**t['params']),
                        status='completed' if t['status']=='complete' else 'failed',score=t['score'],
                        duration_seconds=t['duration_seconds'],error=t['error'],kind='optuna') for t in state['trials']]
            if result.get('status')!='completed':
                progress = Path(self.run_dir/'runs'/run.experiment_id/'progress.json')
                if (result.get('stop_reason')=='time_limit' and progress.exists()
                        and json.loads(progress.read_text(encoding='utf-8')).get('full_refit')):
                    result = dict(json.loads(progress.read_text(encoding='utf-8')),trials=result.get('trials',json.loads(progress.read_text(encoding='utf-8')).get('trials',[])),stop_reason='time_limit_best_model',
                                  duration_seconds=result['duration_seconds'],peak_memory_gb=result['peak_memory_gb'])
            run = ExperimentRun.model_validate(dict(run.model_dump(),**{k:v for k,v in result.items() if k in ExperimentRun.model_fields}))
            try:
                require_worker_infrastructure(result, self.run_dir/'runs'/run.experiment_id)
                if run.status!='completed': raise ValueError(run.error or run.stop_reason or 'Training failed')
                if run.role!='evaluation' or run.split_id!=split.split_id: raise ValueError('Incomparable evaluation artifact')
                training_entry=dict(candidate=candidate.model_dump(mode='json'),plan=plan.model_dump(mode='json'),run=run.model_dump(mode='json'))
                if not run.incumbent:
                    trained[run.experiment_id]=training_entry
                    self.register_component(training_entry)
                    self.store.put('experiment',run.experiment_id,run)
                if (not run.incumbent and plan.search_group_kind!='single' and
                    run.experiment_id not in evaluation_ready and not self.store.get('evaluation',run.experiment_id) and
                    not (previous is None and not self.last_submission)):
                    # Family and recipe selection happens entirely on internal metrics.
                    self.store.put('search_group_progress',round_id+'-'+plan.search_group_id,
                        dict(group_id=plan.search_group_id,experiment_ids=[key for key,row in trained.items()
                             if row['plan'].get('search_group_id')==plan.search_group_id]))
                    return run
                if any(row['run']['experiment_id']==run.experiment_id for row in results):
                    return run
                saved_eval = self.store.get('evaluation',run.experiment_id)
                if saved_eval: evaluation = EvaluationResult.model_validate(saved_eval)
                else:
                    metric_kind = prediction_kind(context.task.primary_metric,context.task.task_type)
                    label_kind = 'label' if 'classification' in context.task.task_type else metric_kind
                    p = self._prediction(candidate,run,features(holdout,context),[metric_kind,label_kind])
                    unbox = lambda a:a[:,0] if a.shape[1]==1 else a
                    evaluation = evaluate(run.experiment_id,split.split_id,targets(holdout,context),unbox(p[metric_kind]),context,
                        X=features(holdout,context),groups=group_keys(holdout,context.task.group_columns),label_prediction=unbox(p[label_kind]),
                        weights=holdout[context.task.weight_column].to_numpy() if context.task.weight_column else None,
                        reference_X=features(self.frame.iloc[split.train_indices],context),reference_y=targets(self.frame.iloc[split.train_indices],context))
                    evaluation.validation_protocol_ref=str(directory/'split.json')
                    if not run.incumbent and self.config.diagnostic_seconds>0:
                        diag_dir=self.run_dir/'diagnostics'/run.experiment_id
                        try:
                            diagnosed=self.sandbox.run(dict(action='diagnose',code_dir=candidate.code_dir,code_hash=candidate.code_hash,
                                model_artifact=run.model_artifact,context=self.sanitized_context(train_path),holdout_path=str(holdout_path),
                                max_rows=self.config.diagnostic_rows,max_features=self.config.diagnostic_features),diag_dir,
                                seconds=min(self.config.diagnostic_seconds,max(.1,self.budget.snapshot().research_seconds)),
                                cpus=plan.cpus,memory_gb=plan.memory_gb)
                            evaluation.diagnostics['model_inspection']=diagnosed.get('diagnostics',dict(status='unavailable',reason=diagnosed.get('stop_reason') or diagnosed.get('error')))
                        except (GlobalAPIError, WorkerInfrastructureError):
                            raise
                        except Exception as error:evaluation.diagnostics['model_inspection']=dict(status='unavailable',reason=str(error)[:300])
                        evaluation.diagnostics['artifact_ref']=str(diag_dir/'report.json')
                        atomic_json(diag_dir/'report.json',evaluation.diagnostics)
                    self.store.put('evaluation',run.experiment_id,evaluation,immutable=True)
                emit('evaluation.completed',evaluation.model_dump(),candidate_id=candidate.candidate_id,experiment_id=run.experiment_id)
                entry = dict(candidate=candidate.model_dump(mode='json'),plan=plan.model_dump(mode='json'),run=run.model_dump(mode='json'),evaluation=evaluation.model_dump())
                results.append(entry)
                if not run.incumbent:self.register_component(entry)
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
                fit_seconds=max((t.duration_seconds for t in run.trials if t.status=='completed'),default=run.duration_seconds)
                self.store.put('cost',uuid.uuid4().hex,dict(stage='fit',seconds=fit_seconds))
                self.budget.reserve=max(self.budget.reserve,min(self.config.total_seconds*.5,
                    fit_seconds/(1-self.config.holdout_fraction)*1.5+self.config.prediction_seconds))
                if self.budget.reserve!=old_reserve:
                    self.store.put('budget','current',{'final_reserve_seconds':self.budget.reserve})
                    emit('budget.reserve_updated',{'final_reserve_seconds':self.budget.reserve})
            except WorkerInfrastructureError as error:
                run.status='failed';run.error=str(error);run.stop_reason=error.reason
                self.store.put('experiment',run.experiment_id,run)
                raise
            except GlobalAPIError:
                raise
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
                        if min(self.budget.snapshot().research_seconds,group_remaining(plan))<=1:continue
                        try:
                            repair_config=self.config.model_copy(update=dict(coding_seconds=max(.1,
                                min(self.config.coding_seconds,group_remaining(plan)))))
                            repaired=self.prepare(plan,round_id,feedback=error,prior=candidate,coding_config=repair_config)
                            if repaired:pending.append((repaired,plan,False,repairs+1))
                        except (GlobalAPIError, WorkerInfrastructureError):
                            raise
                        except LLMUnavailable:pass
                    repair_queue=[]
                available=self.budget.snapshot().research_seconds
                for entry in list(pending):
                    candidate,plan,incumbent,repairs = entry
                    experiment_id=f'{round_id}-incumbent' if incumbent else candidate.candidate_id+('-hpo' if plan.strategy=='optuna' else '')
                    existing=self.store.get('experiment',experiment_id)
                    if existing and existing['status']=='completed':
                        finished(candidate,plan,ExperimentRun.model_validate(existing),existing);pending.remove(entry);continue
                    remaining=min(available,group_remaining(plan)) if not incumbent else available
                    if remaining < .5:
                        pending.remove(entry)
                        group_skips.append(dict(group_id=plan.search_group_id,variant_id=plan.hypothesis.hypothesis_id,reason='group_or_research_budget_exhausted'))
                        self.store.event('budget_skipped',experiment_id,{'remaining_seconds':remaining});continue
                    # One admission ledger covers training and synchronous checks/predictions.
                    if not self.sandbox.can_start(cpus=plan.cpus,memory_gb=plan.memory_gb): continue
                    # can_start reaps exited workers and releases their slots. Consume
                    # those results (including prediction/diagnostics) before another
                    # training job takes the slot needed to evaluate the incumbent.
                    if any(job.poll() is not None for _,job,_ in active):break
                    run=ExperimentRun(experiment_id=experiment_id,round_id=round_id,candidate_id=candidate.candidate_id,
                        split_id=split.split_id,status='running',incumbent=incumbent,resource_cpus=plan.cpus,resource_memory_gb=plan.memory_gb)
                    old_worker=self.store.get('worker',experiment_id)
                    if old_worker:
                        import psutil
                        try:
                            if recorded_process_matches(old_worker): terminate_tree(old_worker['pid'])
                        except psutil.Error: pass
                    run_dir=self.run_dir/'runs'/experiment_id
                    # Reuse a completed worker result after an interrupted controller, before relaunching.
                    completed=run_dir/'result.json'
                    if completed.exists() and (not existing or existing['status']=='running'):
                        result=json.loads(completed.read_text(encoding='utf-8'))
                        if result.get('status')=='completed':
                            finished(candidate,plan,run,result);pending.remove(entry);continue
                    self.store.put('experiment',experiment_id,run)
                    self.store.put('training_plan',experiment_id,plan)
                    req=dict(action='train',code_dir=candidate.code_dir,code_hash=candidate.code_hash,context=self.sanitized_context(train_path),
                        plan=plan.model_dump(mode='json'),train_path=str(train_path),seed=split.seed,inner_fraction=self.config.inner_fraction,fixed=incumbent,
                        outer_split_id=split.split_id, outer_train_indices=split.train_indices,
                        fit_stage='incumbent_recheck' if incumbent else 'full_refit')
                    req.update({key:getattr(self.config,key) for key in ('hpo_batch_size','hpo_max_trials','hpo_startup_trials','hpo_gain_threshold','hpo_seconds')})
                    req.update(training_control=not incumbent,training_control_seconds=self.config.training_control_seconds)
                    initial=next((r for r in trained.values() if r['candidate']['candidate_id']==candidate.candidate_id and
                                  r['plan']['strategy']=='single'),None)
                    if initial and initial['run'].get('validation_scope','model')=='model':
                        source=self.run_dir/'runs'/initial['run']['experiment_id']
                        req['baseline_evidence']=dict(request_ref=str(source/'request.json'),selection_ref=str(source/'selection.json'))
                    seconds=min(plan.time_limit_seconds,self.config.experiment_seconds,remaining)
                    if state['round_number']==1 or not self.last_submission:
                        if plan.strategy!='optuna':seconds=min(seconds,self.config.initial_experiment_seconds)
                    with bind(hypothesis_id=plan.hypothesis.hypothesis_id,candidate_id=candidate.candidate_id,experiment_id=experiment_id):
                        job=self.sandbox.start(req,run_dir,seconds=seconds,memory_gb=plan.memory_gb,cpus=plan.cpus)
                    import psutil
                    try:self.store.put('worker',experiment_id,worker_record(job.process.pid))
                    except psutil.Error:pass
                    active.append((entry,job,run));pending.remove(entry)
                for entry,job,run in list(active):
                    self.training_checkpoints(job,run)
                    if job.poll() is None: continue
                    candidate,plan,incumbent,repairs=entry
                    result=job.finish(); completed_run=finished(candidate,plan,run,result)
                    active.remove((entry,job,run))
                    from .recovery import repair_allowed
                    if (completed_run.status=='failed' and not incumbent and self.budget.snapshot().research_seconds>1 and
                            repair_allowed(self.store,round_id+':'+plan.hypothesis.hypothesis_id,candidate,plan,
                                           completed_run.error,repairs,self.config.runtime_repairs)):
                        repair_queue.append((candidate,plan,completed_run.error,repairs))
                if active: time.sleep(.05)
                elif pending:
                    # All individual plans were validated against capacity; reject stale/infeasible queues.
                    impossible=[e for e in pending if e[1].cpus>self.budget.cpus or e[1].memory_gb>self.budget.memory_gb]
                    for e in impossible:
                        pending.remove(e);self.store.event('resource_skipped',e[0].candidate_id,{})

        feature_handled = False

        def feature_branch():
            nonlocal feature_handled
            if feature_handled or champion is None:
                return
            feature_handled = True
            from .feature_search import (run_feature_session, complete_feature_session,
                                         feature_session_key, select_feature_baseline)
            # The coordinator gets a source/config snapshot, never an outer evaluation.
            try:
                baseline=select_feature_baseline(self,champion,trained,state['round_number'])
                selected=run_feature_session(self,baseline,split,train_path)
                if selected:
                    completed_run=finished(CandidateVersion.model_validate(selected['candidate']),
                        HypothesisPlan.model_validate(selected['plan']),ExperimentRun.model_validate(selected['run']),selected['run'])
                    complete_feature_session(self,selected,
                        completed_run.status=='completed' and champion is not None and
                        champion.run.experiment_id==selected['run']['experiment_id'])
            except (GlobalAPIError, WorkerInfrastructureError):
                raise
            except Exception as error:
                self.store.event('feature_session_failed',feature_session_key(round_id),{'error':str(error)[:1500]})

        plans=sorted([HypothesisPlan.model_validate(p) for p in state['plans']],key=lambda p:(not p.hypothesis.baseline,p.priority))
        # Preparation shares controller persistence and independent workspaces. Resource leases
        # remain in the sandbox, including optional smoke workers started by a coder.
        futures = {}
        prepared_plans = {}
        alternative_future = None
        parallel_prepare = self.config.preparation_workers > 1 and (len(plans)>1 or state.get('deferred_hypotheses'))
        def queue_preparation(items):
            queued, assignments = {}, {}
            for item in items:
                room = group_remaining(item)
                limit = min(self.allocated_fit_seconds(item,state['round_number']),room)
                prepared_plan = item.model_copy(update={'time_limit_seconds':limit}) if limit < item.time_limit_seconds else item
                assignments[item.hypothesis.hypothesis_id] = prepared_plan
                config = self.config.model_copy(update=dict(coding_seconds=max(.1,min(self.config.coding_seconds,room))))
                queued[item.hypothesis.hypothesis_id] = self._preparation_pool.submit(
                    self.prepare,prepared_plan,round_id,coding_config=config)
            return queued, assignments
        if parallel_prepare:
            from .concurrency import PreparationPool
            self._preparation_pool = getattr(self,'_preparation_pool',None) or PreparationPool(self.config.preparation_workers)
            futures,prepared_plans = queue_preparation(plans)
            if state.get('deferred_hypotheses'):
                def prepare_alternatives():
                    output = self.planning_call(state,mode='alternatives',hypotheses=state['deferred_hypotheses'])
                    items = [HypothesisPlan.model_validate(value) for value in output.get('plans',[])] if output.get('valid') else []
                    queued,assignments = queue_preparation(items)
                    return output,queued,assignments
                alternative_future = self._preparation_pool.submit(prepare_alternatives)
        entries=[]
        if previous:
            from .model_spec import frozen_plan
            incumbent_plan=frozen_plan(previous.plan,previous.run)
            entries.append((previous.candidate,incumbent_plan,True,0))
        for number,plan in enumerate(plans):
            plan = prepared_plans.get(plan.hypothesis.hypothesis_id,plan)
            if self.budget.snapshot().research_seconds<.5: break
            group_room=group_remaining(plan)
            if group_room<.5:
                group_skips.append(dict(group_id=plan.search_group_id,variant_id=plan.hypothesis.hypothesis_id,reason='group_budget_exhausted'))
                continue
            fit_limit=min(self.allocated_fit_seconds(plan,state['round_number']),group_room)
            if fit_limit<plan.time_limit_seconds:
                emit('budget.plan_trimmed',dict(hypothesis_id=plan.hypothesis.hypothesis_id,
                    requested_seconds=plan.time_limit_seconds,allocated_seconds=fit_limit))
                plan=plan.model_copy(update={'time_limit_seconds':fit_limit})
            coding_config=self.config.model_copy(update=dict(coding_seconds=max(.1,min(self.config.coding_seconds,group_room))))
            try:
                future = futures.get(plan.hypothesis.hypothesis_id)
                candidate=future.result() if future is not None else self.prepare(plan,round_id,coding_config=coding_config)
            except (GlobalAPIError, WorkerInfrastructureError):
                raise
            except LLMUnavailable:
                self.store.event('llm_unavailable','coding',dict(tool_id=plan.tool_id))
                if initial_sweep:
                    group_skips.append(dict(group_id=plan.search_group_id,variant_id=plan.hypothesis.hypothesis_id,
                                            reason='coding_llm_unavailable'))
                    continue
                break
            if candidate:
                entry=(candidate,plan,False,0)
                if previous is None and number==0 and not initial_sweep:
                    process_jobs([entry])  # obtain a first valid file before preparing alternatives
                    feature_branch()
                    if self.last_submission and self.config.hpo_enabled and plan.strategy=='single' and plan.search_group_kind=='single':
                        space=self.search_space(plan)
                        if space and self.budget.snapshot().research_seconds>.5:
                            from .model_spec import frozen_plan
                            hpo_base=champion if champion is not None else None
                            hpo_candidate=hpo_base.candidate if hpo_base else candidate
                            hpo_plan=frozen_plan(hpo_base.plan,hpo_base.run) if hpo_base else plan
                            search=hpo_plan.model_copy(update={'strategy':'optuna','search_space':space,'max_trials':hpo_plan.hpo_trial_budget or self.config.hpo_initial_trials,
                                'time_limit_seconds':min(self.config.experiment_seconds,self.config.hpo_seconds*1.35)})
                            from .model_spec import hpo_child
                            parent=hpo_base.run.experiment_id if hpo_base else None
                            search=hpo_child(search,parent)
                            search.intervention.base_candidate_id=hpo_candidate.candidate_id
                            search.hypothesis.based_on=hpo_candidate.candidate_id
                            parent_row=next((r for r in results if r['run']['experiment_id']==parent),None)
                            if parent_row:
                                from .execution import config_hash
                                search.base_config_hash=config_hash(parent_row['run'].get('model_spec') or parent_row['run']['best_params'])
                            entries.append((hpo_candidate,search,False,0))
                else: entries.append(entry)
            elif plan.search_group_id:
                group_skips.append(dict(group_id=plan.search_group_id,variant_id=plan.hypothesis.hypothesis_id,reason='coding_failed_or_not_admitted'))
            if number==0 and state.get('deferred_hypotheses'):
                try:
                    if alternative_future is not None:
                        alternative,queued,assignments=alternative_future.result()
                        futures.update(queued);prepared_plans.update(assignments)
                    else:
                        alternative=self.planning_call(state,mode='alternatives',hypotheses=state['deferred_hypotheses'])
                    if alternative.get('valid'):
                        plans.extend(HypothesisPlan.model_validate(p) for p in alternative['plans'])
                except (GlobalAPIError, WorkerInfrastructureError):
                    raise
                except (LLMUnavailable,ValueError) as error:
                    emit('planning.alternative_failed',{'error':str(error)})
        process_jobs(entries)
        if previous is None:
            feature_branch()
        from .search_groups import collect_groups, internal_winner
        group_results=[]
        for group in collect_groups(plans):
            if group.kind=='single':continue
            key=group.hypothesis.hypothesis_id
            saved_group=self.store.get('search_group',round_id+'-'+key)
            group.skipped=[*(saved_group or {}).get('skipped',[]),*[s for s in group_skips if s.get('group_id')==key]]
            self.store.put('search_group',round_id+'-'+key,group)
            members=lambda:[row for row in trained.values() if row['plan'].get('search_group_id')==key]
            winner=internal_winner(members(),context.task.metric_direction)
            if winner and group.kind=='family' and self.config.hpo_enabled:
                selected_plan=HypothesisPlan.model_validate(winner['plan'])
                if selected_plan.search_space and min(group_remaining(selected_plan),self.budget.snapshot().research_seconds)>.5:
                    from .model_spec import hpo_child, frozen_plan
                    frozen=frozen_plan(selected_plan,ExperimentRun.model_validate(winner['run']))
                    frozen.search_space=selected_plan.search_space
                    search=hpo_child(frozen,winner['run']['experiment_id'])
                    search.strategy='optuna';search.max_trials=selected_plan.hpo_trial_budget or self.config.hpo_initial_trials
                    search.time_limit_seconds=min(self.config.experiment_seconds,self.config.hpo_seconds*1.35,group_remaining(selected_plan))
                    search.hypothesis.based_on=winner['candidate']['candidate_id']
                    search.intervention.base_candidate_id=winner['candidate']['candidate_id']
                    from .execution import config_hash
                    search.base_config_hash=config_hash(winner['run'].get('model_spec') or winner['run']['best_params'])
                    process_jobs([(CandidateVersion.model_validate(winner['candidate']),search,False,0)])
                    winner=internal_winner(members(),context.task.metric_direction)
            if winner:
                evaluation_ready.add(winner['run']['experiment_id'])
                finished(CandidateVersion.model_validate(winner['candidate']),HypothesisPlan.model_validate(winner['plan']),
                         ExperimentRun.model_validate(winner['run']),winner['run'])
            outcome=dict(group_id=key,kind=group.kind,
                winner_experiment_id=winner['run']['experiment_id'] if winner else None,
                variants=[dict(experiment_id=row['run']['experiment_id'],candidate_id=row['candidate']['candidate_id'],
                    tool_id=row['plan']['tool_id'],method=row['plan']['model_params'].get('method'),
                    internal_score=row['run']['internal_score'],split_id=row['run']['split_id'],
                    score_kind=row['run'].get('score_kind')) for row in members()],
                skipped=group.skipped,
                selection='internal_primary_metric',status='completed' if winner else 'no_successful_variants')
            outcome['failures']=[dict(experiment_id=r['experiment_id'],error=r.get('error'),reason=r.get('stop_reason'))
                for r in self.store.list('experiment') if r['round_id']==round_id and r['status']=='failed' and
                (self.store.get('training_plan',r['experiment_id']) or {}).get('search_group_id')==key]
            self.store.put('search_group_result',round_id+'-'+key,outcome)
            group_results.append(outcome)
        # Later cycles use the newly rechecked champion after model/family selection.
        # The initial boosting sweep reaches this branch after all family fits/HPO.
        # Legacy single baselines already ran it before their HPO.
        feature_branch()
        ensemble_status=self.store.get('ensemble_planning',round_id)
        if state['round_number']>1 and ensemble_status and not ensemble_status.get('plans'):
            group_results.append(dict(group_id=round_id+'-ensemble',kind='ensemble',status='skipped',
                                      reason=ensemble_status.get('reason'),error=ensemble_status.get('error')))
        output=dict(champion=champion.model_dump(mode='json') if champion else None,round_results=results,
            incumbent_succeeded=incumbent_ok,promoted=promoted,submission=self.last_submission,search_groups=group_results)
        self.store.put('round_result',round_id,output,immutable=True)
        if champion:self.store.put('champion','current',champion)
        return output

    def allocated_fit_seconds(self,plan,round_number):
        available=self.budget.snapshot().research_seconds
        return min(plan.time_limit_seconds,self.config.experiment_seconds,max(0.,available))

    def search_space(self,plan):
        # Generated implementations own their constructor contract; never guess parameters.
        return plan.search_space

    def register_component(self,row):
        """Retain admitted single fits even when their search group did not win."""
        plan=row['plan'];run=row['run']
        tool=self.tools.get(plan['tool_id'])
        if run.get('role', 'evaluation') != 'evaluation' or plan.get('ensemble_components') or plan.get('ensemble_sources') or not tool or tool.native_hpo:
            return
        if run.get('internal_score') is None:return
        from .execution import config_hash
        self.store.put('component',run['experiment_id'],dict(row,round_id=run['round_id'],
            experiment_id=run['experiment_id'],configuration_hash=config_hash(run.get('model_spec') or run['best_params'])))

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
                    execution_context=execution_context(row['plan'],row['run'],row['evaluation'],candidate=row['candidate']),
                    plan=row['plan'],evaluation=row['evaluation'],trials=row['run']['trials'],duration_seconds=row['run']['duration_seconds'],stop_reason=row['run']['stop_reason']))
        failures=[{'candidate_id':r['candidate_id'],'error':r.get('error'),'stop_reason':r.get('stop_reason')}
                  for r in self.store.list('experiment') if r['round_id']==round_id and r['status']=='failed'][-10:]
        coding_failures=self.store.db.execute("SELECT object_id,body FROM events WHERE kind='coding_failed' ORDER BY seq DESC LIMIT 10").fetchall()
        failures.extend(dict(stage='coding',hypothesis_id=key,errors=value.get('errors',[]))
            for key,body in coding_failures for value in [json.loads(body)] if value.get('round_id')==round_id)
        champion_id=state['champion']['candidate']['candidate_id'] if state.get('champion') else None
        champion_experiment=state['champion']['run']['experiment_id'] if state.get('champion') else None
        reference=next((r['evaluation']['score'] for r in rows if r['run']['incumbent']),None)
        decisions=[dict(candidate_id=r['candidate']['candidate_id'],experiment_id=r['run']['experiment_id'],
            accepted=r['run']['experiment_id']==champion_experiment,
            reason=('incumbent_recheck_failed' if not state.get('incumbent_succeeded',True) else
                    'selected_best_on_cycle_holdout' if r['run']['experiment_id']==champion_experiment else 'no_outer_gain'),
            score=r['evaluation']['score'],reference_score=reference,parameters=r['run']['best_params'],
            evaluation=evaluation_summary(r['evaluation']),configuration_diff=r['run'].get('configuration_diff',{}),
            internal_score=r['run']['internal_score'],hpo_stop_reason=r['run']['stop_reason'],
            diagnostics=compact_diagnostics(r['evaluation']['diagnostics'])) for r in rows if not r['run']['incumbent']]
        previous=SelectedPreviousResults(round_id=round_id,selected=selected,failures=failures,
            champion_id=champion_id,decisions=decisions,search_groups=state.get('search_groups',[]))
        self.store.put('analysis',round_id,previous,immutable=True)
        complete=bool(state.get('incumbent_succeeded',True) and
                      [r for r in rows if not r['run']['incumbent'] and r['evaluation']['diagnostics']])
        evaluation_complete = complete
        cycle_submission = None
        refit = None
        if self.config.cycle_full_refit and state.get('champion'):
            refit = self.refit_submission(Champion.model_validate(state['champion']), round_id=round_id,
                experiment_id=round_id+'-submission-refit')
            atomic_json(self.run_dir/'cycles'/round_id/'refit.json', refit)
            complete = complete and refit['status'] == 'passed'
        if self.last_submission:
            if self.test is None:
                self.test = read_table(self.files.test, table_format=context.table_formats.get(self.files.test))
            cycle_submission = snapshot_cycle_submission(context, self.test,
                SubmissionArtifact.model_validate(self.last_submission), self.run_dir, round_id)
            self.store.put('cycle_submission', round_id, cycle_submission, immutable=True)
            emit('submission.cycle_saved', cycle_submission, round_id=round_id)
        self.store.put('cycle',round_id,dict(round_id=round_id,completed=complete,
            evaluation_completed=evaluation_complete,
            candidate_ids=[r['candidate']['candidate_id'] for r in rows if not r['run']['incumbent']],
            analysis_ref=f'analysis:{round_id}',champion_id=previous.champion_id,submission=cycle_submission,
            submission_refit=refit))
        stale=state.get('stale',0)
        if state['round_number']>1 and state.get('incumbent_succeeded'):
            stale=0 if state.get('promoted') else stale+1
        stop=state.get('stop_reason')
        completed=sum(c['completed'] for c in self.store.list('cycle'))
        if self.config.plateau_rounds is not None and stale>=self.config.plateau_rounds and completed>=self.config.target_completed_rounds:stop='plateau'
        if not rows and not state.get('champion') and not self.can_retry_round(state):stop='no_successful_candidates'
        if self.config.max_rounds is not None and state['round_number']>=self.config.max_rounds:stop='max_rounds'
        if self.config.target_completed_rounds and completed>=self.config.target_completed_rounds:stop='target_completed_rounds'
        if self.budget.snapshot().research_seconds<1:stop='research_deadline'
        # Reviews advise the next round. Its planning_payload loads follow-up
        # reconnaissance lazily; a terminal cycle needs neither advisory role.
        if stop is None:
            try:
                self.review_results(rows,context)
            except (GlobalAPIError, WorkerInfrastructureError):
                raise
            except Exception as error:
                self.store.event('advisory_failed',round_id,{'error':str(error)[:300]})
        return {'selected':previous.model_dump(mode='json'),'stale':stale,'stop_reason':stop,
                'submission':self.last_submission,'round_number':state['round_number']+1}

    def review_results(self,rows,context):
        for row in rows:
            if row['run']['incumbent']:continue
            key=row['run']['experiment_id']
            if self.store.get('result_review',key):continue
            seconds=min(self.config.result_validation_seconds,self.budget.snapshot().research_seconds)
            if not self.config.timeouts_enabled:seconds=self.budget.snapshot().research_seconds
            if seconds<.5:continue
            code_root=Path(row['candidate']['code_dir'])
            sources={p.relative_to(code_root).as_posix():p.read_text(encoding='utf-8') for p in code_root.rglob('*.py')}
            from .review_context import review_payload
            review=review_result(self.client,review_payload(row,rows,sources=sources,store=self.store),
                self.skills.load([],context.task.task_type,role='result_validator'),key,seconds)
            self.store.put('result_review',key,review,immutable=True)
            atomic_json(self.run_dir/'reviews'/f'{key}.json',review)
            emit('result.reviewed',dict(experiment_id=key,report=review.model_dump(mode='json')))

    def finalize(self,state):
        self.restore(state)
        champion=Champion.model_validate(state['champion']) if state.get('champion') else None
        latest_source = self.store.get('submission_refit_source', self.last_submission['experiment_id']) if self.last_submission else None
        current_full = bool(champion and self.last_submission and self.last_submission.get('model_role') == 'submission_refit' and
            latest_source and latest_source['champion_experiment_id'] == champion.run.experiment_id)
        if champion and not current_full and self.budget.snapshot().remaining_seconds>self.config.prediction_seconds+2:
            self.refit_submission(champion, round_id='final', experiment_id='final-refit', final=True)
        output=dict(status='completed' if self.last_submission else 'failed',stop_reason=state.get('stop_reason') or 'finished',
            result_available=bool(self.last_submission and self.last_submission.get('model_role')=='submission_refit'),
            submission=self.last_submission,champion=state.get('champion'),run_dir=str(self.run_dir),
            trace=str(self.run_dir/'trace.md'),first_submission=self.store.get('milestone','first_submission'))
        output['failure_causes']=self.failure_causes()
        output['feature_search_sessions']=[{key:session.get(key) for key in
            ('session_id','round_id','status','reason','attempts','stagnant','best_feature_set_id','outer_experiment_id')}
            for session in self.store.list('feature_session')]
        output['feature_search']=(output['feature_search_sessions'][-1] if output['feature_search_sessions'] else None)
        output['warnings']=self.store.list('warning')
        output['process_acceptance']=dict(target_completed_rounds=self.config.target_completed_rounds,
            completed_rounds=sum(c['completed'] for c in self.store.list('cycle')),
            imported_baselines=sum(t.get('origin')=='baseline'
                for run in self.store.list('experiment') for t in run['trials']),
            completed_optuna_trials=sum(t['status']=='completed' and t['kind']=='optuna' and t.get('origin')!='baseline'
                for run in self.store.list('experiment') if not run['incumbent'] for t in run['trials']))
        acceptance=output['process_acceptance']
        acceptance['passed']=(acceptance['completed_rounds']>=self.config.target_completed_rounds and
            (not self.config.target_completed_rounds or acceptance['completed_optuna_trials']>=self.config.hpo_batch_size))
        if self.config.cycle_full_refit:
            acceptance['full_data_submission'] = bool(self.last_submission and
                self.last_submission.get('model_role') == 'submission_refit')
            acceptance['passed'] = acceptance['passed'] and acceptance['full_data_submission']
        atomic_json(self.run_dir/'capabilities.json',capability_matrix(self.tools,
            [self.context.task.task_type] if acceptance['completed_rounds'] else []))
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
        graph.add_conditional_edges('plan',lambda s:'execute' if s.get('plans') else 'plan' if s.get('stop_reason')=='planning_retry' else 'finalize',
                                    {'execute':'execute','plan':'plan','finalize':'finalize'})
        graph.add_edge('execute','analyze')
        graph.add_conditional_edges('analyze',lambda s:'finalize' if s.get('stop_reason') else 'plan',{'plan':'plan','finalize':'finalize'})
        graph.add_edge('finalize',END)
        # Continuous runs checkpoint after each cycle, then resume in a fresh
        # invocation so LangGraph's per-invocation step guard is not a run limit.
        return graph.compile(checkpointer=self.checkpointer,
            interrupt_after=['plan', 'analyze'] if self.config.max_rounds is None else None)

    def run(self,*,resume=False):
        with self.trace.activate():
            return self._run(resume=resume)

    def _run(self,*,resume=False):
        try:
            emit('run.resumed' if resume else 'run.started',dict(deadline=self.budget.deadline,
                cpus=self.budget.cpus,memory_gb=self.budget.memory_gb,config=self.config.model_dump()))
            self.preflight_runtime()
            graph=self.graph()
            graph_config={'configurable':{'thread_id':'run'},
                'recursion_limit':max(100,self.config.max_rounds*5+10) if self.config.max_rounds is not None else 100}
            graph.invoke(None if resume and self.checkpointer.get_tuple(graph_config) else {},graph_config)
            if self.config.max_rounds is None:
                while graph.get_state(graph_config).next:
                    graph.invoke(None,graph_config)
            output=json.loads((self.run_dir/'summary.json').read_text(encoding='utf-8'))
            emit('run.completed' if output['status']=='completed' else 'run.failed',output)
            return output
        except DeferredTask as error:
            output=dict(status='deferred',stop_reason='deferred_time_series_support',error=self.trace.scrub(str(error)),submission=None,run_dir=str(self.run_dir),trace=str(self.run_dir/'trace.md'))
            emit('run.deferred',output)
            atomic_json(self.run_dir/'summary.json',output)
            return output
        except WorkerInfrastructureError as error:
            failure=dict(reason=error.reason,error=str(error),artifacts=error.directory)
            self.store.event('worker_infrastructure_failed',error.directory,failure)
            output=dict(status='failed',stop_reason='worker_infrastructure_error',error=str(error),
                submission=self.last_submission,run_dir=str(self.run_dir),trace=str(self.run_dir/'trace.md'),
                process_acceptance=dict(target_completed_rounds=self.config.target_completed_rounds,
                    completed_rounds=sum(c['completed'] for c in self.store.list('cycle')),passed=False),
                failure_causes=self.failure_causes())
            atomic_json(self.run_dir/'worker-infrastructure-failure.json',failure)
            atomic_json(self.run_dir/'summary.json',output)
            emit('run.failed',output)
            return output
        except GlobalAPIError as error:
            output=dict(status='failed',stop_reason='global_api_failure',error=str(error),
                submission=self.last_submission,run_dir=str(self.run_dir),trace=str(self.run_dir/'trace.md'),
                process_acceptance=dict(target_completed_rounds=self.config.target_completed_rounds,
                    completed_rounds=sum(c['completed'] for c in self.store.list('cycle')),
                    passed=False),failure_causes=self.failure_causes())
            atomic_json(self.run_dir/'provider-failure.json',dict(reason=str(error)))
            atomic_json(self.run_dir/'summary.json',output)
            emit('run.failed',output)
            return output
        except KeyboardInterrupt:
            self._cancel.set()
            if isinstance(self.client,RoleClient): self.client.cancel_event.set()
            if self._preparation_pool is not None: self._preparation_pool.close()
            if isinstance(self.sandbox,SandboxManager):
                for job in self.sandbox.jobs:
                    if job.process.poll() is None: job.reason='external_stop'
            self.sandbox.close()
            active_studies=[]
            for kind in ('experiment','submission_refit'):
                for record in self.store.list(kind):
                    if record.get('status')=='running':
                        directory=self.run_dir/'runs'/record['experiment_id']/'hpo'
                        if directory.is_dir(): active_studies.append(directory)
                        record.update(status='interrupted',stop_reason='external_stop')
                        self.store.put(kind,record['experiment_id'],record)
            submission_champion = (self.store.get('submission_champion', self.last_submission['experiment_id'])
                                   if self.last_submission else None)
            output = dict(status='interrupted', stop_reason='external_stop',
                result_available=bool(self.last_submission and self.last_submission.get('model_role')=='submission_refit'),
                submission=self.last_submission, champion=submission_champion or self.store.get('champion','current'),
                first_submission=self.store.get('milestone','first_submission'), failure_causes=self.failure_causes(),
                run_dir=str(self.run_dir), trace=str(self.run_dir/'trace.md'))
            atomic_json(self.run_dir/'summary.json', output)
            emit('run.stopped', output)
            from .hpo import interrupt_search
            for directory in active_studies:
                try: interrupt_search(directory,'external_stop')
                except Exception as error:
                    self.store.event('stop_accounting_failed',str(directory),dict(error=str(error)))
            return output
        except Exception as error:
            output=dict(status='failed',stop_reason='worker_runtime_error' if isinstance(error,WorkerRuntimeError) else 'error',error=self.trace.scrub(str(error)),submission=self.last_submission,run_dir=str(self.run_dir),trace=str(self.run_dir/'trace.md'),failure_causes=self.failure_causes())
            emit('run.error',output)
            emit('run.failed',output)
            atomic_json(self.run_dir/'summary.json',output)
            return output
        finally:
            self._cancel.set()
            if isinstance(self.client,RoleClient): self.client.cancel_event.set()
            if self._preparation_pool is not None: self._preparation_pool.close()
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
                        self.index_artifacts()

    def failure_causes(self):
        rows=self.store.db.execute("SELECT kind,object_id,body FROM events WHERE kind IN "
            "('planning_failed','coding_failed','llm_unavailable','submission_failed','refit_failed','experiment_failed','adapter_failure','worker_runtime_failed','worker_infrastructure_failed') ORDER BY seq").fetchall()
        return [dict(kind=kind,object_id=key,details=json.loads(body)) for kind,key,body in rows]

    def index_artifacts(self):
        files=[]
        for path in sorted(self.run_dir.rglob('*')):
            if path.is_file() and path.name!='artifact_index.json' and path.suffix not in {'.tmp','.pyc'}:
                files.append(dict(path=path.relative_to(self.run_dir).as_posix(),bytes=path.stat().st_size,
                                  sha256=file_hash(path)))
        atomic_json(self.run_dir/'artifact_index.json',dict(run_dir=str(self.run_dir),files=files))
