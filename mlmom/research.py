"""Numerical evidence and bounded advisory roles, independent of champion selection."""
from __future__ import annotations

from .agents import GlobalAPIError
from .resources import WorkerInfrastructureError

import math
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

from .agents import role_window
from .pipeline_contracts import Observation, ReconnaissanceReport, ResultReview, ToolResult
from .contracts import TaskDefinition
from .csv_tools import CSVTools
from .inputs import read_table
from .persistence import atomic_json, file_hash
from .task_validator import TaskValidator
from .tools import ToolSet, ToolInputError


RECONNAISSANCE_PROMPT = '''Read the supplied task description and deterministic profile first.
For initial reconnaissance, call confirm_task with a complete TaskDefinition BEFORE deep research.
This is the only task-understanding role. Match explicit target, metric/options/direction, ID,
excluded columns, validation and submission requirements from the task, dictionary and manifest.
validation_strategy selects INTERNAL validation only; outer holdout rotates independently each cycle.
Choose holdout explicitly when permitted, or kfold/stratified_kfold/group_kfold with validation_folds
(default 5, minimum 2). Do not claim k-fold for a single split or reduce folds to hide infeasibility.
Inspector heuristics are suggestions. Fix validator errors; never invent missing requirements.
Once confirmed, the task is immutable. Later rounds must not change it.
Investigate the registered train and test features using list_tables, preview_csv and query_csv.
Use Polars 1.26 SQL SELECT/WITH, e.g. SELECT category, COUNT(*) AS n, AVG(target) AS mean_target
FROM train GROUP BY category. Quote column names with double quotes. Only registered tables exist.
No file functions, paths or mutations. SQL aggregates can scan every row; tool output is bounded.
Use correlate_csv for Pearson correlations: CORR is unavailable in Polars 1.26 SQL.
Inspect missingness, duplicates, rare classes/categories, tails, suspicious values, feature-target
relationships and train/test distribution shifts where relevant. Never use sample submission
prediction values as labels: they are placeholders. Holdout is available only through aggregates.
Persist each useful finding with save_observation using its supplied profile/CSV evidence reference
and exact measured numbers. Distinguish observations from assumptions. Proposals are advisory ideas:
describe an expected effect and a check on comparable splits. Include saved observation IDs when
available; references are optional and do not gate proposal retention. Do not invent numbers.
Return ReconnaissanceReport, including unfinished_checks for work to continue next round.
Budget covers all requests, tools and repairs. Confirm requirements early and save findings often.'''


class ResearchTools(ToolSet):
    def __init__(self, root):
        super().__init__()
        self.root = root

    def _invoke(self, name, arguments):
        result = super()._invoke(name,arguments)
        if len(result.model_dump_json()) <= 32000:
            return result
        identity = hashlib.sha256(result.model_dump_json().encode()).hexdigest()
        path = self.root/'tool_responses'/f'{identity}.json'
        atomic_json(path,result)
        return ToolResult(status='ok',duration_seconds=result.duration_seconds,
            data=dict(truncated=True,artifact_ref=str(path),
                      message='Response exceeds 32000 characters. Narrow or paginate the request.'))


class ReconnaissanceSession:
    """One persisted deadline shared by requirements, research, tools and recovery."""
    def __init__(self, *, client, profile, files, sandbox, budget, store, config, key,
                 history, skills, context=None, train_path=None, observations=()):
        self.client, self.profile, self.files = client, profile, files
        self.budget, self.store, self.config, self.key = budget, store, config, key
        self.context, self.history, self.skills = context, history, skills
        seconds = config.initial_reconnaissance_seconds if key == 'initial' else config.reconnaissance_seconds
        if not getattr(config,'timeouts_enabled',True):
            from .time_policy import UNBOUNDED_SECONDS
            seconds=UNBOUNDED_SECONDS-budget.now()
        window = store.get('reconnaissance_budget', key)
        if window is None:
            window = dict(deadline=budget.now()+min(seconds, budget.snapshot().research_seconds))
            store.put('reconnaissance_budget', key, window, immutable=True)
        self.deadline = window['deadline']
        self.profile_ref = 'profile:'+file_hash(Path(context.profile_ref) if context else store.root/'inspection'/'profile.json')
        self.observations = {o.observation_id:o for o in observations}
        self.observations.update({o['observation_id']:Observation.model_validate(o)
            for o in (store.get('reconnaissance_progress',key) or {}).get('observations',[])})
        progress = store.get('reconnaissance_progress',key) or {}
        self.unfinished = progress.get('unfinished_checks',[])
        tables = {}
        for name,path,role in [('train',train_path or files.train,'training'), ('test',files.test,'test_features'),
                               *[(f'dictionary_{i+1}',p,'dictionary') for i,p in enumerate(files.dictionaries)]]:
            fmt = profile.table_formats.get(str(path))
            columns = list(read_table(path, table_format=fmt, nrows=0).columns)
            if name == 'test':
                confirmed = store.get('confirmed_task','task')
                targets = context.task.target_columns if context else confirmed['target_columns'] if confirmed else profile.task.target_columns
                columns = [c for c in columns if c not in targets]
            tables[name] = dict(path=str(path), format=fmt.model_dump() if fmt else None,
                                role=role, columns=columns, hash=file_hash(Path(path)))
        if files.sample_submission:
            tables['sample_submission'] = dict(path=files.sample_submission, role='submission_template_placeholders',
                schema_only=True, hash=file_hash(Path(files.sample_submission)))
        self.csv = CSVTools(tables=tables, sandbox=sandbox, budget=budget, store=store, key=key,
                            deadline=self.deadline, query_seconds=config.reconnaissance_query_seconds)
        self.tools = ResearchTools(store.root/'reconnaissance'/key)
        for name in ('list_tables','preview_csv','query_csv','correlate_csv'):
            self.tools.register(name, getattr(self.csv,name), dict(list_tables='List registered tables and paginated schemas.',
                preview_csv='Read a bounded window of selected CSV columns.',
                query_csv='Run read-only Polars SQL on registered tables; full aggregates, bounded output.',
                correlate_csv='Measure Pearson correlations of numeric columns with a target, excluding non-finite pairs.')[name])
        self.tools.register('read_task', self.read_task, 'Read the original task text in bounded pages.')
        self.tools.register('confirm_task', self.confirm_task, 'Validate and persist the task before research; returns validation errors.')
        self.tools.register('save_observation', self.save_observation, 'Persist a measured finding with a supplied evidence reference.')
        from .context_views import paged_evidence
        self.previous_reports=self.store.list('reconnaissance',limit=3)
        evidence={f'previous_report:{i}':report for i,report in enumerate(self.previous_reports)}
        self.tools.tools.update(paged_evidence(evidence).tools)

    def _check_time(self):
        if self.budget.now() >= self.deadline or self.budget.snapshot().research_seconds <= 0:
            raise ValueError('Reconnaissance deadline reached')

    def read_task(self, offset: int = 0, chars: int = 8000) -> dict:
        self._check_time()
        if offset < 0 or not 1 <= chars <= 8000:
            raise ValueError('Use non-negative offset and 1..8000 characters')
        text = Path(self.files.task).read_text(encoding='utf-8-sig')
        return dict(text=text[offset:offset+chars], offset=offset, total_chars=len(text),
                    truncated=offset+chars<len(text))

    def confirm_task(self, task: TaskDefinition | str) -> dict:
        self._check_time()
        task = TaskDefinition.model_validate_json(task) if isinstance(task, str) else TaskDefinition.model_validate(task)
        task = task.model_copy(update={'excluded_feature_columns':list(dict.fromkeys(
            task.excluded_feature_columns+task.target_columns+task.id_columns+task.group_columns+
            ([task.weight_column] if task.weight_column else [])))})
        saved = self.store.get('confirmed_task','task')
        locked = self.context.task if self.context else TaskDefinition.model_validate(saved) if saved else None
        if locked:
            if locked.model_dump(exclude={'agent_notes'}) != task.model_dump(exclude={'agent_notes'}):
                raise ValueError('The confirmed task is immutable')
            return dict(valid=True, task=locked.model_dump(mode='json'))
        report = TaskValidator().validate(self.profile, task)
        # Do not commit requirements that the executor must reject later.
        extra = []
        if task.task_type == 'unknown': extra.append('Unknown task type')
        if task.task_type == 'ranking' and not task.group_columns: extra.append('Ranking requires query groups')
        if task.task_type == 'anomaly_detection' and not task.target_columns: extra.append('Anomaly evaluation requires labels')
        fmt = task.result_format
        if not fmt.file_name or not fmt.columns or not fmt.prediction_columns:
            extra.append('Output filename, columns and prediction columns are required')
        if not report.valid or extra:
            return dict(valid=False, findings=report.model_dump(mode='json')['findings'], errors=extra)
        train_columns = list(read_table(self.files.train, table_format=self.profile.table_formats.get(self.files.train), nrows=0))
        test_columns = list(read_table(self.files.test, table_format=self.profile.table_formats.get(self.files.test), nrows=0))
        features = [c for c in train_columns if c not in task.excluded_feature_columns]
        if not features or set(features+task.id_columns+task.group_columns)-set(test_columns):
            return dict(valid=False, errors=['No usable features or required columns missing from test'])
        self.csv.tables['test']['columns'] = [c for c in test_columns if c not in task.target_columns]
        self.store.put('confirmed_task','task',task,immutable=True)
        atomic_json(self.store.root/'task.json',dict(status='validated',task=task.model_dump(mode='json'),
            validation=report.model_dump(mode='json'),model=self.client.model))
        return dict(valid=True, task=task.model_dump(mode='json'))

    def save_observation(self, topic: str, description: str, rows: int | None = None,
                         numbers: dict[str,float] | None = None, source_ref: str = '') -> dict:
        self._check_time()
        numbers = numbers or {}
        evidence = self.store.get('csv_result', source_ref[4:]) if source_ref.startswith('csv:') else None
        result_rows = evidence.get('total_rows') if evidence else None
        identity = 'obs-'+hashlib.sha256(json.dumps([topic,description,rows,numbers,source_ref],sort_keys=True).encode()).hexdigest()[:20]
        item = Observation(observation_id=identity,topic=topic,description=description,rows=rows,numbers=numbers,source_ref=source_ref,result_rows=result_rows)
        self.observations[identity] = item
        self._persist()
        return item.model_dump(mode='json')

    def _persist(self):
        self.store.put('reconnaissance_progress',self.key,dict(observations=[o.model_dump(mode='json') for o in self.observations.values()],
            unfinished_checks=self.unfinished))

    def reconcile(self, answer):
        rejected = []
        for item in answer.observations:
            if item.observation_id in self.observations and item == self.observations[item.observation_id]:
                continue
            try:
                saved = self.save_observation(item.topic,item.description,item.rows,item.numbers,item.source_ref)
                if item.agent_notes:
                    self.observations[saved['observation_id']] = Observation.model_validate(dict(saved,agent_notes=item.agent_notes))
            except ValueError as exc:
                rejected.append(f'Observation {item.observation_id}: {exc}')
        unfinished = list(dict.fromkeys(answer.unfinished_checks+self.unfinished+rejected))
        # Proposals remain advisory, with their original optional references intact.
        return answer.model_copy(update=dict(observations=list(self.observations.values()),
            status='partial' if rejected or self.unfinished else answer.status,
            reason='; '.join(rejected)[:1000] if rejected else answer.reason,unfinished_checks=unfinished))

    def run(self):
        from .data_inspector import build_llm_context
        from .metric_contracts import metric_capabilities
        text = Path(self.files.task).read_text(encoding='utf-8-sig')
        saved = self.store.get('confirmed_task','task')
        original_steps = getattr(self.client,'max_steps',None)
        original_deadline = getattr(self.client,'deadline',None)
        answer = None
        error = None
        try:
            self._check_time()
            if original_steps is not None: self.client.max_steps = self.config.reconnaissance_steps
            if original_deadline is not None: self.client.deadline = min(original_deadline,self.deadline)
            with role_window(self.client,'reconnaissance_agent',self.key,max(0,self.deadline-self.budget.now())):
                answer = self.client.generate('reconnaissance_agent', RECONNAISSANCE_PROMPT,
                    dict(initial=self.key=='initial', dataset_inspection=build_llm_context(self.profile),
                         task_description=text[:16000],task_chars=len(text),profile_ref=self.profile_ref,
                         confirmed_task=self.context.task.model_dump(mode='json') if self.context else saved,
                         observations=[o.model_dump(mode='json') for o in self.observations.values()],
                         history=self.history,skills=self.skills,
                         previous_reports=[dict(status=report.get('status'),unfinished_checks=report.get('unfinished_checks',[]),
                             observation_ids=[o['observation_id'] for o in report.get('observations',[])],
                             evidence_ref=f'previous_report:{i}') for i,report in enumerate(self.previous_reports)],
                         allowed_metric_options=metric_capabilities(),
                         task_schema=TaskDefinition.model_json_schema()), ReconnaissanceReport,self.tools)
            if not self.context and not self.store.get('confirmed_task','task'):
                raise ValueError('Reconnaissance did not confirm a valid task')
            if self.budget.now() >= self.deadline:
                raise ValueError('Reconnaissance deadline reached')
            answer = self.reconcile(answer)
        except (GlobalAPIError, WorkerInfrastructureError):
            raise
        except Exception as exc:
            error = str(exc)[:300]
            answer = ReconnaissanceReport(status='partial' if self.context or self.store.get('confirmed_task','task') else 'unavailable',
                observations=list(self.observations.values()),reason=error,
                unfinished_checks=self.unfinished+['Continue unfinished investigation from saved observations and CSV results.'])
        finally:
            if original_steps is not None: self.client.max_steps = original_steps
            if original_deadline is not None: self.client.deadline = original_deadline
            self._persist()
        self.store.put('reconnaissance',self.key,answer,immutable=True)
        atomic_json(self.store.root/'reconnaissance'/f'{self.key}.json',answer)
        if not self.context and not self.store.get('confirmed_task','task'):
            raise ValueError('No validated task after reconnaissance: '+(error or 'missing confirmation'))
        return answer


def data_observations(frame, context, source_ref):
    observations=[]
    def add(topic, description, numbers):
        observations.append(Observation(observation_id=f'data-{len(observations)+1}',topic=topic,
            description=description,rows=len(frame),numbers=numbers,source_ref=source_ref))
    X=frame[context.feature_columns]
    add('duplicates','Repeated feature rows and conflicting targets',
        dict(duplicate_rows=int(X.duplicated().sum())))
    if context.task.target_columns:
        hashes=pd.util.hash_pandas_object(X,index=False)
        labels=frame[context.task.target_columns].astype(str).agg('\x1f'.join,axis=1)
        conflicts=pd.DataFrame({'key':hashes.to_numpy(),'label':labels.to_numpy()}).groupby('key')['label'].nunique().gt(1).sum()
        observations[-1].numbers['conflicting_groups']=int(conflicts)
    for column in X:
        s=X[column]
        if s.isna().any():add('missing',f'Missing values in {column}',dict(count=int(s.isna().sum()),rate=float(s.isna().mean())))
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            counts=s.value_counts(dropna=False)
            rare=counts[counts<max(20,math.ceil(.01*len(s)))]
            add('rare_groups',f'Rare categories in {column}',dict(groups=len(rare),rows=int(rare.sum())))
        else:
            numeric=pd.to_numeric(s,errors='coerce').to_numpy(dtype=float,na_value=np.nan)
            count=int(np.isinf(numeric).sum())
            if count:add('suspicious',f'Infinite values in {column}',dict(count=count))
    for name in context.task.target_columns:
        s=frame[name]
        if 'regression' in context.task.task_type:
            finite=pd.to_numeric(s,errors='coerce').replace([np.inf,-np.inf],np.nan).dropna()
            if len(finite):
                lo,hi=finite.quantile([.01,.99])
                add('target_tail',f'Target tails of {name}',dict(q01=float(lo),q99=float(hi),
                    lower_rows=int((finite<=lo).sum()),upper_rows=int((finite>=hi).sum())))
        else:
            counts=s.value_counts()
            if len(counts):add('rare_classes',f'Class support of {name}',dict(classes=len(counts),min_count=int(counts.min()),max_count=int(counts.max())))
    return observations


def error_slices(y, prediction, X, reference_X, reference_y, task_type):
    """Group definitions are fitted on training data; only aggregates leave evaluation."""
    if y is None or reference_X is None or not any(kind in task_type for kind in ('regression','classification')):return []
    truth,pred=np.asarray(y),np.asarray(prediction)
    if truth.shape!=pred.shape:return []
    error=(truth!=pred).astype(float) if 'classification' in task_type else np.abs(truth-pred)
    if error.ndim>1:error=error.mean(axis=1)
    result=[]
    def add(topic,label,mask):
        mask=np.asarray(mask,dtype=bool);rows=int(mask.sum())
        from .evaluation import error_metrics
        metrics=error_metrics(truth[mask],pred[mask],task_type) if rows else {}
        result.append(dict(topic=topic,slice=label,rows=rows,status='completed' if rows>=20 else 'insufficient_support',**metrics))
    if 'regression' in task_type and reference_y is not None:
        a=np.asarray(reference_y);a=a[:,None] if a.ndim==1 else a
        b=truth[:,None] if truth.ndim==1 else truth
        for i in range(a.shape[1]):
            finite=a[:,i][np.isfinite(a[:,i])]
            if len(finite):
                lo,hi=np.quantile(finite,[.01,.99])
                add('target_tail',f'target_{i} <= train q01 ({lo:g})',b[:,i]<=lo)
                add('target_tail',f'target_{i} >= train q99 ({hi:g})',b[:,i]>=hi)
    elif 'classification' in task_type and reference_y is not None:
        a=np.asarray(reference_y);a=a[:,None] if a.ndim==1 else a
        b=truth[:,None] if truth.ndim==1 else truth
        for i in range(a.shape[1]):
            counts=pd.Series(a[:,i]).value_counts()
            support=pd.Series(b[:,i]).map(counts).fillna(0).to_numpy()
            add('rare_classes',f'target_{i}: rare or unseen class in train',support<max(20,math.ceil(.01*len(a))))
    if X is not None:
        add('missing','any missing feature',X.isna().any(axis=1))
        add('duplicates','feature tuple seen in training',pd.util.hash_pandas_object(X,index=False).isin(pd.util.hash_pandas_object(reference_X,index=False)))
        add('duplicates','repeated feature rows in evaluation',X.duplicated(keep=False))
        for column in X.columns[:20]:
            s,r=X[column],reference_X[column]
            if not pd.api.types.is_numeric_dtype(r) or pd.api.types.is_bool_dtype(r):
                counts=r.fillna('__missing__').astype(str).value_counts()
                support=s.fillna('__missing__').astype(str).map(counts).fillna(0)
                add('rare_groups',f'{column}: rare or unseen in train',support<max(20,math.ceil(.01*len(r))))
            else:
                rr=pd.to_numeric(r,errors='coerce').replace([np.inf,-np.inf],np.nan).dropna()
                if len(rr):
                    lo,hi=rr.quantile([.01,.99])
                    add('suspicious',f'{column}: outside train q01/q99',(s<lo)|(s>hi)|np.isinf(s))
    return result


def review_result(client,payload,skills,key,seconds):
    from .context_views import paged_evidence
    sources=payload.get('source',{})
    values=dict(payload.get('evidence',{}))
    values.update({f'source:{name}':text for name,text in sources.items()})
    evaluation=payload.get('evaluation') or {}
    if evaluation.get('experiment_id'):
        values['evaluation:'+evaluation['experiment_id']]=evaluation
    payload=dict(payload,source={name:dict(evidence_ref=f'source:{name}',characters=len(text)) for name,text in sources.items()},
                 evidence_refs=list(values))
    payload.pop('evidence',None)
    review_tools=paged_evidence(values) if values else None
    from .evaluation import comparable_scores,relative_gain
    reference=payload.get('comparable_reference')
    comparable=comparable_scores(payload.get('evaluation',{}),reference)
    payload=dict(payload,comparable_reference=reference if comparable else None,
                 relative_gain=relative_gain(payload.get('evaluation',{}),reference))
    try:
        with role_window(client,'result_validator',key,seconds):
            result=client.generate('result_validator',
                'Advisory review only. Compare the hypothesis with actual source, transformations and trial '
                'parameters. Separately assess implementation fidelity and empirical support. Use only supplied '
                'evidence_refs, scores and slices. An overall score gain does not prove a subgroup claim. '
                'Cross-split historical scores are not comparable. Missing baseline/support means inconclusive. '
                'Use internal_validation for fold stability and OOF; smoke is synthetic and cannot establish full refit. '
                'execution_context contains runtime facts and the actual SDK composition, including feature programs. '
                'Their absence from preprocessing.py does not mean they were skipped. Never rewrite these facts in your report. '
                'Respect hypothesis_prose_role: inherited historical baseline prose is not a current requirement '
                'and cannot establish an implementation mismatch. Judge the controller-selected feature bundle '
                'from its actual intervention/configuration and distinguish it from the old baseline mechanism. '
                'references.parent_control tests HPO; a gain against references.baseline only measures the whole bundle. '
                'Use trial_history for observed trial counts; plan.max_trials is an initial renewable budget, not a measured count. '
                'Missing metrics are unknown. Read source evidence through read_evidence when needed. '
                'Never change code, champion selection or submission.',dict(payload,skills=skills),ResultReview,review_tools)
        return ResultReview.model_validate(result).model_copy(update=dict(execution_context=payload.get('execution_context',{})))
    except (GlobalAPIError, WorkerInfrastructureError):
        raise
    except Exception as error:
        return ResultReview(implementation='unknown',effect='inconclusive',explanation='Review unavailable: '+str(error)[:300],
                            execution_context=payload.get('execution_context',{}))
