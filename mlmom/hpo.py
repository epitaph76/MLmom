"""Persistent, bounded parameter search; never invokes an LLM or changes code."""
from __future__ import annotations

import hashlib
import json
import time
import pickle
from pathlib import Path

import optuna

from .persistence import atomic_json


def relative_gain(old, new, direction):
    return ((new-old) if direction == 'maximize' else (old-new))/max(abs(old), 1e-12)


def interrupt_search(directory,reason):
    """Controller-side accounting after a worker was killed, without new trials."""
    directory=Path(directory)
    if not (directory/'studies.sqlite3').exists():return []
    from sqlalchemy.pool import NullPool
    storage=optuna.storages.RDBStorage('sqlite:///'+(directory/'studies.sqlite3').as_posix(),engine_kwargs={'poolclass':NullPool})
    states=[]
    try:
        for path in directory.glob('*.json'):
            state=json.loads(path.read_text(encoding='utf-8'))
            study=optuna.load_study(study_name=state['study_name'],storage=storage)
            for trial in study.get_trials(deepcopy=False):
                if trial.state==optuna.trial.TrialState.RUNNING:
                    storage.set_trial_user_attr(trial._trial_id,'error',reason)
                    study.tell(trial.number,state=optuna.trial.TrialState.FAIL)
            state['trials']=[dict(number=t.number,status=t.state.name.lower(),params=t.params,score=t.value,
                duration_seconds=t.duration.total_seconds() if t.duration else 0.,error=t.user_attrs.get('error'))
                for t in study.get_trials(deepcopy=False)]
            state['interruption_reason']=reason
            atomic_json(path,state);states.append(state)
    finally:
        storage.remove_session();storage.engine.dispose()
    return states


def default_space(tool, params):
    """Small backend-specific spaces; custom implementations supply their own."""
    if tool == 'lightgbm':
        return {'num_leaves':dict(type='int',low=15,high=63),
                'min_child_samples':dict(type='int',low=10,high=80),
                'learning_rate':dict(type='float',low=.025,high=.15,log=True)}
    if tool == 'catboost':
        return {'depth':dict(type='int',low=4,high=8),
                'l2_leaf_reg':dict(type='float',low=1.,high=20.,log=True)}
    if tool == 'xgboost':
        return {'max_depth':dict(type='int',low=3,high=8),
                'learning_rate':dict(type='float',low=.025,high=.15,log=True)}
    if tool == 'sklearn':
        if params.get('estimator')=='linear':
            key='C' if 'C' in params else 'alpha'
            return {key:dict(type='float',low=.01,high=10.,log=True)}
        return {'max_depth':dict(type='int',low=4,high=24),
                'min_samples_leaf':dict(type='int',low=1,high=12)}
    if tool == 'text_linear':
        key = 'C' if 'C' in params else 'alpha'
        return {key:dict(type='float',low=.1 if key=='C' else .01,high=10.,log=True)}
    return {}


def run_search(objective, *, directory, identity, space, direction, seed, max_trials,
               batch_size=5, gain_threshold=.005, seconds=600, baseline_score=None, checkpoint=None, trial_checkpoint=None,timeouts_enabled=True):
    """Continue a study in batches. Persist decisions and every finished trial."""
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(json.dumps(dict(identity=identity,space=space,direction=direction),
        sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:24]
    database=directory/'studies.sqlite3'
    path=directory/f'{digest}.json'
    state=json.loads(path.read_text(encoding='utf-8')) if path.exists() else dict(
        identity=identity,study_name=digest,batches=[],spent_seconds=0.,deadline=time.time()+seconds if timeouts_enabled else None,timeouts_enabled=timeouts_enabled,
        baseline_score=baseline_score,stop_reason=None)
    # Resume consumes the original wall-clock window, including interruptions.
    if state.get('timeouts_enabled',True)!=timeouts_enabled:raise ValueError('Resume requires the same timeout policy')
    from .time_policy import UNBOUNDED_SECONDS
    until=min(state['deadline'],time.time()+seconds) if timeouts_enabled else UNBOUNDED_SECONDS
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from sqlalchemy.pool import NullPool
    storage=optuna.storages.RDBStorage('sqlite:///'+database.as_posix(),engine_kwargs={'poolclass':NullPool})
    study=optuna.create_study(study_name=digest,storage=storage,
        load_if_exists=True,direction=direction,sampler=optuna.samplers.TPESampler(seed=seed))
    sampler_path=directory/f'{digest}.sampler'
    if sampler_path.exists():study.sampler=pickle.loads(sampler_path.read_bytes())
    attempted = lambda: sum(t.state != optuna.trial.TrialState.WAITING for t in study.trials)
    try:
        for trial in study.get_trials(deepcopy=False):
            if trial.state == optuna.trial.TrialState.RUNNING:
                study.tell(trial.number,state=optuna.trial.TrialState.FAIL)
        def persist():
            state['trials']=[dict(number=t.number,status=t.state.name.lower(),params=t.params,
                score=t.value,duration_seconds=t.duration.total_seconds() if t.duration else 0.,
                error=t.user_attrs.get('error')) for t in study.get_trials(deepcopy=False) if t.state != optuna.trial.TrialState.WAITING]
            state['study_ref']=str(database)
            sampler_path.write_bytes(pickle.dumps(study.sampler))
            atomic_json(path,state)
        def apply_decision(decision):
            state['control_action']=decision.action
            if decision.action in {'stop','refit'}:
                state['stop_reason']='training_agent_'+decision.action
            elif time.time()<until and attempted()<max_trials:
                state['stop_reason']=None
                for params in decision.next_params:study.enqueue_trial(params,skip_if_exists=True)
        def request_control(kind, details):
            state['pending_control']=dict(kind=kind,details=details)
            persist()
            details=dict(details,remaining_seconds=max(0.,min(details['remaining_seconds'],until-time.time())),
                         remaining_trials=max(0,max_trials-attempted()))
            if details['remaining_seconds']<=0:
                from .pipeline_contracts import TrainingDecision
                decision=TrainingDecision(action='refit',reason='Original HPO deadline reached')
            else:
                decision=(trial_checkpoint if kind=='trial' else checkpoint)(details)
            apply_decision(decision)
            state.pop('pending_control',None)
            persist()
        persist()
        if state.get('pending_control'):
            pending=state['pending_control']
            request_control(pending['kind'],pending['details'])
        if state['stop_reason']:
            return state
        started=time.monotonic()
        while attempted()<max_trials and time.time()<until:
            done=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
            if 'batch_before' not in state:
                state['batch_before']=study.best_value if done else state['baseline_score']
            before=state['batch_before']
            # Interrupted batches are completed before their continuation decision.
            batch_start=state['batches'][-1]['end_trial'] if state['batches'] else 0
            count=min(batch_size-(attempted()-batch_start),max_trials-attempted())
            for _ in range(max(0,count)):
                if time.time()>=until: break
                trial=study.ask()
                try:
                    params={}
                    for name,spec in space.items():
                        if spec['type']=='categorical': params[name]=trial.suggest_categorical(name,spec['choices'])
                        elif spec['type']=='int': params[name]=trial.suggest_int(name,spec['low'],spec['high'],log=spec.get('log',False))
                        else: params[name]=trial.suggest_float(name,spec['low'],spec['high'],log=spec.get('log',False))
                    value=objective(params,trial.number,max(.1,until-time.time()))
                    study.tell(trial,float(value))
                except Exception as error:
                    trial.set_user_attr('error',str(error)[:2000])
                    study.tell(trial,state=optuna.trial.TrialState.FAIL)
                persist()
                if trial_checkpoint is not None and (attempted()==1 or study.trials[trial.number].state==optuna.trial.TrialState.FAIL):
                    finished=study.get_trials(deepcopy=False)
                    success=[t for t in finished if t.state==optuna.trial.TrialState.COMPLETE]
                    request_control('trial',dict(stage='failure' if study.trials[trial.number].state==optuna.trial.TrialState.FAIL else 'initial_fit',
                        trials=state['trials'],best_score=study.best_value if success else None,
                        best_params=study.best_params if success else {},remaining_seconds=max(0.,until-time.time()),
                        remaining_trials=max(0,max_trials-attempted()),default_action='continue' if attempted()<max_trials and time.time()<until else 'refit',
                        error=study.trials[trial.number].user_attrs.get('error')))
                    if state['stop_reason']:break
            done=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
            after=study.best_value if done else None
            # First batch without a reference can continue once; later compare bests.
            gain=relative_gain(before,after,direction) if before is not None and after is not None else None
            complete=attempted()-batch_start>=batch_size
            if complete or attempted()>=max_trials:
                state['batches'].append(dict(start_trial=batch_start,end_trial=attempted(),
                    before=before,after=after,relative_gain=gain))
                state.pop('batch_before',None)
            if state['stop_reason']:pass
            elif after is None: state['stop_reason']='no_successful_trials'
            elif complete and gain is not None and gain<gain_threshold: state['stop_reason']='hpo_plateau'
            elif attempted()>=max_trials: state['stop_reason']='max_trials'
            elif time.time()>=until: state['stop_reason']='hpo_deadline'
            if checkpoint is not None and not str(state['stop_reason']).startswith('training_agent_') and (complete or state['stop_reason'] or attempted()>=max_trials):
                persist()  # Commit study/sampler before handing control to the LLM.
                request_control('batch',dict(trials=state['trials'],best_score=after,
                    best_params=study.best_params if done else {},remaining_seconds=max(0.,until-time.time()),
                    remaining_trials=max(0,max_trials-attempted()),
                    default_action='refit' if state['stop_reason'] else 'continue'))
            if state['stop_reason']: break
        state['spent_seconds']+=time.monotonic()-started
        state['stop_reason']=state['stop_reason'] or 'hpo_deadline'
        persist()
        return state
    finally:
        storage.remove_session()
        storage.engine.dispose()
