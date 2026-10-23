"""Small planning views; full profiles, source and trial histories stay on disk."""
from __future__ import annotations

from .evaluation import compact_diagnostics


def profile_summary(profile):
    columns=[]
    for col in profile.columns:
        item=dict(name=col.name,role=col.role,type=col.semantic_type)
        if col.train.missing_rate:item['missing_rate']=col.train.missing_rate
        if str(col.semantic_type) in {'categorical','text','datetime'}:
            item['unique_count']=col.train.unique_count
        if col.tags:item['tags']=col.tags[:5]
        columns.append(item)
    return dict(dataset=profile.dataset_name,
        tables=[dict(role=f.role,rows=f.rows,columns=f.columns) for f in profile.files],
        columns=columns,checks=profile.checks.model_dump(mode='json'),
        targets={name:dict(summary={k:v for k,v in col.train.summary.items()
                                  if k in {'min','q01','median','mean','q99','max','skew','std'}},
                          distribution=[dict(value=b.label,count=b.count,rate=b.rate)
                                        for b in col.train.distribution[:20]])
                 for name in profile.task.target_columns for col in profile.columns if col.name==name},
        issues=[dict(code=i.code,severity=i.severity,columns=i.columns[:8],message=i.message[:240])
                for i in sorted(profile.issues,key=lambda item:{'error':0,'warning':1,'info':2}[item.severity])[:12]],
        hints=profile.preprocessing_hints[:8])


def plan_summary(plan):
    return {key:plan[key] for key in ('hypothesis','tool_id','skill_ids','model_params','preprocessing',
        'strategy','search_space','execution_mode','prediction_kind','contract_version','target_transform',
        'weight_policy','prediction_policy','intervention','ensemble_components','stopping_policy',
        'search_group_id','search_group_kind','parent_hypothesis','ensemble_sources','feature_set_id','feature_set') if key in plan}


def experiment_summary(run):
    trials=run.get('trials',[])
    return dict(experiment_id=run['experiment_id'],candidate_id=run.get('candidate_id'),
        round_id=run.get('round_id'),status=run.get('status'),stop_reason=run.get('stop_reason'),
        error=run.get('error'),internal_score=run.get('internal_score'),
        score_kind=run.get('score_kind','unknown_legacy'),outer_split_id=run.get('split_id'),
        validation_protocol_ref=run.get('validation_protocol_ref'),
        duration_seconds=run.get('duration_seconds'),best_params=run.get('best_params'),
        trials=dict(total=len(trials),completed=sum(t.get('status')=='completed' for t in trials),
                    failed=sum(t.get('status')=='failed' for t in trials)),
        batches=run.get('hpo_batches',[]),study_ref=run.get('study_ref'))


def evaluation_summary(evaluation):
    return {**{k:evaluation[k] for k in ('experiment_id','split_id','metric','direction','score')},
            **{k:evaluation.get(k) for k in ('score_kind','dataset_hash','metric_options','weighting','validation_protocol_ref')},
            'diagnostics':compact_diagnostics(evaluation.get('diagnostics',{}))}


def champion_snapshot(champion):
    from .execution import config_hash
    params=champion.run.best_params
    plan=plan_summary(champion.plan.model_dump(mode='json'))
    original=plan.pop('hypothesis',None)
    plan.update(model_params=params,strategy='single',search_space={})
    spec=champion.run.model_spec
    if spec:
        from .model_spec import with_spec
        plan=with_spec(plan,spec)
    return dict(candidate_id=champion.candidate.candidate_id,code_hash=champion.candidate.code_hash,
        preprocessing_hash=champion.candidate.preprocessing_hash,configuration_hash=config_hash(spec or params),model_spec=spec,
        model_params=params,plan=plan,original_hypothesis=original,
        evaluation=evaluation_summary(champion.evaluation.model_dump()),
        execution_status=(champion.run.execution_report or {}).get('status','unknown_legacy'))
