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
                for i in profile.issues[:12]],
        hints=profile.preprocessing_hints[:8])


def plan_summary(plan):
    return {key:plan[key] for key in ('hypothesis','tool_id','skill_ids','model_params','preprocessing',
        'strategy','search_space','execution_mode','prediction_kind') if key in plan}


def experiment_summary(run):
    trials=run.get('trials',[])
    return dict(experiment_id=run['experiment_id'],candidate_id=run.get('candidate_id'),
        round_id=run.get('round_id'),status=run.get('status'),stop_reason=run.get('stop_reason'),
        error=run.get('error'),internal_score=run.get('internal_score'),
        duration_seconds=run.get('duration_seconds'),best_params=run.get('best_params'),
        trials=dict(total=len(trials),completed=sum(t.get('status')=='completed' for t in trials),
                    failed=sum(t.get('status')=='failed' for t in trials)),
        batches=run.get('hpo_batches',[]),study_ref=run.get('study_ref'))


def evaluation_summary(evaluation):
    return {**{k:evaluation[k] for k in ('experiment_id','split_id','metric','direction','score')},
            'diagnostics':compact_diagnostics(evaluation.get('diagnostics',{}))}
