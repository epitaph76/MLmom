"""Bounded model inspection in a supervised worker. No new dependencies."""
from __future__ import annotations

import time
import numpy as np
import pandas as pd


def inspect_model(candidate,X,y,context,*,seconds=60,max_rows=1000,max_features=20,groups=None,weights=None):
    from .evaluation import score_predictions,prediction_kind
    started=time.monotonic()
    rng=np.random.default_rng(42)
    indices=np.sort(rng.choice(len(X),min(max_rows,len(X)),replace=False))
    X=X.iloc[indices].copy();y=y.iloc[indices] if y is not None else None
    groups=np.asarray(groups)[indices] if groups is not None else None
    weights=np.asarray(weights)[indices] if weights is not None else None
    report=dict(rows=len(X),feature_importance={'status':'unavailable'},
        contributions={'status':'unavailable'},permutation_importance={'status':'unavailable'})
    model=getattr(candidate,'model',None)
    importance=getattr(model,'feature_importances_',None)
    if importance is not None and len(importance)==len(X.columns):
        ranked=sorted(zip(X.columns,map(float,importance)),key=lambda item:-item[1])
        report['feature_importance']=dict(status='completed',values=dict(ranked[:max_features]))
    else:ranked=[(c,0.) for c in X.columns]
    try:
        if getattr(candidate,'tool',None) in {'lightgbm','catboost'} and time.monotonic()-started<seconds*.4:
            values=candidate._clean(X.head(min(128,len(X))))
            if candidate.tool=='lightgbm':
                for column,categories in getattr(candidate,'native_categories',{}).items():
                    values[column]=pd.Categorical(values[column],categories=categories)
                if candidate.preprocessor is not None:values=candidate.preprocessor.transform(values)
                contribution=np.asarray(model.predict(values,pred_contrib=True))
                size=values.shape[1]+1
                contribution=contribution.reshape(len(values),-1,size)
            else:
                from catboost import Pool
                for column in candidate.category:values[column]=values[column].fillna('__missing__').astype(str)
                contribution=np.asarray(model.get_feature_importance(Pool(values,cat_features=candidate.category),
                    type='ShapValues',thread_count=candidate.context.cpus))
                if contribution.ndim==2:contribution=contribution[:,None,:]
            aggregate=np.abs(contribution[...,:-1]).mean(axis=(0,1))
            names=list(X.columns) if len(aggregate)==len(X.columns) else [f'encoded_{i}' for i in range(len(aggregate))]
            report['contributions']=dict(status='completed',mean_absolute=dict(sorted(zip(names,map(float,aggregate)),key=lambda item:-item[1])[:max_features]))
    except Exception as error:report['contributions']=dict(status='unavailable',reason=str(error)[:300])
    try:
        kind=prediction_kind(context.task.primary_metric,context.task.task_type)
        base=score_predictions(y,candidate.predict(X,{'kind':kind}),context,X=X,groups=groups,weights=weights)
        values={}
        for column,_ in ranked[:max_features]:
            if time.monotonic()-started>=seconds*.85:break
            shuffled=X.copy();shuffled[column]=rng.permutation(shuffled[column].to_numpy())
            score=score_predictions(y,candidate.predict(shuffled,{'kind':kind}),context,X=shuffled,groups=groups,weights=weights)
            values[column]=float(base-score if context.task.metric_direction=='maximize' else score-base)
        report['permutation_importance']=dict(status='completed' if len(values)==min(max_features,len(X.columns)) else 'partial',
            baseline=base,values=values)
    except Exception as error:report['permutation_importance']=dict(status='unavailable',reason=str(error)[:300])
    report['duration_seconds']=time.monotonic()-started
    return report
