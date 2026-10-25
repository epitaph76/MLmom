"""Short real-role diagnostics on small datasets, separate from release acceptance."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
MODEL='deepseek/deepseek-v4.1-flash-1017'


def create_task(root, name):
    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_classification
    root.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(42)
    count=1000 if name=='multiclass' else 640
    if name=='text_linear':
        labels=np.where(np.arange(count)%2,'spam','ham')
        words={'spam':['offer buy now','bonus скидка акция','limited free prize'],
               'ham':['meeting report today','привет документ встреча','team project update']}
        texts=[words[label][i%3]+' '+str(i%17) for i,label in enumerate(labels)]
        frame=pd.DataFrame(dict(text=texts,message_length=[len(t) for t in texts],
                                sent_links=rng.integers(0,3,count)))
        frame['target']=labels
        kind,metric='binary_classification','macro_f1'
        semantics='Classify messages using a sparse classical TF-IDF representation and a linear sklearn estimator. No boosting models: this diagnostic exercises the sparse non-boosting path. Numeric message_length and sent_links are available at inference.'
    elif name=='multiclass':
        values,labels=make_classification(n_samples=count,n_features=10,n_informative=7,
            n_redundant=1,n_classes=10,n_clusters_per_class=1,random_state=42)
        frame=pd.DataFrame(values,columns=[f'sensor_{i}' for i in range(10)])
        frame['target']=[f'category_{i}' for i in labels]
        kind,metric='multiclass_classification','macro_f1'
        semantics='Predict one of ten device categories from numerical sensor measurements. All sensor values are available at inference. Test useful train-only feature interactions against a frozen control.'
    else:
        area=rng.lognormal(3,1,count);load=rng.uniform(.2,3,count)
        frame=pd.DataFrame(dict(floor_area=area,load=load,temperature=rng.normal(15,8,count),
                                age=rng.integers(0,80,count),facility_type=rng.choice(['office','school','shop'],count)))
        frame['target']=area*load+.5*frame.temperature**2+rng.normal(0,4,count)
        kind,metric='regression','rmse'
        semantics='Predict building energy consumption. floor_area is building size, load is activity per unit area, temperature is outdoor temperature, and age is building age. These inputs are available at inference; area/activity interactions are plausible feature experiments.'
    frame.insert(0,'id',np.arange(count))
    test=frame.iloc[-100:].drop(columns='target').copy()
    test['id']+=100000
    train=frame.iloc[:-100].copy()
    train.to_csv(root/'train.csv',index=False);test.to_csv(root/'test.csv',index=False)
    test[['id']].assign(target=0 if kind=='regression' else train.target.iloc[0]).to_csv(root/'sample_submission.csv',index=False)
    pd.DataFrame(dict(feature=train.columns,role=['id']+['feature']*(len(train.columns)-2)+['target'])).to_csv(root/'data_dictionary.csv',index=False)
    (root/'dataset_manifest.json').write_text(json.dumps(dict(train='train.csv',dictionaries=['data_dictionary.csv'],
        task_description='task.md',task_type=kind,target_columns=['target'],id_columns=['id'],primary_metric=metric)),encoding='utf-8')
    (root/'task.md').write_text(semantics+f'\nPredict target; primary metric {metric}. Use a small fixed baseline and 3-fold internal validation. '
        'File submission.csv: UTF-8 CSV, comma delimiter, no index, columns id,target. '
        'One finite value or original class label per test ID, in test.csv order. Train/test files are in this directory.',encoding='utf-8')
    return root/'task.md',metric,len(test)


def inspect_trace(run):
    events=[json.loads(line) for line in (run/'trace.jsonl').read_text(encoding='utf-8').splitlines()]
    def payload(event):
        return json.loads((run/event['payload']['path']).read_text(encoding='utf-8')) if event.get('payload') else {}
    feature=[e['seq'] for e in events if e['kind']=='feature.proposal_prepared']
    hpo=[e['seq'] for e in events if e['kind']=='scheduler.admitted' and payload(e).get('directory','').endswith('-hpo')]
    first=[e['seq'] for e in events if e['kind']=='submission.first']
    usage=[payload(e) for e in events if e['kind']=='runtime.llm_usage']
    return dict(feature_proposal_events=feature,hpo_admissions=hpo,first_full_submission_events=first,
        feature_before_hpo=bool(feature and (not hpo or min(feature)<min(hpo))),
        first_submission_before_hpo=bool(first and (not hpo or min(first)<min(hpo))),
        feature_measurements=sum(e['kind']=='feature.evaluated' for e in events),
        prompt_tokens=sum(u.get('prompt_tokens',0) for u in usage),
        completion_tokens=sum(u.get('completion_tokens',0) for u in usage),
        provider_cost=sum(float(u['cost']) for u in usage if u.get('cost') is not None) if any(u.get('cost') is not None for u in usage) else None,
        provider_currency='unspecified')


def verify_sparse_path(run, summary):
    import sqlite3
    from scipy.sparse import issparse
    from mlmom.pipeline_contracts import TaskContext
    from mlmom.sdk import load_module
    from mlmom.inputs import read_table,features
    with sqlite3.connect(run/'experiments.sqlite3') as db:
        context=TaskContext.model_validate(json.loads(db.execute("SELECT body FROM objects WHERE kind='context' AND id='task'").fetchone()[0]))
        tools={json.loads(row[0])['tool_id'] for row in db.execute("SELECT body FROM objects WHERE kind='training_plan'")}
    if tools != {'sklearn'}: raise ValueError('The non-boosting scenario did not use the requested sklearn path')
    records=[json.loads(path.read_text(encoding='utf-8')) for path in (run/'submissions').glob('*.json')]
    metadata=next(record for record in records if record['artifact']['sha256']==summary['submission']['sha256'])
    model=load_module(metadata['code_dir']).load_candidate(metadata['model_artifact'])
    matrix=model.preprocessor.transform(features(read_table(context.files.test),context))
    if not issparse(matrix): raise ValueError('The text scenario did not preserve its sparse representation')
    return dict(tools=sorted(tools),sparse=True,transformed_shape=list(matrix.shape))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case',choices=['regression','multiclass','text_linear','all'],default='all')
    parser.add_argument('--verify-only',action='store_true',help='Recheck saved results without API or fitting')
    args=parser.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    from mlmom.pipeline import Pipeline
    from mlmom.pipeline_contracts import RunConfig
    from mlmom.persistence import atomic_json
    from scripts.release_cases import verify_run
    # Explicit diagnostic selection; never changes the environment file or release model.
    os.environ['OPENAI_MODEL']=MODEL
    reports=[]
    for name in ['regression','multiclass','text_linear'] if args.case=='all' else [args.case]:
        case=args.output/name
        if args.verify_only:
            import pandas as pd
            result=json.loads((case/'run/summary.json').read_text(encoding='utf-8'))
            manifest=json.loads((case/'task/dataset_manifest.json').read_text(encoding='utf-8'))
            metric=manifest['primary_metric'];rows=len(pd.read_csv(case/'task/test.csv'))
            previous=json.loads((case/'verification.json').read_text(encoding='utf-8'))
            if not (case/'verification.initial.json').exists(): atomic_json(case/'verification.initial.json',previous)
            duration=previous['duration_seconds']
        else:
            task,metric,rows=create_task(case/'task',name)
            config=RunConfig(profile='dev',timeouts_enabled=False,max_rounds=1,hpo_initial_trials=5,hpo_max_trials=5,
                hpo_batch_size=5,feature_max_configurations=2,feature_plateau_limit=2,
                reconnaissance_steps=8,agent_steps=12,diagnostic_seconds=0,reserve_cpus=0,reserve_memory_gb=1,
                max_workers=3,preparation_workers=3)
            atomic_json(case/'config.json',config)
            started=time.monotonic()
            result=Pipeline(task,config,run_dir=case/'run').run()
            duration=time.monotonic()-started
        report=dict(name=name,model=MODEL,status=result['status'],duration_seconds=duration,
                    stop_reason=result.get('stop_reason'),trace=inspect_trace(case/'run'))
        try:
            report['replay']=verify_run(case/'run',rows,metric,minimum_cycles=1,require_observation_handoff=True)
            if name=='text_linear': report['sparse_path']=verify_sparse_path(case/'run',result)
            if not report['trace']['feature_before_hpo'] or not report['trace']['first_submission_before_hpo']:
                raise ValueError('Early feature proposal or first full-data submission was not observed')
            if name=='regression' and not report['trace']['feature_measurements']:
                raise ValueError('The interaction scenario produced no actual feature measurement')
            report['status']='passed'
        except Exception as error:
            report.update(status='failed',error=str(error))
        atomic_json(case/'verification.json',report);reports.append(report)
        atomic_json(args.output/'report.json',dict(model=MODEL,cases=reports,status='passed' if all(r['status']=='passed' for r in reports) else 'failed'))
        print(json.dumps(report,ensure_ascii=False),flush=True)
    return 0 if all(r['status']=='passed' for r in reports) else 1


if __name__=='__main__':raise SystemExit(main())
