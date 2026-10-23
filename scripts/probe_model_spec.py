"""Supplemental worker-level v2 policy/HPO/refit and isolated inference evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    runtime=out/'runtime';runtime.mkdir(exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:archive.extractall(runtime)
    sys.path.insert(0,str(runtime));sys.path.append(str(ROOT))
    import solution
    import numpy as np
    from mlmom.model_spec import TargetTransform,WeightPolicy,PredictionPolicy,with_spec
    from mlmom.persistence import tree_hash
    from mlmom.pipeline_contracts import HypothesisPlan
    from mlmom.resources import SandboxManager
    from mlmom.inputs import features,read_table
    from tests.test_execution import execution_case
    reports=[]
    for kind in ('regression','multiclass_classification'):
        directory=out/kind;directory.mkdir(exist_ok=True)
        frame,context,plan,code=execution_case(directory,kind)
        if kind=='regression':
            target=context.task.target_columns[0]
            frame[target]=np.exp(np.linspace(0,4,len(frame)))
            plan.target_transform=TargetTransform(name='log1p',clip_min=0)
            plan.weight_policy=WeightPolicy(name='tail_aware')
            plan.prediction_policy=PredictionPolicy(clip_min=0)
            plan.strategy='optuna';plan.max_trials=2
            plan.search_space={'policy_params.weight_policy.quantile':dict(type='categorical',choices=[.9,.99])}
        else:
            plan.weight_policy=WeightPolicy(name='sqrt_balanced')
        train=directory/'train.csv';frame.to_csv(train,index=False)
        with SandboxManager(timeouts_enabled=True) as manager:
            smoke=manager.run(dict(action='smoke',code_dir=str(code),code_hash=tree_hash(code),config=plan.model_dump(mode='json'),
                task_type=kind,metric=context.task.primary_metric,prediction_kind=plan.prediction_kind),
                directory/'smoke',seconds=45,memory_gb=2,cpus=1)
            if smoke['status']!='completed':raise ValueError('Policy smoke failed: '+str(smoke))
            fitted=manager.run(dict(action='train',code_dir=str(code),code_hash=tree_hash(code),context=context.model_dump(mode='json'),
                plan=plan.model_dump(mode='json'),train_path=str(train),seed=42,hpo_max_trials=2,hpo_batch_size=2),
                directory/'fit',seconds=100,memory_gb=2,cpus=1)
            if fitted['status']!='completed':raise ValueError('Policy train failed: '+str(fitted))
            if any(key.startswith('policy_params.') for key in fitted['best_params']):raise ValueError('Policy leaked into backend params')
            spec=fitted['model_spec']
            frozen=HypothesisPlan.model_validate(dict(with_spec(plan.model_dump(mode='json'),spec),strategy='single',search_space={}))
            final=manager.run(dict(action='train',phase='final',fixed=True,code_dir=str(code),code_hash=tree_hash(code),
                context=context.model_dump(mode='json'),plan=frozen.model_dump(mode='json'),train_path=str(train),seed=42),
                directory/'final',seconds=60,memory_gb=2,cpus=1)
            if final['status']!='completed':raise ValueError('Policy final refit failed: '+str(final))
            if final['model_spec']!=spec:raise ValueError('Final refit changed the selected ModelSpec')
            X=features(frame,context).head(32);input_path=directory/'features.csv';X.to_csv(input_path,index=False)
            result=manager.run(dict(action='predict',code_dir=str(code),code_hash=tree_hash(code),model_artifact=final['model_artifact'],
                features_path=str(input_path),kinds=[plan.prediction_kind]),directory/'prediction',seconds=30,memory_gb=2,cpus=1)
            if result['status']!='completed':raise ValueError('Policy inference failed: '+str(result))
        proof=out/'isolated'/kind;proof.mkdir(parents=True,exist_ok=True)
        for entry in ('solution.py','mlmom_bundle.py','requirements.txt'):shutil.copyfile(runtime/entry,proof/entry)
        shutil.copytree(code,proof/'code',dirs_exist_ok=True)
        shutil.copytree(final['model_artifact'],proof/'model',dirs_exist_ok=True)
        shutil.copyfile(input_path,proof/'features.csv')
        predictions=read_table(result['predictions'][plan.prediction_kind]).to_numpy()
        (proof/'expected.json').write_text(json.dumps(predictions.tolist()),encoding='utf-8')
        (proof/'prediction.json').write_text(json.dumps({'kind':plan.prediction_kind}),encoding='utf-8')
        (proof/'verify.py').write_text('''import json
from pathlib import Path
import solution
import numpy as np
import pandas as pd
from mlmom.sdk import load_module
root=Path(__file__).resolve().parent
model=load_module(root/'code').load_candidate(str(root/'model'))
expected=np.asarray(json.loads((root/'expected.json').read_text()))
kind=json.loads((root/'prediction.json').read_text())['kind']
actual=np.asarray(model.predict(pd.read_csv(root/'features.csv'),{'kind':kind})).reshape(expected.shape)
assert np.array_equal(expected,actual) or (kind=='value' and np.allclose(expected,actual,rtol=1e-9,atol=1e-9))
print(json.dumps({'status':'passed','rows':len(actual),'original_case_mounted':False}))
''',encoding='utf-8')
        reports.append(dict(task_type=kind,status='passed',smoke_roundtrip=smoke['execution_report']['roundtrip'],
            policy=spec['weight_policy']['name'],target_transform=spec['target_transform']['name'],
            selected_quantile=spec['weight_policy']['quantile'],hpo_trials=len(fitted['trials']),
            full_refit=final['full_refit'],model_spec_preserved=True,isolated_directory=str(proof)))
    report=dict(status='passed',archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        checks=reports,failures=0,errors=0,network_used=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
