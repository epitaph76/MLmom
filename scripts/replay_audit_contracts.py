"""Read-only replay of the two October 22 audits against the current contracts."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    from mlmom.execution import audit_fit,FidelityError
    from mlmom.persistence import ExperimentStore,atomic_json
    from mlmom.research import ReconnaissanceSession
    from mlmom.sdk import FitContext,load_module
    results=[]
    for label in ['qwen-real-review-20261022','critical-real-review-20261022']:
        original=ROOT/'artifacts'/label/'regression/attempt-1/task/run'
        db_path=original/'experiments.sqlite3'
        before=hashlib.sha256(db_path.read_bytes()).hexdigest()
        with sqlite3.connect(db_path.as_uri()+'?mode=ro',uri=True) as db:
            historical=json.loads(db.execute("SELECT body FROM objects WHERE kind='reconnaissance' AND id='initial'").fetchone()[0])
            csv_results=[json.loads(row[0]) for row in db.execute("SELECT body FROM objects WHERE kind='csv_result'")]
            experiments=[json.loads(row[0]) for row in db.execute("SELECT body FROM objects WHERE kind='experiment'")]
            candidates={row[0]:json.loads(row[1]) for row in db.execute("SELECT id,body FROM objects WHERE kind='candidate'")}
        with tempfile.TemporaryDirectory() as tmp:
            store=ExperimentStore(Path(tmp))
            session=ReconnaissanceSession.__new__(ReconnaissanceSession)
            session.store=store;session.key='replay';session.profile_ref='profile:unused'
            session.observations={};session.failed_saves={};session.unfinished=[];session._check_time=lambda:None
            attempts=0
            try:
                for result in csv_results:
                    source=result.get('source_ref')
                    rows=result.get('rows',[])
                    if not source or not rows:continue
                    numbers={f'row_{i}.{column}':value for i,row in enumerate(rows[:3]) for column,value in row.items()
                             if type(value) in {int,float} and math.isfinite(value)}
                    if not numbers:continue
                    attempts+=1;store.put('csv_result',source[4:],result,immutable=True)
                    session.save_observation('replayed_measurements','Exact numerical cells from a stored calculation',
                        numbers=numbers,source_ref=source)
                report=dict(run=label,csv_results=len(csv_results),historical_initial_observations=len(historical['observations']),
                    historical_initial_proposals=len(historical['proposals']),replay_attempts=attempts,
                    replay_saved_observations=len(session.observations))
                assert attempts>0 and attempts==len(session.observations),report
                atomic_json(out/(label+'-observations.json'),list(session.observations.values()))
            finally:store.close()
        if label.startswith('critical'):
            blend=next(run for run in experiments if 'h2_lgbm_catboost_blend' in run['candidate_id'] and run.get('model_artifact'))
            candidate=candidates[blend['candidate_id']]
            module=load_module(candidate['code_dir'])
            saved=module.load_candidate(blend['model_artifact'])
            estimator=getattr(saved,'estimator',saved)
            components=estimator.models
            assert components['catboost'].tree_count_==1
            plan=json.loads(Path(candidate['plan_ref']).read_text(encoding='utf-8'))
            # The shim exposes existing backend objects only; it neither refits nor repairs history.
            shim=SimpleNamespace(export_model_params=estimator.export_model_params,fitted_components=lambda:components)
            context=FitContext('regression','rmse','minimize',42,1,2,30,timeouts_enabled=False)
            path=out/'historical-blend-execution.json'
            try:
                audit_fit(shim,copy.deepcopy(plan['model_params']),plan,context,path,stage='historical_read_only')
            except FidelityError as exc:
                assert any('export' in error.lower() for error in exc.report['errors']),exc.report
                assert any('completed 1' in error for error in exc.report['errors']),exc.report
                report.update(historical_catboost_trees=1,execution_gate='rejected',execution_report=str(path))
            else:raise AssertionError('Historical broken blend passed the machine gate')
        assert before==hashlib.sha256(db_path.read_bytes()).hexdigest(),'Historical database was modified'
        report['historical_database_unchanged']=True
        results.append(report)
    atomic_json(out/'report.json',dict(status='passed',runs=results))
    print(json.dumps(dict(status='passed',runs=results),ensure_ascii=False))


if __name__=='__main__':main()
