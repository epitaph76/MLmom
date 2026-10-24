"""Replay the original failed LightGBM candidates through the repaired runtime.

No new LLM request, candidate editing, parameter search, or fallback model is used.
The original outer split is retained; the selected model is refitted on all train.
"""
import argparse
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mlmom.inputs import read_table, features, targets
from mlmom.persistence import atomic_json, file_hash, tree_hash
from mlmom.pipeline_contracts import CandidateVersion, ExperimentRun, RunConfig, TaskContext
from mlmom.resources import Budget, SandboxManager
from mlmom.submission import publish_submission, snapshot_cycle_submission
from mlmom.tools import verify_source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    run_source = source/'multiclass'/'run'
    task = output/'task'
    task.mkdir()
    for name in ['task.md', 'train.csv', 'test.csv', 'sample_submission.csv', 'data_dictionary.csv']:
        shutil.copyfile(source/'multiclass'/'task'/name, task/name)
    # A copied WAL ledger can be opened without writing lock files into the archived run.
    ledger = output/'original-experiments.sqlite3'
    shutil.copyfile(run_source/'experiments.sqlite3', ledger)
    wal = run_source/'experiments.sqlite3-wal'
    if wal.exists():
        shutil.copyfile(wal, ledger.with_name(ledger.name+'-wal'))
    with sqlite3.connect(ledger) as db:
        context = TaskContext.model_validate(json.loads(db.execute(
            "SELECT body FROM objects WHERE kind='context' AND id='task'").fetchone()[0]))
    context.files = context.files.model_copy(update=dict(root=str(task), task=str(task/'task.md'),
        train=str(task/'train.csv'), test=str(task/'test.csv'), dictionaries=[str(task/'data_dictionary.csv')],
        sample_submission=str(task/'sample_submission.csv'), manifest=None))
    context.table_formats.update({str(task/Path(path).name): fmt for path,fmt in
                                 list(context.table_formats.items()) if Path(path).name.endswith('.csv')})
    split = json.loads((run_source/'splits'/'r1'/'split.json').read_text())
    holdout = read_table(run_source/'splits'/'r1'/'holdout.csv')
    config = RunConfig(timeouts_enabled=True, total_seconds=3540, final_reserve_seconds=600,
                       reserve_cpus=4, reserve_memory_gb=2, max_workers=1)
    budget = Budget(config, time.time())
    report = dict(status='running', source=str(source), split_id=split['split_id'],
        runtime_hashes={p.name:file_hash(p) for p in [ROOT/'mlmom/execution.py']},
        inputs={p.name:file_hash(p) for p in task.iterdir()}, candidates=[])
    atomic_json(output/'report.json', report)

    def sanitized(train_path):
        value = context.model_dump(mode='json')
        value['profile_ref'] = ''
        value['files'] = dict(value['files'], root=str(train_path.parent), train=str(train_path),
                             test='', task='', dictionaries=[], sample_submission=None, manifest=None)
        return value

    with SandboxManager(budget=budget, max_workers=1) as manager:
        def worker(request, directory, seconds=600):
            result = manager.run(request, directory, seconds=min(seconds, budget.snapshot().remaining_seconds),
                                 cpus=4, memory_gb=4)
            if result.get('status') != 'completed':
                raise ValueError(str(result.get('error') or result.get('stop_reason') or result))
            return result

        def fitted(plan, candidate, identity, train_path, final=False):
            initial = ExperimentRun(experiment_id=identity, round_id='r1', candidate_id=candidate.candidate_id,
                split_id='all-train' if final else split['split_id'], role='submission_refit' if final else 'evaluation',
                status='running', resource_cpus=4, resource_memory_gb=4)
            result = worker(dict(action='train', code_dir=candidate.code_dir, code_hash=candidate.code_hash,
                context=sanitized(train_path), plan=plan, train_path=str(train_path), seed=42,
                fixed=final, phase='final' if final else 'initial'), output/'runs'/identity)
            run = ExperimentRun.model_validate(dict(initial.model_dump(mode='json'),
                **{k:v for k,v in result.items() if k in ExperimentRun.model_fields}))
            return run

        def predict(candidate, run, frame, directory):
            directory.mkdir(parents=True, exist_ok=True)
            path = directory/'features.csv'
            features(frame, context).to_csv(path, index=False)
            result = worker(dict(action='predict', code_dir=candidate.code_dir, code_hash=candidate.code_hash,
                model_artifact=run.model_artifact, features_path=str(path), kinds=['label']), directory/'worker', 120)
            return read_table(result['predictions']['label']).to_numpy().ravel()

        best = None
        for name, version in [('baseline-lgbm-multiclass', 'v1'), ('rare-class-weighted-lgbm', 'v2')]:
            row = dict(name=name, version=version, status='running')
            report['candidates'].append(row)
            atomic_json(output/'report.json', report)
            print(json.dumps(dict(event='candidate.started', name=name)), flush=True)
            try:
                code = run_source/'candidates'/'r1'/name/version
                candidate = CandidateVersion.model_validate_json((code/'metadata.json').read_text())
                candidate.code_dir = str(code)
                if tree_hash(code) != candidate.code_hash:
                    raise ValueError('Original candidate source hash changed')
                source_verification = verify_source(code)
                row['source_verification'] = source_verification.model_dump(mode='json')
                if not source_verification.valid:
                    raise ValueError('Original candidate violates source policy: ' + '; '.join(source_verification.errors))
                plan = json.loads((run_source/'plans'/f'r1-{name}.json').read_text())
                # More wall-time for replay; model parameters and all policies stay identical.
                plan['time_limit_seconds'] = 600
                smoke_request = json.loads((run_source/'checks'/f'r1-{name}-{version}'/'request.json').read_text())
                smoke_request.update(action='smoke', code_dir=str(code), code_hash=candidate.code_hash)
                smoke = worker(smoke_request, output/'checks'/name, 90)
                candidate.verification = source_verification
                candidate.execution_report = candidate.execution_report_ref = None
                trained = fitted(plan, candidate, name, run_source/'splits'/'r1'/'train.csv')
                predicted = predict(candidate, trained, holdout, output/'holdout_predictions'/name)
                from sklearn.metrics import accuracy_score, f1_score
                truth = targets(holdout, context)
                metrics = dict(macro_f1=f1_score(truth, predicted, average='macro'),
                    accuracy=accuracy_score(truth, predicted), weighted_f1=f1_score(truth, predicted, average='weighted'))
                row.update(status='passed', code_hash=candidate.code_hash, metrics=metrics,
                           internal_score=trained.internal_score, model_artifact=trained.model_artifact)
                if best is None or metrics['macro_f1'] > best[0]:
                    best = (metrics['macro_f1'], candidate, plan, trained, name)
            except Exception as exc:
                row.update(status='failed', error=str(exc))
            atomic_json(output/'report.json', report)
            print(json.dumps(dict(event='candidate.finished', **row)), flush=True)
        if best is None:
            report.update(status='failed', error='Neither original candidate passed')
            atomic_json(output/'report.json', report)
            return 2
        _, candidate, plan, trained, name = best
        final = fitted(plan, candidate, 'final-refit', task/'train.csv', final=True)
        test = read_table(task/'test.csv')
        prediction = predict(candidate, final, test, output/'test_predictions')
        artifact = publish_submission(context, test, prediction, final, candidate)
        snapshot = snapshot_cycle_submission(context, test, artifact, output, 'r1')
        shutil.copyfile(artifact.path, output/'submission.csv')
        from scripts.run_docker_queue import audit_csv
        verification = audit_csv(output/'submission.csv', task, 'attack_cat', 41166)
        replay = predict(candidate, final, test.iloc[:128], output/'saved_model_replay')
        import numpy as np
        if not np.array_equal(replay, prediction[:128]):
            raise ValueError('Saved-model replay differs from submission')
        report.update(status='passed', winner=name, holdout_macro_f1=best[0], final_submission=artifact.model_dump(),
            verification=verification, cycle_submission=snapshot, replay_equal=True, replay_rows=128)
        atomic_json(output/'report.json', report)
        print(json.dumps(dict(event='replay.completed', winner=name, holdout_macro_f1=best[0], verification=verification)), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
