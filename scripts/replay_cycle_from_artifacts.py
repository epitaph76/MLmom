"""Start a fresh runtime at cycle two using completed first-cycle evidence only.

Run inside Docker. The source is mounted read-only; no old session, checkpoints,
role windows, workers, studies or pending decisions are resumed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sqlite3
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlmom.persistence import atomic_json, file_hash, tree_hash
from mlmom.pipeline import Pipeline
from mlmom.pipeline_contracts import RunConfig, TaskContext, Champion, CandidateVersion, ExperimentRun
from mlmom.inputs import read_table
from mlmom.submission import validate_submission_file


def rows(source):
    with sqlite3.connect('file:' + str(source / 'experiments.sqlite3') + '?mode=ro', uri=True) as db:
        return {(kind, key): json.loads(body)
                for kind, key, body in db.execute('SELECT kind,id,body FROM objects')}


def rebase(value, old, new):
    if isinstance(value, str):
        return new + value[len(old):] if value == old or value.startswith(old + '/') else value
    if isinstance(value, list):
        return [rebase(item, old, new) for item in value]
    if isinstance(value, dict):
        return {rebase(key, old, new): rebase(item, old, new) for key, item in value.items()}
    return value


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from strings(item)


def seed(source_case: Path, work: Path):
    source_run = source_case / 'run'
    source_session = json.loads((source_run / 'session.json').read_text(encoding='utf-8'))
    old_case = str(Path(source_session['task']).parent.parent)
    old_run = old_case + '/run'
    records = rows(source_run)
    cycle = records['cycle', 'r1']
    if not cycle['completed']:
        raise ValueError('The source first cycle is not completed')
    if (work / 'run' / 'session.json').exists():
        raise ValueError('Seed requires a fresh output directory')
    work.mkdir(parents=True, exist_ok=True)
    task = work / 'task'
    task.mkdir()
    inputs = []
    for original, expected in source_session['input_hashes'].items():
        source_file = source_case / 'task' / Path(original).name
        if file_hash(source_file) != expected:
            raise ValueError('Source input changed: ' + source_file.name)
        target = task / source_file.name
        shutil.copy2(source_file, target)
        inputs.append(dict(name=target.name, sha256=file_hash(target)))

    kept = {}
    by_round = {'experiment', 'evaluation', 'component', 'submission_refit', 'feature_evidence'}
    by_prefix = {'candidate', 'plan', 'training_plan', 'preprocessing', 'result_review',
                 'search_group', 'search_group_result', 'submission_refit_status',
                 'submission_champion'}
    fixed = {('context', 'task'), ('confirmed_task', 'task'), ('round_result', 'r1'),
             ('analysis', 'r1'), ('cycle', 'r1'), ('cycle_submission', 'r1'),
             ('split', 'r1'), ('reconnaissance', 'initial'), ('reconnaissance', 'r1'),
             ('milestone', 'first_submission')}
    for (kind, key), value in records.items():
        if ((kind, key) in fixed or
            (kind in by_round and (value.get('round_id') == 'r1' or key.startswith('r1-'))) or
            (kind in by_prefix and key.startswith('r1-')) or
            (kind == 'feature_registry' and value.get('evidence', {}).get('round_id') == 'r1')):
            kept[kind, key] = value
    feature_ids = {'empty'}
    for (kind, _), value in kept.items():
        if kind == 'feature_evidence':
            feature_ids.update([value['feature_set_id'], value['parent_feature_set_id']])
        if kind == 'component' and value['plan'].get('feature_set_id'):
            feature_ids.add(value['plan']['feature_set_id'])
    for key in feature_ids:
        if ('feature_set', key) in records:
            kept['feature_set', key] = records['feature_set', key]

    copied, missing = set(), set()

    def copy_reference(original):
        if not original.startswith(old_run + '/'):
            return
        rel = Path(original[len(old_run) + 1:])
        # Only immutable initial evidence and r1 artifacts can cross the boundary.
        allowed = (rel.parts[0] == 'inspection' or
                   (rel.parts[0] == 'feature_search' and not rel.parts[1].startswith(('r2', 'r3', 'r4', 'r5'))) or
                   any(part == 'r1' or part.startswith('r1-') for part in rel.parts))
        if not allowed or rel.suffix in {'.sqlite3', '.db'}:
            return
        original_path, destination = source_run / rel, work / 'run' / rel
        if not original_path.exists():
            missing.add(str(rel))
            return
        paths = list(original_path.rglob('*')) if original_path.is_dir() else [original_path]
        for item in paths:
            if not item.is_file() or '__pycache__' in item.parts or item.suffix in {'.sqlite3', '.db', '.pyc'}:
                continue
            relative = item.relative_to(source_run)
            if str(relative) in copied:
                continue
            target = work / 'run' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if item.suffix == '.json':
                atomic_json(target, rebase(json.loads(item.read_text(encoding='utf-8')), old_case, str(work)))
            else:
                shutil.copy2(item, target)
            copied.add(str(relative))

    for value in kept.values():
        for reference in strings(value):
            copy_reference(reference)
    # The task's current CSV belongs to the final old cycle. Restore r1's immutable CSV instead.
    shutil.copy2(source_run / 'cycles/r1/submission.csv', task / 'submission.csv')
    config = RunConfig.model_validate(dict(source_session['config'], max_rounds=2,
        plateau_rounds=None, target_completed_rounds=0))
    atomic_json(work / 'config.json', config)
    state = dict(context=kept['context', 'task'], round_number=2,
        champion=kept['round_result', 'r1']['champion'], selected=kept['analysis', 'r1'],
        stale=0, stop_reason=None, planning_attempt=0, plans=[], deferred_hypotheses=None,
        submission=cycle['submission_refit']['artifact'])
    state = rebase(state, old_case, str(work))
    atomic_json(work / 'seed-state.json', state)
    manifest = dict(status='preparing', source_runtime_contract=source_session['runtime_contract_version'],
        new_runtime_contract=6, boundary='after completed r1; before planning r2',
        source_case=str(source_case), imported_objects=dict(Counter(kind for kind, _ in kept)),
        imported_keys=[dict(kind=kind, id=key) for kind, key in sorted(kept)],
        excluded_state=['session', 'checkpoints', 'old deadlines', 'workers', 'planning caches',
                        'HPO studies', 'later cycles', 'old trace and API usage'],
        copied_files=len(copied), unresolved_optional_references=sorted(missing), inputs=inputs,
        first_cycle_retrained=False, live_model='deepseek/deepseek-v4.1-flash-1017')
    atomic_json(work / 'seed-manifest.json', manifest)
    # A NEW v5 session/checkpointer is built from validated, completed evidence.
    p = Pipeline(task / Path(source_session['task']).name, config, run_dir=work / 'run',
                 client=SimpleNamespace(model='offline/seed-verification'))
    try:
        with p.trace.activate():
            for (kind, key), value in kept.items():
                p.store.put(kind, key, rebase(value, old_case, str(work)))
            context = TaskContext.model_validate(state['context'])
            Champion.model_validate(state['champion'])
            for candidate in p.store.list('candidate'):
                if tree_hash(Path(candidate['code_dir'])) != candidate['code_hash']:
                    raise ValueError('Copied candidate source differs: ' + candidate['candidate_id'])
            p.restore(state)
            p.last_submission = state['submission']
            p.store.put('submission', 'current', state['submission'])
            p.store.put('champion', 'current', state['champion'])
            p.store.put('replay_origin', 'seed', manifest)
            test = read_table(p.files.test, table_format=context.table_formats.get(p.files.test))
            csv_audit = validate_submission_file(context, test, Path(state['submission']['path']),
                                                 expected_hash=state['submission']['sha256'])
            p.preflight_runtime()
            full = p.store.get('submission_refit', 'r1-submission-refit')
            candidate = p.store.get('candidate', full['candidate_id'])
            prediction = p._prediction(CandidateVersion.model_validate(candidate),
                ExperimentRun.model_validate(full), test.iloc[:128], [context.submission_kind])
            import numpy as np
            import pandas as pd
            expected = pd.read_csv(task / 'submission.csv')[context.task.result_format.prediction_columns].iloc[:128].to_numpy()
            actual = prediction[context.submission_kind].reshape(expected.shape)
            if context.submission_kind == 'label':
                np.testing.assert_array_equal(actual, expected)
            else:
                np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
            from mlmom.search_groups import component_pool
            pool = component_pool(p.store, 2)
            if len(pool) < 2 or any(row['round_id'] != 'r1' for row in pool):
                raise ValueError('Seed has no clean first-cycle component pool')
            graph_config = {'configurable': {'thread_id': 'run'}, 'recursion_limit': 100}
            graph = p.graph()
            graph.update_state(graph_config, state, as_node='inspect')
            if graph.get_state(graph_config).next != ('plan',):
                raise ValueError('Fresh checkpoint does not enter second-cycle planning')
            manifest.update(status='ready', component_count=len(pool),
                components=[dict(experiment_id=row['experiment_id'], tool_id=row['plan']['tool_id'],
                    feature_set_id=row['plan'].get('feature_set_id')) for row in pool],
                first_cycle_csv=csv_audit, saved_model_replay=dict(rows=128, equal=True), next_stage='plan',
                seed_session_id=p.trace.run_id)
            p.store.put('replay_origin', 'seed', manifest)
            from mlmom.tracing import emit
            emit('replay.seed_ready', manifest)
    finally:
        p.sandbox.close()
        p.checkpointer.close()
        p.store.close()
        p.trace.close()
    atomic_json(work / 'seed-manifest.json', manifest)
    print(json.dumps(dict(status='ready', components=manifest['component_count'],
        first_cycle_retrained=False, saved_model_replay=manifest['saved_model_replay'],
        next_stage='plan', round=2)), flush=True)


def report(work, output):
    values = rows(work / 'run')
    events = [json.loads(line) for line in (work / 'run/trace.jsonl').read_text(encoding='utf-8').splitlines()]
    config = json.loads((work / 'config.json').read_text())
    context = TaskContext.model_validate(values['context', 'task'])
    test = read_table(context.files.test, table_format=context.table_formats.get(context.files.test))
    csv_audit = None
    submission = values.get(('submission', 'current'))
    if submission:
        csv_audit = validate_submission_file(context, test, Path(submission['path']), expected_hash=submission['sha256'])
    groups = [value for (kind, key), value in values.items() if kind == 'search_group_result' and key.startswith('r2-')]
    experiments = [value for (kind, _), value in values.items() if kind == 'experiment' and value['round_id'] == 'r2']
    result = dict(status=output['status'], stop_reason=output.get('stop_reason'),
        cycle=values.get(('cycle', 'r2')), second_cycle_groups=groups,
        new_experiments=[{k:row.get(k) for k in ['experiment_id','role','status','internal_score','score_kind',
            'split_id','duration_seconds','stop_reason','error','model_artifact']} for row in experiments],
        ensemble_proposal=values.get(('ensemble_proposal', 'r2')),
        ensemble_planning=values.get(('ensemble_planning', 'r2')),
        worker_jobs=sum(e['kind'] == 'scheduler.admitted' for e in events),
        completed_llm_calls=dict(Counter(e.get('role') for e in events if e['kind'] == 'llm.call.completed')),
        stage_entries=[dict(stage=e.get('stage'), round_id=e.get('round_id')) for e in events if e['kind'] == 'stage.started'],
        api_usage=[e for e in events if e['kind'] in {'llm.usage','runtime.llm_usage'}],
        csv_audit=csv_audit, configured_max_rounds=config['max_rounds'], summary=output)
    atomic_json(work / 'replay-report.json', result)
    print(json.dumps(dict(event='replay.finished', status=result['status'],
        ensemble_groups=[dict(group_id=g['group_id'], status=g['status'],
                             methods=[v['method'] for v in g['variants']]) for g in groups if g['kind'] == 'ensemble'],
        csv_audit=csv_audit)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['seed', 'run'])
    parser.add_argument('--source-case', type=Path)
    parser.add_argument('--work', type=Path, default=Path('/work'))
    args = parser.parse_args()
    if args.phase == 'seed':
        seed(args.source_case, args.work)
        return 0
    config = RunConfig.model_validate_json((args.work / 'config.json').read_text())
    state = json.loads((args.work / 'seed-state.json').read_text())
    p = Pipeline(Path(state['context']['files']['task']), config, run_dir=args.work / 'run', resume=True)
    output = p.run(resume=True)
    report(args.work, output)
    return 0 if output['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
