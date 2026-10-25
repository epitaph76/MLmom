"""Run organizer tasks sequentially in Docker, auditing every saved cycle CSV."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
import signal
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mlmom.persistence import atomic_json, file_hash
from mlmom.resources import terminate_tree

SPECS = [('regression', 'site_eui', 22449), ('multiclass', 'attack_cat', 41166)]


def now():
    return datetime.now(timezone.utc).isoformat()


def audit_csv(path, task, target, expected_rows):
    """Independent stdlib-only check of the actual UTF-8 organizer file."""
    with (task / 'test.csv').open(encoding='utf-8-sig', newline='') as stream:
        ids = [row['id'] for row in csv.DictReader(stream)]
    if len(ids) != expected_rows or len(set(ids)) != len(ids):
        raise ValueError('Unexpected test size or duplicate IDs')
    allowed = None
    if target == 'attack_cat':
        with (task / 'train.csv').open(encoding='utf-8-sig', newline='') as stream:
            allowed = {row[target] for row in csv.DictReader(stream)}
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.reader(stream, delimiter=',')
        if next(reader) != ['id', target]:
            raise ValueError('Organizer header mismatch')
        count = 0
        for count, row in enumerate(reader, 1):
            if len(row) != 2 or count > len(ids) or row[0] != ids[count - 1]:
                raise ValueError(f'Invalid columns or test ID/order at row {count}')
            if not row[1] or (allowed is not None and row[1] not in allowed):
                raise ValueError(f'Missing/unknown label at row {count}')
            if allowed is None and not math.isfinite(float(row[1])):
                raise ValueError(f'Nonfinite regression prediction at row {count}')
        if count != expected_rows:
            raise ValueError('Submission row count mismatch')
    return dict(status='passed', rows=count, columns=['id', target],
                sha256=file_hash(path), bytes=path.stat().st_size)


def audit_cycles(case, target, rows, checked):
    for metadata_path in sorted((case / 'run' / 'cycles').glob('r*/submission.json')):
        round_id = metadata_path.parent.name
        if round_id in checked:
            continue
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        path = metadata_path.with_suffix('.csv')
        verification = audit_csv(path, case / 'task', target, rows)
        if verification['sha256'] != metadata['artifact']['sha256']:
            raise ValueError(f'{round_id} snapshot checksum mismatch')
        checked[round_id] = dict(path=str(path), checked_at=now(), **verification)
        atomic_json(case / 'cycle_checks.json', checked)
        print(json.dumps(dict(event='cycle.checked', case=case.name, round=round_id,
                              rows=rows, sha256=verification['sha256']), ensure_ascii=False), flush=True)


def save_report(output, report):
    atomic_json(output / 'report.json', report)
    lines = ['# Последовательный запуск в Docker', '',
             f"Состояние: {report['status']}. " + (f"Лимит: {report['seconds_per_task']} секунд на задачу." if report['seconds_per_task'] else 'Общего дедлайна нет; завершение по числу полных циклов.'), '',
             '| Задача | Состояние | Проверено CSV циклов | Итоговый CSV |',
             '|---|---|---:|---|']
    for case in report['cases']:
        lines.append(f"| {case['name']} | {case['status']} | {len(case.get('cycle_checks', {}))} | "
                     f"{case.get('final_csv', {}).get('status', 'ожидание')} |")
        if case.get('error'):
            lines.extend(['', f"{case['name']}: {case['error']}"])
    lines.extend(['', 'Ответы циклов: `<задача>/run/cycles/rN/submission.csv`.',
                  'Итоговый ответ: `<задача>/submission.csv` и рядом с исходным `task.md`.',
                  'Все CSV проверяются независимо от runtime: UTF-8, заголовок, число строк, ID, порядок и значения.',
                  'Локальные holdout-метрики не являются оценкой организаторов.'])
    (output / 'REPORT_RU.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def stop_pipeline(process, grace_seconds=10):
    """Let SIGTERM persist the last completed full-data answer before forced cleanup."""
    process.terminate()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        terminate_tree(process.pid)
        process.wait()


def external_stop(signum, frame):
    raise KeyboardInterrupt


def run_case(output, record, seconds):
    name, target, rows = next(spec for spec in SPECS if spec[0] == record['name'])
    case = output / name
    run = case / 'run'
    config = output / 'config.json'
    record.update(status='running', started_at=now(), cycle_checks={})
    started = time.monotonic()
    process = None
    try:
        with (case / 'controller.stdout.log').open('w', encoding='utf-8') as out, \
             (case / 'controller.stderr.log').open('w', encoding='utf-8') as err:
            process = subprocess.Popen([sys.executable, '-u', str(case / 'task' / 'solution.py'),
                str(case / 'task' / 'task.md'), '--config', str(config), '--run-dir', str(run)],
                cwd=case / 'task', stdout=out, stderr=err)
            record['pid'] = process.pid
            while process.poll() is None:
                audit_cycles(case, target, rows, record['cycle_checks'])
                yield
                # An external wall-time guard exists only when explicitly requested.
                if seconds and time.monotonic() - started >= seconds - 55:
                    record['guard'] = 'pipeline_wall_time_limit'
                    stop_pipeline(process)
                    break
                delay = min(5, max(.1, seconds - 55 - (time.monotonic() - started))) if seconds else 5
                time.sleep(delay)
            process.wait()
        record['exit_code'] = process.returncode
        audit_cycles(case, target, rows, record['cycle_checks'])
        summary_path = run / 'summary.json'
        summary = {}
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            record['result'] = {key: summary.get(key) for key in
                ['status', 'stop_reason', 'first_submission', 'process_acceptance', 'failure_causes']}
            champion = summary.get('champion')
            if champion:
                record['holdout'] = {key: champion['evaluation'].get(key) for key in ['score', 'metric', 'split_id']}
            if json.loads(config.read_text(encoding='utf-8')).get('cycle_full_refit') and (
                    (summary.get('submission') or {}).get('model_role') != 'submission_refit'):
                raise ValueError('Only a preliminary split-trained CSV is available; no successful full-data submission')
        verification = audit_csv(case / 'task' / 'submission.csv', case / 'task', target, rows)
        shutil.copyfile(case / 'task' / 'submission.csv', case / 'submission.csv')
        record['final_csv'] = dict(path=str(case / 'submission.csv'), **verification)
        # Detect a missing answer even if the runtime reported a completed cycle.
        import sqlite3
        with sqlite3.connect(run / 'experiments.sqlite3') as db:
            cycles = [json.loads(row[0]) for row in db.execute("SELECT body FROM objects WHERE kind='cycle'")]
        missing = [cycle['round_id'] for cycle in cycles if cycle['round_id'] not in record['cycle_checks']]
        if missing:
            raise ValueError('Cycles without a verified CSV: ' + ', '.join(missing))
        if not cycles and summary.get('status') != 'interrupted':
            raise ValueError('No completed cycle records')
        record['cycles_recorded'] = len(cycles)
        remaining = seconds - (time.monotonic() - started) if seconds else None
        if summary_path.exists() and (remaining is None or remaining > 3) and summary.get('status') != 'interrupted':
            from scripts.run_real_diagnostics import validate
            record['saved_model_replay'] = validate(run, rows, min(45, remaining - 2) if remaining is not None else 120,
                                                   timeouts_enabled=bool(seconds))
        else:
            record['saved_model_replay'] = dict(status='not_run', reason='No saved summary or verification time')
        record['status'] = ('interrupted' if summary.get('status') == 'interrupted' else 'passed' if process.returncode == 0 and
            record['saved_model_replay'].get('status') == 'passed' else 'failed')
    except KeyboardInterrupt:
        if process is not None and process.poll() is None:
            stop_pipeline(process)
        record.update(status='interrupted', stop_reason='external_stop')
        raise
    except Exception as error:
        if process is not None and process.poll() is None:
            terminate_tree(process.pid)
        stopped = False
        if (run/'summary.json').is_file():
            stopped = json.loads((run/'summary.json').read_text(encoding='utf-8')).get('status') == 'interrupted'
        record.update(status='interrupted' if stopped else 'failed', error=str(error))
    finally:
        record.pop('pid', None)
        record.update(finished_at=now(), duration_seconds=round(time.monotonic() - started, 3))
    yield


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('/work'))
    parser.add_argument('--seconds', type=int, default=0, help='0 disables the external task deadline')
    args = parser.parse_args()
    output = args.output.resolve()
    if (output / 'report.json').exists():
        raise ValueError('Queue already exists; use a new output directory')
    if args.seconds < 0:
        raise ValueError('seconds must be nonnegative')
    signal.signal(signal.SIGTERM, external_stop)
    report = dict(status='running', started_at=now(), seconds_per_task=args.seconds or None,
                  model=os.environ.get('OPENAI_MODEL'), base_url=os.environ.get('OPENAI_BASE_URL'),
                  cases=[dict(name=name, status='pending') for name, _, _ in SPECS])
    save_report(output, report)
    try:
        for record in report['cases']:
            print(json.dumps(dict(event='case.started', case=record['name'], at=now())), flush=True)
            for _ in run_case(output, record, args.seconds):
                save_report(output, report)
            print(json.dumps(dict(event='case.finished', case=record['name'], status=record['status'],
                                  duration_seconds=record['duration_seconds'], at=now())), flush=True)
    except KeyboardInterrupt:
        report.update(status='interrupted', stop_reason='external_stop', finished_at=now())
        save_report(output, report)
        return 130
    report.update(status='passed' if all(c['status'] == 'passed' for c in report['cases']) else 'failed', finished_at=now())
    save_report(output, report)
    return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
