"""Summarize actual cycle/trial evidence; no inference of private platform scores."""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mlmom.persistence import atomic_json,file_hash


def report(root):
    attempts=json.loads((root/'reports.json').read_text(encoding='utf-8'))
    models=[]
    for attempt in attempts:
        manifest=Path(attempt['run_dir'])/'input_manifest.json'
        if manifest.exists():
            model=json.loads(manifest.read_text(encoding='utf-8')).get('model')
            if model and model not in models:models.append(model)
    lines=['# Реальные циклы улучшения MLmom','',
        'Модели API: '+', '.join(models or ['не указаны'])+'. Реальные обучение и Optuna. Метрики — локальный внешний holdout; '
        'test-разметка и платформенный score недоступны. Отдельный audit не использовался. Активные попытки добавляются после завершения.','',
        'Лимиты: час на задачу с учётом повторов и проверки; 4 CPU / 4 ГБ workers, один worker. '
        'Исходные ZIP сохранены. Ключи в отчёт и runtime ZIP не включаются.','',
        '| Задача / попытка | Статус | Завершённые циклы | Optuna trials | Метрика | Время, с | RSS дерева, ГБ |',
        '|---|---|---:|---:|---:|---:|---:|']
    evidence=[];deliverables=root/'deliverables';deliverables.mkdir(exist_ok=True)
    for attempt in attempts:
        run=Path(attempt['run_dir']);summary=attempt.get('result',{})
        acceptance=summary.get('process_acceptance',{})
        score=summary.get('champion',{}).get('evaluation',{}).get('score') if summary.get('champion') else None
        lines.append(f"| {attempt['case']} / {attempt['attempt']} | {summary.get('status','failed')} | "
            f"{acceptance.get('completed_rounds',0)} | {acceptance.get('completed_optuna_trials',0)} | "
            f"{score if score is not None else '—'} | {attempt['duration_seconds']:.1f} | {attempt['peak_rss_gb']:.3f} |")
        with sqlite3.connect(run/'experiments.sqlite3') as db:
            objects={}
            for kind,key,body in db.execute('SELECT kind,id,body FROM objects ORDER BY rowid'):
                objects.setdefault(kind,{})[key]=json.loads(body)
            usages=[json.loads(body) for body, in db.execute("SELECT body FROM events WHERE kind='llm_usage'")]
        usage={name:sum(u.get(name,0) or 0 for u in usages) for name in ['prompt_tokens','completion_tokens','total_tokens']}
        usage['responses']=len(usages)
        usage['provider_cost_raw']=sum(u['cost'] for u in usages if isinstance(u.get('cost'),(float,int)))
        usage['provider_cost_currency']='not specified by usage payload'
        usage['http_attempts']=sum(json.loads(line).get('kind')=='llm.request' for line in (run/'trace.jsonl').read_text(encoding='utf-8').splitlines())
        entry=dict(case=attempt['case'],attempt=attempt['attempt'],run_dir=str(run),acceptance=acceptance,
            verification=attempt.get('verification'),llm_usage=usage,cycles=list(objects.get('cycle',{}).values()),
            experiments=list(objects.get('experiment',{}).values()),evaluations=objects.get('evaluation',{}),
            stop_reason=summary.get('stop_reason'),failure_causes=summary.get('failure_causes',[]),
            planning_failures=[dict(planning_id=k,errors=v.get('errors',[]),warnings=v.get('warnings',[])) for k,v in objects.get('planning',{}).items() if v.get('errors') or v.get('warnings')])
        manifest=run/'input_manifest.json'
        if manifest.exists():
            provider=json.loads(manifest.read_text(encoding='utf-8'))
            entry['provider']={k:provider.get(k) for k in ('model','base_url')}
        evidence.append(entry)
        if summary.get('submission') and attempt.get('verification',{}).get('status')=='passed':
            output=deliverables/f"{attempt['case']}-attempt-{attempt['attempt']}-submission.csv"
            shutil.copyfile(attempt['verification']['saved_submission'],output)
            entry['deliverable']=dict(path=str(output),sha256=file_hash(output),rows=attempt['verification']['rows'],
                model_artifact=attempt['verification']['model_artifact'],candidate_id=attempt['verification']['candidate_id'])
    for entry in evidence:
        lines+=['',f"## {entry['case']}, попытка {entry['attempt']}",'',
            f"Приёмка процесса: **{'выполнена' if entry['acceptance'].get('passed') else 'не выполнена'}**. "
            f"Причина остановки: `{entry['stop_reason']}`. Проверка модели: `{(entry.get('verification') or {}).get('status','not_run')}`.",'',
            f"LLM: {entry['llm_usage']['http_attempts']} HTTP-попыток, {entry['llm_usage']['responses']} ответов с usage, {entry['llm_usage']['prompt_tokens']} входных / "
            f"{entry['llm_usage']['completion_tokens']} выходных tokens. Сумма поля cost провайдера: {entry['llm_usage']['provider_cost_raw']:.6f}; валюта не указана, в рубли не пересчитывается.",'',
            '| Цикл / эксперимент | Тип | Внутренняя метрика | Внешняя метрика | Trials | Остановка |',
            '|---|---|---:|---:|---:|---|']
        batch_lines=[]
        for experiment in entry['experiments']:
            if experiment.get('role')=='submission_refit':continue
            evaluation=entry['evaluations'].get(experiment['experiment_id'],{})
            lines.append(f"| {experiment['round_id']} / {experiment['experiment_id']} | "
                f"{'incumbent' if experiment['incumbent'] else 'candidate'} | {experiment.get('internal_score')} | "
                f"{evaluation.get('score','—')} | {len(experiment['trials'])} | {experiment.get('stop_reason')} |")
            for batch in experiment.get('hpo_batches',[]):
                batch_lines.append(f"- Пакет `{experiment['experiment_id']}` {batch['start_trial']}–{batch['end_trial']}: "
                    f"{batch['before']} → {batch['after']}, относительный прирост {batch['relative_gain']}.")
        if batch_lines:lines+=['','Пакеты Optuna:','']+batch_lines
        run=Path(entry['run_dir'])
        lines+=['',f"[Полная трасса](<{(run/'trace.md').as_posix()}>); [индекс SHA256](<{(run/'artifact_index.json').as_posix()}>)."]
        for cycle in entry['cycles']:
            # Read immutable per-round result: plans contain the exact evidence cited by the LLM.
            with sqlite3.connect(run/'experiments.sqlite3') as db:
                row=db.execute("SELECT body FROM objects WHERE kind='round_result' AND id=?",(cycle['round_id'],)).fetchone()
            if row:
                body=json.loads(row[0]);lines+=['',f"### {cycle['round_id']}: гипотезы и диагностика",'']
                seen=set()
                for item in body['round_results']:
                    if item['run']['incumbent']:continue
                    hypothesis=item['plan']['hypothesis'];key=hypothesis['hypothesis_id']
                    if key not in seen:
                        lines.append(f"- **{key}:** {hypothesis['description']} Обоснование: {hypothesis['rationale']}")
                        seen.add(key)
                    diagnosis=item['evaluation']['diagnostics']
                    inspection=diagnosis.get('model_inspection',{})
                    states={k:v.get('status') for k,v in inspection.items() if isinstance(v,dict)}
                    lines.append(f"- `{item['run']['experiment_id']}`: score={item['evaluation']['score']}; "
                        f"диагностика {states}; [артефакт](<{Path(diagnosis['artifact_ref']).as_posix()}>)" if diagnosis.get('artifact_ref') else
                        f"- `{item['run']['experiment_id']}`: score={item['evaluation']['score']}; базовая диагностика сохранена в evaluation.")
        if entry['planning_failures']:
            lines+=['','Отклонённые планы / деградации проверки:','']+[f"- `{failure['planning_id']}`: {failure['errors']}; warnings={failure['warnings']}" for failure in entry['planning_failures']]
        failed_workers=[e for e in entry['experiments'] if e.get('status')=='failed']
        if failed_workers:
            lines+=['','Ошибки workers:','']+[f"- `{e['experiment_id']}`: {(e.get('error') or e.get('stop_reason') or 'unknown').splitlines()[-1]}" for e in failed_workers]
        if entry['failure_causes']:
            lines+=['','Ошибки и отказы:','']+[f"- `{failure['kind']}`: {failure.get('details')}" for failure in entry['failure_causes']]
    latest={item['case']:item for item in evidence}
    passed=set(latest)=={'regression','multiclass'} and all(e['acceptance'].get('passed') and
        (e.get('verification') or {}).get('status')=='passed' for e in latest.values())
    audit=dict(status='passed' if passed else 'incomplete',attempts=evidence)
    atomic_json(root/'audit.json',audit)
    verified={e['case']:dict(e['deliverable']) for e in evidence if e.get('deliverable')}
    for name,item in verified.items():
        canonical=deliverables/f'{name}-submission.csv'
        shutil.copyfile(item['path'],canonical)
        item['saved_submission']=item['path'];item['path']=str(canonical)
    atomic_json(deliverables/'model_manifest.json',list(verified.values()))
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=audit['status'],report=str(root/'REPORT.md'))))
    return 0 if passed else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    raise SystemExit(report(parser.parse_args().root.resolve()))
