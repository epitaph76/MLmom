"""Offline integration test on organizer archives. This is not a runtime fallback."""
from __future__ import annotations
import argparse
import json
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from mlmom.pipeline import Pipeline
from mlmom.pipeline_contracts import RunConfig
from mlmom.registry import installed_tools
from tests.fakes import OfflineLLM


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('archives',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts'/'organizer-smoke')
    parser.add_argument('--tool',choices=['sklearn','catboost'],default='catboost')
    args=parser.parse_args();reports=[]
    for i,archive in enumerate(args.archives):
        destination=(args.output/f'case-{i}').resolve();destination.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                if not (destination/member.filename).resolve().is_relative_to(destination):raise ValueError('Unsafe archive path')
            source.extractall(destination)
        tasks=list(destination.rglob('task.md'))
        if len(tasks)!=1:raise ValueError('Archive must contain one task.md')
        config=RunConfig(total_seconds=360,final_reserve_seconds=60,smoke_seconds=25,prediction_seconds=30,
                         max_rounds=2,experiment_seconds=60)
        params={'iterations':30,'depth':4} if args.tool=='catboost' else {'n_estimators':12,'max_depth':8}
        spec=installed_tools()[args.tool]
        run_dir=args.output/f'run-{i}'
        resume=(run_dir/'session.json').exists()
        with patch('mlmom.pipeline.installed_tools',lambda:{args.tool:spec}):
            result=Pipeline(tasks[0],config,run_dir=run_dir,client=OfflineLLM(tool_id=args.tool,params=params,seconds=60,cpus=2),resume=resume).run(resume=resume)
        reports.append({k:v for k,v in result.items() if k!='champion'})
        print(json.dumps(reports[-1],ensure_ascii=False))
        if result['status']!='completed':return 2
    (args.output/'reports.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
    return 0


if __name__=='__main__':raise SystemExit(main())
