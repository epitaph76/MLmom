"""Organizer entrypoint: python solution.py path/to/task.md."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

STARTED = time.time()

# Boosters copies only root .py files. Restore our source and skills at runtime.
if Path(__file__).with_name('mlmom_bundle.py').is_file():
    from mlmom_bundle import bootstrap
    bootstrap(Path(__file__).resolve().parent)


def main():
    import sys
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description='MLmom autonomous tabular research pipeline')
    parser.add_argument('task',type=Path,nargs='?',default=Path('task.md'))
    parser.add_argument('--config',type=Path)
    parser.add_argument('--profile',choices=['dev','platform'])
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    from mlmom.pipeline_contracts import RunConfig
    from mlmom.pipeline import Pipeline
    raw=json.loads(args.config.read_text(encoding='utf-8')) if args.config else {}
    if args.profile:raw['profile']=args.profile
    config=RunConfig.model_validate(raw)
    try:
        pipeline=Pipeline(args.task,config,run_dir=args.run_dir,resume=args.resume,started=STARTED)
        result=pipeline.run(resume=args.resume)
    except Exception as error:
        result={'status':'failed','error':str(error)}
    print(json.dumps({k:v for k,v in result.items() if k!='champion'},ensure_ascii=False))
    return 0 if result['status']=='completed' else 2


if __name__=='__main__':raise SystemExit(main())
