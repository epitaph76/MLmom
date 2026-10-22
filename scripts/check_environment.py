"""Record the actual supplied-dependency match, without installing anything."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform


def inspect(requirements):
    packages={}
    for line in requirements.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):continue
        name,expected=line.split('==',1);name=name.split('[')[0]
        try:actual=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:actual=None
        match=actual is not None and (actual==expected or name=='torch' and actual==expected+'+cpu')
        packages[name]=dict(expected=expected,actual=actual,matches=match)
    return dict(status='passed' if platform.system()=='Linux' and platform.python_version().startswith('3.12.') and all(p['matches'] for p in packages.values()) else 'failed',
        os=platform.system(),python=platform.python_version(),packages=packages,
        requirements_sha256=hashlib.sha256(requirements.read_bytes()).hexdigest(),
        limitations=['CPU-only Torch may carry +cpu; Python patch release may differ from organizer.',
                     'Matching dependencies does not establish organizer-platform equivalence.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--requirements',type=Path,default=Path(__file__).resolve().parents[1]/'requirements.txt')
    args=parser.parse_args();report=inspect(args.requirements)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],python=report['python'],mismatches={n:p for n,p in report['packages'].items() if not p['matches']})))
    raise SystemExit(0 if report['status']=='passed' else 1)
