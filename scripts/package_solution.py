"""Build Boosters' flat ZIP from reviewable source, without secrets."""
from __future__ import annotations
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


BOOTSTRAP = '''
def bootstrap(directory):
    import hashlib
    import sys
    from pathlib import Path
    root = Path(directory).resolve() / ('.mlmom-runtime-' + DIGEST[:16])
    for name, source in FILES.items():
        target = root / name
        if not target.resolve().is_relative_to(root):
            raise ValueError('Invalid bundled source path')
        content = source.encode('utf-8')
        if not target.is_file() or target.read_bytes() != content:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + '.tmp')
            temporary.write_bytes(content)
            temporary.replace(target)
    actual = hashlib.sha256(json.dumps(FILES,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    if actual != DIGEST:
        raise ValueError('Bundled source integrity check failed')
    sys.path.insert(0, str(root))
    return root
'''


def package(root: Path, output: Path):
    paths = [p for folder in ['mlmom','skills','profiles'] for p in (root/folder).rglob('*')
             if p.is_file() and p.suffix in {'.py','.json','.md'} and '__pycache__' not in p.parts]
    files = {p.relative_to(root).as_posix():p.read_text(encoding='utf-8') for p in sorted(paths)}
    digest = hashlib.sha256(json.dumps(files,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    bundle = ('# Generated source bundle for Boosters. No third-party packages.\nimport json\n'
              + 'FILES = '+repr(files)+'\nDIGEST = '+repr(digest)+'\n'+BOOTSTRAP)
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.write(root/'solution.py','solution.py')
        archive.write(root/'requirements.txt','requirements.txt')
        archive.writestr('mlmom_bundle.py',bundle)
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {'solution.py','requirements.txt','mlmom_bundle.py'}
        assert not any('/' in name for name in archive.namelist())
    return 3


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('dist/mlmom-solution.zip'))
    args=parser.parse_args();root=Path(__file__).resolve().parent.parent
    print(f'Packaged {package(root,args.output.resolve())} root files: {args.output.resolve()}')
