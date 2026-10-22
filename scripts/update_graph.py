"""Refresh the code graph locally without semantic models or ML runtime dependencies."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


def main():
    root=Path(__file__).resolve().parents[1]
    lock=json.loads((root/'graphify.lock.json').read_text(encoding='utf-8'))
    executable=root/lock['environment_directory']/('Scripts/graphify.exe' if os.name=='nt' else 'bin/graphify')
    if not executable.is_file():
        raise SystemExit('Install the pinned Graphify checkout into .tools/graphify-env; see docs/GRAPHIFY.md')
    python=executable.with_name('python.exe' if os.name=='nt' else 'python')
    version=subprocess.check_output([str(python),'-c',
        'from importlib.metadata import version; print(version("graphifyy"))'],text=True).strip()
    if version!=lock['version']:raise SystemExit('Installed Graphify version differs from graphify.lock.json')
    source=root/lock['source_directory']
    if source.exists():
        commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
        if commit!=lock['commit']:raise SystemExit('Graphify source differs from graphify.lock.json')
    env=dict(os.environ,GRAPHIFY_NO_AUTO_REFRESH='1',GRAPHIFY_OUT=str(root/'graphify-out'),PYTHONUTF8='1')
    # No discovery of a semantic backend through ambient credentials.
    for key in list(env):
        if key in {'PYTHONPATH','PYTHONHOME'} or key.endswith('_API_KEY') or key.startswith(('OPENAI_','ANTHROPIC_','GOOGLE_','GEMINI_','GRAPHIFY_BACKEND')):
            env.pop(key,None)
    for args in (['extract',str(root),'--code-only','--no-dedup','--force'],
                 ['cluster-only',str(root),'--no-label']):
        subprocess.run([str(executable),*args],cwd=root,env=env,check=True)
    for name in ('graph.json','graph.html','GRAPH_REPORT.md'):
        path=root/'graphify-out'/name
        if not path.is_file():raise SystemExit(f'Missing graph output: {name}')
    print('Local code graph refreshed: graphify-out/graph.html')


if __name__=='__main__':main()
