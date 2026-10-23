"""Typed capability tools. Only explicitly registered tools reach an LLM role."""
from __future__ import annotations

import ast
import inspect
import json
import time
from pathlib import Path
from typing import Any, Callable
from pydantic import create_model,ConfigDict
from .inputs import inside
from .pipeline_contracts import ToolResult, ValidationReport
from .tracing import span, emit


class ToolInputError(ValueError):
    def __init__(self, message, *, code='invalid_arguments', fields=(), hint=None, artifacts=()):
        super().__init__(message)
        self.code, self.fields, self.hint, self.artifacts = code, list(fields), hint, list(artifacts)


class ToolSet:
    def __init__(self):
        self.tools = {}

    def register(self, name: str, fn: Callable, description: str):
        hints = __import__('typing').get_type_hints(fn)
        fields = {}
        for key, parameter in inspect.signature(fn).parameters.items():
            default = ... if parameter.default is inspect.Parameter.empty else parameter.default
            fields[key] = (hints.get(key, Any), default)
        model = create_model(name + 'Args', __config__=ConfigDict(extra='forbid'), **fields)
        self.tools[name] = (fn, model, description)
        return self

    def schemas(self):
        return [{'type':'function','function':{'name':name,'description':description,'parameters':model.model_json_schema()}}
                for name, (_, model, description) in self.tools.items()]

    def invoke(self, name: str, arguments: str | dict) -> ToolResult:
        with span('tool', {'name':name,'arguments':arguments}, tool_name=name):
            result = self._invoke(name,arguments)
            emit('tool.result', result, status=result.status)
            return result

    def _invoke(self, name: str, arguments: str | dict) -> ToolResult:
        start = time.monotonic()
        try:
            fn, model, _ = self.tools[name]
            args = model.model_validate(json.loads(arguments) if isinstance(arguments,str) else arguments)
            result = fn(**args.model_dump())
            return ToolResult(status='ok', data=result, duration_seconds=time.monotonic()-start)
        except Exception as error:
            return ToolResult(status='error', error=f'{type(error).__name__}: {str(error)[:2500]}',
                              error_code=getattr(error,'code',type(error).__name__),
                              field_errors=getattr(error,'fields',[]), hint=getattr(error,'hint',None),
                              artifacts=getattr(error,'artifacts',[]),
                              duration_seconds=time.monotonic()-start)


class Workspace:
    def __init__(self, root: Path, *, revision_dir: Path | None = None, writable=None):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.revision_dir = revision_dir
        self.writable = writable

    def path(self, path: str):
        p = inside(self.root, path)
        if p.suffix not in {'.py','.json','.md','.txt'} or any(part.startswith('.') for part in p.relative_to(self.root).parts):
            raise ValueError('Only code and metadata files are allowed')
        return p

    def list_files(self) -> list[str]:
        return [p.relative_to(self.root).as_posix() for p in sorted(self.root.rglob('*'))
                if p.is_file() and p.suffix in {'.py','.json','.md','.txt'}][:100]

    def read_file(self, path: str) -> str:
        p = self.path(path)
        if p.stat().st_size > 64000: raise ValueError('File exceeds tool read limit')
        return p.read_text(encoding='utf-8')

    def write_file(self, path: str, content: str) -> str:
        if len(content) > 64000: raise ValueError('File exceeds write limit')
        p = self.path(path)
        if self.writable is not None and not self.writable(p.relative_to(self.root).as_posix()):
            raise ValueError('This file belongs to another coding role')
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding='utf-8',newline='\n')
        if self.revision_dir is not None:
            import hashlib
            from .persistence import atomic_json
            revision = self.revision_dir / f'{time.time_ns()}-{hashlib.sha256(content.encode()).hexdigest()[:12]}'
            revision.mkdir(parents=True)
            for source in self.root.rglob('*.py'):
                target = revision/source.relative_to(self.root)
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(source.read_bytes())
            atomic_json(revision/'change.json',dict(path=path,sha256=hashlib.sha256(content.encode()).hexdigest()))
        return path

    def apply_patch(self, path: str, old_text: str, new_text: str) -> str:
        text = self.read_file(path)
        if not old_text or text.count(old_text) != 1:
            raise ValueError('Patch must identify exactly one nonempty source block')
        return self.write_file(path, text.replace(old_text,new_text,1))

    def search(self, text: str) -> list[dict]:
        results = []
        for name in self.list_files():
            for line, value in enumerate(self.read_file(name).splitlines(),1):
                if text in value: results.append({'path':name,'line':line,'text':value[:300]})
                if len(results) >= 50: return results
        return results

    def apply_patches(self, patches):
        import hashlib
        pending={}
        originals={}
        for patch in patches:
            path=patch['path']
            target=self.path(path)
            if self.writable is not None and not self.writable(target.relative_to(self.root).as_posix()):
                raise ValueError('This file belongs to another coding role')
            if path not in originals:
                originals[path]=self.read_file(path)
                pending[path]=originals[path]
            if hashlib.sha256(target.read_bytes()).hexdigest()!=patch['sha256']:
                raise ValueError(f'Stale patch for {path}: source SHA256 differs')
            old,new=patch['old_text'],patch['new_text']
            if not old or pending[path].count(old)!=1:
                raise ValueError('Patch must identify exactly one nonempty source block')
            if old.strip()==originals[path].strip():
                raise ValueError('Repair cannot replace the entire existing file')
            pending[path]=pending[path].replace(old,new,1)
        for path,text in pending.items():
            self.write_file(path,text)
        return [dict(path=path,sha256=hashlib.sha256(self.path(path).read_bytes()).hexdigest()) for path in pending]

    def tools(self, *, repair=False):
        result = ToolSet()
        for name in ['list_files','read_file','write_file','apply_patch','search']:
            result.register(name, getattr(self,name), {
                'list_files':'List candidate workspace files.', 'read_file':'Read candidate source.',
                'write_file':'Create or update candidate source.', 'apply_patch':'Replace exactly one source block.',
                'search':'Search candidate source files.'}[name])
        if repair:
            result.tools.pop('write_file',None)
            def read_file(path: str):
                import hashlib
                text=self.read_file(path)
                return dict(path=path,source=text,sha256=hashlib.sha256(self.path(path).read_bytes()).hexdigest())
            result.register('read_file',read_file,'Read current source and its SHA256 for scoped patches.')
            def apply_patch(path: str, sha256: str, old_text: str, new_text: str):
                return self.apply_patches([dict(path=path,sha256=sha256,old_text=old_text,new_text=new_text)])
            result.register('apply_patch',apply_patch,'Apply one scoped repair to an existing file, with its current SHA256.')
        return result


FORBIDDEN = {'os','subprocess','socket','requests','httpx','urllib','ftplib','pip','ctypes','multiprocessing',
             'psutil','sqlite3','keyring','shutil'}
ALLOWED = {'__future__','mlmom','numpy','pandas','scipy','sklearn','catboost','lightgbm','xgboost','flaml','autogluon',
           'lightautoml','statsmodels','torch','imblearn','category_encoders','feature_engine','optuna',
           'polars','pyarrow','joblib','cloudpickle','math','statistics','typing','collections','functools',
           'itertools','dataclasses','pathlib','json','re','time','random','copy','warnings','pickle'}


def verify_source(root: Path) -> ValidationReport:
    errors, warnings = [], []
    files = list(root.rglob('*.py'))
    if not (root/'candidate.py').is_file(): errors.append('Missing candidate.py')
    local = {p.stem for p in files}
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=path.name)
        except (SyntaxError,UnicodeError) as error:
            errors.append(f'{path.name}: {error}')
            continue
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import): names = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom): names = [node.module or '']
            for name in names:
                base = name.split('.')[0]
                if base in FORBIDDEN or (base and base not in ALLOWED and base not in local):
                    errors.append(f'{path.name}: unsupported import {name}')
                if base == 'mlmom' and name != 'mlmom.sdk':
                    errors.append(f'{path.name}: infrastructure import is forbidden: {name}')
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {'eval','exec','compile','__import__'}:
                errors.append(f'{path.name}: dynamic execution is forbidden')
            if isinstance(node,ast.Attribute) and node.attr in {'environ','getenv','system','popen','from_pretrained','load_state_dict_from_url','hub'}:
                errors.append(f'{path.name}: forbidden API {node.attr}')
            if isinstance(node,ast.Constant) and isinstance(node.value,str) and ('http://' in node.value or 'https://' in node.value):
                errors.append(f'{path.name}: network URLs are forbidden')
        if path.name == 'candidate.py':
            functions = {node.name for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
            for name in ['create_candidate','load_candidate']:
                if name not in functions: errors.append(f'candidate.py must define {name}')
    return ValidationReport(valid=not errors, errors=sorted(set(errors)), warnings=warnings)
