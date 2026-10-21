"""Local, replayable execution traces with isolated context and secret scrubbing.

The SQLite journal is authoritative. JSONL is flushed on every event for live
inspection; large, immutable payloads are addressed by their SHA256 digest.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
import uuid


_sink: ContextVar[TraceSink | None] = ContextVar('mlmom_trace_sink', default=None)
_context: ContextVar[dict] = ContextVar('mlmom_trace_context', default={})
_SENSITIVE = {'api_key', 'apikey', 'authorization', 'proxy_authorization', 'cookie',
              'set_cookie', 'password', 'secret', 'access_token', 'refresh_token'}


def _plain(value):
    if hasattr(value, 'model_dump'):
        return value.model_dump(mode='json')
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    if isinstance(value, BaseException):
        return {'type': type(value).__name__, 'message': str(value)}
    return str(value)


def current_sink():
    return _sink.get()


def current_context():
    """Copy correlation fields before dispatching work to another thread."""
    return dict(_context.get())


def emit(kind: str, data=None, **fields):
    sink = current_sink()
    if sink is not None:
        return sink.emit(kind, data, **fields)


@contextmanager
def bind(**context):
    token = _context.set(dict(_context.get(), **context))
    try:
        yield
    finally:
        _context.reset(token)


@contextmanager
def span(kind: str, data=None, **context):
    sink = current_sink()
    if sink is None:
        with bind(**context):
            yield None
    else:
        with sink.span(kind, data, **context) as span_id:
            yield span_id


class TraceSink:
    def __init__(self, root: Path, run_id: str, task_id: str | None = None, *, secrets=()):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / 'trace_payloads').mkdir(exist_ok=True)
        self.run_id = run_id
        self.task_id = task_id
        self.started = time.monotonic()
        self.lock = threading.RLock()
        self.closed = False
        self.secrets = set()
        self.add_secrets(secrets)
        self.add_secrets(value for key, value in os.environ.items()
                         if key.upper().endswith('_TOKEN') or any(part in key.upper() for part in
                             ('API_KEY', 'APIKEY', 'ACCESS_TOKEN', 'AUTH_TOKEN', 'PASSWORD', 'SECRET', 'PRIVATE_KEY')))
        self.db = sqlite3.connect(self.root / 'trace.sqlite3', check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS trace_events (seq INTEGER PRIMARY KEY AUTOINCREMENT, body TEXT NOT NULL)')
        self.db.commit()
        # Repair a truncated or missing JSONL tail after interruption from the
        # committed journal before accepting new events.
        self.stream = (self.root / 'trace.jsonl').open('w', encoding='utf-8')
        for (body,) in self.db.execute('SELECT body FROM trace_events ORDER BY seq'):
            self.stream.write(body+'\n')
        self.stream.flush()

    def add_secrets(self, secrets):
        if isinstance(secrets,str):
            secrets=[secrets]
        with self.lock:
            self.secrets.update(str(value) for value in secrets if value)

    def scrub(self, value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                normalized = str(key).lower().replace('-', '_')
                if normalized in {'headers', 'request_headers', 'response_headers'}:
                    continue
                result[self.scrub(str(key))] = '[REDACTED]' if normalized in _SENSITIVE else self.scrub(item)
            return result
        if isinstance(value, (list, tuple, set)):
            return [self.scrub(item) for item in value]
        if not isinstance(value, (str, int, float, bool, type(None))):
            return self.scrub(_plain(value))
        if isinstance(value, str):
            with self.lock:
                secrets=sorted(self.secrets, key=len, reverse=True)
            for secret in secrets:
                value = value.replace(secret, '[REDACTED]')
            value = re.sub(r'(?i)\bBearer\s+[^\s\"\'<>]+', 'Bearer [REDACTED]', value)
        return value

    @contextmanager
    def activate(self, **context):
        sink_token = _sink.set(self)
        context_token = _context.set(dict(_context.get(), **context))
        try:
            yield self
        finally:
            _context.reset(context_token)
            _sink.reset(sink_token)

    bind = activate

    def payload(self, value):
        cleaned = self.scrub(value)
        body = json.dumps(cleaned, ensure_ascii=False, allow_nan=False, default=_plain).encode('utf-8')
        digest = hashlib.sha256(body).hexdigest()
        relative = f'trace_payloads/{digest}.json'
        path = self.root / relative
        # Never expose a partial hash-named file, including after abrupt death.
        with self.lock:
            try:
                valid_existing=hashlib.sha256(path.read_bytes()).hexdigest()==digest
            except OSError:
                valid_existing=False
            if not valid_existing:
                temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
                try:
                    temporary.write_bytes(body)
                    os.replace(temporary,path)
                finally:
                    temporary.unlink(missing_ok=True)
        return {'path': relative, 'sha256': digest, 'bytes': len(body)}

    def emit(self, kind: str, data=None, **fields):
        with self.lock:
            if self.closed:
                raise RuntimeError('Trace sink is closed')
            record = dict(schema_version=1, event_id=uuid.uuid4().hex,
                          run_id=self.run_id, task_id=self.task_id,
                          timestamp=datetime.now(timezone.utc).isoformat(),
                          elapsed_seconds=time.monotonic()-self.started)
            record.update(_context.get())
            record.update(kind=kind, **fields)
            if data is not None:
                # Keep metadata readable, put full replay data in immutable files.
                record['payload'] = self.payload(data)
            record = self.scrub(record)
            body = json.dumps(record, ensure_ascii=False, allow_nan=False, default=_plain)
            with self.db:
                cursor = self.db.execute('INSERT INTO trace_events(body) VALUES (?)', (body,))
                record['seq'] = cursor.lastrowid
                body = json.dumps(record, ensure_ascii=False, allow_nan=False)
                self.db.execute('UPDATE trace_events SET body=? WHERE seq=?', (body, cursor.lastrowid))
            self.stream.write(body+'\n')
            self.stream.flush()
            return record

    @contextmanager
    def span(self, kind: str, data=None, **context):
        started = time.monotonic()
        span_id = uuid.uuid4().hex
        parent_id = _context.get().get('span_id')
        with self.activate(**dict(context, span_id=span_id, parent_span_id=parent_id)):
            self.emit(kind+'.started', data)
            try:
                yield span_id
            except BaseException as error:
                self.emit(kind+'.failed', {'error_type': type(error).__name__, 'error': str(error)},
                          duration_seconds=time.monotonic()-started)
                raise
            else:
                self.emit(kind+'.completed', duration_seconds=time.monotonic()-started)

    def report(self):
        with self.lock:
            records = [json.loads(row[0]) for row in self.db.execute('SELECT body FROM trace_events ORDER BY seq')]
            counts = Counter(record['kind'] for record in records)
            usage = defaultdict(Counter)
            payload_cache={}
            integrity=[]
            for record in records:
                reference=record.get('payload')
                raw=None
                if reference:
                    relative=reference.get('path','')
                    if relative not in payload_cache:
                        try:
                            path=(self.root/relative).resolve()
                            if not path.is_relative_to(self.root/'trace_payloads'):
                                raise ValueError('Payload path is outside trace_payloads')
                            body=path.read_bytes()
                            if len(body)!=reference['bytes'] or hashlib.sha256(body).hexdigest()!=reference['sha256']:
                                raise ValueError('Payload size or SHA256 mismatch')
                            payload_cache[relative]=json.loads(body.decode('utf-8'))
                        except (OSError,ValueError,KeyError) as error:
                            payload_cache[relative]=None
                            integrity.append((record['seq'],relative,self.scrub(f'{type(error).__name__}: {error}')))
                    raw=payload_cache[relative]
                if record['kind']!='llm.response':
                    continue
                row=usage[(record.get('task_id') or '',record.get('role') or '')]
                row['responses']+=1
                for key in ('prompt_tokens','completion_tokens','total_tokens','cost'):
                    value=((raw or {}).get('usage') or {}).get(key)
                    if isinstance(value,(int,float)):
                        row[key]+=value
                        row[key+'_known']+=1
            lines = ['# Execution trace', '', f'Run: `{self.scrub(self.run_id)}`', '',
                     f'Events: {len(records)}. Full requests, responses, tools and skills: `trace_payloads/`.', '',
                     '| Task | Role | Responses | Prompt tokens | Completion tokens | Total tokens | Provider cost |',
                     '|---|---|---:|---:|---:|---:|---:|']
            clean = lambda value: str(value or '').replace('|', '\\|').replace('\n', ' ')
            for (task,role),row in sorted(usage.items()):
                numbers=[]
                for key in ('prompt_tokens','completion_tokens','total_tokens','cost'):
                    value = f"{row[key]:.6f}" if key=='cost' else str(row[key])
                    numbers.append(value+(' + unknown' if row[key+'_known']<row['responses'] else '')
                                   if row[key+'_known'] else 'unknown')
                lines.append(f"| {clean(task)} | {clean(role)} | {row['responses']} | "+' | '.join(numbers)+' |')
            lines.extend(['', 'Costs are provider-reported values; absent usage is unknown.', '', '| Event | Count |', '|---|---:|'])
            lines.extend(f'| {kind} | {count} |' for kind, count in sorted(counts.items()))
            if integrity:
                lines.extend(['', '## Trace integrity warnings', '',
                    'Missing or corrupt payloads prevent complete replay. Their usage is unknown; other events remain readable.', '',
                    '| First event | Payload | Error |','|---:|---|---|'])
                lines.extend(f'| {seq} | {clean(relative)} | {clean(error)} |' for seq,relative,error in integrity)
            lines.extend(['', '| Seq | Task | Stage / role | Event | Seconds | Payload |', '|---:|---|---|---|---:|---|'])
            for record in records:
                link = record.get('payload', {}).get('path')
                lines.append(f"| {record['seq']} | {clean(record.get('task_id'))} | {clean(record.get('stage'))} / {clean(record.get('role'))} | {clean(record['kind'])} | {record.get('duration_seconds', 0):.3f} | "
                             + (f'[JSON]({link})' if link else '')+' |')
            path = self.root / 'trace.md'
            path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
            return path

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.report()
            self.stream.close()
            self.db.close()
            self.closed = True
