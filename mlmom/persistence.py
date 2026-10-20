"""Single-writer experiment ledger and atomic, immutable candidate snapshots."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any
from pydantic import BaseModel


def plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode='json')
    return value


def dumps(value: Any) -> str:
    return json.dumps(plain(value), ensure_ascii=False, allow_nan=False, default=plain)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(dumps(value), encoding='utf-8')
    os.replace(temp, path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class ExperimentStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.root / 'experiments.sqlite3', check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS objects (kind TEXT, id TEXT, body TEXT NOT NULL, PRIMARY KEY(kind,id))')
        self.db.execute('CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, kind TEXT, object_id TEXT, body TEXT)')
        self.db.commit()

    def put(self, kind: str, key: str, value: Any, *, immutable: bool = False) -> None:
        body = dumps(value)
        with self.lock, self.db:
            old = self.db.execute('SELECT body FROM objects WHERE kind=? AND id=?', (kind, key)).fetchone()
            if immutable and old and old[0] != body:
                raise ValueError(f'Attempt to overwrite immutable {kind}:{key}')
            self.db.execute('INSERT OR REPLACE INTO objects VALUES (?,?,?)', (kind, key, body))

    def get(self, kind: str, key: str) -> dict | None:
        with self.lock:
            row = self.db.execute('SELECT body FROM objects WHERE kind=? AND id=?', (kind, key)).fetchone()
        return json.loads(row[0]) if row else None

    def list(self, kind: str, *, limit: int = 1000) -> list[dict]:
        with self.lock:
            rows = self.db.execute('SELECT body FROM objects WHERE kind=? ORDER BY rowid DESC LIMIT ?', (kind, limit)).fetchall()
        return [json.loads(row[0]) for row in reversed(rows)]

    def event(self, kind: str, key: str, value: Any) -> None:
        with self.lock, self.db:
            self.db.execute('INSERT INTO events(kind,object_id,body) VALUES (?,?,?)', (kind, key, dumps(value)))

    def close(self):
        self.db.close()
