"""SQLite checkpointer for JSON-only LangGraph state; no extra SQLite package."""
from __future__ import annotations
import asyncio
import json
import sqlite3
import threading
from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple, WRITES_IDX_MAP


class JsonSerializer:
    def dumps_typed(self, value):
        return 'json', json.dumps(value, ensure_ascii=False, allow_nan=False,
                                 default=lambda x: {'error':str(x)} if isinstance(x,BaseException) else _unsupported(x)).encode('utf-8')

    def loads_typed(self, value):
        if value[0] != 'json': raise ValueError('Unsupported checkpoint encoding')
        return json.loads(value[1])


def _unsupported(value):
    raise TypeError(f'Non-JSON checkpoint value: {type(value).__name__}')


class SQLiteCheckpointer(BaseCheckpointSaver):
    def __init__(self, path):
        super().__init__(serde=JsonSerializer())
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.RLock()
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS checkpoints(thread TEXT,ns TEXT,id TEXT,parent TEXT,body BLOB,metadata TEXT,PRIMARY KEY(thread,ns,id))')
        self.db.execute('CREATE TABLE IF NOT EXISTS writes(thread TEXT,ns TEXT,id TEXT,task TEXT,idx INTEGER,channel TEXT,value BLOB,PRIMARY KEY(thread,ns,id,task,idx))')
        self.db.commit()

    def put(self, config, checkpoint, metadata, new_versions):
        c = config['configurable']; ns = c.get('checkpoint_ns','')
        _, body = self.serde.dumps_typed(checkpoint)
        with self.lock, self.db:
            self.db.execute('INSERT OR REPLACE INTO checkpoints VALUES(?,?,?,?,?,?)',
                (c['thread_id'],ns,checkpoint['id'],c.get('checkpoint_id'),body,json.dumps(metadata)))
        return {'configurable':{'thread_id':c['thread_id'],'checkpoint_ns':ns,'checkpoint_id':checkpoint['id']}}

    def get_tuple(self, config):
        c = config['configurable']; ns = c.get('checkpoint_ns','')
        query = 'SELECT id,parent,body,metadata FROM checkpoints WHERE thread=? AND ns=?'
        args = [c['thread_id'],ns]
        if c.get('checkpoint_id'): query += ' AND id=?'; args.append(c['checkpoint_id'])
        query += ' ORDER BY id DESC LIMIT 1'
        with self.lock:
            row = self.db.execute(query,args).fetchone()
            if not row: return None
            ident,parent,body,metadata = row
            writes = self.db.execute('SELECT task,channel,value FROM writes WHERE thread=? AND ns=? AND id=? ORDER BY task,idx',
                                     (c['thread_id'],ns,ident)).fetchall()
        make_config = lambda id: {'configurable':{'thread_id':c['thread_id'],'checkpoint_ns':ns,'checkpoint_id':id}}
        return CheckpointTuple(make_config(ident), self.serde.loads_typed(('json',body)),json.loads(metadata),
            make_config(parent) if parent else None,
            [(task,channel,self.serde.loads_typed(('json',value))) for task,channel,value in writes])

    def put_writes(self, config, writes, task_id, task_path=''):
        c = config['configurable']
        with self.lock, self.db:
            for i,(channel,value) in enumerate(writes):
                idx = WRITES_IDX_MAP.get(channel,i)
                mode = 'REPLACE' if channel in WRITES_IDX_MAP else 'IGNORE'
                self.db.execute(f'INSERT OR {mode} INTO writes VALUES(?,?,?,?,?,?,?)',
                    (c['thread_id'],c.get('checkpoint_ns',''),c['checkpoint_id'],task_id,idx,channel,self.serde.dumps_typed(value)[1]))

    def list(self, config, *, filter=None, before=None, limit=None):
        with self.lock:
            rows = self.db.execute('SELECT thread,ns,id FROM checkpoints ORDER BY id DESC').fetchall()
        count = 0
        for thread,ns,ident in rows:
            if config:
                c = config['configurable']
                if thread != c['thread_id'] or ns != c.get('checkpoint_ns',''): continue
            if before and ident >= before['configurable']['checkpoint_id']: continue
            item = self.get_tuple({'configurable':{'thread_id':thread,'checkpoint_ns':ns,'checkpoint_id':ident}})
            if filter and any(item.metadata.get(k)!=v for k,v in filter.items()): continue
            if limit is not None and count >= limit: break
            count += 1
            yield item

    def delete_thread(self, thread_id):
        with self.lock, self.db:
            self.db.execute('DELETE FROM writes WHERE thread=?',(thread_id,))
            self.db.execute('DELETE FROM checkpoints WHERE thread=?',(thread_id,))

    async def aget_tuple(self, config): return await asyncio.to_thread(self.get_tuple,config)
    async def aput(self, config, checkpoint, metadata, new_versions):
        return await asyncio.to_thread(self.put,config,checkpoint,metadata,new_versions)
    async def aput_writes(self, config, writes, task_id, task_path=''):
        await asyncio.to_thread(self.put_writes,config,writes,task_id,task_path)
    async def alist(self, config, **kwargs):
        for item in await asyncio.to_thread(lambda:list(self.list(config,**kwargs))): yield item
    async def adelete_thread(self, thread_id): await asyncio.to_thread(self.delete_thread,thread_id)
    def close(self): self.db.close()
