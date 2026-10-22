"""Bounded CSV exploration over registered tables using the pinned Polars SQL engine."""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path

from .inputs import detect_table_format, read_table
from .persistence import atomic_json, file_hash


def validate_sql(sql: str) -> str:
    """Lex SQL before Polars sees it; file functions must never reach its planner."""
    if not sql.strip() or len(sql) > 16000:
        raise ValueError('SQL must contain 1..16000 characters')
    pattern = re.compile(r'''\s+|--[^\n]*|/\*.*?\*/|'(?:''|[^'])*'|"(?:""|[^"])*"|[\w]+|[(),.*+/<>=!%:;|\-]''', re.S)
    tokens = []
    position = 0
    for match in pattern.finditer(sql):
        if match.start() != position:
            raise ValueError('Unsupported SQL token')
        position = match.end()
        token = match.group()
        if token.isspace() or token.startswith(('--', '/*')):
            continue
        tokens.append(token)
    if position != len(sql) or not tokens or tokens[0].upper() not in {'SELECT', 'WITH'}:
        raise ValueError('Only SELECT/WITH queries are allowed')
    if tokens[-1] == ';':
        tokens.pop()
    if ';' in tokens:
        raise ValueError('Only one SQL statement is allowed')
    forbidden = {'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'COPY', 'ATTACH',
                 'DETACH', 'INSTALL', 'LOAD', 'PRAGMA', 'SET', 'TRUNCATE', 'INTO', 'EXECUTE'}
    for token in tokens:
        if token.startswith("'"):
            continue
        name = token.strip('"').replace('""', '"').lower()
        if name.startswith(('read_', 'scan_')) or (not token.startswith('"') and token.upper() in forbidden):
            raise ValueError('File functions and SQL mutations are forbidden')
    return sql.rstrip().removesuffix(';')


def _lazy_table(spec, cache):
    import polars as pl
    path = Path(spec['path'])
    options = spec.get('format')
    if options is None:
        try:
            options = detect_table_format(path).model_dump()
        except UnicodeDecodeError:
            options = detect_table_format(path,encoding='cp1251').model_dump()
    encoding = options['encoding'].lower().replace('_', '-')
    if encoding in {'utf-8', 'utf-8-sig', 'utf8'}:
        frame = pl.scan_csv(path, separator=options['delimiter'], infer_schema_length=10000,
                            try_parse_dates=False, ignore_errors=False)
    else:
        # pandas preserves the existing reader's encoding/delimiter semantics.
        # Atomic publication prevents an interrupted conversion becoming a valid cache.
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        identity = hashlib.sha256((str(path.resolve())+file_hash(path)+json.dumps(options, sort_keys=True)).encode()).hexdigest()
        target = cache / (identity+'.parquet')
        if not target.exists():
            temporary = target.with_suffix('.partial')
            schema = None
            writer = None
            try:
                # Strings avoid different chunk-level guesses; Polars infers numeric
                # columns below only when every non-null value can be converted.
                for chunk in pd.read_csv(path, sep=options['delimiter'], encoding=options['encoding'],
                                         chunksize=10000, dtype='string', on_bad_lines='error'):
                    table = pa.Table.from_pandas(chunk, preserve_index=False)
                    if writer is None:
                        schema = table.schema
                        writer = pq.ParquetWriter(temporary, schema)
                    writer.write_table(table.cast(schema))
                if writer is None:
                    table = pa.Table.from_pandas(read_table(path, table_format=options, nrows=0).astype('string'), preserve_index=False)
                    pq.write_table(table, temporary)
                else:
                    writer.close(); writer = None
                temporary.replace(target)
            finally:
                if writer is not None:
                    writer.close()
                temporary.unlink(missing_ok=True)
        frame = pl.scan_parquet(target)
        names = frame.collect_schema().names()
        if names:
            convertible = frame.select([
                (pl.col(c).is_null() | pl.col(c).cast(pl.Float64, strict=False).is_not_null()).all().alias(c)
                for c in names]).collect(streaming=True).row(0, named=True)
            frame = frame.with_columns([pl.col(c).cast(pl.Float64) for c in names if convertible[c]])
    if spec.get('columns') is not None:
        frame = frame.select(spec['columns'])
    return frame


def _json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value[:1000] if isinstance(value, str) else value
    return str(value)[:1000]


def csv_worker(request, root):
    """Runs only in an admitted worker; full results stay outside the LLM context."""
    import polars as pl
    cache = Path(request['cache_dir'])
    cache.mkdir(parents=True, exist_ok=True)
    specs = request['tables']
    operation = request['operation']
    if operation == 'schema':
        table = request.get('table')
        if table is not None and table not in specs:
            raise ValueError('Unknown registered table')
        entries = []
        for name in ([table] if table else list(specs)[:100]):
            spec = specs[name]
            frame = _lazy_table(spec, cache)
            schema = frame.collect_schema()
            offset = request.get('offset', 0)
            entries.append(dict(table=name, role=spec['role'], schema_only=spec.get('schema_only', False),
                total_columns=len(schema), column_offset=offset,
                total_rows=frame.select(pl.len()).collect(streaming=True).item(),
                columns=[dict(name=c, dtype=str(t)) for c,t in list(schema.items())[offset:offset+20]],
                columns_truncated=offset+20<len(schema)))
        return dict(status='completed', tables=entries)
    tables = {name: spec for name,spec in specs.items() if not spec.get('schema_only')}
    if operation == 'preview':
        name = request['table']
        if name not in tables:
            raise ValueError('Unknown table or metadata-only table')
        frame = _lazy_table(tables[name], cache)
        columns = request.get('columns') or frame.collect_schema().names()[:20]
        if len(columns) > 20:
            raise ValueError('Choose at most 20 preview columns')
        result = frame.select(columns).slice(request.get('offset', 0), request.get('rows', 10))
    elif operation == 'correlate':
        name = request['table']
        if name not in tables:
            raise ValueError('Unknown table or metadata-only table')
        frame = _lazy_table(tables[name], cache)
        target = request['target']
        results = []
        for column in request['columns']:
            pair = frame.select(pl.col(column).cast(pl.Float64,strict=False).alias('x'),
                                pl.col(target).cast(pl.Float64,strict=False).alias('y')).filter(
                                    pl.col('x').is_finite() & pl.col('y').is_finite())
            values = pair.select(pl.corr('x','y').alias('correlation'),pl.len().alias('pairs')).collect(streaming=True).row(0,named=True)
            results.append(dict(feature=column,target=target,**values))
        result = pl.DataFrame(results).lazy()
    elif operation == 'query':
        sql = validate_sql(request['sql'])
        context = pl.SQLContext(frames={name:_lazy_table(spec, cache) for name,spec in tables.items()}, register_globals=False)
        result = context.execute(sql, eager=False)
    else:
        raise ValueError('Unknown CSV operation')
    artifact = root/'full.parquet'
    # Streaming sink avoids materializing a raw SELECT * merely to bound its preview.
    try:
        result.sink_parquet(artifact)
    except pl.exceptions.InvalidOperationError:
        result.collect(streaming=True).write_parquet(artifact)
    full = pl.scan_parquet(artifact)
    names = full.collect_schema().names()
    total = full.select(pl.len()).collect().item()
    preview = full.select(names[:20]).head(100).collect().to_dicts()
    cells_truncated = any(isinstance(value,str) and len(value)>1000
                          for row in preview for value in row.values())
    rows = [{c:_json_value(v) for c,v in row.items()} for row in preview]
    answer = dict(status='completed', columns=names[:20], total_columns=len(names), total_rows=total,
                  rows=rows, truncated=total>100 or len(names)>20 or cells_truncated,
                  cells_truncated=cells_truncated, artifact_ref=str(artifact))
    while len(json.dumps(answer, ensure_ascii=False)) > 31000 and answer['rows']:
        answer['rows'].pop(); answer['truncated'] = True
    return answer


class CSVTools:
    def __init__(self, *, tables, sandbox, budget, store, key, deadline, query_seconds=30):
        self.tables, self.sandbox, self.budget, self.store = tables, sandbox, budget, store
        self.key, self.deadline, self.query_seconds = key, deadline, query_seconds

    def _run(self, operation, **arguments):
        identity = hashlib.sha256(json.dumps(dict(key=self.key, operation=operation, arguments=arguments,
            tables=self.tables), sort_keys=True).encode()).hexdigest()
        saved = self.store.get('csv_result', identity)
        if saved:
            return saved
        seconds = min(self.query_seconds, self.deadline-self.budget.now(), self.budget.snapshot().research_seconds)
        if not getattr(self.budget,'timeouts_enabled',True):
            from .time_policy import UNBOUNDED_SECONDS
            seconds=UNBOUNDED_SECONDS
        if seconds <= .1:
            raise ValueError('No reconnaissance calculation time remains')
        directory = self.store.root/'reconnaissance'/self.key/'queries'/identity
        result = self.sandbox.run(dict(action='csv', operation=operation, tables=self.tables,
            cache_dir=str(self.store.root/'csv_cache'), **arguments), directory, seconds=seconds,
            cpus=min(2,self.budget.cpus), memory_gb=min(4.,self.budget.memory_gb))
        if result['status'] != 'completed':
            raise ValueError(result.get('error') or result.get('stop_reason') or 'CSV calculation failed')
        result = {k:v for k,v in result.items() if k not in {'duration_seconds','peak_memory_gb'}}
        result['source_ref'] = 'csv:'+identity
        while len(json.dumps(result,ensure_ascii=False)) > 32000 and result.get('tables'):
            result['tables'].pop();result['truncated']=True
        if len(json.dumps(result,ensure_ascii=False)) > 32000:
            raise ValueError('CSV metadata exceeds the response size limit; select fewer columns')
        self.store.put('csv_result', identity, result, immutable=True)
        return result

    def list_tables(self, table: str | None = None, offset: int = 0) -> dict:
        if offset < 0:
            raise ValueError('offset must be non-negative')
        return self._run('schema', table=table, offset=offset)

    def preview_csv(self, table: str, columns: list[str] | None = None, offset: int = 0, rows: int = 10) -> dict:
        if offset < 0 or not 1 <= rows <= 100:
            raise ValueError('Use a non-negative offset and 1..100 rows')
        return self._run('preview', table=table, columns=columns, offset=offset, rows=rows)

    def query_csv(self, sql: str) -> dict:
        return self._run('query', sql=validate_sql(sql))

    def correlate_csv(self, table: str, columns: list[str], target: str) -> dict:
        if not 1 <= len(columns) <= 20:
            raise ValueError('Choose 1..20 numeric columns')
        return self._run('correlate',table=table,columns=columns,target=target)
