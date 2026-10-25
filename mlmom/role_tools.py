"""Train-scoped evidence tools shared by feature proposal calls."""
from functools import wraps
from pathlib import Path

from .context_views import paged_evidence
from .csv_tools import CSVTools
from .persistence import file_hash
from .time_policy import UNBOUNDED_SECONDS
from .tools import ToolSet


def feature_data_tools(pipeline, train_path, session_id):
    """Register only this cycle's allowed train columns; references survive resume."""
    p = pipeline
    task = p.context.task
    columns = list(dict.fromkeys([*p.context.feature_columns, *task.target_columns,
        *task.group_columns, *([task.weight_column] if task.weight_column else [])]))
    fmt = p.context.table_formats.get(str(train_path))
    tables = {'train':dict(path=str(train_path), role='feature_training', columns=columns,
        format=fmt.model_dump() if fmt else None, hash=file_hash(Path(train_path)))}
    deadline = (p.budget.now()+min(p.config.feature_agent_seconds,p.budget.snapshot().research_seconds)
                if p.config.timeouts_enabled else UNBOUNDED_SECONDS)
    csv = CSVTools(tables=tables, sandbox=p.sandbox, budget=p.budget, store=p.store,
        key='feature-'+session_id, deadline=deadline, query_seconds=p.config.reconnaissance_query_seconds)
    saved = p.store.get('feature_query_refs', session_id) or {'references':[]}
    values = {ref:p.store.get('csv_result',ref[4:]) for ref in saved['references']}
    values = {ref:value for ref,value in values.items() if value is not None}
    tools = ToolSet()

    def remember(fn):
        @wraps(fn)
        def measured(*args, **kwargs):
            result = fn(*args, **kwargs)
            ref = result.get('source_ref')
            if ref:
                values[ref] = result
                previous = p.store.get('feature_query_refs',session_id) or {'references':[]}
                p.store.put('feature_query_refs', session_id, {'references':sorted(set(values)|set(previous['references']))})
            return result
        return measured

    for name,description in (
        ('list_tables','Read paginated schemas for the registered train table only.'),
        ('preview_csv','Preview bounded allowed training values, never outer holdout or test.'),
        ('query_csv','Measure read-only Polars SQL on allowed training rows.'),
        ('correlate_csv','Measure train-only numeric Pearson correlations.')):
        tools.register(name,remember(getattr(csv,name)),description)
    tools.tools.update(paged_evidence(values).tools)
    def read_query_result(reference: str, row_offset: int=0, column_offset: int=0, rows: int=100) -> dict:
        value = values.get(reference)
        if value is None or not value.get('artifact_ref'):
            raise ValueError('Choose a saved tabular source_ref from this feature session; metadata uses read_evidence')
        if row_offset < 0 or column_offset < 0 or not 1 <= rows <= 100:
            raise ValueError('Use non-negative offsets and 1..100 rows')
        return csv._run('result_page',result_path=value['artifact_ref'],row_offset=row_offset,
                        column_offset=column_offset,rows=rows)
    tools.register('read_query_result',remember(read_query_result),
        'Read a bounded row/column page of a saved train query, including results beyond its initial preview. Use its source_ref.')
    tools.evidence_refs = sorted(values)
    return tools
