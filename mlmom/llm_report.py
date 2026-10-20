"""Compact Markdown rendering for the LLM-facing dataset analysis."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from .contracts import ColumnProfile, ColumnRole, DatasetProfile, Severity


def _escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _number(value: int | None) -> str:
    return "n/a" if value is None else f"{value:,}".replace(",", " ")


def _distribution_text(column: ColumnProfile) -> str:
    buckets = column.train.distribution
    if not buckets:
        summary = column.train.summary
        keys = ("min", "median", "mean", "max", "length_median", "length_p95")
        return "; ".join(f"{key}={summary[key]}" for key in keys if key in summary) or "n/a"
    return "; ".join(f"{_escape(bucket.label)}: {_percent(bucket.rate)}" for bucket in buckets)


def _target_matrix(profile: DatasetProfile, target: str) -> list[str]:
    summary = profile.target_summaries[target]
    buckets = summary.get("distribution", [])
    lines = [f"### Target: `{_escape(target)}`", ""]
    if not buckets:
        lines.append("No distribution could be calculated.")
        return lines
    labels = [str(bucket["label"]) for bucket in buckets]
    shares = [_percent(float(bucket["rate"])) for bucket in buckets]
    counts = [_number(int(bucket["count"])) for bucket in buckets]
    lines.extend([
        "| Range or class | " + " | ".join(_escape(label) for label in labels) + " |",
        "|---|" + "---|" * len(labels),
        "| Share | " + " | ".join(shares) + " |",
        "| Rows | " + " | ".join(counts) + " |",
        "",
    ])
    scalar_summary = summary.get("summary", {})
    if scalar_summary:
        selected = ("min", "q01", "q25", "median", "mean", "q75", "q99", "max", "std", "skew")
        pairs = [f"{key}={scalar_summary[key]}" for key in selected if key in scalar_summary]
        if pairs:
            lines.append("Numeric summary: " + ", ".join(pairs) + ".")
            lines.append("")
    return lines


def _ordered_columns(profile: DatasetProfile) -> list[ColumnProfile]:
    important = {
        column
        for issue in profile.issues
        if issue.severity != Severity.INFO
        for column in issue.columns
    } | set(profile.task.target_columns) | set(profile.task.id_columns)
    return sorted(
        profile.columns,
        key=lambda column: (
            column.name not in important,
            column.role == ColumnRole.FEATURE,
            -column.train.missing_rate,
            column.name,
        ),
    )


def _render(
    profile: DatasetProfile,
    columns: Iterable[ColumnProfile],
    *,
    include_descriptions: bool,
    omitted: int,
) -> str:
    lines: list[str] = [f"# Dataset analysis: {_escape(profile.dataset_name)}", ""]
    task = profile.task
    lines.extend([
        "## Task",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Type | `{task.task_type}` |",
        f"| Targets | {_escape(', '.join(task.target_columns) or 'not detected')} |",
        f"| IDs | {_escape(', '.join(task.id_columns) or 'not detected')} |",
        f"| Time column | {_escape(task.time_column or 'not detected')} |",
        f"| Group columns | {_escape(', '.join(task.group_columns) or 'not detected')} |",
        f"| Primary metric | {_escape(task.primary_metric or 'not detected')} ({task.metric_direction}) |",
        f"| Inference confidence | `{task.inference_confidence}` |",
        "",
    ])
    if task.inference_evidence:
        lines.append("Inference evidence: " + " ".join(task.inference_evidence))
        lines.append("")

    result = task.result_format
    lines.extend([
        "## Required result",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| File | {_escape(result.file_name or 'not specified')} |",
        f"| Columns | {_escape(', '.join(result.columns) or 'not specified')} |",
        f"| Encoding | {_escape(result.encoding or 'not specified')} |",
        f"| Delimiter | `{_escape(result.delimiter or 'not specified')}` |",
        f"| Include index | {_escape(result.include_index if result.include_index is not None else 'not specified')} |",
        f"| Row order | {_escape(result.row_order or 'not specified')} |",
        "",
    ])
    if task.requirements:
        lines.append("Requirements: " + " ".join(task.requirements))
        lines.append("")

    lines.extend([
        "## Input tables",
        "",
        "| Role | File | Rows | Columns | Duplicate rows | Read status |",
        "|---|---|---:|---:|---:|---|",
    ])
    for file in profile.files:
        lines.append(
            f"| `{file.role}` | {_escape(file.file_name)} | {_number(file.rows)} | "
            f"{_number(file.columns)} | {_number(file.duplicate_rows)} | "
            f"{'error: ' + _escape(file.read_error) if file.read_error else 'ok'} |"
        )
    lines.append("")

    if profile.target_summaries:
        lines.extend(["## Target distributions", ""])
        for target in task.target_columns:
            if target in profile.target_summaries:
                lines.extend(_target_matrix(profile, target))

    lines.extend(["## Data quality and validation risks", ""])
    if profile.issues:
        grouped = Counter(issue.code for issue in profile.issues)
        lines.extend(["| Severity | Issue | Count | Affected columns | Example |", "|---|---|---:|---|---|"])
        for code in sorted(grouped):
            matches = [issue for issue in profile.issues if issue.code == code]
            severity = max(matches, key=lambda issue: {Severity.INFO: 0, Severity.WARNING: 1, Severity.ERROR: 2}[issue.severity]).severity
            affected = list(dict.fromkeys(column for issue in matches for column in issue.columns))
            lines.append(
                f"| `{severity}` | `{_escape(code)}` | {len(matches)} | "
                f"{_escape(', '.join(affected[:15]) or 'dataset-level')} | {_escape(matches[0].message)} |"
            )
    else:
        lines.append("No issues were detected by deterministic checks.")
    lines.append("")

    lines.extend([
        "## Column overview",
        "",
        "| Column | Role | Type | Missing | Unique | Distribution | Tags"
        + (" | Description |" if include_descriptions else " |"),
        "|---|---|---|---:|---:|---|---|"
        + ("---|" if include_descriptions else ""),
    ])
    for column in columns:
        base = (
            f"| `{_escape(column.name)}` | `{column.role}` | `{column.semantic_type}` | "
            f"{_percent(column.train.missing_rate)} | "
            f"{_number(column.train.unique_count)} | {_escape(_distribution_text(column))} | "
            f"{_escape(', '.join(column.tags) or 'none')} |"
        )
        if include_descriptions:
            description = column.description or ""
            if column.unit:
                description = f"{description} [{column.unit}]".strip()
            base += f" {_escape(description[:240])} |"
        lines.append(base)
    if omitted:
        lines.extend(["", f"Omitted from this prompt-sized table: {omitted} lower-priority columns. Full details remain in profile JSON."])
    lines.append("")

    if profile.preprocessing_hints:
        lines.extend(["## Preprocessing and validation hints", ""])
        lines.extend(f"- {_escape(hint)}" for hint in profile.preprocessing_hints)
        lines.append("")
    if profile.report_notes:
        lines.extend(["## Notes", ""])
        lines.extend(f"- {_escape(note)}" for note in profile.report_notes)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def build_llm_markdown(profile: DatasetProfile, max_chars: int = 30_000) -> str:
    """Render a bounded table-first report suitable for direct LLM input."""

    columns = _ordered_columns(profile)
    report = _render(profile, columns, include_descriptions=True, omitted=0)
    if len(report) <= max_chars:
        return report
    report = _render(profile, columns, include_descriptions=False, omitted=0)
    while len(report) > max_chars and columns:
        columns.pop()
        report = _render(
            profile,
            columns,
            include_descriptions=False,
            omitted=len(profile.columns) - len(columns),
        )
    if len(report) > max_chars:
        raise ValueError("max_chars is too small for the mandatory Markdown report")
    return report
