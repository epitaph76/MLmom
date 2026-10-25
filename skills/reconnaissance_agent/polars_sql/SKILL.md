---
name: polars-sql
description: Write read-only Polars 1.26.0 SQL for the reconnaissance agent's registered CSV tools, including aggregates, bins, duplicate keys and train/test comparisons.
---
# Polars 1.26.0 SQL

Use these patterns when calling query_csv. The engine is Polars 1.26.0 SQLContext,
not PostgreSQL, SQLite or DuckDB. Current online documentation may describe newer
syntax. Examples below are executed by tests/test_polars_sql_skill.py through the
same CSV worker as real reconnaissance.

## Tool and schema basics

- Read list_tables first; paginate its schema when columns_truncated is true.
  Replace example names category, x, target, key_a and key_b with exact columns.
  Read-only SELECT/WITH can use registered tables only. Do not submit Python,
  read_csv/scan_parquet functions, file paths or SQL that changes data.
- Use single quotes for string values and double quotes for identifiers needing
  quoting, such as "feature name". Write x IS NULL, never x = NULL.
- Train has labels; test has features only. sample_submission is schema-only.
- Polars compiles SQL to lazy expressions. Aggregates cover the full registered
  input unless your query filters/limits it. Returned rows are a bounded preview
  (up to 100 rows / 20 columns); inspect truncated before claiming completeness.
  Prefer a count/aggregate query over fetching all groups to count them yourself.
- COUNT(*) counts source rows; COUNT(x) counts non-null x. Include COUNT(*) AS n
  when a finding needs measured support. total_rows is output cardinality metadata,
  not the number of source rows in an aggregate.

## Reusable queries

Category frequencies and target distribution; test queries must omit target:

```sql
SELECT category, COUNT(*) AS n, COUNT(target) AS labeled,
       AVG(target) AS mean_target, MIN(target) AS min_target,
       MAX(target) AS max_target
FROM train
GROUP BY category
ORDER BY n DESC, category
```

Missing and conditional counts use SUM(CASE ...), avoiding COUNT(*) FILTER:

```sql
SELECT COUNT(*) AS n, COUNT(x) AS nonmissing,
       SUM(CASE WHEN x IS NULL THEN 1 ELSE 0 END) AS missing,
       SUM(CASE WHEN x < 0 THEN 1 ELSE 0 END) AS negative
FROM train
```

For bins, compute CASE in an inner SELECT. Group by the resulting column in an
outer SELECT; do not GROUP BY a CASE alias created in that same SELECT. Keep null
as its own bin rather than letting ELSE silently mix it with high values:

```sql
SELECT bin, COUNT(*) AS n, AVG(target) AS mean_target
FROM (
    SELECT CASE WHEN x IS NULL THEN 'missing'
                WHEN x < 0 THEN 'negative' ELSE 'nonnegative' END AS bin,
           target
    FROM train
) AS b
GROUP BY bin
ORDER BY bin
```

Count distinct combinations with a derived DISTINCT table. Polars 1.26 does not
accept the tuple expression in COUNT(DISTINCT (key_a, key_b)). Do not concatenate
keys: delimiters, types and nulls can collide. This query counts null-containing
combinations too; add an explicit WHERE if only complete keys should count:

```sql
SELECT COUNT(*) AS unique_keys
FROM (SELECT DISTINCT key_a, key_b FROM train) AS keys
```

Count repeated key groups and their support without materializing every group:

```sql
SELECT COUNT(*) AS duplicate_groups, SUM(n) AS rows_in_duplicate_groups,
       SUM(n - 1) AS extra_rows
FROM (
    SELECT key_a, key_b, COUNT(*) AS n
    FROM train
    GROUP BY key_a, key_b
) AS groups
WHERE n > 1
```

These are duplicates on the chosen keys only. To measure identical feature rows,
use every confirmed feature column, excluding target and ID, rather than a handy
subset. GROUP BY key_a, key_b uses separate columns, not a tuple expression.

To count overlapping distinct complete keys, deduplicate both sides before JOIN
so repeated rows do not multiply the count. Explicitly exclude null keys here;
ordinary equality joins do not match NULL to NULL. Do not report this as overlap
of all keys or overlapping rows:

```sql
SELECT COUNT(*) AS overlapping_complete_keys
FROM (
    SELECT DISTINCT key_a, key_b FROM train
    WHERE key_a IS NOT NULL AND key_b IS NOT NULL
) AS tr
INNER JOIN (
    SELECT DISTINCT key_a, key_b FROM test
    WHERE key_a IS NOT NULL AND key_b IS NOT NULL
) AS te ON tr.key_a = te.key_a AND tr.key_b = te.key_b
```

For train/test comparisons, use compatible UNION ALL branches with the same
aliases and types. This also avoids scalar subqueries inside SELECT:

```sql
SELECT 'train' AS source, COUNT(*) AS n,
       SUM(CASE WHEN x IS NULL THEN 1 ELSE 0 END) AS missing
FROM train
UNION ALL
SELECT 'test' AS source, COUNT(*) AS n,
       SUM(CASE WHEN x IS NULL THEN 1 ELSE 0 END) AS missing
FROM test
```

Measure tail counts directly rather than adding bins from memory. Preserve the
exact boundary (>= versus >); this example counts values at least 300:

```sql
SELECT COUNT(*) AS n,
       SUM(CASE WHEN target >= 300 THEN 1 ELSE 0 END) AS at_least_300
FROM train
```

## Functions and recovery

For UNION, select the exact same named columns in the same order and cast incompatible
values explicitly. SELECT * across different aggregates does not align schemas:

```sql
SELECT 'train' AS source, CAST(COUNT(*) AS INT) AS n FROM train
UNION ALL
SELECT 'test' AS source, CAST(COUNT(*) AS INT) AS n FROM test
```

JOIN conditions use equality between identifiers. Calculate expressions in derived
tables before joining; do not put COALESCE, inequality or OR expressions in ON:

```sql
SELECT COUNT(*) AS overlapping_bins
FROM (
    SELECT DISTINCT CASE WHEN x < 0 THEN 'negative' ELSE 'nonnegative' END AS bin
    FROM train WHERE x IS NOT NULL
) AS tr
INNER JOIN (
    SELECT DISTINCT CASE WHEN x < 0 THEN 'negative' ELSE 'nonnegative' END AS bin
    FROM test WHERE x IS NOT NULL
) AS te ON tr.bin = te.bin
```

An integer flag is not a Boolean predicate. Compare it explicitly with its intended
value; do not pass an integer series as a filter mask:

```sql
SELECT COUNT(*) AS nonzero FROM train WHERE x <> 0
```

Missing columns and aliases require schema inspection, not guesses. list_tables exposes
the actual names and dtypes for each table; an outer SELECT can only reference columns
projected by its inner SELECT. Test features need not contain the training target.
Repeated identical failed requests return their saved error; change or simplify the
operation using the concrete error and current schema before executing again.

Use COUNT, SUM, AVG, MIN, MAX, MEDIAN and STDDEV for basic distributions. For
Pearson correlation use correlate_csv with existing numeric column names and an
exact target column name; CORR is unavailable in the pinned SQL dialect. Do not
pass LN(1 + target) or another expression as correlate_csv.target.

Prefer the derived-table patterns above over scalar subqueries, tuple DISTINCT,
same-SELECT alias grouping, COUNT FILTER or ROW_NUMBER. These are compatibility
choices for this tool/version, not restrictions on all SQL engines.

After an error, follow field_errors/hint: inspect the schema for a missing input
column, move a computed bin into an inner SELECT for a missing alias, or replace
tuple DISTINCT with SELECT DISTINCT columns followed by COUNT(*). Simplify the
failed operation instead of repeatedly probing nearby unsupported syntax. If a
simpler supported form still fails, record the unfinished check and continue.

Save exact numbers from one returned source_ref per observation. Keep computed
interpretations separate from measured cells. Use IDs returned by
save_observation for proposals; SQL success alone does not create an observation.

## Sources

- [Polars SQL overview](https://docs.pola.rs/user-guide/sql/intro/): execution model.
- [Polars 1.26.0 SQL grouping tests](https://github.com/pola-rs/polars/blob/py-1.26.0/py-polars/tests/unit/sql/test_group_by.py): pinned upstream behavior.
- Local worker examples/tests take precedence over examples for newer releases.
