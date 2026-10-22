---
name: schema-quality
description: Turn EDA findings about types, identifiers, duplicates, labels and shift into safe preprocessing decisions.
---
# Schema and data quality

Treat the validated TaskContext and dataset profile as the source of column roles. Distinguish numeric, categorical, boolean, datetime, text, ID, target, group and weight columns explicitly; a numeric dtype does not make an identifier continuous. Coerce strings or dates only with a recorded rule and track parse failures rather than silently producing missing values.

Use train-only summaries to flag impossible ranges, constants, near-constants, duplicate rows and identical features with conflicting targets. Remove only confirmed technical duplicates or impossible records. Rare valid events, fraud cases and minority examples are not data errors merely because they are unusual. Keep duplicate entities or feature hashes in one validation group when feasible.

Target checks include missing labels, allowed classes, class counts and conflicting repeated labels. Never repair labels from a model prediction automatically. Suspected label noise is an audit finding unless a trusted rule confirms the correction.

If the trusted inspector provides train-versus-inference summaries, use distribution differences as risk evidence only. Do not fit transforms, select thresholds or tune models on inference values. Combine effect magnitude, domain meaning and stability; a p-value alone is not a preprocessing command.

Fail clearly when mandatory columns are absent or an inference schema cannot be aligned. Persist the expected feature names, order and type/coercion policy with the candidate.
