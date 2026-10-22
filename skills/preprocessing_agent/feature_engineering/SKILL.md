---
name: feature-engineering
description: Implement cheap reproducible tabular features and train-fitted aggregates after establishing a baseline.
---
# Feature engineering

Test a specific hypothesis: date parts, missing indicators, safe logs, selected ratios/interactions or train-only frequency/group aggregates. Avoid exhaustive polynomials and arbitrary ID interactions. plan.preprocessing does not execute features: implement/persist a transformer or wrapper.

Protect zero denominators, validate log domains and clean infinities. Fix datetime/timezone parsing and reference dates. Store output order and train-derived clipping cutoffs/counts/maps; unseen groups get stored fallback values. Do not recompute maps from prediction batches.

Target encoding needs OOF training values and full-train maps only for new rows. sklearn TargetEncoder.fit_transform cross-fits; fit().transform() is not equivalent. Its ordinary folds do not guarantee group isolation: use explicit group-safe OOF maps when needed. Outer validation cannot fit encoders.

Compare with baseline on the same runner split/metric, including CPU/RAM cost. Check persisted transforms after load and on a single new row without batch statistics.
