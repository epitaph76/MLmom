---
name: hypothesis-generator
description: Form one evidence-backed, falsifiable model improvement from reconnaissance and experiment history.
---
# Hypothesis generation

Use measured findings, task meaning and current-cycle diagnostics to choose one
change whose effect can be checked against a reference on the same outer split.
Describe the observation, its evidence IDs, proposed mechanism and expected metric
effect in the existing description/rationale fields. Distinguish a measured fact
from a suspected explanation; do not invent effect sizes or causal evidence.

Prefer findings with sufficient support and a plausible effect on the primary
metric. Describe computational cost, failure risk and the comparison that would
reject the idea. A tail-error claim needs a supported tail slice, not just total RMSE.

Read previous hypotheses, implementations and failures before proposing a change.
Do not repeat a tested idea under another name or call another hyperparameter set
a new hypothesis: parameter search belongs inside the existing implementation.
A repeat is justified only by changed data/evidence, a repaired implementation or
an explicitly different test; state that reason and use existing lineage IDs.

Keep a necessary baseline simple within the established first-cycle policy.
Later hypotheses should change one principal behavior, such as category handling
or target transformation, so that the result can inform the next experiment.
Do not combine unrelated transforms to claim evidence for each of them.

Respect the confirmed internal validation strategy and allow time for every fold
and a full-train refit. The outer holdout rotates; historical scores are context,
never a directly comparable reference. Candidate ensembles, blending and stacking
are disabled. Return only the existing HypothesisBatch schema.
