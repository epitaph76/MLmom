---
name: hypothesis-generator
description: Turn measured findings and experiment history into one-change, falsifiable ML hypotheses.
---
# Hypothesis generator

Select an observation that can change a modeling decision. Prefer confirmed task
requirements, critical findings, measured reconnaissance observations, comparable
experiment results and diagnosed failures. Use only supplied observation, evaluation,
candidate and experiment references. A proposal, correlation or plausible mechanism is
not a measured benefit. Empty history is normal before the first fit; in that case create
the required cheap baseline rather than inventing evidence.

Propose one falsifiable intervention per hypothesis. One intervention may be a model
family, representation, feature transformation, loss/weighting policy, sampling policy
or decision rule. Keep supporting operations only when they are necessary to execute
that intervention. A new parameter value, wider range, longer search or another Optuna
trial is not a new hypothesis: parameter search belongs inside the existing hypothesis.
If two independent changes are desirable, choose the higher-value one now and leave the
other for a later ablation.

For later rounds, connect the intervention to numeric findings or a previous experiment.
Use `based_on` only for a supplied verified candidate ID and
`based_on_hypothesis_id` only for lineage. Do not compare scores from different outer
splits as if they were paired. Reject a repeated idea when the same mechanism and
intervention were already tested, rejected, or failed for the same unresolved reason.
Revisit it only when new evidence changes the mechanism, the intervention is materially
different, or a named fix removes the previous blocker.

Estimate value against the requested metric direction and the affected population.
State a qualitative effect unless supplied evidence supports a numeric estimate. Balance
that value against runtime/memory cost, implementation complexity, leakage or instability
risk, and the remaining experiment budget. Prefer the cheapest experiment that can
falsify the claim.

Return at most the requested number of hypotheses. Preserve the required baseline flag.
`description` names the single intervention. Write `rationale` compactly in this order:
`Evidence: ...; Mechanism: ...; Expected effect: ...; Cost: ...; Risk: ...; Verify: ...`.
Verification must name the comparable reference, metric and acceptance/rejection signal.

Decision examples:

- Good: a saved rare-class observation motivates class weights, tested against the same
  candidate and split with macro-F1 and per-class recall.
- Reject: "try depth 8 instead of 6" is another parameter trial, not a new hypothesis.
- Reject: repeat target encoding after it already failed from leakage unless the new idea
  names and tests the fold-local correction.
