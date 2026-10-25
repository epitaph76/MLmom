---
name: hypothesis-generator
description: Turn measured findings and experiment history into one-change, falsifiable ML hypotheses.
---
# Hypothesis generator

Use champion.execution_context for the actual model composition and observed execution.
An estimator name alone does not describe its learned SDK feature programs. Unknown facts
are not evidence that features were skipped. Historical baseline prose and reviewer explanations
do not override runtime facts. Champion-derived plans inherit saved configuration automatically;
recommend additions relative to it, or explicitly request removals/replacements. Keep score
comparisons scoped to their split and separate bundle improvements from HPO-specific effects.

feature_sets contains accepted, versioned feature definitions and their measured evidence.
They are options for later model hypotheses, not universal proof across backends. Cite the
set ID when proposing reuse. Do not repeat a feature search already resolved by session
history; the coordinator prepares initial proposals after context and measures them on a fixed control before long HPO.
Feature sets may contain Python modules beyond DSL, including supervised representations
and inferred entity history. An unavailable DSL operator is not a modeling restriction.

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
Treat equivalent backend implementations as variants of one approach: use kind=family,
model_family=gradient_boosting and explicitly select useful family_tools for the task. Boosting is optional; other supported tasks need appropriate models. Other model families name
up to three family_tools. The runner compares initial configurations on the same inner
split and tunes only the best implementation. Do not use multiple parent slots for
boosting library names. Ensemble search has its own slot after the first cycle; prefer
ordinary representation/model interventions here instead of duplicating that search.
`description` names the single intervention. Write `rationale` compactly in this order:
`Evidence: ...; Mechanism: ...; Expected effect: ...; Cost: ...; Risk: ...; Verify: ...`.
Verification must name the comparable reference, metric and acceptance/rejection signal.

Decision examples:

- Good: a saved rare-class observation motivates class weights, tested against the same
  candidate and split with macro-F1 and per-class recall.
- Reject: "try depth 8 instead of 6" is another parameter trial, not a new hypothesis.
- Reject: repeat target encoding after it already failed from leakage unless the new idea
  names and tests the fold-local correction.

Blending and OOF stacking are allowed hypotheses when diverse errors justify
them; account for every component and preprocessing cost. The runner owns
validation_strategy, all internal folds and fresh full-train refit. Timeouts are
opt-in; do not invent a hard deadline when timeouts_enabled=false.

## Authoritative current evidence
Use Champion.model_params/configuration_hash and scoped evaluation; original_hypothesis is historical prose with potentially obsolete numbers. A single-change ablation must inherit the current configuration; extra tuning makes it a bundled experiment. Never interpret mae as signed bias: mean_residual=y_true-y_pred, positive means underprediction. Scores have explicit score_kind, split and weighting; historical outer holdout is not an internal CV baseline. A runtime failure means the hypothesis was not tested; use actual scores and errors, without requiring provenance reports. Promise duplicate-aware validation only when the immutable task duplicate_policy is group_features.

Choose hypotheses/families and lineage. The controller owns search-group IDs and
membership; based_on fields describe historical lineage, not group assignment.
