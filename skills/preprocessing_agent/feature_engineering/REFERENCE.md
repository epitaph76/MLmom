# Feature research families and controlled comparisons

This is a method guide for the assigned transform. Preprocessing owns fold-fitted feature
representation; Feature/experiment agents choose and measure new hypotheses. Record the
semantic argument and proposed comparison in agent_notes when this call cannot run it.
Do not train the predictive estimator from preprocessing, invent measured gains, or duplicate
registered feature_set programs. A required representation repair is not evidence of gain.

## Transformation card

For each candidate record input columns/units, the physical or behavioral relation, what
information it adds beyond existing inputs, expected effect, prediction-time availability,
learned state, domain/fallback, supported rows/segments and the controlled experiment.
Start with a small supported family; choose subsequent operations from measured residuals
or a concrete mechanism. An exact algebraic identity may add no new information but make
an existing relation easier for a limited model to learn. Distinguish these explanations.

## 1. Arithmetic invariants

Look for sums, conservation constraints, component-total relationships, products and scale
identities justified by units or domain definitions. Examples include part sums minus an
observed total, mass minus volume*density, or a normalized discrepancy. Confirm that all
inputs exist before prediction and that none is the target/post-outcome quantity. Check
rounding, sensor precision, unit conversions, missing components and different conventions
before calling a discrepancy an anomaly. Do not search arbitrary arithmetic expressions just
because one correlates with y. Compare the invariant/residual feature separately with the base.

## 2. Relative quantities

Test rates, shares, per-unit quantities, scale normalization and positive-domain log ratios
when the denominator describes meaningful exposure/size. Equal ratios can hide very different
sample sizes, so retain the denominator or a justified reliability indicator. State when a
share is bounded and whether numerator/denominator refer to the same interval/population.
Define zero/negative/missing denominator and tiny-exposure behavior in the correct units;
a universal epsilon can change the quantity's meaning. Train-fitted clipping and fallbacks
remain fold-local and persisted. Compare the relative quantity, not an uncontrolled bundle
of every possible numerator/denominator pair.

## 3. Group deviations

Compare an object with a meaningful reference population: x-centre(group), relative deviation,
or standardized deviation using train-fitted group scale. Choose mean/median/robust scale
from the distribution and task, not automatically. Inspect group support, zero variance,
rare/new groups and shrinkage to a stored population reference. Do not normalize from the
prediction batch. Target-based group statistics require OOF/exclusion and corresponding
priors/fallbacks; ordinary TargetEncoder cross-fitting does not ensure group isolation.
For a historical interpretation exclude the current event and unavailable future records.
Compare the deviation with the base while keeping original x available unless removal is a
separate explicitly measured intervention.

## 4. Interactions

Choose conditional effects with an interpretable mechanism: an exposure changing a rate,
different operating regimes, a category modifying a numeric response, or a justified product.
Explain why the baseline representation/model may not already express it. For categorical
crosses measure joint support, rare combinations, cardinality, memory and unseen-combination
fallbacks. Increase interaction order only when simpler comparisons justify it; do not create
every polynomial or cross of identifier-like fields. Parsing/coding choices preserve meaning
and are not themselves proof of predictive utility.

## 5. Repeated events

Separate persistent entity, observation/event, time and availability fields. Test prior-event
counts, time since last known event, lag/delta and rolling rate/variability where available
history can explain the task. State window endpoints, ties, irregular sampling and delayed
target availability. Training history must exclude own outcome and impermissible future;
target summaries require cross-fitting/exclusion. Inference uses frozen permitted training
history plus the query row, not other prediction rows as newly fitted history. Return an
explicit no-history fallback/indicator. Temporal recipe validity does not create a deferred
time-aware executor or authorize replacing the controller's splits.

## 6. Relational aggregation

Define entity/event views and validated parent-child relationships from supplied inputs.
Choose counts, sums, means, diversity or conditional aggregates with explicit population,
denominator, units and cutoff. Check join cardinality, duplicate event IDs, multiple parents
and many-to-many paths: row expansion can double-count and change entity weights. Distinguish
number of events from number of unique related objects. Identity/linkage uncertainty must
be propagated as ambiguity/confidence or a fallback, not silently converted into hard truth.
Additional tables are available only through an explicit runtime contract. Featuretools-style
composition can organize valid paths but does not validate the identities or temporal scope.

## Individual comparisons and follow-up

Specify one transformation versus the same frozen base: preserve model parameters, policies,
splits, seed, metric options/weights and data scope. Report baseline/candidate/delta, supported
row/entity counts, relevant segment changes and fit/inference cost. If operations are jointly
necessary, explain dependencies and compare feasible component ablations; do not attribute a
combined gain to every member without separate evidence. An atomic feature replacement is one
final-set comparison, without a prerequisite removal-only improvement.

If a feature helps only a small/ambiguous segment, inspect sampling, linkage error, missingness
and exposure as alternative explanations. Use new internal comparisons for further choices;
outer labels must not fit transformations or choose masks. Store learned rules, output names
and order and verify the selected implementation after load and on individual/new-group rows.
These are tools for making a defensible experiment, not mandatory estimator/smoke audits.
