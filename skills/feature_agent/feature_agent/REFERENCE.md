# Entity discovery, linkage and relational feature research

Use the six-stage hierarchy in SKILL.md for a concrete entity/history hypothesis. This is
a research procedure, not an extra admission gate or a claim that identity is identifiable
in every dataset. Existing tools expose the current cycle's registered train only. Additional
tables are usable only if explicitly supplied by the runtime; a library example is not
permission to open files or external datasets. In a flat table, derive entity/event views
only where row semantics and measured cardinalities justify them.

## 1. Discover structure before choosing keys

State what one row represents: an object, event, object-period measurement or aggregate.
Distinguish observation IDs, entity IDs, group/parent IDs and ordering fields. List units,
conversions, categorical code meanings, missingness, repeated observations and sampling
frequency. A timestamp may be an event date, ingestion date or future outcome date.
For each potential relation state one-to-one, one-to-many or many-to-many and measure its
fan-out; joining an exploded child table can silently reweight entities or duplicate events.

Begin with schema/column pages, dictionary meanings and cheap train aggregates. Keep a queue:
`hypothesis -> supporting source_ref -> unresolved question -> next query -> feature family`.
For each entry record row/segment support, linkage uncertainty, inference inputs, cost and
the expected mechanism of improvement. Separate observations from hypotheses and measured gains.

## 2. Find invariants and a small key frontier

Use trusted IDs or independently anchored repeats where available. For a categorical field,
inspect conditional distinct counts/entropy given the anchor; for numeric fields compare
within-anchor spread with measurement resolution and between-anchor spread. Show nonmissing
support and the number of repeated anchors. Singletons and identifiers copied into the
candidate grouping cannot certify invariance. If no independent anchors exist, call apparent
stability conditional on a provisional grouping; test it with independent attributes and
alternative groupings instead of presenting it as established identity.

Normalize units, precision, case and missing markers using declared semantics and train-fitted
rules. Keep original distinctions when they might matter: rounding can merge different objects,
and a changing category/address can split a real object. Missing values are weak evidence, not
agreement. Do not put the prediction target or post-outcome measurements into a linkage key.

Start from plausible stable field families rather than all combinations. Profile a small
frontier of exact keys, then add a field or tolerance only to address a measured failure.
Keep the coverage/ambiguity/cost frontier; discard duplicate partitions and redundant aggregates.
Very fine keys can achieve apparent purity by producing only singletons. Very broad keys
can cover all rows while merging many objects. Neither result establishes useful linkage.

At each iteration update a diagnostic queue and an experiment queue. First resolve missing
row-grain/unit/cutoff facts that could invalidate the leading hypotheses. Then count coverage
and block sizes for plausible keys before any pair comparisons. If a leading key is ambiguous,
query independent contradictions or label evidence and try one informative extension/relaxation;
otherwise specify its feature family and as-of fallback. Recompute the frontier after each query.
Among candidates at the same evidence tier, prefer non-dominated coverage/ambiguity/cost;
break ties by the strongest supported task/error mechanism, then larger usable segment, then
lower marginal pair/fit cost. Keep labelled accuracy separate from unlabelled proxies. Do not
promote an unmeasured high-correlation idea above measured candidates merely for a larger score.
Stop expanding a branch when another retained candidate dominates it or the next check cannot
change a useful experiment choice within resources. These rules prioritize research; uncertainty
can justify a diagnostic experiment and is not an automatic execution refusal.

The following SQL examples are a small synthetic diagnostic, not universal column names.
`anchor_id` is an independent trusted identity available for a labelled subset; `category`
is a categorical attribute and `measure` a numeric attribute in a common unit. Adapt all
names to the real schema. The three queries are exercised through the pinned CSV worker.

```sql
WITH by_anchor AS (
    SELECT anchor_id, COUNT(*) AS observations,
           COUNT(category) AS category_observed,
           COUNT(DISTINCT category) AS category_levels,
           COUNT(measure) AS measure_observed,
           MIN(measure) AS min_measure, MAX(measure) AS max_measure
    FROM train WHERE anchor_id IS NOT NULL GROUP BY anchor_id
)
SELECT * FROM by_anchor WHERE observations > 1
ORDER BY anchor_id
```

## 3. Diagnose keys, blocking and linkage errors

For a candidate key K report total rows N, complete-key rows U, rows in repeated groups R,
cross-period repeat rows, unmatched/ambiguous fractions and the block-size distribution.
Give coverage U/N and repeated-row support R/U with denominators. Repeats across periods are
different from several events in the same period; the latter are not automatically errors.

For deduplication, within-block pair cost is sum n_b*(n_b-1)/2; link-only between two tables
uses sum n_left,b*n_right,b. Count the union across blocking rules, not the duplicated sum.
Measure the largest blocks before computing distances. Prefer selective equality blocks and
incrementally relax or union complementary rules; expensive comparisons run inside candidates.
Do not materialize an unbounded all-pairs join or silently discard large blocks without
reporting lost coverage. Chunking/blocking must preserve the chosen method's comparison scope.

```sql
WITH blocks AS (
    SELECT region, category, size_bucket, COUNT(*) AS n,
           COUNT(DISTINCT period) AS periods,
           COUNT(DISTINCT anchor_id) AS anchors
    FROM train
    WHERE region IS NOT NULL AND category IS NOT NULL AND size_bucket IS NOT NULL
    GROUP BY region, category, size_bucket
)
SELECT COUNT(*) AS key_groups, SUM(n) AS covered_rows,
       SUM(CASE WHEN n > 1 THEN n ELSE 0 END) AS repeated_rows,
       SUM(CASE WHEN periods > 1 THEN n ELSE 0 END) AS cross_period_rows,
       SUM(CASE WHEN anchors > 1 THEN n ELSE 0 END) AS conflict_rows,
       MAX(n) AS largest_block, SUM(n*(n-1)/2) AS candidate_pairs
FROM blocks
```

Anchor conflicts are evidence of collisions in the anchored subset, not a calibrated global
false-link rate. A numeric tolerance must follow units/resolution or independent repeated
records. For fuzzy matching, compare informative fields, account for common values and missing
comparisons, keep best/second-best margins and permit abstention. Model scores are not measured
identity probabilities unless calibration is supported. Freeze rules and thresholds inside
each training fold; any independent linkage labels must also respect that scope.

With trusted entity labels, enumerate conceptual record pairs independently of the proposed
key. TP are accepted same-entity pairs, FP accepted different-entity pairs, FN true pairs not
accepted, including those missed by blocking. Precision=TP/(TP+FP), recall=TP/(TP+FN).
Do not call an undefined denominator zero or one. The next query gives counts for a strict
same-key rule on the anchored subset without building a pair table. Null anchors are excluded
from truth counts; known anchors with missing candidate keys still belong to the recall
denominator. Repeated rows are distinct observations here; exclude duplicate ingestions first
when they are not genuine independent records.

```sql
WITH labelled AS (
    SELECT * FROM train WHERE anchor_id IS NOT NULL
), blocks AS (
    SELECT region, category, size_bucket, COUNT(*) AS n FROM labelled
    WHERE region IS NOT NULL AND category IS NOT NULL AND size_bucket IS NOT NULL
    GROUP BY region, category, size_bucket
), anchor_blocks AS (
    SELECT region, category, size_bucket, anchor_id, COUNT(*) AS n FROM labelled
    WHERE region IS NOT NULL AND category IS NOT NULL AND size_bucket IS NOT NULL
    GROUP BY region, category, size_bucket, anchor_id
), entities AS (
    SELECT anchor_id, COUNT(*) AS n FROM labelled GROUP BY anchor_id
)
SELECT proposed.proposed_pairs, correct.true_positive_pairs, truth.true_pairs
FROM (SELECT COALESCE(SUM(n*(n-1)/2),0) AS proposed_pairs FROM blocks) AS proposed
CROSS JOIN (SELECT COALESCE(SUM(n*(n-1)/2),0) AS true_positive_pairs FROM anchor_blocks) AS correct
CROSS JOIN (SELECT COALESCE(SUM(n*(n-1)/2),0) AS true_pairs FROM entities) AS truth
```

For pair labels instead of entity IDs, use an independent held-out evaluation sample and
state its sampling design. Inspect high/low scores, threshold-near pairs, ambiguous records,
rare/missing attributes and unmatched records; a representative or appropriately weighted
sample estimates population precision/recall. Active review near a threshold is diagnostic,
not a representative accuracy estimate on its own. Recall needs true links outside predicted
blocks as well; labelling only emitted pairs cannot measure missed links there. Uncertain
clerical labels remain uncertain. Report sample counts, coverage, intervals where supported
and selection bias. Do not invent gold labels from the candidate key, from y, or from a
downstream score improvement. Without reference labels, report contradiction rates, threshold
sensitivity, ambiguity and unmatchedness as proxies; actual FP/FN rates remain unknown.

Choose thresholds using the relative cost of false merges/missed links and observed validation
trade-offs, not a universal probability cutoff. Review cluster-level contradictions and bridge
edges: connected components can amplify one false link. Preserve ambiguous clusters separately
or use confidence/fallback features instead of forcing an identity for every row.

## 4. Verify relationships and time

Check relation fan-out, duplicate events, parent uniqueness, cross-object consistency and
entity changes through time. Separate persistent entities from temporary states. A relation
valid at one date need not hold forever; check effective dates, ties, source updates and gaps.
For every historical source identify event_time, available_time and the prediction cutoff.
Use only information available by that cutoff; a backfilled record with an old event date
can still be future information. Specify whether at-cutoff records are observable. Treat
target availability as a separate, often later, time.

Keep the controller's confirmed validation protocol. If it cannot support the required
deployment claim, describe the mismatch or deferred execution path; do not secretly replace
folds, fit on future rows, or present random-split gains as temporal validation. Structural
research and a proposed temporal recipe are not evidence of a completed temporal executor.

## 5. Build features with an inference contract

Select one interpretable family: historical counts/recency, entity-vs-peer deviations,
lag/delta/rolling summaries, parent-child aggregates, relationship diversity, or confidence,
missing-history and ambiguity indicators. Define the unit, cutoff, denominator, source path
and fallback for each feature. Deduplicate source events before aggregating across relations.

`fit_transform(X,y,context)` learns matching rules, vocabularies, reference entities/history
and priors only from the current training fold. Training history excludes the current event
and unavailable future values. Target-derived training features use OOF/exclusion, including
priors and fallback statistics; unsupervised learned transforms still stay fold-local.
`transform(X)` resolves each query against frozen training reference state and permitted row
attributes, applies its own cutoff, and never uses query labels or other query rows as newly
learned history. An online history update needs an explicit runtime contract; it is not implied
by a batch of predictions. Give unseen/no-history and ambiguous records documented fallbacks.
Persist learned state and restore exact row order, width and names after all joins.

Check representative known/new/ambiguous records, history-poor segments, cutoff boundaries and
save/load. Predictions for the same rows should agree when batches are reordered or partitioned;
moving a future event later must not change an earlier as-of feature. These are useful checks
for the chosen recipe, not a mandatory synthetic smoke or an estimator audit before admission.

## 6. Prioritize experiments and ablate by segment

Schedule the next query that can cheaply separate competing explanations. Prioritize families
whose mechanism fits the task, have meaningful supported coverage and acceptable linkage
uncertainty, add information beyond measured features and fit actual CPU/RAM. Use supplied
OOF residual slices to identify supported error concentrations; do not manufacture OOF evidence
before a control exists. Keep uncertain high-value ideas in the queue alongside cheap checks.
Expand primitive depth/key complexity only after simpler results justify the extra search.
Pause low-value/redundant branches instead of scanning every possible column combination.

Compare the family with the same frozen model, policies and internal splits. The primary
metric uses the task's options/weights. Measure added families and, when useful, remove one
family from the combined set or compare strict/fuzzy/no-link alternatives. A replacement is
one final-set experiment; a removal-only improvement is not a prerequisite. Shuffled-link
controls that preserve group sizes can help distinguish relationship signal from generic
grouping effects; they are not identity ground truth or a mandatory run.

Predeclare relevant segments from training semantics: high-confidence/ambiguous/unmatched,
known/new entities, sparse/dense history, time periods and missing/rare attributes. Report
row/entity support and baseline/candidate/delta by segment, with the same metric definition
and undefined values marked. Account for repeated-entity dependence when estimating uncertainty;
fold dispersion is diagnostic, not an automatic stability veto. Avoid choosing from hundreds
of segment/threshold variants on the same evidence and calling the winner independent proof.
Runtime selects by its internal comparison; segment harms guide interpretation and follow-ups.

Store a compact queue and linkage/availability/ablation notes in existing agent_notes and cite
returned source_ref values. Do not add invented required FeatureProposal fields, copy all raw
pairs into the prompt, or embed exploratory full-train maps in a program.

## Optional method implementations

Featuretools Deep Feature Synthesis composes aggregation/transform primitives across an
EntitySet's declared relationships. It does not infer reliable identity links. After validating
relations, inspect definitions with features_only, restrict allowed_paths and primitive_options,
and choose max_depth against measured redundancy/cost. Supply per-instance cutoff_time and,
where justified, training_window; deliberately choose include_cutoff_time. Keep availability
times and delayed labels explicit, restore prediction-row alignment and persist definitions
plus the fold-fitted reference state. Broad automatic synthesis still needs the preceding
prioritization and the final ablations. See [DFS](https://docs.featuretools.com/en/stable/getting_started/afe.html),
[time semantics](https://docs.featuretools.com/en/stable/getting_started/handling_time.html) and
[DFS parameters](https://docs.featuretools.com/en/stable/generated/featuretools.dfs.html).

Splink provides probabilistic record linkage with explicit comparisons, blocking, parameter
estimation and match scores. Its labelled accuracy analysis/threshold diagnostics support the
error procedure above; they require appropriate reference labels and sampling. Inspect
pair errors as well as clusters, estimate blocking loss, and retain unresolved records.
See [evaluation](https://moj-analytical-services.github.io/splink/demos/tutorials/07_Evaluation.html),
[edge review](https://moj-analytical-services.github.io/splink/topic_guides/evaluation/edge_overview.html)
and [blocking cost](https://moj-analytical-services.github.io/splink/topic_guides/blocking/performance.html).

These packages are optional method references, not added requirements. Use a package only
when the runtime reports it installed and a compatible fold-fitted implementation is possible.
Otherwise use existing Polars/pandas grouping, bounded candidate joins and explicit comparison
rules, or an installed train-fitted pair model. Do not call heuristic similarity a calibrated
match probability. Absence of these packages does not prevent testing a useful entity feature.
