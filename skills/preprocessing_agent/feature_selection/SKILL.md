---
name: feature-selection
description: Diagnose feature contribution with repeated/grouped importance and ablation, then fit justified selection inside folds.
---
# Feature selection

Use [REFERENCE.md](REFERENCE.md) for repeated permutation, joint permutation of correlated
feature families, drop-column retraining and fold-wise selection stability. Importance is
model/data/split dependent; verify predictive quality first. Never remove a feature because
one importance estimate is near zero or negative. A one-shot diagnostic is preliminary.

Exclude declared service/target columns and confirmed unavailable/post-outcome inputs.
An identifier-like name, high correlation or low variance alone does not prove uselessness
or leakage. Correlated substitutes can hide each other's individual contribution; inspect
their joint importance and refitted ablations before choosing a representative.

Learn masks, correlation groups, importance-based thresholds and reductions inside each
training fold. Use training-internal held-out/OOF diagnostics or explicitly planned nested
selection; never choose on outer labels or a global all-labelled-data ranking before CV.
Preprocessing has no outer validation scorer/model: request missing diagnostics through the
experiment owner, or fit only an assigned train-only selector. Do not redesign the estimator.

Compare retained/dropped sets with the same primary metric/options/weights and matched
budgets; retain repeat/fold distributions and group-level selection frequencies. Instability
is a diagnostic, not an automatic refusal. Preserve sparse structure and fit-time schema;
persist selected names/indices and handle reordered, extra, missing and new-level inputs.
