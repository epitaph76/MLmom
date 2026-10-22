---
name: text-features
description: Build bounded sparse classical text representations instead of treating free text as ordinary categories.
---
# Text features

Distinguish free text from categorical strings using profile evidence such as length, token diversity and uniqueness. Empty text needs an explicit policy. Check duplicate or template text and suspicious target-bearing metadata before modeling.

For classical ML, start with bounded word or character TF-IDF and a sparse-compatible linear estimator. Control ngram range, min/max document frequency and maximum features under the memory budget. Hashing is an option when a fixed dimension is required, but its collisions and lack of an inspectable vocabulary are tradeoffs.

Fit vocabulary and IDF only on the current training fold and persist them. Combine sparse text and tabular branches with a sparse ColumnTransformer; do not densify a large text matrix for a tree by accident. Unknown inference tokens must be harmless.

Class weights or conservative undersampling are usually better first imbalance experiments for sparse text than vanilla SMOTE in a high-dimensional neighbour space. Compare word, character or combined representations on identical folds and include fit/prediction cost.
