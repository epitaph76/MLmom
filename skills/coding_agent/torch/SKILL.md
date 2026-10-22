---
name: torch
description: Train a compact CPU tabular MLP with PyTorch 2.5.1 when data size and budget justify it after a boosting baseline.
---
# PyTorch 2.5.1

Prefer boosting for small/category-heavy data. A small reference baseline uses hidden_size (64), epochs (20), lr (.001), batch_size (256). The model is Linear-ReLU-Linear on dense scaled OHE, without validation patience or sample weights. Generate other layers/loss policies explicitly when planned.

Start width 32-128, batch 128-512 and bounded epochs. Estimate dense arrays/tensor copies. Set torch threads from context; keep loaders serial and check remaining_seconds inside minibatches.

Single-label classification uses encoded long targets and raw logits with CrossEntropyLoss, no prior softmax. Binary/multilabel can use float targets with BCEWithLogitsLoss; regression uses MSE/MAE. Apply softmax/sigmoid only at inference; map labels/output columns back to the SDK.

Custom stopping monitors internal validation in eval/no_grad, deep-copies/restores the best state_dict and exports the selected epoch count for fixed refit. Predict in bounded batches. Persist architecture, state_dict, preprocessing and label mapping; load on CPU only from this solution's trusted artifacts. No external checkpoints.
