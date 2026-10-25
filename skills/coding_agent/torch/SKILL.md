---
name: torch
description: Train a compact weighted CPU PyTorch 2.5.1 MLP on prepared dense data.
---
# PyTorch 2.5.1

The complete REFERENCE.md uses a dense Linear-ReLU-Linear baseline with hidden_size, epochs, lr,
batch_size and patience_epochs. Estimate tensor memory; set torch threads from context.cpus.
Single-label classification uses encoded long targets and CrossEntropyLoss on logits; multilabel
uses BCEWithLogitsLoss, regression MSE. Sample weights apply to per-row losses.

Optional stopping calls context.validation_scorer under eval/no_grad, restores a copied best
state_dict and exports the selected epoch count. Otherwise train fixed epochs and record why.
Check remaining_seconds only when timeouts_enabled. No outer or internal HPO loop belongs here.

Predict in bounded CPU batches; softmax/sigmoid apply only for classification inference.
Persist architecture, label mapping and state_dict. Load weights with map_location='cpu' and
weights_only=True from this solution's artifacts. Do not pickle FitContext, validation closures
or optimizer state. No external checkpoints or downloads.