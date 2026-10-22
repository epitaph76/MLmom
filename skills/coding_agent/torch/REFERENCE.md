> Role boundary: the SDK supplies already transformed X and validation_X. Any feature cleaning/encoding in the standalone API excerpts below belongs to the preprocessing agent; omit it from model code. Use preprocessing_spec for category indices/dtypes. Persist only estimator and output/target state; SDK persists preprocessing.

# Direct CPU PyTorch reference
```python
import torch
# Fit and save imputer/encoder/scaler on train before building tensors.
torch.set_num_threads(context.cpus)
torch.manual_seed(context.seed)
self.model = torch.nn.Sequential(torch.nn.Linear(width,hidden), torch.nn.ReLU(), torch.nn.Linear(hidden,outputs))
optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
for epoch in range(epochs):
    if context.remaining_seconds < 1: break
    self.model.train()
    for ids in torch.randperm(len(values)).split(batch_size):
        optimizer.zero_grad()
        loss = criterion(self.model(values[ids]), target[ids])
        loss.backward()
        optimizer.step()
        if context.remaining_seconds < 1: break
```
Implement bounded CPU tensors and preprocessing; classification labels use train-only encoding. Raw logits feed CrossEntropyLoss or BCEWithLogitsLoss; softmax/sigmoid apply at inference only. Weighted losses need reduction='none' and aligned weights. Monitor exact internal validation metric under eval/no_grad, clone best state_dict and restore it. Export chosen epochs for validation-free refit. Persist architecture, state and transforms; no external checkpoints. No nested HPO.
