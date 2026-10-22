# Generated candidate persistence

Implement your own Candidate with fit/predict/save and module factories:
```python
from pathlib import Path
import cloudpickle

def create_candidate(config):
    return Candidate(config)

def load_candidate(artifact_dir):
    return cloudpickle.loads((Path(artifact_dir) / 'candidate.pkl').read_bytes())

# Candidate.save writes cloudpickle.dumps(self) to artifact_dir/candidate.pkl.
```
Store only inference state, not FitContext or X/y. Native predictors referencing external model files must copy files into artifact_dir and rebase on load. Trace every promised operation to code: categorical casts with column names, missing policy, inverse target transform, clipping, class order and parameter consumption. Preserve column order and fit transforms on allowed training rows only. No project adapter imports.
