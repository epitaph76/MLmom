# Estimator-only torch reference

Runnable CPU MLP for dense regression, multioutput regression, binary/multiclass and multilabel classification. Sample weights are applied to per-row losses.
The SDK supplies transformed X and owns target/weight/output policies.

```python
from pathlib import Path
import copy
import cloudpickle
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

class Candidate:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.params = dict(config.get('model_params', {}))

    def fit(self, X, y, context):
        torch.set_num_threads(context.cpus)
        torch.manual_seed(context.seed)
        self.kind = context.task_type
        self.classes = list(context.classes)
        self.single_class = self.kind in {'binary_classification','multiclass_classification'}
        self.multilabel = self.kind == 'multilabel_classification'
        self.encoder = LabelEncoder().fit(context.classes) if self.single_class else None
        values = torch.as_tensor(np.asarray(X,dtype=np.float32))
        array = np.asarray(y)
        if self.single_class:
            target = torch.tensor(self.encoder.transform(array),dtype=torch.long)
            outputs = len(self.encoder.classes_)
            criterion = torch.nn.CrossEntropyLoss(reduction='none')
        else:
            array = array[:,None] if array.ndim == 1 else array
            target = torch.tensor(array,dtype=torch.float32)
            outputs = array.shape[1]
            criterion = torch.nn.BCEWithLogitsLoss(reduction='none') if self.multilabel else torch.nn.MSELoss(reduction='none')
        params = dict(hidden_size=64, epochs=20, lr=.001, batch_size=256, patience_epochs=5) | self.params
        unknown = set(params)-{'hidden_size','epochs','lr','batch_size','patience_epochs'}
        if unknown:
            raise ValueError('Unsupported MLP parameters: '+', '.join(sorted(unknown)))
        self.params = params
        self.shape = (values.shape[1],int(params['hidden_size']),outputs)
        self.model = torch.nn.Sequential(torch.nn.Linear(self.shape[0],self.shape[1]),
                                        torch.nn.ReLU(),torch.nn.Linear(int(params['hidden_size']),outputs))
        optimizer = torch.optim.Adam(self.model.parameters(),lr=float(params['lr']))
        weights = torch.as_tensor(np.asarray(context.sample_weight,dtype=np.float32)) if context.sample_weight is not None else None
        scorer = context.validation_scorer
        self.use_stopping = self.config.get('stopping_policy') == 'early_stopping' and context.validation_X is not None and scorer is not None
        best = None
        best_state = None
        best_epoch = 0
        stale = 0
        trained = 0
        self.stopping_info = dict(mode='primary_metric' if self.use_stopping else 'fixed',
            reason='SDK original-scale primary metric' if self.use_stopping else
                   'Fixed epochs: no exact holdout scorer or fixed request', metric=context.metric)
        for epoch in range(int(params['epochs'])):
            if context.timeouts_enabled and context.remaining_seconds < .2:
                break
            self.model.train()
            for indices in torch.randperm(len(values)).split(int(params['batch_size'])):
                if context.timeouts_enabled and context.remaining_seconds < .2:
                    break
                optimizer.zero_grad()
                losses = criterion(self.model(values[indices]),target[indices])
                losses = losses.mean(dim=1) if losses.ndim > 1 else losses
                loss = losses.mean() if weights is None else (losses*weights[indices]).sum()/weights[indices].sum().clamp_min(1e-12)
                loss.backward()
                optimizer.step()
            trained = epoch+1
            if self.use_stopping:
                score = scorer(self.predict(context.validation_X,{'kind':scorer.kind}),scorer.kind)
                improved = best is None or (score > best if context.direction == 'maximize' else score < best)
                if improved:
                    best, best_state, best_epoch, stale = score, copy.deepcopy(self.model.state_dict()), trained, 0
                else:
                    stale += 1
                if stale >= int(params['patience_epochs']):
                    break
        if trained == 0:
            raise TimeoutError('No MLP epoch completed')
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.params['epochs'] = best_epoch
        else:
            self.params['epochs'] = trained
        self.model.eval()
        return self

    def predict(self, X, prediction_spec):
        self.model.eval()
        array = torch.as_tensor(np.asarray(X,dtype=np.float32))
        with torch.no_grad():
            values = torch.cat([self.model(batch) for batch in array.split(int(self.params['batch_size']))]).numpy()
        if self.single_class:
            probability = np.exp(values-values.max(axis=1,keepdims=True))
            probability /= probability.sum(axis=1,keepdims=True)
            if prediction_spec['kind'] == 'probability':
                return probability[:,self.encoder.transform(self.classes)]
            return self.encoder.inverse_transform(probability.argmax(axis=1))
        if self.multilabel:
            probability = 1/(1+np.exp(-values))
            return probability if prediction_spec['kind'] == 'probability' else (probability >= .5).astype(int)
        return values[:,0] if values.shape[1] == 1 else values

    def export_model_params(self):
        return dict(self.params)

    def save(self, artifact_dir):
        torch.save(self.model.state_dict(),Path(artifact_dir)/'weights.pt')
        state={k:v for k,v in vars(self).items() if k!='model'}
        (Path(artifact_dir)/'state.pkl').write_bytes(cloudpickle.dumps(state))

def create_candidate(config):
    return Candidate(config)

def load_candidate(artifact_dir):
    candidate=Candidate({})
    vars(candidate).update(cloudpickle.loads((Path(artifact_dir)/'state.pkl').read_bytes()))
    width,hidden,outputs=candidate.shape
    torch.set_num_threads(1)
    candidate.model=torch.nn.Sequential(torch.nn.Linear(width,hidden),torch.nn.ReLU(),torch.nn.Linear(hidden,outputs))
    candidate.model.load_state_dict(torch.load(Path(artifact_dir)/'weights.pt',map_location='cpu',weights_only=True))
    candidate.model.eval()
    return candidate
```

No preprocessing or SDK output policies are duplicated. The optimizer, targets, FitContext and validation scorer remain local to fit. Restore the selected state and epoch count for fixed refit. Inference is batched; artifacts contain only this solution's trusted CPU inference state.
