"""Offline LLM fixture. Never imported by solution.py or shipped as a fallback."""
import json
from mlmom.contracts import TaskDefinition,ValidationStrategy
from mlmom.pipeline_contracts import HypothesisBatch,HypothesisPlan,PlanBatch,ValidationReport
from mlmom.coding import CodeBundle

ADAPTER_CODE = '''from mlmom.adapters import LibraryCandidate, load_adapter
def create_candidate(config):
    return LibraryCandidate(config)
def load_candidate(artifact_dir):
    return load_adapter(artifact_dir)
'''


class OfflineLLM:
    model='offline/test'
    def __init__(self, *, tool_id='sklearn',params=None,fail=False,seconds=15,cpus=1):
        self.tool_id=tool_id;self.params=params or {'n_estimators':8};self.fail=fail;self.calls=[];self.seconds=seconds;self.cpus=cpus
    def complete(self,messages):
        if self.fail: raise RuntimeError('API unavailable')
        payload=json.loads(messages[-1]['content'])
        inferred=payload['dataset_inspection']['task']
        kind=inferred['task_type']
        task=TaskDefinition(objective='Solve the supplied task.',task_type=kind,
            target_columns=inferred['target_columns'],id_columns=inferred['id_columns'],
            time_column=inferred.get('time_column'),group_columns=inferred.get('group_columns',[]),
            weight_column=inferred.get('weight_column'),excluded_feature_columns=inferred['id_columns']+inferred['target_columns'],
            primary_metric=inferred['primary_metric'],metric_direction=inferred['metric_direction'],
            validation_strategy=ValidationStrategy.TIME_SERIES_SPLIT if inferred.get('time_column') else
                ValidationStrategy.GROUP_KFOLD if inferred.get('group_columns') else
                ValidationStrategy.STRATIFIED_KFOLD if 'classification' in kind else ValidationStrategy.KFOLD,
            result_format=inferred['result_format'])
        return task.model_dump_json()
    def generate(self,role,instruction,payload,schema,tools=None):
        self.calls.append((role,payload))
        if role=='hypothesis_generator':
            return HypothesisBatch(hypotheses=[dict(hypothesis_id=f"h{payload['round_number']}",description='Small tree baseline',rationale='Fast CPU baseline',baseline=True)])
        if role=='planner':
            from mlmom.evaluation import prediction_kind
            return PlanBatch(plans=[HypothesisPlan(hypothesis=h,tool_id=self.tool_id,skill_ids=[self.tool_id],model_params=self.params,
                time_limit_seconds=self.seconds,cpus=self.cpus,memory_gb=2,max_trials=1,
                prediction_kind=prediction_kind(payload['task']['primary_metric'],payload['task']['task_type'])) for h in payload['hypotheses']['hypotheses']])
        if role=='plan_validator':return ValidationReport(valid=True)
        if role=='coding_agent':return CodeBundle(files={'candidate.py':ADAPTER_CODE})
        raise AssertionError(role)
