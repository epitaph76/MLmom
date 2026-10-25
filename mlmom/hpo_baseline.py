"""Reuse measured CV evidence only when it describes this exact experiment."""
import json
from pathlib import Path

from .feature_contracts import feature_hash
from .persistence import file_hash
from .tracing import emit
from .execution_resources import model_parameters


def same_fixed_settings(original, current, exported, space):
    """Compare actual assignment knobs; exports may include source-owned defaults."""
    old = model_parameters(original)
    new = model_parameters(current)
    measured = model_parameters(exported)
    # Removing/changing an explicit old knob is a different experiment. Newly frozen
    # defaults are reusable only when their measured export has the same value.
    return (all(key in space or (key in new and value == new[key]) for key,value in old.items()) and
            all(key in space or key in old or (key in measured and value == measured[key])
                for key,value in new.items()))


def importable_baseline(request, protocol):
    source = request.get('baseline_evidence')
    if not source:
        return None
    try:
        original = json.loads(Path(source['request_ref']).read_text(encoding='utf-8'))
        selection = json.loads(Path(source['selection_ref']).read_text(encoding='utf-8'))
        old, current = original['plan'], request['plan']
        space = current['search_space']
        comparable = (
            selection.get('execution_identity') == dict(code_hash=request['code_hash'],train_hash=file_hash(Path(request['train_path']))) and
            original['code_hash'] == request['code_hash'] and
            file_hash(Path(original['train_path'])) == file_hash(Path(request['train_path'])) and
            original['context']['task'] == request['context']['task'] and
            all(original['context'].get(k) == request['context'].get(k)
                for k in ('feature_columns','classes','submission_kind','table_formats')) and
            original['context'].get('metric_options', {}) == request['context'].get('metric_options', {}) and
            selection['protocol'] == protocol and
            old.get('search_space',{}) == space and
            feature_hash(old.get('feature_set')) == feature_hash(current.get('feature_set')) and
            old.get('stopping_policy', 'fixed') == current.get('stopping_policy', 'fixed') == 'fixed' and
            all(old.get(k) == current.get(k) for k in ('tool_id','preprocessing','target_transform','weight_policy','prediction_policy')) and
            not any('.' in k for k in space) and
            same_fixed_settings(old['model_params'],current['model_params'],selection['params'],space))
        if not comparable:
            emit('hpo.baseline_not_imported', {'reason':'Implementation, configuration, data or evaluation differs'})
            return None
        return dict(params={k:old['model_params'][k] if k in old['model_params'] else selection['params'][k] for k in space}, score=selection['score'],
                    selection_ref=source['selection_ref'])
    except (OSError, ValueError, KeyError, TypeError) as error:
        emit('hpo.baseline_not_imported', {'reason':str(error)})
        return None
