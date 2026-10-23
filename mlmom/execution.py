"""Trusted inspection of fitted estimators and the export/recreate contract."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from .persistence import atomic_json


COUNT_KEYS = {'iterations','n_estimators','num_iterations','num_boost_round','epochs'}
ALIASES = {'num_iterations':'n_estimators','num_boost_round':'n_estimators',
           'eta':'learning_rate','lambda':'reg_lambda','alpha':'reg_alpha',
           'num_leaf':'num_leaves','min_data_in_leaf':'min_child_samples',
           'feature_fraction':'colsample_bytree','bagging_fraction':'subsample'}


class FidelityError(ValueError):
    def __init__(self, report, path):
        self.report, self.path = report, str(path)
        super().__init__('Execution fidelity failed: '+'; '.join(report['errors'])[:3500]+f' [report: {path}]')


def config_hash(params):
    return hashlib.sha256(json.dumps(params,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def _missing(expected, exported, prefix='model_params'):
    errors = []
    for key,value in expected.items():
        path = prefix+'.'+key
        if key not in exported:
            errors.append(f'Incomplete export: missing {path}')
        elif isinstance(value,dict):
            if not isinstance(exported[key],dict): errors.append(f'Export changes object {path}')
            else: errors.extend(_missing(value,exported[key],path))
        elif key not in COUNT_KEYS and value != exported[key]:
            errors.append(f'Export changes {path}: {value!r} -> {exported[key]!r}')
    return errors


def complete_export(model, requested):
    export = getattr(model,'export_model_params',None)
    exported = export() if callable(export) else copy.deepcopy(requested)
    if not isinstance(exported,dict): raise ValueError('export_model_params() must return a complete JSON object')
    # JSON is the actual persistence boundary; never accept NaN or opaque objects.
    exported = json.loads(json.dumps(exported,allow_nan=False))
    errors = _missing(requested,exported)
    if errors: raise ValueError('; '.join(errors))
    return exported


def fitted_components(candidate):
    """Unwrap only documented candidate/component boundaries, never arbitrary object graphs."""
    method = getattr(candidate,'fitted_components',None)
    if callable(method):
        components = method()
        if not isinstance(components,dict) or not components:
            raise ValueError('fitted_components() must return a nonempty name -> fitted estimator mapping')
        return components
    model = getattr(candidate,'model',None)
    if model is None: raise ValueError('Expose model or implement fitted_components() for every trained component')
    return {'model':model}


def _json_params(values):
    result = {}
    for key,value in values.items():
        try: result[key] = json.loads(json.dumps(value,allow_nan=False))
        except (TypeError,ValueError): continue
    return result


def inspect_component(model):
    """Numbers below come from fitted backend APIs, not candidate-provided JSON."""
    family = type(model).__module__.split('.')[0]
    if any(cls.__module__.startswith('torch.nn.') for cls in type(model).__mro__): family='torch'
    if hasattr(model,'steps') and any(cls.__module__.startswith('sklearn.') for cls in type(model).__mro__):family='sklearn'
    if family=='flaml' and hasattr(getattr(model,'model',None),'estimator'):
        return dict(backend='flaml',children={'selected':inspect_component(model.model.estimator)})
    if family == 'mlmom' or callable(getattr(model,'fitted_components',None)):
        return dict(backend=family,children={key:inspect_component(value) for key,value in fitted_components(model).items()})
    if family == 'sklearn' and hasattr(model,'steps'):
        leaf = inspect_component(model.steps[-1][1])
        leaf['actual_params'].update(_json_params(model.get_params(deep=True)))
        return leaf
    params = {}
    iterations = trees = best = retained = patience = None
    exhausted = False
    if family == 'catboost':
        if not model.is_fitted(): raise ValueError('CatBoost component is not fitted')
        params = model.get_all_params()
        trees = int(model.tree_count_)
        retained=trees
        history = model.get_evals_result()
        lengths = [len(series) for metrics in history.values() for series in metrics.values()]
        # metric_period can make the history sparse. Retained backend rounds
        # are an independent lower bound on how much training actually ran.
        iterations = max([trees,*lengths])
        best = model.get_best_iteration()
        best = int(best)+1 if best is not None and best >= 0 else None
        patience=params.get('od_wait') if params.get('od_type')=='Iter' else None
    elif family == 'lightgbm':
        params = model.get_params()
        params.update(model.booster_.params)
        retained=int(model.booster_.current_iteration())
        history=getattr(model,'evals_result_',{})
        lengths=[len(series) for metrics in history.values() for series in metrics.values()]
        iterations = max([retained,*lengths])
        trees = int(model.booster_.num_trees())
        best = int(model.best_iteration_) if getattr(model,'best_iteration_',0) > 0 else None
        dump = model.booster_.dump_model()
        exhausted = iterations == 1 and bool(dump['tree_info']) and dump['tree_info'][0]['num_leaves'] == 1
    elif family == 'xgboost':
        booster = model.get_booster() if hasattr(model,'get_booster') else model
        params = model.get_params() if hasattr(model,'get_params') else {}
        iterations = int(booster.num_boosted_rounds())
        retained=iterations
        trees = len(booster.get_dump())
        raw_best = booster.attr('best_iteration')
        best = int(raw_best)+1 if raw_best is not None else None
        if best is not None and hasattr(model,'get_booster'):retained=best
    elif family == 'sklearn':
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(model)
        params = model.get_params(deep=True)
        if type(model).__module__.startswith('sklearn.multioutput'):
            return dict(backend=family,children={f'target_{i}':inspect_component(value) for i,value in enumerate(model.estimators_)})
        if 'n_estimators' in params and hasattr(model,'estimators_'):
            iterations = int(getattr(model,'n_estimators_',len(model.estimators_)))
            trees = len(model.estimators_)
            retained=iterations
        elif getattr(model,'n_iter_',None) is not None:
            import numpy as np
            iterations = int(np.max(model.n_iter_))
    elif family == 'statsmodels':
        if not hasattr(model,'params'): raise ValueError('statsmodels component is not fitted')
        params = dict(add_intercept=bool(getattr(model.model,'k_constant',0)))
    elif family in {'flaml','autogluon','lightautoml'}:
        # Native frozen configuration is separately exercised by smoke/refit.
        if family == 'flaml':
            params = dict(getattr(model,'best_config',{}) or {})
        elif family == 'autogluon':
            params = model.model_info(model.model_best).get('hyperparameters',{})
        else: params = {}
    elif family == 'torch':
        import torch
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise ValueError('Torch model has non-finite fitted parameters')
        linear = [layer for layer in model.modules() if isinstance(layer,torch.nn.Linear)]
        params = dict(hidden_size=linear[0].out_features) if len(linear)>1 else {}
        return dict(backend=family,actual_params=params,completed_iterations=None,retained_trees=None,
                    best_iteration=None,backend_exhausted=False,inspection='unknown')
    else:
        raise ValueError(f'No trusted fitted-state inspector for {type(model).__module__}.{type(model).__name__}')
    return dict(backend=family,actual_params=_json_params(params),completed_iterations=iterations,
                retained_iterations=retained,retained_trees=trees,best_iteration=best,stopping_patience=patience,
                backend_exhausted=exhausted,inspection='verified')


def _equal(left,right):
    if isinstance(left,(int,float)) and isinstance(right,(int,float)):
        return math.isclose(left,right,rel_tol=1e-6,abs_tol=1e-9)
    return left == right


def _leaves(reports):
    for name,report in reports.items():
        if 'children' in report:
            for child,value in _leaves(report['children']): yield name+'.'+child,value
        else: yield name,report


def _component_objects(components):
    for name,component in components.items():
        if type(component).__module__.startswith('sklearn.multioutput'):
            for index,value in enumerate(component.estimators_):yield name+f'.target_{index}',value
        elif type(component).__module__.split('.')[0]=='flaml' and hasattr(getattr(component,'model',None),'estimator'):
            yield name+'.selected',component.model.estimator
        elif type(component).__module__.split('.')[0]=='mlmom' or callable(getattr(component,'fitted_components',None)):
            for child,value in _component_objects(fitted_components(component)):yield name+'.'+child,value
        else:yield name,component


def audit_fit(model, requested, config, context, path, *, stage, protocol_ref=None):
    version = 2 if config.get('contract_version') == 2 else 1
    report = dict(version=version,stage=stage,status='failed',requested_params=copy.deepcopy(requested),
                  code_hash=config.get('_code_hash'),
                  exported_params=None,components={},errors=[],roundtrip='not_checked',
                  validation_protocol_ref=protocol_ref,timeouts_enabled=context.timeouts_enabled,
                  seed=context.seed,cpus=context.cpus)
    try:
        try:
            exported = complete_export(model,requested)
        except (ValueError,TypeError) as exc:
            report['errors'].append(str(exc))
            # Inspection is independent: a bad export must not hide a one-tree fit.
            exported = {}
            raw=getattr(model,'export_model_params',None)
            if callable(raw):
                try:
                    value=json.loads(json.dumps(raw(),allow_nan=False))
                    if isinstance(value,dict):exported=value
                except (ValueError,TypeError):pass
        report['exported_params'] = exported
        components = fitted_components(model)
        if config.get('ensemble_components') and not callable(getattr(model,'fitted_components',None)):
            raise ValueError('Ensembles must implement fitted_components() for every component and meta learner')
        report['components'] = {name:inspect_component(component) for name,component in components.items()}
        if version == 2:
            from .model_spec import model_spec, ModelSpec
            from .fit_provenance import record_for, records_for
            method = getattr(model, 'export_model_spec', None)
            spec = method() if callable(method) else model_spec(config, exported)
            spec = ModelSpec.model_validate(spec).model_dump(mode='json')
            if spec['backend_params'] != exported:
                report['errors'].append('ModelSpec.backend_params differs from exported backend parameters')
            report['model_spec'] = spec
            objects = dict(_component_objects(components))
            for name, actual in _leaves(report['components']):
                component = objects.get(name)
                trace = record_for(component, records_for(context)) if component is not None else None
                if trace is None:
                    if actual['backend'] != 'torch':
                        report['errors'].append(f'{name}: missing trusted backend fit-entry provenance')
                    continue
                actual['fit_entry'] = trace
                if config.get('ensemble_sources'):
                    source=next((s for s in config['ensemble_sources'] if s['name']==name.split('.')[0]),None)
                    if source:
                        expected_spec=model_spec(source['config'])
                        if trace.get('code_hash')!=source['code_hash']:
                            report['errors'].append(f'{name}: fitted source differs from the frozen component')
                        for policy in ('target_transform','weight_policy','prediction_policy'):
                            if trace.get('model_spec',{}).get(policy)!=expected_spec[policy]:
                                report['errors'].append(f'{name}: component {policy} differs from its frozen policy')
                        if trace.get('rows')!=context.fit_rows:
                            report['errors'].append(f'{name}: component did not refit on every current training row')
                # Constructor state at fit entry takes precedence over post-fit mutable attributes.
                actual['actual_params'].update(trace['params'])
                if not trace['completed']:
                    report['errors'].append(f'{name}: backend fit did not complete')
                input_trace=trace.get('native_parent_entry') or trace
                if input_trace['target_fingerprint'] != trace['expected_target_fingerprint']:
                    columns=trace.get('target_columns')
                    column_name=name.rsplit('.',1)[-1]
                    if not (columns and column_name.startswith('target_') and
                            int(column_name[7:])<len(columns) and trace['target_fingerprint']==columns[int(column_name[7:])]):
                        report['errors'].append(f'{name}: backend target differs from the declared runtime transformation')
                if input_trace['weight_fingerprint'] != trace['expected_weight_fingerprint']:
                    report['errors'].append(f'{name}: backend sample_weight differs from the declared runtime weight policy')
        declared = config.get('ensemble_components',[])
        for entry in declared:
            identity=entry.get('name',entry.get('tool_id'))
            if identity and identity not in report['components']:
                report['errors'].append(f'Declared component {identity} is not exposed by fitted_components()')
        if declared and len(list(_leaves(report['components']))) < len(declared):
            report['errors'].append('Not every declared ensemble component is exposed for inspection')
        for name,actual in _leaves(report['components']):
            top_name=name.split('.')[0]
            if actual['backend']=='torch':
                # PreparedCandidate passes dataclass clones; shared per-optimizer traces
                # keep independent components' actual steps attached to their tensors.
                fitted=dict(_component_objects(components))[name]
                tensor_ids={id(parameter) for parameter in fitted.parameters()}
                traces=[trace for ids,trace in context.optimizer_traces.items() if set(ids) <= tensor_ids]
                if len(traces)==1:
                    trace=traces[0]
                    actual.update(completed_iterations=trace['steps']/trace['steps_per_epoch'],optimizer_steps=trace['steps'],
                                  steps_per_epoch=trace['steps_per_epoch'],inspection='verified')
                    actual['actual_params'].update(trace['params'])
            native = actual['actual_params']
            declaration = next((entry for entry in declared if entry.get('name',entry.get('tool_id'))==top_name),{})
            expected_backend=declaration.get('tool_id') if declared else config.get('tool_id')
            frozen=exported.get('__native_frozen',{})
            if expected_backend=='flaml':
                expected_backend={'lgbm':'lightgbm','rf':'sklearn','extra_tree':'sklearn','xgboost':'xgboost','catboost':'catboost'}.get(frozen.get('estimator'),'flaml')
            if expected_backend and actual['backend'] != expected_backend:
                report['errors'].append(f'{name}: planned backend {expected_backend}, actual {actual["backend"]}')
            expected = dict(declaration.get('model_params',{}))
            if config.get('tool_id')=='flaml':
                expected.update(frozen.get('config',{}))
                if 'log_max_bin' in expected:expected['max_bin']=2**int(expected.pop('log_max_bin'))-1
                if 'max_leaves' in expected and actual['backend']=='sklearn':expected['max_leaf_nodes']=expected.pop('max_leaves')
            if isinstance(requested.get(top_name),dict):expected.update(requested[top_name])
            # Flat Optuna overrides retain their precedence over component defaults.
            expected.update({key:value for key,value in requested.items() if not isinstance(value,dict) and
                             (not declared or config.get('parameter_routes',{}).get(key)==top_name)})
            actual['requested_params'] = expected
            policies=config.get('component_stopping_policies',{})
            policy = policies.get(name,policies.get(top_name,config.get('stopping_policy','fixed')))
            actual['stopping_policy'] = policy
            for key,value in expected.items():
                if key in COUNT_KEYS: continue
                canonical = ALIASES.get(key,key)
                present=key in native or canonical in native
                observed = native.get(key,native.get(canonical))
                if key in {'metric','eval_metric'}:
                    if isinstance(value,str) and isinstance(observed,list) and len(observed)==1:observed=observed[0]
                if present and not _equal(value,observed):
                    report['errors'].append(f'{name}.{key}: requested {value!r}, actual {observed!r}')
                elif version==2 and not present and key not in {'estimator','method','weights','components','__native_frozen','__ensemble_frozen'}:
                    report['errors'].append(f'{name}.{key}: declared backend parameter has no fit-time evidence')
            count = next((expected[key] for key in ('iterations','n_estimators','num_iterations','num_boost_round','epochs') if key in expected),None)
            if count is None:
                count = native.get('iterations',native.get('n_estimators'))
                if actual['backend']=='torch':count=exported.get('epochs')
            completed = actual['completed_iterations']
            retained=actual.get('retained_iterations')
            selected = actual.get('best_iteration')
            actual['requested_iterations'] = count
            if actual['inspection'] != 'verified':
                actual['stopping_reason'] = 'unknown'
                report['errors'].append(f'{name}: backend execution cannot be verified')
            elif count is None:
                actual['stopping_reason'] = 'fit_completed'
            elif completed is None:
                actual['stopping_reason'] = 'unknown'
                report['errors'].append(f'{name}: no actual iteration counter for requested {count}')
            elif completed > count:
                actual['stopping_reason'] = 'iteration_limit_exceeded'
                report['errors'].append(f'{name}: requested at most {count} iterations, completed {completed}; iteration limit exceeded')
            elif completed == count and (retained is None or retained >= count):
                actual['stopping_reason'] = 'iteration_limit'
            elif (policy == 'early_stopping' and context.validation_y is not None and selected is not None and
                  (completed >= count or (actual['backend']=='catboost' and actual.get('stopping_patience') is not None and
                                         completed-selected >= actual['stopping_patience']) or
                   (actual['backend']!='catboost' and completed>selected))):
                actual['stopping_reason'] = 'iteration_limit_best_model_selection' if completed>=count else 'early_stopping'
            elif actual['backend_exhausted']:
                actual['stopping_reason'] = 'backend_no_valid_split'
            else:
                actual['stopping_reason'] = 'unknown'
                report['errors'].append(f'{name}: requested {count} iterations, completed {completed}; unverified early termination')
            # Export may change the iteration cap only to a measured selected count.
            if declared:
                frozen_scope=dict(exported.get(top_name,{})) if isinstance(exported.get(top_name),dict) else {}
                frozen_scope.update({key:value for key,value in exported.items() if not isinstance(value,dict) and
                                     config.get('parameter_routes',{}).get(key)==top_name})
            else:frozen_scope=exported
            if actual['stopping_reason'] in {'early_stopping','iteration_limit_best_model_selection'} and retained is not None and count is not None and retained<count:
                count_key='iterations' if actual['backend']=='catboost' else 'n_estimators'
                frozen_count=frozen_scope.get(count_key,frozen_scope.get('num_iterations',frozen_scope.get('num_boost_round')))
                if frozen_count != retained:
                    report['errors'].append(f'{name}: export must freeze measured selected {count_key}={retained}, received {frozen_count!r}')
            for key in COUNT_KEYS & set(expected) & set(frozen_scope):
                frozen_count = frozen_scope[key]
                if frozen_count != expected[key] and not (policy == 'early_stopping' and
                        context.validation_y is not None and frozen_count == (selected or completed)):
                    report['errors'].append(f'{name}.{key}: export changes fixed iterations {expected[key]} -> {frozen_count}')
        report['configuration_hash'] = config_hash(report['model_spec'] if version == 2 else exported)
    except (ValueError,TypeError,AttributeError) as exc:
        report['errors'].append(str(exc))
    report['status'] = 'passed' if not report['errors'] else 'failed'
    atomic_json(Path(path),report)
    if report['errors']: raise FidelityError(report,path)
    return report


def compare_roundtrip(first, second):
    errors = []
    if first.get('version') == 2 and first.get('model_spec') != second.get('model_spec'):
        errors.append('export/recreate changes the complete ModelSpec')
    before,after = dict(_leaves(first['components'])),dict(_leaves(second['components']))
    if before.keys() != after.keys(): return ['export/recreate changes fitted component names']
    for name,actual in before.items():
        other = after[name]
        if actual['backend'] != other['backend']: errors.append(f'{name}: export/recreate changes backend')
        for key,value in actual['actual_params'].items():
            if key not in COUNT_KEYS and key in other['actual_params'] and not _equal(value,other['actual_params'][key]):
                errors.append(f'{name}.{key}: export/recreate changes effective parameter')
        if actual['completed_iterations'] != other['completed_iterations'] and actual['stopping_reason'] not in {'early_stopping','iteration_limit_best_model_selection'}:
            errors.append(f'{name}: export/recreate changes actual iterations')
    return errors


def require_admitted(run, candidate):
    report = run.execution_report
    if not report or report.get('status') != 'passed' or report.get('errors'):
        raise ValueError('Machine execution gate: missing or failed execution provenance')
    if report.get('version') not in {1,2} or not report.get('components') or any(item.get('inspection')!='verified' for _,item in _leaves(report['components'])):
        raise ValueError('Machine execution gate: actual fitted components are not verified')
    smoke=getattr(candidate,'execution_report',None)
    if not smoke or smoke.get('roundtrip')!='passed' or smoke.get('code_hash')!=candidate.code_hash:
        raise ValueError('Machine execution gate: export/recreate smoke provenance is missing')
    if report.get('version') == 2 and smoke.get('version') != 2:
        raise ValueError('Machine execution gate: ModelSpec v2 requires a v2 export/recreate smoke check')
    if report.get('code_hash') != candidate.code_hash:
        raise ValueError('Machine execution gate: execution code hash differs from candidate')
    identity = report.get('model_spec') if report.get('version') == 2 else run.best_params
    if report.get('exported_params') != run.best_params or report.get('configuration_hash') != config_hash(identity):
        raise ValueError('Machine execution gate: saved parameters differ from verified export')
    if report.get('version') == 2 and getattr(run, 'model_spec', None) != report.get('model_spec'):
        raise ValueError('Machine execution gate: saved ModelSpec differs from verified execution')
    if not run.full_refit: raise ValueError('Machine execution gate: full-train refit is missing')
