"""Family ownership, OOF isolation, frozen policies and independent ensemble slots."""
import copy
import json
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from mlmom.pipeline_contracts import (Hypothesis, HypothesisPlan, SearchGroupPlan,
    EnsembleProposal, PlanBatch, RunConfig)
from mlmom.search_groups import (expand_families, collect_groups, component_pool,
    configure_composite, ensemble_recipes, internal_winner)
from mlmom.search_groups import limit_parent_groups
from mlmom.persistence import ExperimentStore, tree_hash
from mlmom.sdk import FrozenEnsemble, FitContext, load_module
from mlmom.planning import build_planning_graph
from mlmom.worker import synthetic
from tests.fakes import MODEL_CODE, PREPROCESSING_CODE, OfflineLLM
from tests.test_execution import execution_case
from tests.test_planning_limits import case, Planner, Skills

COMPOSITION_CODE = '''from pathlib import Path
from mlmom.sdk import FrozenEnsemble
def create_candidate(config):
    return FrozenEnsemble(config, Path(__file__).resolve().parent / 'components')
def load_candidate(artifact_dir):
    return FrozenEnsemble.load(artifact_dir)
'''


def composite_case(root, kind='regression', method='mean', different_policies=False):
    root.mkdir(parents=True,exist_ok=True)
    _, context, plan, _ = execution_case(root, kind)
    code = root/'composite'
    code.mkdir()
    (code/'candidate.py').write_text(COMPOSITION_CODE,encoding='utf-8')
    sources = []
    for i in range(2):
        component = code/'components'/f'base_{i}'
        component.mkdir(parents=True)
        (component/'candidate.py').write_text(MODEL_CODE,encoding='utf-8')
        (component/'preprocessing.py').write_text(PREPROCESSING_CODE,encoding='utf-8')
        base = plan.model_copy(deep=True)
        base.model_params = dict(n_estimators=3+i)
        if different_policies:
            if 'regression' in kind and i==1:
                base.target_transform.name='log1p'
                if kind=='regression':base.weight_policy.name='tail_aware'
            elif kind in {'binary_classification','multiclass_classification'} and i==1:
                base.weight_policy.name='sqrt_balanced'
        sources.append(dict(name=f'base_{i}',candidate_id=f'c{i}',experiment_id=f'e{i}',
            code_hash=tree_hash(component),config=base.model_dump(mode='json')))
    plan = configure_composite(plan,sources,3)
    plan.model_params['method'] = method
    X,y = synthetic(kind)
    if 'regression' in kind:
        y = y-y.min()+1
    fit = FitContext(task_type=kind,metric=context.task.primary_metric,direction=context.task.metric_direction,
        seed=42,cpus=1,memory_gb=2,time_limit_seconds=60,timeouts_enabled=False,
        classes=context.classes,task_context=context,work_dir=str(root/'fit'))
    return code,context,plan,X,y,fit


class FamilyPlanTests(unittest.TestCase):
    def test_single_group_label_is_normalized_without_rewriting_historical_lineage(self):
        plan = case()[3]
        plan.hypothesis.hypothesis_id = 'h2-log1p-target-transform'
        plan.hypothesis.based_on_hypothesis_id = 'old-backend-child'
        plan.parent_hypothesis = plan.hypothesis.model_copy(update={'hypothesis_id': 'old-family'})
        plan.search_group_id = 'h2-single-change-ablation'
        group = collect_groups([plan])[0]
        variant = group.variants[0]
        self.assertEqual(group.hypothesis.hypothesis_id, plan.hypothesis.hypothesis_id)
        self.assertEqual(variant.search_group_id, plan.hypothesis.hypothesis_id)
        self.assertEqual(variant.parent_hypothesis.hypothesis_id, 'old-family')
        self.assertEqual(variant.hypothesis.based_on_hypothesis_id, 'old-backend-child')
        self.assertEqual(variant.agent_notes['requested_search_group_id'], 'h2-single-change-ablation')
        self.assertEqual(plan.search_group_id, 'h2-single-change-ablation')

    def test_standalone_hypotheses_do_not_merge_by_an_invented_shared_group_label(self):
        plans = []
        for key in ['first', 'second']:
            plan = case()[3]
            plan.hypothesis.hypothesis_id, plan.search_group_id = key, 'shared-label'
            plans.append(plan)
        groups = collect_groups(plans)
        self.assertEqual([g.hypothesis.hypothesis_id for g in groups], ['first', 'second'])
        self.assertTrue(all(len(g.variants) == 1 for g in groups))

    def test_family_group_mismatch_remains_an_error(self):
        plan = case()[3]
        plan.parent_hypothesis = plan.hypothesis.model_copy(update={'hypothesis_id': 'actual-family'})
        plan.search_group_kind, plan.search_group_id = 'family', 'wrong-family'
        with self.assertRaisesRegex(ValueError, 'different search group'):
            collect_groups([plan])

    def test_accepted_cached_single_group_is_repaired_without_another_llm_call(self):
        context, tools, budget, _ = case()
        saved = {}
        def save(progress):
            saved.update(json.loads(json.dumps(progress)))
        first = build_planning_graph(Planner(), context, tools, Skills(), budget,
            save_progress=save).invoke({'payload': {'round_number': 2}})
        self.assertTrue(first['valid'], first)
        for phase in ['proposed', 'accepted']:
            saved[phase]['base']['search_group_id'] = 'narrative-ablation-label'
            saved[phase]['base']['parent_hypothesis'] = dict(
                saved[phase]['base']['hypothesis'], hypothesis_id='old-family')
        client = Planner()
        resumed = build_planning_graph(client, context, tools, Skills(), budget,
            saved_progress=saved, save_progress=save).invoke(dict(
                payload={'round_number': 2}, hypotheses=saved['hypotheses']))
        self.assertTrue(resumed['valid'], resumed)
        self.assertEqual(client.calls, [])
        base = next(p for p in resumed['plans'] if p['hypothesis']['hypothesis_id'] == 'base')
        self.assertEqual(base['search_group_id'], 'base')
        self.assertEqual(base['parent_hypothesis']['hypothesis_id'], 'old-family')
        self.assertEqual(saved['accepted']['base']['search_group_id'], 'base')

    def test_boosting_is_one_parent_with_three_stable_variants(self):
        context,tools,budget,_ = case()
        parent = Hypothesis(hypothesis_id='trees',kind='family',model_family='gradient_boosting',
                            baseline=True,description='Boosting family',rationale='Measured evidence')
        all_children,meta,skipped = expand_families([parent],tools,'regression')
        first,_,_ = expand_families([parent],tools,'regression','baseline')
        rest,_,_ = expand_families([parent],tools,'regression','alternatives')
        self.assertEqual([meta[c.hypothesis_id][1] for c in all_children],['lightgbm','catboost','xgboost'])
        self.assertEqual([c.hypothesis_id for c in [*first,*rest]],[c.hypothesis_id for c in all_children])
        self.assertEqual(sum(c.baseline for c in all_children),1)
        self.assertEqual(skipped,[])
        class FamilyPlanner(Planner):
            def generate(self,role,instruction,payload,schema,tools=None):
                if role=='planner':
                    _,_,_,plan=case()
                    plan.hypothesis=Hypothesis.model_validate(payload['hypotheses']['hypotheses'][0])
                    plan.tool_id=payload['required_tool_id'];plan.skill_ids=[plan.tool_id]
                    return PlanBatch(plans=[plan])
                return super().generate(role,instruction,payload,schema,tools)
        output = build_planning_graph(FamilyPlanner(),context,tools,Skills(),budget,lightweight=False).invoke(
            dict(payload={'round_number':2},hypotheses={'hypotheses':[parent.model_dump()]}))
        self.assertTrue(output['valid'],output)
        self.assertEqual(len(output['plans']),3)
        self.assertEqual(len(output['groups']),1)
        self.assertEqual(output['groups'][0]['hypothesis']['hypothesis_id'],'trees')
        self.assertTrue(all(p['strategy']=='single' for p in output['plans']))
        self.assertEqual(len(limit_parent_groups(output['plans'],1)),3)

    def test_unavailable_family_member_has_explicit_reason(self):
        _,tools,_,_=case();tools['catboost'].available=False;tools['catboost'].reason='distribution not installed'
        parent=Hypothesis(hypothesis_id='g',kind='family',model_family='gradient_boosting',description='GBDT',rationale='test')
        children,_,skipped=expand_families([parent],tools,'regression')
        self.assertEqual(len(children),3)
        self.assertEqual(skipped,[])

    def test_pipeline_plans_every_selected_family_variant_and_resumes_deferred_members(self):
        from mlmom.pipeline import Pipeline
        from mlmom.tools import ToolSet
        from types import SimpleNamespace

        class FamilyPlanner(Planner):
            def generate(self, role, instruction, payload, schema, tools=None):
                if role == 'planner':
                    self.calls.append((role, payload))
                    _, _, _, plan = case()
                    plan.hypothesis = Hypothesis.model_validate(payload['hypotheses']['hypotheses'][0])
                    plan.tool_id = payload['required_tool_id']
                    plan.skill_ids = [plan.tool_id]
                    return PlanBatch(plans=[plan])
                return super().generate(role, instruction, payload, schema, tools)

        # The first accepted plan may only defer work in baseline mode. Calls
        # through Pipeline must retain all members in later rounds/alternatives.
        for baseline in (False, True):
            with self.subTest(baseline=baseline), tempfile.TemporaryDirectory() as tmp:
                context, tools, budget, _ = case()
                parent = Hypothesis(hypothesis_id='trees', kind='family', model_family='gradient_boosting',
                    baseline=baseline, description='Boosting family', rationale='Measured evidence')
                hypotheses = {'hypotheses': [parent.model_dump()]}
                pipeline = object.__new__(Pipeline)
                pipeline.config = RunConfig(coding_seconds=60, prediction_seconds=10)
                pipeline.client = FamilyPlanner()
                pipeline.tools, pipeline.skills, pipeline.role_tools = tools, Skills(), ToolSet()
                pipeline.restore = lambda state: context
                pipeline.budget = SimpleNamespace(snapshot=lambda: budget)
                pipeline.last_submission = None if baseline else {'path': 'submission.csv'}
                pipeline.planning_payload = lambda state: dict(round_number=state['round_number'], history={'evaluation': []})
                # Unselected libraries must not be probed before planning.
                pipeline.preflight_tools = lambda **kwargs: self.fail('Unexpected eager capability probe')
                pipeline.store = ExperimentStore(Path(tmp))
                try:
                    if baseline:
                        state = {'round_number': 1}
                        first = pipeline.planning_call(state, mode='baseline', hypotheses=hypotheses)
                        self.assertTrue(first['valid'], first)
                        self.assertEqual([p['tool_id'] for p in first['plans']], ['lightgbm'])
                        pipeline.last_submission = {'path': 'submission.csv'}
                        deferred = pipeline.planning_call(state, mode='alternatives', hypotheses=first['hypotheses'])
                        self.assertTrue(deferred['valid'], deferred)
                        self.assertEqual([p['tool_id'] for p in deferred['plans']], ['catboost', 'xgboost'])
                        all_plans = first['plans'] + deferred['plans']
                        calls = len(pipeline.client.calls)
                        self.assertEqual(pipeline.planning_call(state, mode='alternatives'), deferred)
                        self.assertEqual(len(pipeline.client.calls), calls)
                    else:
                        result = pipeline.planning_call({'round_number': 2}, hypotheses=hypotheses)
                        self.assertTrue(result['valid'], result)
                        all_plans = result['plans']
                    self.assertEqual([p['tool_id'] for p in all_plans], ['lightgbm', 'catboost', 'xgboost'])
                    self.assertEqual(len({p['hypothesis']['hypothesis_id'] for p in all_plans}), 3)
                    group = pipeline.store.get('search_group', ('r1-' if baseline else 'r2-') + 'trees')
                    self.assertEqual(len(group['variants']), 3)
                finally:
                    pipeline.store.close()

    def test_legacy_single_group_and_internal_winner_direction(self):
        _,_,_,plan=case()
        self.assertEqual(collect_groups([plan])[0].kind,'single')
        rows=[dict(run=dict(status='completed',internal_score=s)) for s in [3.,1.,2.]]
        self.assertEqual(internal_winner(rows,'minimize')['run']['internal_score'],1.)
        self.assertEqual(internal_winner(rows,'maximize')['run']['internal_score'],3.)
        tied=[dict(run=dict(status='completed',internal_score=1.,experiment_id=method),
                   plan=dict(model_params={'method':method})) for method in ['stack','mean']]
        self.assertEqual(internal_winner(tied,'minimize')['run']['experiment_id'],'mean')
        self.assertEqual(internal_winner(list(reversed(tied)),'minimize')['run']['experiment_id'],'mean')
        with self.assertRaises(ValueError):
            EnsembleProposal(component_experiment_ids=['a','a'],rationale='duplicate')
        with self.assertRaises(ValueError):
            SearchGroupPlan(hypothesis=plan.hypothesis,variants=[plan,plan])

    def test_component_pool_excludes_current_incomplete_and_duplicate_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=ExperimentStore(Path(tmp))
            store.put('round_result','r1',dict(round_id='r1'))
            store.put('cycle','r1',dict(round_id='r1',completed=True))
            for key,round_id,digest in [('a','r1','x'),('b','r1','y'),('dup','r1','x'),('c','r2','z')]:
                store.put('component',key,dict(experiment_id=key,round_id=round_id,
                    candidate={'code_hash':digest},configuration_hash=digest))
            self.assertEqual({r['experiment_id'] for r in component_pool(store,2)},{'dup','b'})
            self.assertEqual(component_pool(store,1),[])
            store.close()


class CompositeTests(unittest.TestCase):
    def test_all_regression_recipes_preserve_component_policies_and_roundtrip(self):
        for method in ['mean','median','weighted','stack']:
            with self.subTest(method=method),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                code,context,plan,X,y,c=composite_case(root,method=method,different_policies=True)
                config=dict(plan.model_dump(mode='json'),_code_hash=tree_hash(code))
                module=load_module(code);model=module.create_candidate(config)
                model.fit(X,y,c)
                prediction=model.predict(X,{'kind':'value'})
                self.assertTrue(np.isfinite(prediction).all())
                frozen=dict(config,model_params=model.export_model_params())
                recreated=module.create_candidate(frozen)
                c2=copy.copy(c);c2.fit_id=uuid.uuid4().hex;c2.work_dir=str(root/'recreate')
                recreated.fit(X,y,c2)
                np.testing.assert_allclose(prediction,recreated.predict(X,{'kind':'value'}))
                artifact=root/'artifact';model.save(artifact)
                moved=root/'moved';artifact.rename(moved)
                shutil.rmtree(code)
                restored=load_module(moved/'source').load_candidate(moved)
                np.testing.assert_allclose(prediction,restored.predict(X,{'kind':'value'}))
                self.assertEqual(restored.source_specs['base_1']['target_transform']['name'],'log1p')
                if method in {'weighted','stack'}:
                    self.assertEqual(len(model.fold_evidence),6)
                    for fold in model.fold_evidence:
                        self.assertFalse(set(fold['train_indices']) & set(fold['validation_indices']))
                if method=='weighted':
                    self.assertAlmostEqual(model.weights.sum(),1.)
                    self.assertTrue((model.weights>=0).all())

    def test_classification_multilabel_and_multioutput_probabilities(self):
        for kind in ['binary_classification','multiclass_classification','multilabel_classification','multioutput_regression']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                code,context,plan,X,y,c=composite_case(Path(tmp),kind,'stack',True)
                if context.classes:
                    c.classes=list(reversed(context.classes))
                config=dict(plan.model_dump(mode='json'),_code_hash=tree_hash(code))
                model=load_module(code).create_candidate(config);model.fit(X,y,c)
                probabilities=model.predict(X,{'kind':'probability' if 'classification' in kind else 'value'})
                self.assertTrue(np.isfinite(probabilities).all())
                if context.classes:
                    np.testing.assert_allclose(probabilities.sum(axis=1),1.)
                    labels=model.predict(X,{'kind':'label'})
                    self.assertTrue(set(labels)<=set(context.classes))
                if kind=='multilabel_classification':self.assertEqual(probabilities.shape,np.asarray(y).shape)

    def test_group_oof_keeps_duplicate_groups_separate_and_rejects_insufficient_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            code,_,plan,X,y,c=composite_case(Path(tmp),method='stack')
            c.groups=np.repeat(np.arange(15),4)
            model=load_module(code).create_candidate(plan.model_dump())
            model.fit(X,y,c)
            for fold in model.fold_evidence:
                self.assertFalse(set(c.groups[fold['train_indices']]) & set(c.groups[fold['validation_indices']]))
            c.groups=np.zeros(len(X))
            with self.assertRaisesRegex(ValueError,'Insufficient independent groups'):
                model.fit(X,y,c)

    def test_component_sources_are_immutable_and_outer_policies_not_applied_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            code,_,plan,X,y,c=composite_case(Path(tmp),different_policies=True)
            source=code/'components'/'base_0'/'candidate.py'
            source.write_text(source.read_text()+'\n# modified\n')
            model=load_module(code).create_candidate(plan.model_dump())
            with self.assertRaisesRegex(ValueError,'hash mismatch'):model.fit(X,y,c)

    def test_generated_composition_assignment_is_left_to_agents(self):
        with tempfile.TemporaryDirectory() as tmp:
            code,_,plan,_,_,_=composite_case(Path(tmp),different_policies=True)
            changed=COMPOSITION_CODE.replace("return FrozenEnsemble(config,", "config['ensemble_sources'][1]['config']['target_transform']['name']='identity'\n    return FrozenEnsemble(config,")
            (code/'candidate.py').write_text(changed,encoding='utf-8')
            model=load_module(code).create_candidate(plan.model_dump())
            self.assertEqual(model.config['ensemble_sources'][1]['config']['target_transform']['name'], 'identity')

    def test_fixed_full_data_refit_relearns_weights_without_recipe_search(self):
        from mlmom.worker import train
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            code,context,plan,_,_,_=composite_case(root,method='weighted')
            result=train(dict(code_dir=str(code),code_hash=tree_hash(code),context=context.model_dump(mode='json'),
                plan=plan.model_dump(mode='json'),train_path=context.files.train,seed=42,
                fixed=True,phase='final',timeouts_enabled=False,worker_seconds=60,worker_cpus=1,
                worker_memory_gb=2),root/'final')
            self.assertTrue(result['full_refit'])
            self.assertEqual(result['stop_reason'],'fixed_configuration')
            self.assertEqual(result['trials'],[])
            self.assertIsNone(result['internal_score'])
            model=load_module(code).load_candidate(result['model_artifact'])
            self.assertEqual(model.method,'weighted')
            self.assertLessEqual(model.weight_fit['trials'],3)
            self.assertEqual(len(model.models),2)


class GroupPipelineTests(unittest.TestCase):
    def test_ensemble_budget_skip_is_explicit_and_cached_without_calling_llm(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from mlmom.ensemble_planning import plan_ensemble
        with tempfile.TemporaryDirectory() as tmp:
            store=ExperimentStore(Path(tmp))
            try:
                store.put('cycle','r1',dict(round_id='r1',completed=True))
                for index in range(2):
                    store.put('component',str(index),dict(experiment_id=str(index),round_id='r1',
                        candidate={'code_hash':str(index)},configuration_hash=str(index),plan={'tool_id':'sklearn'}))
                from mlmom.registry import installed_tools
                pipeline=SimpleNamespace(store=store,config=RunConfig(timeouts_enabled=True),client=Mock(),
                    tools=installed_tools(),
                    context=SimpleNamespace(task=SimpleNamespace(task_type='regression')),
                    budget=SimpleNamespace(snapshot=lambda:SimpleNamespace(timeouts_enabled=True,research_seconds=1)),
                    admission_cost=lambda:100)
                result=plan_ensemble(pipeline,{'round_number':2})
                self.assertEqual(result['reason'],'insufficient_complete_ensemble_budget')
                self.assertEqual(plan_ensemble(pipeline,{'round_number':2}),result)
                pipeline.client.generate.assert_not_called()
            finally:store.close()

    def test_ordinary_planning_failure_does_not_remove_ensemble_slot(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from mlmom.pipeline import Pipeline
        pipeline=object.__new__(Pipeline)
        pipeline.config=RunConfig()
        pipeline.last_submission={'path':'saved.csv'}
        pipeline.budget=SimpleNamespace(snapshot=lambda:SimpleNamespace(research_seconds=10000))
        pipeline.store=SimpleNamespace(event=Mock())
        pipeline.planning_call=Mock(return_value={'valid':False,'plans':[],'errors':['ordinary plan rejected']})
        expected=[{'marker':'independent ensemble group'}]
        with patch('mlmom.ensemble_planning.plan_ensemble',return_value={'plans':expected}):
            result=pipeline.plan({'round_number':2})
        self.assertEqual(result['plans'],expected)
        self.assertIsNone(result['stop_reason'])

    def test_family_then_ensemble_queue_internal_selection_and_resume(self):
        from mlmom.pipeline import Pipeline
        from mlmom.registry import installed_tools
        from tests.test_pipeline import dataset
        class FamilyClient(OfflineLLM):
            def complete(self,messages):
                task=json.loads(super().complete(messages));task['validation_folds']=2
                return json.dumps(task)
            def generate(self,role,instruction,payload,schema,tools=None):
                if role=='hypothesis_generator':
                    return schema(hypotheses=[dict(hypothesis_id=f"idea{payload['round_number']}",
                        description='Compare tree implementations',rationale='Synthetic comparison',
                        baseline=payload['round_number']==1,kind='family',model_family='tree_implementations',
                        family_tools=['sklearn','lightgbm'])])
                if role=='ensemble_agent':
                    self.calls.append((role,payload))
                    chosen={}
                    for component in payload['components']:
                        chosen.setdefault(component['tool_id'],component['experiment_id'])
                    return schema(component_experiment_ids=list(chosen.values())[:2],rationale='Different tree implementations')
                if role=='planner':
                    result=super().generate(role,instruction,payload,schema,tools)
                    for plan in result.plans:
                        tool=payload.get('required_tool_id') or self.tool_id
                        plan.tool_id=tool;plan.skill_ids=[tool]
                        plan.model_params=dict(n_estimators=3,max_depth=3)
                        plan.search_space={'n_estimators':{'type':'int','low':4,'high':6}}
                        if tool=='lightgbm':plan.model_params.update(min_child_samples=2,verbosity=-1)
                        plan.time_limit_seconds=90
                    return result
                if role=='coding_agent':
                    self.calls.append((role,payload))
                    if payload['plan'].get('ensemble_sources'):
                        return schema(files={'candidate.py':COMPOSITION_CODE})
                    source=MODEL_CODE
                    if payload['plan']['tool_id']=='lightgbm':
                        source=source.replace('from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, IsolationForest',
                            'from lightgbm import LGBMClassifier as ExtraTreesClassifier, LGBMRegressor as ExtraTreesRegressor\nfrom sklearn.ensemble import IsolationForest')
                    return schema(files={'candidate.py':source})
                return super().generate(role,instruction,payload,schema,tools)
        def tools():
            return {k:v for k,v in installed_tools().items() if k in {'sklearn','lightgbm'}}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);task=dataset(root/'input');client=FamilyClient(params={'n_estimators':3})
            config=RunConfig(timeouts_enabled=False,initial_boosting_sweep=False,max_rounds=2,hpo_enabled=True,hpo_max_trials=2,hpo_batch_size=2,
                diagnostic_seconds=0,max_workers=2,reserve_cpus=0,reserve_memory_gb=0)
            with patch('mlmom.pipeline.installed_tools',tools):
                pipeline=Pipeline(task,config,run_dir=root/'run',client=client)
                output=pipeline.run()
            self.assertEqual(output['status'],'completed',output)
            store=ExperimentStore(root/'run')
            try:
                first=store.get('round_result','r1');second=store.get('round_result','r2')
                self.assertIsNotNone(first)
                self.assertIsNotNone(second,output)
                first_group=next(g for g in first['search_groups'] if g['kind']=='family')
                self.assertEqual({r['tool_id'] for r in first_group['variants']},{'sklearn','lightgbm'})
                self.assertEqual(len({r['split_id'] for r in first_group['variants']}),1)
                hpo=[r for r in store.list('experiment') if r['round_id']=='r1' and r['experiment_id'].endswith('-hpo')]
                self.assertEqual(len(hpo),1)
                initial=[r for r in first_group['variants'] if not r['experiment_id'].endswith('-hpo')]
                selected=min(initial,key=lambda r:r['internal_score'])
                self.assertEqual(hpo[0]['candidate_id'],selected['candidate_id'])
                self.assertEqual(len([role for role,_ in client.calls if role=='ensemble_agent']),1)
                proposed=store.get('ensemble_proposal','r2')['component_experiment_ids']
                self.assertTrue(all(store.get('experiment',key)['round_id']=='r1' for key in proposed))
                ensemble=next(g for g in second['search_groups'] if g['kind']=='ensemble')
                self.assertEqual(ensemble['status'],'completed',ensemble)
                self.assertEqual({r['method'] for r in ensemble['variants']},{'mean','median','weighted','stack'},ensemble)
                evaluations={r['experiment_id'] for r in store.list('evaluation')}
                self.assertEqual(evaluations & {r['experiment_id'] for r in ensemble['variants']},
                                 {ensemble['winner_experiment_id']})
                self.assertGreaterEqual(len(component_pool(store,3)),2)
                self.assertEqual(store.get('ensemble_planning','r1')['plans'],[])
                count=len(store.list('experiment'))
            finally:store.close()
            with patch('mlmom.pipeline.installed_tools',tools):
                resumed=Pipeline(task,config,run_dir=root/'run',client=OfflineLLM(fail=True),resume=True).run(resume=True)
            self.assertEqual(resumed['submission'],output['submission'])
            store=ExperimentStore(root/'run')
            try:self.assertEqual(len(store.list('experiment')),count)
            finally:store.close()


if __name__=='__main__':unittest.main()
