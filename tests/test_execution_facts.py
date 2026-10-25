"""Observed fit facts survive failures and preserve the actual diamonds composition."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mlmom.execution_facts import execution_context, record_fit, read_fit_facts
from mlmom.pipeline_contracts import Champion, HypothesisPlan, PlanBatch
from mlmom.planning_context import champion_snapshot
from mlmom.planning_inheritance import inherit_champion
from mlmom.research import review_result
from mlmom.review_context import review_payload
from mlmom.sdk import FitContext
from mlmom.worker import train, failure_result
from scripts.replay_diamonds_context import hydrate, seed_pipeline
from tests.fakes import OfflineLLM
from tests import test_feature_engineering as feature_fixtures
from tests.test_feature_programs import RATIO_PROGRAM

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'tests/fixtures/diamonds_r1_boundary.json'
PROGRAMS=['diamond_domain_features','carat_spline_quality_interactions','rbf_kernel_diamond','gbm_oof_representation']


class FitFactsTests(unittest.TestCase):
    def test_worker_folds_refit_resume_and_known_empty_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            _,_,_,_,protocol,request=feature_fixtures.FeatureWorkerTests().request(root)
            work=root/'run';work.mkdir()
            result=train(request,work)
            facts=result['execution_facts']
            self.assertEqual(len(facts),len(protocol['folds'])+1)
            self.assertEqual([f['fold_number'] for f in facts[:-1]],list(range(len(protocol['folds']))))
            self.assertEqual(facts[-1]['stage'],'full_refit')
            self.assertTrue(all(f['feature_programs']==[] and f['derived_columns']==0 for f in facts))
            self.assertTrue(all(f['split_id']=='r1' for f in facts))
            self.assertTrue(all(f['trial_number']==0 for f in facts[:-1]))
            resumed=train(request,work)
            self.assertEqual(len(resumed['execution_facts']),len(facts)+1)
            self.assertEqual(len({f['fit_id'] for f in resumed['execution_facts']}),len(facts)+1)

    def test_program_config_column_counts_and_submission_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            _,_,_,_,_,request=feature_fixtures.FeatureWorkerTests().request(root)
            request['plan'].update(feature_set_id='ratio',feature_set=dict(feature_set_id='ratio',
                programs=[dict(name='ratio',source=RATIO_PROGRAM,config={'intent':'ratio'})]))
            request.update(fixed=True,phase='submission')
            request.pop('outer_split_id')
            work=root/'submission';work.mkdir()
            result=train(request,work)
            fact=result['execution_facts'][0]
            self.assertEqual((fact['stage'],fact['split_id']),('submission_refit','all-train'))
            self.assertEqual(fact['total_columns'],fact['base_columns']+fact['derived_columns'])
            self.assertEqual(fact['derived_columns'],1)
            self.assertEqual(fact['feature_programs'][0]['config'],{'intent':'ratio'})
            self.assertEqual(fact['feature_programs'][0]['columns'],1)
            self.assertEqual(fact['feature_execution_owner'],'SDK')
            self.assertIsNone(result['internal_score'])

    def test_prediction_failure_keeps_completed_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            _,_,_,_,_,request=feature_fixtures.FeatureWorkerTests().request(root)
            work=root/'failed';work.mkdir()
            with patch('mlmom.sdk.PreparedCandidate.predict',side_effect=ValueError('prediction failed')):
                try:train(request,work)
                except ValueError as error:failure=failure_result(error,work)
                else:self.fail('Expected prediction failure')
            self.assertEqual(len(failure['execution_facts']),1)
            self.assertEqual(failure['execution_facts'][0]['stage'],'inner_fold')
            self.assertEqual(read_fit_facts(work),failure['execution_facts'])

    def test_missing_or_broken_optional_manifest_is_unknown(self):
        class Broken:
            def feature_manifest(self):raise RuntimeError('optional manifest unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            for model in (object(),Broken()):
                c=FitContext('regression','rmse','minimize',42,1,2,60,execution_facts_dir=tmp)
                fact=record_fit(model,{'model_params':{'alpha':2}},c)
                self.assertEqual(fact['status'],'unknown')
                self.assertIsNone(fact['feature_programs'])
                self.assertIsNone(fact['derived_columns'])
                self.assertEqual(fact['parameters'],{'alpha':2})

    def test_sdk_ensemble_records_each_component_with_parent_identity(self):
        from tests.test_search_groups import composite_case
        from mlmom.sdk import load_module
        from mlmom.feature_contracts import FeatureSet, FeatureProgram
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            code,_,plan,X,y,context=composite_case(root)
            source=RATIO_PROGRAM.replace("['x0']","['x']").replace("['x1']","['z']")
            definition=FeatureSet(feature_set_id='component-program',programs=[FeatureProgram(name='ratio',source=source)])
            plan.ensemble_sources[0]['config'].update(feature_set_id=definition.feature_set_id,feature_set=definition.model_dump(mode='json'))
            context.execution_facts_dir=str(root/'fit-execution')
            context.outer_split_id='outer-r1'
            model=load_module(code).create_candidate(plan.model_dump(mode='json'))
            model.fit(X,y,context)
            facts=read_fit_facts(root)
            self.assertEqual(len(facts),2)
            self.assertEqual({f['component'] for f in facts},{'base_0','base_1'})
            self.assertTrue(all(f['parent_fit_id']==context.fit_id for f in facts))
            self.assertEqual(facts[0]['feature_programs'][0]['name'],'ratio')
            self.assertEqual(facts[1]['feature_programs'],[])


class DiamondsBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.fixture=json.loads(FIXTURE.read_text(encoding='utf-8'))
        self.data=hydrate(self.fixture)
        self.champion=Champion.model_validate(self.data['champion'])
        self.snapshot=champion_snapshot(self.champion)

    def proposed(self,**changes):
        values=dict(hypothesis=dict(hypothesis_id='interactions',description='Add carat grade interactions',
            rationale='Test interactions against the saved champion',based_on=self.champion.candidate.candidate_id),
            tool_id='sklearn',skill_ids=['sklearn'],feature_set_id=None)
        values.update(changes)
        # Exercise the actual PlannerPlan -> HypothesisPlan boundary, including unset fields.
        original=PlanBatch.model_validate({'plans':[values]}).plans[0]
        supplied=original.model_dump(mode='json',exclude_unset=True)
        return inherit_champion(HypothesisPlan.model_validate(supplied),self.snapshot,supplied)

    def test_null_and_empty_inherit_exact_champion(self):
        plan=self.proposed(model_params={},preprocessing=[],target_transform=None)
        self.assertEqual(plan.model_params,self.champion.run.best_params)
        self.assertEqual(plan.preprocessing,self.champion.plan.preprocessing)
        self.assertEqual([p.name for p in plan.feature_set.programs],PROGRAMS)
        self.assertEqual(plan.prediction_policy,self.champion.plan.prediction_policy)

    def test_interactions_preserve_programs_and_record_the_actual_change(self):
        plan=self.proposed(preprocessing=[*self.champion.plan.preprocessing,'Append carat x grade interactions'])
        self.assertEqual([p.name for p in plan.feature_set.programs],PROGRAMS)
        self.assertEqual(plan.configuration_diff['features'],dict(added=[],removed=[],replaced=[]))
        self.assertEqual(plan.intervention.changed_fields,['preprocessing'])
        self.assertEqual(plan.configuration_diff['preprocessing']['before'],self.champion.plan.preprocessing)
        from mlmom.coding import reuse_assignment
        previous=dict(plan=self.snapshot['plan'],files=self.data['sources'],dependencies_match=True,
            preprocessing_spec=dict(description='Saved base preprocessing',output='dense'))
        reuse_pre,reuse_model=reuse_assignment(previous,plan)
        self.assertFalse(reuse_pre)
        self.assertTrue(reuse_model)

    def test_explicit_removals_resets_replacement_and_independent_model(self):
        one=self.proposed(remove_feature_names=['gbm_oof_representation'])
        self.assertEqual([p.name for p in one.feature_set.programs],PROGRAMS[:-1])
        self.assertEqual(one.configuration_diff['features']['removed'],['gbm_oof_representation'])
        all_removed=self.proposed(reset_fields=['feature_set','preprocessing','model_params'],model_params={'alpha':3})
        self.assertIsNone(all_removed.feature_set)
        self.assertEqual(all_removed.preprocessing,[])
        self.assertEqual(all_removed.model_params,{'alpha':3})
        replaced=self.proposed(feature_set_id='replacement',feature_set=dict(feature_set_id='replacement',
            features=[dict(name='interaction',op='product',columns=['carat','depth'])]))
        self.assertEqual(replaced.configuration_diff['features']['removed'],sorted(PROGRAMS))
        self.assertEqual(replaced.configuration_diff['features']['added'],['interaction'])
        independent=self.proposed(hypothesis=dict(hypothesis_id='new',description='Independent model',rationale='New model'))
        self.assertIsNone(independent.feature_set)
        self.assertEqual(independent.configuration_diff,{})
        other_backend=self.proposed(tool_id='catboost',model_params={'iterations':20})
        self.assertEqual(other_backend.model_params,{'iterations':20})
        self.assertEqual([p.name for p in other_backend.feature_set.programs],PROGRAMS)

    def test_payload_context_cache_and_all_advertised_evidence(self):
        client=OfflineLLM()
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as cleanup:
            pipeline,state=seed_pipeline(self.fixture,Path(tmp),client)
            cleanup.callback(pipeline.store.close)
            payload=pipeline.planning_payload(state)
            context_call=next(p for role,p in client.calls if role=='context_agent')
            self.assertEqual(context_call['champion']['execution_context'],payload['execution_context'])
            self.assertEqual([p['name'] for p in payload['execution_context']['current_configuration']['feature_set']['programs']],PROGRAMS)
            observed=payload['execution_context']['observed_execution']
            self.assertEqual(observed['status'],'unknown')
            self.assertEqual(observed['fits'],[])
            saved=observed['related_fits'][0]
            self.assertEqual((saved['base_columns'],saved['derived_columns'],saved['total_columns']),(28,282,310))
            self.assertEqual((saved['stage'],saved['split_id']),('submission_refit','all-train'))
            for ref in payload['evidence_refs']:
                first=pipeline.read_planning_evidence(ref,chars=37)
                second=pipeline.read_planning_evidence(ref,offset=37,chars=37)
                self.assertEqual(first['reference'],ref)
                self.assertEqual(second['offset'],37)
            first_hash=payload['planning_brief']['input_hash']
            changed=copy.deepcopy(state)
            changed['champion']['run']['best_params']['alpha']=7
            changed['champion']['run']['model_spec']['backend_params']['alpha']=7
            changed_payload=pipeline.planning_payload(changed)
            self.assertNotEqual(first_hash,changed_payload['planning_brief']['input_hash'])

    def test_review_refs_open_and_reviewer_cannot_overwrite_facts(self):
        from mlmom.pipeline_contracts import ResultReview
        self.assertNotIn('execution_context',ResultReview.model_json_schema()['properties'])
        parent=self
        class Reviewer:
            def generate(self,role,instruction,payload,schema,tools=None):
                parent.assertIn('candidate:'+parent.champion.candidate.candidate_id,payload['evidence_refs'])
                parent.assertIn('source:feature_programs/gbm_oof_representation.py',payload['evidence_refs'])
                for ref in payload['evidence_refs']:
                    read=tools.invoke('read_evidence',dict(reference=ref,offset=0,chars=41))
                    parent.assertEqual(read.status,'ok',read.error)
                    parent.assertEqual(tools.invoke('read_evidence',dict(reference=ref,offset=41,chars=41)).status,'ok')
                return schema(implementation='matches',effect='supported',explanation='GBM was not used',
                    execution_context={'current_configuration':{'feature_set':None}})
        payload=review_payload(self.data['champion'],self.data['rows'],sources=self.data['sources'])
        self.assertIn('Historical baseline rationale',payload['hypothesis_prose_role'])
        self.assertEqual(payload['trial_history']['total'],len(self.data['champion']['run']['trials']))
        before=copy.deepcopy(payload['execution_context'])
        review=review_result(Reviewer(),payload,[],'review',5)
        self.assertEqual(review.explanation,'GBM was not used')
        self.assertEqual(review.execution_context,before)
        self.assertEqual([p['name'] for p in review.execution_context['current_configuration']['feature_set']['programs']],PROGRAMS)

    def test_real_planning_and_ensemble_role_boundaries_preserve_programs(self):
        from mlmom.planning import build_planning_graph
        from mlmom.ensemble_planning import plan_ensemble
        parent=self
        class Client(OfflineLLM):
            def generate(self,role,instruction,payload,schema,tools=None):
                if role=='hypothesis_generator':
                    self.calls.append((role,payload))
                    return schema(hypotheses=[dict(hypothesis_id='interactions',description='Add carat grade interactions',
                        rationale='Test interactions relative to the current representation',based_on=parent.champion.candidate.candidate_id)])
                if role=='planner':
                    self.calls.append((role,payload))
                    return schema(plans=[dict(hypothesis=payload['hypotheses']['hypotheses'][0],tool_id='sklearn',
                        skill_ids=['sklearn'],model_params={},feature_set_id=None,cpus=1,memory_gb=2,
                        preprocessing=[*parent.champion.plan.preprocessing,'Append carat x grade interactions'])])
                return super().generate(role,instruction,payload,schema,tools)
        client=Client()
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as cleanup:
            pipeline,state=seed_pipeline(self.fixture,Path(tmp),client)
            cleanup.callback(pipeline.store.close)
            payload=pipeline.planning_payload(state)
            graph=build_planning_graph(client,pipeline.context,pipeline.tools,pipeline.skills,pipeline.budget.snapshot(),
                max_candidates=1,known_candidates={self.champion.candidate.candidate_id},lightweight=False)
            output=graph.invoke(dict(payload=payload),{'recursion_limit':100})
            self.assertTrue(output['valid'],output)
            self.assertEqual([p['name'] for p in output['plans'][0]['feature_set']['programs']],PROGRAMS)
            plan_ensemble(pipeline,state)
            for role in ('context_agent','hypothesis_generator','planner','plan_validator','ensemble_agent'):
                call=next(p for r,p in client.calls if r==role)
                common=call['champion']['execution_context']
                self.assertEqual([p['name'] for p in common['current_configuration']['feature_set']['programs']],PROGRAMS,role)
                self.assertEqual(common['observed_execution']['related_fits'][0]['split_id'],'all-train',role)
            reviewed=next(p for r,p in client.calls if r=='plan_validator')
            self.assertEqual([p['name'] for p in reviewed['execution_contract']['effective_feature_set']['programs']],PROGRAMS)
            # A committed effective removal can resume without reapplying it or requiring registration again.
            removed=self.proposed(remove_feature_names=['gbm_oof_representation'])
            saved=removed.model_dump(mode='json')
            resumed=build_planning_graph(client,pipeline.context,pipeline.tools,pipeline.skills,pipeline.budget.snapshot(),
                max_candidates=1,lightweight=False,saved_progress=dict(proposed={'interactions':saved},accepted={}))
            result=resumed.invoke(dict(payload=payload,hypotheses={'hypotheses':[saved['hypothesis']]}),{'recursion_limit':100})
            self.assertTrue(result['valid'],result)
            self.assertEqual([p['name'] for p in result['plans'][0]['feature_set']['programs']],PROGRAMS[:-1])

    def test_hpo_delta_is_relative_to_its_immediate_parent(self):
        from mlmom.model_spec import hpo_child
        inherited=self.proposed(model_params={'alpha':7})
        child=hpo_child(inherited,'immediate-parent')
        self.assertEqual(child.configuration_diff['base_parameters']['alpha'],7)
        self.assertEqual(child.configuration_diff['base_experiment_id'],'immediate-parent')
        self.assertEqual(child.configuration_diff['features'],dict(added=[],removed=[],replaced=[]))

    def test_hpo_without_parent_does_not_use_baseline_as_tuning_evidence(self):
        row=copy.deepcopy(self.data['champion'])
        row['run']['intervention']=dict(kind='hpo',parent_experiment_id='missing-parent')
        payload=review_payload(row,self.data['rows'])
        self.assertIsNone(payload['comparable_reference'])
        self.assertIsNone(payload['relative_gain'])
        self.assertIsNotNone(payload['references']['baseline'])
        parent=copy.deepcopy(row)
        parent['run']['experiment_id']='parent'
        parent['evaluation']=dict(parent['evaluation'],experiment_id='parent',split_id='different')
        row['run']['intervention']['parent_experiment_id']='parent'
        payload=review_payload(row,[*self.data['rows'],parent])
        self.assertIsNone(payload['relative_gain'])

    def test_saved_alpha_trial_uses_feature_parent_instead_of_plain_baseline(self):
        row=next(row for row in self.data['rows'] if row['run']['experiment_id']==self.champion.run.experiment_id+'-hpo')
        payload=review_payload(row,self.data['rows'])
        self.assertEqual(payload['references']['parent_control']['experiment_id'],self.champion.run.experiment_id)
        self.assertEqual(payload['comparable_reference'],payload['references']['parent_control'])
        self.assertNotEqual(payload['comparable_reference'],payload['references']['baseline'])
        self.assertIn('Historical baseline rationale',payload['hypothesis_prose_role'])


if __name__=='__main__':unittest.main()
