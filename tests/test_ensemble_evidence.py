"""Measured diversity is not inferred from library names or missing predictions."""
import copy
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from mlmom.ensemble_evidence import diversity_evidence, oof_snapshot
from mlmom.ensemble_planning import plan_ensemble
from mlmom.persistence import atomic_json
from mlmom.splitting import inner_protocol
from tests import test_controller_groups as controller_cases
from tests import test_execution as execution_cases
from tests import test_inner_validation as inner_cases


class DiversityEvidenceTests(unittest.TestCase):
    def context(self, root, task_type='regression'):
        frame, context, *_ = execution_cases.execution_case(root, task_type)
        return frame, context

    def row(self, root, context, frame, key, predictions, *, weights=None, protocol=None, order=None):
        root.mkdir(exist_ok=True)
        if protocol is None:
            protocol = inner_protocol(frame, context, outer_split_id='outer',
                source_indices=list(range(1000, 1000+len(frame))))
        indices = sorted(i for f in protocol['folds'] for i in f['validation_indices'])
        if order is not None:
            indices = np.asarray(indices)[order].tolist()
        from mlmom.inputs import targets
        from mlmom.evaluation import prediction_kind
        snapshot = oof_snapshot(protocol, context, indices, np.asarray(predictions)[indices],
            prediction_kind(context.task.primary_metric, context.task.task_type), targets(frame, context), weights)
        atomic_json(root/'oof.json', snapshot)
        atomic_json(root/'protocol.json', protocol)
        return dict(experiment_id=key, run=dict(split_id='outer', oof_ref=str(root/'oof.json'),
            validation_protocol_ref=str(root/'protocol.json'), duration_seconds=100000,
            resource_cpus=2, resource_memory_gb=3, execution_facts=[]))

    def test_opposite_residuals_aligned_by_source_row_and_exact_weighted_gain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame, context = self.context(root)
            context.metric_options = {'multioutput':'uniform_average'}
            y = frame[context.task.target_columns[0]].to_numpy()
            error = np.linspace(-3, 5, len(frame))
            w = np.arange(1, len(frame)+1)
            a = self.row(root/'a', context, frame, 'a', y+error, weights=w)
            b = self.row(root/'b', context, frame, 'b', y-error, weights=w, order=np.arange(len(frame))[::-1])
            # Measurements must work without a raw train/holdout path or pipeline frame.
            Path(context.files.train).unlink()
            evidence = diversity_evidence([a, b], context)
            self.assertEqual(evidence['status'], 'historical_gain_observed')
            pair = evidence['pairs'][0]
            self.assertAlmostEqual(pair['error_correlation']['per_output'][0]['value'], -1)
            self.assertAlmostEqual(pair['loss_correlation']['value'], 1)
            self.assertEqual(pair['support_rows'], len(y))
            self.assertAlmostEqual(pair['error_overlap']['weighted_jaccard'], 1)
            gain = evidence['incremental_benefit']
            protocol = json.loads(Path(a['run']['validation_protocol_ref']).read_text())
            fold_rmse = [np.sqrt(np.average(error[f['validation_indices']]**2,
                weights=w[f['validation_indices']])) for f in protocol['folds']]
            self.assertAlmostEqual(gain['components']['a']['mean_fold_score'], np.mean(fold_rmse))
            self.assertAlmostEqual(gain['components']['a']['pooled_oof_score'], np.sqrt(np.average(error**2, weights=w)))
            self.assertAlmostEqual(gain['improvement_over_best'], np.mean(fold_rmse))
            self.assertAlmostEqual(gain['blend']['mean_fold_score'], 0, places=12)
            self.assertEqual(evidence['full_refit_cost']['status'], 'unknown')
            self.assertIsNone(evidence['full_refit_cost']['historical_component_fit_sum_seconds'])

    def test_multiclass_probability_error_overlap_and_log_loss_not_label_correlation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame, context = self.context(root, 'multiclass_classification')
            context.classes = ['amber', 'blue', 'cyan']
            y = np.asarray(context.classes)[np.arange(len(frame)) % 3]
            frame[context.task.target_columns[0]] = y
            context.task.primary_metric = 'log_loss'
            context.task.metric_direction = 'minimize'
            context.metric_options = {}
            p = np.full((len(y), 3), .1)
            p[np.arange(len(y)), np.arange(len(y)) % 3] = .8
            q = p.copy()
            p[::4] = np.roll(p[::4], 1, axis=1)
            q[1::4] = np.roll(q[1::4], 1, axis=1)
            w = 1+np.arange(len(y)) % 5
            a = self.row(root/'a', context, frame, 'a', p, weights=w)
            b = self.row(root/'b', context, frame, 'b', q, weights=w)
            evidence = diversity_evidence([a, b], context)
            pair = evidence['pairs'][0]
            self.assertEqual(pair['error_overlap']['intersection_rows'], 0)
            self.assertEqual(pair['error_overlap']['union_rows'], 30)
            self.assertAlmostEqual(pair['error_overlap']['weighted_jaccard'], 0)
            self.assertEqual(pair['class_order'], context.classes)
            score = log_loss(y, (p+q)/2, labels=context.classes, sample_weight=w)
            self.assertAlmostEqual(evidence['incremental_benefit']['blend']['pooled_oof_score'], score)
            self.assertNotIn('targets', json.dumps(evidence))
            snapshot = json.loads(Path(b['run']['oof_ref']).read_text())
            snapshot['classes'] = context.classes[::-1]
            atomic_json(Path(b['run']['oof_ref']), snapshot)
            self.assertEqual(diversity_evidence([a,b], context)['status'], 'unverified_hypothesis')

    def test_labels_allow_error_diagnostics_but_not_a_probability_blend_and_constant_is_undefined(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame, context = self.context(root, 'binary_classification')
            context.classes = ['no', 'yes']
            context.task.primary_metric = 'mcc'
            frame[context.task.target_columns[0]] = np.where(np.arange(len(frame)) % 2, 'yes', 'no')
            y = frame[context.task.target_columns[0]].to_numpy()
            a = self.row(root/'a', context, frame, 'a', y)
            b = self.row(root/'b', context, frame, 'b', y)
            report = diversity_evidence([a,b], context)
            self.assertEqual(report['status'], 'measured_diversity')
            self.assertIsNone(report['pairs'][0]['error_correlation']['per_output'][0]['value'])
            self.assertIsNone(report['pairs'][0]['error_overlap']['weighted_jaccard'])
            self.assertEqual(report['incremental_benefit']['status'], 'unavailable')

    def test_binary_probability_shapes_maximize_direction_and_metric_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame, context = self.context(root, 'binary_classification')
            context.classes = ['no', 'yes']
            truth = np.arange(len(frame)) % 2
            frame[context.task.target_columns[0]] = np.asarray(context.classes)[truth]
            context.task.primary_metric = 'roc_auc'
            context.task.metric_direction = 'maximize'
            context.metric_options = {'max_fpr':.8}
            p = np.clip(.25+.5*truth+.2*np.sin(np.arange(len(frame))), .01,.99)
            q = np.clip(.4+.2*truth+.3*np.cos(np.arange(len(frame))), .01,.99)
            w = 1+np.arange(len(frame)) % 4
            a = self.row(root/'a', context, frame, 'a', p, weights=w)
            b = self.row(root/'b', context, frame, 'b', np.column_stack([1-q,q]), weights=w)
            gain = diversity_evidence([a,b], context)['incremental_benefit']
            self.assertEqual(gain['status'], 'measured')
            self.assertAlmostEqual(gain['blend']['pooled_oof_score'],
                roc_auc_score(truth,(p+q)/2,sample_weight=w,max_fpr=.8))
            best = max(s['mean_fold_score'] for s in gain['components'].values())
            self.assertAlmostEqual(gain['improvement_over_best'],gain['blend']['mean_fold_score']-best)

    def test_multioutput_errors_and_original_scale_weighted_primary_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame, context = self.context(root, 'multioutput_regression')
            y = frame[context.task.target_columns].to_numpy()
            context.task.primary_metric = 'mae'
            context.metric_options = {'multioutput':[.2,.8]}
            error = np.column_stack([np.linspace(-4,2,len(y)),np.linspace(-1,8,len(y))])
            w = 1+np.arange(len(y)) % 7
            a = self.row(root/'a',context,frame,'a',y+error,weights=w)
            b = self.row(root/'b',context,frame,'b',y-error,weights=w)
            result = diversity_evidence([a,b],context)
            self.assertEqual(len(result['pairs'][0]['error_correlation']['per_output']),2)
            self.assertTrue(all(abs(r['value']+1)<1e-12 for r in result['pairs'][0]['error_correlation']['per_output']))
            oracle = np.average(np.average(abs(error),weights=w,axis=0),weights=[.2,.8])
            self.assertAlmostEqual(result['incremental_benefit']['components']['a']['pooled_oof_score'],oracle)
            self.assertAlmostEqual(result['incremental_benefit']['blend']['pooled_oof_score'],0,places=12)

    def test_incompatible_protocol_corrupt_mapping_and_legacy_do_not_invent_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame, context = self.context(root)
            p = frame[context.task.target_columns[0]].to_numpy()
            a = self.row(root/'a', context, frame, 'a', p)
            protocol = inner_protocol(frame, context, seed=108, outer_split_id='outer',
                source_indices=list(range(1000, 1000+len(frame))))
            b = self.row(root/'b', context, frame, 'b', p, protocol=protocol)
            result = diversity_evidence([a,b], context)
            self.assertIn('Different validation protocols', result['pairs'][0]['reason'])
            b = self.row(root/'b', context, frame, 'b', p)
            artifact = Path(b['run']['oof_ref'])
            data = json.loads(artifact.read_text())
            data['source_indices'][0] += 500
            atomic_json(artifact, data)
            self.assertIn('alignment', diversity_evidence([a,b], context)['pairs'][0]['reason'])
            atomic_json(artifact, dict(predictions=p.tolist(), indices=list(range(len(p)))))
            self.assertIn('Legacy OOF', diversity_evidence([a,b], context)['pairs'][0]['reason'])
            b['run']['oof_ref'] = str(root/'missing.json')
            self.assertEqual(diversity_evidence([a,b], context)['status'], 'unverified_hypothesis')

    def test_triple_does_not_inherit_pair_gain_and_refit_cost_is_advisory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame, context = self.context(root)
            y = frame[context.task.target_columns[0]].to_numpy()
            rows = [self.row(root/key, context, frame, key, y+offset) for key,offset in [('a',1),('b',-1),('c',10)]]
            for index,row in enumerate(rows):
                row['run']['execution_facts'] = [dict(stage='inner_fold',duration_seconds=10000,fit_id='fold'),
                    dict(stage='full_refit', duration_seconds=2+index,fit_id='refit',fit_rows=len(frame)),
                    dict(stage='full_refit',component='nested',duration_seconds=9000,fit_id='base')]
            result = diversity_evidence(rows[:2], context, methods=['mean','stack'], oof_folds=4)
            self.assertEqual(result['full_refit_cost']['historical_component_fit_sum_seconds'], 5)
            self.assertEqual(result['full_refit_cost']['recipes']['stack']['oof_component_fits'], 8)
            self.assertEqual(result['full_refit_cost']['status'], 'estimated')
            complete = diversity_evidence(rows,context)['incremental_benefit']
            self.assertAlmostEqual(complete['leave_one_component_out']['c']['improvement_from_adding_component'],
                -10/3,places=12)
            rows[2]['run'].pop('oof_ref')
            result = diversity_evidence(rows, context)
            self.assertEqual([p['status'] for p in result['pairs']], ['measured','unavailable','unavailable'])
            self.assertEqual(result['status'], 'unverified_hypothesis')
            self.assertEqual(result['incremental_benefit']['status'], 'unavailable')
            rows[2] = self.row(root/'c', context, frame, 'c', y+10)
            self.assertEqual(diversity_evidence([rows[0],rows[2]], context)['status'], 'historical_gain_not_observed')

    def test_real_role_tool_persistence_planner_payload_and_resume_without_new_gates(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as cleanup:
            pipeline, *_ = controller_cases.IndependentRecipeTests().make_pipeline(Path(tmp))
            cleanup.callback(pipeline.store.close)
            generate = pipeline.client.generate
            received = []
            def client(role, instruction, payload, schema, tools=None):
                if role == 'ensemble_agent':
                    self.assertEqual(payload['diversity_evidence']['status'], 'unverified_hypothesis')
                    self.assertEqual(tools.invoke('compare_oof_components', {'experiment_ids':['e0','unknown']}).status, 'error')
                    measured = tools.invoke('compare_oof_components', {'experiment_ids':['e0','e1']})
                    self.assertEqual(measured.status, 'ok')
                    self.assertEqual(measured.data['status'], 'unverified_hypothesis')
                    self.assertTrue(measured.data['reference'].startswith('ensemble_diversity:'))
                if role == 'planner':
                    received.append(copy.deepcopy(payload['diversity_evidence']))
                return generate(role,instruction,payload,schema,tools)
            pipeline.client.generate = client
            result = plan_ensemble(pipeline, {'round_number':2})
            self.assertEqual(len(result['plans']), 3, result)
            self.assertEqual(len(received), 3)
            evidence = result['diversity_evidence']
            self.assertEqual(evidence['status'], 'unverified_hypothesis')
            self.assertTrue(all(review['diversity_evidence']==evidence for review in pipeline.client.reviews))
            kind, key = evidence['reference'].split(':',1)
            self.assertEqual(pipeline.store.get(kind,key), evidence)
            calls = len(pipeline.client.calls)
            self.assertEqual(plan_ensemble(pipeline, {'round_number':2}), result)
            self.assertEqual(len(pipeline.client.calls),calls)

    def test_actual_worker_artifacts_support_the_diagnostic_without_raw_data_access(self):
        from mlmom.worker import train
        from mlmom.persistence import file_hash
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame, context, _, _, request = inner_cases.InnerExecutionTests().request(root)
            frame[context.task.target_columns[0]] = 0.
            frame['_weight'] = 1+np.arange(len(frame)) % 7
            context.task.weight_column = '_weight'
            frame.to_csv(request['train_path'],index=False)
            context.dataset_hash = file_hash(Path(request['train_path']))
            request['context'] = context.model_dump(mode='json')
            rows = []
            for key,offset in [('a',-20),('b',20)]:
                job = root/key
                job.mkdir()
                current = copy.deepcopy(request)
                current['plan']['model_params'] = {'constant':offset}
                result = train(current,job)
                rows.append(dict(experiment_id=key,run=dict(result,split_id=request['outer_split_id'],
                    resource_cpus=1,resource_memory_gb=2)))
            Path(request['train_path']).unlink()
            report = diversity_evidence(rows,context)
            self.assertEqual(report['status'],'historical_gain_observed',report)
            self.assertAlmostEqual(report['incremental_benefit']['improvement_over_best'],20)
            self.assertAlmostEqual(report['incremental_benefit']['blend']['mean_fold_score'],0)
            self.assertEqual(report['pairs'][0]['weighting'],'sample_weight')
            self.assertAlmostEqual(report['pairs'][0]['weight_sum'],frame['_weight'].sum())
            self.assertEqual(report['full_refit_cost']['status'],'estimated')
            self.assertTrue(all(c['samples'][0]['seconds']>=0 for c in report['full_refit_cost']['components'].values()))


if __name__ == '__main__':
    unittest.main()
