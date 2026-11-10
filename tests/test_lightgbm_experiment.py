"""Synthetic-only checks for the fixed two-model LightGBM comparison."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

from src import baseline_rf as b, weighted_rf as w, lightgbm_experiment as l
from src.evaluate import evaluate_predictions
from src.preprocess import FEATURE_COLUMNS, sha256


def synthetic_data():
    rows = []
    metadata = {}
    for rid, split, count, event in (('chb01_01','train',80,[100,150]),
                                      ('chb01_18','validation',40,[100,125]),
                                      ('chb01_26','test',40,[100,125]),
                                      ('chb01_23','supplementary',40,[])):
        metadata[rid+'.edf'] = {'duration_sec': count*5, 'seizures_sec': [event] if event else []}
        for i in range(count):
            label = int(bool(event) and event[0] <= i*5 < event[1])
            value = label * 10 + (i % 3) / 10
            rows.append({**{f: value if split in ('train','validation') else np.nan for f in FEATURE_COLUMNS},
                         'patient_id':'chb01','recording_id':rid,'source_file':rid+'.edf','split':split,
                         'start_sec':i*5,'end_sec':i*5+5,'decision_time_sec':i*5+5,'label':label,
                         'seizure_overlap_fraction':label,'leaky_numeric_metadata':label*1000})
    return pd.DataFrame(rows), metadata


def small_rf_models():
    return {'dummy': b.DummyClassifier(strategy='most_frequent'),
            'random_forest': RandomForestClassifier(n_estimators=5,random_state=42,n_jobs=1)}


class LightGBMUnitTests(unittest.TestCase):
    def setUp(self):
        frame,_=synthetic_data()
        self.train=frame[frame.split.eq('train')].copy()
        self.validation=frame[frame.split.eq('validation')].copy()

    def test_exact_configuration_and_training_only_weight(self):
        models=l.make_models(self.train)
        a,c=models['ordinary_lgbm'],models['weighted_lgbm']
        expected={'objective':'binary','n_estimators':150,'learning_rate':.05,'max_depth':3,
                  'num_leaves':7,'min_child_samples':10,'random_state':42,'n_jobs':4}
        for key,value in expected.items(): self.assertEqual(a.get_params()[key],value)
        self.assertEqual(a.scale_pos_weight,1.)
        self.assertEqual(c.scale_pos_weight,7.)
        self.assertEqual([k for k in a.get_params() if a.get_params()[k]!=c.get_params()[k]],['scale_pos_weight'])
        self.assertIsNone(a.class_weight)
        self.assertNotIn('is_unbalance',a.get_params())
        self.assertTrue(a.get_params()['deterministic'])

    def test_weight_rejects_nontraining_or_single_class(self):
        for bad in (self.validation, self.train.iloc[:0], self.train.assign(label=0), self.train.assign(split='test')):
            with self.assertRaises(ValueError): l.make_models(bad)

    def test_fits_receive_same_whitelisted_features_without_validation_or_early_stopping(self):
        models=l.make_models(self.train)
        a,c=models.values()
        with patch.object(a,'fit',wraps=a.fit) as af, patch.object(c,'fit',wraps=c.fit) as cf:
            b.fit_baselines(self.train,models)
        pd.testing.assert_frame_equal(af.call_args.args[0],cf.call_args.args[0])
        self.assertEqual(list(af.call_args.args[0].columns),FEATURE_COLUMNS)
        np.testing.assert_array_equal(af.call_args.args[1],self.train.label)
        self.assertEqual(af.call_args.kwargs,{})
        self.assertEqual(cf.call_args.kwargs,{})
        self.assertEqual(a.booster_.feature_name(),FEATURE_COLUMNS)
        with self.assertRaises(ValueError): b.fit_baselines(self.validation,models)

    def test_prediction_alignment_serialization_and_evaluator(self):
        models=l.make_models(self.train);b.fit_baselines(self.train,models)
        self.validation.index=np.arange(40)*3+7
        with tempfile.TemporaryDirectory() as directory:
            for name,model in models.items():
                table=b.validation_predictions(model,self.validation)
                pd.testing.assert_frame_equal(table[b.PREDICTION_METADATA],self.validation[b.PREDICTION_METADATA])
                np.testing.assert_array_equal(table.prediction,(table.score.to_numpy()>.5).astype(int))
                path=Path(directory)/f'{name}.joblib';joblib.dump(model,path)
                pd.testing.assert_frame_equal(table,b.validation_predictions(joblib.load(path),self.validation))
                report=evaluate_predictions(table,{'chb01_18':{'duration_sec':200,'seizures_sec':[[100,125]]}})
                self.assertEqual(report['event_metrics']['n_seizures'],1)
                self.assertEqual(report['evaluation_policy']['decision_time'],'window end')
                with self.assertRaises(ValueError): b.validation_predictions(model,self.validation.assign(split='test'))

    def test_known_prediction_changes_and_alignment_failure(self):
        frame=self.validation.iloc[:3][b.PREDICTION_METADATA].assign(label=[1,1,0])
        a=frame.assign(prediction=[0,1,0],score=[.2,.8,.2])
        c=frame.assign(prediction=[1,0,1],score=[.8,.2,.8])
        self.assertEqual(l.prediction_changes(a,c),{'changed':3,'misses_rescued':1,'new_ictal_misses':1,'new_fp_windows':1,'fp_windows_removed':0})
        with self.assertRaises(AssertionError): l.comparison_table({'a':a,'c':c.iloc[::-1]})


class LightGBMRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.data=self.root/'data.parquet';self.manifest=self.root/'manifest.csv'
        frame,metadata=synthetic_data();frame.to_parquet(self.data,index=False)
        rows=frame[frame.split.ne('supplementary')][['source_file','split']].drop_duplicates().rename(columns={'source_file':'filename'})
        rows['annotation_source']='synthetic_summary';rows.to_csv(self.manifest,index=False)
        b.write_json(self.data.with_suffix('.json'),{'output_sha256':sha256(self.data),'manifest_sha256':sha256(self.manifest),
                    'feature_columns':FEATURE_COLUMNS,'raw_recordings':metadata})
        self.rf,self.weighted=self.root/'rf',self.root/'weighted'
        with patch.object(b,'make_models',side_effect=small_rf_models): b.run_experiment(self.data,self.manifest,self.rf)
        w.run_experiment(self.data,self.manifest,self.rf,self.weighted)

    def test_full_runner_two_fits_test_exclusion_and_preservation(self):
        output=self.root/'lightgbm'
        before={str(p):sha256(p) for folder in (self.rf,self.weighted) for p in folder.iterdir()}
        read_parquet=pd.read_parquet
        with patch.object(l,'fit_baselines',wraps=b.fit_baselines) as fit, patch.object(pd,'read_parquet',wraps=read_parquet) as read:
            reports=l.run_experiment(self.data,self.manifest,self.rf,self.weighted,output)
        self.assertEqual(fit.call_count,1)
        self.assertEqual(set(fit.call_args.args[1]),{'ordinary_lgbm','weighted_lgbm'})
        self.assertTrue(fit.call_args.args[0].split.eq('train').all())
        for call in read.call_args_list:
            if Path(call.args[0])==self.data:
                self.assertEqual(call.kwargs['filters'],[('split','in',['train','validation'])])
            else:
                self.assertIn(Path(call.args[0]),[self.rf/'random_forest_validation.parquet',self.weighted/'weighted_validation.parquet'])
        self.assertTrue(all(sha256(p)==digest for p,digest in before.items()))
        self.assertEqual(len(reports),4)
        compare=pd.read_parquet(output/'window_comparison.parquet')
        self.assertEqual(set(compare.recording_id),{'chb01_18'})
        config=json.loads((output/'configuration.json').read_text())
        self.assertEqual(config['changed_parameters'],['scale_pos_weight'])
        self.assertEqual(config['feature_columns'],FEATURE_COLUMNS)
        self.assertEqual(json.loads((output/'verification.json').read_text())['model_fits'],2)
        for name,digest in json.loads((output/'artifact_sha256.json').read_text()).items(): self.assertEqual(sha256(output/name),digest)
        with self.assertRaises(FileExistsError): l.run_experiment(self.data,self.manifest,self.rf,self.weighted,output)

    def test_changed_reference_artifact_fails_before_training(self):
        with (self.rf/'random_forest_validation.parquet').open('ab') as stream: stream.write(b'x')
        with patch.object(l,'fit_baselines') as fit,self.assertRaisesRegex(ValueError,'Historical RF artifact changed'):
            l.run_experiment(self.data,self.manifest,self.rf,self.weighted,self.root/'out')
        fit.assert_not_called()


if __name__=='__main__': unittest.main()
