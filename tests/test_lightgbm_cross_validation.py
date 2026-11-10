"""Synthetic fold isolation, fold-local weights, alignment and aggregation checks."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from src import baseline_rf as b, rf_cross_validation as rf, lightgbm_cross_validation as cv
from src.lightgbm_experiment import make_models
from src.preprocess import FEATURE_COLUMNS,sha256


def synthetic_data():
    rows,metadata=[],{}
    positives=dict(zip(rf.HOLDOUT_IDS,(3,4,5,6)))
    for rid in rf.TRAIN_IDS:
        n=positives.get(rid,0)
        metadata[rid+'.edf']={'duration_sec':200,'seizures_sec':[[50,50+5*n]] if n else []}
        for i in range(40):
            label=int(10 <= i < 10+n)
            rows.append({'recording_id':rid,'source_file':rid+'.edf','patient_id':'chb01','split':'train',
                         'start_sec':i*5,'end_sec':i*5+5,'decision_time_sec':i*5+5,'label':label,
                         'annotation_leak':label,'numeric_metadata':999,
                         **{name:label*10+(i%3)/10 for name in FEATURE_COLUMNS}})
    return pd.DataFrame(rows),metadata


def tiny_rf_models():
    return {'random_forest':RandomForestClassifier(n_estimators=3,random_state=42,n_jobs=1)}


class CVUnitTests(unittest.TestCase):
    def test_fold_isolation_and_weights_from_each_subset(self):
        frame,_=synthetic_data();ratios=[]
        for rid in rf.HOLDOUT_IDS:
            train,held=rf.fold_data(frame,rid)
            self.assertNotIn(rid,set(train.recording_id))
            self.assertEqual(set(train.recording_id),set(rf.TRAIN_IDS)-{rid})
            self.assertEqual(set(held.recording_id),{rid})
            models=make_models(train);a,c=models.values()
            ratio=train.label.eq(0).sum()/train.label.eq(1).sum();ratios.append(ratio)
            self.assertEqual(c.scale_pos_weight,ratio)
            self.assertEqual(a.scale_pos_weight,1.)
            self.assertEqual([k for k in a.get_params() if a.get_params()[k]!=c.get_params()[k]],['scale_pos_weight'])
        self.assertEqual(len(set(ratios)),4)

    def test_continuity_counts_true_positive_runs_without_counting_fp_as_seizure(self):
        t=pd.DataFrame({'recording_id':['chb01_03']*7,'start_sec':np.arange(7)*5,
                        'end_sec':np.arange(1,8)*5,'label':[0,1,1,1,1,1,0],
                        'prediction':[1,1,1,0,1,1,1]})
        c=cv.ictal_continuity(t)
        self.assertEqual((c['ictal_windows'],c['true_positive_windows'],c['true_positive_runs']), (5,4,2))
        self.assertEqual(c['longest_run_sec'],10.)
        self.assertEqual(c['runs'],[{'start_sec':5.,'end_sec':15.,'windows':2},
                                    {'start_sec':20.,'end_sec':30.,'windows':2}])
        self.assertEqual(cv.ictal_continuity(t.assign(prediction=0))['longest_run_windows'],0)
        with self.assertRaises(ValueError):cv.ictal_continuity(t.iloc[:0])

    def test_aggregation_unique_events_and_undefined_precision(self):
        frame,raw=synthetic_data();metadata={Path(k).stem:v for k,v in raw.items()}
        tables=[frame[frame.recording_id.eq(rid)][b.PREDICTION_METADATA].assign(prediction=0,score=0.) for rid in rf.HOLDOUT_IDS]
        report=rf.aggregate_predictions(tables,metadata)
        self.assertEqual(report['event_metrics']['n_seizures'],4)
        self.assertEqual(report['event_metrics']['n_missed'],4)
        self.assertEqual(report['window_metrics']['n_windows'],160)
        self.assertIsNone(report['macro_window_metrics']['precision']['mean'])
        self.assertEqual(report['macro_window_metrics']['precision']['n_defined'],0)
        with self.assertRaises(ValueError):rf.aggregate_predictions(tables[:3]+[tables[0]],metadata)


class CVRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.data=self.root/'data.parquet';self.manifest=self.root/'manifest.csv'
        frame,metadata=synthetic_data();self.frame=frame
        forbidden=frame.iloc[:8].copy()
        forbidden['recording_id']=['chb01_17','chb01_18','chb01_21','chb01_22','chb01_26','chb01_23','chb01_24','chb01_25']
        forbidden['split']=['validation']*2+['test']*3+['supplementary']*3
        forbidden['source_file']=forbidden.recording_id+'.edf';forbidden[FEATURE_COLUMNS]=np.nan
        pd.concat([frame,forbidden],ignore_index=True).to_parquet(self.data,index=False)
        manifest=pd.concat([frame[['source_file','split']].drop_duplicates(),forbidden.iloc[:5][['source_file','split']]],ignore_index=True)
        manifest=manifest.rename(columns={'source_file':'filename'});manifest['annotation_source']='synthetic'
        manifest.to_csv(self.manifest,index=False)
        b.write_json(self.data.with_suffix('.json'),{'output_sha256':sha256(self.data),'manifest_sha256':sha256(self.manifest),
                    'feature_columns':FEATURE_COLUMNS,'raw_recordings':metadata})
        base=self.root/'baseline.json'
        b.write_json(base,{'data_sha256':sha256(self.data),'manifest_sha256':sha256(self.manifest),
            'split_audit_sha256':sha256(self.data.with_suffix('.json')),'versions':{},'source_sha256':{},
            'feature_columns':FEATURE_COLUMNS,'models':{'random_forest':tiny_rf_models()['random_forest'].get_params()}})
        self.rf=self.root/'rf'
        with patch.object(rf,'make_models',side_effect=tiny_rf_models):rf.run_experiment(self.data,self.manifest,base,self.rf)
        self.lgbm_config=self.root/'lightgbm.json'
        b.write_json(self.lgbm_config,{'data_sha256':sha256(self.data),'manifest_sha256':sha256(self.manifest),
            'feature_columns':FEATURE_COLUMNS,'versions':{},'source_sha256':{},'decision_rule':'score >0.5; ties 0',
            'models':{name:m.get_params() for name,m in make_models(frame).items()}})

    def test_runner_eight_independent_fits_and_no_reserved_data(self):
        output=self.root/'run';real_make=make_models;created=[];inputs=[]
        def capture_make(train):
            models=real_make(train);created.extend(models.values())
            return models
        original_fit=LGBMClassifier.fit
        def capture_fit(model,x,y,**kwargs):
            inputs.append((x.copy(),np.array(y),dict(kwargs)))
            return original_fit(model,x,y,**kwargs)
        old_hashes={str(p):sha256(p) for p in self.rf.rglob('*') if p.is_file()}
        with patch.object(cv,'make_models',side_effect=capture_make),patch.object(LGBMClassifier,'fit',new=capture_fit),patch.object(cv,'fit_baselines',wraps=b.fit_baselines) as fit,patch.object(pd,'read_parquet',wraps=pd.read_parquet) as read:
            reports=cv.run_experiment(self.data,self.manifest,self.rf,self.lgbm_config,output)
        self.assertEqual(len(inputs),8)
        self.assertEqual(len({id(m) for m in created}),8)
        self.assertEqual(fit.call_count,4)
        for index,rid in enumerate(rf.HOLDOUT_IDS):
            train,models=fit.call_args_list[index].args
            self.assertNotIn(rid,set(train.recording_id))
            x,y,kwargs=inputs[index*2];xx,yy,otherkwargs=inputs[index*2+1]
            pd.testing.assert_frame_equal(x,xx);np.testing.assert_array_equal(y,yy)
            self.assertEqual(list(x.columns),FEATURE_COLUMNS)
            self.assertEqual(kwargs,{})
            self.assertEqual(otherkwargs,{})
        for call in read.call_args_list:
            path=Path(call.args[0])
            if path==self.data:self.assertEqual(call.kwargs['filters'],[('split','==','train')])
            else:
                self.assertTrue(path.is_relative_to(self.rf))
                self.assertTrue(path.name.endswith('_predictions.parquet'))
        self.assertTrue(all(sha256(p)==digest for p,digest in old_hashes.items()))
        for name in reports:self.assertEqual(reports[name]['event_metrics']['n_seizures'],4)
        for file in output.rglob('*_lgbm_predictions.parquet'):
            table=pd.read_parquet(file)
            held=self.frame[self.frame.recording_id.eq(table.recording_id.iloc[0])]
            pd.testing.assert_frame_equal(table[b.PREDICTION_METADATA],held[b.PREDICTION_METADATA].reset_index(drop=True))
        with self.assertRaises(FileExistsError):cv.run_experiment(self.data,self.manifest,self.rf,self.lgbm_config,output)

    def test_fixed_setting_reproduction_and_artifact_hashes(self):
        a,bdir=self.root/'a',self.root/'b'
        first=cv.run_experiment(self.data,self.manifest,self.rf,self.lgbm_config,a)
        second=cv.run_experiment(self.data,self.manifest,self.rf,self.lgbm_config,bdir)
        self.assertEqual(first,second)
        for path in a.rglob('*_lgbm_predictions.parquet'):
            pd.testing.assert_frame_equal(pd.read_parquet(path),pd.read_parquet(bdir/path.relative_to(a)))
        for name,digest in json.loads((a/'artifact_sha256.json').read_text()).items():self.assertEqual(sha256(a/name),digest)

    def test_changed_rf_artifact_rejected_before_fit(self):
        with (self.rf/'fold_1_chb01_03/ordinary_predictions.parquet').open('ab') as stream:stream.write(b'x')
        with patch.object(cv,'fit_baselines') as fit,self.assertRaisesRegex(ValueError,'RF artifact changed'):
            cv.run_experiment(self.data,self.manifest,self.rf,self.lgbm_config,self.root/'out')
        fit.assert_not_called()


if __name__=='__main__':unittest.main()
