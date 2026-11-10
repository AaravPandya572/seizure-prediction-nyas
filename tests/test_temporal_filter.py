"""Hand-calculated causal rules and saved-prediction integration; no fitting."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.temporal_filter import RULES, filter_binary, filter_predictions
from src import temporal_filter_experiment as experiment
from src.evaluate import evaluate_predictions
from src.preprocess import sha256
from src.baseline_rf import write_json


def recording(values, rid='a', labels=None):
    n=len(values)
    return pd.DataFrame({'patient_id':['chb01']*n,'recording_id':[rid]*n,'source_file':[rid+'.edf']*n,
        'split':['train']*n,'start_sec':np.arange(n)*5,'end_sec':np.arange(1,n+1)*5,
        'decision_time_sec':np.arange(1,n+1)*5,'label':[0]*n if labels is None else labels,
        'prediction':values,'score':np.asarray(values)*.8+.1})


class CausalRuleTests(unittest.TestCase):
    def test_none_is_copy_and_isolated_positives_removed(self):
        values=np.array([0,1,0,0,1,0])
        result=filter_binary(values,'none')
        np.testing.assert_array_equal(result,values)
        result[0]=1;self.assertEqual(values[0],0)
        for rule in RULES[1:]:np.testing.assert_array_equal(filter_binary(values,rule),np.zeros(6))

    def test_startup_and_one_decision_confirmation_delay(self):
        for rule in RULES[1:]:
            np.testing.assert_array_equal(filter_binary([1],rule),[0])
            np.testing.assert_array_equal(filter_binary([1,1,1],rule),[0,1,1])
            np.testing.assert_array_equal(filter_binary([0,1,1,1],rule),[0,0,1,1])

    def test_intermittent_positives_and_raw_history(self):
        np.testing.assert_array_equal(filter_binary([1,0,1,0,1,0],'two_consecutive'),[0]*6)
        np.testing.assert_array_equal(filter_binary([1,0,1,0,1,0],'two_of_three'),[0,0,1,0,1,0])
        # Two-of-three can stay positive on a current negative raw prediction.
        np.testing.assert_array_equal(filter_binary([1,1,0,0],'two_of_three'),[0,1,1,0])
        np.testing.assert_array_equal(filter_binary([1,1,0,0],'two_consecutive'),[0,1,0,0])

    def test_prefix_invariance_proves_no_future_or_backdating(self):
        values=np.random.default_rng(42).integers(0,2,100)
        for rule in RULES:
            complete=filter_binary(values,rule)
            for k in range(1,len(values)+1):np.testing.assert_array_equal(filter_binary(values[:k],rule),complete[:k])

    def test_invalid_inputs(self):
        for values in ([],[[1]], [0,np.nan], [0,np.inf], [0,2], ['0','1']):
            with self.subTest(values=values),self.assertRaises(ValueError):filter_binary(values,'two_consecutive')
        with self.assertRaises(ValueError):filter_binary([0,1],'unknown')


class TimelineAndEventTests(unittest.TestCase):
    def test_metadata_labels_scores_and_row_alignment_unchanged(self):
        frame=recording([0,1,1,0,1]);frame.index=[7,2,7,99,0];original=frame.copy()
        meta={'a':{'duration_sec':25,'seizures_sec':[]}}
        for rule in RULES:
            result=filter_predictions(frame,meta,rule)
            pd.testing.assert_frame_equal(frame,original)
            pd.testing.assert_frame_equal(result.drop(columns=['prediction','unfiltered_prediction']),frame.drop(columns='prediction'))
            np.testing.assert_array_equal(result.unfiltered_prediction,frame.prediction)
            if rule=='none':pd.testing.assert_frame_equal(result.drop(columns='unfiltered_prediction'),frame)
            with self.assertRaises(ValueError):filter_predictions(result,meta,rule)

    def test_state_resets_even_with_interleaved_recordings_and_duplicate_indices(self):
        a,b=recording([0,1],'a'),recording([1,0],'b')
        frame=pd.concat([a.iloc[:1],b.iloc[:1],a.iloc[1:],b.iloc[1:]])
        frame.index=[7,7,3,3]
        meta={rid:{'duration_sec':10,'seizures_sec':[]} for rid in ('a','b')}
        for rule in RULES[1:]:np.testing.assert_array_equal(filter_predictions(frame,meta,rule).prediction,[0]*4)

    def test_gaps_partial_missing_duplicate_or_out_of_order_windows_rejected(self):
        frame=recording([0,1,1,0]);meta={'a':{'duration_sec':20,'seizures_sec':[]}}
        for bad in (frame.drop(index=1),frame.iloc[1:],frame.iloc[:-1],frame.iloc[::-1],pd.concat([frame,frame.iloc[-1:]])):
            with self.assertRaises(ValueError):filter_predictions(bad,meta,'two_of_three')
        half=frame.iloc[:2].copy();half['start_sec']=[0,2.5];half['end_sec']=[2.5,5];half['decision_time_sec']=half.end_sec
        with self.assertRaisesRegex(ValueError,'five-second'):filter_predictions(half,{'a':{'duration_sec':5,'seizures_sec':[]}},'none')
        bad=frame.copy();bad.loc[1,'decision_time_sec']=9
        with self.assertRaises(ValueError):filter_predictions(bad,meta,'none')

    def test_isolated_short_seizure_detection_can_be_lost(self):
        frame=recording([0,1,0,0],labels=[0,1,0,0]);meta={'a':{'duration_sec':20,'seizures_sec':[[6,11]]}}
        raw=evaluate_predictions(frame,meta)
        self.assertEqual(raw['event_metrics']['latencies_sec'],[4.])
        for rule in RULES[1:]:
            report=evaluate_predictions(filter_predictions(frame,meta,rule),meta)
            self.assertEqual(report['event_metrics']['n_missed'],1)
            self.assertIsNone(report['event_metrics']['mean_latency_sec'])

    def test_offset_boundary_is_not_retroactively_detected(self):
        frame=recording([0,1,1,0],labels=[0,1,1,0]);meta={'a':{'duration_sec':20,'seizures_sec':[[6,15]]}}
        for rule in RULES[1:]:
            report=evaluate_predictions(filter_predictions(frame,meta,rule),meta)
            event=report['per_recording']['a']['event_metrics']
            self.assertEqual(event['n_missed'],1)
            self.assertEqual(event['alarms'][0]['start_sec'],15.)
            self.assertEqual(event['n_false_alarms'],1)

    def test_two_of_three_bridges_interruption_without_extra_event(self):
        frame=recording([1,1,0,1,1,0,0],labels=[1]*6+[0]);meta={'a':{'duration_sec':35,'seizures_sec':[[0,30]]}}
        raw=evaluate_predictions(frame,meta)['event_metrics']
        filtered=evaluate_predictions(filter_predictions(frame,meta,'two_of_three'),meta)['event_metrics']
        self.assertEqual(raw['n_repeat_alarms'],1)
        self.assertEqual(filtered['n_repeat_alarms'],0)
        self.assertEqual(filtered['n_detected'],1)
        self.assertEqual(filtered['latencies_sec'],[10.])


class SavedArtifactTests(unittest.TestCase):
    def test_full_analysis_uses_only_saved_tables_and_protects_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folders={key:root/key for key in ('rf_cv','lgbm_cv','rf_val','lgbm_val')}
            base=[0,1,1,0,1,0,0,1]
            cv_metadata={rid:{'duration_sec':40,'seizures_sec':[[6,25]]} for rid in experiment.HOLDOUTS}
            val_metadata={'chb01_17':{'duration_sec':40,'seizures_sec':[]},'chb01_18':{'duration_sec':40,'seizures_sec':[[6,25]]}}
            for key,folder in folders.items():
                folder.mkdir();metadata=cv_metadata if key.endswith('cv') else val_metadata
                write_json(folder/'configuration.json',{'metadata':metadata} if key.endswith('cv') else {'validation_metadata':metadata})
                frames=[]
                for i,(rid,m) in enumerate(metadata.items(),1):
                    labels=[0,1,1,1,1,0,0,0] if m['seizures_sec'] else [0]*8
                    table=recording(base if m['seizures_sec'] else [0,0,1,0,0,0,0,0],rid,labels)
                    if key.endswith('cv'):
                        sub=folder/f'fold_{i}_{rid}';sub.mkdir();stem='ordinary' if key=='rf_cv' else 'ordinary_lgbm'
                        table.to_parquet(sub/f'{stem}_predictions.parquet',index=False)
                        write_json(sub/f'{stem}_evaluation.json',evaluate_predictions(table,{rid:m}))
                    else:table['split']='validation'
                    frames.append(table)
                joined=pd.concat(frames,ignore_index=True);report=evaluate_predictions(joined,metadata)
                if key.endswith('cv'):
                    write_json(folder/'summary.json',{'ordinary' if key=='rf_cv' else 'ordinary_lgbm':report})
                else:
                    stem='random_forest' if key=='rf_val' else 'ordinary_lgbm'
                    joined.to_parquet(folder/f'{stem}_validation.parquet',index=False)
                    write_json(folder/f'{stem}_evaluation.json',report)
                write_json(folder/'artifact_sha256.json',{str(p.relative_to(folder)):sha256(p) for p in folder.rglob('*') if p.is_file()})
            original={str(p):sha256(p) for folder in folders.values() for p in folder.rglob('*') if p.is_file()}
            with patch.object(pd,'read_parquet',wraps=pd.read_parquet) as read:
                result=experiment.run_experiment(folders['rf_cv'],folders['lgbm_cv'],folders['rf_val'],folders['lgbm_val'],root/'out')
            self.assertEqual(read.call_count,10)
            self.assertTrue(all(Path(call.args[0]).is_relative_to(root) for call in read.call_args_list))
            self.assertTrue(all(sha256(p)==digest for p,digest in original.items()))
            self.assertEqual(result['primary_cv']['ordinary_lgbm']['none']['event_metrics']['n_seizures'],4)
            for partition in ('primary_cv','secondary_validation'):
                changed=pd.read_parquet(root/'out'/partition/'changed_predictions.parquet')
                self.assertTrue(changed.prediction.ne(changed.unfiltered_prediction).all())
                self.assertTrue(changed.rule.ne('none').all())
                self.assertTrue(changed.decision_time_sec.eq(changed.end_sec).all())
            for name,digest in json.loads((root/'out/artifact_sha256.json').read_text()).items():self.assertEqual(sha256(root/'out'/name),digest)
            with self.assertRaises(FileExistsError):experiment.run_experiment(*folders.values(),root/'out')


if __name__=='__main__':unittest.main()
