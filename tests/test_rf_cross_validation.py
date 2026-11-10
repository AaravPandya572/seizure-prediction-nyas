"""Training-only fold isolation, complete timelines and aggregation safeguards."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src import rf_cross_validation as cv
from src import baseline_rf as b
from src.preprocess import FEATURE_COLUMNS, sha256


def synthetic_data():
    rows, metadata = [], {}
    for rid in cv.TRAIN_IDS:
        metadata[rid] = {"duration_sec": 20, "seizures_sec": [[5, 15]] if rid in cv.HOLDOUT_IDS else []}
        for i in range(4):
            label = int(rid in cv.HOLDOUT_IDS and i in (1, 2))
            rows.append({**{name: float(10 * label + i / 10) for name in FEATURE_COLUMNS},
                         "patient_id": "chb01", "recording_id": rid, "source_file": rid + ".edf",
                         "split": "train", "start_sec": i * 5, "end_sec": i * 5 + 5,
                         "decision_time_sec": i * 5 + 5, "label": label,
                         "seizure_overlap_fraction": label, "numeric_metadata": 999})
    return pd.DataFrame(rows), metadata


def tiny_models():
    return {"random_forest": RandomForestClassifier(n_estimators=3, random_state=42, n_jobs=1)}


class FoldTests(unittest.TestCase):
    def setUp(self):
        self.frame, self.metadata = synthetic_data()

    def test_exact_four_folds_disjoint_and_normals_retained(self):
        seen = []
        for rid in cv.HOLDOUT_IDS:
            train, held = cv.fold_data(self.frame, rid)
            self.assertEqual(set(train.recording_id), set(cv.TRAIN_IDS) - {rid})
            self.assertEqual(set(held.recording_id), {rid})
            self.assertTrue({"chb01_01", "chb01_14"} <= set(train.recording_id))
            self.assertFalse(set(train.index) & set(held.index))
            seen.append(rid)
        self.assertEqual(len(set(seen)), 4)

    def test_forbidden_partitions_and_recordings_rejected(self):
        for rid in ("chb01_17", "chb01_21", "chb01_23"):
            bad = self.frame.copy()
            bad.loc[0, "recording_id"] = rid
            with self.assertRaises(ValueError):
                cv.fold_data(bad, "chb01_03")
        with self.assertRaises(ValueError):
            cv.fold_data(self.frame.assign(split="validation"), "chb01_03")
        with self.assertRaises(ValueError):
            cv.fold_data(self.frame, "chb01_01")

    def test_predictions_align_and_only_eeg_features_enter_fit(self):
        train, held = cv.fold_data(self.frame, "chb01_03")
        model = tiny_models()["random_forest"]
        with patch.object(model, "fit", wraps=model.fit) as fit:
            b.fit_baselines(train, {"ordinary": model})
        self.assertEqual(list(fit.call_args.args[0].columns), FEATURE_COLUMNS)
        held.index = [20, 3, 91, 8]
        prediction = cv.predict_holdout(model, held, "chb01_03")
        pd.testing.assert_frame_equal(prediction[b.PREDICTION_METADATA], held[b.PREDICTION_METADATA])
        np.testing.assert_array_equal(prediction.prediction, model.predict(held[FEATURE_COLUMNS]))
        with self.assertRaises(ValueError):
            cv.predict_holdout(model, held.assign(split="test"), "chb01_03")

    def test_aggregation_counts_events_once_and_preserves_undefined_metrics(self):
        tables = [self.frame[self.frame.recording_id.eq(rid)][b.PREDICTION_METADATA].assign(prediction=0, score=0.)
                  for rid in cv.HOLDOUT_IDS]
        summary = cv.aggregate_predictions(tables, self.metadata)
        self.assertEqual(summary["event_metrics"]["n_seizures"], 4)
        self.assertEqual(summary["event_metrics"]["n_missed"], 4)
        self.assertEqual(summary["window_metrics"]["confusion_matrix"], [[8, 0], [8, 0]])
        self.assertEqual(summary["macro_window_metrics"]["precision"]["n_defined"], 0)
        self.assertIsNone(summary["macro_window_metrics"]["precision"]["mean"])
        self.assertIsNone(summary["event_metrics"]["mean_latency_sec"])
        self.assertEqual(summary["event_metrics"]["evaluated_duration_sec"], 80)
        with self.assertRaises(ValueError):
            cv.aggregate_predictions(tables[:-1] + [tables[0]], self.metadata)


class LoaderAndRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data, self.manifest, self.config = (self.root / name for name in ("data.parquet", "manifest.csv", "baseline.json"))
        self.frame, metadata = synthetic_data()
        forbidden = self.frame.iloc[:3].copy()
        forbidden["recording_id"] = ["chb01_17", "chb01_21", "chb01_23"]
        forbidden["source_file"] = forbidden.recording_id + ".edf"
        forbidden["split"] = ["validation", "test", "supplementary"]
        forbidden[FEATURE_COLUMNS] = np.nan
        pd.concat([self.frame, forbidden], ignore_index=True).to_parquet(self.data, index=False)
        manifest = pd.DataFrame({"filename": [rid + ".edf" for rid in cv.TRAIN_IDS] + ["chb01_17.edf", "chb01_21.edf"],
                                 "split": ["train"] * 6 + ["validation", "test"], "annotation_source": "synthetic"})
        manifest.to_csv(self.manifest, index=False)
        self.audit = {"output_sha256": sha256(self.data), "manifest_sha256": sha256(self.manifest),
                      "feature_columns": FEATURE_COLUMNS, "raw_recordings": {rid + ".edf": m for rid, m in metadata.items()}}
        b.write_json(self.data.with_suffix('.json'), self.audit)
        b.write_json(self.config, {"data_sha256": sha256(self.data), "manifest_sha256": sha256(self.manifest),
                     "split_audit_sha256": sha256(self.data.with_suffix('.json')), "versions": {}, "source_sha256": {},
                     "feature_columns": FEATURE_COLUMNS, "models": {"random_forest": tiny_models()["random_forest"].get_params()}})

    def test_only_training_rows_decoded(self):
        with patch.object(pd, "read_parquet", wraps=pd.read_parquet) as read:
            frame, metadata = cv.load_training_data(self.data, self.manifest)
        self.assertEqual(read.call_count, 1)
        self.assertEqual(read.call_args.kwargs['filters'], [("split", "==", "train")])
        self.assertEqual(set(frame.recording_id), set(cv.TRAIN_IDS))
        self.assertEqual(set(metadata), set(cv.TRAIN_IDS))

    def test_incomplete_timeline_rejected(self):
        self.frame.iloc[1:].to_parquet(self.data, index=False)
        self.audit['output_sha256'] = sha256(self.data)
        b.write_json(self.data.with_suffix('.json'), self.audit)
        with self.assertRaises(ValueError):
            cv.load_training_data(self.data, self.manifest)

    def test_independent_models_determinism_and_artifact_reload(self):
        output = self.root / 'run'
        with patch.object(cv, 'make_models', side_effect=tiny_models), patch.object(cv, 'fit_baselines', wraps=b.fit_baselines) as fit:
            summary = cv.run_experiment(self.data, self.manifest, self.config, output)
        self.assertEqual(fit.call_count, 4)
        estimators = []
        for rid, call in zip(cv.HOLDOUT_IDS, fit.call_args_list):
            train, models = call.args
            self.assertNotIn(rid, set(train.recording_id))
            self.assertEqual(len(train), 20)
            a, c = models['ordinary'], models['weighted']
            self.assertEqual([k for k in a.get_params() if a.get_params()[k] != c.get_params()[k]], ['class_weight'])
            estimators.extend(models.values())
        self.assertEqual(len({id(m) for m in estimators}), 8)
        with patch.object(cv, 'make_models', side_effect=tiny_models):
            repeated = cv.run_experiment(self.data, self.manifest, self.config, self.root / 'repeat')
        self.assertEqual(summary, repeated)
        for file in output.rglob('*_predictions.parquet'):
            pd.testing.assert_frame_equal(pd.read_parquet(file), pd.read_parquet(self.root / 'repeat' / file.relative_to(output)))
        for name, digest in json.loads((output / 'artifact_sha256.json').read_text()).items():
            self.assertEqual(sha256(output / name), digest)
        with self.assertRaises(FileExistsError):
            cv.run_experiment(self.data, self.manifest, self.config, output)


if __name__ == '__main__':
    unittest.main()
