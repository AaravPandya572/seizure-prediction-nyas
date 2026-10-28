"""Leakage, alignment and reproducibility checks on tiny synthetic recordings."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

from src import baseline_rf as b
from src.evaluate import evaluate_predictions
from src.preprocess import FEATURE_COLUMNS, sha256


def synthetic_frame():
    rows = []
    for number, split in ((1, "train"), (18, "validation"), (26, "test")):
        for i in range(4):
            label = int(i in (1, 2))
            rows.append({**{name: float(label * 10 + i / 10) for name in FEATURE_COLUMNS},
                         "patient_id": "chb01", "recording_id": f"chb01_{number:02d}",
                         "source_file": f"chb01_{number:02d}.edf", "split": split,
                         "start_sec": 5 * i, "end_sec": 5 * (i + 1),
                         "decision_time_sec": 5 * (i + 1), "label": label,
                         "seizure_overlap_fraction": label, "filter_warmup": i < 2,
                         "unknown_numeric_metadata": 99999})
    return pd.DataFrame(rows)


def small_models():
    return {"dummy": DummyClassifier(strategy="most_frequent"),
            "random_forest": RandomForestClassifier(n_estimators=5, random_state=42, n_jobs=1)}


class BaselineTests(unittest.TestCase):
    def setUp(self):
        frame = synthetic_frame()
        self.train = frame[frame.split == "train"].copy()
        self.validation = frame[frame.split == "validation"].copy()

    def test_exact_features_ignore_metadata_and_restore_canonical_order(self):
        frame = self.train[self.train.columns[::-1]]
        features = b.select_features(frame)
        self.assertEqual(list(features.columns), FEATURE_COLUMNS)
        self.assertEqual(features.shape, (4, 126))
        self.assertFalse(set(b.PREDICTION_METADATA) & set(features.columns))
        pd.testing.assert_frame_equal(features, self.train[FEATURE_COLUMNS])

    def test_missing_duplicate_and_nonfinite_features_fail(self):
        bad_nan, bad_inf = self.train.copy(), self.train.copy()
        bad_nan.loc[0, FEATURE_COLUMNS[0]] = np.nan
        bad_inf.loc[0, FEATURE_COLUMNS[0]] = np.inf
        for frame in (self.train.drop(columns=FEATURE_COLUMNS[0]), bad_nan, bad_inf,
                      pd.concat([self.train, self.train[[FEATURE_COLUMNS[0]]]], axis=1)):
            with self.subTest(shape=frame.shape), self.assertRaises(ValueError):
                b.select_features(frame)

    def test_fixed_production_configuration(self):
        models = b.make_models()
        self.assertEqual(models["dummy"].strategy, "most_frequent")
        rf = models["random_forest"]
        self.assertEqual((rf.n_estimators, rf.random_state, rf.class_weight, rf.n_jobs), (200, 42, None, 4))
        self.assertIsNone(rf.max_depth)
        self.assertEqual(rf.max_features, "sqrt")

    def test_fit_rejects_validation_test_or_mixed_rows(self):
        for frame in (self.validation, synthetic_frame(), self.train.iloc[:0]):
            with self.assertRaisesRegex(ValueError, "Only training"):
                b.fit_baselines(frame, small_models())

    def test_fit_receives_only_train_features(self):
        models = small_models()
        with patch.object(models["random_forest"], "fit", wraps=models["random_forest"].fit) as fit:
            b.fit_baselines(self.train, models)
        features, labels = fit.call_args.args
        pd.testing.assert_frame_equal(features, self.train[FEATURE_COLUMNS])
        np.testing.assert_array_equal(labels, self.train.label)
        self.assertEqual(list(models["random_forest"].feature_names_in_), FEATURE_COLUMNS)

    def test_prediction_alignment_with_nonconsecutive_index(self):
        models = small_models()
        b.fit_baselines(self.train, models)
        validation = self.validation.copy()
        validation.index = [19, 2, 83, 7]
        result = b.validation_predictions(models["random_forest"], validation)
        pd.testing.assert_frame_equal(result[b.PREDICTION_METADATA], validation[b.PREDICTION_METADATA])
        np.testing.assert_array_equal(result.prediction, models["random_forest"].predict(validation[FEATURE_COLUMNS]))
        np.testing.assert_array_equal(result.score, models["random_forest"].predict_proba(validation[FEATURE_COLUMNS])[:, 1])
        report = evaluate_predictions(result, {"chb01_18": {"duration_sec": 20, "seizures_sec": [[5, 15]]}})
        self.assertEqual(report["window_metrics"]["n_windows"], 4)
        self.assertEqual(report["event_metrics"]["n_seizures"], 1)

    def test_test_predictions_are_rejected(self):
        for split in ("train", "test"):
            frame = self.validation.assign(split=split)
            with self.assertRaisesRegex(ValueError, "validation rows only"):
                b.validation_predictions(None, frame)

    def test_serialization_and_standard_tie_rule(self):
        models = small_models()
        b.fit_baselines(self.train, models)
        with tempfile.TemporaryDirectory() as directory:
            for name, model in models.items():
                path = Path(directory) / f"{name}.joblib"
                joblib.dump(model, path)
                pd.testing.assert_frame_equal(b.validation_predictions(model, self.validation),
                                              b.validation_predictions(joblib.load(path), self.validation))
        # A balanced, unsplit tree yields exactly .5; sklearn chooses class 0.
        tie = RandomForestClassifier(n_estimators=1, bootstrap=False, min_samples_split=10, random_state=42)
        tie.fit(self.train[FEATURE_COLUMNS], self.train.label)
        predictions = b.validation_predictions(tie, self.validation)
        self.assertTrue(predictions.score.eq(.5).all())
        self.assertTrue(predictions.prediction.eq(0).all())


class DataAndRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data, self.manifest = self.root / "data.parquet", self.root / "manifest.csv"
        self.frame = synthetic_frame()
        # The test rows deliberately contain invalid features: they must not load.
        self.frame.loc[self.frame.split.eq("test"), FEATURE_COLUMNS] = np.nan
        self.frame.to_parquet(self.data, index=False)
        records = self.frame[["source_file", "split"]].drop_duplicates().rename(columns={"source_file": "filename"})
        records["annotation_source"] = "synthetic_summary.txt"
        records.to_csv(self.manifest, index=False)
        self.audit = {"output_sha256": sha256(self.data), "manifest_sha256": sha256(self.manifest),
                      "feature_columns": FEATURE_COLUMNS,
                      "raw_recordings": {name: {"duration_sec": 20, "seizures_sec": [[5, 15]]}
                                         for name in records.filename}}
        self.save_audit()

    def save_audit(self):
        self.data.with_suffix(".json").write_text(json.dumps(self.audit))

    def update_data(self, frame):
        frame.to_parquet(self.data, index=False)
        self.audit["output_sha256"] = sha256(self.data)
        self.save_audit()

    def test_parquet_filter_and_disjoint_recordings(self):
        with patch.object(b.pd, "read_parquet", wraps=pd.read_parquet) as read:
            partitions, metadata = b.load_development_data(self.data, self.manifest)
        self.assertEqual(read.call_args.kwargs["filters"], [("split", "in", ["train", "validation"])])
        self.assertEqual(set(metadata), {"chb01_01", "chb01_18"})
        self.assertEqual(set(partitions["train"].recording_id), {"chb01_01"})
        self.assertEqual(set(partitions["validation"].recording_id), {"chb01_18"})

    def test_provenance_mismatch_fails(self):
        self.audit["output_sha256"] = "invalid"
        self.save_audit()
        with self.assertRaisesRegex(ValueError, "differs from"):
            b.load_development_data(self.data, self.manifest)

    def test_split_label_timeline_and_duplicate_errors_fail(self):
        for column, value in (("split", "validation"), ("label", 1), ("end_sec", 6),
                              ("recording_id", "chb01_18")):
            bad = self.frame.copy()
            bad.loc[0, column] = value
            self.update_data(bad)
            with self.subTest(column=column), self.assertRaises(ValueError):
                b.load_development_data(self.data, self.manifest)
        self.update_data(pd.concat([self.frame, self.frame.iloc[:1]]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            b.load_development_data(self.data, self.manifest)

    def test_full_runner_artifacts_and_no_overwrite(self):
        output = self.root / "experiment"
        with patch.object(b, "make_models", side_effect=small_models):
            reports = b.run_experiment(self.data, self.manifest, output)
        config = json.loads((output / "configuration.json").read_text())
        self.assertEqual(config["feature_columns"], FEATURE_COLUMNS)
        self.assertEqual(config["counts"]["train"]["windows"], 4)
        for name in ("dummy", "random_forest"):
            predictions = pd.read_parquet(output / f"{name}_validation.parquet")
            self.assertEqual(set(predictions.recording_id), {"chb01_18"})
            self.assertTrue(reports[name]["serialization_verified"])
            report = evaluate_predictions(predictions, config["validation_metadata"])
            self.assertEqual(report["window_metrics"], reports[name]["window_metrics"])
        hashes = json.loads((output / "artifact_sha256.json").read_text())
        for name, expected_hash in hashes.items():
            self.assertEqual(sha256(output / name), expected_hash)
        with self.assertRaises(FileExistsError):
            b.run_experiment(self.data, self.manifest, output)


if __name__ == "__main__":
    unittest.main()
