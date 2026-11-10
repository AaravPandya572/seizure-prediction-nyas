"""Controlled comparison tests; only synthetic EEG-like feature tables."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import baseline_rf as b, weighted_rf as w
from src.preprocess import FEATURE_COLUMNS, sha256
import test_baseline_rf as fixtures
from test_baseline_rf import small_models, synthetic_frame


class ControlledComparisonTests(unittest.TestCase):
    def test_only_class_weight_differs_and_original_is_untouched(self):
        ordinary = b.make_models()["random_forest"]
        weighted = w.weighted_clone(ordinary)
        differences = [key for key in ordinary.get_params()
                       if ordinary.get_params()[key] != weighted.get_params()[key]]
        self.assertEqual(differences, ["class_weight"])
        self.assertEqual(weighted.class_weight, "balanced")
        self.assertIsNone(ordinary.class_weight)

    def test_both_forests_receive_identical_ordered_inputs(self):
        frame = synthetic_frame()
        train = frame[frame.split.eq("train")]
        ordinary = small_models()["random_forest"]
        weighted = w.weighted_clone(ordinary)
        with patch.object(ordinary, "fit", wraps=ordinary.fit) as a, patch.object(weighted, "fit", wraps=weighted.fit) as c:
            b.fit_baselines(train, {"ordinary": ordinary, "weighted": weighted})
        pd.testing.assert_frame_equal(a.call_args.args[0], c.call_args.args[0])
        np.testing.assert_array_equal(a.call_args.args[1], c.call_args.args[1])
        self.assertEqual(list(a.call_args.args[0].columns), FEATURE_COLUMNS)
        self.assertEqual(len(a.call_args.args[1]), 4)

    def test_fingerprint_detects_feature_label_or_row_order_changes(self):
        train = synthetic_frame().query("split == 'train'").copy()
        signature = w.training_fingerprint(train)
        for column in (FEATURE_COLUMNS[0], "label"):
            bad = train.copy()
            bad.loc[0, column] += 1
            self.assertNotEqual(w.training_fingerprint(bad), signature)
        self.assertNotEqual(w.training_fingerprint(train.iloc[::-1]), signature)
        with self.assertRaises(ValueError):
            w.training_fingerprint(synthetic_frame())

    def test_comparison_alignment_and_known_changes(self):
        frame = synthetic_frame().query("split == 'validation'")[b.PREDICTION_METADATA].reset_index(drop=True)
        ordinary = frame.assign(prediction=[0, 0, 1, 0], score=[.1, .4, .8, .1])
        weighted = frame.assign(prediction=[1, 1, 0, 0], score=[.6, .7, .3, .2])
        result = w.compare_predictions(ordinary, weighted)
        self.assertEqual(result.changed.tolist(), [True, True, True, False])
        self.assertEqual(result.corrected.tolist(), [False, True, False, False])
        self.assertEqual(result.regressed.tolist(), [True, False, True, False])
        self.assertEqual(result.previously_missed.tolist(), [False, True, False, False])
        with self.assertRaises(AssertionError):
            w.compare_predictions(ordinary, weighted.iloc[::-1])


class WeightedRunnerTests(unittest.TestCase):
    def setUp(self):
        # Reuse synthetic Parquet/manifest fixture; its test features are NaN.
        fixture = fixtures.DataAndRunnerTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.data, self.manifest, self.root = fixture.data, fixture.manifest, fixture.root
        self.baseline, self.output = self.root / "baseline", self.root / "weighted"
        with patch.object(b, "make_models", side_effect=small_models):
            b.run_experiment(self.data, self.manifest, self.baseline)

    def test_runner_excludes_test_rows_preserves_baseline_and_reloads(self):
        before = {p.name: sha256(p) for p in self.baseline.iterdir()}
        # Capture every Parquet read: development reads must be filtered;
        # the remaining read is the baseline validation predictions only.
        with patch.object(w, "fit_baselines", wraps=w.fit_baselines) as fit, patch.object(pd, "read_parquet", wraps=pd.read_parquet) as read:
            ordinary, weighted = w.run_experiment(self.data, self.manifest, self.baseline, self.output)
        self.assertEqual(fit.call_count, 1)
        self.assertTrue(fit.call_args.args[0].split.eq("train").all())
        for call in read.call_args_list:
            if Path(call.args[0]) == self.data:
                self.assertEqual(call.kwargs["filters"], [("split", "in", ["train", "validation"])])
            else:
                self.assertEqual(Path(call.args[0]), self.baseline / "random_forest_validation.parquet")
        self.assertEqual(before, {p.name: sha256(p) for p in self.baseline.iterdir()})
        check = json.loads((self.output / "verification.json").read_text())
        self.assertTrue(check["weighted_reload_identical"])
        self.assertFalse(check["test_rows_used"])
        predictions = pd.read_parquet(self.output / "weighted_validation.parquet")
        self.assertEqual(set(predictions.recording_id), {"chb01_18"})
        self.assertEqual(weighted["window_metrics"]["n_windows"], 4)
        self.assertEqual(ordinary["window_metrics"]["n_windows"], 4)
        with self.assertRaises(FileExistsError):
            w.run_experiment(self.data, self.manifest, self.baseline, self.output)

    def test_changed_baseline_artifact_rejected_before_fit(self):
        with (self.baseline / "random_forest_evaluation.json").open("a") as stream:
            stream.write("\n")
        with patch.object(w, "fit_baselines") as fit, self.assertRaisesRegex(ValueError, "Baseline artifact changed"):
            w.run_experiment(self.data, self.manifest, self.baseline, self.output)
        fit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
