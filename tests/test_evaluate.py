"""Hand-calculated evaluation examples; no datasets, models, or EDF access."""

import json
import unittest

import numpy as np
import pandas as pd

from src.evaluate import alarm_episodes, event_metrics, evaluate_predictions, window_metrics


def events(prediction, seizures=(), step=5):
    """Make a complete synthetic recording with equally spaced windows."""
    starts = np.arange(len(prediction)) * step
    return event_metrics(starts, starts + step, prediction, seizures,
                         recording_duration_sec=len(prediction) * step)


class WindowMetricTests(unittest.TestCase):
    def test_hand_calculated_confusion_and_rates(self):
        result = window_metrics([0, 0, 0, 1, 1, 1], [0, 1, 0, 1, 0, 1])
        self.assertEqual(result["confusion_matrix"], [[2, 1], [1, 2]])
        for name in ("accuracy", "precision", "recall", "specificity", "f1"):
            self.assertAlmostEqual(result[name], 2 / 3)
        self.assertEqual(result["positive_prevalence"], .5)

    def test_precision_recall_specificity_have_distinct_denominators(self):
        result = window_metrics([1, 1, 1, 0, 0], [1, 0, 0, 1, 0])
        self.assertEqual(result["precision"], 1 / 2)
        self.assertEqual(result["recall"], 1 / 3)
        self.assertEqual(result["specificity"], 1 / 2)
        self.assertEqual(result["f1"], 2 / 5)
        self.assertEqual(result["accuracy"], 2 / 5)

    def test_average_precision_uses_scores(self):
        # Positive ranks are 1 and 3: AP = (1/1 + 2/3) / 2 = 5/6.
        result = window_metrics([1, 0, 1, 0], [1, 1, 0, 0], [.9, .8, .7, .1])
        self.assertAlmostEqual(result["average_precision"], 5 / 6)
        reversed_scores = window_metrics([1, 0, 1, 0], [1, 1, 0, 0], [.1, .7, .8, .9])
        self.assertNotEqual(result["average_precision"], reversed_scores["average_precision"])
        self.assertIsNone(window_metrics([1, 0], [1, 0])["average_precision"])

    def test_tied_scores_ap_equals_prevalence(self):
        result = window_metrics([1, 0, 0, 0], [0, 0, 0, 0], [.5] * 4)
        self.assertEqual(result["average_precision"], .25)

    def test_no_positive_truth_or_predictions_is_not_perfect_recall(self):
        result = window_metrics([0, 0], [0, 0], [.1, .2])
        self.assertEqual(result["confusion_matrix"], [[2, 0], [0, 0]])
        self.assertEqual(result["accuracy"], 1)
        self.assertEqual(result["specificity"], 1)
        for name in ("precision", "recall", "f1", "average_precision"):
            self.assertIsNone(result[name])
            self.assertIn(name, result["undefined_metrics"])
        json.dumps(result, allow_nan=False)

    def test_all_missed_and_all_positive_truth(self):
        result = window_metrics([1, 1], [0, 0], [.3, .2])
        self.assertEqual(result["recall"], 0)
        self.assertEqual(result["f1"], 0)
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["specificity"])
        self.assertEqual(result["average_precision"], 1)

    def test_invalid_inputs(self):
        cases = [([], [], None), ([0], [0, 1], None), ([2], [0], None),
                 ([0], [np.nan], None), ([[0]], [0], None), ([1], [0], []),
                 ([0], [0], [np.inf]), ([0], [0], [0, 1])]
        for truth, pred, score in cases:
            with self.subTest(case=(truth, pred, score)), self.assertRaises(ValueError):
                window_metrics(truth, pred, score)


class EventMetricTests(unittest.TestCase):
    def test_positive_run_starts_at_end_and_closes_on_negative_decision(self):
        alarms = alarm_episodes([0, 5, 10, 15], [5, 10, 15, 20], [1, 1, 0, 1],
                                recording_duration_sec=20)
        self.assertEqual(len(alarms), 2)
        self.assertEqual(alarms[0], {"alarm_id": 1, "start_sec": 5, "end_sec": 15,
                                    "last_positive_decision_sec": 10, "positive_windows": 2,
                                    "right_censored": False})
        self.assertEqual(alarms[1]["start_sec"], 20)
        self.assertEqual(alarms[1]["end_sec"], 20)
        self.assertTrue(alarms[1]["right_censored"])

    def test_future_predictions_cannot_shift_an_alarm_start(self):
        first = events([0, 1, 1, 0])["alarms"][0]["start_sec"]
        changed = events([0, 1, 0, 1])["alarms"][0]["start_sec"]
        self.assertEqual(first, 10)
        self.assertEqual(first, changed)

    def test_detection_miss_false_alarm_and_latency(self):
        # Alerts at 5, 20, 30 s. First is false; seizures at [18,24) and
        # [26,34) are detected; the short event [6,8) is missed.
        result = events([1, 0, 0, 1, 0, 1, 0, 0], [(6, 8), (18, 24), (26, 34)])
        self.assertEqual(result["n_detected"], 2)
        self.assertEqual(result["n_missed"], 1)
        self.assertEqual(result["n_false_alarms"], 1)
        self.assertEqual(result["false_alarms_per_hour"], 90)
        self.assertEqual(result["latencies_sec"], [2, 4])
        self.assertEqual(result["mean_latency_sec"], 3)
        self.assertEqual(result["median_latency_sec"], 3)
        self.assertAlmostEqual(result["event_sensitivity"], 2 / 3)
        self.assertIsNone(result["seizures"][0]["latency_sec"])

    def test_exact_onset_is_hit_exact_offset_is_not(self):
        hit = events([0, 1, 0], [(10, 15)])  # decision at 10
        miss = events([0, 0, 1], [(10, 15)])  # decision at 15
        self.assertEqual(hit["latencies_sec"], [0])
        self.assertEqual(miss["n_detected"], 0)
        self.assertEqual(miss["n_false_alarms"], 1)

    def test_short_seizure_ends_before_positive_window_decision(self):
        result = events([0, 1, 0], [(6, 8)])
        self.assertEqual(result["n_missed"], 1)
        self.assertEqual(result["n_false_alarms"], 1)
        self.assertEqual(result["alarms"][0]["start_sec"], 10)

    def test_ongoing_preonset_alarm_does_not_detect_seizure(self):
        result = events([1, 1, 1, 0], [(6, 14)])
        self.assertEqual(result["n_missed"], 1)
        self.assertEqual(result["n_false_alarms"], 1)

    def test_repeated_alarms_do_not_inflate_seizure_hits(self):
        result = events([1, 0, 1, 0, 1, 0], [(4, 26)])
        self.assertEqual(result["n_alarms"], 3)
        self.assertEqual(result["n_detected"], 1)
        self.assertEqual(result["n_repeat_alarms"], 2)
        self.assertEqual(result["n_false_alarms"], 0)
        self.assertEqual(result["latencies_sec"], [1])

    def test_one_sustained_alarm_cannot_detect_two_seizures(self):
        result = events([1, 1, 1, 1], [(4, 8), (12, 18)])
        self.assertEqual(result["n_detected"], 1)
        self.assertEqual(result["n_missed"], 1)

    def test_adjacent_seizures_boundary_belongs_to_second(self):
        result = events([0, 1, 0], [(5, 10), (10, 15)])
        self.assertFalse(result["seizures"][0]["detected"])
        self.assertTrue(result["seizures"][1]["detected"])

    def test_no_seizures_and_rearming_without_cooldown(self):
        result = events([1, 1, 0, 1, 0, 1])
        self.assertEqual(result["n_false_alarms"], 3)
        self.assertEqual(result["false_alarms_per_hour"], 360)
        self.assertEqual(result["n_detected"], 0)
        self.assertIsNone(result["event_sensitivity"])
        self.assertIsNone(result["mean_latency_sec"])
        self.assertIsNone(result["median_latency_sec"])

    def test_no_alarms_all_seizures_missed(self):
        result = events([0, 0, 0], [(1, 4), (6, 8)])
        self.assertEqual(result["n_missed"], 2)
        self.assertEqual(result["false_alarms_per_hour"], 0)
        self.assertEqual(result["latencies_sec"], [])
        json.dumps(result, allow_nan=False)

    def test_no_seizures_and_no_alarms(self):
        result = events([0, 0, 0])
        self.assertEqual(result["n_seizures"], 0)
        self.assertEqual(result["n_missed"], 0)
        self.assertEqual(result["n_alarms"], 0)
        self.assertEqual(result["false_alarms_per_hour"], 0)
        self.assertIsNone(result["event_sensitivity"])

    def test_invalid_timeline_cannot_inflate_exposure(self):
        cases = [([], [], [], 5),
                 ([0, 10], [5, 15], [0, 1], 15),  # gap
                 ([0, 4], [5, 10], [0, 1], 10),  # overlap
                 ([5, 0], [10, 5], [0, 1], 10),  # unsorted
                 ([5], [10], [0], 10),  # omitted beginning
                 ([0], [5], [0], 10),  # omitted end
                 ([0], [0], [0], 5), ([0], [5], [0], 0),
                 ([0], [5], [0], np.nan), ([0], [5], [0, 1], 5),
                 ([0], [np.inf], [0], 5), ([0], [5], [2], 5)]
        for starts, ends, pred, duration in cases:
            with self.subTest(case=(starts, ends, pred, duration)), self.assertRaises(ValueError):
                event_metrics(starts, ends, pred, [], recording_duration_sec=duration)

    def test_invalid_annotations(self):
        cases = [[(4, 4)], [(-1, 5)], [(1, 21)], [(8, 10), (2, 4)],
                 [(1, 8), (7, 10)], [(1, np.nan)], [1, 2], [[1, 2, 3]]]
        for seizures in cases:
            with self.subTest(seizures=seizures), self.assertRaises(ValueError):
                events([0, 1, 0, 0], seizures)


class RecordingAggregationTests(unittest.TestCase):
    def sample(self):
        # A ends positive and B begins positive: they must be separate alarms.
        frame = pd.DataFrame({"recording_id": ["a", "a", "b", "b", "b", "b"],
                              "start_sec": [0, 5, 0, 5, 10, 15],
                              "end_sec": [5, 10, 5, 10, 15, 20],
                              "label": [0, 1, 0, 0, 0, 0],
                              "prediction": [0, 1, 1, 0, 0, 0],
                              "score": [.1, .8, .7, .3, .2, .1]})
        metadata = {"a": {"duration_sec": 10, "seizures_sec": [(6, 10)]},
                    "b": {"duration_sec": 20, "seizures_sec": []}}
        return frame, metadata

    def test_recordings_reset_and_rates_pool_by_exposure(self):
        frame, metadata = self.sample()
        result = evaluate_predictions(frame, metadata)
        summary = result["event_metrics"]
        self.assertEqual(summary["n_alarms"], 2)
        self.assertEqual(summary["false_alarms_per_hour"], 240)  # 2 / (30/3600)
        self.assertEqual(summary["n_missed"], 1)
        self.assertEqual(result["per_recording"]["a"]["event_metrics"]["false_alarms_per_hour"], 360)
        self.assertEqual(result["per_recording"]["b"]["event_metrics"]["false_alarms_per_hour"], 180)
        self.assertEqual(result["window_metrics"]["confusion_matrix"], [[4, 1], [0, 1]])
        json.dumps(result, allow_nan=False)

    def test_pooled_latencies_and_recording_identity(self):
        frame, metadata = self.sample()
        metadata["a"]["seizures_sec"] = [(6, 10.0)]
        # Decision at 10 is still an offset miss, even for float annotations.
        self.assertIsNone(evaluate_predictions(frame, metadata)["event_metrics"]["mean_latency_sec"])
        metadata["b"]["seizures_sec"] = [(1, 8)]
        result = evaluate_predictions(frame, metadata)
        self.assertEqual(result["event_metrics"]["latencies_sec"], [4])
        self.assertEqual(result["event_metrics"]["n_detected"], 1)

    def test_optional_scores_and_exact_decision_times(self):
        frame, metadata = self.sample()
        frame["decision_time_sec"] = frame.end_sec
        result = evaluate_predictions(frame.drop(columns="score"), metadata)
        self.assertIsNone(result["window_metrics"]["average_precision"])
        frame.loc[0, "decision_time_sec"] = 0
        with self.assertRaisesRegex(ValueError, "actual window end"):
            evaluate_predictions(frame, metadata)

    def test_metadata_and_table_validation(self):
        frame, metadata = self.sample()
        cases = [(frame, {"a": metadata["a"]}), (frame, {**metadata, "c": metadata["a"]}),
                 (frame, {"a": {}, "b": metadata["b"]}),
                 (frame.drop(columns="label"), metadata), (frame.iloc[:0], metadata),
                 (frame.assign(recording_id=None), metadata),
                 (frame.assign(recording_id=""), metadata),
                 (frame, None), (frame, {"a": None, "b": metadata["b"]}),
                 (pd.concat([frame, frame[["recording_id"]]], axis=1), metadata)]
        for table, meta in cases:
            with self.subTest(columns=list(table), metadata=meta), self.assertRaises(ValueError):
                evaluate_predictions(table, meta)


if __name__ == "__main__":
    unittest.main()
