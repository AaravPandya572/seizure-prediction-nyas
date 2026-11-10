"""Hand-calculated descriptive periods and feature contrasts; no model fitting."""

import unittest
import pandas as pd

from src.failure_analysis import score_periods, feature_periods, describe_scores, feature_descriptions
from src.preprocess import FEATURE_COLUMNS
from src.rf_cross_validation import HOLDOUT_IDS


class FailureAnalysisTests(unittest.TestCase):
    def test_score_period_boundaries_are_disjoint_decision_times(self):
        frame = pd.DataFrame({'end_sec': [5, 10, 15, 20, 25, 80]})
        masks = score_periods(frame, 10, 20)
        self.assertEqual(frame.loc[masks['during'], 'end_sec'].tolist(), [10, 15])
        self.assertEqual(frame.loc[masks['post60'], 'end_sec'].tolist(), [20, 25])
        self.assertEqual(frame.loc[masks['remote'], 'end_sec'].tolist(), [80])
        self.assertTrue(pd.DataFrame(masks).sum(axis=1).eq(1).all())

    def test_feature_post_requires_whole_window_and_labels_are_preserved(self):
        frame = pd.DataFrame({'start_sec': [0, 5, 10, 15, 20], 'end_sec': [5, 10, 15, 20, 25], 'label': [0, 0, 1, 1, 0]})
        original = frame.copy()
        masks = feature_periods(frame, 8, 18)
        self.assertEqual(frame.loc[masks['ictal'], 'start_sec'].tolist(), [10, 15])
        self.assertEqual(frame.loc[masks['post60_full'], 'start_sec'].tolist(), [20])
        pd.testing.assert_frame_equal(frame, original)

    def test_score_summary_known_values_and_empty_period(self):
        frame = pd.DataFrame({'end_sec': [10, 15, 20], 'score': [.1, .3, .7], 'prediction': [0, 0, 1]})
        d = {r['period']: r for r in describe_scores(frame, 10, 20)}
        self.assertAlmostEqual(d['during']['mean'], .2)
        self.assertEqual(d['during']['peak_decisions_sec'], [15])
        self.assertEqual(d['post60']['positive_windows'], 1)
        self.assertIsNone(d['pre60']['max'])

    def test_feature_ratios_distinguish_baseline_scaling_from_relative_change(self):
        rows, metadata = [], {}
        for rid in HOLDOUT_IDS:
            factor = 2 if rid == 'chb01_15' else 1
            metadata[rid] = {'seizures_sec': [[95, 100]]}
            for start, label, value in ((10, 0, 1), (95, 1, 3), (150, 0, 2)):
                rows.append({'recording_id': rid, 'start_sec': start, 'end_sec': start+5, 'label': label,
                             **{f: factor * value for f in FEATURE_COLUMNS}})
        distributions, families, differences = feature_descriptions(pd.DataFrame(rows), metadata)
        self.assertEqual(len(distributions), 4 * 3 * 126)
        self.assertEqual(len(families), 4 * 3 * 7)
        self.assertTrue(differences.target_ictal_over_remote.eq(3).all())
        self.assertTrue(differences.other_event_median_ratio.eq(3).all())
        self.assertTrue(differences.difference_of_changes.eq(2).all())


if __name__ == '__main__':
    unittest.main()
