"""Scientific invariants; unittest avoids adding a test dependency."""
import contextlib
import io
import tempfile
from pathlib import Path
import unittest
import warnings
from unittest.mock import patch

import numpy as np

from src import preprocess as p


class AnnotationTests(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.txt"
            path.write_text(text)
            return p.parse_summary_file(path)

    def test_numbered_multiple_and_seizure_free_files(self):
        result = self.parse("""File Name: chb01_03.edf
Number of Seizures in File: 2
Seizure 1 Start Time: 5 seconds
Seizure 1 End Time: 10 seconds
Seizure 2 Start Time: 20 seconds
Seizure 2 End Time: 25 seconds
File Name: chb01_01.edf
Number of Seizures in File: 0
""")
        self.assertEqual(result, {"chb01_03.edf": [(5, 10), (20, 25)], "chb01_01.edf": []})

    def test_real_annotations(self):
        result = p.parse_summary_file(p.PROJECT_ROOT / "data/raw/chb01-summary.txt")
        self.assertEqual(result["chb01_01.edf"], [])
        self.assertEqual(result["chb01_03.edf"], [(2996, 3036)])
        self.assertEqual(result["chb01_04.edf"], [(1467, 1494)])

    def test_malformed_annotations_fail(self):
        cases = [
            "Number of Seizures in File: 1\nSeizure Start Time: 5 seconds",
            "Number of Seizures in File: 0\nSeizure End Time: 5 seconds",
            "Number of Seizures in File: 1\nSeizure 1 Start Time: 5 seconds\nSeizure 2 End Time: 10 seconds",
            "Number of Seizures in File: 1\nSeizure Start Time: 10 seconds\nSeizure End Time: 5 seconds",
            "Number of Seizures in File: 2\nSeizure Start Time: 5 seconds\nSeizure End Time: 10 seconds",
            "Number of Seizures in File: 0\nFile Name: chb01_03.edf\nNumber of Seizures in File: 0",
            "File Start Time: 00:00:00",
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.parse("File Name: chb01_03.edf\n" + case)

    def test_interval_validation(self):
        for intervals in [[(-1, 2)], [(1, 1)], [(5, 10), (9, 12)], [(10, 11), (1, 2)], [(0, np.nan)], [(0, 21)]]:
            with self.subTest(intervals=intervals), self.assertRaises(ValueError):
                p.validate_seizures(intervals, 20)

    def test_half_open_boundaries_and_exact_half(self):
        self.assertEqual(p.window_annotation(0, 5, [(5, 10)]), (0, 0, []))
        self.assertEqual(p.window_annotation(10, 15, [(5, 10)]), (0, 0, []))
        self.assertEqual(p.window_annotation(0, 5, [(2.5, 10)]), (1, 2.5, [1]))
        self.assertEqual(p.window_annotation(0, 5, [(2.6, 10)])[0], 0)

    def test_multiple_seizures_overlap_union(self):
        self.assertEqual(p.window_annotation(0, 5, [(0, 1), (3, 5)]), (1, 3, [1, 2]))

    def test_known_boundary_windows_and_counts(self):
        for intervals, expected in [([], 0), ([(2996, 3036)], 8), ([(1467, 1494)], 6)]:
            self.assertEqual(sum(p.window_annotation(t, t + 5, intervals)[0] for t in range(0, 3600, 5)), expected)
        self.assertEqual(p.window_annotation(3035, 3040, [(2996, 3036)]), (0, 1, [1]))
        self.assertEqual(p.window_annotation(1490, 1495, [(1467, 1494)]), (1, 4, [1]))


class SignalTests(unittest.TestCase):
    def test_channel_order_and_explicit_duplicate_alias(self):
        with self.assertWarnsRegex(UserWarning, "first duplicate"):
            result = p.select_channels([c for c in reversed(p.CHANNELS) if c != "T8-P8"] + ["T8-P8-0", "T8-P8-1"])
        self.assertEqual(result, ["T8-P8-0" if c == "T8-P8" else c for c in p.CHANNELS])
        with self.assertRaises(ValueError):
            p.select_channels(list(p.CHANNELS[1:]))
        with self.assertRaises(ValueError):
            p.select_channels(list(p.CHANNELS) + ["FP1-F7"])

    def test_band_powers_partition_and_known_alpha_signal(self):
        t = np.arange(1280) / 256
        epoch = np.tile(np.sin(2 * np.pi * 10 * t), (18, 1))
        features, low_power = p.extract_epoch_features(epoch, 256)
        matrix = features.reshape(18, 7)
        np.testing.assert_allclose(matrix[:, 2:].sum(axis=1), 1, atol=1e-12)
        self.assertTrue((matrix[:, 4] > .99).all())  # alpha
        np.testing.assert_allclose(matrix[:, 0], .5, atol=1e-12)
        self.assertEqual(low_power, 0)

    def test_flat_and_near_flat_channels_remain_finite(self):
        for scale in (0, 1e-10):
            features, count = p.extract_epoch_features(np.ones((18, 1280)) * scale, 256)
            self.assertTrue(np.isfinite(features).all())
            self.assertEqual(count, 18)
            np.testing.assert_array_equal(features.reshape(18, 7)[:, 2:], 0)

    def test_invalid_windows_fail(self):
        for data in [np.zeros((18, 1279)), np.zeros((17, 1280)), np.full((18, 1280), np.nan)]:
            with self.assertRaises(ValueError):
                p.extract_epoch_features(data, 256)

    def test_filter_does_not_use_future_samples_and_resets(self):
        rng = np.random.default_rng(42)
        data = rng.normal(size=(18, 2560))
        changed = data.copy()
        changed[:, 1280:] += 100
        np.testing.assert_array_equal(p.causal_bandpass(data, 256)[:, :1280],
                                      p.causal_bandpass(changed, 256)[:, :1280])
        np.testing.assert_array_equal(p.causal_bandpass(np.zeros_like(data), 256), 0)
        with self.assertRaises(ValueError):
            p.causal_bandpass(data, 80)

    def test_exact_length_and_partial_tail(self):
        bounds = p.window_bounds(3600 * 256, 256)
        self.assertEqual(len(bounds), 720)
        self.assertEqual(bounds[0], (0, 1280))
        self.assertEqual(bounds[-1], (920320, 921600))
        self.assertTrue(all(a[1] == b[0] for a, b in zip(bounds, bounds[1:])))
        with self.assertWarnsRegex(UserWarning, "Dropping 1"):
            self.assertEqual(len(p.window_bounds(1281, 256)), 1)
        with self.assertRaises(ValueError):
            p.window_bounds(1279, 256)

    def test_invalid_sampling_rates(self):
        for sfreq in (np.nan, np.inf, 0, 80, 256.01):
            with self.subTest(sfreq=sfreq), self.assertRaises(ValueError):
                p.window_bounds(1280, sfreq)

    def test_completed_edf_header_lengths(self):
        # An explicit allowlist ensures this test never touches active downloads.
        for filename in ("chb01_01.edf", "chb01_03.edf", "chb01_04.edf"):
            with self.subTest(filename=filename), warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Channel names are not unique.*", category=RuntimeWarning)
                with p.mne.io.read_raw_edf(p.PROJECT_ROOT / "data/raw" / filename,
                                           preload=False, verbose="WARNING") as raw:
                    self.assertEqual(raw.info["sfreq"], 256)
                    self.assertEqual(raw.n_times, 921600)
                    self.assertEqual(raw.n_times / raw.info["sfreq"], 3600)

    def test_rows_preserve_full_stream_and_metadata(self):
        with patch.object(p, "load_and_filter_edf", return_value=(np.zeros((18, 3840)), 256, list(p.CHANNELS))):
            with self.assertWarnsRegex(UserWarning, "near-zero-power"):
                frame, metadata = p.slice_and_extract("chb01_03.edf", [(5, 10)])
        self.assertEqual(frame.label.tolist(), [0, 1, 0])
        self.assertEqual(frame.start_sec.tolist(), [0, 5, 10])
        self.assertEqual(frame.end_sec.tolist(), [5, 10, 15])
        self.assertEqual(frame.recording_id.unique().tolist(), ["chb01_03"])
        self.assertEqual(frame.patient_id.unique().tolist(), ["chb01"])
        self.assertEqual(metadata["n_samples"], 3840)
        self.assertTrue(frame.loc[2, "postictal_60s"])
        self.assertEqual(frame.decision_time_sec.tolist(), frame.end_sec.tolist())

    def test_out_of_recording_annotations_fail(self):
        with patch.object(p, "load_and_filter_edf", return_value=(np.zeros((18, 1280)), 256, list(p.CHANNELS))):
            with self.assertRaises(ValueError):
                p.slice_and_extract("chb01_03.edf", [(4, 6)])

    def test_legacy_output_is_protected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            p.main(["--output", str(p.PROJECT_ROOT / "data/processed/chb01_features.parquet")])


if __name__ == "__main__":
    unittest.main()
