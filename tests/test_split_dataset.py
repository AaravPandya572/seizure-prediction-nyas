"""Split safeguards and expanded-data checks; no model training."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src import preprocess as p
from src.split_dataset import (_check_edf_size, _summary_metadata, assign_splits,
                               load_manifest, verify_processed)

MANIFEST = p.PROJECT_ROOT / "config/chb01_recordings.csv"
FEATURES = p.PROJECT_ROOT / "data/processed/chb01_detection_features_11recordings.parquet"
SPLIT_DATA = p.PROJECT_ROOT / "data/processed/chb01_detection_features_11recordings_split.parquet"
EXPECTED_SPLITS = {"train": (1, 3, 4, 14, 15, 16), "validation": (17, 18), "test": (21, 22, 26)}


class ManifestTests(unittest.TestCase):
    def parse(self, content):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            path.write_text(content)
            return load_manifest(path)

    def test_fixed_requested_membership(self):
        manifest = load_manifest(MANIFEST)
        self.assertEqual(len(manifest), 11)
        for split, numbers in EXPECTED_SPLITS.items():
            self.assertEqual(set(manifest.loc[manifest.split == split, "filename"]),
                             {f"chb01_{n:02d}.edf" for n in numbers})
        self.assertEqual(manifest.annotation_source.unique().tolist(), ["data/raw/chb01-summary.txt"])

    def test_conflicts_and_invalid_manifest_fail(self):
        base = ("filename,split,annotation_source\nchb01_01.edf,train,summary.txt\n"
                "chb01_18.edf,validation,summary.txt\nchb01_26.edf,test,summary.txt\n")
        cases = [base + "chb01_01.edf,test,summary.txt\n",
                 base + "chb01_01.edf,train,summary.txt\n",
                 base.replace(",test,", ",unknown,"),
                 base.replace(",test,summary.txt", ",test,"),
                 base.replace("chb01_26.edf", "../chb01_26.edf"),
                 base.replace("filename,split,annotation_source", "filename,split,other")]
        for content in cases:
            with self.subTest(content=content), self.assertRaises(ValueError):
                self.parse(content)

    def test_assign_splits_preserves_rows_and_prevents_recording_overlap(self):
        manifest = load_manifest(MANIFEST)
        frame = pd.DataFrame({"source_file": manifest.filename,
                              "recording_id": manifest.filename.map(lambda f: Path(f).stem),
                              "label": [0] * 11})
        result = assign_splits(frame, manifest)
        pd.testing.assert_frame_equal(result.drop(columns="split"), frame)
        self.assertTrue(result.groupby("recording_id").split.nunique().eq(1).all())
        groups = [set(result.loc[result.split == split, "recording_id"]) for split in EXPECTED_SPLITS]
        self.assertTrue(all(not a & b for i, a in enumerate(groups) for b in groups[i + 1:]))
        with self.assertRaises(ValueError):
            assign_splits(result, manifest)
        with self.assertRaises(ValueError):
            assign_splits(frame.iloc[:-1], manifest)
        bad = frame.copy()
        bad.loc[1, "recording_id"] = bad.loc[0, "recording_id"]
        with self.assertRaises(ValueError):
            assign_splits(bad, manifest)


class IntegrityTests(unittest.TestCase):
    def synthetic(self):
        """A complete 15-second stream with a boundary-crossing seizure."""
        filename = "chb01_03.edf"
        base_features = dict(zip(p.FEATURE_COLUMNS, np.tile([1, 2, .2, .2, .2, .2, .2], 18)))
        rows = []
        for start, label, overlap, ids in ((0, 0, 0, []), (5, 1, 4, [1]), (10, 0, 2, [1])):
            rows.append({**base_features, "source_file": filename, "recording_id": "chb01_03",
                         "patient_id": "chb01", "start_sample": start * 256,
                         "end_sample": (start + 5) * 256, "start_sec": start, "end_sec": start + 5,
                         "decision_time_sec": start + 5, "label": label, "sampling_rate_hz": 256,
                         "recording_duration_sec": 15, "seizure_overlap_sec": overlap,
                         "seizure_overlap_fraction": overlap / 5, "overlapping_seizure_ids": json.dumps(ids),
                         "processing_version": p.PROCESSING_VERSION, "filter_warmup": start < 10,
                         "low_power_channel_count": 0})
        metadata = {"duration_sec": 15, "n_samples": 3840, "sampling_rate_hz": 256,
                    "selected_edf_channels": list(p.CHANNELS), "seizures_sec": [[6, 12]], "sha256": "synthetic"}
        provenance = {"recordings": {filename: dict(metadata)}, "channels": list(p.CHANNELS),
                      "window_sec": 5, "processing_version": p.PROCESSING_VERSION}
        return pd.DataFrame(rows), provenance, {filename: metadata}

    def test_known_labels_and_counts(self):
        frame, provenance, raw = self.synthetic()
        counts = verify_processed(frame, provenance, raw)
        self.assertEqual((counts[0]["windows"], counts[0]["class_0"], counts[0]["class_1"], counts[0]["seizure_events"]),
                         (3, 2, 1, 1))

    def test_incorrect_or_missing_windows_and_metadata_fail(self):
        frame, provenance, raw = self.synthetic()
        changes = [("label", 0), ("seizure_overlap_sec", 0), ("overlapping_seizure_ids", "[]"),
                   ("end_sec", 11), ("decision_time_sec", 9), ("start_sample", 0),
                   ("recording_id", "other"), ("patient_id", "chb02"), ("sampling_rate_hz", 128),
                   ("recording_duration_sec", 3600), ("low_power_channel_count", 18)]
        for column, value in changes:
            bad = frame.copy()
            bad.loc[1, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                verify_processed(bad, provenance, raw)
        for bad in (frame.iloc[:-1], pd.concat([frame, frame.iloc[:1]], ignore_index=True),
                    frame.iloc[::-1], frame.drop(columns=p.FEATURE_COLUMNS[0])):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                verify_processed(bad, provenance, raw)

    def test_nonfinite_features_and_wrong_channel_order_fail(self):
        frame, provenance, raw = self.synthetic()
        for value in (np.nan, np.inf):
            bad = frame.copy()
            bad.loc[0, p.FEATURE_COLUMNS[0]] = value
            with self.assertRaises(ValueError):
                verify_processed(bad, provenance, raw)
        columns = list(frame.columns)
        columns[0], columns[7] = columns[7], columns[0]
        with self.assertRaises(ValueError):
            verify_processed(frame[columns], provenance, raw)
        bad = frame.copy()
        bad.loc[0, p.FEATURE_COLUMNS[2]] = .9
        with self.assertRaises(ValueError):
            verify_processed(bad, provenance, raw)

    def test_raw_provenance_mismatch_fails(self):
        frame, provenance, raw = self.synthetic()
        raw["chb01_03.edf"]["sha256"] = "changed"
        with self.assertRaises(ValueError):
            verify_processed(frame, provenance, raw)

    def test_short_recording_sample_bounds(self):
        bounds = p.window_bounds(595200, 256)
        self.assertEqual(len(bounds), 465)
        self.assertEqual(bounds[-1], (2320 * 256, 2325 * 256))
        self.assertEqual(sum(p.window_annotation(a / 256, b / 256, [(1862, 1963)])[0]
                             for a, b in bounds), 21)

    def test_edf_payload_size_detects_truncation(self):
        header = bytearray(b" " * 512)
        header[184:192], header[236:244], header[252:256] = b"512     ", b"2       ", b"1   "
        header[472:480] = b"256     "
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.edf"
            path.write_bytes(header + bytes(2 * 256 * 2))
            _check_edf_size(path)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "payload length mismatch"):
                _check_edf_size(path)

    def test_summary_short_duration_and_extended_clock_hours(self):
        sfreq, records = _summary_metadata(p.PROJECT_ROOT / "data/raw/chb01-summary.txt")
        self.assertEqual(sfreq, 256)
        for filename, duration in (("chb01_13.edf", 3600), ("chb01_26.edf", 2325)):
            times = records[filename]
            self.assertEqual((times["File End Time"] - times["File Start Time"]) % 86400, duration)


@unittest.skipUnless(FEATURES.exists() and SPLIT_DATA.exists(), "Run expansion and split CLIs for artifact integration checks")
class ExpandedArtifactTests(unittest.TestCase):
    def test_all_recordings_labels_features_and_splits(self):
        frame, split_frame = pd.read_parquet(FEATURES), pd.read_parquet(SPLIT_DATA)
        provenance = json.loads(FEATURES.with_suffix(".json").read_text())
        audit = json.loads(SPLIT_DATA.with_suffix(".json").read_text())
        counts = verify_processed(frame, provenance, audit["raw_recordings"])
        manifest = load_manifest(MANIFEST)
        pd.testing.assert_frame_equal(split_frame, assign_splits(frame, manifest))
        self.assertEqual(len(frame), 7665)
        self.assertEqual(int(frame.label.sum()), 90)
        expected = {1: (720, 0, 0), 3: (720, 8, 1), 4: (720, 6, 1), 14: (720, 0, 0),
                    15: (720, 8, 1), 16: (720, 10, 1), 17: (720, 0, 0), 18: (720, 18, 1),
                    21: (720, 19, 1), 22: (720, 0, 0), 26: (465, 21, 1)}
        for record in counts:
            number = int(Path(record["filename"]).stem.split("_")[1])
            self.assertEqual((record["windows"], record["class_1"], record["seizure_events"]), expected[number])
            self.assertEqual(record["trailing_samples_dropped"], 0)
        self.assertEqual(split_frame.groupby("split").label.agg(["count", "sum"]).to_dict("index"),
                         {"train": {"count": 4320, "sum": 32}, "validation": {"count": 1440, "sum": 18},
                          "test": {"count": 1905, "sum": 40}})
        self.assertEqual(p.sha256(FEATURES), audit["input_sha256"])
        self.assertEqual(p.sha256(SPLIT_DATA), audit["output_sha256"])
        self.assertEqual(p.sha256(MANIFEST), audit["manifest_sha256"])


if __name__ == "__main__":
    unittest.main()
