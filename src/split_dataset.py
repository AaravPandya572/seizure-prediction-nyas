"""Verify raw/processed EEG integrity and attach explicit recording-level splits.

Run as `venv/bin/python -m src.split_dataset`. This performs data checks only;
the reserved test recordings are never used for model selection or fitting.
Preprocessing remains in src/preprocess.py and is invoked separately.
"""

import argparse
import json
from pathlib import Path
import re
import warnings

import mne
import numpy as np
import pandas as pd

from .preprocess import (CHANNELS, FEATURE_COLUMNS, FILTER_WARMUP_SEC,
                         PROCESSING_VERSION, PROJECT_ROOT, WINDOW_SEC,
                         parse_summary_file, select_channels, sha256, validate_seizures)

SPLITS = ("train", "validation", "test")


def load_manifest(path):
    """One row per EDF; repeated filenames are errors even within one split."""
    manifest = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(manifest.columns) != ["filename", "split", "annotation_source"] or manifest.empty:
        raise ValueError("Manifest needs filename,split,annotation_source columns and at least one row")
    if manifest.filename.duplicated().any():
        raise ValueError("A recording may occur only once in the manifest (no split overlap)")
    if set(manifest.split) != set(SPLITS):
        raise ValueError("Manifest must contain only train, validation, test and include all three")
    if not manifest.filename.str.fullmatch(r"chb\d+_\d+[a-z]?\.edf").all():
        raise ValueError("Manifest filenames must be CHB-MIT EDF basenames")
    if not manifest.annotation_source.map(lambda s: bool(s.strip())).all():
        raise ValueError("Every recording needs an explicit annotation source")
    return manifest


def _summary_metadata(path):
    """Read clocks only for duration/header checks; seizure parsing stays shared."""
    text = Path(path).read_text()
    rate = re.search(r"Data Sampling Rate:\s*([\d.]+) Hz", text)
    if rate is None:
        raise ValueError(f"No sampling rate in {path}")
    records, current = {}, None
    for line in text.splitlines():
        if line.startswith("File Name:"):
            current = line.split(":", 1)[1].strip()
            records[current] = {}
        elif current and line.startswith(("File Start Time:", "File End Time:")):
            key, clock = line.split(":", 1)
            h, m, s = map(int, clock.strip().split(":"))
            records[current][key] = h * 3600 + m * 60 + s
    return float(rate.group(1)), records


def _check_edf_size(path):
    """Check declared EDF payload length so a truncated download cannot pass.

    EDF has a 256-byte fixed header, followed by per-signal fields; each sample
    is two bytes. The samples-per-record field follows 216 bytes per signal.
    """
    with Path(path).open("rb") as stream:
        header = stream.read(256)
        header_bytes = int(header[184:192])
        n_records = int(header[236:244])
        n_channels = int(header[252:256])
        stream.seek(256 + 216 * n_channels)
        samples = [int(stream.read(8)) for _ in range(n_channels)]
    if n_records <= 0 or n_channels <= 0 or header_bytes != 256 * (n_channels + 1):
        raise ValueError(f"Unsupported or invalid EDF header: {path}")
    expected_bytes = header_bytes + n_records * sum(samples) * 2
    if Path(path).stat().st_size != expected_bytes:
        raise ValueError(f"EDF payload length mismatch (possibly incomplete): {path}")


def verify_raw_recordings(manifest, root=PROJECT_ROOT):
    """Read only manifest EDFs; compare headers/annotations and return hashes."""
    summaries, verified, messages = {}, {}, []
    reference_channels = None
    for row in manifest.itertuples(index=False):
        source = Path(root) / row.annotation_source
        if row.annotation_source not in summaries:
            summaries[row.annotation_source] = (parse_summary_file(source), *_summary_metadata(source), sha256(source))
        annotations, expected_rate, clocks, summary_hash = summaries[row.annotation_source]
        if row.filename not in annotations or row.filename not in clocks:
            raise ValueError(f"Missing summary entry: {row.filename}")
        path = Path(root) / "data/raw" / row.filename
        _check_edf_size(path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with mne.io.read_raw_edf(path, preload=False, verbose="WARNING") as raw:
                selected = select_channels(raw.ch_names)
                sfreq, n_samples = float(raw.info["sfreq"]), int(raw.n_times)
                duration = n_samples / sfreq
                clock = clocks[row.filename]
                expected_duration = (clock["File End Time"] - clock["File Start Time"]) % 86400
                if sfreq != expected_rate or duration != expected_duration:
                    raise ValueError(f"EDF rate/duration disagrees with summary: {row.filename}")
                date = raw.info["meas_date"]
                if date is None or date.hour * 3600 + date.minute * 60 + date.second != clock["File Start Time"] % 86400:
                    raise ValueError(f"EDF start clock disagrees with summary: {row.filename}")
                if reference_channels is None:
                    reference_channels = raw.ch_names
                if raw.ch_names != reference_channels:
                    raise ValueError(f"EDF channel names/order differ: {row.filename}")
                # Probe both ends without loading the full signal a second time.
                for start in (0, n_samples - 1):
                    if not np.isfinite(raw.get_data(picks=selected, start=start, stop=start + 1)).all():
                        raise ValueError(f"Nonfinite EDF boundary sample: {row.filename}")
                validate_seizures(annotations[row.filename], duration)
                verified[row.filename] = {
                    "duration_sec": duration, "n_samples": n_samples, "sampling_rate_hz": sfreq,
                    "selected_edf_channels": selected, "raw_channels": raw.ch_names,
                    "seizures_sec": [list(pair) for pair in annotations[row.filename]],
                    "annotation_source": row.annotation_source, "annotation_sha256": summary_hash,
                    "sha256": sha256(path),
                }
        messages.extend(f"{row.filename}: {w.message}" for w in caught)
    return verified, list(dict.fromkeys(messages))


def assign_splits(frame, manifest):
    """Attach a split by filename without shuffling, dropping, or duplicating rows."""
    if "split" in frame:
        raise ValueError("Input already has splits; use the original expanded feature table")
    if set(frame.source_file) != set(manifest.filename):
        raise ValueError("Processed recordings must match the manifest exactly")
    if manifest.filename.duplicated().any():
        raise ValueError("Duplicate recording in manifest")
    if frame.groupby("source_file").recording_id.nunique().ne(1).any():
        raise ValueError("A source file maps to multiple recording IDs")
    if frame.groupby("recording_id").source_file.nunique().ne(1).any():
        raise ValueError("A recording ID maps to multiple source files")
    result = frame.copy()
    result["split"] = result.source_file.map(manifest.set_index("filename").split)
    if result.groupby("recording_id").split.nunique().ne(1).any():
        raise ValueError("Recording overlaps splits")
    return result


def verify_processed(frame, provenance, raw):
    """Independently reconstruct labels/times; never estimate counts from hours."""
    required = {"source_file", "recording_id", "patient_id", "start_sample", "end_sample",
                "start_sec", "end_sec", "decision_time_sec", "label", "sampling_rate_hz",
                "recording_duration_sec", "seizure_overlap_sec", "seizure_overlap_fraction",
                "overlapping_seizure_ids", "processing_version", "filter_warmup", "low_power_channel_count"}
    if not frame.columns.is_unique or not required <= set(frame):
        raise ValueError("Missing or duplicated processed metadata columns")
    if [c for c in frame if c in FEATURE_COLUMNS] != FEATURE_COLUMNS:
        raise ValueError("Expected all 126 canonical feature columns in channel order")
    if frame.isna().any().any() or not np.isfinite(frame[FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("NaN or infinite features/metadata")
    if set(frame.source_file) != set(raw) or set(provenance["recordings"]) != set(raw):
        raise ValueError("Raw, processed, and provenance recording sets differ")
    if frame.duplicated(["source_file", "start_sample"]).any():
        raise ValueError("Duplicate windows")
    if (provenance["channels"] != list(CHANNELS) or provenance["window_sec"] != WINDOW_SEC
            or provenance["processing_version"] != PROCESSING_VERSION):
        raise ValueError("Preprocessing settings disagree with the established pipeline")
    per_recording = []
    for filename, group in frame.groupby("source_file", sort=False):
        metadata, saved = raw[filename], provenance["recordings"][filename]
        for key in ("duration_sec", "n_samples", "sampling_rate_hz", "selected_edf_channels", "seizures_sec", "sha256"):
            if saved[key] != metadata[key]:
                raise ValueError(f"Provenance/raw mismatch: {filename}, {key}")
        size = round(WINDOW_SEC * metadata["sampling_rate_hz"])
        starts = np.arange(0, metadata["n_samples"] - size + 1, size)
        ends = starts + size
        if len(group) != len(starts):
            raise ValueError(f"Incorrect number of windows: {filename}")
        expected = {"start_sample": starts, "end_sample": ends,
                    "start_sec": starts / metadata["sampling_rate_hz"],
                    "end_sec": ends / metadata["sampling_rate_hz"],
                    "decision_time_sec": ends / metadata["sampling_rate_hz"]}
        for key, values in expected.items():
            if not np.array_equal(group[key].to_numpy(), values):
                raise ValueError(f"Invalid timeline/order: {filename}, {key}")
        for key, value in (("sampling_rate_hz", metadata["sampling_rate_hz"]),
                           ("recording_duration_sec", metadata["duration_sec"]),
                           ("recording_id", Path(filename).stem), ("patient_id", filename.split("_")[0]),
                           ("processing_version", PROCESSING_VERSION)):
            if not group[key].eq(value).all():
                raise ValueError(f"Invalid {key}: {filename}")
        overlap = np.zeros(len(group))
        ids = [[] for _ in range(len(group))]
        for i, (onset, offset) in enumerate(metadata["seizures_sec"], 1):
            amount = np.maximum(0, np.minimum(expected["end_sec"], offset) - np.maximum(expected["start_sec"], onset))
            overlap += amount
            for position in np.flatnonzero(amount > 0):
                ids[position].append(i)
        if (not np.array_equal(group.label.to_numpy(), (overlap >= WINDOW_SEC / 2).astype(int))
                or not np.allclose(group.seizure_overlap_sec, overlap, rtol=0, atol=1e-12)
                or not np.allclose(group.seizure_overlap_fraction, overlap / WINDOW_SEC, rtol=0, atol=1e-12)
                or group.overlapping_seizure_ids.map(json.loads).tolist() != ids):
            raise ValueError(f"Incorrect labels or seizure overlap: {filename}")
        if not np.array_equal(group.filter_warmup, expected["start_sec"] < FILTER_WARMUP_SEC):
            raise ValueError(f"Incorrect filter warmup flags: {filename}")
        power = group[FEATURE_COLUMNS].to_numpy().reshape(len(group), len(CHANNELS), 7)[:, :, 2:]
        totals = power.sum(axis=2)
        if ((power < 0).any() or (power > 1).any()
                or not np.all(np.isclose(totals, 1, atol=1e-12, rtol=0) | (totals == 0))
                or not np.array_equal((totals == 0).sum(axis=1), group.low_power_channel_count)):
            raise ValueError(f"Invalid relative spectral powers: {filename}")
        per_recording.append({"filename": filename, "duration_sec": metadata["duration_sec"],
                              "windows": len(group), "class_0": int((group.label == 0).sum()),
                              "class_1": int(group.label.sum()), "seizure_events": len(metadata["seizures_sec"]),
                              "trailing_samples_dropped": int(metadata["n_samples"] % size)})
    return per_recording


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "config/chb01_recordings.csv")
    parser.add_argument("--features", type=Path, default=PROJECT_ROOT / "data/processed/chb01_detection_features_11recordings.parquet")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/processed/chb01_detection_features_11recordings_split.parquet")
    args = parser.parse_args(argv)
    # A new output is required so no old data or input can be silently replaced.
    if args.output.suffix != ".parquet" or args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("Choose a new .parquet output path (neither it nor its JSON may already exist)")
    manifest = load_manifest(args.manifest)
    raw, messages = verify_raw_recordings(manifest)
    frame = pd.read_parquet(args.features)
    provenance_path = args.features.with_suffix(".json")
    provenance = json.loads(provenance_path.read_text())
    records = verify_processed(frame, provenance, raw)
    result = assign_splits(frame, manifest)
    counts = pd.DataFrame(records).merge(manifest, on="filename", validate="one_to_one")
    summary = counts.groupby("split", sort=False)[["windows", "class_0", "class_1", "seizure_events", "duration_sec"]].sum().reindex(SPLITS)
    summary["recordings"] = counts.groupby("split").size()
    report = {"manifest": str(args.manifest), "manifest_sha256": sha256(args.manifest),
              "input_sha256": sha256(args.features), "preprocessing_provenance_sha256": sha256(provenance_path),
              "raw_recordings": raw, "per_recording": counts.to_dict(orient="records"),
              "per_split": summary.reset_index().to_dict(orient="records"),
              "warnings": messages, "preprocessing_warnings": provenance["warnings"],
              "feature_columns": FEATURE_COLUMNS, "test_policy": "reserved for final evaluation; integrity and preprocessing only",
              "checks": "EDF lengths/clocks/rates/channels/annotations, finite features, complete fixed-grid windows, labels/overlaps, feature order, provenance hashes, unique windows, exclusive recording splits"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    report["output_sha256"] = sha256(args.output)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(counts[["filename", "split", "duration_sec", "windows", "class_0", "class_1", "seizure_events"]].to_string(index=False))
    print("\nSplit summary:\n", summary.to_string())
    print(f"\nVerified and saved {args.output}; raw EDFs and input features were read only.")


if __name__ == "__main__":
    main()
