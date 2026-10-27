"""Extract patient-specific seizure DETECTION features from explicitly selected EDFs.

Labels describe [start, end) windows: >=50% annotated ictal overlap is positive.
The original preictal pipeline is preserved in Git commit 4454197.
Run from the repository root: venv/bin/python src/preprocess.py --help
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import re
import warnings

import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, welch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDINGS = ("chb01_01.edf", "chb01_03.edf", "chb01_04.edf")
WINDOW_SEC = 5.0
FILTER_WARMUP_SEC = 10.0  # Flag, not discard: this is a pragmatic QC interval.
POWER_FLOOR_UV2 = 1e-12
PROCESSING_VERSION = "detection-v1"
CHANNELS = (
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3",
    "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FZ-CZ", "CZ-PZ",
)
FREQ_BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
              "beta": (13, 30), "gamma": (30, 45)}
STAT_NAMES = ("var", "line_length", *FREQ_BANDS)
FEATURE_COLUMNS = [f"{ch}_{stat}" for ch in CHANNELS for stat in STAT_NAMES]


def validate_seizures(intervals, duration_sec=None):
    """Reject malformed annotations rather than silently creating normal labels."""
    previous_end = -1.0
    for start, end in intervals:
        if not np.isfinite([start, end]).all() or not 0 <= start < end:
            raise ValueError(f"Invalid seizure interval: {(start, end)}")
        if start < previous_end:
            raise ValueError("Seizure intervals must be ordered and nonoverlapping")
        if duration_sec is not None and end > duration_sec:
            raise ValueError(f"Seizure ends at {end}, beyond recording length {duration_sec}")
        previous_end = end


def parse_summary_file(summary_path):
    """Read unnumbered or numbered seizure pairs, checking declared event counts."""
    seizures = {}
    filename, count, pending = None, None, None

    def finish_recording():
        if filename is not None:
            if pending is not None or count is None or len(seizures[filename]) != count:
                raise ValueError(f"Incomplete annotations/count mismatch for {filename}")
            validate_seizures(seizures[filename])

    for raw_line in Path(summary_path).read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("File Name:"):
            finish_recording()
            filename = line.split(":", 1)[1].strip()
            if not filename or filename in seizures:
                raise ValueError(f"Missing or duplicate recording name: {filename}")
            seizures[filename] = []
            count, pending = None, None
        elif line.startswith("Number of Seizures in File:"):
            if filename is None or count is not None:
                raise ValueError("Misplaced or repeated seizure count")
            count = int(line.split(":", 1)[1])
            if count < 0:
                raise ValueError("Negative seizure count")
        elif line.startswith("Seizure") and ("Start Time:" in line or "End Time:" in line):
            match = re.fullmatch(r"Seizure(?: (\d+))? (Start|End) Time: (\d+(?:\.\d+)?) seconds", line)
            if filename is None or match is None:
                raise ValueError(f"Malformed annotation: {line}")
            number, kind, seconds = match.groups()
            if kind == "Start":
                if pending is not None:
                    raise ValueError(f"Unpaired seizure start in {filename}")
                pending = (number, float(seconds))
            else:
                if pending is None or pending[0] != number:
                    raise ValueError(f"Unpaired/mismatched seizure end in {filename}")
                seizures[filename].append((pending[1], float(seconds)))
                pending = None
    finish_recording()
    if not seizures:
        raise ValueError("Summary contains no recording annotations")
    return seizures


def select_channels(names):
    """Use a fixed bipolar order; explicitly resolve CHB-MIT's duplicate T8-P8."""
    selected = []
    for channel in CHANNELS:
        if names.count(channel) == 1:
            selected.append(channel)
        elif channel == "T8-P8" and "T8-P8-0" in names and "T8-P8-1" in names:
            # MNE numbers duplicate EDF labels in original order. Keep the first.
            warnings.warn("Using first duplicate T8-P8 signal (T8-P8-0)", UserWarning)
            selected.append("T8-P8-0")
        else:
            raise ValueError(f"Missing or ambiguous required bipolar channel: {channel}")
    return selected


def causal_bandpass(data, sfreq):
    """Filter the complete file forward once, with fresh zero state for each file.

    Keeping state across all samples avoids per-window edge artifacts. No future
    samples are used. A 60 Hz notch is omitted because features stop at 45 Hz.
    """
    if not np.isfinite(sfreq) or sfreq <= 90:
        raise ValueError("Sampling frequency must exceed twice the 45 Hz cutoff")
    if data.ndim != 2 or data.shape[1] == 0 or not np.isfinite(data).all():
        raise ValueError("EEG must be a finite channels-by-samples matrix")
    sos = butter(4, [0.5, 45.0], btype="bandpass", fs=sfreq, output="sos")
    return sosfilt(sos, data, axis=1)


def load_and_filter_edf(edf_path):
    with mne.io.read_raw_edf(edf_path, preload=False, verbose="WARNING") as raw:
        selected = select_channels(raw.ch_names)
        sfreq = float(raw.info["sfreq"])
        # Read only the selected bipolar EEG; MNE converts EDF amplitudes to volts.
        data = raw.get_data(picks=selected) * 1e6
    return causal_bandpass(data, sfreq), sfreq, selected


def extract_epoch_features(epoch_matrix, sfreq):
    """Return 7 features/channel and the number of near-zero-power channels."""
    if (epoch_matrix.ndim != 2 or epoch_matrix.shape[0] != len(CHANNELS)
            or epoch_matrix.shape[1] != round(WINDOW_SEC * sfreq)
            or not np.isfinite(epoch_matrix).all()):
        raise ValueError("Expected a finite, complete five-second 18-channel window")
    freqs, psd = welch(epoch_matrix, fs=sfreq, nperseg=round(sfreq * 2),
                       window="hann", noverlap=round(sfreq), detrend="constant", axis=1)
    # Identical bin integration in numerator and denominator partitions all power.
    # Bands are half-open, except gamma includes the final 45 Hz bin.
    dfreq = freqs[1] - freqs[0]
    total = psd[:, (freqs >= 0.5) & (freqs <= 45)].sum(axis=1) * dfreq
    low_power = total <= POWER_FLOOR_UV2
    relative = []
    for name, (low, high) in FREQ_BANDS.items():
        mask = (freqs >= low) & ((freqs <= high) if name == "gamma" else (freqs < high))
        power = psd[:, mask].sum(axis=1) * dfreq
        relative.append(np.divide(power, total, out=np.zeros_like(total), where=~low_power))
    features = np.column_stack((np.var(epoch_matrix, axis=1),
                               np.abs(np.diff(epoch_matrix, axis=1)).sum(axis=1), *relative))
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite extracted features")
    return features.ravel(), int(low_power.sum())


def window_bounds(n_samples, sfreq):
    """Tile integer sample bounds; warn about an incomplete final window."""
    if not np.isfinite(sfreq) or sfreq <= 90:
        raise ValueError("Sampling frequency must be finite and >90 Hz")
    exact = WINDOW_SEC * sfreq
    size = round(exact)
    if not np.isfinite(sfreq) or sfreq <= 90 or not np.isclose(size, exact, rtol=0, atol=1e-9):
        raise ValueError("Five seconds must contain an integer number of samples at sfreq >90 Hz")
    if n_samples < size:
        raise ValueError("Recording is shorter than one complete window")
    if n_samples % size:
        warnings.warn(f"Dropping {n_samples % size} trailing samples (incomplete window)")
    return [(start, start + size) for start in range(0, n_samples - size + 1, size)]


def window_annotation(start, end, intervals):
    """Half-open interval overlap; exactly 50% is ictal, touching edges is zero."""
    overlaps = [max(0.0, min(end, off) - max(start, on)) for on, off in intervals]
    overlap = sum(overlaps)
    return int(overlap >= 0.5 * (end - start)), overlap, [i + 1 for i, v in enumerate(overlaps) if v > 0]


def slice_and_extract(edf_filename, seizure_times, raw_dir=PROJECT_ROOT / "data/raw"):
    if not re.fullmatch(r"chb\d+_\d+[a-z]?\.edf", edf_filename):
        raise ValueError(f"Expected a CHB-MIT EDF basename: {edf_filename}")
    data, sfreq, selected = load_and_filter_edf(Path(raw_dir) / edf_filename)
    duration = data.shape[1] / sfreq
    validate_seizures(seizure_times, duration)
    rows = []
    for start_idx, end_idx in window_bounds(data.shape[1], sfreq):
        start, end = start_idx / sfreq, end_idx / sfreq
        label, overlap, ids = window_annotation(start, end, seizure_times)
        features, low_power = extract_epoch_features(data[:, start_idx:end_idx], sfreq)
        row = dict(zip(FEATURE_COLUMNS, features))
        row.update(patient_id=edf_filename.split("_")[0], recording_id=Path(edf_filename).stem,
                   source_file=edf_filename, start_sample=start_idx, end_sample=end_idx,
                   start_sec=start, end_sec=end, decision_time_sec=end,
                   sampling_rate_hz=sfreq, recording_duration_sec=duration,
                   label=label, seizure_overlap_sec=overlap, seizure_overlap_fraction=overlap / WINDOW_SEC,
                   overlapping_seizure_ids=json.dumps(ids), low_power_channel_count=low_power,
                   filter_warmup=start < FILTER_WARMUP_SEC,
                   postictal_60s=any(start < off + 60 and end > off for _, off in seizure_times),
                   processing_version=PROCESSING_VERSION)
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.low_power_channel_count.any():
        warnings.warn(f"{edf_filename}: {int((result.low_power_channel_count > 0).sum())} windows with near-zero-power channels")
    return result, {"duration_sec": duration, "n_samples": data.shape[1], "sampling_rate_hz": sfreq,
                    "selected_edf_channels": selected, "seizures_sec": seizure_times}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", nargs="+", default=list(DEFAULT_RECORDINGS),
                        help="Explicit basenames of COMPLETED EDFs only; never auto-discovers files")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument("--summary", type=Path, default=PROJECT_ROOT / "data/raw/chb01-summary.txt")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/processed/chb01_detection_features.parquet")
    args = parser.parse_args(argv)
    if args.output.resolve() == (PROJECT_ROOT / "data/processed/chb01_features.parquet").resolve():
        parser.error("The legacy prediction dataset must be preserved; choose a detection filename")
    if args.output.suffix != ".parquet":
        parser.error("Output must use .parquet extension")
    if len(set(args.recordings)) != len(args.recordings):
        parser.error("Duplicate recording names")
    # Validate the entire selection before opening any EDF. No directory scan.
    annotations = parse_summary_file(args.summary)
    for filename in args.recordings:
        if filename not in annotations:
            parser.error(f"No annotations for {filename}; never assume seizure-free")
    frames, recordings, captured = [], {}, []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for filename in args.recordings:
            print(f"Processing {filename}", flush=True)
            frame, metadata = slice_and_extract(filename, annotations[filename], args.raw_dir)
            metadata["sha256"] = sha256(args.raw_dir / filename)
            frames.append(frame)
            recordings[filename] = metadata
        captured = list(dict.fromkeys(str(w.message) for w in caught))
    table = pd.concat(frames, ignore_index=True)
    report = {"processing_version": PROCESSING_VERSION, "label_rule": ">=50% overlap of half-open intervals",
              "window_sec": WINDOW_SEC, "channels": CHANNELS, "bands_hz": FREQ_BANDS,
              "filter": "scipy Butterworth order=4 bandpass 0.5-45Hz SOS forward, zero initial state per EDF; no notch",
              "filter_warmup_flag_sec": FILTER_WARMUP_SEC, "power_floor_uv2": POWER_FLOOR_UV2,
              "postictal_flag_sec": 60, "postictal_policy": "retained; labels determined by ictal overlap only",
              "annotation_sha256": sha256(args.summary), "recordings": recordings,
              "preprocessing_source_sha256": sha256(__file__), "python_version": platform.python_version(),
              "versions": {name: importlib.metadata.version(name) for name in ("mne", "numpy", "pandas", "scipy", "pyarrow")},
              "warnings": captured, "counts": {"windows": len(table), "ictal": int(table.label.sum()),
                                               "nonictal": int((table.label == 0).sum())}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output, index=False)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(table.groupby(["source_file", "label"]).size().unstack(fill_value=0).rename(columns={0: "nonictal", 1: "ictal"}))
    print(f"Total: {len(table)} windows; ictal {table.label.mean():.4%}")
    print(f"Warnings: {captured or 'none'}")
    print(f"Saved {args.output} and matching JSON provenance report")


if __name__ == "__main__":
    main()
