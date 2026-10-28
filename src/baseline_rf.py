"""Phase 2B: fixed, unweighted baselines; train -> validation only.

Run `venv/bin/python -m src.baseline_rf --output results/baseline_v1`.
No scaling, sampling, threshold search, temporal filtering, or test predictions.
"""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

from .evaluate import evaluate_predictions
from .preprocess import FEATURE_COLUMNS, PROJECT_ROOT, sha256, window_annotation
from .split_dataset import load_manifest

DEFAULT_DATA = PROJECT_ROOT / "data/processed/chb01_detection_features_11recordings_split.parquet"
DEFAULT_MANIFEST = PROJECT_ROOT / "config/chb01_recordings.csv"
PREDICTION_METADATA = ["patient_id", "recording_id", "source_file", "split",
                       "start_sec", "end_sec", "decision_time_sec", "label"]


def select_features(frame):
    """Explicit whitelist, in preprocessing order; never infer from dtype."""
    if not frame.columns.is_unique or not set(FEATURE_COLUMNS) <= set(frame.columns):
        raise ValueError("Missing or duplicate EEG feature columns")
    features = frame.loc[:, FEATURE_COLUMNS].copy()
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("EEG features must be finite")
    return features


def load_development_data(path, manifest_path):
    """Verify provenance and load only train/validation rows via Parquet filters."""
    path, manifest_path = Path(path), Path(manifest_path)
    manifest = load_manifest(manifest_path)
    audit = json.loads(path.with_suffix(".json").read_text())
    if sha256(path) != audit["output_sha256"] or sha256(manifest_path) != audit["manifest_sha256"]:
        raise ValueError("Dataset or manifest differs from the verified split artifact")
    if audit["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("Feature ordering differs from the verified split artifact")
    frame = pd.read_parquet(path, filters=[("split", "in", ["train", "validation"])])
    expected = manifest[manifest.split.isin(["train", "validation"])]
    if set(frame.source_file) != set(expected.filename):
        raise ValueError("Development recordings differ from the manifest")
    assignments = expected.set_index("filename").split
    if not frame.split.eq(frame.source_file.map(assignments)).all():
        raise ValueError("Incorrect recording split assignment")
    if not frame.recording_id.eq(frame.source_file.map(lambda name: Path(name).stem)).all():
        raise ValueError("Recording ID differs from its source filename")
    if frame.duplicated(["recording_id", "start_sec", "end_sec"]).any():
        raise ValueError("Duplicate windows")
    select_features(frame)
    metadata = {}
    for recording_id, group in frame.groupby("recording_id", sort=False):
        source_file = group.source_file.iloc[0]
        raw = audit["raw_recordings"][source_file]
        metadata[recording_id] = {key: raw[key] for key in ("duration_sec", "seizures_sec")}
        expected_labels = [window_annotation(a, b, raw["seizures_sec"])[0]
                           for a, b in zip(group.start_sec, group.end_sec)]
        if not np.array_equal(group.label, expected_labels):
            raise ValueError(f"Labels disagree with annotations: {recording_id}")
    # Reuse the evaluator's complete, ordered timeline and annotation checks.
    checked = frame[PREDICTION_METADATA].copy()
    checked["prediction"] = 0
    evaluate_predictions(checked, metadata)
    partitions = {split: frame.loc[frame.split == split].copy()
                  for split in ("train", "validation")}
    if set(partitions["train"].label) != {0, 1}:
        raise ValueError("Training requires both classes")
    return partitions, metadata


def make_models():
    """Fixed before observing validation results; other sklearn defaults saved."""
    return {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42,
                                                 class_weight=None, n_jobs=4),
    }


def fit_baselines(train, models):
    if train.empty or not train.split.eq("train").all():
        raise ValueError("Only training rows may be fitted")
    features = select_features(train)
    if set(train.label) != {0, 1}:
        raise ValueError("Training requires both classes")
    for model in models.values():
        model.fit(features, train.label.to_numpy())


def validation_predictions(model, validation):
    """Keep metadata in input order; ndarray assignments are positional."""
    if validation.empty or not validation.split.eq("validation").all():
        raise ValueError("This experiment predicts validation rows only")
    features = select_features(validation)
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("Expected fitted binary classes [0, 1]")
    result = validation[PREDICTION_METADATA].copy()
    result["prediction"] = model.predict(features)
    result["score"] = model.predict_proba(features)[:, 1]
    # sklearn predict uses argmax: an exact probability tie selects class 0.
    return result


def score_summary(predictions):
    """Descriptive validation scores, without threshold selection."""
    summaries = []
    for (recording_id, label), group in predictions.groupby(["recording_id", "label"]):
        values = group.score
        summaries.append({"recording_id": recording_id, "label": int(label),
                          "windows": len(group), "predicted_positive": int(group.prediction.sum()),
                          "mean": float(values.mean()),
                          **{name: float(values.quantile(q)) for name, q in
                             (("min", 0), ("q25", .25), ("median", .5), ("q75", .75),
                              ("q95", .95), ("max", 1))}})
    return summaries


def plot_scores(predictions, metadata, output):
    """Plot only validation recordings with seizures, using end-time decisions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for recording_id, record in metadata.items():
        if not record["seizures_sec"]:
            continue
        group = predictions[predictions.recording_id == recording_id]
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), layout="constrained")
        for ax in axes:
            ax.plot(group.end_sec, group.score, color="#1768ac", linewidth=1, label="RF seizure score")
            ax.axhline(.5, color="#a33a22", linestyle="--", label="Default boundary (ties → class 0)")
            for i, (onset, offset) in enumerate(record["seizures_sec"]):
                ax.axvspan(onset, offset, color="#e9b44c", alpha=.3,
                           label="Annotated seizure" if i == 0 else None)
            ax.set(ylabel="Seizure-class score", ylim=(-.02, 1.02))
            ax.grid(alpha=.2)
        axes[0].set(title=f"{recording_id}: validation scores at window end", xlim=(0, record["duration_sec"]))
        axes[0].legend(loc="upper right", fontsize=8)
        axes[1].set(xlabel="Decision time from recording start (seconds)",
                    xlim=(max(0, record["seizures_sec"][0][0] - 120),
                          min(record["duration_sec"], record["seizures_sec"][-1][1] + 120)))
        fig.savefig(output / f"{recording_id}_scores.png", dpi=160)
        plt.close(fig)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_experiment(data, manifest, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Use a new output directory; preserving {output}")
    partitions, metadata = load_development_data(data, manifest)
    train, validation = partitions["train"], partitions["validation"]
    validation_metadata = {rid: metadata[rid] for rid in validation.recording_id.unique()}
    models = make_models()
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data": str(Path(data).resolve()), "data_sha256": sha256(data),
        "manifest": str(Path(manifest).resolve()), "manifest_sha256": sha256(manifest),
        "split_audit_sha256": sha256(Path(data).with_suffix(".json")),
        "source_sha256": {name: sha256(PROJECT_ROOT / "src" / name)
                          for name in ("baseline_rf.py", "preprocess.py", "split_dataset.py", "evaluate.py")},
        "feature_columns": FEATURE_COLUMNS, "models": {name: model.get_params() for name, model in models.items()},
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {name: version(name) for name in
                     ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "pyarrow", "matplotlib", "mne")},
        "decision_rule": "sklearn predict; binary RF score > 0.5, exact ties select class 0",
        "policy": "train only; validation evaluation only; no tuning, weighting, resampling or temporal filtering",
        "counts": {split: {"windows": len(frame), "class_0": int((frame.label == 0).sum()),
                           "class_1": int((frame.label == 1).sum()),
                           "recordings": list(frame.recording_id.unique()),
                           "seizures": sum(len(metadata[rid]["seizures_sec"]) for rid in frame.recording_id.unique())}
                   for split, frame in partitions.items()},
        "validation_metadata": validation_metadata,
    }
    write_json(output / "configuration.json", config)
    fit_baselines(train, models)
    reports = {}
    for name, model in models.items():
        predictions = validation_predictions(model, validation)
        model_path = output / f"{name}.joblib"
        joblib.dump(model, model_path, compress=3)
        restored = validation_predictions(joblib.load(model_path), validation)
        pd.testing.assert_frame_equal(predictions, restored)
        predictions.to_parquet(output / f"{name}_validation.parquet", index=False)
        report = evaluate_predictions(predictions, validation_metadata)
        report["score_distribution"] = score_summary(predictions)
        report["serialization_verified"] = True
        write_json(output / f"{name}_evaluation.json", report)
        reports[name] = report
        if name == "random_forest":
            # Include every error and positive call, with its original window identity.
            predictions.loc[(predictions.prediction != predictions.label) | predictions.prediction.eq(1)].to_parquet(
                output / "random_forest_review_windows.parquet", index=False)
            plot_scores(predictions, validation_metadata, output)
    write_json(output / "artifact_sha256.json", {path.name: sha256(path) for path in sorted(output.iterdir())})
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/baseline_v1")
    args = parser.parse_args()
    reports = run_experiment(args.data, args.manifest, args.output)
    print(json.dumps({name: {key: report[key] for key in ("window_metrics", "event_metrics")}
                      for name, report in reports.items()}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
