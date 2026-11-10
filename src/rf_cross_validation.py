"""Four fixed training-only leave-one-seizure-recording-out RF comparisons."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from .baseline_rf import (DEFAULT_DATA, DEFAULT_MANIFEST, PREDICTION_METADATA,
                          fit_baselines, make_models, score_summary, select_features, write_json)
from .evaluate import evaluate_predictions
from .preprocess import FEATURE_COLUMNS, PROJECT_ROOT, sha256, window_annotation
from .split_dataset import load_manifest
from .weighted_rf import training_fingerprint, weighted_clone

TRAIN_IDS = tuple(f"chb01_{n:02d}" for n in (1, 3, 4, 14, 15, 16))
HOLDOUT_IDS = tuple(f"chb01_{n:02d}" for n in (3, 4, 15, 16))
RATE_METRICS = ("accuracy", "precision", "recall", "specificity", "f1", "average_precision")


def check_training_scope(frame):
    if frame.empty or not frame.split.eq("train").all() or set(frame.recording_id) != set(TRAIN_IDS):
        raise ValueError("CV requires exactly the six original training recordings")


def load_training_data(data, manifest):
    """Decode training rows only; reuse the split audit's existing annotations."""
    data, manifest = Path(data), Path(manifest)
    assignments = load_manifest(manifest)
    audit = json.loads(data.with_suffix(".json").read_text())
    if sha256(data) != audit["output_sha256"] or sha256(manifest) != audit["manifest_sha256"]:
        raise ValueError("Data or manifest differs from verified split artifact")
    expected = {f"{rid}.edf" for rid in TRAIN_IDS}
    if set(assignments.loc[assignments.split.eq("train"), "filename"]) != expected:
        raise ValueError("Frozen training assignments changed")
    if audit["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("Canonical feature ordering changed")
    frame = pd.read_parquet(data, filters=[("split", "==", "train")])
    check_training_scope(frame)
    if not frame.source_file.eq(frame.recording_id + ".edf").all():
        raise ValueError("Source filename and recording ID disagree")
    if frame.duplicated(["recording_id", "start_sec", "end_sec"]).any():
        raise ValueError("Duplicate windows")
    select_features(frame)
    metadata = {}
    for rid, group in frame.groupby("recording_id", sort=False):
        raw = audit["raw_recordings"][f"{rid}.edf"]
        metadata[rid] = {key: raw[key] for key in ("duration_sec", "seizures_sec")}
        if not np.array_equal(group.end_sec - group.start_sec, np.full(len(group), 5.0)):
            raise ValueError("Expected five-second windows")
        labels = [window_annotation(a, b, raw["seizures_sec"])[0] for a, b in zip(group.start_sec, group.end_sec)]
        if not np.array_equal(group.label, labels):
            raise ValueError("Labels disagree with annotations")
        if len(raw["seizures_sec"]) != int(rid in HOLDOUT_IDS):
            raise ValueError("Expected one seizure in each held-out recording, none in normal recordings")
    # Validate complete ordered timelines, annotations, decision times and labels.
    checked = frame[PREDICTION_METADATA].copy()
    checked["prediction"] = 0
    evaluate_predictions(checked, metadata)
    return frame, metadata


def fold_data(frame, held_out):
    check_training_scope(frame)
    if held_out not in HOLDOUT_IDS:
        raise ValueError("Holdout must be one of the four training seizure recordings")
    return (frame.loc[frame.recording_id.ne(held_out)].copy(),
            frame.loc[frame.recording_id.eq(held_out)].copy())


def predict_holdout(model, held, held_out):
    if held.empty or not held.split.eq("train").all() or set(held.recording_id) != {held_out} or held_out not in HOLDOUT_IDS:
        raise ValueError("Predictions require the specified training-partition holdout only")
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("Expected fitted classes [0,1]")
    features = select_features(held)
    result = held[PREDICTION_METADATA].copy()
    # Preserve the original split; fold role is explicit in fold definitions.
    result["prediction"] = model.predict(features)
    result["score"] = model.predict_proba(features)[:, 1]
    return result


def aggregate_predictions(tables, metadata):
    """Each seizure recording contributes exactly once per model, never per alarm."""
    if len(tables) != len(HOLDOUT_IDS) or any(table.recording_id.nunique() != 1 for table in tables):
        raise ValueError("Expected one complete prediction table per fold")
    ids = [table.recording_id.iloc[0] for table in tables]
    if len(set(ids)) != 4 or set(ids) != set(HOLDOUT_IDS):
        raise ValueError("Duplicate or missing held-out recordings")
    pooled = pd.concat(tables, ignore_index=True)
    if not pooled.split.eq("train").all():
        raise ValueError("Only original training recordings may be aggregated")
    report = evaluate_predictions(pooled, {rid: metadata[rid] for rid in ids})
    macro = {}
    for name in RATE_METRICS:
        values = [r["window_metrics"][name] for r in report["per_recording"].values()
                  if r["window_metrics"][name] is not None]
        macro[name] = {"n_defined": len(values), "mean": float(np.mean(values)) if values else None,
                       "std_population": float(np.std(values)) if values else None,
                       "min": min(values) if values else None, "max": max(values) if values else None}
    report["macro_window_metrics"] = macro
    report["macro_event_sensitivity"] = float(np.mean([r["event_metrics"]["event_sensitivity"]
                                                       for r in report["per_recording"].values()]))
    return report


def run_experiment(data, manifest, baseline_config, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Use a new output directory; preserving {output}")
    original = json.loads(Path(baseline_config).read_text())
    for path, expected in ((data, original["data_sha256"]), (manifest, original["manifest_sha256"]),
                           (Path(data).with_suffix(".json"), original["split_audit_sha256"])):
        if sha256(path) != expected:
            raise ValueError("Baseline dataset or assignments changed")
    for name, expected in original["versions"].items():
        if version(name) != expected:
            raise ValueError(f"Baseline library version changed: {name}")
    for name, expected in original["source_sha256"].items():
        if sha256(PROJECT_ROOT / "src" / name) != expected:
            raise ValueError(f"Methodology source changed: {name}")
    template = make_models()["random_forest"]
    if template.get_params() != original["models"]["random_forest"] or original["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("Baseline forest parameters or features changed")
    frame, metadata = load_training_data(data, manifest)
    folds, all_predictions = [], {"ordinary": [], "weighted": []}
    output.mkdir(parents=True, exist_ok=False)
    config = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "data_sha256": sha256(data), "manifest_sha256": sha256(manifest),
              "baseline_configuration_sha256": sha256(baseline_config),
              "source_sha256": {name: sha256(PROJECT_ROOT / "src" / name) for name in
                                ("rf_cross_validation.py", "baseline_rf.py", "weighted_rf.py", "evaluate.py")},
              "python": platform.python_version(), "platform": platform.platform(), "versions": original["versions"],
              "feature_columns": FEATURE_COLUMNS, "training_pool": list(TRAIN_IDS),
              "holdout_order": list(HOLDOUT_IDS), "metadata": metadata,
              "models": {"ordinary": template.get_params(), "weighted": weighted_clone(template).get_params()},
              "decision_rule": "sklearn predict; score >0.5, exact ties select class 0",
              "policy": "exploratory recording-level CV on frozen training pool only; no tuning"}
    write_json(output / "configuration.json", config)
    for number, rid in enumerate(HOLDOUT_IDS, 1):
        train, held = fold_data(frame, rid)
        folder = output / f"fold_{number}_{rid}"
        folder.mkdir()
        signature = training_fingerprint(train)
        definition = {"fold": number, "held_out": rid, "train_recordings": list(train.recording_id.unique()),
                      "training_windows": len(train), "training_ictal": int(train.label.sum()),
                      "evaluated_windows": len(held), "evaluated_ictal": int(held.label.sum()),
                      "training_inputs_sha256": signature,
                      "balanced_weights": {str(c): len(train) / (2 * int(train.label.eq(c).sum())) for c in (0, 1)}}
        folds.append(definition)
        write_json(folder / "definition.json", definition)
        # Fresh independent estimators per fold; neither forest has seen this EDF.
        models = {"ordinary": clone(template), "weighted": weighted_clone(template)}
        fit_baselines(train, models)
        if training_fingerprint(train) != signature:
            raise ValueError("Training inputs were mutated")
        for name, model in models.items():
            predictions = predict_holdout(model, held, rid)
            joblib.dump(model, folder / f"{name}.joblib", compress=3)
            pd.testing.assert_frame_equal(predictions, predict_holdout(joblib.load(folder / f"{name}.joblib"), held, rid))
            predictions.to_parquet(folder / f"{name}_predictions.parquet", index=False)
            report = evaluate_predictions(predictions, {rid: metadata[rid]})
            report["score_distribution"] = score_summary(predictions)
            write_json(folder / f"{name}_evaluation.json", report)
            all_predictions[name].append(predictions)
    write_json(output / "folds.json", folds)
    summaries = {name: aggregate_predictions(tables, metadata) for name, tables in all_predictions.items()}
    write_json(output / "summary.json", summaries)
    write_json(output / "verification.json", {"independent_models_fitted": 8, "reloads_identical": True,
               "unique_evaluated_recordings_per_model": list(HOLDOUT_IDS),
               "validation_test_or_new_recordings_used": False, "training_inputs_unchanged": True})
    write_json(output / "artifact_sha256.json", {str(p.relative_to(output)): sha256(p)
               for p in sorted(output.rglob("*")) if p.is_file()})
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline-config", type=Path, default=PROJECT_ROOT / "results/baseline_v1/configuration.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/rf_cv_v1")
    args = parser.parse_args()
    summaries = run_experiment(args.data, args.manifest, args.baseline_config, args.output)
    print(json.dumps({name: {k: report[k] for k in ("window_metrics", "event_metrics", "macro_window_metrics")}
                      for name, report in summaries.items()}, indent=2))


if __name__ == "__main__":
    main()
