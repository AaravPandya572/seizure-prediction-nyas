"""One controlled class-weight experiment, using Phase 2B artifacts read-only."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

from .baseline_rf import (DEFAULT_DATA, DEFAULT_MANIFEST, PREDICTION_METADATA,
                          fit_baselines, load_development_data, score_summary,
                          select_features, validation_predictions, write_json)
from .evaluate import evaluate_predictions
from .preprocess import FEATURE_COLUMNS, PROJECT_ROOT, sha256


def weighted_clone(baseline):
    """Clone every saved hyperparameter, changing only the requested weight."""
    if not isinstance(baseline, RandomForestClassifier) or baseline.class_weight is not None:
        raise ValueError("Expected an ordinary, unweighted Random Forest")
    return clone(baseline).set_params(class_weight="balanced")


def training_fingerprint(train):
    """Ordered features, labels and identities used by both experiment phases."""
    import hashlib
    if train.empty or not train.split.eq("train").all():
        raise ValueError("Fingerprint requires training rows only")
    features = select_features(train)
    digest = hashlib.sha256()
    digest.update(json.dumps(list(features.columns)).encode())
    digest.update(features.to_numpy(dtype="<f8").tobytes())
    digest.update(train.label.to_numpy(dtype="<i8").tobytes())
    digest.update(train[PREDICTION_METADATA].to_json(orient="records", double_precision=15).encode())
    return digest.hexdigest()


def compare_predictions(ordinary, weighted):
    """Reject misalignment rather than joining rows ambiguously by timestamps."""
    pd.testing.assert_frame_equal(ordinary[PREDICTION_METADATA], weighted[PREDICTION_METADATA])
    result = ordinary.rename(columns={"prediction": "ordinary_prediction", "score": "ordinary_score"}).copy()
    result["weighted_prediction"] = weighted.prediction.to_numpy()
    result["weighted_score"] = weighted.score.to_numpy()
    result["changed"] = result.ordinary_prediction != result.weighted_prediction
    result["previously_missed"] = result.label.eq(1) & result.ordinary_prediction.eq(0)
    result["corrected"] = result.ordinary_prediction.ne(result.label) & result.weighted_prediction.eq(result.label)
    result["regressed"] = result.ordinary_prediction.eq(result.label) & result.weighted_prediction.ne(result.label)
    return result


def plot_comparison(comparison, metadata, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for rid, record in metadata.items():
        if not record["seizures_sec"]:
            continue
        group = comparison[comparison.recording_id == rid]
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), layout="constrained")
        for ax in axes:
            for column, label, color in (("ordinary_score", "Ordinary RF", "#1768ac"),
                                          ("weighted_score", "Weighted RF", "#b23a48")):
                ax.plot(group.end_sec, group[column], marker=".", markersize=3,
                        linewidth=1, label=label, color=color)
            ax.axhline(.5, linestyle="--", color="gray", label="Default boundary; ties select 0")
            for i, (onset, offset) in enumerate(record["seizures_sec"]):
                ax.axvspan(onset, offset, alpha=.2, color="#e9b44c", label="Seizure" if i == 0 else None)
            ax.set(ylabel="Seizure-class score", ylim=(-.02, 1.02))
            ax.grid(alpha=.2)
        axes[0].set(title=f"{rid}: fixed RF comparison on validation", xlim=(0, record["duration_sec"]))
        axes[0].legend(fontsize=8, loc="upper right")
        axes[1].set(xlim=(record["seizures_sec"][0][0] - 30, record["seizures_sec"][-1][1] + 30),
                    xlabel="Window-end decision time (seconds from recording start)")
        fig.savefig(output / f"{rid}_comparison.png", dpi=160)
        plt.close(fig)


def run_experiment(data, manifest, baseline_dir, output):
    baseline_dir, output = Path(baseline_dir), Path(output)
    if output.exists():
        raise FileExistsError(f"Use a new output directory; preserving {output}")
    config = json.loads((baseline_dir / "configuration.json").read_text())
    for name, expected in json.loads((baseline_dir / "artifact_sha256.json").read_text()).items():
        if sha256(baseline_dir / name) != expected:
            raise ValueError(f"Baseline artifact changed: {name}")
    for path, expected in ((data, config["data_sha256"]), (manifest, config["manifest_sha256"]),
                           (Path(data).with_suffix(".json"), config["split_audit_sha256"])):
        if sha256(path) != expected:
            raise ValueError("Baseline data or recording assignments changed")
    if config["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("Baseline feature ordering changed")
    for name, expected in config["source_sha256"].items():
        if sha256(PROJECT_ROOT / "src" / name) != expected:
            raise ValueError(f"Baseline methodology source changed: {name}")
    for name, expected in config["versions"].items():
        if version(name) != expected:
            raise ValueError(f"Library version changed: {name}; restore the baseline environment")
    partitions, metadata = load_development_data(data, manifest)
    train, validation = partitions["train"], partitions["validation"]
    metadata = {rid: metadata[rid] for rid in validation.recording_id.unique()}
    if metadata != config["validation_metadata"]:
        raise ValueError("Validation annotation metadata changed")
    baseline = joblib.load(baseline_dir / "random_forest.joblib")
    if baseline.get_params() != config["models"]["random_forest"]:
        raise ValueError("Saved model parameters differ from configuration")
    if list(baseline.feature_names_in_) != FEATURE_COLUMNS:
        raise ValueError("Saved model feature names differ")
    ordinary = validation_predictions(baseline, validation).reset_index(drop=True)
    pd.testing.assert_frame_equal(ordinary, pd.read_parquet(baseline_dir / "random_forest_validation.parquet"))
    ordinary_report = evaluate_predictions(ordinary, metadata)
    saved_report = json.loads((baseline_dir / "random_forest_evaluation.json").read_text())
    for key, value in ordinary_report.items():
        if value != saved_report[key]:
            raise ValueError(f"Baseline evaluation changed: {key}")
    weighted = weighted_clone(baseline)
    output.mkdir(parents=True, exist_ok=False)
    experiment = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_directory": str(baseline_dir.resolve()), "baseline_configuration_sha256": sha256(baseline_dir / "configuration.json"),
        "data_sha256": sha256(data), "manifest_sha256": sha256(manifest),
        "training_inputs_sha256": training_fingerprint(train),
        "training_identity_evidence": "Same verified dataset hash, assignments, feature order, labels, source and library versions as Phase 2B",
        "feature_columns": FEATURE_COLUMNS, "counts": config["counts"], "versions": config["versions"],
        "python": config["python"], "decision_rule": config["decision_rule"],
        "validation_metadata": metadata, "ordinary_parameters": baseline.get_params(),
        "weighted_parameters": weighted.get_params(), "changed_parameters": ["class_weight"],
        "source_sha256": sha256(Path(__file__)),
        "balanced_weights": {str(c): len(train) / (2 * int((train.label == c).sum())) for c in (0, 1)},
    }
    write_json(output / "configuration.json", experiment)
    fit_baselines(train, {"weighted": weighted})
    if training_fingerprint(train) != experiment["training_inputs_sha256"]:
        raise ValueError("Training inputs were mutated")
    predictions = validation_predictions(weighted, validation).reset_index(drop=True)
    joblib.dump(weighted, output / "weighted_rf.joblib", compress=3)
    restored = validation_predictions(joblib.load(output / "weighted_rf.joblib"), validation).reset_index(drop=True)
    pd.testing.assert_frame_equal(predictions, restored)
    predictions.to_parquet(output / "weighted_validation.parquet", index=False)
    weighted_report = evaluate_predictions(predictions, metadata)
    for name, report, table in (("ordinary", ordinary_report, ordinary), ("weighted", weighted_report, predictions)):
        report["score_distribution"] = score_summary(table)
        write_json(output / f"{name}_evaluation.json", report)
    comparison = compare_predictions(ordinary, predictions)
    comparison.to_parquet(output / "validation_comparison.parquet", index=False)
    comparison.loc[comparison.previously_missed | comparison.changed].to_parquet(output / "review_windows.parquet", index=False)
    write_json(output / "verification.json", {
        "baseline_predictions_and_evaluation_unchanged": True, "weighted_reload_identical": True,
        "training_inputs_unchanged": True, "test_rows_used": False,
        "changed_windows": int(comparison.changed.sum()), "corrected_windows": int(comparison.corrected.sum()),
        "regressed_windows": int(comparison.regressed.sum()),
        "previous_misses_corrected": int((comparison.previously_missed & comparison.corrected).sum()),
    })
    plot_comparison(comparison, metadata, output)
    write_json(output / "artifact_sha256.json", {p.name: sha256(p) for p in sorted(output.iterdir())})
    return ordinary_report, weighted_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline", type=Path, default=PROJECT_ROOT / "results/baseline_v1")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/rf_weighted_v1")
    args = parser.parse_args()
    reports = run_experiment(args.data, args.manifest, args.baseline, args.output)
    print(json.dumps({name: {k: report[k] for k in ("window_metrics", "event_metrics")}
                      for name, report in zip(("ordinary", "weighted"), reports)}, indent=2))


if __name__ == "__main__":
    main()
