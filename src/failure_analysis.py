"""Analysis only: saved out-of-fold scores and existing training EEG features."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .baseline_rf import DEFAULT_DATA, DEFAULT_MANIFEST, PREDICTION_METADATA, write_json
from .evaluate import evaluate_predictions
from .preprocess import FEATURE_COLUMNS, STAT_NAMES, CHANNELS, PROJECT_ROOT, sha256
from .rf_cross_validation import HOLDOUT_IDS, load_training_data, predict_holdout


def score_periods(frame, onset, offset):
    """Disjoint descriptive bins at causal decision times; no relabeling."""
    times = frame.end_sec
    return {
        "pre60": (times >= onset - 60) & (times < onset),
        "during": (times >= onset) & (times < offset),
        "post60": (times >= offset) & (times < offset + 60),
        "remote": (times < onset - 60) | (times >= offset + 60),
    }


def feature_periods(frame, onset, offset):
    """Ictal uses existing labels; post uses fully postictal windows only."""
    return {
        "remote_nonictal": frame.label.eq(0) & frame.start_sec.ge(10) &
            (frame.end_sec.le(onset - 60) | frame.start_sec.ge(offset + 60)),
        "ictal": frame.label.eq(1),
        "post60_full": frame.start_sec.ge(offset) & frame.end_sec.le(offset + 60),
    }


def describe_scores(frame, onset, offset):
    rows = []
    for name, mask in score_periods(frame, onset, offset).items():
        group = frame.loc[mask]
        maximum = group.score.max() if len(group) else None
        rows.append({"period": name, "windows": len(group),
                     "mean": float(group.score.mean()) if len(group) else None,
                     "median": float(group.score.median()) if len(group) else None,
                     "min": float(group.score.min()) if len(group) else None,
                     "max": float(maximum) if maximum is not None else None,
                     "peak_decisions_sec": group.loc[group.score.eq(maximum), "end_sec"].tolist(),
                     "positive_windows": int(group.prediction.sum())})
    return rows


def feature_descriptions(frame, metadata):
    rows, families = [], []
    for rid in HOLDOUT_IDS:
        group = frame[frame.recording_id.eq(rid)]
        onset, offset = metadata[rid]["seizures_sec"][0]
        for period, mask in feature_periods(group, onset, offset).items():
            selected = group.loc[mask]
            for feature in FEATURE_COLUMNS:
                values = selected[feature]
                rows.append({"recording_id": rid, "period": period, "feature": feature,
                             "family": feature.rsplit("_", 1)[-1] if not feature.endswith("line_length") else "line_length",
                             "n": len(values), "median": float(values.median()),
                             "q25": float(values.quantile(.25)), "q75": float(values.quantile(.75)),
                             "min": float(values.min()), "max": float(values.max())})
            for family in STAT_NAMES:
                # Fixed across-channel summary, not a learned transformation.
                values = selected[[f"{ch}_{family}" for ch in CHANNELS]].median(axis=1)
                families.append({"recording_id": rid, "period": period, "family": family,
                                 "n": len(values), "median": float(values.median()),
                                 "q25": float(values.quantile(.25)), "q75": float(values.quantile(.75))})
    table = pd.DataFrame(rows)
    # Compare event medians equally, avoiding dominance by longer seizures.
    pivot = table.pivot(index="feature", columns=["recording_id", "period"], values="median")
    comparisons = []
    for feature in FEATURE_COLUMNS:
        med = pivot.loc[feature]
        other = [rid for rid in HOLDOUT_IDS if rid != "chb01_15"]
        baseline = [med[rid, "remote_nonictal"] for rid in other]
        ictal = [med[rid, "ictal"] for rid in other]
        ratios = [a / b if b else None for a, b in zip(ictal, baseline)]
        target_base, target_ictal = med["chb01_15", "remote_nonictal"], med["chb01_15", "ictal"]
        reference_delta = float(np.median(np.array(ictal) - baseline))
        comparisons.append({"feature": feature, "target_ictal": target_ictal,
                            "other_event_median_ictal": float(np.median(ictal)),
                            "target_remote": target_base, "other_event_median_remote": float(np.median(baseline)),
                            "target_ictal_minus_remote": target_ictal - target_base,
                            "other_event_median_change": reference_delta,
                            "target_ictal_over_remote": target_ictal / target_base if target_base else None,
                            "other_event_median_ratio": float(np.median(ratios)) if all(v is not None for v in ratios) else None,
                            "difference_of_changes": target_ictal - target_base - reference_delta})
    return table, pd.DataFrame(families), pd.DataFrame(comparisons)


def figures(frame, metadata, predictions, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for mode in ("missed", "comparison"):
        ids = ["chb01_15"] if mode == "missed" else list(HOLDOUT_IDS)
        fig, axes = plt.subplots(2 if mode == "missed" else 4, 1, figsize=(11, 7 if mode == "missed" else 10), layout="constrained")
        for i, ax in enumerate(axes):
            rid = ids[0] if mode == "missed" else ids[i]
            onset, offset = metadata[rid]["seizures_sec"][0]
            for name, color in (("ordinary", "#1768ac"), ("weighted", "#b23a48")):
                table, report = predictions[rid, name]
                ax.plot(table.end_sec, table.score, marker=".", markersize=3, linewidth=1, label=name, color=color)
                for alarm in report["event_metrics"]["alarms"]:
                    ax.scatter(alarm["start_sec"], table.loc[table.end_sec.eq(alarm["start_sec"]), "score"].iloc[0],
                               marker="x", s=65, color=color, zorder=5)
            ax.axvspan(onset, offset, color="#e9b44c", alpha=.25, label="Annotated seizure")
            ax.axvline(onset, color="#a07813", linewidth=.8)
            ax.axvline(offset, color="#a07813", linewidth=.8)
            ax.axhline(.5, color="gray", linestyle="--", label="Default boundary")
            ax.set(title=f"{rid}: onset {onset:g}, offset {offset:g}; × = alarm start", ylabel="Score", ylim=(-.02, 1.02),
                   xlim=(0, metadata[rid]["duration_sec"]) if mode == "missed" and i == 0 else (onset - 60, offset + 90))
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=8, loc="upper right")
        axes[-1].set_xlabel("Actual window-end decision time (s)")
        fig.savefig(output / f"scores_{mode}.png", dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(4, 2, figsize=(12, 11), layout="constrained")
    colors = ("#777777", "#b23a48", "#1768ac")
    for ax, family in zip(axes.flat, STAT_NAMES):
        data, positions, labels = [], [], []
        for i, rid in enumerate(HOLDOUT_IDS):
            group = frame[frame.recording_id.eq(rid)]
            on, off = metadata[rid]["seizures_sec"][0]
            for j, (_, mask) in enumerate(feature_periods(group, on, off).items()):
                data.append(group.loc[mask, [f"{ch}_{family}" for ch in CHANNELS]].median(axis=1))
                positions.append(i * 4 + j)
            labels.append(rid.replace("chb01_", ""))
        box = ax.boxplot(data, positions=positions, widths=.65, patch_artist=True, showfliers=False)
        for i, patch in enumerate(box["boxes"]): patch.set_facecolor(colors[i % 3])
        ax.set(title=family, xticks=[1, 5, 9, 13], xticklabels=labels, xlabel="Recording suffix")
        if family in ("var", "line_length"): ax.set_yscale("log")
        ax.grid(axis="y", alpha=.2)
    axes.flat[-1].axis("off")
    fig.suptitle("Per-window median across 18 channels: gray remote / red ictal / blue post60\nBoxes: IQR and median; whiskers 1.5 IQR; outliers hidden, all values saved")
    fig.savefig(output / "feature_families.png", dpi=160)
    plt.close(fig)
    target = frame.loc[frame.recording_id.eq("chb01_15")]
    on, off = metadata["chb01_15"]["seizures_sec"][0]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), layout="constrained")
    # Exploratory illustrations, not feature selection for a future classifier.
    for ax, feature, unit in zip(axes, ("FP2-F8_gamma", "T8-P8_line_length", "FZ-CZ_var"),
                                 ("Relative power", "µV (sum)", "µV²")):
        ax.plot(target.end_sec, target[feature], marker=".", linewidth=1)
        ax.axvspan(on, off, color="#e9b44c", alpha=.25, label="Seizure")
        ax.axvline(1785, color="#b23a48", linestyle="--", label="Both alarm starts")
        ax.set(title=feature, ylabel=unit, xlim=(on - 60, off + 60))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("Window-end time (s); illustrative post hoc features")
    fig.savefig(output / "feature_timelines.png", dpi=160)
    plt.close(fig)


def run_analysis(data, manifest, cv_dir, output):
    cv_dir, output = Path(cv_dir), Path(output)
    if output.exists(): raise FileExistsError(f"Preserving existing directory {output}")
    config = json.loads((cv_dir / "configuration.json").read_text())
    if sha256(data) != config["data_sha256"] or sha256(manifest) != config["manifest_sha256"]:
        raise ValueError("CV dataset/manifest changed")
    for name, expected in json.loads((cv_dir / "artifact_sha256.json").read_text()).items():
        if sha256(cv_dir / name) != expected: raise ValueError(f"CV artifact changed: {name}")
    frame, metadata = load_training_data(data, manifest)
    output.mkdir(parents=True)
    predictions, scores, alarms = {}, [], []
    for i, rid in enumerate(HOLDOUT_IDS, 1):
        folder = cv_dir / f"fold_{i}_{rid}"
        held = frame[frame.recording_id.eq(rid)]
        for name in ("ordinary", "weighted"):
            table = pd.read_parquet(folder / f"{name}_predictions.parquet")
            pd.testing.assert_frame_equal(table[PREDICTION_METADATA], held[PREDICTION_METADATA].reset_index(drop=True))
            model = joblib.load(folder / f"{name}.joblib")
            pd.testing.assert_frame_equal(table, predict_holdout(model, held, rid).reset_index(drop=True))
            evaluated = evaluate_predictions(table, {rid: metadata[rid]})
            saved = json.loads((folder / f"{name}_evaluation.json").read_text())
            for key, value in evaluated.items():
                if value != saved[key]: raise ValueError("Saved evaluation mismatch")
            report = evaluated["per_recording"][rid]
            predictions[rid, name] = (table, report)
            on, off = metadata[rid]["seizures_sec"][0]
            scores.extend({"recording_id": rid, "model": name, **row} for row in describe_scores(table, on, off))
            for alarm in report["event_metrics"]["alarms"]:
                row = table.loc[table.end_sec.eq(alarm["start_sec"])].iloc[0]
                alarms.append({"recording_id": rid, "model": name, **alarm,
                               "seconds_after_offset": alarm["start_sec"] - off,
                               "window_start_sec": float(row.start_sec), "window_end_sec": float(row.end_sec),
                               "score": float(row.score), "label": int(row.label)})
            if rid == "chb01_15": table.loc[table.end_sec.between(on - 60, off + 60)].to_parquet(output / f"{name}_local_scores.parquet", index=False)
    desc, families, differences = feature_descriptions(frame, metadata)
    desc.to_parquet(output / "all_feature_distributions.parquet", index=False)
    families.to_parquet(output / "family_distributions.parquet", index=False)
    differences.to_parquet(output / "feature_differences.parquet", index=False)
    local = frame.loc[frame.recording_id.eq("chb01_15")].copy()
    for family in STAT_NAMES:
        local["family_" + family] = local[[f"{ch}_{family}" for ch in CHANNELS]].median(axis=1)
    on, off = metadata["chb01_15"]["seizures_sec"][0]
    columns = ["start_sec", "end_sec", "label"] + ["family_" + s for s in STAT_NAMES] + ["FP2-F8_gamma", "T8-P8_line_length"]
    local.loc[local.end_sec.between(on - 60, off + 60), columns].to_parquet(output / "local_feature_timeline.parquet", index=False)
    write_json(output / "score_periods.json", scores)
    write_json(output / "alarms.json", alarms)
    write_json(output / "quality.json", {rid: {"windows": len(g), "low_power_windows": int(g.low_power_channel_count.gt(0).sum()),
        "warmup_windows": int(g.filter_warmup.sum()), "max_low_power_channels": int(g.low_power_channel_count.max())}
        for rid, g in frame.groupby("recording_id")})
    write_json(output / "configuration.json", {"source_sha256": sha256(Path(__file__)), "cv_configuration_sha256": sha256(cv_dir / "configuration.json"),
        "data_sha256": sha256(data), "metadata": metadata, "learned_transformations": None,
        "feature_periods": "remote nonictal outside +/-60s, excluding first10s; ictal=original label; post60 fully inside [offset,offset+60]",
        "score_periods": "decision end in pre60 [on-60,on), during [on,off), post60 [off,off+60), remote otherwise"})
    figures(frame, metadata, predictions, output)
    write_json(output / "verification.json", {"models_trained": 0, "saved_model_predictions_verified": 8,
        "evaluation_reports_reproduced": 8, "all_126_features_finite": True, "protected_recordings_used": False})
    write_json(output / "artifact_sha256.json", {p.name: sha256(p) for p in sorted(output.iterdir())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cv-dir", type=Path, default=PROJECT_ROOT / "results/rf_cv_v1")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/failure_analysis_v1")
    args = parser.parse_args()
    run_analysis(args.data, args.manifest, args.cv_dir, args.output)


if __name__ == "__main__":
    main()
