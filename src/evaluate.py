"""Offline evaluation of binary window decisions and seizure alarm episodes.

This module reads no files, trains no models, and performs no temporal filtering.
Times are seconds relative to one recording. See docs/evaluation.md for the
input contract, strict onset-matching policy, and interpretation of undefined
metrics. Results contain plain Python values and can be serialized as JSON.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def _vector(values, name):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty, finite one-dimensional array")
    return array


def _binary(values, name):
    array = _vector(values, name)
    if not np.isin(array, [0, 1]).all():
        raise ValueError(f"{name} must contain only 0 and 1")
    return array.astype(int)


def _ratio(numerator, denominator):
    # None distinguishes an undefined rate from a measured zero, without NaN.
    return float(numerator / denominator) if denominator else None


def window_metrics(y_true, y_pred, y_score=None):
    """Return pooled binary metrics; class 1 is ictal, class 0 is nonictal.

    Confusion matrix: rows=true, columns=predicted, order=[0, 1]. Scores must
    rank higher for ictal EEG. Average precision is the step-weighted PR summary
    from sklearn, not trapezoidal PR area and not a metric of binary predictions.
    Undefined rates and unavailable AP are None, with reasons in the result.
    """
    truth, prediction = _binary(y_true, "y_true"), _binary(y_pred, "y_pred")
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have matching lengths")
    tn = int(np.sum((truth == 0) & (prediction == 0)))
    fp = int(np.sum((truth == 0) & (prediction == 1)))
    fn = int(np.sum((truth == 1) & (prediction == 0)))
    tp = int(np.sum((truth == 1) & (prediction == 1)))
    result = {
        "n_windows": len(truth), "n_positive": tp + fn,
        "positive_prevalence": float(truth.mean()),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "accuracy": float((tn + tp) / len(truth)),
        "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "average_precision": None,
    }
    undefined = {}
    for name, denominator in (("precision", tp + fp), ("recall", tp + fn),
                              ("specificity", tn + fp), ("f1", 2 * tp + fp + fn)):
        if denominator == 0:
            undefined[name] = "zero denominator"
    if y_score is None:
        undefined["average_precision"] = "no continuous scores supplied"
    else:
        scores = _vector(y_score, "y_score")
        if scores.shape != truth.shape:
            raise ValueError("y_score and y_true must have matching lengths")
        if tp + fn == 0:
            undefined["average_precision"] = "no positive ground-truth windows"
        else:
            result["average_precision"] = float(average_precision_score(truth, scores))
    result["undefined_metrics"] = undefined
    return result


def _timeline(start_sec, end_sec, y_pred, recording_duration_sec):
    starts, ends = _vector(start_sec, "start_sec"), _vector(end_sec, "end_sec")
    prediction = _binary(y_pred, "y_pred")
    if starts.shape != ends.shape or starts.shape != prediction.shape:
        raise ValueError("Window times and predictions must have matching lengths")
    duration = float(recording_duration_sec)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("recording_duration_sec must be finite and positive")
    if (starts < 0).any() or (ends <= starts).any() or (ends > duration).any():
        raise ValueError("Windows must have positive lengths within the recording")
    # Require a complete stream to prevent selected snippets from understating
    # false alarms/hour or counting seizures during missing EEG as model misses.
    if (not np.isclose(starts[0], 0, rtol=0, atol=1e-9)
            or not np.isclose(ends[-1], duration, rtol=0, atol=1e-9)
            or not np.allclose(starts[1:], ends[:-1], rtol=0, atol=1e-9)):
        raise ValueError("Windows must be ordered, contiguous, and cover [0, recording_duration_sec)")
    return starts, ends, prediction


def alarm_episodes(start_sec, end_sec, y_pred, *, recording_duration_sec):
    """Group positive runs using decisions available at each window's END.

    One negative decision ends an episode and immediately re-arms the detector.
    No cooldown, majority rule, smoothing, or gap merging is applied. Episode
    end is the first negative decision time, or recording end (right-censored).
    It is retrospective bookkeeping and never changes the alarm's start time.
    A last-window alert may have equal start/end times; its decision still counts.
    Call separately for every recording so an episode cannot cross a file gap.
    """
    _, ends, prediction = _timeline(start_sec, end_sec, y_pred, recording_duration_sec)
    episodes, active = [], None
    for decision_time, positive in zip(ends, prediction):
        if positive:
            if active is None:
                active = {"alarm_id": len(episodes) + 1,
                          "start_sec": float(decision_time), "positive_windows": 0}
            active["positive_windows"] += 1
            active["last_positive_decision_sec"] = float(decision_time)
        elif active is not None:
            episodes.append({**active, "end_sec": float(decision_time), "right_censored": False})
            active = None
    if active is not None:
        episodes.append({**active, "end_sec": float(recording_duration_sec), "right_censored": True})
    return episodes


def _seizure_intervals(seizures_sec, duration):
    intervals = np.asarray(seizures_sec, dtype=float)
    if intervals.shape == (0,):
        return []
    if intervals.ndim != 2 or intervals.shape[1] != 2 or not np.isfinite(intervals).all():
        raise ValueError("seizures_sec must be a sequence of finite (onset, offset) pairs")
    previous_end = 0.0
    for onset, offset in intervals:
        if not 0 <= onset < offset <= duration or onset < previous_end:
            raise ValueError("Seizures must be ordered, nonoverlapping, and inside the recording")
        previous_end = offset
    return intervals.tolist()


def _event_summary(n_seizures, n_detected, n_false, duration_sec, latencies):
    return {
        "n_seizures": n_seizures, "n_detected": n_detected,
        "n_missed": n_seizures - n_detected,
        "event_sensitivity": _ratio(n_detected, n_seizures),
        "n_false_alarms": n_false, "evaluated_duration_sec": float(duration_sec),
        "evaluated_hours": float(duration_sec / 3600),
        "false_alarms_per_hour": float(n_false * 3600 / duration_sec),
        "latencies_sec": latencies,
        "mean_latency_sec": float(np.mean(latencies)) if latencies else None,
        "median_latency_sec": float(np.median(latencies)) if latencies else None,
    }


def event_metrics(start_sec, end_sec, y_pred, seizures_sec, *, recording_duration_sec):
    """Evaluate one complete recording against exact [onset, offset) annotations.

    An alarm matches only if its START is >= onset and < offset. The first such
    alarm detects that seizure; later starts within it are repeat alarms. Starts
    outside seizures are false alarms, including pre-onset/post-offset starts.
    An alarm already active before onset cannot detect a new seizure. No early
    tolerance, late grace period, or annotation-dependent suppression is used.
    """
    alarms = alarm_episodes(start_sec, end_sec, y_pred, recording_duration_sec=recording_duration_sec)
    duration = float(recording_duration_sec)
    intervals = _seizure_intervals(seizures_sec, duration)
    events = [{"seizure_id": i + 1, "onset_sec": on, "offset_sec": off,
               "detected": False, "alarm_id": None, "latency_sec": None}
              for i, (on, off) in enumerate(intervals)]
    for alarm in alarms:
        alarm.update(status="false_alarm", seizure_id=None)
        for event in events:
            if event["onset_sec"] <= alarm["start_sec"] < event["offset_sec"]:
                alarm["seizure_id"] = event["seizure_id"]
                alarm["status"] = "repeat" if event["detected"] else "detection"
                if not event["detected"]:
                    event.update(detected=True, alarm_id=alarm["alarm_id"],
                                 latency_sec=alarm["start_sec"] - event["onset_sec"])
                break
    latencies = [event["latency_sec"] for event in events if event["detected"]]
    result = _event_summary(len(events), len(latencies),
                            sum(a["status"] == "false_alarm" for a in alarms), duration, latencies)
    result.update(n_alarms=len(alarms), n_repeat_alarms=sum(a["status"] == "repeat" for a in alarms),
                  alarms=alarms, seizures=events)
    return result


def evaluate_predictions(predictions, recording_metadata):
    """Evaluate a DataFrame and explicit metadata, without any file I/O.

    Required columns: recording_id, start_sec, end_sec, label, prediction.
    Optional: score; decision_time_sec (if supplied, must equal end_sec).
    Metadata maps each globally unique recording ID to duration_sec and
    seizures_sec. Exactly the same recordings must occur in both inputs.
    Returns pooled window metrics, per-recording reports, and event totals.
    """
    required = {"recording_id", "start_sec", "end_sec", "label", "prediction"}
    if not isinstance(predictions, pd.DataFrame) or not required <= set(predictions.columns):
        raise ValueError(f"predictions must be a DataFrame with columns {sorted(required)}")
    if not predictions.columns.is_unique:
        raise ValueError("Prediction column names must be unique")
    if predictions.empty or not predictions.recording_id.map(lambda value: isinstance(value, str) and bool(value)).all():
        raise ValueError("Predictions require nonempty string recording IDs and at least one row")
    if not isinstance(recording_metadata, Mapping):
        raise ValueError("recording_metadata must map recording IDs to metadata")
    if set(predictions.recording_id) != set(recording_metadata):
        raise ValueError("Recording IDs must match metadata exactly")
    if "decision_time_sec" in predictions:
        decisions = _vector(predictions.decision_time_sec, "decision_time_sec")
        ends = _vector(predictions.end_sec, "end_sec")
        if not np.array_equal(decisions, ends):
            raise ValueError("decision_time_sec must equal the actual window end_sec")

    def windows(frame):
        return window_metrics(frame.label, frame.prediction, frame.score if "score" in frame else None)

    per_recording = {}
    for recording_id, frame in predictions.groupby("recording_id", sort=False):
        metadata = recording_metadata[recording_id]
        if not isinstance(metadata, Mapping) or not {"duration_sec", "seizures_sec"} <= set(metadata):
            raise ValueError(f"Missing duration_sec or seizures_sec for {recording_id}")
        per_recording[recording_id] = {
            "window_metrics": windows(frame),
            "event_metrics": event_metrics(frame.start_sec, frame.end_sec, frame.prediction,
                                           metadata["seizures_sec"], recording_duration_sec=metadata["duration_sec"]),
        }
    reports = [r["event_metrics"] for r in per_recording.values()]
    latencies = [latency for r in reports for latency in r["latencies_sec"]]
    totals = _event_summary(sum(r["n_seizures"] for r in reports), sum(r["n_detected"] for r in reports),
                            sum(r["n_false_alarms"] for r in reports),
                            sum(r["evaluated_duration_sec"] for r in reports), latencies)
    totals.update(n_alarms=sum(r["n_alarms"] for r in reports),
                  n_repeat_alarms=sum(r["n_repeat_alarms"] for r in reports))
    return {"window_metrics": windows(predictions), "event_metrics": totals,
            "per_recording": per_recording,
            "evaluation_policy": {"decision_time": "window end", "seizure_interval": "[onset, offset)",
                                  "matching": "new alarm start within seizure; first hit only",
                                  "rearm": "one negative decision; no cooldown", "temporal_filter": None,
                                  "false_alarm_denominator": "total evaluated EEG hours, including ictal time"}}
