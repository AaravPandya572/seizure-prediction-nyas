# Evaluation contract

`src/evaluate.py` evaluates supplied predictions in memory. It performs no file
discovery, EDF access, model fitting, threshold tuning, or temporal filtering.
Tests use synthetic arrays only:

```sh
venv/bin/python -m unittest discover -s tests -p 'test_evaluate.py' -v
```

## Public functions

- `window_metrics(y_true, y_pred, y_score=None)` evaluates aligned binary arrays.
- `alarm_episodes(start_sec, end_sec, y_pred, recording_duration_sec=...)` builds
  raw positive-run episodes for one recording.
- `event_metrics(start_sec, end_sec, y_pred, seizures_sec, recording_duration_sec=...)`
  matches those episodes to exact annotations for one recording.
- `evaluate_predictions(predictions, recording_metadata)` returns pooled window
  metrics, pooled event totals, per-recording reports, and the evaluation policy.

All results contain dictionaries/lists and ordinary numeric values; undefined
metrics are `None` (JSON `null`). Results can be saved with
`json.dumps(result, allow_nan=False)`. IDs for alarms and seizures are one-based
within each recording; interpret them together with the recording ID.

## Window metrics

Class 1 is ictal; class 0 is nonictal. The evaluator consumes the existing
preprocessing labels (>=50% ictal overlap); it does not reconstruct or modify
them. Input predictions and scores must be aligned to the same rows as labels.

The confusion matrix is `[[TN, FP], [FN, TP]]`, with true class on rows and
predicted class on columns, both ordered `[0, 1]`.

| Metric | Definition |
| --- | --- |
| Accuracy | `(TP + TN) / N` |
| Precision | `TP / (TP + FP)` |
| Recall / sensitivity | `TP / (TP + FN)` |
| Specificity | `TN / (TN + FP)` |
| F1 | `2*TP / (2*TP + FP + FN)` |
| Average precision (AP) | `sum((recall_i - recall_previous) * precision_i)` over score thresholds |

`average_precision` uses scikit-learn's step-weighted AP definition. It is the
chosen PR summary, distinct from trapezoidal PR-AUC. Supply the continuous
positive-class score, not the hard decisions, to preserve ranking information.
Scores may be probabilities or finite decision scores; higher must mean more
ictal. The evaluator does not require or imply probability calibration.

A zero-denominator rate is `None`, with a reason under `undefined_metrics`.
For example, no positive predictions makes precision undefined. Recall is
undefined when there are no positive labels. AP is `None` without scores or
without positive labels; an all-positive recording has AP=1, which alone says
nothing about discrimination. F1 is zero when there are false positives or
false negatives but no true positives; it is undefined when TP=FP=FN=0.
Empty arrays, nonbinary labels/decisions, mismatched lengths, and nonfinite
inputs are rejected. Pooled window metrics are computed from all supplied rows,
not by averaging recording rates. Keep window lengths consistent for comparison.

## Alarm episode and matching rules

1. Within each recording, process windows in their supplied chronological order.
   A decision becomes available at **`end_sec`**, never at window start or center.
2. The first positive decision when inactive starts an alarm. Consecutive positive
   decisions continue that same episode. One negative decision closes it and
   immediately re-arms the detector. There is no cooldown or smoothing.
3. An episode's reported end is the first negative **decision time**, or the
   recording end if still active (`right_censored=True`). Its last positive
   decision time is also recorded. The endpoint is retrospective bookkeeping;
   future decisions never change when the alarm began. A final-window alert may
   have equal start/end times and is still counted.
4. Each seizure is an exact half-open annotation `[onset_sec, offset_sec)`.
   A **new alarm start** in this interval detects it. Exactly at onset counts;
   exactly at offset does not. There is no early tolerance or late grace period.
5. The first matching alarm detects that seizure once. Additional alarm starts
   inside the same seizure are `repeat`, reported as `n_repeat_alarms`, not extra
   detections or false alarms. They remain visible in the alarm ledger.
6. Alarm starts outside all seizure intervals are `false_alarm`, including
   postictal starts. An alarm that started before onset and continues through a
   seizure remains a false alarm and cannot detect it. One sustained episode
   cannot detect a later seizure without re-arming and a new positive decision.
7. Latency is `first_matching_alarm_start - annotated_onset`, in seconds. Missed
   seizures have `latency_sec=None`; they are never assigned zero latency.
   Mean/median latency uses detected seizures only and must be read beside the
   missed count and event sensitivity.

`n_alarms = n_detected + n_repeat_alarms + n_false_alarms`.
The evaluator returns the alarm and seizure ledgers for inspecting each decision.

## Exposure, recording boundaries, and input validation

False alarms/hour is `number of false alarm episodes / total evaluated EEG hours`.
The denominator includes ictal and nonictal time; it does not include unrecorded
time between EDFs. Across recordings it is **sum of false alarms / sum of hours**,
not the mean of individual rates. Event sensitivity is detected/annotated seizures,
or `None` if no seizures were evaluated. Counts remain available in that case.

Each recording must have a complete ordered timeline covering `[0, duration_sec)`
with finite, positive-length, contiguous, nonoverlapping windows. Equality at
adjacent boundaries is checked to an absolute tolerance of 1e-9 seconds; seizure
matching uses the exact supplied times. Missing snippets, omitted initial/final
windows, overlaps, duplicates, and shuffled windows fail rather than silently
inflate exposure or treat unobserved seizures as misses. The current complete
one-hour files meet this contract. Recordings whose incomplete trailing window
was dropped by preprocessing are intentionally rejected until an explicit
evaluated-coverage policy is added; do not simply inflate their duration.

Seizure annotations must be finite, ordered, nonoverlapping onset/offset pairs
inside the recording. An explicit empty list means no seizures; missing annotation
metadata is an error. Adjacent events are allowed. Recordings are evaluated
independently with fresh alarm state. A last-window positive in one file and
first-window positive in another are separate episodes, even at similar clock times.

## Model handoff

The DataFrame must contain:

| Column | Meaning |
| --- | --- |
| `recording_id` | Globally unique ID, e.g. `chb01_03`; do not reuse across patients |
| `start_sec`, `end_sec` | Window times relative to that recording |
| `label` | Existing binary ictal ground truth |
| `prediction` | Binary model decision using an already selected threshold |
| `score` (optional) | Positive-class continuous score used for AP |
| `decision_time_sec` (optional) | Must equal `end_sec` if included |

Metadata is keyed by exactly the IDs present in the prediction table:

```python
metadata = {
    "chb01_03": {"duration_sec": 3600.0, "seizures_sec": [(2996.0, 3036.0)]},
}
```

The preprocessing JSON keys its recording entries by **filename**. For future
model output on an already processed, held-out table `test_df`, build the handoff
as follows (this is illustrative, not a training command):

```python
from src.evaluate import evaluate_predictions

# scores must be in exactly test_df's row order. Choose threshold on validation
# data beforehand; never adjust it based on this evaluator's final test results.
predictions = test_df[
    ["recording_id", "start_sec", "end_sec", "decision_time_sec", "label"]
].copy()
predictions["score"] = scores
predictions["prediction"] = (scores >= threshold).astype(int)
metadata = {
    row.recording_id: {
        "duration_sec": provenance["recordings"][row.source_file]["duration_sec"],
        "seizures_sec": provenance["recordings"][row.source_file]["seizures_sec"],
    }
    for row in test_df[["recording_id", "source_file"]].drop_duplicates().itertuples()
}
result = evaluate_predictions(predictions, metadata)
```

Use only feature columns for model inputs; annotations and labels belong only in
evaluation. The caller is responsible for holding out whole recordings and for
causal prediction generation. The evaluator cannot prove a model avoided future
information. It does not check that supplied labels agree with annotations.
Its decision-time latency includes window accumulation, but does not include
model runtime, transmission, or display delay.

## Runnable synthetic example

```python
import pandas as pd
from src.evaluate import evaluate_predictions

predictions = pd.DataFrame({
    "recording_id": ["synthetic"] * 4,
    "start_sec": [0, 5, 10, 15], "end_sec": [5, 10, 15, 20],
    "label": [0, 1, 1, 0], "prediction": [0, 1, 0, 1],
    "score": [.1, .9, .4, .8],
})
result = evaluate_predictions(predictions, {
    "synthetic": {"duration_sec": 20, "seizures_sec": [(6, 16)]},
})
assert result["window_metrics"]["confusion_matrix"] == [[1, 1], [1, 1]]
assert result["event_metrics"]["latencies_sec"] == [4.0]
assert result["event_metrics"]["false_alarms_per_hour"] == 180.0
```

These deliberately tiny examples verify arithmetic, not detector performance.
With five-second decisions, a short seizure may end before any decision occurs;
it then counts as missed even if a positive window overlaps it. Window recall
and event sensitivity can therefore disagree. Results depend on the stated
episode/matching rules, and small numbers of independent seizures cannot support
strong generalization claims. Repeated alarms and sustained false-alarm duration
should be inspected in addition to false alarm counts. Temporal filters and
alternative matching tolerances require separate, predeclared comparisons later.
