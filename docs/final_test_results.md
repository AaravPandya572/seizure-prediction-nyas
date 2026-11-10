# First final held-out evaluation — Phase 3C

Executed once on **2026-11-10**, following explicit authorization and frozen
protocol `nyas-p01-final-evaluation-v1`. Both saved models detected **both test
seizures**, with first-detection delays of **8 and 13 seconds**. Neither produced
a false-positive window or false-alarm episode over **2.645833 hours**. Each
missed nine of 40 ictal windows. LightGBM produced one repeat ictal alarm;
Random Forest produced three. These are observed results from two seizures in
one patient, not evidence of perfect sensitivity or a zero underlying alarm rate.

No research model was trained, replaced or adjusted. Supplementary recordings
`chb01_23`, `chb01_24`, and `chb01_25` were not opened or evaluated. This phase
does not authorize further model development.

## Protocol verification and model identity

The working tree was clean before execution at commit
`8798eb59f22716edbbcb0a7e981c0fc92bb1e687` (`main`). That commit preserves the
previously uncommitted development work described in the freeze. It descends
from the recorded freeze-time HEAD
`a1359fa4b64f94dd4eab45c82b1d03952f3d79f6`. All frozen content hashes match;
the newer commit is a provenance improvement, not a method change.

The authoritative [protocol](final_evaluation_protocol.md) and
[configuration](../config/final_evaluation_protocol.json) were unchanged.
Configuration SHA-256:
`cb5b77949755ba0aaf796e6199f388a83a195257e5f3b9459d136bbf2d4c7f3b`.
Protocol document SHA-256:
`a23efac5d6214d2ee5db32abb6b0e8bfc63022fcb1750525614fe4be3f1ceae4`.

| Role | Saved artifact | Fixed configuration |
| --- | --- | --- |
| Primary | `results/lightgbm_v1/ordinary_lgbm.joblib` | Phase 2G ordinary LightGBM; 150 trees, learning rate 0.05, depth 3, seven leaves, minimum child samples 10, seed 42, scale_pos_weight 1.0 |
| Reference | `results/baseline_v1/random_forest.joblib` | Phase 2B ordinary Random Forest; 200 trees, seed 42, class_weight None; all original settings |

LightGBM model SHA-256:
`7324d3199850abd2e78d9190ff08946ecfd056e1a32e4b5d0f901d519b7b2b5f`.
Random Forest model SHA-256:
`05fb53387283e82892aed6885469070fc85782f2155b460ba6b91cc4534d60c0`.
Exact full parameters and original configuration hashes remain in the frozen JSON.

Both models reloaded with the expected classes `[0,1]`, exact parameters and
126 ordered feature names. The six training IDs remain `chb01_01`, `03`, `04`,
`14`, `15`, `16`: 4,320 rows, 32 ictal windows, four events. Original training
configurations, sources, data hashes, LightGBM's original training fingerprint
and the models' training sample counts agree. The original RF binary does not
embed row identities; its training scope is supported by the preserved code,
configuration and data provenance. Saved development prediction artifacts and
CV folds contained no reserved test or supplementary IDs. No test-based model
selection is recorded in the verified development workflow.

Dataset, manifest, summary, sources/tests, model configurations, model binaries,
package versions and native libraries all matched the freeze. The three primary
EDFs matched their earlier SHA-256 integrity records. The exact evaluation block
was extracted unchanged from the frozen document; its SHA-256 is
`1f26887237c0508c3c00714f00732f26b5c5c72728c71293830bf2e71575b65a`.
`results/final_test_v1/` did not exist before execution.

The fixed decision remains **score > 0.5**, with exact ties assigned class 0;
no filtering or calibration. Inputs retain five-second nonoverlapping windows,
causal preprocessing and original ictal-overlap labels. No metadata is used as a
predictor. Alarm decisions occur at actual window ends. A new alarm must start
inside annotated `[onset, offset)`; an already-active early alarm or post-offset
start cannot detect the seizure. One negative decision re-arms detection; repeats
inside the same event are reported separately. Definitions were not changed.

## Dataset and exposure

Input: `data/processed/chb01_detection_features_11recordings_split.parquet`,
SHA-256 `957cda683c9f09f1912e22d264f715904861e3bb8ab044d657485d13ef121ae0`.
Only rows assigned `split="test"` for the three frozen IDs entered inference.
The natural class distribution was retained: **1,865 class-0 and 40 class-1
windows**, or **2.0997% ictal**. Both models used the same aligned 1,905 rows.

| Recording | Duration (s) | Hours | Windows | Class 0 | Class 1 | Events |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chb01_21 | 3,600 | 1.000000 | 720 | 701 | 19 | 1 |
| chb01_22 | 3,600 | 1.000000 | 720 | 720 | 0 | 0 |
| chb01_26 | 2,325 | 0.645833 | 465 | 444 | 21 | 1 |
| **Total** | **9,525** | **2.645833** | **1,905** | **1,865** | **40** | **2** |

The shorter `chb01_26` recording contributes 465 windows, not 720. All timelines
are ordered, contiguous, duplicate-free and complete through the actual recording
end. No trailing samples or difficult windows were excluded.

## Pooled results

Window percentages below are rounded; saved JSON retains full precision.
AP is the existing step-weighted average precision on uncalibrated scores.

| Metric | LightGBM primary | Random Forest reference |
| --- | ---: | ---: |
| Accuracy | 99.5276% | 99.5276% |
| Precision | 100.0000% | 100.0000% |
| Recall | 77.5000% | 77.5000% |
| Specificity | 100.0000% | 100.0000% |
| F1 | 87.3239% | 87.3239% |
| Average precision | 0.947176 | 0.942249 |
| TN / FP / FN / TP | 1865 / 0 / 9 / 31 | 1865 / 0 / 9 / 31 |
| Annotated / detected / missed seizures | 2 / 2 / 0 | 2 / 2 / 0 |
| Observed event sensitivity | 100% (2/2) | 100% (2/2) |
| Per-event detection delays | 8 s, 13 s | 8 s, 13 s |
| Mean / median detection latency | 10.5 s / 10.5 s | 10.5 s / 10.5 s |
| False-positive windows | 0 | 0 |
| False-alarm episodes | 0 | 0 |
| False-alarm episodes/hour | 0.0000 | 0.0000 |
| All alarm episodes | 3 | 5 |
| Repeat ictal alarm episodes | 1 | 3 |

Both pooled confusion matrices are `[[1865, 0], [9, 31]]`, with true rows,
predicted columns and class order `[0,1]`. The pooled totals do **not** mean
identical predictions: four window decisions differ. Two ictal windows are
positive only for LightGBM and two only for RF; neither introduces a false positive.
The exact differences are saved in `prediction_differences.csv`.

![Confusion matrices with counts and true-class percentages](/Users/aaravpandya/seizure-prediction-nyas/results/final_test_v1/figures/confusion_matrices.png)

## Per-recording window results

Accuracy, precision, recall, specificity and F1 are percentages. U = undefined.

| Model | Recording | Accuracy | Precision | Recall | Specificity | F1 | AP | TN / FP / FN / TP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LightGBM | chb01_21 | 99.17 | 100 | 68.42 | 100 | 81.25 | 0.949826 | 701 / 0 / 6 / 13 |
| RF | chb01_21 | 99.31 | 100 | 73.68 | 100 | 84.85 | 0.929146 | 701 / 0 / 5 / 14 |
| LightGBM | chb01_22 | 100 | U | U | 100 | U | U | 720 / 0 / 0 / 0 |
| RF | chb01_22 | 100 | U | U | 100 | U | U | 720 / 0 / 0 / 0 |
| LightGBM | chb01_26 | 99.35 | 100 | 85.71 | 100 | 92.31 | 0.965038 | 444 / 0 / 3 / 18 |
| RF | chb01_26 | 99.14 | 100 | 80.95 | 100 | 89.47 | 0.959788 | 444 / 0 / 4 / 17 |

`chb01_22` has no seizures and no positive decisions. For both models, window
precision/recall/F1/AP, event sensitivity, and detection latency are undefined,
represented by JSON `null` with reasons. Its 100% accuracy/specificity and zero
observed false alarms are defined. Pooled metrics are all defined.

## Seizure outcomes and exact alarm timestamps

Every time below is seconds from the start of its recording, using actual
window-end decision times. Both detections occur strictly before seizure offset.
Neither model detects an event missed by the other; neither is earlier.

| Recording / event | Annotation [onset, offset) | LightGBM first alarm | RF first alarm | Latency for both | Detected / missed for both |
| --- | --- | ---: | ---: | ---: | --- |
| chb01_21 / 1 | [327, 420) s | 335 s | 335 s | 8 s | 1 / 0 |
| chb01_26 / 1 | [1862, 1963) s | 1875 s | 1875 s | 13 s | 1 / 0 |

Full episode bookkeeping is shown here; the end timestamp is the first negative
decision, not the last positive decision. These episode ends do not alter starts
or create post-offset detections.

| Model | Recording | Alarm start (s) | Episode end (s) | Last positive decision (s) | Positive windows | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| LightGBM | chb01_21 | 335 | 395 | 390 | 12 | Detection |
| LightGBM | chb01_21 | 400 | 405 | 400 | 1 | Repeat |
| RF | chb01_21 | 335 | 385 | 380 | 10 | Detection |
| RF | chb01_21 | 390 | 405 | 400 | 3 | Repeat |
| RF | chb01_21 | 410 | 415 | 410 | 1 | Repeat |
| LightGBM | chb01_26 | 1875 | 1965 | 1960 | 18 | Detection |
| RF | chb01_26 | 1875 | 1880 | 1875 | 1 | Detection |
| RF | chb01_26 | 1885 | 1965 | 1960 | 16 | Repeat |

There are no alarms of any status in `chb01_22`. False alarms and false-positive
windows are zero for **each** recording/model. The pooled rate is
`0 / (9525 / 3600) = 0` false-alarm episodes/hour, including ictal exposure.

Successful event detection does not mean all seizure windows were classified
correctly: LightGBM misses six windows in `chb01_21` and three in `chb01_26`;
RF misses five and four, respectively. One negative decision can split an
otherwise detected seizure into repeated alarms. LightGBM has fewer repeats in
this sample without a persistence rule. No new filter was tested on these data.

![Model scores at actual window-end times around each seizure](/Users/aaravpandya/seizure-prediction-nyas/results/final_test_v1/figures/seizure_score_timelines.png)

Scores are uncalibrated. Dots mark actual decisions; step segments hold the last
available value. The shaded intervals are the unchanged annotations, and the
vertical dotted lines mark the shared first alarm times.

![Exact first and repeat alarm starts relative to seizure onset](/Users/aaravpandya/seizure-prediction-nyas/results/final_test_v1/figures/alarm_timing.png)

## Comparison with frozen development findings

Development numbers below are copied from preserved reports; no model was
refitted or rerun on development data. Training CV used four held-out training
recordings and fold-specific fitted estimators, rather than the final saved
estimators. Its pooled out-of-fold AP is descriptive of scores across those folds.
Validation contained one event; this final test contains two different events.

| Assessment | Model | Detected / annotated | Window recall | F1 | AP | False alarms/hour | Mean latency (s) | Repeats |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Training-recording CV, 4 h | LightGBM | 4/4 | 65.625% | 75.00% | 0.867054 | 0.50 | 7.50 | 2 |
| Training-recording CV, 4 h | RF | 3/4 | 53.125% | 68.00% | 0.844954 | 0.25 | 7.33 | 2 |
| Original validation, 2 h | LightGBM | 1/1 | 55.556% | 68.97% | 0.847912 | 0.50 | 5.00 | 0 |
| Original validation, 2 h | RF | 1/1 | 61.111% | 75.86% | 0.982948 | 0.00 | 10.00 | 2 |
| Final test, 2.646 h | LightGBM | 2/2 | 77.500% | 87.32% | 0.947176 | 0.00 | 10.50 | 1 |
| Final test, 2.646 h | RF | 2/2 | 77.500% | 87.32% | 0.942249 | 0.00 | 10.50 | 3 |

The CV event-detection advantage for LightGBM did **not** remain a separating
result on the final test: event detection, latency, recall, F1 and false alarms
tie. LightGBM has slightly higher AP and fewer repeat alarms here, which does
not establish broad superiority. RF's earlier false-alarm advantage also becomes
a tie. Keep the preselected primary/reference roles unchanged.

Both models have higher window recall here than on validation, while LightGBM
loses its validation latency advantage and both have a 10.5-second mean delay.
Validation, CV and test favor different aspects of each model. This demonstrates
recording-dependent performance and supports reporting several complementary
metrics; it does not prove either model generalizes reliably beyond Patient 01.
Different prevalence (validation 1.25%, CV 1.11%, test 2.10%), event composition
and small event counts also limit direct AP/precision comparisons. No additional
experiment was run to explain these outcomes.

![Event, false-alarm and repeat-alarm summary](/Users/aaravpandya/seizure-prediction-nyas/results/final_test_v1/figures/event_false_alarm_summary.png)

## Execution, environment and outputs

The exact frozen shell block is archived as
`results/final_test_v1/execution_command.sh`. It was invoked once using:

```sh
/bin/bash /tmp/nyas_phase3c_8798eb5/execution_command.sh
```

The archive is byte-identical to that temporary execution file and the original
block in the frozen protocol. Its contents include the checksum check followed
by `NYAS_FINAL_TEST_AUTHORIZED=YES venv/bin/python - <<'PY'` and the unchanged
evaluation code. Execution began at `2026-11-10T09:54:24.761328+00:00` and
completed at `2026-11-10T09:54:26.839332+00:00` with exit status 0. The command
calls `predict` and `predict_proba` once each per selected model as specified;
no additional research-model predictions were requested.

Do not rerun inference to reproduce tables or figures. The equivalent archived
command `/bin/bash results/final_test_v1/execution_command.sh` intentionally
refuses the now-existing output directory. Do not remove the guard or overwrite
the first outputs.

Environment matched the freeze: Python 3.14.5, macOS 26.3.2 ARM64, LightGBM 4.7.0,
scikit-learn 1.9.1, NumPy 2.5.4, SciPy 1.18.1, pandas 3.0.6, joblib 1.6.0,
PyArrow 26.0.0, Matplotlib 3.11.2, MNE 1.13.2 and libomp 23.1.4. Native library
hashes are in the configuration and pre-execution log. No dependencies were installed.

Saved under `results/final_test_v1/`:

- `primary_predictions.parquet`, `reference_predictions.parquet`: all aligned
  window IDs, labels, times, original decisions and scores.
- `primary_evaluation.json`, `reference_evaluation.json`: full pooled/per-record
  metrics, undefined reasons, alarms and seizure ledgers.
- `dataset_counts.csv`, `window_metrics.csv`, `event_metrics.csv`,
  `seizure_outcomes.csv`, `alarm_episodes.csv`, `prediction_differences.csv`,
  `development_comparison.csv`: readable tables derived only from saved outputs.
- `figures/`: four figures in PNG and editable SVG; the report script reads saved
  predictions/reports only and does not load any fitted estimator.
- Frozen configuration copy, exact execution command/log/record, pre-execution
  verification and source, before/after integrity evidence, verification tests/logs,
  original-output checksums and final delivery checksums.

## Post-evaluation verification

**62 checks passed, zero failures, zero skips**:

- 54 existing preprocessing, evaluation and split checks, including the previously
  deferred complete expanded-dataset integration test. No selected-model fitting
  or inference is part of these checks.
- Eight saved-output checks: exact metadata/alignment and labels; short-recording
  lengths/exposure; finite scores and threshold semantics; unchanged-evaluator
  report reproduction from saved predictions; alarm/re-arming/matching/latency;
  undefined no-seizure metrics; preservation of historical inputs/results; and
  byte-identical frozen command/first outputs.

The verification recomputes metrics from already saved decisions without calling
either research model. All **271** snapshotted historical source, test, protocol,
dataset, primary raw EDF and earlier-result files retained their hashes.
The originally generated final prediction/report files also retain their first
checksums. No earlier experiment, preprocessing implementation, label, evaluator,
split or model artifact was changed. No commit or push was made.

Commands to rerun these checks or regenerate presentation artifacts **without
generating new predictions**:

```sh
PYTHONPATH=tests venv/bin/python -m unittest test_evaluate test_preprocess test_split_dataset -v
PYTHONPATH=. venv/bin/python results/final_test_v1/test_final_outputs.py
MPLCONFIGDIR=/tmp/nyas_phase3c_matplotlib venv/bin/python results/final_test_v1/create_figures_and_tables.py
```

There was no protocol mismatch or evaluation fault. The first plotting run
reported an unwritable default Matplotlib cache and used a temporary fallback;
final figure rendering used an explicit writable temporary cache. This affected
neither predictions nor metric definitions. Figures were visually checked for
timing, readability and unclipped labels.

## Defensible interpretation and presentation recommendation

Present this as **two independently held-out recordings with seizures, plus one
seizure-free test recording, all from the same patient**. Both models detected
2/2 events after 8 and 13 seconds with no false alarms observed over 2.646 hours.
Their 77.5% window recall and repeat alarms show why successful event detection
is different from continuous seizure segmentation. Present the development
failure analysis and filtering trade-off alongside this favorable final result;
do not claim that the test erases the earlier missed seizure or proves a winner.

The test was reserved during model selection, but recordings share a patient,
acquisition setting and potentially correlated physiology. Two events are too
few for a precise sensitivity estimate; 2.646 hours is too little to establish
a low long-term false-alarm rate. Windows are correlated, scores uncalibrated,
and offline latency excludes computation, transport and clinical workflow.
No patient-independent, prospective or clinical validation claim is supported.
Five-second labels/decision times also limit boundary precision.

Retain ordinary LightGBM as the frozen primary and RF as reference. After review
and separate authorization, assess `chb01_23`, `24`, `25` solely for supplementary
false alarms with the same artifacts, threshold and rules. Report their exposure
and alarm timestamps separately; do not merge their hours into this primary
table, use them for tuning, or imply that seizure-free data can measure seizure
sensitivity or latency. No supplementary evaluation has been performed here.
