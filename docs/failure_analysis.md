# Phase 2E: why chb01_15 was missed

Both saved out-of-fold forests assign subthreshold scores throughout the
annotated seizure `[1732,1772)` seconds. Their only positive window is
`[1780,1785)`, fully after offset, and their only alarm starts at its end,
1785 seconds: **13 seconds after offset and 53 seconds after onset**. The
models respond during the seizure but peak afterward. The late response
coincides with a transient high-frequency/line-length increase; this is a clue,
not proof of a causal feature mechanism or of an EEG artifact.

## Inspected material and verification

Reviewed the Git status, existing CV implementation/report, split metadata,
fold definitions, all eight saved fold models, prediction tables, evaluator
reports and exact annotations. Existing training-only loading reads only the
original six training recordings from the processed dataset. Saved predictions
match original window identities/times/labels exactly, reproduce from all eight
loaded models, and reproduce the original evaluation reports. No models were
fitted, no thresholds or feature definitions changed, and no previous outputs
were rewritten. Original validation/test recordings and new `chb01_23`–`25`
were not opened or used. No EDF was opened for this analysis.

## Scores and delayed alarms

For `chb01_15`, the 60 seconds before onset have mean scores 0.000417 ordinary
and 0 weighted. During the event they rise irregularly, not monotonically:

| Window end (s) | Ordinary | Weighted |
| --- | ---: | ---: |
| 1735 | 0.000 | 0.015 |
| 1740 | 0.310 | 0.035 |
| 1745 | 0.025 | 0.025 |
| 1750 | 0.080 | 0.165 |
| 1755 | 0.125 | 0.185 |
| 1760 | 0.325 | 0.040 |
| 1765 | 0.395 | 0.015 |
| 1770 | 0.375 | 0.010 |
| 1775 (after offset) | 0.385 | 0.035 |
| 1780 | 0.090 | 0.015 |
| 1785 (alarm) | 0.520 | 0.680 |
| 1790 | 0.050 | 0.025 |
| 1795 | 0.235 | 0.455 |

| Decision-time period | Windows | Ordinary mean / median / maximum | Weighted mean / median / maximum |
| --- | ---: | --- | --- |
| Pre60: [1672,1732) | 12 | 0.000417 / 0 / 0.005 | 0 / 0 / 0 |
| During: [1732,1772) | 8 | 0.204375 / 0.2175 / 0.395 | 0.061250 / 0.0300 / 0.185 |
| Post60: [1772,1832) | 12 | 0.155833 / 0.1000 / 0.520 | 0.175833 / 0.0725 / 0.680 |
| Other decision times | 688 | 0.000676 / 0 / 0.050 | 0.000560 / 0 / 0.035 |

The recording-wide maxima occur at 1785 for both forests. Each alarm episode
lasts from 1785 until the first negative decision at 1790, with one positive
window. `[1780,1785)` lies 8–13 seconds after offset and is inside the existing
60-second postictal interval. It overlaps **zero** annotated seizure seconds.
The preceding `[1770,1775)` overlaps two seizure seconds but is correctly
nonictal under the unchanged >=50% window rule, and neither model calls it
positive anyway. No positive predicted window straddles the boundary.

Primary evaluation is unchanged: new alarm starts must fall within exact
`[onset,offset)`. Both therefore miss the seizure, have undefined detection
latency and generate one false alarm/hour. Their matrices remain
`[[711,1],[8,0]]`. The value 13 seconds is a descriptive post-offset delay,
**not** a detected-seizure latency. Giving a late grace period would change the
scientific endpoint; no such relabeling or secondary credit is applied here.

The fixed threshold explains the decision mechanically, since all ictal scores
are below it, but does not explain why the scores are low. No lower thresholds
were searched, evaluated or recommended from this held-out event.

## Comparison with detected seizures

All scores below are from each recording's own held-out fold model. They are
uncalibrated scores, not clinical probabilities, and different folds use
models trained on different recordings.

| Recording | Annotated interval (s) | Ictal windows | Ordinary mean / max ictal score | Weighted mean / max ictal score | Recall ordinary / weighted | First alarm; latency |
| --- | --- | ---: | --- | --- | --- | --- |
| chb01_03 | [2996,3036) | 8 | 0.571875 / 0.890 | 0.783750 / 0.990 | 75% / 87.5% | 3005; 9 s |
| chb01_04 | [1467,1494) | 6 | 0.683333 / 0.970 | 0.712500 / 0.975 | 66.67% / 66.67% | 1475; 8 s |
| chb01_15 | [1732,1772) | 8 | 0.204375 / 0.395 | 0.061250 / 0.185 | 0% / 0% | 1785, false alarm; undefined latency |
| chb01_16 | [1015,1066) | 10 | 0.637000 / 0.910 | 0.744500 / 0.980 | 70% / 80% | 1020; 5 s |

The saved `score_periods.json` contains pre/during/post/remote distributions for
all four recordings and both models, including ranges and peak decision times.
Decision-time bins differ from ictal **window labels** at annotation boundaries:
`chb01_04` has five decisions during the exact event but six ictal-labeled
windows; `16` has eleven during-event decisions but ten ictal windows. These
descriptive bins never replace labels or evaluator definitions.

## EEG feature comparisons

All 126 features are described separately by recording and period, including
median, IQR and min/max. No learned scaler, PCA or new predictor is used.
Figures summarize each window by the median across its 18 channels for each
of the seven fixed feature families; these summaries are descriptive only.
Cross-channel medians of relative powers need not sum to one.

Feature periods are: original ictal label; fully postictal windows with
`start >= offset` and `end <= offset+60`; and remote nonictal windows outside
60 seconds before onset/after offset, excluding the first ten filter-warmup
seconds. Boundary-straddling windows are excluded from the fully postictal
feature group. Counts are stored with each statistic. Normal here means
annotation-based nonictal, not clinician-confirmed artifact-free EEG.

| Per-window channel-median feature, then median over ictal windows | chb01_03 | chb01_04 | chb01_15 | chb01_16 |
| --- | ---: | ---: | ---: | ---: |
| Variance (µV²) | 8561.88 | 6851.31 | 9584.14 | 9621.10 |
| Line length (µV, fixed window) | 10624.04 | 9904.24 | 8047.60 | 12171.83 |
| Relative delta | 0.76234 | 0.77198 | 0.85555 | 0.85518 |
| Relative theta | 0.16568 | 0.16449 | 0.12063 | 0.08753 |
| Relative alpha | 0.01324 | 0.01938 | 0.01357 | 0.01120 |
| Relative beta | 0.01324 | 0.01580 | 0.00620 | 0.01452 |
| Relative gamma | 0.00651 | 0.00691 | 0.00054 | 0.01294 |

`15` has lower line length and gamma than the other events, but not lower
variance. It is not uniformly weak across the feature space. Its remote variance
baseline is higher (1883.54 versus 973.03 in `03` and 638.00 in `04`), but similar
to `16` (1836.73). Ictal/remote variance ratios are 8.80, 10.74, 5.09 and 5.24;
`16` is detected despite a contrast similar to `15`. This weakens a simple
variance-contrast explanation. Line-length ratios are 2.06, 2.02, 1.70 and 2.53.

The low beta/gamma levels in `15` are partly recording-wide: its remote beta is
0.00999, near `16` at 0.00978 and below `03`/`04` at 0.04097/0.05558. Remote
gamma is also similar in `15` and `16` (0.00172 versus 0.00177), yet ictal gamma
falls to 0.00054 in `15` and rises to 0.01294 in `16`. Thus a recording-wide shift
alone does not explain all spectral differences. Relative power is compositional;
a smaller gamma fraction does not alone prove lower absolute gamma power.

Illustrative channel-specific contrasts (post hoc, not model feature selection):

- `T8-P8_line_length`: `15` ictal median 10069.35 versus median of the other
  three event medians 30622.43. Remote medians are similar: 3814.19 versus 3822.87.
  Ictal/remote ratio is 2.64 versus median reference ratio 7.97.
- `P7-O1_var`: 7725.46 versus reference 2107.95, but remote levels also differ
  (1456.53 versus 644.14). Raw ictal contrast alone overstates uniqueness.
- `FP2-F8_gamma`: ictal 0.000228 versus reference event-median 0.05804;
  within-record change is -0.00176 versus reference median change +0.04269.
- `T7-P7_theta`: ictal 0.07521 versus reference 0.22723; within-record change
  -0.10241 versus reference +0.01503.

Reference comparisons weight the three other events equally, not their window
counts. `feature_differences.parquet` preserves all 126 contrasts, including
raw medians, background levels, within-record changes and ratios; no feature
was selected to modify a model, and no significance or causal claim is made.

## Postictal feature burst and signal quality

At the alarm-producing `[1780,1785)` window, `FP2-F8_gamma` is 0.22068 versus
its ictal median 0.000228; `T8-P8_line_length` is 27401.71 versus ictal median
10069.35. The channel-median gamma is 0.01329 versus ictal median 0.00054.
These measurements coincide with score peaks 0.520 and 0.680 and resemble
some higher-frequency/line-length levels observed in detected events. They
provide a plausible feature-space account of the late alert, but individual
feature contributions were not established and many features change together.
The line-length increase also persists across later windows without positive
predictions, so one feature value alone is not a sufficient explanation.

All features are finite, and no window in the original training pool has a
low-power-channel flag. Two initial warmup windows exist per recording, far
from the analyzed seizures. These checks argue against a numerical or near-flat
signal failure; they do not detect motion, muscle, electrode or clipping
artifacts. A postictal high-frequency burst could be physiological, artifactual
or mixed. Extracted features alone cannot distinguish these possibilities;
raw EEG and annotation review would be needed. No artifact-free claim is made.
Causal filter phase effects also cannot be isolated here; the observed 13-second
delay must not be attributed to filtering without further evidence.

## Figures and limitations

- `scores_missed.png`: entire `15` recording and local zoom, exact annotations,
  default boundary and every alarm start.
- `scores_comparison.png`: both models around each of the four events.
- `feature_families.png`: remote/ictal/postictal feature-family distributions.
- `feature_timelines.png`: three illustrative channel features around `15`.

Markers are window-end observations; connecting lines are visual guides, not
additional decisions or interpolation used by the detector. Box-plot outliers
are hidden for readability but all feature ranges are retained in tables.

Four correlated seizures in one patient cannot establish a general failure
mechanism. Fold 3 trained on only three other seizures (24 ictal windows).
Inter-seizure variability and sparse positive examples are plausible limitations,
not proven causes. Reusing held-out events for this exploratory analysis can
inform hypotheses but cannot supply independent evidence for a redesigned model.
Future modifications would need a predeclared evaluation protocol; original
validation/test recordings remain protected.

The next useful analysis is a targeted raw-EEG and annotation review of `15`
around 1732–1800, especially the post-offset high-frequency burst, with a
qualified EEG reviewer if available. Check physiological versus artifact
explanations before changing features or choosing another classifier. This
recommendation was not performed; no new model or threshold experiment was run.

## Reproduction and checks

Added `src/failure_analysis.py`, four focused synthetic tests in
`tests/test_failure_analysis.py`, this report and a README link. Analysis outputs
are separate under `results/failure_analysis_v1/`. Existing experiments and
uncommitted work are preserved. No dependencies installed, commits or pushes.

```sh
MPLCONFIGDIR=/tmp/nyas-matplotlib venv/bin/python -m src.failure_analysis \
  --output results/failure_analysis_reproduction
venv/bin/python -m unittest discover -s tests -p test_failure_analysis.py -v
venv/bin/python -m unittest discover -s tests -p test_evaluate.py -v
```

Use a fresh output directory; overwrite is refused. All 29 analysis/evaluation
tests passed (4 new + 25 existing). Training tests were not run in this
analysis-only phase. Eight saved model predictions and evaluator reports were
verified. The analysis fits no model or learned transformation and retains all
existing feature, label, threshold and event definitions. Artifact hashes,
verification, tests and a before/after preservation audit accompany the outputs.
