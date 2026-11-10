# Phase 2D: training-only recording-level RF stability

Both forests detected the same three of four held-out seizures, missed
`chb01_15`, and had identical first detection delays. Weighting increased pooled
window recall from 53.125% to 59.375%, but increased false-alarm episodes from
one to four and reduced macro F1 and AP. This is exploratory recording-level
stability analysis within one patient, not prospective chronological validation.

## Methodology and scope

`src/rf_cross_validation.py` reads only `split == "train"` rows from the existing
11-recording Parquet. The earlier train/validation loader is deliberately not
called. Data, frozen manifest, feature order, baseline parameter settings,
methodology source and library versions are checked against existing hashes.
Only the ordered 126 EEG features enter model fitting; target, annotations,
overlap, identifiers, timestamps, split and quality metadata are excluded.
Complete five-second timelines, labels and annotations are validated before fitting.
Whole-file processed-data hashing is an integrity check without decoding protected
rows. No original validation/test predictions or reports are loaded.

Each fold trains fresh ordinary and weighted estimators on identical matrices and
labels. Both use 200 trees, seed 42, four workers and all other original defaults.
Only `class_weight` differs: `None` versus `"balanced"`. Balanced weights use each
fold's training labels only. No resampling, scaling changes, tuning, threshold
search or filtering is performed. The standard sklearn decision is score >0.5,
with exact ties choosing class 0. Eight independent models, input fingerprints,
full parameters, versions and model reload verification are saved.

The user reports completed seizure-free raw files `chb01_23.edf`, `chb01_24.edf`
and `chb01_25.edf`. Their existence is documented from that supplied information;
they were not opened, inspected, preprocessed, assigned splits or included here.
They remain reserved for later supplementary false-alarm evaluation. No EDFs were
modified, and the original 11-recording assignments are unchanged.

## Fold definitions and observed results

Each fold fits three distinct annotated events and evaluates one. Both normal
recordings remain in every training portion. Each held-out recording contributes
its entire one-hour timeline exactly once per model: 2,880 unique evaluated
windows, 2,848 nonictal and 32 ictal, four hours and four events. Training sets
overlap across folds and must not be pooled as independent observations.

| Fold | Holdout | Training recordings | Train windows | Train ictal | Evaluated windows | Evaluated ictal |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | chb01_03 | chb01_01, chb01_04, chb01_14, chb01_15, chb01_16 | 3600 | 24 | 720 | 8 |
| 2 | chb01_04 | chb01_01, chb01_03, chb01_14, chb01_15, chb01_16 | 3600 | 26 | 720 | 6 |
| 3 | chb01_15 | chb01_01, chb01_03, chb01_04, chb01_14, chb01_16 | 3600 | 24 | 720 | 8 |
| 4 | chb01_16 | chb01_01, chb01_03, chb01_04, chb01_14, chb01_15 | 3600 | 22 | 720 | 10 |

| Holdout | Model | Accuracy | Precision | Recall | Specificity | F1 | AP | Confusion matrix |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| chb01_03 | ordinary | 0.997222 | 1.000000 | 0.750000 | 1.000000 | 0.857143 | 0.860764 | [[712, 0], [2, 6]] |
| chb01_03 | weighted | 0.997222 | 0.875000 | 0.875000 | 0.998596 | 0.875000 | 0.876389 | [[711, 1], [1, 7]] |
| chb01_04 | ordinary | 0.997222 | 1.000000 | 0.666667 | 1.000000 | 0.800000 | 1.000000 | [[714, 0], [2, 4]] |
| chb01_04 | weighted | 0.995833 | 0.800000 | 0.666667 | 0.998599 | 0.727273 | 0.930556 | [[713, 1], [2, 4]] |
| chb01_15 | ordinary | 0.987500 | 0.000000 | 0.000000 | 0.998596 | 0.000000 | 0.440566 | [[711, 1], [8, 0]] |
| chb01_15 | weighted | 0.987500 | 0.000000 | 0.000000 | 0.998596 | 0.000000 | 0.271631 | [[711, 1], [8, 0]] |
| chb01_16 | ordinary | 0.995833 | 1.000000 | 0.700000 | 1.000000 | 0.823529 | 0.966923 | [[710, 0], [3, 7]] |
| chb01_16 | weighted | 0.994444 | 0.800000 | 0.800000 | 0.997183 | 0.800000 | 0.965152 | [[708, 2], [2, 8]] |

| Holdout | Model | Detected / missed | Latency (s) | FP windows | False episodes/h | Repeats | Alarm starts (s) |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| chb01_03 | ordinary | 1 / 0 | 9.0 | 0 | 0 | 0 | 3005 (detection) |
| chb01_03 | weighted | 1 / 0 | 9.0 | 1 | 1 | 0 | 3005 (detection), 3050 (false_alarm) |
| chb01_04 | ordinary | 1 / 0 | 8.0 | 0 | 0 | 1 | 1475 (detection), 1485 (repeat) |
| chb01_04 | weighted | 1 / 0 | 8.0 | 1 | 0 | 1 | 1475 (detection), 1490 (repeat) |
| chb01_15 | ordinary | 0 / 1 | undefined | 1 | 1 | 0 | 1785 (false_alarm) |
| chb01_15 | weighted | 0 / 1 | undefined | 1 | 1 | 0 | 1785 (false_alarm) |
| chb01_16 | ordinary | 1 / 0 | 5.0 | 0 | 0 | 1 | 1020 (detection), 1035 (repeat) |
| chb01_16 | weighted | 1 / 0 | 5.0 | 2 | 2 | 1 | 870 (false_alarm), 1010 (false_alarm), 1020 (detection), 1035 (repeat) |

| Metric | Ordinary pooled | Weighted pooled | Ordinary macro mean | Weighted macro mean |
| --- | ---: | ---: | ---: | ---: |
| accuracy | 0.994444 | 0.993750 | 0.994444 | 0.993750 |
| precision | 0.944444 | 0.791667 | 0.750000 | 0.618750 |
| recall | 0.531250 | 0.593750 | 0.529167 | 0.585417 |
| specificity | 0.999649 | 0.998244 | 0.999649 | 0.998243 |
| f1 | 0.680000 | 0.678571 | 0.620168 | 0.600568 |
| average_precision | 0.844954 | 0.773728 | 0.817063 | 0.760932 |

Pooled confusion matrices: ordinary `[[2847,1],[15,17]]`; weighted `[[2843,5],[13,19]]`.

## Definitions and interpretation

Confusion matrices use `[[TN,FP],[FN,TP]]`. Actual window ends are decision times;
consecutive positives form one episode, and a negative decision re-arms it.
Only a new start within exact `[onset,offset)` detects the event; later starts
within it are repeats. Pre-onset active alarms cannot detect a seizure. False
alarms/hour uses total evaluated EEG hours including ictal time. Missed seizures
have undefined latency. See the unchanged [evaluation contract](evaluation.md).

Weighted `chb01_04` has one FP window but zero false-alarm episodes: the decision
for `[1495,1500)` is after offset 1494 yet continues the valid repeat alarm that
began at 1490. Weighted `03` has a new postictal false start at 3050, after offset
3036. Both models' `15` alarm at 1785 is after offset 1772 and cannot detect that
event. Weighted `16` false starts at 870 and 1010 precede onset 1015. False-positive
windows and false-alarm episodes therefore need separate counts.

Pooled metrics concatenate out-of-fold predictions once per recording and retain
separate recording alarm state. Events are counted from annotations, not alarms.
Macro window rates weight folds equally and exclude undefined values, with their
number reported explicitly; JSON also reports population SD and min/max. All six
actual macro window rates are defined in all four folds. Precision in fold 15 is
zero, since its false-positive prediction makes the denominator nonzero.

Both models detect 3/4 events (macro sensitivity 0.75), miss one, and have delays
`[9,8,5]` seconds: mean 7.333 s, median 8 s. These delay summaries exclude the
missed event and must be read beside 3/4. Both have two repeats. Ordinary has one
false episode / four hours = 0.25/h; weighted has four / four hours = 1.0/h.
Total alarms are six and nine. One-hour folds make macro and pooled false-alarm
rates equal. Pooled AP combines scores from differently fitted models without
established cross-fold calibration; inspect fold and macro AP alongside it.
AP is the existing step-weighted PR summary, not trapezoidal area. Scores are
not calibrated clinical probabilities.

Weighting improves recall in `03` (0.75 to 0.875) and `16` (0.70 to 0.80), while
recall is unchanged in `04` (2/3) and `15` (zero). Neither model uniquely detects
any event, and no latency improves. Ordinary recall ranges 0–0.75; weighted
0–0.875. Macro recall population SD is 0.30695 versus 0.34613, descriptive of
these folds rather than a population uncertainty estimate. Weighting's two extra
ictal detections come with four extra FP windows and three extra false episodes.
Macro F1 declines from 0.620168 to 0.600568. AP improves in `03` and declines in
all other folds. Its benefit is mixed and does not extend to event detection here.

The shared failure is `chb01_15`: all eight ictal windows are missed. Maximum
ictal scores are 0.395 ordinary and 0.185 weighted, means 0.204375 and 0.06125.
Both instead generate a postictal false alarm. These are model-output observations;
no feature-level cause has been demonstrated. Different seizure durations,
boundary positions on the five-second grid and EEG feature patterns could
plausibly contribute. Exact durations are 40, 27, 40 and 51 seconds for `03`, `04`,
`15`, `16`. Duration alone cannot explain success in `03` and failure in `15`,
both 40 seconds. Weighted split criteria and leaf distributions can change
nonuniformly across events; weighting need not raise every ictal score.

Only 22–26 ictal windows and three seizures train each fold. EEG windows within an
event are correlated, training sets overlap, and all events belong to one patient.
Four events do not support clinical or population generalization. Training can
include recordings acquired after a fold's holdout: this is not chronological
validation. False-alarm estimates concern background EEG in seizure-containing
recordings; normal-only training files have no out-of-fold estimates here.

Continue studying ordinary RF as the reference before adding LightGBM. Weighted
RF has no event/latency advantage and generates more false alarms here. A useful
next separately approved step is a frozen-protocol supplementary false-alarm
assessment on the three new normal files, declaring models and rules before
reading results and keeping the original split unchanged. No such evaluation,
LightGBM, threshold search, filtering or dashboard work was started.

## Files, verification and reproduction

Added `src/rf_cross_validation.py`, seven tests in `tests/test_rf_cross_validation.py`,
this report and a README link. Earlier Phase 2C uncommitted changes were preserved.
New `results/rf_cv_v1/` outputs include configuration, fold definitions, eight
models, eight complete prediction tables, per-fold metrics/score distributions,
pooled/macro summaries, verification and artifact hashes. Initial-run `tests.log`
and `preservation.json` record tests and before/after hashes of protected files.

The suite ran 79 tests: **78 passed, one skipped**, no failures. All seven new
tests passed, covering fold isolation, retained normal recordings, excluded
validation/test/new IDs, feature-only fit matrices, aligned timelines, unique
event aggregation, undefined metrics, training-only reads, missing windows,
eight distinct model instances, deterministic repeated fits, reloads and output
protection. The existing expanded-data integration test is skipped because it
reads original validation/test rows. Other prior model tests use synthetic
fixtures; existing preprocessing tests read only the original three training EDF
headers. No protected raw recordings or new supplementary EDFs are read.

From the repository root, using the existing Apple Silicon environment and a
fresh output directory:

```sh
venv/bin/python -m src.rf_cross_validation \
  --data data/processed/chb01_detection_features_11recordings_split.parquet \
  --manifest config/chb01_recordings.csv \
  --baseline-config results/baseline_v1/configuration.json \
  --output results/rf_cv_reproduction

venv/bin/python -m unittest discover -s tests -p test_rf_cross_validation.py -v
```

Existing output directories are refused. The baseline configuration freezes
settings; previous model files and scores are not loaded. No dependencies were
installed or changes committed/pushed. Reproduce the protected full suite with:

```sh
MPLCONFIGDIR=/tmp/nyas-matplotlib venv/bin/python - <<'PY'
import unittest
suite = unittest.defaultTestLoader.discover('tests')
def embargo(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            embargo(item)
        elif item.id() == 'test_split_dataset.ExpandedArtifactTests.test_all_recordings_labels_features_and_splits':
            method = getattr(type(item), item._testMethodName)
            setattr(type(item), item._testMethodName,
                    unittest.skip('Phase 2D: original validation/test rows remain protected')(method))
embargo(suite)
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
PY
```
