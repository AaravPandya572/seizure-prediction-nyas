# Phase 2C: controlled class-weight experiment

The weighted forest detected the single validation seizure five seconds earlier,
but detected fewer ictal windows and had lower average precision. No false
positives occurred for either forest. These observations do not establish an
overall improvement from class weighting.

## Controlled comparison

`src/weighted_rf.py` reuses Phase 2B's loading, explicit 126-feature whitelist,
training, validation prediction and evaluator functions. The saved baseline is
loaded, never refitted. All 1,440 validation decisions and scores reproduce its
stored Parquet exactly; all evaluator outputs match the previous report.

The runner clones that estimator, changing only `class_weight=None` to
`class_weight="balanced"`. Both have 200 trees, seed 42, four workers, bootstrap
sampling, Gini splits, sqrt feature subsampling and all other saved defaults.
Full parameters and library versions are in the new `configuration.json`.
The standard `predict` boundary is preserved: score >0.5 selects class 1, and
an exact tie selects class 0. No tuning, sampling, feature or normalization
changes were made. No parameters changed after observing validation outcomes.

Training is the same 4,320 windows (4,288 nonictal; 32 ictal) from `chb01_01`,
`03`, `04`, `14`, `15`, `16`, with four distinct annotated seizures. Validation
is 1,440 windows (1,422 nonictal; 18 ictal) from `chb01_17` and `18`: two hours,
one seizure. Only `FEATURE_COLUMNS`, in canonical order, enter the models;
labels, timing, overlap, annotations, identifiers, assignments and quality flags
are excluded. Balanced weights use training labels only, `N/(2*N_class)`:
class 0 = 0.5037313433, class 1 = 67.5, a ratio of 134.

Data, manifest, feature-order, methodology source and library versions match
Phase 2B. The new runner fingerprints ordered training features, labels and window
identities and checks they remain unchanged by fitting. Phase 2B did not save this
matrix fingerprint; input identity with that run is established through its saved
dataset, manifest, feature-order and source hashes. Synthetic tests directly
capture and compare both fit matrices and labels.

Only train/validation rows are loaded through a Parquet filter. The experiment
opens no EDFs and creates no test scores or test evaluations. Whole-file hashing
of the existing combined dataset is an integrity check without decoding test rows.
All previous results, preprocessing, assignments and evaluation definitions are
preserved. No packages were installed and nothing was committed or pushed.

## Combined validation results

| Metric | Ordinary RF | Balanced RF |
| --- | ---: | ---: |
| Accuracy | 99.5139% | 99.3750% |
| Precision | 100% | 100% |
| Recall | 61.1111% | 50.0000% |
| Specificity | 100% | 100% |
| F1 | 0.758621 | 0.666667 |
| Average precision | 0.982948 | 0.940438 |
| True positive / false negative windows | 11 / 7 | 9 / 9 |
| False positive windows | 0 | 0 |
| Events detected / missed | 1 / 0 | 1 / 0 |
| Event sensitivity | 1/1 | 1/1 |
| False alarm episodes / hour | 0 | 0 |
| First detection latency | 10 s | 5 s |
| Repeated alarm episodes | 2 | 1 |

Confusion matrices, rows=true, columns=predicted, class order `[0,1]`:

```text
Ordinary: [[1422, 0], [7, 11]]
Weighted: [[1422, 0], [9,  9]]
```

AP is the unchanged evaluator's step-weighted PR summary, not trapezoidal area.
Pooled AP uses all validation scores and does not average recording APs.

## Per-recording results

| Recording / model | Accuracy | Precision | Recall | Specificity | F1 | AP | Confusion matrix |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| chb01_17 / ordinary | 100% | undefined | undefined | 100% | undefined | undefined | [[720,0],[0,0]] |
| chb01_17 / weighted | 100% | undefined | undefined | 100% | undefined | undefined | [[720,0],[0,0]] |
| chb01_18 / ordinary | 99.0278% | 100% | 61.1111% | 100% | 0.758621 | 1.000000 | [[702,0],[7,11]] |
| chb01_18 / weighted | 98.7500% | 100% | 50.0000% | 100% | 0.666667 | 0.961645 | [[702,0],[9,9]] |

`chb01_17` has no annotated seizure. Both forests have zero alarms, zero detected
or missed events and zero false alarms/hour over its one hour. Event sensitivity,
latency, recall and AP are undefined; undefined metrics are JSON `null`.
`chb01_18` has one seizure at `[1720,1810)` seconds. Both detect it, miss zero
events and produce zero false alarms/hour over its one hour.

| Model / episode | Start decision (s) | End decision (s) | Positive windows | Status |
| --- | ---: | ---: | ---: | --- |
| Ordinary / 1 | 1730 | 1735 | 1 | Detection, latency 10 s |
| Ordinary / 2 | 1740 | 1785 | 9 | Repeat |
| Ordinary / 3 | 1795 | 1800 | 1 | Repeat |
| Weighted / 1 | 1725 | 1735 | 2 | Detection, latency 5 s |
| Weighted / 2 | 1740 | 1775 | 7 | Repeat |

Actual window ends are decision times. Consecutive positive decisions continue
an episode; one negative decision closes it and re-arms. A new alarm start must
fall within `[onset,offset)` to detect the seizure. Further starts in that interval
are repeats. Episode end is the first negative decision. False alarms/hour counts
false episodes per total EEG hours including ictal time. False-positive windows
and false-alarm episodes are distinct quantities; both happen to be zero here.
See the unchanged [evaluation contract](evaluation.md).

## Prediction changes

Exactly four decisions changed. One original miss became correct, three original
true positives became misses, and no nonictal decision changed. All seven
original misses and the three regressions are included in `review_windows.parquet`.

| Window (s) | Ordinary score | Weighted score | Outcome |
| --- | ---: | ---: | --- |
| [1720,1725) | 0.215 | 0.515 | Original miss corrected; alarm 5 s earlier |
| [1730,1735) | 0.235 | 0.345 | Still missed |
| [1770,1775) | 0.530 | 0.310 | New miss |
| [1775,1780) | 0.600 | 0.490 | New miss |
| [1780,1785) | 0.445 | 0.110 | Still missed |
| [1785,1790) | 0.370 | 0.080 | Still missed |
| [1790,1795) | 0.520 | 0.145 | New miss |
| [1795,1800) | 0.350 | 0.070 | Still missed |
| [1800,1805) | 0.360 | 0.060 | Still missed |
| [1805,1810) | 0.285 | 0.010 | Still missed |

| Score group | Ordinary mean / median / range | Weighted mean / median / range |
| --- | --- | --- |
| chb01_17 nonictal | 0.005708 / 0 / 0–0.340 | 0.000549 / 0 / 0–0.150 |
| chb01_18 nonictal | 0.001168 / 0 / 0–0.160 | 0.000798 / 0 / 0–0.085 |
| chb01_18 ictal | 0.583333 / 0.565 / 0.215–0.975 | 0.523889 / 0.5025 / 0.010–0.990 |

Weighting raises some early ictal scores and lowers several later ones; it does
not uniformly raise seizure scores. Its ranking worsens on this validation set,
as shown by declining AP. Scores are uncalibrated and are not clinical probabilities.

![Validation comparison](../results/rf_weighted_v1/chb01_18_comparison.png)

Markers show window-end decisions. Connecting lines are visual guides, not
additional decisions, interpolation used by the detector, or temporal filtering.

## Scientific interpretation

Earlier detection and fewer repeats coexist with lower recall and F1. Event
sensitivity remains 1/1 and cannot increase on this already-detected event.
Precision and specificity remain unchanged. There is no observed sensitivity-
specificity trade-off: window sensitivity declines while specificity stays fixed.

Class weighting changes weighted split criteria and leaf class proportions and
can change tree structure, rather than just shifting a threshold. Different
structures might emphasize patterns that help at onset but hurt later in this
seizure. That is a plausible explanation, not a demonstrated cause of these
particular errors. The run establishes the observed effect under this fixed seed,
not a general benefit or harm from cost-sensitive learning.

Only 32 positive training windows arise from four distinct seizures in one
patient. Adjacent EEG windows are highly correlated; separate seizures within
this patient do not provide independent patient samples. One validation seizure
makes event sensitivity and latency unstable measures of future performance.
Zero false alarms over two hours does not imply zero future false-alarm risk.
Neither clinical reliability nor population-level generalization is supported.
Test recordings were not used to resolve uncertainty.

After review, a useful next experiment is predeclared leave-one-seizure-recording-
out comparisons using only the six training recordings, with both configurations,
features and seed fixed. Assess event-to-event stability, keeping validation and
final test data out of that experiment. This is a recommendation only.

## Files, tests and reproduction

Added `src/weighted_rf.py`, six synthetic tests in `tests/test_weighted_rf.py`,
this report and a README link. New artifacts in `results/rf_weighted_v1/` include
`weighted_rf.joblib`, complete `configuration.json`, aligned weighted validation
predictions, ordinary/weighted evaluator reports (pooled and per recording with
alarm ledgers), the all-window comparison, review windows, plot, verification
and artifact hashes. The initial run additionally stores `tests.log` and a
before/after audit of protected files in `preservation.json`.

The suite ran 72 tests: **71 passed, one skipped**, no failures. The one skipped
existing `ExpandedArtifactTests.test_all_recordings_labels_features_and_splits`
reads held-out rows and is excluded under this phase's explicit embargo. All
six new tests passed, covering identical fit matrices, sole parameter difference,
feature/label/order fingerprints, alignment and known changes, test-row exclusion,
baseline preservation and weighted-model reload. Actual baseline predictions and
evaluation reproduced exactly; weighted predictions and scores reproduce on reload.

From the repository root in the existing ARM64 environment, using a new directory:

```sh
MPLCONFIGDIR=/tmp/nyas-matplotlib venv/bin/python -m src.weighted_rf \
  --data data/processed/chb01_detection_features_11recordings_split.parquet \
  --manifest config/chb01_recordings.csv \
  --baseline results/baseline_v1 \
  --output results/rf_weighted_reproduction
```

Existing output directories are refused. `MPLCONFIGDIR` places the plot font
cache in a writable location. Reproduce the suite with its temporary exclusion:

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
                    unittest.skip('Phase 2C: held-out test rows must not be loaded')(method))
embargo(suite)
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
PY
```

No threshold search, temporal filtering, LightGBM, EEGNet, dashboard or final test
evaluation was performed. This phase stops at the controlled comparison.
