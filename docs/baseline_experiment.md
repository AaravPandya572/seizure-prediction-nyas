# Phase 2B: initial DummyClassifier and Random Forest baselines

The fixed Random Forest detected 11/18 validation seizure windows and the one
validation seizure, with a 10-second onset-to-alarm latency. There were no false
positive windows. This is a small, patient-specific development result, not a
final test result. No threshold or hyperparameter search was performed.

## Scope, inputs and leakage safeguards

`src/baseline_rf.py` reads the verified split dataset with a Parquet filter for
`train` and `validation` only. It checks dataset and manifest hashes against the
Phase 2A report, verifies each recording's assigned split, checks labels against
exact seizure annotations, and uses the existing evaluator to check complete,
ordered timelines. Whole-file hashing is an integrity check, not test inference.

Only `src.preprocess.FEATURE_COLUMNS` enters either classifier: the canonical
18-channel order, each channel's variance, line length, and five relative powers,
for exactly 126 features. The complete ordered list is also saved in
`results/baseline_v1/configuration.json`. All 19 other dataset columns are excluded,
including labels, overlap, annotations, identifiers, split, timestamps and quality
flags. Feature selection is explicit even if new numeric metadata is added later.
No feature scaling, feature selection search, resampling, or row exclusion occurs.

| Partition used | Recordings | Class 0 | Class 1 | Windows | Distinct annotated seizures |
| --- | --- | ---: | ---: | ---: | ---: |
| Training | 01, 03, 04, 14, 15, 16 | 4,288 | 32 | 4,320 | 4 |
| Validation | 17, 18 | 1,422 | 18 | 1,440 | 1 |

All filenames have the `chb01_` prefix. Training prevalence is 32/4320 = 0.7407%;
validation prevalence is 18/1440 = 1.25%. Neither classifier receives test rows.
The runner rejects non-training rows at fitting and non-validation rows at
prediction. Existing full-suite dataset integrity tests still inspect all splits;
they do not perform model inference or evaluation on the reserved test set.

## Fixed model configurations

- DummyClassifier: `strategy="most_frequent"`, `constant=None`, `random_state=None`.
  This strategy is deterministic and always predicts the training majority, class 0.
- RandomForestClassifier: `n_estimators=200`, `random_state=42`, `class_weight=None`,
  `n_jobs=4`. Unchanged scikit-learn defaults: `criterion="gini"`,
  `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`,
  `min_weight_fraction_leaf=0.0`, `max_features="sqrt"`, `max_leaf_nodes=None`,
  `min_impurity_decrease=0.0`, `bootstrap=True`, `oob_score=False`, `verbose=0`,
  `warm_start=False`, `ccp_alpha=0.0`, `max_samples=None`, `monotonic_cst=None`.

The runner saves every parameter from `get_params()`, source/data hashes, feature
order, counts, exact validation annotations and library versions. This run used
Python 3.14.5 on ARM64, scikit-learn 1.9.1, NumPy 2.5.4, pandas 3.0.6, SciPy 1.18.1,
joblib 1.6.0, PyArrow 26.0.0, matplotlib 3.11.2 and MNE 1.13.2.

Predictions use sklearn's standard `predict`; scores are `predict_proba` for
class 1. For this binary forest, class 1 requires a score **greater than 0.5**;
an exact 0.5 tie selects class 0. There is no threshold optimization or smoothing.

## Actual validation results

| Metric | Dummy | Random Forest |
| --- | ---: | ---: |
| Accuracy | 98.7500% | 99.5139% |
| Precision | Undefined (no positive predictions) | 100% |
| Recall | 0% | 61.1111% |
| Specificity | 100% | 100% |
| F1 | 0 | 0.758621 |
| Average precision | 0.012500 | 0.982948 |
| Positive predictions | 0 | 11 |
| Seizures detected / missed | 0 / 1 | 1 / 0 |
| False alarm episodes / evaluated hours | 0 / 2 | 0 / 2 |
| False alarms/hour | 0 | 0 |
| Detection latency | Undefined (missed) | 10 seconds |
| Repeat alarm episodes within the seizure | 0 | 2 |

Confusion matrices use rows=true and columns=predicted, both in class order `[0,1]`:

```text
Dummy:          [[1422, 0],     Random Forest: [[1422,  0],
                 [  18, 0]]                    [   7, 11]]
```

Dummy accuracy is `(TN + TP)/N = (1422 + 0)/1440 = 98.75%`, despite recall
`0/18 = 0` and event detection `0/1 = 0`. Predicting the majority class is enough
to achieve high accuracy here. Its constant zero scores give AP equal to
validation prevalence, 0.0125. AP is the evaluator's step-weighted precision-recall
summary, not trapezoidal PR area. Undefined precision/latency remain JSON `null`.

The forest improves accuracy by only 0.7639 percentage points, while detecting
11 seizure windows that the dummy misses. Accuracy alone conceals this difference.

## Failure analysis: observations

The validation seizure in `chb01_18` is `[1720,1810)` seconds. All 11 positive
windows occur within it. Their window spans form three runs:
`[1725,1730)`, `[1735,1780)` (nine windows), and `[1790,1795)`.

Using actual window ends, alarms begin at 1730, 1740 and 1795 seconds. The first
detects the seizure with latency `1730 - 1720 = 10` seconds. The later two are
repeat alarms, not additional detected seizures or false alarms. The first
negative decisions close these episodes at 1735, 1785 and 1800 seconds.

Seven seizure windows are missed: `[1720,1725)`, `[1730,1735)`, `[1780,1785)`,
`[1785,1790)`, `[1795,1800)`, `[1800,1805)`, `[1805,1810)`. Thus the model does
not continuously mark the entire seizure, although it detects that event once.
Window recall (11/18) and event sensitivity (1/1) answer different questions.
All 1,422 nonictal validation windows are negative, including all of `chb01_17`.

| Validation score group | Count | Median | Mean | 95th percentile | Min–max |
| --- | ---: | ---: | ---: | ---: | --- |
| chb01_17, nonictal | 720 | 0 | 0.005708 | 0.030 | 0–0.340 |
| chb01_18, nonictal | 702 | 0 | 0.001168 | 0.005 | 0–0.160 |
| chb01_18, ictal | 18 | 0.565 | 0.583333 | 0.94525 | 0.215–0.975 |

The seizure windows generally rank above nonictal windows, explaining high AP,
but seven scores remain below the fixed decision boundary. Scores fluctuate
within the seizure; they are not calibrated estimates of clinical risk.
The following plot shows discrete scores at window-end decision times; connecting
lines are visual guides, not additional decisions or temporal filtering.

![Validation scores and exact seizure interval](../results/baseline_v1/chb01_18_scores.png)

The observed misses could reflect sparse training examples, differences in seizure
appearance, or the limits of these features and model. These are **hypotheses**;
this experiment does not identify their cause or demonstrate that class imbalance
alone caused them. In particular, the forest did not collapse to an all-negative
classifier, so the result should not be described that way.

## Evaluation definitions and limitations

The existing [evaluation contract](evaluation.md) is unchanged. Consecutive
positive decisions form one episode; one negative decision re-arms it. Only a
new alarm start inside `[onset,offset)` detects an event. An alarm exactly at
offset or already active before onset cannot detect it. Recordings have separate
alarm state. False alarms/hour divides by total evaluated EEG hours (including
ictal time). Latency includes window accumulation but excludes inference and
delivery runtime. Missed seizures never receive a zero latency.

Only four distinct training seizures and one validation seizure are available,
all from one patient. Adjacent windows are correlated, and distinct seizures in
one patient are not independent patient samples. One detected validation event
cannot establish reliable event sensitivity; zero false alarms over two hours
cannot establish a zero false alarm rate in future recordings. No patient-level
or clinical generalization claim follows. The held-out test recordings remain
reserved; their model performance is unknown. Further repeated comparisons on
this same small validation set also risk overfitting development decisions.

## Artifacts and reproduction

New/modified source files:

- `src/baseline_rf.py`: model runner, strict feature/split checks, model saving,
  aligned validation output, existing evaluator handoff, score summaries and plot.
- `tests/test_baseline_rf.py`: 12 synthetic tests for features/metadata, splits,
  positional alignment, serialization, standard probability ties, evaluator
  compatibility, provenance and output protection.
- `docs/baseline_experiment.md` and the README link: methodology, results and commands.

`results/baseline_v1/` contains both `.joblib` models, all parameters/versions in
`configuration.json`, `dummy_validation.parquet` and `random_forest_validation.parquet`
(1,440 aligned rows each), two `*_evaluation.json` reports including per-recording
alarm ledgers and score distributions, `random_forest_review_windows.parquet`
(all errors or positive predictions), the plot, and artifact SHA-256 hashes.
`verification.json` and `tests.log` record the final artifact checks and full-suite
results for this run. All saved predictions retain patient/recording/file IDs,
split, start/end/decision times, labels, prediction and score.

From the repository root, using the existing virtual environment:

```sh
# Fits both baselines and evaluates validation; use a NEW directory on each run.
MPLCONFIGDIR=/tmp/nyas-matplotlib venv/bin/python -m src.baseline_rf \
  --data data/processed/chb01_detection_features_11recordings_split.parquet \
  --manifest config/chb01_recordings.csv \
  --output results/baseline_reproduction

MPLCONFIGDIR=/tmp/nyas-matplotlib venv/bin/python -m unittest discover -s tests -v
```

`MPLCONFIGDIR` just places matplotlib's font cache in a writable location. No
dependencies were installed. The runner refuses to overwrite an existing output
directory. Models are reloaded and their complete validation predictions/scores
compared to the originals during every run. Preserve the saved library versions
for reproduction across environments; sklearn model loading is version-sensitive.

To re-evaluate the saved predictions without fitting any model:

```sh
venv/bin/python - <<'PY'
import json
from pathlib import Path
import pandas as pd
from src.evaluate import evaluate_predictions
root = Path('results/baseline_v1')
metadata = json.loads((root / 'configuration.json').read_text())['validation_metadata']
for name in ('dummy', 'random_forest'):
    predictions = pd.read_parquet(root / f'{name}_validation.parquet')
    report = evaluate_predictions(predictions, metadata)
    saved = json.loads((root / f'{name}_evaluation.json').read_text())
    for key, value in report.items():
        assert value == saved[key]
    print(name, report['window_metrics'], report['event_metrics'])
PY
```

Verification: **66 tests passed** (54 existing + 12 new), with no skips. Both
models reload with identical predictions and scores. Earlier processed datasets,
preprocessing/evaluation/split source and the manifest have unchanged hashes.
The baseline runner opens no EDF files; existing preprocessing tests only read
the original three EDF headers. Raw data and earlier artifacts were not modified.

After review, the suggested next experiment is one predeclared comparison using
the same forest, split, seed and decision rule with `class_weight="balanced"`.
Compare seizure recall, latency, repeats and false alarms against this saved
baseline. It is a proposed experiment only; no weighted model, alternative
threshold, LightGBM, temporal filter, deep model or dashboard was implemented.
