# Phase 2G: fixed LightGBM versus Random Forest comparison

Both LightGBM models detect the single validation seizure at 1725 seconds,
5 seconds after onset, and correctly classify the same 10 of 18 ictal windows.
Each produces one false-alarm episode in two hours. Weighting rescues no ictal
windows; it removes one false-positive window and introduces another elsewhere.
Neither LightGBM model establishes an improvement over ordinary RF: ordinary RF
has higher recall, F1 and AP and no false alarm in this validation set. LightGBM
alarms five seconds earlier than ordinary RF, but matches weighted RF's latency.
There is only one evaluated seizure, so these observations cannot establish a
reliable model ranking or clinical performance.

## Environment and minimum installation

The existing environment is Python 3.14.5 on macOS ARM64, with NumPy 2.5.4,
SciPy 1.18.1, pandas 3.0.6, scikit-learn 1.9.1, joblib 1.6.0, PyArrow 26.0.0,
matplotlib 3.11.2 and MNE 1.13.2. These versions remain unchanged.
LightGBM was absent. The pinned `lightgbm-4.7.0-py3-none-macosx_12_0_arm64.whl`
was installed into the existing venv with no dependency upgrades. Import then
confirmed a missing `libomp.dylib`; only Homebrew's OpenMP runtime, libomp
23.1.4, was added, with automatic Homebrew update/cleanup disabled. No compiler,
GPU runtime or other Python package was installed. Synthetic and actual fits
passed on this environment.

```sh
venv/bin/python -m pip install --only-binary=:all: --no-deps lightgbm==4.7.0
HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_INSTALL_CLEANUP=1 brew install libomp
```

These commands describe the minimal additions for this existing environment;
they are already installed for this run. `requirements.txt` gains only the
LightGBM pin. Runtime binary paths/hashes, wheel tag and commands are saved in
`environment.json` with the results. Restoring recorded library/runtime versions
matters for reproduction; deterministic settings are not a cross-version or
cross-platform guarantee.

Sources: [official package and ARM64 wheel](https://pypi.org/project/lightgbm/),
[installation guide](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html).

## Frozen configurations and data isolation

Both LightGBM estimators use `objective="binary"`, `n_estimators=150`,
`learning_rate=0.05`, `max_depth=3`, `num_leaves=7`, `min_child_samples=10`,
`random_state=42`, `n_jobs=4`. The requested leaves/depth are compatible
(7 <= 2^3); no parameter adjustment was needed. Both actually fitted 150 trees.
Both also set CPU `deterministic=True`, `force_col_wise=True` and `verbosity=-1`
before seeing outcomes. This fixes histogram-building behavior and reduces
logging, identically for both models. The [parameter documentation](https://lightgbm.readthedocs.io/en/stable/Parameters.html)
recommends a fixed histogram orientation with deterministic mode.

Ordinary `scale_pos_weight=1.0`. Weighted `scale_pos_weight=N0/N1=4288/32=134.0`,
computed from training labels. This is the **only parameter difference**. No
`is_unbalance`, `class_weight` weighting or sample weights are combined with it.
The shared constructor value `class_weight=None` applies no additional weighting.
Other fixed defaults are `boosting_type="gbdt"`, `subsample_for_bin=200000`,
`min_split_gain=0.0`, `min_child_weight=0.001`, `subsample=1.0`, `subsample_freq=0`,
`colsample_bytree=1.0`, `reg_alpha=0.0`, `reg_lambda=0.0`, `importance_type="split"`.
Full constructor and fitted booster parameters are saved. No tuning, early
stopping, evaluation set, callback, normalization change or resampling is used.

| Split used | Recordings | Class 0 | Class 1 | Windows | Annotated seizures |
| --- | --- | ---: | ---: | ---: | ---: |
| Train | chb01_01,03,04,14,15,16 | 4288 | 32 | 4320 | 4 |
| Validation | chb01_17,18 | 1422 | 18 | 1440 | 1 |

The existing Parquet loader decodes only train/validation rows. Dataset and
manifest hashes match the RF experiments, and the loader checks recording
assignments, labels and complete timelines. Training selects exactly the ordered
126 `FEATURE_COLUMNS` from preprocessing. All 19 metadata/target columns are
excluded, even numeric ones. An ordered training-input fingerprint is checked
before/after fitting. No test or supplementary inference occurs, no EDF is
opened by the runner, and no new recordings are processed. Existing full-suite
preprocessing tests read only the three original training EDF headers.

RF references are the saved original and weighted validation models; they are
not refitted. Their artifacts pass hashes, and their predictions/scores and
original evaluator outputs reproduce. Preprocessing, manifest, annotations,
feature definitions and evaluator source are unchanged. `chb01_15` belongs to
the training data here; no predictions or new failure claims are made about it.

The default classifier `predict` rule is retained: model score >0.5 gives class 1,
with exact ties selecting class 0. Scores are **model scores**, not calibrated
clinical probabilities. Native model text and joblib copies are both saved.

## Combined and per-recording validation results

| Combined metric | Ordinary RF | Weighted RF | Ordinary LightGBM | Weighted LightGBM |
| --- | ---: | ---: | ---: | ---: |
| accuracy | 0.995139 | 0.993750 | 0.993750 | 0.993750 |
| precision | 1.000000 | 1.000000 | 0.909091 | 0.909091 |
| recall | 0.611111 | 0.500000 | 0.555556 | 0.555556 |
| specificity | 1.000000 | 1.000000 | 0.999297 | 0.999297 |
| f1 | 0.758621 | 0.666667 | 0.689655 | 0.689655 |
| average_precision | 0.982948 | 0.940438 | 0.847912 | 0.690848 |
| confusion_matrix | [[1422, 0], [7, 11]] | [[1422, 0], [9, 9]] | [[1421, 1], [8, 10]] | [[1421, 1], [8, 10]] |
| fp | 0 | 0 | 1 | 1 |
| Detected / missed | 1 / 0 | 1 / 0 | 1 / 0 | 1 / 0 |
| False episodes/hour | 0.0 | 0.0 | 0.5 | 0.5 |
| Latency (s) | 10.0 | 5.0 | 5.0 | 5.0 |
| Repeat episodes | 2 | 1 | 0 | 0 |

| Recording | Model | Accuracy | Precision | Recall | Specificity | F1 | AP | Confusion matrix |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| chb01_17 | Ordinary RF | 1.000000 | undefined | undefined | 1.000000 | undefined | undefined | [[720, 0], [0, 0]] |
| chb01_17 | Weighted RF | 1.000000 | undefined | undefined | 1.000000 | undefined | undefined | [[720, 0], [0, 0]] |
| chb01_17 | Ordinary LightGBM | 0.998611 | 0.000000 | undefined | 0.998611 | 0.000000 | undefined | [[719, 1], [0, 0]] |
| chb01_17 | Weighted LightGBM | 1.000000 | undefined | undefined | 1.000000 | undefined | undefined | [[720, 0], [0, 0]] |
| chb01_18 | Ordinary RF | 0.990278 | 1.000000 | 0.611111 | 1.000000 | 0.758621 | 1.000000 | [[702, 0], [7, 11]] |
| chb01_18 | Weighted RF | 0.987500 | 1.000000 | 0.500000 | 1.000000 | 0.666667 | 0.961645 | [[702, 0], [9, 9]] |
| chb01_18 | Ordinary LightGBM | 0.988889 | 1.000000 | 0.555556 | 1.000000 | 0.714286 | 0.963942 | [[702, 0], [8, 10]] |
| chb01_18 | Weighted LightGBM | 0.987500 | 0.909091 | 0.555556 | 0.998575 | 0.689655 | 0.762839 | [[701, 1], [8, 10]] |

Confusion matrices are `[[TN,FP],[FN,TP]]`, true rows and predicted columns in
class order `[0,1]`. AP uses the unchanged step-weighted PR definition, not
trapezoidal area. Combined AP is computed from all validation scores, not from
averaging recording APs. All recordings are evaluated in full.

For `chb01_17`, there are no seizures: sensitivity and latency are undefined for
all models. Ordinary LightGBM has one false-positive window and one false episode
in that hour; the other models have none. Precision and F1 for ordinary LightGBM
are measured zero, while they are undefined for the others because no positive
labels or predictions occur. Recall and AP are undefined for all four there.
For `chb01_18`, every model detects the one seizure and misses zero events;
weighted LightGBM alone has a false episode in this recording. Per-recording
JSONs retain undefined values as `null` with explanations.

## Alarm times and changed predictions

| Recording | Model | Alarm start (s) | Episode end (s) | Positive windows | Status |
| --- | --- | ---: | ---: | ---: | --- |
| chb01_18 | Ordinary RF | 1730 | 1735 | 1 | detection |
| chb01_18 | Ordinary RF | 1740 | 1785 | 9 | repeat |
| chb01_18 | Ordinary RF | 1795 | 1800 | 1 | repeat |
| chb01_18 | Weighted RF | 1725 | 1735 | 2 | detection |
| chb01_18 | Weighted RF | 1740 | 1775 | 7 | repeat |
| chb01_17 | Ordinary LightGBM | 1825 | 1830 | 1 | false_alarm |
| chb01_18 | Ordinary LightGBM | 1725 | 1775 | 10 | detection |
| chb01_18 | Weighted LightGBM | 1725 | 1775 | 10 | detection |
| chb01_18 | Weighted LightGBM | 2535 | 2540 | 1 | false_alarm |

Each episode end is the first negative **decision** time, not the end of its
last positive EEG window. Decision times equal actual window ends. A new start
must occur inside `[1720,1810)` to detect the validation seizure. Further starts
inside are repeats; outside starts are false alarms. Recordings have separate
state, and one negative decision re-arms. False alarms/hour is total false
episodes divided by total evaluated hours, including ictal time. These rules
are unchanged; see [the evaluator contract](evaluation.md).

Both LightGBM models correctly flag ten consecutive windows covering
`[1720,1770)`, with positive decisions at 1725,1730,...,1770. They miss all eight
later ictal windows covering `[1770,1810)`. The alarm begins during the seizure
and closes at 1775. Neither has an early pre-onset alert or a repeated alarm in
the seizure. Weighted and ordinary LightGBM have identical ictal binary decisions.

Compared with ordinary RF, both LightGBMs rescue `[1720,1725)` and `[1730,1735)`
but lose three RF true positives: `[1770,1775)`, `[1775,1780)` and `[1790,1795)`.
Net ictal true positives fall from 11 to 10. Compared with weighted RF, each
LightGBM adds the `[1730,1735)` true positive (9 to 10), with the same five-second
latency. Window improvements therefore depend on the RF comparator; no additional
independent event is detected.

Ordinary LightGBM's new false positive is `chb01_17` `[1820,1825)`, score
0.957654, versus ordinary RF 0.340 and weighted RF 0.150. Weighted LightGBM reduces
that score to 0.006638 but creates a false positive at `chb01_18` `[2530,2535)`,
score 0.653765. Ordinary LightGBM's score there is 0.014657 (RF 0.160/0.085).
Thus weighting changes exactly **two** hard decisions, both nonictal: one removed
FP and one new FP. Total FP windows and false episodes stay at one, in different
recordings. This is not improvement in false-alarm burden.

Ordinary LightGBM ictal score median/mean are 0.672388/0.517017; weighted values
are 0.998924/0.559797. Despite a higher median score, weighting reduces pooled
AP from 0.847912 to 0.690848 and per-seizure-recording AP from 0.963942 to
0.762839. It changes the ranking of true and false windows beyond the fixed
threshold, without rescuing any ictal decisions. Near-one values and the isolated
high-scoring false positives should not be read as reliable certainty. Detailed
ranges, quartiles and 95th percentiles are in each report's `score_distribution`.
No calibration or threshold experiment was performed.

![Four-model validation score comparison](../results/lightgbm_v1/chb01_18_four_model_scores.png)

The upper panel shows the complete seizure-containing recording, including the
weighted LightGBM false alert at 2535; the lower panel zooms around onset/offset.
Markers are actual window-end scores. Connecting lines are visual guides and do
not create intervening decisions or temporal filtering. The separate normal
recording's false alert is documented in the ledger and comparison table.

## Interpretation, limitations and next phase

These fixed LightGBM settings do not improve on ordinary RF's combined recall,
precision, specificity, F1 or AP. They provide one-step-earlier detection than
ordinary RF and fewer within-seizure repeats, but weighted RF already achieves
the same initial latency without the observed false alarms. Weighting LightGBM
has no ictal decision, latency, event sensitivity or total false-alarm benefit
and worsens AP. Neither model missed the validation event; event sensitivity
remains 1/1 for all four, so window-level changes cannot demonstrate better event
detection. Algorithmic differences, sparse positives and event variability are
plausible contributors, not established causes of particular mistakes.

Only four training seizures produce 32 correlated positive windows, and validation
has one seizure in one patient plus one seizure-free hour. A measured 1/1 event
sensitivity is weak evidence for future detection. Zero RF false alarms or one
LightGBM false alarm in two hours are unstable estimates, not reliable long-term
rates. The same validation set has been examined repeatedly through model
comparisons, increasing development-selection bias. Neither clinical reliability
nor population-level generalization follows. No test or supplementary data were
used to adjudicate uncertainty. No claim is made that training on `chb01_15`
solves that earlier held-out failure.

Retain ordinary RF as the current reference. Before selecting a final classifier,
a useful separately approved experiment is to apply these **unchanged** LightGBM
configurations to the same training-only recording-level folds already used for
RF, with endpoints declared beforehand. Do not tune on the difficult event or
select a winner from this one validation seizure alone. After model review, a
single predeclared causal persistence filter could be compared against the frozen
unfiltered baseline, reporting both false alarms and added detection latency.
The observed LightGBM false positives are isolated one-window episodes, which
makes persistence a testable hypothesis, not a demonstrated solution. Such a
filter can also suppress short true alerts or delay detection. Keep the final
test set sealed; supplementary normal evaluation remains a later distinct step.
No CV extension, filtering, threshold optimization, neural model, final test or
dashboard was implemented in this phase.

## Files, verification and reproduction

Added `src/lightgbm_experiment.py`, seven synthetic tests in
`tests/test_lightgbm_experiment.py`, this report and a README link. The only
requirements change is `lightgbm==4.7.0`. Existing uncommitted work is preserved.
`results/lightgbm_v1/` contains two models in joblib/native text formats, frozen
parameters and versions, 1,440 aligned validation predictions per model, four
model evaluator reports including per-recording ledgers, all-window comparison,
one figure, warning/verification records and artifact hashes. `environment.json`,
`tests.log` and `preservation.json` document the installed runtime, initial test
run and before/after checks. No Python warnings were captured in the real fits.

The full suite ran 94 tests: **93 passed, one skipped**, no failures. The existing
expanded-dataset integration check is skipped because it decodes held-out test
rows. Every other test ran, including the seven new tests. They verify exact
feature order, metadata exclusion, identical fit inputs, training-only weight
calculation, sole parameter difference, no validation fit/early stopping,
alignment, serialization, evaluator compatibility, test/supplementary exclusion,
reference preservation and output protection. Tiny synthetic fits in tests are
separate from the two real-data fits. Saved model reloads reproduce validation
predictions and scores; historical RF results reproduce without refitting.

From the repository root in the existing environment, use a fresh output directory:

```sh
MPLCONFIGDIR=/tmp/nyas-matplotlib venv/bin/python -m src.lightgbm_experiment \
  --data data/processed/chb01_detection_features_11recordings_split.parquet \
  --manifest config/chb01_recordings.csv \
  --rf-dir results/baseline_v1 \
  --weighted-rf-dir results/rf_weighted_v1 \
  --output results/lightgbm_reproduction

MPLCONFIGDIR=/tmp/nyas-matplotlib venv/bin/python -m unittest discover -s tests \
  -p test_lightgbm_experiment.py -v
```

Existing output directories are refused. `MPLCONFIGDIR` only chooses a writable
plot-font cache. To reproduce the full suite with the explicit held-out exclusion:

```sh
MPLCONFIGDIR=/tmp/nyas-matplotlib venv/bin/python - <<'PY'
import unittest
suite=unittest.defaultTestLoader.discover('tests')
def protect(suite):
    for item in suite:
        if isinstance(item,unittest.TestSuite):
            protect(item)
        elif item.id()=='test_split_dataset.ExpandedArtifactTests.test_all_recordings_labels_features_and_splits':
            method=getattr(type(item),item._testMethodName)
            setattr(type(item),item._testMethodName,
                    unittest.skip('Phase 2G: no held-out test feature rows')(method))
protect(suite)
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
PY
```

No original model, preprocessing implementation, label, manifest or prior result
was changed; no commit or push was performed. Work stops after these two fits
and their validation comparison.
