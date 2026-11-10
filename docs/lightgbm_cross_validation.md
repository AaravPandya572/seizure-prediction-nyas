# Phase 2H: training-only LightGBM cross-validation and stability

Ordinary LightGBM detects all four held-out seizures, including `chb01_15`;
the other three models detect three. Its pooled recall is 65.625%, F1 0.7500,
and false-alarm rate 0.5 episodes/hour. Ordinary RF has lower recall (53.125%)
and fewer false alarms (0.25/hour). Weighted LightGBM does not consistently
improve on ordinary LightGBM: it loses the `15` detection, has lower pooled
recall and AP, and doubles its false-alarm rate to 1/hour.

**The `chb01_15` detection is exploratory, not independent confirmation.** That
recording has been studied repeatedly in development. No parameter was altered
in response to these results, and the detection does not demonstrate clinical
or population generalization.

## Scope, fixed models and fold verification

`src/lightgbm_cross_validation.py` reuses the established training-only loader,
recording fold definitions, explicit feature selector, prediction handoff and
unchanged evaluator. The data and manifest hashes, all 126 ordered feature names,
methodology source hashes and library versions must match the original RF/CV
and Phase 2G experiments. Each training-input fingerprint and recording list is
checked against the saved RF fold. Complete five-second timelines are retained.
No validation, final-test or supplementary feature rows are decoded. No EDFs
are opened by this runner. Reading the Phase 2G configuration supplies settings
only; its validation models, prediction tables and reports are not loaded.
Whole-file hashes of existing processed data check integrity without decoding
reserved rows. Historical artifact hashes are preservation checks, not analysis
of protected scores.

Both LightGBMs use binary objective, 150 estimators, learning rate 0.05,
maximum depth 3, seven leaves, minimum child samples 10, seed 42 and four workers.
Phase 2G's `deterministic=True`, `force_col_wise=True`, `verbosity=-1` and every
other parameter are preserved. Ordinary `scale_pos_weight=1.0`; weighted uses
the current fold's training nonictal/ictal ratio. Neither `is_unbalance` nor
additional class/sample weights are applied. No early stopping, evaluation-set
feedback, tuning, resampling, engineered features, threshold search or temporal
filtering occurs. Standard `predict` means score >0.5, ties select class 0.
Full parameters per fold are in `definition.json`.

Each fold trains two fresh estimators: eight actual-data fits, all 150 trees.
Both classifiers receive the same feature matrix and labels in a fold. Only
`scale_pos_weight` differs. Canonical EEG features alone enter fitting; labels,
annotation overlaps, split, IDs, timestamps and quality metadata are excluded.
Saved RF predictions are aligned and re-evaluated without fitting RF models.
Their per-fold and pooled/macro results reproduce the historical results.

The environment remains Python 3.14.5 ARM64 with LightGBM 4.7.0 and the previously
installed libomp runtime. Existing package versions remain unchanged; nothing
was installed in this phase. Reproducibility metadata and source hashes are saved.

| Fold | Holdout | Training recordings | Train 0 / 1 | Evaluated 0 / 1 | Positive weight |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | chb01_03 | chb01_01, chb01_04, chb01_14, chb01_15, chb01_16 | 3576 / 24 | 712 / 8 | 149.000000000 |
| 2 | chb01_04 | chb01_01, chb01_03, chb01_14, chb01_15, chb01_16 | 3574 / 26 | 714 / 6 | 137.461538462 |
| 3 | chb01_15 | chb01_01, chb01_03, chb01_04, chb01_14, chb01_16 | 3576 / 24 | 712 / 8 | 149.000000000 |
| 4 | chb01_16 | chb01_01, chb01_03, chb01_04, chb01_14, chb01_15 | 3578 / 22 | 710 / 10 | 162.636363636 |

Each fit contains five whole recordings, three annotated seizures and 3,600
windows. Normal recordings `chb01_01` and `14` are in every fit. Each evaluation
contains the complete one-hour held-out recording and one annotated event.
Across folds there are 2,880 unique evaluated windows: 2,848 nonictal and 32
ictal, four observed EEG hours and four distinct events per model. Training sets
overlap. The full original training ratio 134 is **not** used for fold weights.

## Per-fold window and event results

| Holdout | Model | Accuracy | Precision | Recall | Specificity | F1 | AP | Confusion matrix |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| chb01_03 | Ordinary RF | 0.997222 | 1.000000 | 0.750000 | 1.000000 | 0.857143 | 0.860764 | [[712, 0], [2, 6]] |
| chb01_03 | Weighted RF | 0.997222 | 0.875000 | 0.875000 | 0.998596 | 0.875000 | 0.876389 | [[711, 1], [1, 7]] |
| chb01_03 | Ordinary LGBM | 0.997222 | 1.000000 | 0.750000 | 1.000000 | 0.857143 | 0.838975 | [[712, 0], [2, 6]] |
| chb01_03 | Weighted LGBM | 0.995833 | 0.777778 | 0.875000 | 0.997191 | 0.823529 | 0.766045 | [[710, 2], [1, 7]] |
| chb01_04 | Ordinary RF | 0.997222 | 1.000000 | 0.666667 | 1.000000 | 0.800000 | 1.000000 | [[714, 0], [2, 4]] |
| chb01_04 | Weighted RF | 0.995833 | 0.800000 | 0.666667 | 0.998599 | 0.727273 | 0.930556 | [[713, 1], [2, 4]] |
| chb01_04 | Ordinary LGBM | 0.997222 | 0.833333 | 0.833333 | 0.998599 | 0.833333 | 0.958333 | [[713, 1], [1, 5]] |
| chb01_04 | Weighted LGBM | 0.997222 | 1.000000 | 0.666667 | 1.000000 | 0.800000 | 0.910714 | [[714, 0], [2, 4]] |
| chb01_15 | Ordinary RF | 0.987500 | 0.000000 | 0.000000 | 0.998596 | 0.000000 | 0.440566 | [[711, 1], [8, 0]] |
| chb01_15 | Weighted RF | 0.987500 | 0.000000 | 0.000000 | 0.998596 | 0.000000 | 0.271631 | [[711, 1], [8, 0]] |
| chb01_15 | Ordinary LGBM | 0.991667 | 0.750000 | 0.375000 | 0.998596 | 0.500000 | 0.649177 | [[711, 1], [5, 3]] |
| chb01_15 | Weighted LGBM | 0.987500 | 0.000000 | 0.000000 | 0.998596 | 0.000000 | 0.265673 | [[711, 1], [8, 0]] |
| chb01_16 | Ordinary RF | 0.995833 | 1.000000 | 0.700000 | 1.000000 | 0.823529 | 0.966923 | [[710, 0], [3, 7]] |
| chb01_16 | Weighted RF | 0.994444 | 0.800000 | 0.800000 | 0.997183 | 0.800000 | 0.965152 | [[708, 2], [2, 8]] |
| chb01_16 | Ordinary LGBM | 0.994444 | 0.875000 | 0.700000 | 0.998592 | 0.777778 | 0.926241 | [[709, 1], [3, 7]] |
| chb01_16 | Weighted LGBM | 0.995833 | 0.818182 | 0.900000 | 0.997183 | 0.857143 | 0.937374 | [[708, 2], [1, 9]] |

Confusion matrices use `[[TN,FP],[FN,TP]]`, true rows and predicted columns in
class order `[0,1]`. AP is the existing step-weighted PR summary, not trapezoidal
PR area. Rates in the tables below are fractions rather than percentages.

| Holdout | Model | Detected / missed | Latency (s) | FP windows | False episodes/h | Repeats | Exact alarm starts (s) |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| chb01_03 | Ordinary RF | 1 / 0 | 9 | 0 | 0 | 0 | 3005 (D) |
| chb01_03 | Weighted RF | 1 / 0 | 9 | 1 | 1 | 0 | 3005 (D), 3050 (FA) |
| chb01_03 | Ordinary LGBM | 1 / 0 | 9 | 0 | 0 | 0 | 3005 (D) |
| chb01_03 | Weighted LGBM | 1 / 0 | 9 | 2 | 1 | 0 | 3005 (D), 3050 (FA) |
| chb01_04 | Ordinary RF | 1 / 0 | 8 | 0 | 0 | 1 | 1475 (D), 1485 (R) |
| chb01_04 | Weighted RF | 1 / 0 | 8 | 1 | 0 | 1 | 1475 (D), 1490 (R) |
| chb01_04 | Ordinary LGBM | 1 / 0 | 8 | 1 | 0 | 0 | 1475 (D) |
| chb01_04 | Weighted LGBM | 1 / 0 | 8 | 0 | 0 | 0 | 1475 (D) |
| chb01_15 | Ordinary RF | 0 / 1 | undefined | 1 | 1 | 0 | 1785 (FA) |
| chb01_15 | Weighted RF | 0 / 1 | undefined | 1 | 1 | 0 | 1785 (FA) |
| chb01_15 | Ordinary LGBM | 1 / 0 | 8 | 1 | 1 | 1 | 1740 (D), 1765 (R), 1785 (FA) |
| chb01_15 | Weighted LGBM | 0 / 1 | undefined | 1 | 1 | 0 | 1785 (FA) |
| chb01_16 | Ordinary RF | 1 / 0 | 5 | 0 | 0 | 1 | 1020 (D), 1035 (R) |
| chb01_16 | Weighted RF | 1 / 0 | 5 | 2 | 2 | 1 | 870 (FA), 1010 (FA), 1020 (D), 1035 (R) |
| chb01_16 | Ordinary LGBM | 1 / 0 | 5 | 1 | 1 | 1 | 870 (FA), 1020 (D), 1035 (R) |
| chb01_16 | Weighted LGBM | 1 / 0 | 5 | 2 | 2 | 0 | 870 (FA), 1010 (FA), 1020 (D) |

D=detection, R=repeat, FA=false alarm. Starts are seconds from each recording's
beginning at actual window-end decision times. The first new positive episode
starting inside exact `[onset,offset)` detects a seizure; subsequent starts
inside it are repeats. Outside starts are false alarms. One negative decision
closes/re-arms an episode; state resets between files. False alarms/hour uses
all evaluated EEG hours including ictal time. Missed seizures have undefined
latency, never zero. No endpoint or label definition was changed.

FP windows and false episodes are distinct. For example, ordinary LightGBM has
one FP window in `04` but zero false episodes: `[1495,1500)` extends a valid
alarm that began at 1475. Weighted LightGBM's `03` episode extends to a positive
decision at 3040 after offset 3036, yet its episode began inside the seizure;
a later new start at 3050 is a false alarm. Exact end times, positive-window
counts and statuses remain in the saved per-recording alarm ledgers.

## Pooled and macro summaries

| Pooled metric | Ordinary RF | Weighted RF | Ordinary LGBM | Weighted LGBM |
| --- | ---: | ---: | ---: | ---: |
| accuracy | 0.994444 | 0.993750 | 0.995139 | 0.994097 |
| precision | 0.944444 | 0.791667 | 0.875000 | 0.800000 |
| recall | 0.531250 | 0.593750 | 0.656250 | 0.625000 |
| specificity | 0.999649 | 0.998244 | 0.998947 | 0.998244 |
| f1 | 0.680000 | 0.678571 | 0.750000 | 0.701754 |
| average_precision | 0.844954 | 0.773728 | 0.867054 | 0.751964 |
| Confusion matrix | [[2847, 1], [15, 17]] | [[2843, 5], [13, 19]] | [[2845, 3], [11, 21]] | [[2843, 5], [12, 20]] |
| Detected / missed | 3 / 1 | 3 / 1 | 4 / 0 | 3 / 1 |
| False episodes / hours | 1 / 4.0 | 4 / 4.0 | 2 / 4.0 | 4 / 4.0 |
| False episodes/hour | 0.25 | 1.0 | 0.5 | 1.0 |
| Repeat episodes | 2 | 2 | 2 | 0 |
| Mean detected latency (s) | 7.333333333333333 | 7.333333333333333 | 7.5 | 7.333333333333333 |

| Macro mean | Ordinary RF | Weighted RF | Ordinary LGBM | Weighted LGBM |
| --- | ---: | ---: | ---: | ---: |
| accuracy | 0.994444 | 0.993750 | 0.995139 | 0.994097 |
| precision | 0.750000 | 0.618750 | 0.864583 | 0.648990 |
| recall | 0.529167 | 0.585417 | 0.664583 | 0.610417 |
| specificity | 0.999649 | 0.998243 | 0.998947 | 0.998242 |
| f1 | 0.620168 | 0.600568 | 0.742063 | 0.620168 |
| average_precision | 0.817063 | 0.760932 | 0.843182 | 0.719951 |

Each recording is concatenated once per model. Pooled confusion counts are
summed before calculating rates; events are counted from distinct annotations,
never from windows or repeated alarms. Macro means weight each fold equally,
using only defined values and reporting their number. All actual macro window
metrics here are defined in four folds; precision in missed-event folds is
zero, because those models do predict a false positive. JSON retains `null`
for truly undefined metrics and includes population SD, min/max and n_defined.

First-detection latency on the three mutually detected events is **identical
for all four models: 9, 8 and 5 seconds**. Ordinary LightGBM additionally detects
`15` at 8 seconds. Its mean latency 7.5 s covers four events; the other models'
7.333 s mean covers only three. Those means cannot be used to claim the other
models detect faster. All medians are 8 s, and latency must be read beside the
miss count. Macro event sensitivities equal 1.0 for ordinary LightGBM and 0.75
for the other models, since each fold contains one event.

Pooled AP combines scores from separately fitted models; cross-fold calibration
has not been established. It is a descriptive ranking measure. Per-fold and
macro AP are also reported. Model scores are uncalibrated and are not clinical
seizure probabilities.

## chb01_15: detected by ordinary LightGBM, still problematic

The fixed ordinary model predicts positive at decision times **1740, 1765 and
1770**: windows `[1735,1740)`, `[1760,1765)` and `[1765,1770)`. Their scores are
approximately 0.749955, 0.755152 and 0.778164. The first decision is inside the
unchanged annotation `[1732,1772)`, giving latency 8 seconds. The later pair starts
one repeat episode. Window recall is only 3/8 (37.5%), despite detecting the event.
The other five ictal windows remain missed.

Weighted LightGBM predicts none of the eight ictal windows. Its maximum ictal
score is about 0.001150, compared with 0.778164 for ordinary LightGBM. Both RFs
also miss the event. All four models still alarm at **1785**, after offset;
ordinary LightGBM's detection therefore does **not** eliminate the previously
investigated postictal false alarm. Its score there is 0.963959, weighted
LightGBM's 0.997788. Those post-offset scores cannot establish physiological
origin or justify altering the annotation.

The ordinary LightGBM fold never trains on `15`; its input fingerprint and
holdout match RF exactly. Nevertheless the experiment was proposed after extensive
inspection of this recording and comparison of model families. This is an
exploratory within-development finding, not an independent test, a confirmation
of a solved failure, or grounds for optimizing specifically for that seizure.
No new raw-EEG investigation or causal feature attribution is performed here.

## Continuity, weighting and stability

| Holdout | Ordinary RF | Weighted RF | Ordinary LGBM | Weighted LGBM |
| --- | --- | --- | --- | --- |
| chb01_03 | 1 runs; longest 6 windows | 1 runs; longest 7 windows | 1 runs; longest 6 windows | 1 runs; longest 7 windows |
| chb01_04 | 2 runs; longest 3 windows | 2 runs; longest 2 windows | 1 runs; longest 5 windows | 1 runs; longest 4 windows |
| chb01_15 | 0 runs; longest 0 windows | 0 runs; longest 0 windows | 2 runs; longest 2 windows | 0 runs; longest 0 windows |
| chb01_16 | 2 runs; longest 5 windows | 2 runs; longest 6 windows | 2 runs; longest 5 windows | 1 runs; longest 9 windows |

The diagnostic above counts consecutive windows with both original label=1 and
prediction=1. It is not smoothing or a new event rule. Spans may extend slightly
beyond exact seizure boundaries because original labels require >=50% overlap;
these spans are not continuous clinical seizure coverage. Complete run intervals
are saved in `continuity.json`.

Ordinary LightGBM is more continuous than ordinary RF in `04` (five consecutive
true-positive windows versus three), equal in `03` and `16`, and detects `15`
in two short runs. Its total repeat episodes remain two, the same as either RF.
Weighted LightGBM has no repeats and a nine-window run in `16`, but misses `15`
entirely and covers fewer ictal windows in `04`. Fewer repeats alone therefore
does not establish a universally better detector or sustained seizure coverage.

Weighting helps LightGBM window recall in `03` (75% to 87.5%) and `16` (70% to
90%), hurts it in `04` (83.33% to 66.67%) and loses `15` (37.5% to zero). Relative
to ordinary LightGBM, it rescues three ictal windows across `03`/`16` but loses
four across `04`/`15`; it adds three FP windows and removes one. Pooled TP falls
21 to 20, FP rises 3 to 5, false episodes rise 2 to 4, and AP falls 0.867054 to
0.751964. These mixed effects do not support a consistent weighting benefit.
No model parameters were changed to investigate this outcome.

| Model | Metric | Macro mean | Population SD | Min–max |
| --- | --- | ---: | ---: | --- |
| Ordinary RF | recall | 0.529167 | 0.306951 | 0.000000–0.750000 |
| Ordinary RF | precision | 0.750000 | 0.433013 | 0.000000–1.000000 |
| Weighted RF | recall | 0.585417 | 0.346128 | 0.000000–0.875000 |
| Weighted RF | precision | 0.618750 | 0.358545 | 0.000000–0.875000 |
| Ordinary LGBM | recall | 0.664583 | 0.173843 | 0.375000–0.833333 |
| Ordinary LGBM | precision | 0.864583 | 0.090211 | 0.750000–1.000000 |
| Weighted LGBM | recall | 0.610417 | 0.363880 | 0.000000–0.900000 |
| Weighted LGBM | precision | 0.648990 | 0.383930 | 0.000000–1.000000 |

Ordinary LightGBM has the lowest observed across-fold recall and precision SDs,
with no zero-recall fold. These are descriptive statistics of four events, not
population uncertainty estimates or confidence intervals. Lower variability
partly reflects its three positive windows in `15`, where the other models have
zero true positives. It does not imply reliable behavior on new patients.

![Four-model out-of-fold score comparison](../results/lightgbm_cv_v1/four_model_seizure_scores.png)

Markers show actual decisions; connecting lines are visual guides, not new
predictions between windows or temporal filtering. The figure is a seizure-local
view; complete tables/evaluator exposure include all one-hour recordings and
false alarms outside the plotted ranges, such as the start at 870 in `16`.

## Interpretation and provisional model recommendation

Ordinary LightGBM improves pooled window recall, F1 and AP and detects one more
event than ordinary RF on these four folds, at a cost of lower precision
(87.5% versus 94.44%) and more false episodes (0.5/h versus 0.25/h). This is a
sensitivity/false-alarm trade-off. Both weighted models produce 1 false episode/h
and miss `15`; weighting has no event-level advantage here. No model is strictly
best on every endpoint.

I recommend **ordinary LightGBM as the provisional primary research candidate**,
with ordinary RF retained as the lower-false-alarm reference. This prioritizes
observed event sensitivity and stability, not a claim of clinical readiness.
Before a final choice, agree on the acceptable event-miss/false-alarm trade-off
and freeze the existing configuration and endpoint rules. A preference for lower
false-alarm burden could defensibly favor ordinary RF instead. Neither weighted
variant has evidence here strong enough to justify priority over these ordinary
models. No model is retrained on a larger pool or selected using reserved data
in this phase. Temporal filtering and final evaluation remain future decisions.

The data contain four seizures from one patient; each fold fits only three events
and 22–26 correlated positive windows. Training sets overlap, and both recordings
and model family choices have been examined repeatedly during development.
This is not prospective chronological evaluation: training can include later
recordings. Events within one patient are not independent patient samples, and
window counts exaggerate the amount of independent evidence. False-alarm exposure
is four hours of seizure-containing recordings; the two normal-only recordings
are always used for fitting and have no out-of-fold false-alarm estimate here.
Observed 4/4 detection cannot establish clinical reliability or population
sensitivity. Original validation/test/supplementary data were not used to resolve
uncertainty. No new experiments were introduced after seeing the results.

## Files, tests and reproduction

Added `src/lightgbm_cross_validation.py`, six focused synthetic tests in
`tests/test_lightgbm_cross_validation.py`, this report and a README link. No
existing source, preprocessing, manifest, label, requirements or RF result was
modified. Earlier uncommitted work is preserved. No dependencies installed,
commits made or changes pushed.

`results/lightgbm_cv_v1/` contains frozen configuration and versions, fold-specific
weights/parameters, eight new LightGBM models in joblib/native text formats,
eight complete prediction tables, aligned four-model fold tables, sixteen
per-fold evaluator reports, pooled/macro summaries, continuity/changed-window
diagnostics, one figure, verification and artifact hashes. Initial-run tests and
a before/after preservation audit accompany the outputs. Both LightGBMs actually
fit 150 trees in every fold; no Python warnings were captured.

The full suite ran 100 tests: **99 passed, one skipped**, no failures. The skipped
existing integration check reads original validation/test feature rows. All other
tests ran; historical experiment tests use synthetic validation/test fixtures,
and existing preprocessing tests read only the original three training EDF headers.
Six new tests verify fold isolation, distinct fold-derived weights, exact feature
selection and identical fit inputs, no validation fit/early stopping, prediction
alignment, eight independent model instances, reloads, deterministic repeated
synthetic fits, unique event aggregation/undefined metrics, RF preservation and
training-only decoding (reserved synthetic rows deliberately contain NaNs).
Synthetic test fits are distinct from the eight actual-data fits. Saved RF
metrics, fold inputs and pooled/macro summaries reproduce without RF refitting.

From the repository root, using the existing environment and a fresh directory:

```sh
MPLCONFIGDIR=/tmp/nyas-matplotlib venv/bin/python -m src.lightgbm_cross_validation \
  --data data/processed/chb01_detection_features_11recordings_split.parquet \
  --manifest config/chb01_recordings.csv \
  --rf-dir results/rf_cv_v1 \
  --lightgbm-config results/lightgbm_v1/configuration.json \
  --output results/lightgbm_cv_reproduction

MPLCONFIGDIR=/tmp/nyas-matplotlib venv/bin/python -m unittest discover -s tests \
  -p test_lightgbm_cross_validation.py -v
```

Existing output directories are refused. The protected full-suite command is:

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
                    unittest.skip('Phase 2H: no reserved validation/test rows')(method))
protect(suite)
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
PY
```

Work stops after the specified eight LightGBM fits and their cross-validation
comparison. No temporal filtering, final testing, new datasets or dashboard work
was performed.
