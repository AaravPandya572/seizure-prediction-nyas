# NYAS BrainTech — technical research evidence package

Prepared 2026-11-10 (Asia/Kolkata). Repository HEAD: `f543e44d378ad3c1846ead8fb12c684ab61323bf`. The working tree was clean at inspection. This is an evidence catalogue, not a research paper. Only this new Markdown document was written; no experiment, model prediction, training, preprocessing or test suite was run. Numerical tables below transcribe saved outputs or summarize existing dataset rows. No raw EEG signal was reprocessed and no fitted estimator was deserialized.


Quick navigation: [chronology](#1-chronological-experiment-and-decision-ledger) · [dataset changes](#2-dataset-progression-and-preprocessing-changes) · [Git milestones](#4-git-milestones-and-execution-chronology) · [gaps and qualifications](#5-contradictions-qualifications-and-missing-context) · [exact configurations](#appendix-a-exact-configurations-and-fold-definitions) · [numerical ledger](#appendix-b-complete-saved-evaluation-result-ledger) · [historical reports](#appendix-c-historical-report-evidence-by-phase) · [tests](#appendix-d-test-evolution-and-saved-verification-records) · [figures](#appendix-e-figure-inventory) · [artifact inventory](#appendix-f-artifact-and-implementation-inventory).

## How to use this package

Sections 1–7 provide the research chronology, data/method changes, provenance and unresolved questions. Appendices A–F provide exact configurations, complete saved evaluation metrics and alarm ledgers, historical report evidence, test records, figures and artifact inventories. Relative links resolve from this document in `docs/`; JSON selectors identify the source fields. Rates are fractions unless a source explicitly marks percentages. `null` means undefined, never a measured zero. Confusion matrices use `[[TN,FP],[FN,TP]]`. Repeated appearances of the same saved RF reference are not independent experiments.

The repository contains a three-week calendar span (2026-10-20 through 2026-11-10), but does not document a daily schedule or exact week-by-week allocation. Phase order is supported by reports and dependencies; commit dates are preservation milestones, not necessarily execution dates. Motivations below are documented objectives or explicitly labeled methodological questions, not reconstructed personal intentions. Original ChatGPT discussions and the initial audit are not repository artifacts.

## 1. Chronological experiment and decision ledger

### Initial prediction proposal and prototype

**Source:** `Git `4454197:src/preprocess.py`; initial scaffold `e0a0f07``

**Objective / hypothesis:** Operational hypothesis: features from the 30 minutes before onset might support a warning classifier. This is inferred from `PREICTAL_HORIZON_SEC=1800` and preictal labels; a written scientific rationale, literature review and prespecified seizure prediction horizon are missing.

**Dataset / methodology / configuration:** Three recordings 01/03/04. First half of normal 01 plus pre-onset 03 are training; second half of 01 plus pre-onset 04 are test. Only first seizure in each file is used. Five-second windows, optional preictal stride, 18 first eligible signals, 126 features, notch plus MNE IIR filtering.

**Results:** Saved legacy dataset has 1,612 rows: 959 class 0, 653 preictal class 1. No saved trained preictal classifier or prediction-performance report was found.

**Limitations / documented subsequent decision:** Prediction labeling is not demonstrated forecasting. Same normal recording spans train/test; first-only seizure handling, channel selection, PSD integration and noncausal-default filtering need care. Source preserved before replacement. The reason/date of the human decision to switch objectives needs conversation context.

### Week 1: seizure detection preprocessing and evaluator

**Source:** [README.md](../README.md); [src/preprocess.py](../src/preprocess.py); [docs/evaluation.md](../docs/evaluation.md)

**Objective / hypothesis:** Reframe the target as identifying ongoing ictal EEG and make labels, features and causal decision timing reproducible. This is documented implementation intent, not a tested superiority hypothesis.

**Dataset / methodology / configuration:** Initially 01/03/04 only; entire recordings, fixed canonical channels, continuous causal SOS filtering, five-second nonoverlapping windows and >=50% ictal overlap. Evaluator separates window labels from exact event matching.

**Results:** 2,160 rows: 2,146 nonictal and 14 ictal; two annotated events. Public evaluator supplies window and event metrics with undefined rates explicit.

**Limitations / documented subsequent decision:** Two events from one patient cannot support generalization. Dropped partial tails in preprocessing conflict with evaluator full-coverage requirements for arbitrary future inputs; selected files have no such tails. Proceeded to recording-level expansion.

### Phase 2A: expansion and fixed splits

**Source:** [docs/dataset_preparation.md](../docs/dataset_preparation.md)

**Objective / hypothesis:** Prepare a reproducible dataset with whole-recording partitions; no classification hypothesis was tested.

**Dataset / methodology / configuration:** 11 recordings; six train, two validation, three test. Verify raw headers/payload, annotation clocks, channels, frequency, features and metadata. Preserve earlier datasets.

**Results:** 7,665 windows, 90 ictal, seven events, 38,325 seconds. Train 4,320/32 ictal/4 events; validation 1,440/18/1; test 1,905/40/2. chb01_26 has 465 windows and 2,325 seconds.

**Limitations / documented subsequent decision:** Local checksums establish local consistency, not independent PhysioNet authenticity. Fixed test reserved for final inference; descriptive integrity and preprocessing were allowed beforehand.

### Phase 2B: majority dummy and ordinary Random Forest

**Source:** [docs/baseline_experiment.md](../docs/baseline_experiment.md)

**Objective / hypothesis:** Establish whether a fixed classifier detects seizures beyond the high accuracy obtainable by predicting the majority class.

**Dataset / methodology / configuration:** Train only six recordings, evaluate 17/18. Dummy most_frequent; RF 200 trees, seed 42, class_weight=None, four threads, original defaults. 126-feature whitelist; no scaling, sampling or tuning.

**Results:** Dummy accuracy 98.75%, recall 0, event 0/1. RF recall 61.1111%, AP 0.982948, 1/1 detected at 10 s; zero false alarms in two hours and two repeats.

**Limitations / documented subsequent decision:** One validation event and correlated windows limit inference. The report proposes a controlled balanced-weight RF comparison next.

### Phase 2C: balanced Random Forest

**Source:** [docs/weighted_experiment.md](../docs/weighted_experiment.md)

**Objective / hypothesis:** Test whether training-derived class weighting improves seizure recall under severe imbalance.

**Dataset / methodology / configuration:** Clone all original RF settings; change only class_weight to balanced. Same training and validation rows; no refitting of original reference. Balanced weights follow N/(2*N_class).

**Results:** Recall fell from 61.1111% to 50%; F1 0.758621 to 0.666667; both detect 1/1 with zero false alarms. Weighted latency improves 10 to 5 s; repeats fall 2 to 1. One original miss rescued, three new ictal misses.

**Limitations / documented subsequent decision:** Weighting does not uniformly raise positive scores. Earlier onset and fewer repeats do not imply better window classification. Broader training-only recording CV follows.

### Phase 2D: Random Forest stability

**Source:** [docs/rf_cross_validation.md](../docs/rf_cross_validation.md)

**Objective / hypothesis:** Assess stability across the four seizure-containing training recordings, comparing ordinary and balanced RF.

**Dataset / methodology / configuration:** Four leave-one-seizure-recording-out folds: hold out 03,04,15,16; always train on normal 01/14 plus the other three seizure recordings. 3,600 rows and 22–26 ictal windows per fit. Fold-local class weights.

**Results:** Both detect 3/4, miss 15, with detected latencies 9/8/5 s. Ordinary recall 53.125%, 0.25 false alarms/h; weighted 59.375%, 1.0/h. Both give a post-offset false alarm at 1785 s in 15.

**Limitations / documented subsequent decision:** Overlapping fold training sets, one patient, and no normal-only OOF exposure. Report recommends ordinary RF reference and proposes supplementary false-alarm work; that proposal was not the next executed phase. Failure analysis followed.

### Phase 2E: missed-seizure feature investigation

**Source:** [docs/failure_analysis.md](../docs/failure_analysis.md)

**Objective / hypothesis:** Investigate why chb01_15 was missed, without changing the endpoint or threshold. Physiological/artifact explanations are hypotheses, not findings.

**Dataset / methodology / configuration:** Saved RF OOF predictions/models and training-only feature data; compare pre/during/post scores, all 126 features and event-level reference contrasts. No EDF opened for this phase.

**Results:** Both max ictal scores stay below 0.5 (0.395 ordinary, 0.185 weighted). Only alarm is 1785 s, 13 s after offset and 53 s after onset. This is not detection latency. Event variance is not uniformly weak; line length and gamma are lower than detected exemplars.

**Limitations / documented subsequent decision:** Feature association is not attribution or causality. A postictal burst coincides with score peaks. The report recommends targeted raw EEG and annotation review before changing features.

### Phase 2F: targeted raw EEG investigation

**Source:** [docs/raw_eeg_investigation.md](../docs/raw_eeg_investigation.md)

**Objective / hypothesis:** Check whether the postictal high-frequency fraction reflects absolute power, filtering artifacts or other signal changes.

**Dataset / methodology / configuration:** Only training EDFs 15/03/04/16. Continuous causal prefix filtering reproduces prior state; inspect 1700–1810 s and 1775–1790 s in 15, fixed midpoint exemplars elsewhere. Same Welch units and channels.

**Results:** All 126 features reproduce exactly for 28 checked windows (maximum absolute error 0). Raw and filtered postictal fast burst exists. FP2-F8 filtered gamma rises from 2.21 to 1690.96 µV² in selected mid-ictal/alarm windows, about 764-fold.

**Limitations / documented subsequent decision:** Burst origin remains unresolved without expert/video/EMG context. No software/label fault found; seizure remains missed. A separate fixed LightGBM comparison is discussed, not a proven cure for the failure.

### Phase 2G: fixed LightGBM comparison

**Source:** [docs/lightgbm_experiment.md](../docs/lightgbm_experiment.md)

**Objective / hypothesis:** Compare ordinary and cost-sensitive LightGBM with saved RF references under the same protocol.

**Dataset / methodology / configuration:** 150 trees, learning_rate .05, depth 3, seven leaves, min_child_samples 10, seed 42, deterministic CPU settings. Ordinary positive weight 1; weighted 4288/32=134.0. Same train/validation only.

**Results:** Both LightGBMs recall 55.5556%, detect 1/1 at 5 s and produce 0.5 false episodes/h. Weighted changes the location of its one false alarm, rescues no ictal window, and lowers AP from 0.847912 to 0.690848.

**Limitations / documented subsequent decision:** Ordinary RF remains stronger on validation recall/F1/AP and false alarms; LightGBM is earlier. Repeated validation use limits selection claims. Report proposes identical training-recording CV.

### Phase 2H: LightGBM stability

**Source:** [docs/lightgbm_cross_validation.md](../docs/lightgbm_cross_validation.md)

**Objective / hypothesis:** Test the fixed LightGBM settings on the same four original training folds; assess stability and missed-event behavior.

**Dataset / methodology / configuration:** Eight LightGBM fits: ordinary/weighted per fold; fold-local positive weights. Saved RF references reused, no RF refits.

**Results:** Ordinary LGBM detects 4/4, recall 65.625%, F1 .75, AP .867054, 0.5 false episodes/h. Weighted LGBM detects 3/4, recall 62.5%, 1.0/h. Only ordinary LGBM detects 15 (8 s initial latency).

**Limitations / documented subsequent decision:** Small overlapping folds and repeated examination of 15 remain exploratory. Report recommends ordinary LGBM provisionally primary, ordinary RF lower-false-alarm reference; no model is best on every endpoint.

### Phase 3A: fixed causal persistence rules

**Source:** [docs/temporal_filtering.md](../docs/temporal_filtering.md)

**Objective / hypothesis:** Can two consecutive positives or two of the latest three reduce false/repeat alarms without unacceptable event loss or delay?

**Dataset / methodology / configuration:** Saved ordinary RF/LGBM OOF predictions primary; original validation secondary. No model fit. State resets per file, available startup history only, reject missing/incomplete windows, never backdate.

**Results:** Both filters remove observed CV false episodes while preserving baseline event totals, but LGBM 15 latency grows from 8 to 38 s, only 2 s before offset. Window recall falls for consecutive filtering. Exact changes and all event delays are in appendices.

**Limitations / documented subsequent decision:** No numerical latency budget was prespecified. Report recommends retaining no filtering because the extra 30 s is not a defensible improvement on this small sample.

### Phase 3B: freeze

**Source:** [docs/final_evaluation_protocol.md](../docs/final_evaluation_protocol.md)

**Objective / hypothesis:** Lock artifacts, features, threshold, alarm policy and evaluation scope before final inference.

**Dataset / methodology / configuration:** Existing Phase 2G ordinary LGBM primary and Phase 2B RF reference; no train+validation refit. 126 ordered features, score >.5 (ties 0), no calibration or filtering. Primary 21/22/26; supplementary 23/24/25 separate.

**Results:** Model reloads and historical validation reproduction passed. 111 tests passed, one test-data integration check deferred. Source/model/config/data hashes recorded despite then-dirty tree.

**Limitations / documented subsequent decision:** RF serialized binary does not embed training row IDs. Exact provenance relies on saved code/config/data. Supplementary preparation remained pending; final evaluation separately authorized later.

### Phase 3C: first final held-out evaluation

**Source:** [docs/final_test_results.md](../docs/final_test_results.md)

**Objective / hypothesis:** Measure both frozen models once on the reserved recording test; descriptive comparison only.

**Dataset / methodology / configuration:** Exact frozen command, unchanged artifacts and 1,905 test windows; all 9,525 seconds included.

**Results:** Both detect 2/2 at 8 and 13 s, recall 77.5%, F1 .873239, 0 false alarms/h observed; both matrices [[1865,0],[9,31]]. AP .947176 LGBM/.942249 RF; repeats 1/3. Four differing window decisions despite equal pooled matrices.

**Limitations / documented subsequent decision:** Two events do not establish clinical sensitivity; short zero-alarm exposure does not prove zero underlying rate. Frozen roles retained; supplementary false-alarm assessment remains unexecuted in this repository.

## 2. Dataset progression and preprocessing changes

| Saved dataset | Rows | Columns | Class 0 | Class 1 | Meaning of class 1 |

| --- | --- | --- | --- | --- | --- |

| [data/processed/chb01_detection_features.parquet](../data/processed/chb01_detection_features.parquet) | 2160 | 144 | 2146 | 14 | ictal |

| [data/processed/chb01_detection_features_11recordings.parquet](../data/processed/chb01_detection_features_11recordings.parquet) | 7665 | 144 | 7575 | 90 | ictal |

| [data/processed/chb01_detection_features_11recordings_split.parquet](../data/processed/chb01_detection_features_11recordings_split.parquet) | 7665 | 145 | 7575 | 90 | ictal |

| [data/processed/chb01_features.parquet](../data/processed/chb01_features.parquet) | 1612 | 130 | 959 | 653 | preictal |



Counts above are read-only summaries of saved `label` columns, not regenerated features. Legacy split: 01 train/test each 360 class-0 rows; 03 train has 239 class-0 and 360 preictal rows; 04 test has 293 preictal rows and no class-0 rows. The 30-minute region is truncated at recording start for 04. Source: legacy Parquet `source_file, split_role, label`, and Git `4454197:src/preprocess.py::slice_and_extract` / main block.

| Aspect | Original code at 4454197 | Current detection evidence |

| --- | --- | --- |

| Target | 30 minutes before first onset; omit seizure/postictal samples in seizure files | >=50% union of validated ictal intervals; all complete recording windows |

| Channels | First 18 eligible names | Fixed ordered 18 bipolar channels; explicit T8-P8-0 duplicate mapping |

| Filtering | MNE notch 60 Hz and IIR bandpass with phase unspecified; not explicitly causal | Continuous forward SciPy SOS, zero initial state per file, .5–45 Hz; no notch |

| PSD | Separate trapezoid integrations for half-open bands and inclusive denominator; edge gaps | Identical frequency-bin sums; .5–45 denominator, gamma includes 45 |

| Annotations | First seizure used downstream; parser lacks count/pair/bounds validation | Numbered/un-numbered pairs, event counts/order/overlap/bounds validated |

| Metadata | time_sec, source_file, split_role and label | Start/end samples and seconds, end-time decision, patient/recording, overlap/event IDs, frequency/duration and quality flags |

| Splits | Half of one normal recording in train and half in test | Whole recording assignments via fixed manifest |



Source: [src/preprocess.py](../src/preprocess.py) functions `parse_summary_file`, `select_channels`, `causal_bandpass`, `extract_epoch_features`, `window_annotation`, `slice_and_extract`; [src/split_dataset.py](../src/split_dataset.py); historical source from Git. Current numerical settings and feature order appear in Appendix A. Historical claims that filtering removes all muscle noise or that gamma is a “key preictal marker” are comments, not validated results. Exact original MNE phase behavior should be checked against the original runtime if a historical noncausality claim is needed.

## 3. Evaluation definitions and interpretation

Source: [docs/evaluation.md](../docs/evaluation.md) and [src/evaluate.py](../src/evaluate.py). AP is step-weighted average precision, not trapezoidal PR area. Event detection requires a NEW alarm at a window-end time in [onset, offset); first hit detects, further starts inside the same event are repeats. A negative decision ends the episode and re-arms; no cooldown. An already-active pre-onset alarm does not detect an event. False-alarm denominator is all evaluated EEG hours including ictal time; recording gaps are excluded. Undefined ratios remain null. Mean latency averages detected events only. Accuracy, event sensitivity, window recall and repeat alarms answer different questions. Temporal-filter AP uses unchanged raw scores and is not evidence that filtered classification improved score ranking.

## 4. Git milestones and execution chronology

| Commit | Author timestamp | Milestone |

| --- | --- | --- |

| e0a0f0778fedf66b5fd1ac6fc6eb35c9263408d6 | 2026-10-20T13:52:39+05:30 | Initial project structure for NYAS BrainTech Seizure Prediction |

| 4454197b0def937de4ee56bd681f861c7cab767d | 2026-10-23T13:12:09+05:30 | Saving original preictal preprocessing code. |

| e36245cb7605787af35cc6b3e1637a30012ea0fc | 2026-10-27T13:03:26+05:30 | Complete EEG dataset preparation and evaluation framework |

| a1359fa4b64f94dd4eab45c82b1d03952f3d79f6 | 2026-10-28T22:40:35+05:30 | Implement Random Forest baseline model and validation. |

| 8798eb59f22716edbbcb0a7e981c0fc92bb1e687 | 2026-11-10T15:20:04+05:30 | Freeze seizure detection models and final evaluation protocol |

| f543e44d378ad3c1846ead8fb12c684ab61323bf | 2026-11-10T17:13:07+05:30 | Final Testing Results |



| Run timestamp source | Recorded UTC time |

| --- | --- |

| [results/lightgbm_cv_v1/configuration.json](../results/lightgbm_cv_v1/configuration.json) | 2026-11-10T07:41:04.477991+00:00 |

| [results/temporal_filtering_v1/configuration.json](../results/temporal_filtering_v1/configuration.json) | 2026-11-10T08:07:04.765626+00:00 |

| [results/baseline_v1/configuration.json](../results/baseline_v1/configuration.json) | 2026-10-27T07:42:02.925447+00:00 |

| [results/rf_weighted_v1/configuration.json](../results/rf_weighted_v1/configuration.json) | 2026-11-07T13:45:19.453422+00:00 |

| [results/rf_cv_v1/configuration.json](../results/rf_cv_v1/configuration.json) | 2026-11-07T14:05:42.761849+00:00 |

| [results/lightgbm_v1/configuration.json](../results/lightgbm_v1/configuration.json) | 2026-11-10T07:12:18.707676+00:00 |

| [results/final_test_v1/execution_record.json](../results/final_test_v1/execution_record.json) | 2026-11-10T09:54:24.761328+00:00 |



The baseline run timestamp precedes its commit date; this is normal run-versus-commit chronology, not a contradiction. Failure/raw-EEG configuration files lack `created_utc`; precise execution dates need external context. Phases 2C–3B were preserved together in 8798eb5, so that commit cannot date each individual action. Original experiment commands are historical records, not commands to rerun during evidence gathering.

## 5. Contradictions, qualifications and missing context

- **Stale status text:** README still says final testing is pending; the frozen protocol also intentionally preserves its pre-authorization status. Phase 3C execution_record.json and final_test_results.md establish completion. Do not silently edit the historical freeze.

- **Original hypothesis:** the code operationalizes a 30-minute preictal zone but does not establish a clinical seizure-prediction horizon, rationale for 30 minutes, or prespecified forecasting endpoint. Supply the original hypothesis and audit discussion from ChatGPT.

- **Initial audit and pivot:** source changes demonstrate corrections, but no standalone original audit or dated decision memo is present. Do not invent the reason the team abandoned forecasting.

- **Meaning of “independent”:** events/recordings are held out, but all are Patient 01; windows and seizures are correlated. Do not describe this as independent-patient validation or prospective chronological validation.

- **15 was not rescued by relabeling:** RF post-offset 1785 s is a false alarm; 13 s after offset is not successful detection latency. LGBM later detects the same event under its own held-out fold; validation LGBM includes 15 in fitting and cannot demonstrate rescue of that held-out failure.

- **Model ranking varies by endpoint and dataset:** RF leads validation window recall/AP; ordinary LGBM leads CV event detection at higher false alarms; final core outcomes tie. This is a documented trade-off, not inconsistent reporting.

- **False-positive windows versus false episodes:** an FP window can continue an existing ictal alarm without starting a new false episode. Two-of-three can add positive windows, including after offset. These are expected consequences of the frozen rules.

- **Artifact attribution unresolved:** the raw postictal burst exists but physiology versus movement/muscle/electrode origin remains unknown. No expert adjudication/video/EMG evidence was found; feature associations do not establish model causality.

- **No formal latency budget:** the temporal-filter recommendation is a reasoned development decision, not an optimization against a prespecified numerical acceptable delay.

- **Test access wording:** test EDF integrity, annotation checks and preprocessing occurred in Phase 2A, and earlier test suites could read test feature rows. “Sealed” means reserved from model fitting/selection/prediction, not never read for any purpose. Later integration skips are documented separately.

- **Absent experiments:** EEGNet/training files and dashboard scaffold must not be presented as executed models. Dependency availability (e.g. torch, SHAP, Streamlit) does not establish their use. No supplementary 23/24/25 evaluation artifacts were found.

- **Provenance limits:** local raw hashes were not checked against independent PhysioNet download checksums in the documented workflow. RF does not serialize training row IDs. Hashes establish artifact consistency, not clinical correctness or complete history outside this repository.

- **Missing early test logs:** current test source and later suite logs establish coverage, but do not prove exact first-day counts/timing where no historical log exists. Test counts across phases use different scopes and cannot be added as unique tests.

- **Calendar detail:** exact daily work, hypothesis conversations, mentor feedback, author contributions and reasons for choosing the fixed model sizes need user-supplied context. Git milestones support the calendar span but not a three-week daily narrative.



## 6. Evidence strength and defensible endpoints

Observed results support an offline, patient-specific case study of severe imbalance, class weighting, model-dependent missed events and the delay cost of persistence. They do not establish clinical deployment, calibrated probabilities, a causal EEG failure mechanism, or robust population superiority. Keep unsuccessful weighting, the missed RF event and the rejected temporal filters in the research history. Preserve both frozen final model results and the natural class distribution. Supplementary zero-seizure data, if later authorized, can assess additional false-alarm exposure but not seizure sensitivity or latency.

## 7. Reading map

Appendix A: exact settings and folds. Appendix B: saved classifier/filter metrics and event/alarm ledgers. Appendix C: historical report prose and tables (all numerical observations retained; executable examples omitted). Appendix D: historical test outcomes and source coverage. Appendix E: every available research figure. Appendix F: numerical artifacts, models, provenance and source inventory. Appendix G: preservation checks performed for this document.

## Appendix A. Exact configurations and fold definitions

### Frozen features

Source: [config/final_evaluation_protocol.json](../config/final_evaluation_protocol.json) → `features`.

```json
{
  "columns": [
    "FP1-F7_var",
    "FP1-F7_line_length",
    "FP1-F7_delta",
    "FP1-F7_theta",
    "FP1-F7_alpha",
    "FP1-F7_beta",
    "FP1-F7_gamma",
    "F7-T7_var",
    "F7-T7_line_length",
    "F7-T7_delta",
    "F7-T7_theta",
    "F7-T7_alpha",
    "F7-T7_beta",
    "F7-T7_gamma",
    "T7-P7_var",
    "T7-P7_line_length",
    "T7-P7_delta",
    "T7-P7_theta",
    "T7-P7_alpha",
    "T7-P7_beta",
    "T7-P7_gamma",
    "P7-O1_var",
    "P7-O1_line_length",
    "P7-O1_delta",
    "P7-O1_theta",
    "P7-O1_alpha",
    "P7-O1_beta",
    "P7-O1_gamma",
    "FP1-F3_var",
    "FP1-F3_line_length",
    "FP1-F3_delta",
    "FP1-F3_theta",
    "FP1-F3_alpha",
    "FP1-F3_beta",
    "FP1-F3_gamma",
    "F3-C3_var",
    "F3-C3_line_length",
    "F3-C3_delta",
    "F3-C3_theta",
    "F3-C3_alpha",
    "F3-C3_beta",
    "F3-C3_gamma",
    "C3-P3_var",
    "C3-P3_line_length",
    "C3-P3_delta",
    "C3-P3_theta",
    "C3-P3_alpha",
    "C3-P3_beta",
    "C3-P3_gamma",
    "P3-O1_var",
    "P3-O1_line_length",
    "P3-O1_delta",
    "P3-O1_theta",
    "P3-O1_alpha",
    "P3-O1_beta",
    "P3-O1_gamma",
    "FP2-F4_var",
    "FP2-F4_line_length",
    "FP2-F4_delta",
    "FP2-F4_theta",
    "FP2-F4_alpha",
    "FP2-F4_beta",
    "FP2-F4_gamma",
    "F4-C4_var",
    "F4-C4_line_length",
    "F4-C4_delta",
    "F4-C4_theta",
    "F4-C4_alpha",
    "F4-C4_beta",
    "F4-C4_gamma",
    "C4-P4_var",
    "C4-P4_line_length",
    "C4-P4_delta",
    "C4-P4_theta",
    "C4-P4_alpha",
    "C4-P4_beta",
    "C4-P4_gamma",
    "P4-O2_var",
    "P4-O2_line_length",
    "P4-O2_delta",
    "P4-O2_theta",
    "P4-O2_alpha",
    "P4-O2_beta",
    "P4-O2_gamma",
    "FP2-F8_var",
    "FP2-F8_line_length",
    "FP2-F8_delta",
    "FP2-F8_theta",
    "FP2-F8_alpha",
    "FP2-F8_beta",
    "FP2-F8_gamma",
    "F8-T8_var",
    "F8-T8_line_length",
    "F8-T8_delta",
    "F8-T8_theta",
    "F8-T8_alpha",
    "F8-T8_beta",
    "F8-T8_gamma",
    "T8-P8_var",
    "T8-P8_line_length",
    "T8-P8_delta",
    "T8-P8_theta",
    "T8-P8_alpha",
    "T8-P8_beta",
    "T8-P8_gamma",
    "P8-O2_var",
    "P8-O2_line_length",
    "P8-O2_delta",
    "P8-O2_theta",
    "P8-O2_alpha",
    "P8-O2_beta",
    "P8-O2_gamma",
    "FZ-CZ_var",
    "FZ-CZ_line_length",
    "FZ-CZ_delta",
    "FZ-CZ_theta",
    "FZ-CZ_alpha",
    "FZ-CZ_beta",
    "FZ-CZ_gamma",
    "CZ-PZ_var",
    "CZ-PZ_line_length",
    "CZ-PZ_delta",
    "CZ-PZ_theta",
    "CZ-PZ_alpha",
    "CZ-PZ_beta",
    "CZ-PZ_gamma"
  ],
  "count": 126,
  "channels": [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ"
  ],
  "window_sec": 5.0,
  "stride_sec": 5.0,
  "sampling_rate_hz": 256.0,
  "per_channel": [
    "variance_uv2",
    "line_length_uv",
    "delta",
    "theta",
    "alpha",
    "beta",
    "gamma"
  ],
  "relative_psd_denominator_hz": [
    0.5,
    45.0
  ],
  "scaling_or_new_transform": null,
  "metadata_predictors": false
}
```

### Frozen preprocessing

Source: [config/final_evaluation_protocol.json](../config/final_evaluation_protocol.json) → `preprocessing`.

```json
{
  "processing_version": "detection-v1",
  "causal_filter": "existing fourth-order Butterworth 0.5-45Hz SOS, zero state per file, continuous forward filtering",
  "notch": false,
  "welch": "2s Hann; 1s overlap; constant detrending; 0.5Hz bin sums",
  "band_edges_hz": {
    "delta": [
      0.5,
      4
    ],
    "theta": [
      4,
      8
    ],
    "alpha": [
      8,
      13
    ],
    "beta": [
      13,
      30
    ],
    "gamma": [
      30,
      45
    ]
  },
  "band_endpoints": "half-open except gamma includes 45Hz",
  "power_floor_uv2": 1e-12,
  "duplicate_channel": "canonical T8-P8 uses first duplicate T8-P8-0",
  "warmup_and_postictal": "retained; quality flags are metadata, not predictors",
  "label": "1 iff >=50% overlap with union of validated half-open ictal intervals",
  "partial_tail": "existing preprocessing drops an incomplete tail with warning; evaluator requires complete coverage, so stop for review if any assessment has incomplete coverage"
}
```

### Frozen decisions

Source: [config/final_evaluation_protocol.json](../config/final_evaluation_protocol.json) → `decisions`.

```json
{
  "positive_class": 1,
  "threshold": 0.5,
  "comparison": ">",
  "class_at_exact_tie": 0,
  "implementation": "model.predict and score from predict_proba[:,1], classes=[0,1]",
  "timestamp": "actual end_sec; decision_time_sec=end_sec",
  "temporal_filter": null,
  "probability_calibration": null
}
```

### Frozen software

Source: [config/final_evaluation_protocol.json](../config/final_evaluation_protocol.json) → `software`.

```json
{
  "python": "3.14.5",
  "platform": "macOS-26.3.2-arm64-arm-64bit-Mach-O",
  "architecture": "arm64",
  "packages": {
    "lightgbm": "4.7.0",
    "numpy": "2.5.4",
    "pandas": "3.0.6",
    "scipy": "1.18.1",
    "scikit-learn": "1.9.1",
    "joblib": "1.6.0",
    "pyarrow": "26.0.0",
    "matplotlib": "3.11.2",
    "mne": "1.13.2"
  },
  "libomp": {
    "path": "/opt/homebrew/Cellar/libomp/23.1.4/lib/libomp.dylib",
    "version": "23.1.4",
    "sha256": "432b8d15fc1b2431f937bba89796a32daaa2dd464eda12a744d9cb2818842477"
  },
  "lightgbm_native_library": {
    "path": "/Users/aaravpandya/seizure-prediction-nyas/venv/lib/python3.14/site-packages/lightgbm/lib/lib_lightgbm.dylib",
    "sha256": "bc392db609d97730a9ed2acec7a56529b356c03dfad39bebf689523fed7dff18"
  }
}
```

### results/lightgbm_cv_v1/configuration.json

Source: [results/lightgbm_cv_v1/configuration.json](../results/lightgbm_cv_v1/configuration.json). Hash/provenance fields remain in the linked original.

```json
{
  "created_utc": "2026-11-10T07:41:04.477991+00:00",
  "python": "3.14.5",
  "platform": "macOS-26.3.2-arm64-arm-64bit-Mach-O",
  "ordinary_parameters": {
    "boosting_type": "gbdt",
    "class_weight": null,
    "colsample_bytree": 1.0,
    "importance_type": "split",
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_child_samples": 10,
    "min_child_weight": 0.001,
    "min_split_gain": 0.0,
    "n_estimators": 150,
    "n_jobs": 4,
    "num_leaves": 7,
    "objective": "binary",
    "random_state": 42,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "subsample": 1.0,
    "subsample_for_bin": 200000,
    "subsample_freq": 0,
    "scale_pos_weight": 1.0,
    "deterministic": true,
    "force_col_wise": true,
    "verbosity": -1
  },
  "weighted_parameter_rule": "same parameters, scale_pos_weight=fold-training nonictal/ictal",
  "decision_rule": "sklearn-style predict; positive score >0.5, exact tie selects class 0",
  "policy": "four frozen training-only folds; no tuning or early stopping"
}
```

### results/failure_analysis_v1/configuration.json

Source: [results/failure_analysis_v1/configuration.json](../results/failure_analysis_v1/configuration.json). Hash/provenance fields remain in the linked original.

```json
{
  "learned_transformations": null,
  "feature_periods": "remote nonictal outside +/-60s, excluding first10s; ictal=original label; post60 fully inside [offset,offset+60]",
  "score_periods": "decision end in pre60 [on-60,on), during [on,off), post60 [off,off+60), remote otherwise"
}
```

### results/temporal_filtering_v1/configuration.json

Source: [results/temporal_filtering_v1/configuration.json](../results/temporal_filtering_v1/configuration.json). Hash/provenance fields remain in the linked original.

```json
{
  "created_utc": "2026-11-10T08:07:04.765626+00:00",
  "rules": {
    "none": "output current raw decision",
    "two_consecutive": "output 1 iff current and immediately previous raw decisions are 1",
    "two_of_three": "output 1 iff at least 2 of current and up to 2 previous raw decisions are 1"
  },
  "startup": "For confirmation rules only: first decision 0; second may fire after two observed positives",
  "missing_windows": "Reject missing/incomplete/duplicate/shuffled/variable-duration windows; no imputation",
  "timing": "Current window end; no future values, backdating or within-recording state carry across gaps",
  "rearm": "Existing evaluator: one negative filtered decision closes and re-arms an episode",
  "primary": "four original training CV holdouts",
  "secondary": "original validation, descriptive only; rules unchanged",
  "decision_priorities": [
    "preserve event detection",
    "reduce false alarms",
    "minimize delay",
    "simplicity"
  ]
}
```

### results/baseline_v1/configuration.json

Source: [results/baseline_v1/configuration.json](../results/baseline_v1/configuration.json). Hash/provenance fields remain in the linked original.

```json
{
  "created_utc": "2026-10-27T07:42:02.925447+00:00",
  "data": "/Users/aaravpandya/seizure-prediction-nyas/data/processed/chb01_detection_features_11recordings_split.parquet",
  "manifest": "/Users/aaravpandya/seizure-prediction-nyas/config/chb01_recordings.csv",
  "models": {
    "dummy": {
      "constant": null,
      "random_state": null,
      "strategy": "most_frequent"
    },
    "random_forest": {
      "bootstrap": true,
      "ccp_alpha": 0.0,
      "class_weight": null,
      "criterion": "gini",
      "max_depth": null,
      "max_features": "sqrt",
      "max_leaf_nodes": null,
      "max_samples": null,
      "min_impurity_decrease": 0.0,
      "min_samples_leaf": 1,
      "min_samples_split": 2,
      "min_weight_fraction_leaf": 0.0,
      "monotonic_cst": null,
      "n_estimators": 200,
      "n_jobs": 4,
      "oob_score": false,
      "random_state": 42,
      "verbose": 0,
      "warm_start": false
    }
  },
  "python": "3.14.5",
  "platform": "macOS-26.3.2-arm64-arm-64bit-Mach-O",
  "decision_rule": "sklearn predict; binary RF score > 0.5, exact ties select class 0",
  "policy": "train only; validation evaluation only; no tuning, weighting, resampling or temporal filtering",
  "counts": {
    "train": {
      "windows": 4320,
      "class_0": 4288,
      "class_1": 32,
      "recordings": [
        "chb01_01",
        "chb01_03",
        "chb01_04",
        "chb01_14",
        "chb01_15",
        "chb01_16"
      ],
      "seizures": 4
    },
    "validation": {
      "windows": 1440,
      "class_0": 1422,
      "class_1": 18,
      "recordings": [
        "chb01_17",
        "chb01_18"
      ],
      "seizures": 1
    }
  }
}
```

### results/raw_eeg_investigation_v1/configuration.json

Source: [results/raw_eeg_investigation_v1/configuration.json](../results/raw_eeg_investigation_v1/configuration.json). Hash/provenance fields remain in the linked original.

```json
{
  "raw_files": [
    "chb01_15",
    "chb01_03",
    "chb01_04",
    "chb01_16"
  ],
  "filter": "existing causal_bandpass; prefix from t=0, zero initial state; no cleaning/notch",
  "psd": "Welch 2s Hann, 1s overlap, constant detrending; 0.5Hz bin sums times 0.5Hz; gamma[30,45], total[0.5,45]",
  "waveform_units": "MNE volts multiplied by 1e6 => µV; no trace normalization or display decimation",
  "raw_prefix_note": "earlier samples read only for correct causal state; interpretation restricted to target crops/exemplars",
  "model": "saved CV fold_3_chb01_15 only; verified predictions/evaluation, no training"
}
```

### results/rf_weighted_v1/configuration.json

Source: [results/rf_weighted_v1/configuration.json](../results/rf_weighted_v1/configuration.json). Hash/provenance fields remain in the linked original.

```json
{
  "created_utc": "2026-11-07T13:45:19.453422+00:00",
  "baseline_directory": "/Users/aaravpandya/seizure-prediction-nyas/results/baseline_v1",
  "training_identity_evidence": "Same verified dataset hash, assignments, feature order, labels, source and library versions as Phase 2B",
  "counts": {
    "train": {
      "windows": 4320,
      "class_0": 4288,
      "class_1": 32,
      "recordings": [
        "chb01_01",
        "chb01_03",
        "chb01_04",
        "chb01_14",
        "chb01_15",
        "chb01_16"
      ],
      "seizures": 4
    },
    "validation": {
      "windows": 1440,
      "class_0": 1422,
      "class_1": 18,
      "recordings": [
        "chb01_17",
        "chb01_18"
      ],
      "seizures": 1
    }
  },
  "python": "3.14.5",
  "decision_rule": "sklearn predict; binary RF score > 0.5, exact ties select class 0",
  "ordinary_parameters": {
    "bootstrap": true,
    "ccp_alpha": 0.0,
    "class_weight": null,
    "criterion": "gini",
    "max_depth": null,
    "max_features": "sqrt",
    "max_leaf_nodes": null,
    "max_samples": null,
    "min_impurity_decrease": 0.0,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "min_weight_fraction_leaf": 0.0,
    "monotonic_cst": null,
    "n_estimators": 200,
    "n_jobs": 4,
    "oob_score": false,
    "random_state": 42,
    "verbose": 0,
    "warm_start": false
  },
  "weighted_parameters": {
    "bootstrap": true,
    "ccp_alpha": 0.0,
    "class_weight": "balanced",
    "criterion": "gini",
    "max_depth": null,
    "max_features": "sqrt",
    "max_leaf_nodes": null,
    "max_samples": null,
    "min_impurity_decrease": 0.0,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "min_weight_fraction_leaf": 0.0,
    "monotonic_cst": null,
    "n_estimators": 200,
    "n_jobs": 4,
    "oob_score": false,
    "random_state": 42,
    "verbose": 0,
    "warm_start": false
  },
  "changed_parameters": [
    "class_weight"
  ],
  "balanced_weights": {
    "0": 0.503731343283582,
    "1": 67.5
  }
}
```

### results/rf_cv_v1/configuration.json

Source: [results/rf_cv_v1/configuration.json](../results/rf_cv_v1/configuration.json). Hash/provenance fields remain in the linked original.

```json
{
  "created_utc": "2026-11-07T14:05:42.761849+00:00",
  "python": "3.14.5",
  "platform": "macOS-26.3.2-arm64-arm-64bit-Mach-O",
  "training_pool": [
    "chb01_01",
    "chb01_03",
    "chb01_04",
    "chb01_14",
    "chb01_15",
    "chb01_16"
  ],
  "holdout_order": [
    "chb01_03",
    "chb01_04",
    "chb01_15",
    "chb01_16"
  ],
  "models": {
    "ordinary": {
      "bootstrap": true,
      "ccp_alpha": 0.0,
      "class_weight": null,
      "criterion": "gini",
      "max_depth": null,
      "max_features": "sqrt",
      "max_leaf_nodes": null,
      "max_samples": null,
      "min_impurity_decrease": 0.0,
      "min_samples_leaf": 1,
      "min_samples_split": 2,
      "min_weight_fraction_leaf": 0.0,
      "monotonic_cst": null,
      "n_estimators": 200,
      "n_jobs": 4,
      "oob_score": false,
      "random_state": 42,
      "verbose": 0,
      "warm_start": false
    },
    "weighted": {
      "bootstrap": true,
      "ccp_alpha": 0.0,
      "class_weight": "balanced",
      "criterion": "gini",
      "max_depth": null,
      "max_features": "sqrt",
      "max_leaf_nodes": null,
      "max_samples": null,
      "min_impurity_decrease": 0.0,
      "min_samples_leaf": 1,
      "min_samples_split": 2,
      "min_weight_fraction_leaf": 0.0,
      "monotonic_cst": null,
      "n_estimators": 200,
      "n_jobs": 4,
      "oob_score": false,
      "random_state": 42,
      "verbose": 0,
      "warm_start": false
    }
  },
  "decision_rule": "sklearn predict; score >0.5, exact ties select class 0",
  "policy": "exploratory recording-level CV on frozen training pool only; no tuning"
}
```

### results/lightgbm_v1/configuration.json

Source: [results/lightgbm_v1/configuration.json](../results/lightgbm_v1/configuration.json). Hash/provenance fields remain in the linked original.

```json
{
  "created_utc": "2026-11-10T07:12:18.707676+00:00",
  "python": "3.14.5",
  "platform": "macOS-26.3.2-arm64-arm-64bit-Mach-O",
  "data": "/Users/aaravpandya/seizure-prediction-nyas/data/processed/chb01_detection_features_11recordings_split.parquet",
  "models": {
    "ordinary_lgbm": {
      "boosting_type": "gbdt",
      "class_weight": null,
      "colsample_bytree": 1.0,
      "importance_type": "split",
      "learning_rate": 0.05,
      "max_depth": 3,
      "min_child_samples": 10,
      "min_child_weight": 0.001,
      "min_split_gain": 0.0,
      "n_estimators": 150,
      "n_jobs": 4,
      "num_leaves": 7,
      "objective": "binary",
      "random_state": 42,
      "reg_alpha": 0.0,
      "reg_lambda": 0.0,
      "subsample": 1.0,
      "subsample_for_bin": 200000,
      "subsample_freq": 0,
      "scale_pos_weight": 1.0,
      "deterministic": true,
      "force_col_wise": true,
      "verbosity": -1
    },
    "weighted_lgbm": {
      "boosting_type": "gbdt",
      "class_weight": null,
      "colsample_bytree": 1.0,
      "importance_type": "split",
      "learning_rate": 0.05,
      "max_depth": 3,
      "min_child_samples": 10,
      "min_child_weight": 0.001,
      "min_split_gain": 0.0,
      "n_estimators": 150,
      "n_jobs": 4,
      "num_leaves": 7,
      "objective": "binary",
      "random_state": 42,
      "reg_alpha": 0.0,
      "reg_lambda": 0.0,
      "subsample": 1.0,
      "subsample_for_bin": 200000,
      "subsample_freq": 0,
      "scale_pos_weight": 134.0,
      "deterministic": true,
      "force_col_wise": true,
      "verbosity": -1
    }
  },
  "changed_parameters": [
    "scale_pos_weight"
  ],
  "counts": {
    "train": {
      "windows": 4320,
      "class_0": 4288,
      "class_1": 32,
      "recordings": [
        "chb01_01",
        "chb01_03",
        "chb01_04",
        "chb01_14",
        "chb01_15",
        "chb01_16"
      ],
      "seizures": 4
    },
    "validation": {
      "windows": 1440,
      "class_0": 1422,
      "class_1": 18,
      "recordings": [
        "chb01_17",
        "chb01_18"
      ],
      "seizures": 1
    }
  },
  "fit_policy": "train only; no eval_set, callbacks, early stopping, sample weighting or resampling",
  "decision_rule": "sklearn-style predict; positive score >0.5, exact tie selects class 0"
}
```

### results/lightgbm_cv_v1/folds.json

Source: [results/lightgbm_cv_v1/folds.json](../results/lightgbm_cv_v1/folds.json)

```json
[
  {
    "fold": 1,
    "held_out": "chb01_03",
    "train_recordings": [
      "chb01_01",
      "chb01_04",
      "chb01_14",
      "chb01_15",
      "chb01_16"
    ],
    "training_windows": 3600,
    "training_ictal": 24,
    "evaluated_windows": 720,
    "evaluated_ictal": 8,
    "training_inputs_sha256": "628bd602416a6ca3fd1defb37e70ffb65c3cb3410a05a7770f8b3a528deff3b1",
    "training_class_0": 3576,
    "scale_pos_weight": 149.0,
    "models": {
      "ordinary_lgbm": {
        "boosting_type": "gbdt",
        "class_weight": null,
        "colsample_bytree": 1.0,
        "importance_type": "split",
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_child_samples": 10,
        "min_child_weight": 0.001,
        "min_split_gain": 0.0,
        "n_estimators": 150,
        "n_jobs": 4,
        "num_leaves": 7,
        "objective": "binary",
        "random_state": 42,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "subsample": 1.0,
        "subsample_for_bin": 200000,
        "subsample_freq": 0,
        "scale_pos_weight": 1.0,
        "deterministic": true,
        "force_col_wise": true,
        "verbosity": -1
      },
      "weighted_lgbm": {
        "boosting_type": "gbdt",
        "class_weight": null,
        "colsample_bytree": 1.0,
        "importance_type": "split",
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_child_samples": 10,
        "min_child_weight": 0.001,
        "min_split_gain": 0.0,
        "n_estimators": 150,
        "n_jobs": 4,
        "num_leaves": 7,
        "objective": "binary",
        "random_state": 42,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "subsample": 1.0,
        "subsample_for_bin": 200000,
        "subsample_freq": 0,
        "scale_pos_weight": 149.0,
        "deterministic": true,
        "force_col_wise": true,
        "verbosity": -1
      }
    }
  },
  {
    "fold": 2,
    "held_out": "chb01_04",
    "train_recordings": [
      "chb01_01",
      "chb01_03",
      "chb01_14",
      "chb01_15",
      "chb01_16"
    ],
    "training_windows": 3600,
    "training_ictal": 26,
    "evaluated_windows": 720,
    "evaluated_ictal": 6,
    "training_inputs_sha256": "e6ea246ec98d3d69002bbbe2b43f25b91af70255e6fa99e06b3ade98f63dba90",
    "training_class_0": 3574,
    "scale_pos_weight": 137.46153846153845,
    "models": {
      "ordinary_lgbm": {
        "boosting_type": "gbdt",
        "class_weight": null,
        "colsample_bytree": 1.0,
        "importance_type": "split",
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_child_samples": 10,
        "min_child_weight": 0.001,
        "min_split_gain": 0.0,
        "n_estimators": 150,
        "n_jobs": 4,
        "num_leaves": 7,
        "objective": "binary",
        "random_state": 42,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "subsample": 1.0,
        "subsample_for_bin": 200000,
        "subsample_freq": 0,
        "scale_pos_weight": 1.0,
        "deterministic": true,
        "force_col_wise": true,
        "verbosity": -1
      },
      "weighted_lgbm": {
        "boosting_type": "gbdt",
        "class_weight": null,
        "colsample_bytree": 1.0,
        "importance_type": "split",
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_child_samples": 10,
        "min_child_weight": 0.001,
        "min_split_gain": 0.0,
        "n_estimators": 150,
        "n_jobs": 4,
        "num_leaves": 7,
        "objective": "binary",
        "random_state": 42,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "subsample": 1.0,
        "subsample_for_bin": 200000,
        "subsample_freq": 0,
        "scale_pos_weight": 137.46153846153845,
        "deterministic": true,
        "force_col_wise": true,
        "verbosity": -1
      }
    }
  },
  {
    "fold": 3,
    "held_out": "chb01_15",
    "train_recordings": [
      "chb01_01",
      "chb01_03",
      "chb01_04",
      "chb01_14",
      "chb01_16"
    ],
    "training_windows": 3600,
    "training_ictal": 24,
    "evaluated_windows": 720,
    "evaluated_ictal": 8,
    "training_inputs_sha256": "87d167c0e95e4d36e1ed2f56ec5a1319340e48e583284eaecbfd8c229ce4212e",
    "training_class_0": 3576,
    "scale_pos_weight": 149.0,
    "models": {
      "ordinary_lgbm": {
        "boosting_type": "gbdt",
        "class_weight": null,
        "colsample_bytree": 1.0,
        "importance_type": "split",
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_child_samples": 10,
        "min_child_weight": 0.001,
        "min_split_gain": 0.0,
        "n_estimators": 150,
        "n_jobs": 4,
        "num_leaves": 7,
        "objective": "binary",
        "random_state": 42,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "subsample": 1.0,
        "subsample_for_bin": 200000,
        "subsample_freq": 0,
        "scale_pos_weight": 1.0,
        "deterministic": true,
        "force_col_wise": true,
        "verbosity": -1
      },
      "weighted_lgbm": {
        "boosting_type": "gbdt",
        "class_weight": null,
        "colsample_bytree": 1.0,
        "importance_type": "split",
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_child_samples": 10,
        "min_child_weight": 0.001,
        "min_split_gain": 0.0,
        "n_estimators": 150,
        "n_jobs": 4,
        "num_leaves": 7,
        "objective": "binary",
        "random_state": 42,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "subsample": 1.0,
        "subsample_for_bin": 200000,
        "subsample_freq": 0,
        "scale_pos_weight": 149.0,
        "deterministic": true,
        "force_col_wise": true,
        "verbosity": -1
      }
    }
  },
  {
    "fold": 4,
    "held_out": "chb01_16",
    "train_recordings": [
      "chb01_01",
      "chb01_03",
      "chb01_04",
      "chb01_14",
      "chb01_15"
    ],
    "training_windows": 3600,
    "training_ictal": 22,
    "evaluated_windows": 720,
    "evaluated_ictal": 10,
    "training_inputs_sha256": "bf473b60c7f31ed54ce3ad721df833463eb419cddbf37100acb6f15663c62f2f",
    "training_class_0": 3578,
    "scale_pos_weight": 162.63636363636363,
    "models": {
      "ordinary_lgbm": {
        "boosting_type": "gbdt",
        "class_weight": null,
        "colsample_bytree": 1.0,
        "importance_type": "split",
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_child_samples": 10,
        "min_child_weight": 0.001,
        "min_split_gain": 0.0,
        "n_estimators": 150,
        "n_jobs": 4,
        "num_leaves": 7,
        "objective": "binary",
        "random_state": 42,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "subsample": 1.0,
        "subsample_for_bin": 200000,
        "subsample_freq": 0,
        "scale_pos_weight": 1.0,
        "deterministic": true,
        "force_col_wise": true,
        "verbosity": -1
      },
      "weighted_lgbm": {
        "boosting_type": "gbdt",
        "class_weight": null,
        "colsample_bytree": 1.0,
        "importance_type": "split",
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_child_samples": 10,
        "min_child_weight": 0.001,
        "min_split_gain": 0.0,
        "n_estimators": 150,
        "n_jobs": 4,
        "num_leaves": 7,
        "objective": "binary",
        "random_state": 42,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "subsample": 1.0,
        "subsample_for_bin": 200000,
        "subsample_freq": 0,
        "scale_pos_weight": 162.63636363636363,
        "deterministic": true,
        "force_col_wise": true,
        "verbosity": -1
      }
    }
  }
]
```

### results/rf_cv_v1/folds.json

Source: [results/rf_cv_v1/folds.json](../results/rf_cv_v1/folds.json)

```json
[
  {
    "fold": 1,
    "held_out": "chb01_03",
    "train_recordings": [
      "chb01_01",
      "chb01_04",
      "chb01_14",
      "chb01_15",
      "chb01_16"
    ],
    "training_windows": 3600,
    "training_ictal": 24,
    "evaluated_windows": 720,
    "evaluated_ictal": 8,
    "training_inputs_sha256": "628bd602416a6ca3fd1defb37e70ffb65c3cb3410a05a7770f8b3a528deff3b1",
    "balanced_weights": {
      "0": 0.5033557046979866,
      "1": 75.0
    }
  },
  {
    "fold": 2,
    "held_out": "chb01_04",
    "train_recordings": [
      "chb01_01",
      "chb01_03",
      "chb01_14",
      "chb01_15",
      "chb01_16"
    ],
    "training_windows": 3600,
    "training_ictal": 26,
    "evaluated_windows": 720,
    "evaluated_ictal": 6,
    "training_inputs_sha256": "e6ea246ec98d3d69002bbbe2b43f25b91af70255e6fa99e06b3ade98f63dba90",
    "balanced_weights": {
      "0": 0.5036373810856184,
      "1": 69.23076923076923
    }
  },
  {
    "fold": 3,
    "held_out": "chb01_15",
    "train_recordings": [
      "chb01_01",
      "chb01_03",
      "chb01_04",
      "chb01_14",
      "chb01_16"
    ],
    "training_windows": 3600,
    "training_ictal": 24,
    "evaluated_windows": 720,
    "evaluated_ictal": 8,
    "training_inputs_sha256": "87d167c0e95e4d36e1ed2f56ec5a1319340e48e583284eaecbfd8c229ce4212e",
    "balanced_weights": {
      "0": 0.5033557046979866,
      "1": 75.0
    }
  },
  {
    "fold": 4,
    "held_out": "chb01_16",
    "train_recordings": [
      "chb01_01",
      "chb01_03",
      "chb01_04",
      "chb01_14",
      "chb01_15"
    ],
    "training_windows": 3600,
    "training_ictal": 22,
    "evaluated_windows": 720,
    "evaluated_ictal": 10,
    "training_inputs_sha256": "bf473b60c7f31ed54ce3ad721df833463eb419cddbf37100acb6f15663c62f2f",
    "balanced_weights": {
      "0": 0.5030743432084964,
      "1": 81.81818181818181
    }
  }
]
```

## Appendix B. Complete saved evaluation result ledger

Each block cites its file and JSON selector. Reused references are intentionally retained with their provenance. Ratios below use 12 significant digits for readability; original JSON is authoritative at full stored precision. Seizure/episode ledgers retain all original fields. No evaluator was called to make these tables.

### results/final_test_v1/primary_evaluation.json — root

Source: [results/final_test_v1/primary_evaluation.json](../results/final_test_v1/primary_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1905 | 40 | 0.995275590551 | 1 | 0.775 | 1 | 0.87323943662 | 0.947176280995 | [[1865,0],[9,31]] |

| chb01_21 | 720 | 19 | 0.991666666667 | 1 | 0.684210526316 | 1 | 0.8125 | 0.9498264893 | [[701,0],[6,13]] |

| chb01_22 | 720 | 0 | 1 | undefined | undefined | 1 | undefined | undefined | [[720,0],[0,0]] |

| chb01_26 | 465 | 21 | 0.993548387097 | 1 | 0.857142857143 | 1 | 0.923076923077 | 0.965037696785 | [[444,0],[3,18]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 2 | 2 | 0 | 1 | 0 | 2.64583333333 | 0 | 1 | [8.0,13.0] | 10.5 | 10.5 |

| chb01_21 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | [8.0] | 8 | 8 |

| chb01_22 | 0 | 0 | 0 | undefined | 0 | 1 | 0 | 0 | [] | undefined | undefined |

| chb01_26 | 1 | 1 | 0 | 1 | 0 | 0.645833333333 | 0 | 0 | [13.0] | 13 | 13 |



`pooled` event/alarm details and undefined reasons:

**undefined_metrics**

`{}`


`chb01_21` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 327.0 | 420.0 | true | 1 | 8.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 335.0 | 12 | 390.0 | 395.0 | false | "detection" | 1 |
| 2 | 400.0 | 1 | 400.0 | 405.0 | false | "repeat" | 1 |

**undefined_metrics**

`{}`


`chb01_22` event/alarm details and undefined reasons:

**seizures**

`[]`

**alarms**

`[]`

**undefined_metrics**

`{"event_sensitivity": "no annotated seizures", "mean_latency_sec": "no detected seizures", "median_latency_sec": "no detected seizures"}`

**undefined_window_metrics**

`{"precision": "zero denominator", "recall": "zero denominator", "f1": "zero denominator", "average_precision": "no positive ground-truth windows"}`


`chb01_26` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1862.0 | 1963.0 | true | 1 | 13.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1875.0 | 18 | 1960.0 | 1965.0 | false | "detection" | 1 |

**undefined_metrics**

`{}`


### results/final_test_v1/reference_evaluation.json — root

Source: [results/final_test_v1/reference_evaluation.json](../results/final_test_v1/reference_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1905 | 40 | 0.995275590551 | 1 | 0.775 | 1 | 0.87323943662 | 0.942248598152 | [[1865,0],[9,31]] |

| chb01_21 | 720 | 19 | 0.993055555556 | 1 | 0.736842105263 | 1 | 0.848484848485 | 0.929145546342 | [[701,0],[5,14]] |

| chb01_22 | 720 | 0 | 1 | undefined | undefined | 1 | undefined | undefined | [[720,0],[0,0]] |

| chb01_26 | 465 | 21 | 0.991397849462 | 1 | 0.809523809524 | 1 | 0.894736842105 | 0.959788359788 | [[444,0],[4,17]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 2 | 2 | 0 | 1 | 0 | 2.64583333333 | 0 | 3 | [8.0,13.0] | 10.5 | 10.5 |

| chb01_21 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 2 | [8.0] | 8 | 8 |

| chb01_22 | 0 | 0 | 0 | undefined | 0 | 1 | 0 | 0 | [] | undefined | undefined |

| chb01_26 | 1 | 1 | 0 | 1 | 0 | 0.645833333333 | 0 | 1 | [13.0] | 13 | 13 |



`pooled` event/alarm details and undefined reasons:

**undefined_metrics**

`{}`


`chb01_21` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 327.0 | 420.0 | true | 1 | 8.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 335.0 | 10 | 380.0 | 385.0 | false | "detection" | 1 |
| 2 | 390.0 | 3 | 400.0 | 405.0 | false | "repeat" | 1 |
| 3 | 410.0 | 1 | 410.0 | 415.0 | false | "repeat" | 1 |

**undefined_metrics**

`{}`


`chb01_22` event/alarm details and undefined reasons:

**seizures**

`[]`

**alarms**

`[]`

**undefined_metrics**

`{"event_sensitivity": "no annotated seizures", "mean_latency_sec": "no detected seizures", "median_latency_sec": "no detected seizures"}`

**undefined_window_metrics**

`{"precision": "zero denominator", "recall": "zero denominator", "f1": "zero denominator", "average_precision": "no positive ground-truth windows"}`


`chb01_26` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1862.0 | 1963.0 | true | 1 | 13.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1875.0 | 1 | 1875.0 | 1880.0 | false | "detection" | 1 |
| 2 | 1885.0 | 16 | 1960.0 | 1965.0 | false | "repeat" | 1 |

**undefined_metrics**

`{}`


### results/lightgbm_cv_v1/summary.json — root.ordinary_rf

Source: [results/lightgbm_cv_v1/summary.json](../results/lightgbm_cv_v1/summary.json) → `root.ordinary_rf`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 2880 | 32 | 0.994444444444 | 0.944444444444 | 0.53125 | 0.999648876404 | 0.68 | 0.844954325253 | [[2847,1],[15,17]] |

| chb01_03 | 720 | 8 | 0.997222222222 | 1 | 0.75 | 1 | 0.857142857143 | 0.860763888889 | [[712,0],[2,6]] |

| chb01_04 | 720 | 6 | 0.997222222222 | 1 | 0.666666666667 | 1 | 0.8 | 1 | [[714,0],[2,4]] |

| chb01_15 | 720 | 8 | 0.9875 | 0 | 0 | 0.998595505618 | 0 | 0.440566378066 | [[711,1],[8,0]] |

| chb01_16 | 720 | 10 | 0.995833333333 | 1 | 0.7 | 1 | 0.823529411765 | 0.966923076923 | [[710,0],[3,7]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 4 | 3 | 1 | 0.75 | 1 | 4 | 0.25 | 2 | [9.0,8.0,5.0] | 7.33333333333 | 8 |

| chb01_03 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [9.0] | 9 | 9 |

| chb01_04 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | [8.0] | 8 | 8 |

| chb01_15 | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | [] | undefined | undefined |

| chb01_16 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | [5.0] | 5 | 5 |



`chb01_03` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 2996.0 | 3036.0 | true | 1 | 9.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3005.0 | 6 | 3030.0 | 3035.0 | false | "detection" | 1 |


`chb01_04` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1467.0 | 1494.0 | true | 1 | 8.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1475.0 | 1 | 1475.0 | 1480.0 | false | "detection" | 1 |
| 2 | 1485.0 | 3 | 1495.0 | 1500.0 | false | "repeat" | 1 |


`chb01_15` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1732.0 | 1772.0 | false | null | null |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1785.0 | 1 | 1785.0 | 1790.0 | false | "false_alarm" | null |


`chb01_16` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1015.0 | 1066.0 | true | 1 | 5.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1020.0 | 2 | 1025.0 | 1030.0 | false | "detection" | 1 |
| 2 | 1035.0 | 5 | 1055.0 | 1060.0 | false | "repeat" | 1 |


### results/lightgbm_cv_v1/summary.json — root.weighted_rf

Source: [results/lightgbm_cv_v1/summary.json](../results/lightgbm_cv_v1/summary.json) → `root.weighted_rf`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 2880 | 32 | 0.99375 | 0.791666666667 | 0.59375 | 0.998244382022 | 0.678571428571 | 0.77372755841 | [[2843,5],[13,19]] |

| chb01_03 | 720 | 8 | 0.997222222222 | 0.875 | 0.875 | 0.998595505618 | 0.875 | 0.876388888889 | [[711,1],[1,7]] |

| chb01_04 | 720 | 6 | 0.995833333333 | 0.8 | 0.666666666667 | 0.998599439776 | 0.727272727273 | 0.930555555556 | [[713,1],[2,4]] |

| chb01_15 | 720 | 8 | 0.9875 | 0 | 0 | 0.998595505618 | 0 | 0.271631000578 | [[711,1],[8,0]] |

| chb01_16 | 720 | 10 | 0.994444444444 | 0.8 | 0.8 | 0.997183098592 | 0.8 | 0.965151515152 | [[708,2],[2,8]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 4 | 3 | 1 | 0.75 | 4 | 4 | 1 | 2 | [9.0,8.0,5.0] | 7.33333333333 | 8 |

| chb01_03 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 0 | [9.0] | 9 | 9 |

| chb01_04 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | [8.0] | 8 | 8 |

| chb01_15 | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | [] | undefined | undefined |

| chb01_16 | 1 | 1 | 0 | 1 | 2 | 1 | 2 | 1 | [5.0] | 5 | 5 |



`chb01_03` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 2996.0 | 3036.0 | true | 1 | 9.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3005.0 | 7 | 3035.0 | 3040.0 | false | "detection" | 1 |
| 2 | 3050.0 | 1 | 3050.0 | 3055.0 | false | "false_alarm" | null |


`chb01_04` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1467.0 | 1494.0 | true | 1 | 8.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1475.0 | 2 | 1480.0 | 1485.0 | false | "detection" | 1 |
| 2 | 1490.0 | 3 | 1500.0 | 1505.0 | false | "repeat" | 1 |


`chb01_15` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1732.0 | 1772.0 | false | null | null |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1785.0 | 1 | 1785.0 | 1790.0 | false | "false_alarm" | null |


`chb01_16` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1015.0 | 1066.0 | true | 3 | 5.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 870.0 | 1 | 870.0 | 875.0 | false | "false_alarm" | null |
| 2 | 1010.0 | 1 | 1010.0 | 1015.0 | false | "false_alarm" | null |
| 3 | 1020.0 | 2 | 1025.0 | 1030.0 | false | "detection" | 1 |
| 4 | 1035.0 | 6 | 1060.0 | 1065.0 | false | "repeat" | 1 |


### results/lightgbm_cv_v1/summary.json — root.ordinary_lgbm

Source: [results/lightgbm_cv_v1/summary.json](../results/lightgbm_cv_v1/summary.json) → `root.ordinary_lgbm`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 2880 | 32 | 0.995138888889 | 0.875 | 0.65625 | 0.998946629213 | 0.75 | 0.867054072542 | [[2845,3],[11,21]] |

| chb01_03 | 720 | 8 | 0.997222222222 | 1 | 0.75 | 1 | 0.857142857143 | 0.838974926254 | [[712,0],[2,6]] |

| chb01_04 | 720 | 6 | 0.997222222222 | 0.833333333333 | 0.833333333333 | 0.998599439776 | 0.833333333333 | 0.958333333333 | [[713,1],[1,5]] |

| chb01_15 | 720 | 8 | 0.991666666667 | 0.75 | 0.375 | 0.998595505618 | 0.5 | 0.649176864802 | [[711,1],[5,3]] |

| chb01_16 | 720 | 10 | 0.994444444444 | 0.875 | 0.7 | 0.998591549296 | 0.777777777778 | 0.926241258741 | [[709,1],[3,7]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 4 | 4 | 0 | 1 | 2 | 4 | 0.5 | 2 | [9.0,8.0,8.0,5.0] | 7.5 | 8 |

| chb01_03 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [9.0] | 9 | 9 |

| chb01_04 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [8.0] | 8 | 8 |

| chb01_15 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | [8.0] | 8 | 8 |

| chb01_16 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | [5.0] | 5 | 5 |



`chb01_03` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 2996.0 | 3036.0 | true | 1 | 9.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3005.0 | 6 | 3030.0 | 3035.0 | false | "detection" | 1 |


`chb01_04` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1467.0 | 1494.0 | true | 1 | 8.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1475.0 | 6 | 1500.0 | 1505.0 | false | "detection" | 1 |


`chb01_15` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1732.0 | 1772.0 | true | 1 | 8.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1740.0 | 1 | 1740.0 | 1745.0 | false | "detection" | 1 |
| 2 | 1765.0 | 2 | 1770.0 | 1775.0 | false | "repeat" | 1 |
| 3 | 1785.0 | 1 | 1785.0 | 1790.0 | false | "false_alarm" | null |


`chb01_16` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1015.0 | 1066.0 | true | 2 | 5.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 870.0 | 1 | 870.0 | 875.0 | false | "false_alarm" | null |
| 2 | 1020.0 | 2 | 1025.0 | 1030.0 | false | "detection" | 1 |
| 3 | 1035.0 | 5 | 1055.0 | 1060.0 | false | "repeat" | 1 |


### results/lightgbm_cv_v1/summary.json — root.weighted_lgbm

Source: [results/lightgbm_cv_v1/summary.json](../results/lightgbm_cv_v1/summary.json) → `root.weighted_lgbm`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 2880 | 32 | 0.994097222222 | 0.8 | 0.625 | 0.998244382022 | 0.701754385965 | 0.751963599621 | [[2843,5],[12,20]] |

| chb01_03 | 720 | 8 | 0.995833333333 | 0.777777777778 | 0.875 | 0.997191011236 | 0.823529411765 | 0.766044829768 | [[710,2],[1,7]] |

| chb01_04 | 720 | 6 | 0.997222222222 | 1 | 0.666666666667 | 1 | 0.8 | 0.910714285714 | [[714,0],[2,4]] |

| chb01_15 | 720 | 8 | 0.9875 | 0 | 0 | 0.998595505618 | 0 | 0.265673144258 | [[711,1],[8,0]] |

| chb01_16 | 720 | 10 | 0.995833333333 | 0.818181818182 | 0.9 | 0.997183098592 | 0.857142857143 | 0.937373737374 | [[708,2],[1,9]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 4 | 3 | 1 | 0.75 | 4 | 4 | 1 | 0 | [9.0,8.0,5.0] | 7.33333333333 | 8 |

| chb01_03 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 0 | [9.0] | 9 | 9 |

| chb01_04 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [8.0] | 8 | 8 |

| chb01_15 | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | [] | undefined | undefined |

| chb01_16 | 1 | 1 | 0 | 1 | 2 | 1 | 2 | 0 | [5.0] | 5 | 5 |



`chb01_03` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 2996.0 | 3036.0 | true | 1 | 9.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3005.0 | 8 | 3040.0 | 3045.0 | false | "detection" | 1 |
| 2 | 3050.0 | 1 | 3050.0 | 3055.0 | false | "false_alarm" | null |


`chb01_04` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1467.0 | 1494.0 | true | 1 | 8.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1475.0 | 4 | 1490.0 | 1495.0 | false | "detection" | 1 |


`chb01_15` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1732.0 | 1772.0 | false | null | null |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1785.0 | 1 | 1785.0 | 1790.0 | false | "false_alarm" | null |


`chb01_16` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1015.0 | 1066.0 | true | 3 | 5.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 870.0 | 1 | 870.0 | 875.0 | false | "false_alarm" | null |
| 2 | 1010.0 | 1 | 1010.0 | 1015.0 | false | "false_alarm" | null |
| 3 | 1020.0 | 9 | 1060.0 | 1065.0 | false | "detection" | 1 |


### results/baseline_v1/dummy_evaluation.json — root

Source: [results/baseline_v1/dummy_evaluation.json](../results/baseline_v1/dummy_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1440 | 18 | 0.9875 | undefined | 0 | 1 | 0 | 0.0125 | [[1422,0],[18,0]] |

| chb01_17 | 720 | 0 | 1 | undefined | undefined | 1 | undefined | undefined | [[720,0],[0,0]] |

| chb01_18 | 720 | 18 | 0.975 | undefined | 0 | 1 | 0 | 0.025 | [[702,0],[18,0]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 0 | 1 | 0 | 0 | 2 | 0 | 0 | [] | undefined | undefined |

| chb01_17 | 0 | 0 | 0 | undefined | 0 | 1 | 0 | 0 | [] | undefined | undefined |

| chb01_18 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | [] | undefined | undefined |



`pooled` event/alarm details and undefined reasons:

**undefined_window_metrics**

`{"precision": "zero denominator"}`


`chb01_17` event/alarm details and undefined reasons:

**seizures**

`[]`

**alarms**

`[]`

**undefined_window_metrics**

`{"precision": "zero denominator", "recall": "zero denominator", "f1": "zero denominator", "average_precision": "no positive ground-truth windows"}`


`chb01_18` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1720.0 | 1810.0 | false | null | null |

**alarms**

`[]`

**undefined_window_metrics**

`{"precision": "zero denominator"}`


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_17",
    "label": 0,
    "windows": 720,
    "predicted_positive": 0,
    "mean": 0.0,
    "min": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 0.0,
    "q95": 0.0,
    "max": 0.0
  },
  {
    "recording_id": "chb01_18",
    "label": 0,
    "windows": 702,
    "predicted_positive": 0,
    "mean": 0.0,
    "min": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 0.0,
    "q95": 0.0,
    "max": 0.0
  },
  {
    "recording_id": "chb01_18",
    "label": 1,
    "windows": 18,
    "predicted_positive": 0,
    "mean": 0.0,
    "min": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 0.0,
    "q95": 0.0,
    "max": 0.0
  }
]
```

### results/baseline_v1/random_forest_evaluation.json — root

Source: [results/baseline_v1/random_forest_evaluation.json](../results/baseline_v1/random_forest_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1440 | 18 | 0.995138888889 | 1 | 0.611111111111 | 1 | 0.758620689655 | 0.982947564991 | [[1422,0],[7,11]] |

| chb01_17 | 720 | 0 | 1 | undefined | undefined | 1 | undefined | undefined | [[720,0],[0,0]] |

| chb01_18 | 720 | 18 | 0.990277777778 | 1 | 0.611111111111 | 1 | 0.758620689655 | 1 | [[702,0],[7,11]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 0 | 2 | 0 | 2 | [10.0] | 10 | 10 |

| chb01_17 | 0 | 0 | 0 | undefined | 0 | 1 | 0 | 0 | [] | undefined | undefined |

| chb01_18 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 2 | [10.0] | 10 | 10 |



`chb01_17` event/alarm details and undefined reasons:

**seizures**

`[]`

**alarms**

`[]`

**undefined_window_metrics**

`{"precision": "zero denominator", "recall": "zero denominator", "f1": "zero denominator", "average_precision": "no positive ground-truth windows"}`


`chb01_18` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1720.0 | 1810.0 | true | 1 | 10.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1730.0 | 1 | 1730.0 | 1735.0 | false | "detection" | 1 |
| 2 | 1740.0 | 9 | 1780.0 | 1785.0 | false | "repeat" | 1 |
| 3 | 1795.0 | 1 | 1795.0 | 1800.0 | false | "repeat" | 1 |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_17",
    "label": 0,
    "windows": 720,
    "predicted_positive": 0,
    "mean": 0.0057083333333333335,
    "min": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 0.0,
    "q95": 0.03,
    "max": 0.34
  },
  {
    "recording_id": "chb01_18",
    "label": 0,
    "windows": 702,
    "predicted_positive": 0,
    "mean": 0.001168091168091168,
    "min": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 0.0,
    "q95": 0.005,
    "max": 0.16
  },
  {
    "recording_id": "chb01_18",
    "label": 1,
    "windows": 18,
    "predicted_positive": 11,
    "mean": 0.5833333333333334,
    "min": 0.215,
    "q25": 0.3625,
    "median": 0.565,
    "q75": 0.7887500000000001,
    "q95": 0.9452499999999999,
    "max": 0.975
  }
]
```

### results/rf_weighted_v1/ordinary_evaluation.json — root

Source: [results/rf_weighted_v1/ordinary_evaluation.json](../results/rf_weighted_v1/ordinary_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1440 | 18 | 0.995138888889 | 1 | 0.611111111111 | 1 | 0.758620689655 | 0.982947564991 | [[1422,0],[7,11]] |

| chb01_17 | 720 | 0 | 1 | undefined | undefined | 1 | undefined | undefined | [[720,0],[0,0]] |

| chb01_18 | 720 | 18 | 0.990277777778 | 1 | 0.611111111111 | 1 | 0.758620689655 | 1 | [[702,0],[7,11]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 0 | 2 | 0 | 2 | [10.0] | 10 | 10 |

| chb01_17 | 0 | 0 | 0 | undefined | 0 | 1 | 0 | 0 | [] | undefined | undefined |

| chb01_18 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 2 | [10.0] | 10 | 10 |



`chb01_17` event/alarm details and undefined reasons:

**seizures**

`[]`

**alarms**

`[]`

**undefined_window_metrics**

`{"precision": "zero denominator", "recall": "zero denominator", "f1": "zero denominator", "average_precision": "no positive ground-truth windows"}`


`chb01_18` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1720.0 | 1810.0 | true | 1 | 10.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1730.0 | 1 | 1730.0 | 1735.0 | false | "detection" | 1 |
| 2 | 1740.0 | 9 | 1780.0 | 1785.0 | false | "repeat" | 1 |
| 3 | 1795.0 | 1 | 1795.0 | 1800.0 | false | "repeat" | 1 |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_17",
    "label": 0,
    "windows": 720,
    "predicted_positive": 0,
    "mean": 0.0057083333333333335,
    "min": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 0.0,
    "q95": 0.03,
    "max": 0.34
  },
  {
    "recording_id": "chb01_18",
    "label": 0,
    "windows": 702,
    "predicted_positive": 0,
    "mean": 0.001168091168091168,
    "min": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 0.0,
    "q95": 0.005,
    "max": 0.16
  },
  {
    "recording_id": "chb01_18",
    "label": 1,
    "windows": 18,
    "predicted_positive": 11,
    "mean": 0.5833333333333334,
    "min": 0.215,
    "q25": 0.3625,
    "median": 0.565,
    "q75": 0.7887500000000001,
    "q95": 0.9452499999999999,
    "max": 0.975
  }
]
```

### results/rf_weighted_v1/weighted_evaluation.json — root

Source: [results/rf_weighted_v1/weighted_evaluation.json](../results/rf_weighted_v1/weighted_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1440 | 18 | 0.99375 | 1 | 0.5 | 1 | 0.666666666667 | 0.94043796856 | [[1422,0],[9,9]] |

| chb01_17 | 720 | 0 | 1 | undefined | undefined | 1 | undefined | undefined | [[720,0],[0,0]] |

| chb01_18 | 720 | 18 | 0.9875 | 1 | 0.5 | 1 | 0.666666666667 | 0.96164485564 | [[702,0],[9,9]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 0 | 2 | 0 | 1 | [5.0] | 5 | 5 |

| chb01_17 | 0 | 0 | 0 | undefined | 0 | 1 | 0 | 0 | [] | undefined | undefined |

| chb01_18 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | [5.0] | 5 | 5 |



`chb01_17` event/alarm details and undefined reasons:

**seizures**

`[]`

**alarms**

`[]`

**undefined_window_metrics**

`{"precision": "zero denominator", "recall": "zero denominator", "f1": "zero denominator", "average_precision": "no positive ground-truth windows"}`


`chb01_18` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1720.0 | 1810.0 | true | 1 | 5.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1725.0 | 2 | 1730.0 | 1735.0 | false | "detection" | 1 |
| 2 | 1740.0 | 7 | 1770.0 | 1775.0 | false | "repeat" | 1 |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_17",
    "label": 0,
    "windows": 720,
    "predicted_positive": 0,
    "mean": 0.0005486111111111112,
    "min": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 0.0,
    "q95": 0.0,
    "max": 0.15
  },
  {
    "recording_id": "chb01_18",
    "label": 0,
    "windows": 702,
    "predicted_positive": 0,
    "mean": 0.0007977207977207978,
    "min": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 0.0,
    "q95": 0.005,
    "max": 0.085
  },
  {
    "recording_id": "chb01_18",
    "label": 1,
    "windows": 18,
    "predicted_positive": 9,
    "mean": 0.5238888888888888,
    "min": 0.01,
    "q25": 0.11875,
    "median": 0.5025,
    "q75": 0.93375,
    "q95": 0.99,
    "max": 0.99
  }
]
```

### results/rf_cv_v1/summary.json — root.ordinary

Source: [results/rf_cv_v1/summary.json](../results/rf_cv_v1/summary.json) → `root.ordinary`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 2880 | 32 | 0.994444444444 | 0.944444444444 | 0.53125 | 0.999648876404 | 0.68 | 0.844954325253 | [[2847,1],[15,17]] |

| chb01_03 | 720 | 8 | 0.997222222222 | 1 | 0.75 | 1 | 0.857142857143 | 0.860763888889 | [[712,0],[2,6]] |

| chb01_04 | 720 | 6 | 0.997222222222 | 1 | 0.666666666667 | 1 | 0.8 | 1 | [[714,0],[2,4]] |

| chb01_15 | 720 | 8 | 0.9875 | 0 | 0 | 0.998595505618 | 0 | 0.440566378066 | [[711,1],[8,0]] |

| chb01_16 | 720 | 10 | 0.995833333333 | 1 | 0.7 | 1 | 0.823529411765 | 0.966923076923 | [[710,0],[3,7]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 4 | 3 | 1 | 0.75 | 1 | 4 | 0.25 | 2 | [9.0,8.0,5.0] | 7.33333333333 | 8 |

| chb01_03 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [9.0] | 9 | 9 |

| chb01_04 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | [8.0] | 8 | 8 |

| chb01_15 | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | [] | undefined | undefined |

| chb01_16 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | [5.0] | 5 | 5 |



`chb01_03` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 2996.0 | 3036.0 | true | 1 | 9.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3005.0 | 6 | 3030.0 | 3035.0 | false | "detection" | 1 |


`chb01_04` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1467.0 | 1494.0 | true | 1 | 8.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1475.0 | 1 | 1475.0 | 1480.0 | false | "detection" | 1 |
| 2 | 1485.0 | 3 | 1495.0 | 1500.0 | false | "repeat" | 1 |


`chb01_15` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1732.0 | 1772.0 | false | null | null |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1785.0 | 1 | 1785.0 | 1790.0 | false | "false_alarm" | null |


`chb01_16` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1015.0 | 1066.0 | true | 1 | 5.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1020.0 | 2 | 1025.0 | 1030.0 | false | "detection" | 1 |
| 2 | 1035.0 | 5 | 1055.0 | 1060.0 | false | "repeat" | 1 |


### results/rf_cv_v1/summary.json — root.weighted

Source: [results/rf_cv_v1/summary.json](../results/rf_cv_v1/summary.json) → `root.weighted`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 2880 | 32 | 0.99375 | 0.791666666667 | 0.59375 | 0.998244382022 | 0.678571428571 | 0.77372755841 | [[2843,5],[13,19]] |

| chb01_03 | 720 | 8 | 0.997222222222 | 0.875 | 0.875 | 0.998595505618 | 0.875 | 0.876388888889 | [[711,1],[1,7]] |

| chb01_04 | 720 | 6 | 0.995833333333 | 0.8 | 0.666666666667 | 0.998599439776 | 0.727272727273 | 0.930555555556 | [[713,1],[2,4]] |

| chb01_15 | 720 | 8 | 0.9875 | 0 | 0 | 0.998595505618 | 0 | 0.271631000578 | [[711,1],[8,0]] |

| chb01_16 | 720 | 10 | 0.994444444444 | 0.8 | 0.8 | 0.997183098592 | 0.8 | 0.965151515152 | [[708,2],[2,8]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 4 | 3 | 1 | 0.75 | 4 | 4 | 1 | 2 | [9.0,8.0,5.0] | 7.33333333333 | 8 |

| chb01_03 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 0 | [9.0] | 9 | 9 |

| chb01_04 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | [8.0] | 8 | 8 |

| chb01_15 | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | [] | undefined | undefined |

| chb01_16 | 1 | 1 | 0 | 1 | 2 | 1 | 2 | 1 | [5.0] | 5 | 5 |



`chb01_03` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 2996.0 | 3036.0 | true | 1 | 9.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3005.0 | 7 | 3035.0 | 3040.0 | false | "detection" | 1 |
| 2 | 3050.0 | 1 | 3050.0 | 3055.0 | false | "false_alarm" | null |


`chb01_04` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1467.0 | 1494.0 | true | 1 | 8.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1475.0 | 2 | 1480.0 | 1485.0 | false | "detection" | 1 |
| 2 | 1490.0 | 3 | 1500.0 | 1505.0 | false | "repeat" | 1 |


`chb01_15` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1732.0 | 1772.0 | false | null | null |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1785.0 | 1 | 1785.0 | 1790.0 | false | "false_alarm" | null |


`chb01_16` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1015.0 | 1066.0 | true | 3 | 5.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 870.0 | 1 | 870.0 | 875.0 | false | "false_alarm" | null |
| 2 | 1010.0 | 1 | 1010.0 | 1015.0 | false | "false_alarm" | null |
| 3 | 1020.0 | 2 | 1025.0 | 1030.0 | false | "detection" | 1 |
| 4 | 1035.0 | 6 | 1060.0 | 1065.0 | false | "repeat" | 1 |


### results/lightgbm_v1/comparison.json — root.ordinary_rf

Source: [results/lightgbm_v1/comparison.json](../results/lightgbm_v1/comparison.json) → `root.ordinary_rf`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1440 | 18 | 0.995138888889 | 1 | 0.611111111111 | 1 | 0.758620689655 | 0.982947564991 | [[1422,0],[7,11]] |

| chb01_17 | 720 | 0 | 1 | undefined | undefined | 1 | undefined | undefined | [[720,0],[0,0]] |

| chb01_18 | 720 | 18 | 0.990277777778 | 1 | 0.611111111111 | 1 | 0.758620689655 | 1 | [[702,0],[7,11]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 0 | 2 | 0 | 2 | [10.0] | 10 | 10 |

| chb01_17 | 0 | 0 | 0 | undefined | 0 | 1 | 0 | 0 | [] | undefined | undefined |

| chb01_18 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 2 | [10.0] | 10 | 10 |



`chb01_17` event/alarm details and undefined reasons:

**seizures**

`[]`

**alarms**

`[]`

**undefined_window_metrics**

`{"precision": "zero denominator", "recall": "zero denominator", "f1": "zero denominator", "average_precision": "no positive ground-truth windows"}`


`chb01_18` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1720.0 | 1810.0 | true | 1 | 10.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1730.0 | 1 | 1730.0 | 1735.0 | false | "detection" | 1 |
| 2 | 1740.0 | 9 | 1780.0 | 1785.0 | false | "repeat" | 1 |
| 3 | 1795.0 | 1 | 1795.0 | 1800.0 | false | "repeat" | 1 |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_17",
    "label": 0,
    "windows": 720,
    "predicted_positive": 0,
    "mean": 0.0057083333333333335,
    "min": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 0.0,
    "q95": 0.03,
    "max": 0.34
  },
  {
    "recording_id": "chb01_18",
    "label": 0,
    "windows": 702,
    "predicted_positive": 0,
    "mean": 0.001168091168091168,
    "min": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 0.0,
    "q95": 0.005,
    "max": 0.16
  },
  {
    "recording_id": "chb01_18",
    "label": 1,
    "windows": 18,
    "predicted_positive": 11,
    "mean": 0.5833333333333334,
    "min": 0.215,
    "q25": 0.3625,
    "median": 0.565,
    "q75": 0.7887500000000001,
    "q95": 0.9452499999999999,
    "max": 0.975
  }
]
```

### results/lightgbm_v1/comparison.json — root.weighted_rf

Source: [results/lightgbm_v1/comparison.json](../results/lightgbm_v1/comparison.json) → `root.weighted_rf`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1440 | 18 | 0.99375 | 1 | 0.5 | 1 | 0.666666666667 | 0.94043796856 | [[1422,0],[9,9]] |

| chb01_17 | 720 | 0 | 1 | undefined | undefined | 1 | undefined | undefined | [[720,0],[0,0]] |

| chb01_18 | 720 | 18 | 0.9875 | 1 | 0.5 | 1 | 0.666666666667 | 0.96164485564 | [[702,0],[9,9]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 0 | 2 | 0 | 1 | [5.0] | 5 | 5 |

| chb01_17 | 0 | 0 | 0 | undefined | 0 | 1 | 0 | 0 | [] | undefined | undefined |

| chb01_18 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | [5.0] | 5 | 5 |



`chb01_17` event/alarm details and undefined reasons:

**seizures**

`[]`

**alarms**

`[]`

**undefined_window_metrics**

`{"precision": "zero denominator", "recall": "zero denominator", "f1": "zero denominator", "average_precision": "no positive ground-truth windows"}`


`chb01_18` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1720.0 | 1810.0 | true | 1 | 5.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1725.0 | 2 | 1730.0 | 1735.0 | false | "detection" | 1 |
| 2 | 1740.0 | 7 | 1770.0 | 1775.0 | false | "repeat" | 1 |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_17",
    "label": 0,
    "windows": 720,
    "predicted_positive": 0,
    "mean": 0.0005486111111111112,
    "min": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 0.0,
    "q95": 0.0,
    "max": 0.15
  },
  {
    "recording_id": "chb01_18",
    "label": 0,
    "windows": 702,
    "predicted_positive": 0,
    "mean": 0.0007977207977207978,
    "min": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 0.0,
    "q95": 0.005,
    "max": 0.085
  },
  {
    "recording_id": "chb01_18",
    "label": 1,
    "windows": 18,
    "predicted_positive": 9,
    "mean": 0.5238888888888888,
    "min": 0.01,
    "q25": 0.11875,
    "median": 0.5025,
    "q75": 0.93375,
    "q95": 0.99,
    "max": 0.99
  }
]
```

### results/lightgbm_v1/comparison.json — root.ordinary_lgbm

Source: [results/lightgbm_v1/comparison.json](../results/lightgbm_v1/comparison.json) → `root.ordinary_lgbm`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1440 | 18 | 0.99375 | 0.909090909091 | 0.555555555556 | 0.99929676512 | 0.689655172414 | 0.847911789967 | [[1421,1],[8,10]] |

| chb01_17 | 720 | 0 | 0.998611111111 | 0 | undefined | 0.998611111111 | 0 | undefined | [[719,1],[0,0]] |

| chb01_18 | 720 | 18 | 0.988888888889 | 1 | 0.555555555556 | 1 | 0.714285714286 | 0.963942242064 | [[702,0],[8,10]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 1 | 2 | 0.5 | 0 | [5.0] | 5 | 5 |

| chb01_17 | 0 | 0 | 0 | undefined | 1 | 1 | 1 | 0 | [] | undefined | undefined |

| chb01_18 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [5.0] | 5 | 5 |



`chb01_17` event/alarm details and undefined reasons:

**seizures**

`[]`

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1825.0 | 1 | 1825.0 | 1830.0 | false | "false_alarm" | null |

**undefined_window_metrics**

`{"recall": "zero denominator", "average_precision": "no positive ground-truth windows"}`


`chb01_18` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1720.0 | 1810.0 | true | 1 | 5.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1725.0 | 10 | 1770.0 | 1775.0 | false | "detection" | 1 |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_17",
    "label": 0,
    "windows": 720,
    "predicted_positive": 1,
    "mean": 0.0014456644641128002,
    "min": 1.915000919008218e-06,
    "q25": 4.1254398079302976e-06,
    "median": 4.1254398079302976e-06,
    "q75": 4.23367696082743e-06,
    "q95": 5.022428214028077e-06,
    "max": 0.9576539316762845
  },
  {
    "recording_id": "chb01_18",
    "label": 0,
    "windows": 702,
    "predicted_positive": 0,
    "mean": 2.667927681818086e-05,
    "min": 4.120072495641895e-06,
    "q25": 4.1254398079302976e-06,
    "median": 4.1254398079302976e-06,
    "q75": 4.1254398079302976e-06,
    "q95": 7.500757899620423e-06,
    "max": 0.014656507619529505
  },
  {
    "recording_id": "chb01_18",
    "label": 1,
    "windows": 18,
    "predicted_positive": 10,
    "mean": 0.5170172535298267,
    "min": 6.282312482062467e-05,
    "q25": 0.001066224598142506,
    "median": 0.67238798037939,
    "q75": 0.9975002528634387,
    "q95": 0.9999293621633811,
    "max": 0.9999548568160628
  }
]
```

### results/lightgbm_v1/comparison.json — root.weighted_lgbm

Source: [results/lightgbm_v1/comparison.json](../results/lightgbm_v1/comparison.json) → `root.weighted_lgbm`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1440 | 18 | 0.99375 | 0.909090909091 | 0.555555555556 | 0.99929676512 | 0.689655172414 | 0.690847975861 | [[1421,1],[8,10]] |

| chb01_17 | 720 | 0 | 1 | undefined | undefined | 1 | undefined | undefined | [[720,0],[0,0]] |

| chb01_18 | 720 | 18 | 0.9875 | 0.909090909091 | 0.555555555556 | 0.998575498575 | 0.689655172414 | 0.76283914534 | [[701,1],[8,10]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 1 | 2 | 0.5 | 0 | [5.0] | 5 | 5 |

| chb01_17 | 0 | 0 | 0 | undefined | 0 | 1 | 0 | 0 | [] | undefined | undefined |

| chb01_18 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 0 | [5.0] | 5 | 5 |



`chb01_17` event/alarm details and undefined reasons:

**seizures**

`[]`

**alarms**

`[]`

**undefined_window_metrics**

`{"precision": "zero denominator", "recall": "zero denominator", "f1": "zero denominator", "average_precision": "no positive ground-truth windows"}`


`chb01_18` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1720.0 | 1810.0 | true | 1 | 5.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1725.0 | 10 | 1770.0 | 1775.0 | false | "detection" | 1 |
| 2 | 2535.0 | 1 | 2535.0 | 2540.0 | false | "false_alarm" | null |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_17",
    "label": 0,
    "windows": 720,
    "predicted_positive": 0,
    "mean": 4.0591793251561746e-05,
    "min": 5.662673483071732e-07,
    "q25": 3.2028483061968295e-06,
    "median": 3.7607555825414913e-06,
    "q75": 4.301734460462898e-06,
    "q95": 3.497351174310493e-05,
    "max": 0.006637899950519555
  },
  {
    "recording_id": "chb01_18",
    "label": 0,
    "windows": 702,
    "predicted_positive": 1,
    "mean": 0.000957528892000567,
    "min": 1.594309976143612e-07,
    "q25": 3.3329367445966057e-06,
    "median": 4.342820876501969e-06,
    "q75": 7.4974657896325965e-06,
    "q95": 1.637606946832231e-05,
    "max": 0.6537654296338538
  },
  {
    "recording_id": "chb01_18",
    "label": 1,
    "windows": 18,
    "predicted_positive": 10,
    "mean": 0.5597969999830863,
    "min": 2.7540202837799114e-07,
    "q25": 9.101478036866963e-05,
    "median": 0.998923814106714,
    "q75": 0.9999004639537648,
    "q95": 0.9999144560113445,
    "max": 0.9999144560113445
  }
]
```

### results/lightgbm_v1/weighted_lgbm_evaluation.json — root

Source: [results/lightgbm_v1/weighted_lgbm_evaluation.json](../results/lightgbm_v1/weighted_lgbm_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1440 | 18 | 0.99375 | 0.909090909091 | 0.555555555556 | 0.99929676512 | 0.689655172414 | 0.690847975861 | [[1421,1],[8,10]] |

| chb01_17 | 720 | 0 | 1 | undefined | undefined | 1 | undefined | undefined | [[720,0],[0,0]] |

| chb01_18 | 720 | 18 | 0.9875 | 0.909090909091 | 0.555555555556 | 0.998575498575 | 0.689655172414 | 0.76283914534 | [[701,1],[8,10]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 1 | 2 | 0.5 | 0 | [5.0] | 5 | 5 |

| chb01_17 | 0 | 0 | 0 | undefined | 0 | 1 | 0 | 0 | [] | undefined | undefined |

| chb01_18 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 0 | [5.0] | 5 | 5 |



`chb01_17` event/alarm details and undefined reasons:

**seizures**

`[]`

**alarms**

`[]`

**undefined_window_metrics**

`{"precision": "zero denominator", "recall": "zero denominator", "f1": "zero denominator", "average_precision": "no positive ground-truth windows"}`


`chb01_18` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1720.0 | 1810.0 | true | 1 | 5.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1725.0 | 10 | 1770.0 | 1775.0 | false | "detection" | 1 |
| 2 | 2535.0 | 1 | 2535.0 | 2540.0 | false | "false_alarm" | null |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_17",
    "label": 0,
    "windows": 720,
    "predicted_positive": 0,
    "mean": 4.0591793251561746e-05,
    "min": 5.662673483071732e-07,
    "q25": 3.2028483061968295e-06,
    "median": 3.7607555825414913e-06,
    "q75": 4.301734460462898e-06,
    "q95": 3.497351174310493e-05,
    "max": 0.006637899950519555
  },
  {
    "recording_id": "chb01_18",
    "label": 0,
    "windows": 702,
    "predicted_positive": 1,
    "mean": 0.000957528892000567,
    "min": 1.594309976143612e-07,
    "q25": 3.3329367445966057e-06,
    "median": 4.342820876501969e-06,
    "q75": 7.4974657896325965e-06,
    "q95": 1.637606946832231e-05,
    "max": 0.6537654296338538
  },
  {
    "recording_id": "chb01_18",
    "label": 1,
    "windows": 18,
    "predicted_positive": 10,
    "mean": 0.5597969999830863,
    "min": 2.7540202837799114e-07,
    "q25": 9.101478036866963e-05,
    "median": 0.998923814106714,
    "q75": 0.9999004639537648,
    "q95": 0.9999144560113445,
    "max": 0.9999144560113445
  }
]
```

### results/lightgbm_v1/ordinary_lgbm_evaluation.json — root

Source: [results/lightgbm_v1/ordinary_lgbm_evaluation.json](../results/lightgbm_v1/ordinary_lgbm_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1440 | 18 | 0.99375 | 0.909090909091 | 0.555555555556 | 0.99929676512 | 0.689655172414 | 0.847911789967 | [[1421,1],[8,10]] |

| chb01_17 | 720 | 0 | 0.998611111111 | 0 | undefined | 0.998611111111 | 0 | undefined | [[719,1],[0,0]] |

| chb01_18 | 720 | 18 | 0.988888888889 | 1 | 0.555555555556 | 1 | 0.714285714286 | 0.963942242064 | [[702,0],[8,10]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 1 | 2 | 0.5 | 0 | [5.0] | 5 | 5 |

| chb01_17 | 0 | 0 | 0 | undefined | 1 | 1 | 1 | 0 | [] | undefined | undefined |

| chb01_18 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [5.0] | 5 | 5 |



`chb01_17` event/alarm details and undefined reasons:

**seizures**

`[]`

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1825.0 | 1 | 1825.0 | 1830.0 | false | "false_alarm" | null |

**undefined_window_metrics**

`{"recall": "zero denominator", "average_precision": "no positive ground-truth windows"}`


`chb01_18` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1720.0 | 1810.0 | true | 1 | 5.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1725.0 | 10 | 1770.0 | 1775.0 | false | "detection" | 1 |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_17",
    "label": 0,
    "windows": 720,
    "predicted_positive": 1,
    "mean": 0.0014456644641128002,
    "min": 1.915000919008218e-06,
    "q25": 4.1254398079302976e-06,
    "median": 4.1254398079302976e-06,
    "q75": 4.23367696082743e-06,
    "q95": 5.022428214028077e-06,
    "max": 0.9576539316762845
  },
  {
    "recording_id": "chb01_18",
    "label": 0,
    "windows": 702,
    "predicted_positive": 0,
    "mean": 2.667927681818086e-05,
    "min": 4.120072495641895e-06,
    "q25": 4.1254398079302976e-06,
    "median": 4.1254398079302976e-06,
    "q75": 4.1254398079302976e-06,
    "q95": 7.500757899620423e-06,
    "max": 0.014656507619529505
  },
  {
    "recording_id": "chb01_18",
    "label": 1,
    "windows": 18,
    "predicted_positive": 10,
    "mean": 0.5170172535298267,
    "min": 6.282312482062467e-05,
    "q25": 0.001066224598142506,
    "median": 0.67238798037939,
    "q75": 0.9975002528634387,
    "q95": 0.9999293621633811,
    "max": 0.9999548568160628
  }
]
```

### results/lightgbm_v1/weighted_rf_evaluation.json — root

Source: [results/lightgbm_v1/weighted_rf_evaluation.json](../results/lightgbm_v1/weighted_rf_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1440 | 18 | 0.99375 | 1 | 0.5 | 1 | 0.666666666667 | 0.94043796856 | [[1422,0],[9,9]] |

| chb01_17 | 720 | 0 | 1 | undefined | undefined | 1 | undefined | undefined | [[720,0],[0,0]] |

| chb01_18 | 720 | 18 | 0.9875 | 1 | 0.5 | 1 | 0.666666666667 | 0.96164485564 | [[702,0],[9,9]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 0 | 2 | 0 | 1 | [5.0] | 5 | 5 |

| chb01_17 | 0 | 0 | 0 | undefined | 0 | 1 | 0 | 0 | [] | undefined | undefined |

| chb01_18 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | [5.0] | 5 | 5 |



`chb01_17` event/alarm details and undefined reasons:

**seizures**

`[]`

**alarms**

`[]`

**undefined_window_metrics**

`{"precision": "zero denominator", "recall": "zero denominator", "f1": "zero denominator", "average_precision": "no positive ground-truth windows"}`


`chb01_18` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1720.0 | 1810.0 | true | 1 | 5.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1725.0 | 2 | 1730.0 | 1735.0 | false | "detection" | 1 |
| 2 | 1740.0 | 7 | 1770.0 | 1775.0 | false | "repeat" | 1 |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_17",
    "label": 0,
    "windows": 720,
    "predicted_positive": 0,
    "mean": 0.0005486111111111112,
    "min": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 0.0,
    "q95": 0.0,
    "max": 0.15
  },
  {
    "recording_id": "chb01_18",
    "label": 0,
    "windows": 702,
    "predicted_positive": 0,
    "mean": 0.0007977207977207978,
    "min": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 0.0,
    "q95": 0.005,
    "max": 0.085
  },
  {
    "recording_id": "chb01_18",
    "label": 1,
    "windows": 18,
    "predicted_positive": 9,
    "mean": 0.5238888888888888,
    "min": 0.01,
    "q25": 0.11875,
    "median": 0.5025,
    "q75": 0.93375,
    "q95": 0.99,
    "max": 0.99
  }
]
```

### results/lightgbm_v1/ordinary_rf_evaluation.json — root

Source: [results/lightgbm_v1/ordinary_rf_evaluation.json](../results/lightgbm_v1/ordinary_rf_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1440 | 18 | 0.995138888889 | 1 | 0.611111111111 | 1 | 0.758620689655 | 0.982947564991 | [[1422,0],[7,11]] |

| chb01_17 | 720 | 0 | 1 | undefined | undefined | 1 | undefined | undefined | [[720,0],[0,0]] |

| chb01_18 | 720 | 18 | 0.990277777778 | 1 | 0.611111111111 | 1 | 0.758620689655 | 1 | [[702,0],[7,11]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 0 | 2 | 0 | 2 | [10.0] | 10 | 10 |

| chb01_17 | 0 | 0 | 0 | undefined | 0 | 1 | 0 | 0 | [] | undefined | undefined |

| chb01_18 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 2 | [10.0] | 10 | 10 |



`chb01_17` event/alarm details and undefined reasons:

**seizures**

`[]`

**alarms**

`[]`

**undefined_window_metrics**

`{"precision": "zero denominator", "recall": "zero denominator", "f1": "zero denominator", "average_precision": "no positive ground-truth windows"}`


`chb01_18` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1720.0 | 1810.0 | true | 1 | 10.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1730.0 | 1 | 1730.0 | 1735.0 | false | "detection" | 1 |
| 2 | 1740.0 | 9 | 1780.0 | 1785.0 | false | "repeat" | 1 |
| 3 | 1795.0 | 1 | 1795.0 | 1800.0 | false | "repeat" | 1 |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_17",
    "label": 0,
    "windows": 720,
    "predicted_positive": 0,
    "mean": 0.0057083333333333335,
    "min": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 0.0,
    "q95": 0.03,
    "max": 0.34
  },
  {
    "recording_id": "chb01_18",
    "label": 0,
    "windows": 702,
    "predicted_positive": 0,
    "mean": 0.001168091168091168,
    "min": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 0.0,
    "q95": 0.005,
    "max": 0.16
  },
  {
    "recording_id": "chb01_18",
    "label": 1,
    "windows": 18,
    "predicted_positive": 11,
    "mean": 0.5833333333333334,
    "min": 0.215,
    "q25": 0.3625,
    "median": 0.565,
    "q75": 0.7887500000000001,
    "q95": 0.9452499999999999,
    "max": 0.975
  }
]
```

### results/rf_cv_v1/fold_2_chb01_04/ordinary_evaluation.json — root

Source: [results/rf_cv_v1/fold_2_chb01_04/ordinary_evaluation.json](../results/rf_cv_v1/fold_2_chb01_04/ordinary_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 720 | 6 | 0.997222222222 | 1 | 0.666666666667 | 1 | 0.8 | 1 | [[714,0],[2,4]] |

| chb01_04 | 720 | 6 | 0.997222222222 | 1 | 0.666666666667 | 1 | 0.8 | 1 | [[714,0],[2,4]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | [8.0] | 8 | 8 |

| chb01_04 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | [8.0] | 8 | 8 |



`chb01_04` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1467.0 | 1494.0 | true | 1 | 8.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1475.0 | 1 | 1475.0 | 1480.0 | false | "detection" | 1 |
| 2 | 1485.0 | 3 | 1495.0 | 1500.0 | false | "repeat" | 1 |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_04",
    "label": 0,
    "windows": 714,
    "predicted_positive": 0,
    "mean": 0.0026750700280112048,
    "min": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 0.0,
    "q95": 0.005,
    "max": 0.42
  },
  {
    "recording_id": "chb01_04",
    "label": 1,
    "windows": 6,
    "predicted_positive": 4,
    "mean": 0.6833333333333332,
    "min": 0.47,
    "q25": 0.5225,
    "median": 0.6225,
    "q75": 0.8500000000000001,
    "q95": 0.9575,
    "max": 0.97
  }
]
```

### results/rf_cv_v1/fold_2_chb01_04/weighted_evaluation.json — root

Source: [results/rf_cv_v1/fold_2_chb01_04/weighted_evaluation.json](../results/rf_cv_v1/fold_2_chb01_04/weighted_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 720 | 6 | 0.995833333333 | 0.8 | 0.666666666667 | 0.998599439776 | 0.727272727273 | 0.930555555556 | [[713,1],[2,4]] |

| chb01_04 | 720 | 6 | 0.995833333333 | 0.8 | 0.666666666667 | 0.998599439776 | 0.727272727273 | 0.930555555556 | [[713,1],[2,4]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | [8.0] | 8 | 8 |

| chb01_04 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | [8.0] | 8 | 8 |



`chb01_04` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1467.0 | 1494.0 | true | 1 | 8.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1475.0 | 2 | 1480.0 | 1485.0 | false | "detection" | 1 |
| 2 | 1490.0 | 3 | 1500.0 | 1505.0 | false | "repeat" | 1 |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_04",
    "label": 0,
    "windows": 714,
    "predicted_positive": 1,
    "mean": 0.0028781512605042018,
    "min": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 0.0,
    "q95": 0.005,
    "max": 0.565
  },
  {
    "recording_id": "chb01_04",
    "label": 1,
    "windows": 6,
    "predicted_positive": 4,
    "mean": 0.7125,
    "min": 0.245,
    "q25": 0.5700000000000001,
    "median": 0.8,
    "q75": 0.9212499999999999,
    "q95": 0.97,
    "max": 0.975
  }
]
```

### results/rf_cv_v1/fold_4_chb01_16/ordinary_evaluation.json — root

Source: [results/rf_cv_v1/fold_4_chb01_16/ordinary_evaluation.json](../results/rf_cv_v1/fold_4_chb01_16/ordinary_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 720 | 10 | 0.995833333333 | 1 | 0.7 | 1 | 0.823529411765 | 0.966923076923 | [[710,0],[3,7]] |

| chb01_16 | 720 | 10 | 0.995833333333 | 1 | 0.7 | 1 | 0.823529411765 | 0.966923076923 | [[710,0],[3,7]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | [5.0] | 5 | 5 |

| chb01_16 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | [5.0] | 5 | 5 |



`chb01_16` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1015.0 | 1066.0 | true | 1 | 5.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1020.0 | 2 | 1025.0 | 1030.0 | false | "detection" | 1 |
| 2 | 1035.0 | 5 | 1055.0 | 1060.0 | false | "repeat" | 1 |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_16",
    "label": 0,
    "windows": 710,
    "predicted_positive": 0,
    "mean": 0.005309859154929578,
    "min": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 0.0,
    "q95": 0.025,
    "max": 0.275
  },
  {
    "recording_id": "chb01_16",
    "label": 1,
    "windows": 10,
    "predicted_positive": 7,
    "mean": 0.637,
    "min": 0.2,
    "q25": 0.46625,
    "median": 0.71,
    "q75": 0.85375,
    "q95": 0.90325,
    "max": 0.91
  }
]
```

### results/rf_cv_v1/fold_4_chb01_16/weighted_evaluation.json — root

Source: [results/rf_cv_v1/fold_4_chb01_16/weighted_evaluation.json](../results/rf_cv_v1/fold_4_chb01_16/weighted_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 720 | 10 | 0.994444444444 | 0.8 | 0.8 | 0.997183098592 | 0.8 | 0.965151515152 | [[708,2],[2,8]] |

| chb01_16 | 720 | 10 | 0.994444444444 | 0.8 | 0.8 | 0.997183098592 | 0.8 | 0.965151515152 | [[708,2],[2,8]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 2 | 1 | 2 | 1 | [5.0] | 5 | 5 |

| chb01_16 | 1 | 1 | 0 | 1 | 2 | 1 | 2 | 1 | [5.0] | 5 | 5 |



`chb01_16` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1015.0 | 1066.0 | true | 3 | 5.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 870.0 | 1 | 870.0 | 875.0 | false | "false_alarm" | null |
| 2 | 1010.0 | 1 | 1010.0 | 1015.0 | false | "false_alarm" | null |
| 3 | 1020.0 | 2 | 1025.0 | 1030.0 | false | "detection" | 1 |
| 4 | 1035.0 | 6 | 1060.0 | 1065.0 | false | "repeat" | 1 |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_16",
    "label": 0,
    "windows": 710,
    "predicted_positive": 2,
    "mean": 0.004619718309859155,
    "min": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 0.0,
    "q95": 0.01,
    "max": 0.625
  },
  {
    "recording_id": "chb01_16",
    "label": 1,
    "windows": 10,
    "predicted_positive": 8,
    "mean": 0.7445,
    "min": 0.395,
    "q25": 0.63875,
    "median": 0.8225,
    "q75": 0.8625,
    "q95": 0.9754999999999999,
    "max": 0.98
  }
]
```

### results/rf_cv_v1/fold_3_chb01_15/ordinary_evaluation.json — root

Source: [results/rf_cv_v1/fold_3_chb01_15/ordinary_evaluation.json](../results/rf_cv_v1/fold_3_chb01_15/ordinary_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 720 | 8 | 0.9875 | 0 | 0 | 0.998595505618 | 0 | 0.440566378066 | [[711,1],[8,0]] |

| chb01_15 | 720 | 8 | 0.9875 | 0 | 0 | 0.998595505618 | 0 | 0.440566378066 | [[711,1],[8,0]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | [] | undefined | undefined |

| chb01_15 | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | [] | undefined | undefined |



`chb01_15` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1732.0 | 1772.0 | false | null | null |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1785.0 | 1 | 1785.0 | 1790.0 | false | "false_alarm" | null |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_15",
    "label": 0,
    "windows": 712,
    "predicted_positive": 1,
    "mean": 0.0032865168539325845,
    "min": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 0.0,
    "q95": 0.005,
    "max": 0.52
  },
  {
    "recording_id": "chb01_15",
    "label": 1,
    "windows": 8,
    "predicted_positive": 0,
    "mean": 0.204375,
    "min": 0.0,
    "q25": 0.06625,
    "median": 0.2175,
    "q75": 0.3375,
    "q95": 0.388,
    "max": 0.395
  }
]
```

### results/rf_cv_v1/fold_3_chb01_15/weighted_evaluation.json — root

Source: [results/rf_cv_v1/fold_3_chb01_15/weighted_evaluation.json](../results/rf_cv_v1/fold_3_chb01_15/weighted_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 720 | 8 | 0.9875 | 0 | 0 | 0.998595505618 | 0 | 0.271631000578 | [[711,1],[8,0]] |

| chb01_15 | 720 | 8 | 0.9875 | 0 | 0 | 0.998595505618 | 0 | 0.271631000578 | [[711,1],[8,0]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | [] | undefined | undefined |

| chb01_15 | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | [] | undefined | undefined |



`chb01_15` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1732.0 | 1772.0 | false | null | null |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1785.0 | 1 | 1785.0 | 1790.0 | false | "false_alarm" | null |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_15",
    "label": 0,
    "windows": 712,
    "predicted_positive": 1,
    "mean": 0.0035042134831460674,
    "min": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 0.0,
    "q95": 0.005,
    "max": 0.68
  },
  {
    "recording_id": "chb01_15",
    "label": 1,
    "windows": 8,
    "predicted_positive": 0,
    "mean": 0.06125,
    "min": 0.01,
    "q25": 0.015,
    "median": 0.030000000000000002,
    "q75": 0.07125000000000001,
    "q95": 0.178,
    "max": 0.185
  }
]
```

### results/rf_cv_v1/fold_1_chb01_03/ordinary_evaluation.json — root

Source: [results/rf_cv_v1/fold_1_chb01_03/ordinary_evaluation.json](../results/rf_cv_v1/fold_1_chb01_03/ordinary_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 720 | 8 | 0.997222222222 | 1 | 0.75 | 1 | 0.857142857143 | 0.860763888889 | [[712,0],[2,6]] |

| chb01_03 | 720 | 8 | 0.997222222222 | 1 | 0.75 | 1 | 0.857142857143 | 0.860763888889 | [[712,0],[2,6]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [9.0] | 9 | 9 |

| chb01_03 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [9.0] | 9 | 9 |



`chb01_03` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 2996.0 | 3036.0 | true | 1 | 9.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3005.0 | 6 | 3030.0 | 3035.0 | false | "detection" | 1 |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_03",
    "label": 0,
    "windows": 712,
    "predicted_positive": 0,
    "mean": 0.001734550561797753,
    "min": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 0.0,
    "q95": 0.005,
    "max": 0.215
  },
  {
    "recording_id": "chb01_03",
    "label": 1,
    "windows": 8,
    "predicted_positive": 6,
    "mean": 0.571875,
    "min": 0.0,
    "q25": 0.45000000000000007,
    "median": 0.6625,
    "q75": 0.81,
    "q95": 0.883,
    "max": 0.89
  }
]
```

### results/rf_cv_v1/fold_1_chb01_03/weighted_evaluation.json — root

Source: [results/rf_cv_v1/fold_1_chb01_03/weighted_evaluation.json](../results/rf_cv_v1/fold_1_chb01_03/weighted_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 720 | 8 | 0.997222222222 | 0.875 | 0.875 | 0.998595505618 | 0.875 | 0.876388888889 | [[711,1],[1,7]] |

| chb01_03 | 720 | 8 | 0.997222222222 | 0.875 | 0.875 | 0.998595505618 | 0.875 | 0.876388888889 | [[711,1],[1,7]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 0 | [9.0] | 9 | 9 |

| chb01_03 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 0 | [9.0] | 9 | 9 |



`chb01_03` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 2996.0 | 3036.0 | true | 1 | 9.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3005.0 | 7 | 3035.0 | 3040.0 | false | "detection" | 1 |
| 2 | 3050.0 | 1 | 3050.0 | 3055.0 | false | "false_alarm" | null |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_03",
    "label": 0,
    "windows": 712,
    "predicted_positive": 1,
    "mean": 0.004199438202247191,
    "min": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 0.0,
    "q95": 0.005,
    "max": 0.725
  },
  {
    "recording_id": "chb01_03",
    "label": 1,
    "windows": 8,
    "predicted_positive": 7,
    "mean": 0.78375,
    "min": 0.0,
    "q25": 0.82125,
    "median": 0.855,
    "q75": 0.96375,
    "q95": 0.99,
    "max": 0.99
  }
]
```

### results/temporal_filtering_v1/secondary_validation/ordinary_lgbm_two_of_three_evaluation.json — root

Source: [results/temporal_filtering_v1/secondary_validation/ordinary_lgbm_two_of_three_evaluation.json](../results/temporal_filtering_v1/secondary_validation/ordinary_lgbm_two_of_three_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1440 | 18 | 0.994444444444 | 1 | 0.555555555556 | 1 | 0.714285714286 | undefined | [[1422,0],[8,10]] |

| chb01_17 | 720 | 0 | 1 | undefined | undefined | 1 | undefined | undefined | [[720,0],[0,0]] |

| chb01_18 | 720 | 18 | 0.988888888889 | 1 | 0.555555555556 | 1 | 0.714285714286 | undefined | [[702,0],[8,10]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 0 | 2 | 0 | 0 | [10.0] | 10 | 10 |

| chb01_17 | 0 | 0 | 0 | undefined | 0 | 1 | 0 | 0 | [] | undefined | undefined |

| chb01_18 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [10.0] | 10 | 10 |



`pooled` event/alarm details and undefined reasons:

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_17` event/alarm details and undefined reasons:

**seizures**

`[]`

**alarms**

`[]`

**undefined_window_metrics**

`{"precision": "zero denominator", "recall": "zero denominator", "f1": "zero denominator", "average_precision": "no continuous scores supplied"}`


`chb01_18` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1720.0 | 1810.0 | true | 1 | 10.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1730.0 | 10 | 1775.0 | 1780.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`score_policy`:

```json
"Original scores retained in prediction files; AP not evaluated for alert strategies"
```

`alert_rule`:

```json
"output 1 iff at least 2 of current and up to 2 previous raw decisions are 1"
```

### results/temporal_filtering_v1/secondary_validation/ordinary_rf_two_consecutive_evaluation.json — root

Source: [results/temporal_filtering_v1/secondary_validation/ordinary_rf_two_consecutive_evaluation.json](../results/temporal_filtering_v1/secondary_validation/ordinary_rf_two_consecutive_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1440 | 18 | 0.993055555556 | 1 | 0.444444444444 | 1 | 0.615384615385 | undefined | [[1422,0],[10,8]] |

| chb01_17 | 720 | 0 | 1 | undefined | undefined | 1 | undefined | undefined | [[720,0],[0,0]] |

| chb01_18 | 720 | 18 | 0.986111111111 | 1 | 0.444444444444 | 1 | 0.615384615385 | undefined | [[702,0],[10,8]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 0 | 2 | 0 | 0 | [25.0] | 25 | 25 |

| chb01_17 | 0 | 0 | 0 | undefined | 0 | 1 | 0 | 0 | [] | undefined | undefined |

| chb01_18 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [25.0] | 25 | 25 |



`pooled` event/alarm details and undefined reasons:

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_17` event/alarm details and undefined reasons:

**seizures**

`[]`

**alarms**

`[]`

**undefined_window_metrics**

`{"precision": "zero denominator", "recall": "zero denominator", "f1": "zero denominator", "average_precision": "no continuous scores supplied"}`


`chb01_18` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1720.0 | 1810.0 | true | 1 | 25.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1745.0 | 8 | 1780.0 | 1785.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`score_policy`:

```json
"Original scores retained in prediction files; AP not evaluated for alert strategies"
```

`alert_rule`:

```json
"output 1 iff current and immediately previous raw decisions are 1"
```

### results/temporal_filtering_v1/secondary_validation/ordinary_lgbm_none_evaluation.json — root

Source: [results/temporal_filtering_v1/secondary_validation/ordinary_lgbm_none_evaluation.json](../results/temporal_filtering_v1/secondary_validation/ordinary_lgbm_none_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1440 | 18 | 0.99375 | 0.909090909091 | 0.555555555556 | 0.99929676512 | 0.689655172414 | undefined | [[1421,1],[8,10]] |

| chb01_17 | 720 | 0 | 0.998611111111 | 0 | undefined | 0.998611111111 | 0 | undefined | [[719,1],[0,0]] |

| chb01_18 | 720 | 18 | 0.988888888889 | 1 | 0.555555555556 | 1 | 0.714285714286 | undefined | [[702,0],[8,10]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 1 | 2 | 0.5 | 0 | [5.0] | 5 | 5 |

| chb01_17 | 0 | 0 | 0 | undefined | 1 | 1 | 1 | 0 | [] | undefined | undefined |

| chb01_18 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [5.0] | 5 | 5 |



`pooled` event/alarm details and undefined reasons:

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_17` event/alarm details and undefined reasons:

**seizures**

`[]`

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1825.0 | 1 | 1825.0 | 1830.0 | false | "false_alarm" | null |

**undefined_window_metrics**

`{"recall": "zero denominator", "average_precision": "no continuous scores supplied"}`


`chb01_18` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1720.0 | 1810.0 | true | 1 | 5.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1725.0 | 10 | 1770.0 | 1775.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`score_policy`:

```json
"Original scores retained in prediction files; AP not evaluated for alert strategies"
```

`alert_rule`:

```json
"output current raw decision"
```

### results/temporal_filtering_v1/secondary_validation/comparison.json — root.ordinary_rf.none

Source: [results/temporal_filtering_v1/secondary_validation/comparison.json](../results/temporal_filtering_v1/secondary_validation/comparison.json) → `root.ordinary_rf.none`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1440 | 18 | 0.995138888889 | 1 | 0.611111111111 | 1 | 0.758620689655 | undefined | [[1422,0],[7,11]] |

| chb01_17 | 720 | 0 | 1 | undefined | undefined | 1 | undefined | undefined | [[720,0],[0,0]] |

| chb01_18 | 720 | 18 | 0.990277777778 | 1 | 0.611111111111 | 1 | 0.758620689655 | undefined | [[702,0],[7,11]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 0 | 2 | 0 | 2 | [10.0] | 10 | 10 |

| chb01_17 | 0 | 0 | 0 | undefined | 0 | 1 | 0 | 0 | [] | undefined | undefined |

| chb01_18 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 2 | [10.0] | 10 | 10 |



`pooled` event/alarm details and undefined reasons:

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_17` event/alarm details and undefined reasons:

**seizures**

`[]`

**alarms**

`[]`

**undefined_window_metrics**

`{"precision": "zero denominator", "recall": "zero denominator", "f1": "zero denominator", "average_precision": "no continuous scores supplied"}`


`chb01_18` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1720.0 | 1810.0 | true | 1 | 10.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1730.0 | 1 | 1730.0 | 1735.0 | false | "detection" | 1 |
| 2 | 1740.0 | 9 | 1780.0 | 1785.0 | false | "repeat" | 1 |
| 3 | 1795.0 | 1 | 1795.0 | 1800.0 | false | "repeat" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`score_policy`:

```json
"Original scores retained in prediction files; AP not evaluated for alert strategies"
```

`alert_rule`:

```json
"output current raw decision"
```

### results/temporal_filtering_v1/secondary_validation/comparison.json — root.ordinary_rf.two_consecutive

Source: [results/temporal_filtering_v1/secondary_validation/comparison.json](../results/temporal_filtering_v1/secondary_validation/comparison.json) → `root.ordinary_rf.two_consecutive`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1440 | 18 | 0.993055555556 | 1 | 0.444444444444 | 1 | 0.615384615385 | undefined | [[1422,0],[10,8]] |

| chb01_17 | 720 | 0 | 1 | undefined | undefined | 1 | undefined | undefined | [[720,0],[0,0]] |

| chb01_18 | 720 | 18 | 0.986111111111 | 1 | 0.444444444444 | 1 | 0.615384615385 | undefined | [[702,0],[10,8]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 0 | 2 | 0 | 0 | [25.0] | 25 | 25 |

| chb01_17 | 0 | 0 | 0 | undefined | 0 | 1 | 0 | 0 | [] | undefined | undefined |

| chb01_18 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [25.0] | 25 | 25 |



`pooled` event/alarm details and undefined reasons:

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_17` event/alarm details and undefined reasons:

**seizures**

`[]`

**alarms**

`[]`

**undefined_window_metrics**

`{"precision": "zero denominator", "recall": "zero denominator", "f1": "zero denominator", "average_precision": "no continuous scores supplied"}`


`chb01_18` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1720.0 | 1810.0 | true | 1 | 25.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1745.0 | 8 | 1780.0 | 1785.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`score_policy`:

```json
"Original scores retained in prediction files; AP not evaluated for alert strategies"
```

`alert_rule`:

```json
"output 1 iff current and immediately previous raw decisions are 1"
```

### results/temporal_filtering_v1/secondary_validation/comparison.json — root.ordinary_rf.two_of_three

Source: [results/temporal_filtering_v1/secondary_validation/comparison.json](../results/temporal_filtering_v1/secondary_validation/comparison.json) → `root.ordinary_rf.two_of_three`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1440 | 18 | 0.994444444444 | 1 | 0.555555555556 | 1 | 0.714285714286 | undefined | [[1422,0],[8,10]] |

| chb01_17 | 720 | 0 | 1 | undefined | undefined | 1 | undefined | undefined | [[720,0],[0,0]] |

| chb01_18 | 720 | 18 | 0.988888888889 | 1 | 0.555555555556 | 1 | 0.714285714286 | undefined | [[702,0],[8,10]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 0 | 2 | 0 | 0 | [20.0] | 20 | 20 |

| chb01_17 | 0 | 0 | 0 | undefined | 0 | 1 | 0 | 0 | [] | undefined | undefined |

| chb01_18 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [20.0] | 20 | 20 |



`pooled` event/alarm details and undefined reasons:

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_17` event/alarm details and undefined reasons:

**seizures**

`[]`

**alarms**

`[]`

**undefined_window_metrics**

`{"precision": "zero denominator", "recall": "zero denominator", "f1": "zero denominator", "average_precision": "no continuous scores supplied"}`


`chb01_18` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1720.0 | 1810.0 | true | 1 | 20.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1740.0 | 10 | 1785.0 | 1790.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`score_policy`:

```json
"Original scores retained in prediction files; AP not evaluated for alert strategies"
```

`alert_rule`:

```json
"output 1 iff at least 2 of current and up to 2 previous raw decisions are 1"
```

### results/temporal_filtering_v1/secondary_validation/comparison.json — root.ordinary_lgbm.none

Source: [results/temporal_filtering_v1/secondary_validation/comparison.json](../results/temporal_filtering_v1/secondary_validation/comparison.json) → `root.ordinary_lgbm.none`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1440 | 18 | 0.99375 | 0.909090909091 | 0.555555555556 | 0.99929676512 | 0.689655172414 | undefined | [[1421,1],[8,10]] |

| chb01_17 | 720 | 0 | 0.998611111111 | 0 | undefined | 0.998611111111 | 0 | undefined | [[719,1],[0,0]] |

| chb01_18 | 720 | 18 | 0.988888888889 | 1 | 0.555555555556 | 1 | 0.714285714286 | undefined | [[702,0],[8,10]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 1 | 2 | 0.5 | 0 | [5.0] | 5 | 5 |

| chb01_17 | 0 | 0 | 0 | undefined | 1 | 1 | 1 | 0 | [] | undefined | undefined |

| chb01_18 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [5.0] | 5 | 5 |



`pooled` event/alarm details and undefined reasons:

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_17` event/alarm details and undefined reasons:

**seizures**

`[]`

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1825.0 | 1 | 1825.0 | 1830.0 | false | "false_alarm" | null |

**undefined_window_metrics**

`{"recall": "zero denominator", "average_precision": "no continuous scores supplied"}`


`chb01_18` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1720.0 | 1810.0 | true | 1 | 5.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1725.0 | 10 | 1770.0 | 1775.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`score_policy`:

```json
"Original scores retained in prediction files; AP not evaluated for alert strategies"
```

`alert_rule`:

```json
"output current raw decision"
```

### results/temporal_filtering_v1/secondary_validation/comparison.json — root.ordinary_lgbm.two_consecutive

Source: [results/temporal_filtering_v1/secondary_validation/comparison.json](../results/temporal_filtering_v1/secondary_validation/comparison.json) → `root.ordinary_lgbm.two_consecutive`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1440 | 18 | 0.99375 | 1 | 0.5 | 1 | 0.666666666667 | undefined | [[1422,0],[9,9]] |

| chb01_17 | 720 | 0 | 1 | undefined | undefined | 1 | undefined | undefined | [[720,0],[0,0]] |

| chb01_18 | 720 | 18 | 0.9875 | 1 | 0.5 | 1 | 0.666666666667 | undefined | [[702,0],[9,9]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 0 | 2 | 0 | 0 | [10.0] | 10 | 10 |

| chb01_17 | 0 | 0 | 0 | undefined | 0 | 1 | 0 | 0 | [] | undefined | undefined |

| chb01_18 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [10.0] | 10 | 10 |



`pooled` event/alarm details and undefined reasons:

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_17` event/alarm details and undefined reasons:

**seizures**

`[]`

**alarms**

`[]`

**undefined_window_metrics**

`{"precision": "zero denominator", "recall": "zero denominator", "f1": "zero denominator", "average_precision": "no continuous scores supplied"}`


`chb01_18` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1720.0 | 1810.0 | true | 1 | 10.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1730.0 | 9 | 1770.0 | 1775.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`score_policy`:

```json
"Original scores retained in prediction files; AP not evaluated for alert strategies"
```

`alert_rule`:

```json
"output 1 iff current and immediately previous raw decisions are 1"
```

### results/temporal_filtering_v1/secondary_validation/comparison.json — root.ordinary_lgbm.two_of_three

Source: [results/temporal_filtering_v1/secondary_validation/comparison.json](../results/temporal_filtering_v1/secondary_validation/comparison.json) → `root.ordinary_lgbm.two_of_three`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1440 | 18 | 0.994444444444 | 1 | 0.555555555556 | 1 | 0.714285714286 | undefined | [[1422,0],[8,10]] |

| chb01_17 | 720 | 0 | 1 | undefined | undefined | 1 | undefined | undefined | [[720,0],[0,0]] |

| chb01_18 | 720 | 18 | 0.988888888889 | 1 | 0.555555555556 | 1 | 0.714285714286 | undefined | [[702,0],[8,10]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 0 | 2 | 0 | 0 | [10.0] | 10 | 10 |

| chb01_17 | 0 | 0 | 0 | undefined | 0 | 1 | 0 | 0 | [] | undefined | undefined |

| chb01_18 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [10.0] | 10 | 10 |



`pooled` event/alarm details and undefined reasons:

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_17` event/alarm details and undefined reasons:

**seizures**

`[]`

**alarms**

`[]`

**undefined_window_metrics**

`{"precision": "zero denominator", "recall": "zero denominator", "f1": "zero denominator", "average_precision": "no continuous scores supplied"}`


`chb01_18` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1720.0 | 1810.0 | true | 1 | 10.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1730.0 | 10 | 1775.0 | 1780.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`score_policy`:

```json
"Original scores retained in prediction files; AP not evaluated for alert strategies"
```

`alert_rule`:

```json
"output 1 iff at least 2 of current and up to 2 previous raw decisions are 1"
```

### results/temporal_filtering_v1/secondary_validation/ordinary_rf_two_of_three_evaluation.json — root

Source: [results/temporal_filtering_v1/secondary_validation/ordinary_rf_two_of_three_evaluation.json](../results/temporal_filtering_v1/secondary_validation/ordinary_rf_two_of_three_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1440 | 18 | 0.994444444444 | 1 | 0.555555555556 | 1 | 0.714285714286 | undefined | [[1422,0],[8,10]] |

| chb01_17 | 720 | 0 | 1 | undefined | undefined | 1 | undefined | undefined | [[720,0],[0,0]] |

| chb01_18 | 720 | 18 | 0.988888888889 | 1 | 0.555555555556 | 1 | 0.714285714286 | undefined | [[702,0],[8,10]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 0 | 2 | 0 | 0 | [20.0] | 20 | 20 |

| chb01_17 | 0 | 0 | 0 | undefined | 0 | 1 | 0 | 0 | [] | undefined | undefined |

| chb01_18 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [20.0] | 20 | 20 |



`pooled` event/alarm details and undefined reasons:

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_17` event/alarm details and undefined reasons:

**seizures**

`[]`

**alarms**

`[]`

**undefined_window_metrics**

`{"precision": "zero denominator", "recall": "zero denominator", "f1": "zero denominator", "average_precision": "no continuous scores supplied"}`


`chb01_18` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1720.0 | 1810.0 | true | 1 | 20.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1740.0 | 10 | 1785.0 | 1790.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`score_policy`:

```json
"Original scores retained in prediction files; AP not evaluated for alert strategies"
```

`alert_rule`:

```json
"output 1 iff at least 2 of current and up to 2 previous raw decisions are 1"
```

### results/temporal_filtering_v1/secondary_validation/ordinary_lgbm_two_consecutive_evaluation.json — root

Source: [results/temporal_filtering_v1/secondary_validation/ordinary_lgbm_two_consecutive_evaluation.json](../results/temporal_filtering_v1/secondary_validation/ordinary_lgbm_two_consecutive_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1440 | 18 | 0.99375 | 1 | 0.5 | 1 | 0.666666666667 | undefined | [[1422,0],[9,9]] |

| chb01_17 | 720 | 0 | 1 | undefined | undefined | 1 | undefined | undefined | [[720,0],[0,0]] |

| chb01_18 | 720 | 18 | 0.9875 | 1 | 0.5 | 1 | 0.666666666667 | undefined | [[702,0],[9,9]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 0 | 2 | 0 | 0 | [10.0] | 10 | 10 |

| chb01_17 | 0 | 0 | 0 | undefined | 0 | 1 | 0 | 0 | [] | undefined | undefined |

| chb01_18 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [10.0] | 10 | 10 |



`pooled` event/alarm details and undefined reasons:

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_17` event/alarm details and undefined reasons:

**seizures**

`[]`

**alarms**

`[]`

**undefined_window_metrics**

`{"precision": "zero denominator", "recall": "zero denominator", "f1": "zero denominator", "average_precision": "no continuous scores supplied"}`


`chb01_18` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1720.0 | 1810.0 | true | 1 | 10.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1730.0 | 9 | 1770.0 | 1775.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`score_policy`:

```json
"Original scores retained in prediction files; AP not evaluated for alert strategies"
```

`alert_rule`:

```json
"output 1 iff current and immediately previous raw decisions are 1"
```

### results/temporal_filtering_v1/secondary_validation/ordinary_rf_none_evaluation.json — root

Source: [results/temporal_filtering_v1/secondary_validation/ordinary_rf_none_evaluation.json](../results/temporal_filtering_v1/secondary_validation/ordinary_rf_none_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1440 | 18 | 0.995138888889 | 1 | 0.611111111111 | 1 | 0.758620689655 | undefined | [[1422,0],[7,11]] |

| chb01_17 | 720 | 0 | 1 | undefined | undefined | 1 | undefined | undefined | [[720,0],[0,0]] |

| chb01_18 | 720 | 18 | 0.990277777778 | 1 | 0.611111111111 | 1 | 0.758620689655 | undefined | [[702,0],[7,11]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 0 | 2 | 0 | 2 | [10.0] | 10 | 10 |

| chb01_17 | 0 | 0 | 0 | undefined | 0 | 1 | 0 | 0 | [] | undefined | undefined |

| chb01_18 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 2 | [10.0] | 10 | 10 |



`pooled` event/alarm details and undefined reasons:

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_17` event/alarm details and undefined reasons:

**seizures**

`[]`

**alarms**

`[]`

**undefined_window_metrics**

`{"precision": "zero denominator", "recall": "zero denominator", "f1": "zero denominator", "average_precision": "no continuous scores supplied"}`


`chb01_18` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1720.0 | 1810.0 | true | 1 | 10.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1730.0 | 1 | 1730.0 | 1735.0 | false | "detection" | 1 |
| 2 | 1740.0 | 9 | 1780.0 | 1785.0 | false | "repeat" | 1 |
| 3 | 1795.0 | 1 | 1795.0 | 1800.0 | false | "repeat" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`score_policy`:

```json
"Original scores retained in prediction files; AP not evaluated for alert strategies"
```

`alert_rule`:

```json
"output current raw decision"
```

### results/temporal_filtering_v1/primary_cv/ordinary_lgbm_two_of_three_evaluation.json — root

Source: [results/temporal_filtering_v1/primary_cv/ordinary_lgbm_two_of_three_evaluation.json](../results/temporal_filtering_v1/primary_cv/ordinary_lgbm_two_of_three_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 2880 | 32 | 0.994444444444 | 0.863636363636 | 0.59375 | 0.998946629213 | 0.703703703704 | undefined | [[2845,3],[13,19]] |

| chb01_03 | 720 | 8 | 0.997222222222 | 1 | 0.75 | 1 | 0.857142857143 | undefined | [[712,0],[2,6]] |

| chb01_04 | 720 | 6 | 0.994444444444 | 0.666666666667 | 0.666666666667 | 0.997198879552 | 0.666666666667 | undefined | [[712,2],[2,4]] |

| chb01_15 | 720 | 8 | 0.988888888889 | 0.5 | 0.125 | 0.998595505618 | 0.2 | undefined | [[711,1],[7,1]] |

| chb01_16 | 720 | 10 | 0.997222222222 | 1 | 0.8 | 1 | 0.888888888889 | undefined | [[710,0],[2,8]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 4 | 4 | 0 | 1 | 0 | 4 | 0 | 0 | [14.0,13.0,38.0,10.0] | 18.75 | 13.5 |

| chb01_03 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [14.0] | 14 | 14 |

| chb01_04 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [13.0] | 13 | 13 |

| chb01_15 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [38.0] | 38 | 38 |

| chb01_16 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [10.0] | 10 | 10 |



`pooled` event/alarm details and undefined reasons:

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_03` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 2996.0 | 3036.0 | true | 1 | 14.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3010.0 | 6 | 3035.0 | 3040.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_04` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1467.0 | 1494.0 | true | 1 | 13.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1480.0 | 6 | 1505.0 | 1510.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_15` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1732.0 | 1772.0 | true | 1 | 38.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1770.0 | 2 | 1775.0 | 1780.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_16` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1015.0 | 1066.0 | true | 1 | 10.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1025.0 | 8 | 1060.0 | 1065.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`score_policy`:

```json
"Original scores retained in prediction files; AP not evaluated for alert strategies"
```

`alert_rule`:

```json
"output 1 iff at least 2 of current and up to 2 previous raw decisions are 1"
```

### results/temporal_filtering_v1/primary_cv/ordinary_rf_two_consecutive_evaluation.json — root

Source: [results/temporal_filtering_v1/primary_cv/ordinary_rf_two_consecutive_evaluation.json](../results/temporal_filtering_v1/primary_cv/ordinary_rf_two_consecutive_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 2880 | 32 | 0.993055555556 | 1 | 0.375 | 1 | 0.545454545455 | undefined | [[2848,0],[20,12]] |

| chb01_03 | 720 | 8 | 0.995833333333 | 1 | 0.625 | 1 | 0.769230769231 | undefined | [[712,0],[3,5]] |

| chb01_04 | 720 | 6 | 0.994444444444 | 1 | 0.333333333333 | 1 | 0.5 | undefined | [[714,0],[4,2]] |

| chb01_15 | 720 | 8 | 0.988888888889 | undefined | 0 | 1 | 0 | undefined | [[712,0],[8,0]] |

| chb01_16 | 720 | 10 | 0.993055555556 | 1 | 0.5 | 1 | 0.666666666667 | undefined | [[710,0],[5,5]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 4 | 3 | 1 | 0.75 | 0 | 4 | 0 | 1 | [14.0,23.0,10.0] | 15.6666666667 | 14 |

| chb01_03 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [14.0] | 14 | 14 |

| chb01_04 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [23.0] | 23 | 23 |

| chb01_15 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | [] | undefined | undefined |

| chb01_16 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | [10.0] | 10 | 10 |



`pooled` event/alarm details and undefined reasons:

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_03` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 2996.0 | 3036.0 | true | 1 | 14.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3010.0 | 5 | 3030.0 | 3035.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_04` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1467.0 | 1494.0 | true | 1 | 23.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1490.0 | 2 | 1495.0 | 1500.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_15` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1732.0 | 1772.0 | false | null | null |

**alarms**

`[]`

**undefined_window_metrics**

`{"precision": "zero denominator", "average_precision": "no continuous scores supplied"}`


`chb01_16` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1015.0 | 1066.0 | true | 1 | 10.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1025.0 | 1 | 1025.0 | 1030.0 | false | "detection" | 1 |
| 2 | 1040.0 | 4 | 1055.0 | 1060.0 | false | "repeat" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`score_policy`:

```json
"Original scores retained in prediction files; AP not evaluated for alert strategies"
```

`alert_rule`:

```json
"output 1 iff current and immediately previous raw decisions are 1"
```

### results/temporal_filtering_v1/primary_cv/ordinary_lgbm_none_evaluation.json — root

Source: [results/temporal_filtering_v1/primary_cv/ordinary_lgbm_none_evaluation.json](../results/temporal_filtering_v1/primary_cv/ordinary_lgbm_none_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 2880 | 32 | 0.995138888889 | 0.875 | 0.65625 | 0.998946629213 | 0.75 | undefined | [[2845,3],[11,21]] |

| chb01_03 | 720 | 8 | 0.997222222222 | 1 | 0.75 | 1 | 0.857142857143 | undefined | [[712,0],[2,6]] |

| chb01_04 | 720 | 6 | 0.997222222222 | 0.833333333333 | 0.833333333333 | 0.998599439776 | 0.833333333333 | undefined | [[713,1],[1,5]] |

| chb01_15 | 720 | 8 | 0.991666666667 | 0.75 | 0.375 | 0.998595505618 | 0.5 | undefined | [[711,1],[5,3]] |

| chb01_16 | 720 | 10 | 0.994444444444 | 0.875 | 0.7 | 0.998591549296 | 0.777777777778 | undefined | [[709,1],[3,7]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 4 | 4 | 0 | 1 | 2 | 4 | 0.5 | 2 | [9.0,8.0,8.0,5.0] | 7.5 | 8 |

| chb01_03 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [9.0] | 9 | 9 |

| chb01_04 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [8.0] | 8 | 8 |

| chb01_15 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | [8.0] | 8 | 8 |

| chb01_16 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | [5.0] | 5 | 5 |



`pooled` event/alarm details and undefined reasons:

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_03` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 2996.0 | 3036.0 | true | 1 | 9.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3005.0 | 6 | 3030.0 | 3035.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_04` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1467.0 | 1494.0 | true | 1 | 8.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1475.0 | 6 | 1500.0 | 1505.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_15` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1732.0 | 1772.0 | true | 1 | 8.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1740.0 | 1 | 1740.0 | 1745.0 | false | "detection" | 1 |
| 2 | 1765.0 | 2 | 1770.0 | 1775.0 | false | "repeat" | 1 |
| 3 | 1785.0 | 1 | 1785.0 | 1790.0 | false | "false_alarm" | null |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_16` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1015.0 | 1066.0 | true | 2 | 5.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 870.0 | 1 | 870.0 | 875.0 | false | "false_alarm" | null |
| 2 | 1020.0 | 2 | 1025.0 | 1030.0 | false | "detection" | 1 |
| 3 | 1035.0 | 5 | 1055.0 | 1060.0 | false | "repeat" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`score_policy`:

```json
"Original scores retained in prediction files; AP not evaluated for alert strategies"
```

`alert_rule`:

```json
"output current raw decision"
```

### results/temporal_filtering_v1/primary_cv/comparison.json — root.ordinary_rf.none

Source: [results/temporal_filtering_v1/primary_cv/comparison.json](../results/temporal_filtering_v1/primary_cv/comparison.json) → `root.ordinary_rf.none`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 2880 | 32 | 0.994444444444 | 0.944444444444 | 0.53125 | 0.999648876404 | 0.68 | undefined | [[2847,1],[15,17]] |

| chb01_03 | 720 | 8 | 0.997222222222 | 1 | 0.75 | 1 | 0.857142857143 | undefined | [[712,0],[2,6]] |

| chb01_04 | 720 | 6 | 0.997222222222 | 1 | 0.666666666667 | 1 | 0.8 | undefined | [[714,0],[2,4]] |

| chb01_15 | 720 | 8 | 0.9875 | 0 | 0 | 0.998595505618 | 0 | undefined | [[711,1],[8,0]] |

| chb01_16 | 720 | 10 | 0.995833333333 | 1 | 0.7 | 1 | 0.823529411765 | undefined | [[710,0],[3,7]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 4 | 3 | 1 | 0.75 | 1 | 4 | 0.25 | 2 | [9.0,8.0,5.0] | 7.33333333333 | 8 |

| chb01_03 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [9.0] | 9 | 9 |

| chb01_04 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | [8.0] | 8 | 8 |

| chb01_15 | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | [] | undefined | undefined |

| chb01_16 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | [5.0] | 5 | 5 |



`pooled` event/alarm details and undefined reasons:

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_03` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 2996.0 | 3036.0 | true | 1 | 9.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3005.0 | 6 | 3030.0 | 3035.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_04` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1467.0 | 1494.0 | true | 1 | 8.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1475.0 | 1 | 1475.0 | 1480.0 | false | "detection" | 1 |
| 2 | 1485.0 | 3 | 1495.0 | 1500.0 | false | "repeat" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_15` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1732.0 | 1772.0 | false | null | null |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1785.0 | 1 | 1785.0 | 1790.0 | false | "false_alarm" | null |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_16` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1015.0 | 1066.0 | true | 1 | 5.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1020.0 | 2 | 1025.0 | 1030.0 | false | "detection" | 1 |
| 2 | 1035.0 | 5 | 1055.0 | 1060.0 | false | "repeat" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`score_policy`:

```json
"Original scores retained in prediction files; AP not evaluated for alert strategies"
```

`alert_rule`:

```json
"output current raw decision"
```

### results/temporal_filtering_v1/primary_cv/comparison.json — root.ordinary_rf.two_consecutive

Source: [results/temporal_filtering_v1/primary_cv/comparison.json](../results/temporal_filtering_v1/primary_cv/comparison.json) → `root.ordinary_rf.two_consecutive`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 2880 | 32 | 0.993055555556 | 1 | 0.375 | 1 | 0.545454545455 | undefined | [[2848,0],[20,12]] |

| chb01_03 | 720 | 8 | 0.995833333333 | 1 | 0.625 | 1 | 0.769230769231 | undefined | [[712,0],[3,5]] |

| chb01_04 | 720 | 6 | 0.994444444444 | 1 | 0.333333333333 | 1 | 0.5 | undefined | [[714,0],[4,2]] |

| chb01_15 | 720 | 8 | 0.988888888889 | undefined | 0 | 1 | 0 | undefined | [[712,0],[8,0]] |

| chb01_16 | 720 | 10 | 0.993055555556 | 1 | 0.5 | 1 | 0.666666666667 | undefined | [[710,0],[5,5]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 4 | 3 | 1 | 0.75 | 0 | 4 | 0 | 1 | [14.0,23.0,10.0] | 15.6666666667 | 14 |

| chb01_03 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [14.0] | 14 | 14 |

| chb01_04 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [23.0] | 23 | 23 |

| chb01_15 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | [] | undefined | undefined |

| chb01_16 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | [10.0] | 10 | 10 |



`pooled` event/alarm details and undefined reasons:

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_03` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 2996.0 | 3036.0 | true | 1 | 14.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3010.0 | 5 | 3030.0 | 3035.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_04` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1467.0 | 1494.0 | true | 1 | 23.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1490.0 | 2 | 1495.0 | 1500.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_15` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1732.0 | 1772.0 | false | null | null |

**alarms**

`[]`

**undefined_window_metrics**

`{"precision": "zero denominator", "average_precision": "no continuous scores supplied"}`


`chb01_16` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1015.0 | 1066.0 | true | 1 | 10.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1025.0 | 1 | 1025.0 | 1030.0 | false | "detection" | 1 |
| 2 | 1040.0 | 4 | 1055.0 | 1060.0 | false | "repeat" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`score_policy`:

```json
"Original scores retained in prediction files; AP not evaluated for alert strategies"
```

`alert_rule`:

```json
"output 1 iff current and immediately previous raw decisions are 1"
```

### results/temporal_filtering_v1/primary_cv/comparison.json — root.ordinary_rf.two_of_three

Source: [results/temporal_filtering_v1/primary_cv/comparison.json](../results/temporal_filtering_v1/primary_cv/comparison.json) → `root.ordinary_rf.two_of_three`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 2880 | 32 | 0.994444444444 | 0.944444444444 | 0.53125 | 0.999648876404 | 0.68 | undefined | [[2847,1],[15,17]] |

| chb01_03 | 720 | 8 | 0.997222222222 | 1 | 0.75 | 1 | 0.857142857143 | undefined | [[712,0],[2,6]] |

| chb01_04 | 720 | 6 | 0.994444444444 | 0.75 | 0.5 | 0.998599439776 | 0.6 | undefined | [[713,1],[3,3]] |

| chb01_15 | 720 | 8 | 0.988888888889 | undefined | 0 | 1 | 0 | undefined | [[712,0],[8,0]] |

| chb01_16 | 720 | 10 | 0.997222222222 | 1 | 0.8 | 1 | 0.888888888889 | undefined | [[710,0],[2,8]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 4 | 3 | 1 | 0.75 | 0 | 4 | 0 | 0 | [14.0,18.0,10.0] | 14 | 14 |

| chb01_03 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [14.0] | 14 | 14 |

| chb01_04 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [18.0] | 18 | 18 |

| chb01_15 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | [] | undefined | undefined |

| chb01_16 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [10.0] | 10 | 10 |



`pooled` event/alarm details and undefined reasons:

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_03` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 2996.0 | 3036.0 | true | 1 | 14.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3010.0 | 6 | 3035.0 | 3040.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_04` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1467.0 | 1494.0 | true | 1 | 18.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1485.0 | 4 | 1500.0 | 1505.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_15` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1732.0 | 1772.0 | false | null | null |

**alarms**

`[]`

**undefined_window_metrics**

`{"precision": "zero denominator", "average_precision": "no continuous scores supplied"}`


`chb01_16` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1015.0 | 1066.0 | true | 1 | 10.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1025.0 | 8 | 1060.0 | 1065.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`score_policy`:

```json
"Original scores retained in prediction files; AP not evaluated for alert strategies"
```

`alert_rule`:

```json
"output 1 iff at least 2 of current and up to 2 previous raw decisions are 1"
```

### results/temporal_filtering_v1/primary_cv/comparison.json — root.ordinary_lgbm.none

Source: [results/temporal_filtering_v1/primary_cv/comparison.json](../results/temporal_filtering_v1/primary_cv/comparison.json) → `root.ordinary_lgbm.none`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 2880 | 32 | 0.995138888889 | 0.875 | 0.65625 | 0.998946629213 | 0.75 | undefined | [[2845,3],[11,21]] |

| chb01_03 | 720 | 8 | 0.997222222222 | 1 | 0.75 | 1 | 0.857142857143 | undefined | [[712,0],[2,6]] |

| chb01_04 | 720 | 6 | 0.997222222222 | 0.833333333333 | 0.833333333333 | 0.998599439776 | 0.833333333333 | undefined | [[713,1],[1,5]] |

| chb01_15 | 720 | 8 | 0.991666666667 | 0.75 | 0.375 | 0.998595505618 | 0.5 | undefined | [[711,1],[5,3]] |

| chb01_16 | 720 | 10 | 0.994444444444 | 0.875 | 0.7 | 0.998591549296 | 0.777777777778 | undefined | [[709,1],[3,7]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 4 | 4 | 0 | 1 | 2 | 4 | 0.5 | 2 | [9.0,8.0,8.0,5.0] | 7.5 | 8 |

| chb01_03 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [9.0] | 9 | 9 |

| chb01_04 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [8.0] | 8 | 8 |

| chb01_15 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | [8.0] | 8 | 8 |

| chb01_16 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | [5.0] | 5 | 5 |



`pooled` event/alarm details and undefined reasons:

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_03` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 2996.0 | 3036.0 | true | 1 | 9.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3005.0 | 6 | 3030.0 | 3035.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_04` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1467.0 | 1494.0 | true | 1 | 8.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1475.0 | 6 | 1500.0 | 1505.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_15` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1732.0 | 1772.0 | true | 1 | 8.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1740.0 | 1 | 1740.0 | 1745.0 | false | "detection" | 1 |
| 2 | 1765.0 | 2 | 1770.0 | 1775.0 | false | "repeat" | 1 |
| 3 | 1785.0 | 1 | 1785.0 | 1790.0 | false | "false_alarm" | null |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_16` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1015.0 | 1066.0 | true | 2 | 5.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 870.0 | 1 | 870.0 | 875.0 | false | "false_alarm" | null |
| 2 | 1020.0 | 2 | 1025.0 | 1030.0 | false | "detection" | 1 |
| 3 | 1035.0 | 5 | 1055.0 | 1060.0 | false | "repeat" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`score_policy`:

```json
"Original scores retained in prediction files; AP not evaluated for alert strategies"
```

`alert_rule`:

```json
"output current raw decision"
```

### results/temporal_filtering_v1/primary_cv/comparison.json — root.ordinary_lgbm.two_consecutive

Source: [results/temporal_filtering_v1/primary_cv/comparison.json](../results/temporal_filtering_v1/primary_cv/comparison.json) → `root.ordinary_lgbm.two_consecutive`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 2880 | 32 | 0.99375 | 0.9375 | 0.46875 | 0.999648876404 | 0.625 | undefined | [[2847,1],[17,15]] |

| chb01_03 | 720 | 8 | 0.995833333333 | 1 | 0.625 | 1 | 0.769230769231 | undefined | [[712,0],[3,5]] |

| chb01_04 | 720 | 6 | 0.995833333333 | 0.8 | 0.666666666667 | 0.998599439776 | 0.727272727273 | undefined | [[713,1],[2,4]] |

| chb01_15 | 720 | 8 | 0.990277777778 | 1 | 0.125 | 1 | 0.222222222222 | undefined | [[712,0],[7,1]] |

| chb01_16 | 720 | 10 | 0.993055555556 | 1 | 0.5 | 1 | 0.666666666667 | undefined | [[710,0],[5,5]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 4 | 4 | 0 | 1 | 0 | 4 | 0 | 1 | [14.0,13.0,38.0,10.0] | 18.75 | 13.5 |

| chb01_03 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [14.0] | 14 | 14 |

| chb01_04 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [13.0] | 13 | 13 |

| chb01_15 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [38.0] | 38 | 38 |

| chb01_16 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | [10.0] | 10 | 10 |



`pooled` event/alarm details and undefined reasons:

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_03` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 2996.0 | 3036.0 | true | 1 | 14.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3010.0 | 5 | 3030.0 | 3035.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_04` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1467.0 | 1494.0 | true | 1 | 13.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1480.0 | 5 | 1500.0 | 1505.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_15` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1732.0 | 1772.0 | true | 1 | 38.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1770.0 | 1 | 1770.0 | 1775.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_16` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1015.0 | 1066.0 | true | 1 | 10.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1025.0 | 1 | 1025.0 | 1030.0 | false | "detection" | 1 |
| 2 | 1040.0 | 4 | 1055.0 | 1060.0 | false | "repeat" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`score_policy`:

```json
"Original scores retained in prediction files; AP not evaluated for alert strategies"
```

`alert_rule`:

```json
"output 1 iff current and immediately previous raw decisions are 1"
```

### results/temporal_filtering_v1/primary_cv/comparison.json — root.ordinary_lgbm.two_of_three

Source: [results/temporal_filtering_v1/primary_cv/comparison.json](../results/temporal_filtering_v1/primary_cv/comparison.json) → `root.ordinary_lgbm.two_of_three`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 2880 | 32 | 0.994444444444 | 0.863636363636 | 0.59375 | 0.998946629213 | 0.703703703704 | undefined | [[2845,3],[13,19]] |

| chb01_03 | 720 | 8 | 0.997222222222 | 1 | 0.75 | 1 | 0.857142857143 | undefined | [[712,0],[2,6]] |

| chb01_04 | 720 | 6 | 0.994444444444 | 0.666666666667 | 0.666666666667 | 0.997198879552 | 0.666666666667 | undefined | [[712,2],[2,4]] |

| chb01_15 | 720 | 8 | 0.988888888889 | 0.5 | 0.125 | 0.998595505618 | 0.2 | undefined | [[711,1],[7,1]] |

| chb01_16 | 720 | 10 | 0.997222222222 | 1 | 0.8 | 1 | 0.888888888889 | undefined | [[710,0],[2,8]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 4 | 4 | 0 | 1 | 0 | 4 | 0 | 0 | [14.0,13.0,38.0,10.0] | 18.75 | 13.5 |

| chb01_03 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [14.0] | 14 | 14 |

| chb01_04 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [13.0] | 13 | 13 |

| chb01_15 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [38.0] | 38 | 38 |

| chb01_16 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [10.0] | 10 | 10 |



`pooled` event/alarm details and undefined reasons:

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_03` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 2996.0 | 3036.0 | true | 1 | 14.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3010.0 | 6 | 3035.0 | 3040.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_04` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1467.0 | 1494.0 | true | 1 | 13.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1480.0 | 6 | 1505.0 | 1510.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_15` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1732.0 | 1772.0 | true | 1 | 38.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1770.0 | 2 | 1775.0 | 1780.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_16` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1015.0 | 1066.0 | true | 1 | 10.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1025.0 | 8 | 1060.0 | 1065.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`score_policy`:

```json
"Original scores retained in prediction files; AP not evaluated for alert strategies"
```

`alert_rule`:

```json
"output 1 iff at least 2 of current and up to 2 previous raw decisions are 1"
```

### results/temporal_filtering_v1/primary_cv/ordinary_rf_two_of_three_evaluation.json — root

Source: [results/temporal_filtering_v1/primary_cv/ordinary_rf_two_of_three_evaluation.json](../results/temporal_filtering_v1/primary_cv/ordinary_rf_two_of_three_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 2880 | 32 | 0.994444444444 | 0.944444444444 | 0.53125 | 0.999648876404 | 0.68 | undefined | [[2847,1],[15,17]] |

| chb01_03 | 720 | 8 | 0.997222222222 | 1 | 0.75 | 1 | 0.857142857143 | undefined | [[712,0],[2,6]] |

| chb01_04 | 720 | 6 | 0.994444444444 | 0.75 | 0.5 | 0.998599439776 | 0.6 | undefined | [[713,1],[3,3]] |

| chb01_15 | 720 | 8 | 0.988888888889 | undefined | 0 | 1 | 0 | undefined | [[712,0],[8,0]] |

| chb01_16 | 720 | 10 | 0.997222222222 | 1 | 0.8 | 1 | 0.888888888889 | undefined | [[710,0],[2,8]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 4 | 3 | 1 | 0.75 | 0 | 4 | 0 | 0 | [14.0,18.0,10.0] | 14 | 14 |

| chb01_03 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [14.0] | 14 | 14 |

| chb01_04 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [18.0] | 18 | 18 |

| chb01_15 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | [] | undefined | undefined |

| chb01_16 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [10.0] | 10 | 10 |



`pooled` event/alarm details and undefined reasons:

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_03` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 2996.0 | 3036.0 | true | 1 | 14.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3010.0 | 6 | 3035.0 | 3040.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_04` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1467.0 | 1494.0 | true | 1 | 18.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1485.0 | 4 | 1500.0 | 1505.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_15` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1732.0 | 1772.0 | false | null | null |

**alarms**

`[]`

**undefined_window_metrics**

`{"precision": "zero denominator", "average_precision": "no continuous scores supplied"}`


`chb01_16` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1015.0 | 1066.0 | true | 1 | 10.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1025.0 | 8 | 1060.0 | 1065.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`score_policy`:

```json
"Original scores retained in prediction files; AP not evaluated for alert strategies"
```

`alert_rule`:

```json
"output 1 iff at least 2 of current and up to 2 previous raw decisions are 1"
```

### results/temporal_filtering_v1/primary_cv/ordinary_lgbm_two_consecutive_evaluation.json — root

Source: [results/temporal_filtering_v1/primary_cv/ordinary_lgbm_two_consecutive_evaluation.json](../results/temporal_filtering_v1/primary_cv/ordinary_lgbm_two_consecutive_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 2880 | 32 | 0.99375 | 0.9375 | 0.46875 | 0.999648876404 | 0.625 | undefined | [[2847,1],[17,15]] |

| chb01_03 | 720 | 8 | 0.995833333333 | 1 | 0.625 | 1 | 0.769230769231 | undefined | [[712,0],[3,5]] |

| chb01_04 | 720 | 6 | 0.995833333333 | 0.8 | 0.666666666667 | 0.998599439776 | 0.727272727273 | undefined | [[713,1],[2,4]] |

| chb01_15 | 720 | 8 | 0.990277777778 | 1 | 0.125 | 1 | 0.222222222222 | undefined | [[712,0],[7,1]] |

| chb01_16 | 720 | 10 | 0.993055555556 | 1 | 0.5 | 1 | 0.666666666667 | undefined | [[710,0],[5,5]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 4 | 4 | 0 | 1 | 0 | 4 | 0 | 1 | [14.0,13.0,38.0,10.0] | 18.75 | 13.5 |

| chb01_03 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [14.0] | 14 | 14 |

| chb01_04 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [13.0] | 13 | 13 |

| chb01_15 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [38.0] | 38 | 38 |

| chb01_16 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | [10.0] | 10 | 10 |



`pooled` event/alarm details and undefined reasons:

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_03` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 2996.0 | 3036.0 | true | 1 | 14.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3010.0 | 5 | 3030.0 | 3035.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_04` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1467.0 | 1494.0 | true | 1 | 13.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1480.0 | 5 | 1500.0 | 1505.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_15` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1732.0 | 1772.0 | true | 1 | 38.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1770.0 | 1 | 1770.0 | 1775.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_16` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1015.0 | 1066.0 | true | 1 | 10.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1025.0 | 1 | 1025.0 | 1030.0 | false | "detection" | 1 |
| 2 | 1040.0 | 4 | 1055.0 | 1060.0 | false | "repeat" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`score_policy`:

```json
"Original scores retained in prediction files; AP not evaluated for alert strategies"
```

`alert_rule`:

```json
"output 1 iff current and immediately previous raw decisions are 1"
```

### results/temporal_filtering_v1/primary_cv/ordinary_rf_none_evaluation.json — root

Source: [results/temporal_filtering_v1/primary_cv/ordinary_rf_none_evaluation.json](../results/temporal_filtering_v1/primary_cv/ordinary_rf_none_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 2880 | 32 | 0.994444444444 | 0.944444444444 | 0.53125 | 0.999648876404 | 0.68 | undefined | [[2847,1],[15,17]] |

| chb01_03 | 720 | 8 | 0.997222222222 | 1 | 0.75 | 1 | 0.857142857143 | undefined | [[712,0],[2,6]] |

| chb01_04 | 720 | 6 | 0.997222222222 | 1 | 0.666666666667 | 1 | 0.8 | undefined | [[714,0],[2,4]] |

| chb01_15 | 720 | 8 | 0.9875 | 0 | 0 | 0.998595505618 | 0 | undefined | [[711,1],[8,0]] |

| chb01_16 | 720 | 10 | 0.995833333333 | 1 | 0.7 | 1 | 0.823529411765 | undefined | [[710,0],[3,7]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 4 | 3 | 1 | 0.75 | 1 | 4 | 0.25 | 2 | [9.0,8.0,5.0] | 7.33333333333 | 8 |

| chb01_03 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [9.0] | 9 | 9 |

| chb01_04 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | [8.0] | 8 | 8 |

| chb01_15 | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | [] | undefined | undefined |

| chb01_16 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | [5.0] | 5 | 5 |



`pooled` event/alarm details and undefined reasons:

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_03` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 2996.0 | 3036.0 | true | 1 | 9.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3005.0 | 6 | 3030.0 | 3035.0 | false | "detection" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_04` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1467.0 | 1494.0 | true | 1 | 8.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1475.0 | 1 | 1475.0 | 1480.0 | false | "detection" | 1 |
| 2 | 1485.0 | 3 | 1495.0 | 1500.0 | false | "repeat" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_15` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1732.0 | 1772.0 | false | null | null |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1785.0 | 1 | 1785.0 | 1790.0 | false | "false_alarm" | null |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`chb01_16` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1015.0 | 1066.0 | true | 1 | 5.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1020.0 | 2 | 1025.0 | 1030.0 | false | "detection" | 1 |
| 2 | 1035.0 | 5 | 1055.0 | 1060.0 | false | "repeat" | 1 |

**undefined_window_metrics**

`{"average_precision": "no continuous scores supplied"}`


`score_policy`:

```json
"Original scores retained in prediction files; AP not evaluated for alert strategies"
```

`alert_rule`:

```json
"output current raw decision"
```

### results/lightgbm_cv_v1/fold_2_chb01_04/weighted_lgbm_evaluation.json — root

Source: [results/lightgbm_cv_v1/fold_2_chb01_04/weighted_lgbm_evaluation.json](../results/lightgbm_cv_v1/fold_2_chb01_04/weighted_lgbm_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 720 | 6 | 0.997222222222 | 1 | 0.666666666667 | 1 | 0.8 | 0.910714285714 | [[714,0],[2,4]] |

| chb01_04 | 720 | 6 | 0.997222222222 | 1 | 0.666666666667 | 1 | 0.8 | 0.910714285714 | [[714,0],[2,4]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [8.0] | 8 | 8 |

| chb01_04 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [8.0] | 8 | 8 |



`chb01_04` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1467.0 | 1494.0 | true | 1 | 8.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1475.0 | 4 | 1490.0 | 1495.0 | false | "detection" | 1 |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_04",
    "label": 0,
    "windows": 714,
    "predicted_positive": 0,
    "mean": 0.0006937680501747988,
    "min": 1.8010086457840012e-07,
    "q25": 2.5305510340511616e-06,
    "median": 2.5759352515432423e-06,
    "q75": 2.8446541568218297e-06,
    "q95": 3.740806149381979e-06,
    "max": 0.368742011006528
  },
  {
    "recording_id": "chb01_04",
    "label": 1,
    "windows": 6,
    "predicted_positive": 4,
    "mean": 0.6844913853387814,
    "min": 0.0220672466761566,
    "q25": 0.3141870881833282,
    "median": 0.9996235459193071,
    "q75": 0.9999086293896373,
    "q95": 0.9999212720524495,
    "max": 0.9999220473668348
  }
]
```

### results/lightgbm_cv_v1/fold_2_chb01_04/ordinary_lgbm_evaluation.json — root

Source: [results/lightgbm_cv_v1/fold_2_chb01_04/ordinary_lgbm_evaluation.json](../results/lightgbm_cv_v1/fold_2_chb01_04/ordinary_lgbm_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 720 | 6 | 0.997222222222 | 0.833333333333 | 0.833333333333 | 0.998599439776 | 0.833333333333 | 0.958333333333 | [[713,1],[1,5]] |

| chb01_04 | 720 | 6 | 0.997222222222 | 0.833333333333 | 0.833333333333 | 0.998599439776 | 0.833333333333 | 0.958333333333 | [[713,1],[1,5]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [8.0] | 8 | 8 |

| chb01_04 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [8.0] | 8 | 8 |



`chb01_04` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1467.0 | 1494.0 | true | 1 | 8.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1475.0 | 6 | 1500.0 | 1505.0 | false | "detection" | 1 |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_04",
    "label": 0,
    "windows": 714,
    "predicted_positive": 1,
    "mean": 0.0013069612713876937,
    "min": 3.7928521907662088e-06,
    "q25": 3.978142328350184e-06,
    "median": 3.978142328350184e-06,
    "q75": 4.108789116671305e-06,
    "q95": 8.157769336556321e-06,
    "max": 0.8070637553090362
  },
  {
    "recording_id": "chb01_04",
    "label": 1,
    "windows": 6,
    "predicted_positive": 5,
    "mean": 0.8346678599421483,
    "min": 0.09687862411388393,
    "q25": 0.9416454920978118,
    "median": 0.992337129889729,
    "q75": 0.9994858698979207,
    "q95": 0.99981849165141,
    "max": 0.9998372130250577
  }
]
```

### results/lightgbm_cv_v1/fold_2_chb01_04/weighted_rf_evaluation.json — root

Source: [results/lightgbm_cv_v1/fold_2_chb01_04/weighted_rf_evaluation.json](../results/lightgbm_cv_v1/fold_2_chb01_04/weighted_rf_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 720 | 6 | 0.995833333333 | 0.8 | 0.666666666667 | 0.998599439776 | 0.727272727273 | 0.930555555556 | [[713,1],[2,4]] |

| chb01_04 | 720 | 6 | 0.995833333333 | 0.8 | 0.666666666667 | 0.998599439776 | 0.727272727273 | 0.930555555556 | [[713,1],[2,4]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | [8.0] | 8 | 8 |

| chb01_04 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | [8.0] | 8 | 8 |



`chb01_04` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1467.0 | 1494.0 | true | 1 | 8.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1475.0 | 2 | 1480.0 | 1485.0 | false | "detection" | 1 |
| 2 | 1490.0 | 3 | 1500.0 | 1505.0 | false | "repeat" | 1 |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_04",
    "label": 0,
    "windows": 714,
    "predicted_positive": 1,
    "mean": 0.0028781512605042018,
    "min": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 0.0,
    "q95": 0.005,
    "max": 0.565
  },
  {
    "recording_id": "chb01_04",
    "label": 1,
    "windows": 6,
    "predicted_positive": 4,
    "mean": 0.7125,
    "min": 0.245,
    "q25": 0.5700000000000001,
    "median": 0.8,
    "q75": 0.9212499999999999,
    "q95": 0.97,
    "max": 0.975
  }
]
```

### results/lightgbm_cv_v1/fold_2_chb01_04/ordinary_rf_evaluation.json — root

Source: [results/lightgbm_cv_v1/fold_2_chb01_04/ordinary_rf_evaluation.json](../results/lightgbm_cv_v1/fold_2_chb01_04/ordinary_rf_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 720 | 6 | 0.997222222222 | 1 | 0.666666666667 | 1 | 0.8 | 1 | [[714,0],[2,4]] |

| chb01_04 | 720 | 6 | 0.997222222222 | 1 | 0.666666666667 | 1 | 0.8 | 1 | [[714,0],[2,4]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | [8.0] | 8 | 8 |

| chb01_04 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | [8.0] | 8 | 8 |



`chb01_04` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1467.0 | 1494.0 | true | 1 | 8.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1475.0 | 1 | 1475.0 | 1480.0 | false | "detection" | 1 |
| 2 | 1485.0 | 3 | 1495.0 | 1500.0 | false | "repeat" | 1 |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_04",
    "label": 0,
    "windows": 714,
    "predicted_positive": 0,
    "mean": 0.0026750700280112048,
    "min": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 0.0,
    "q95": 0.005,
    "max": 0.42
  },
  {
    "recording_id": "chb01_04",
    "label": 1,
    "windows": 6,
    "predicted_positive": 4,
    "mean": 0.6833333333333332,
    "min": 0.47,
    "q25": 0.5225,
    "median": 0.6225,
    "q75": 0.8500000000000001,
    "q95": 0.9575,
    "max": 0.97
  }
]
```

### results/lightgbm_cv_v1/fold_4_chb01_16/weighted_lgbm_evaluation.json — root

Source: [results/lightgbm_cv_v1/fold_4_chb01_16/weighted_lgbm_evaluation.json](../results/lightgbm_cv_v1/fold_4_chb01_16/weighted_lgbm_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 720 | 10 | 0.995833333333 | 0.818181818182 | 0.9 | 0.997183098592 | 0.857142857143 | 0.937373737374 | [[708,2],[1,9]] |

| chb01_16 | 720 | 10 | 0.995833333333 | 0.818181818182 | 0.9 | 0.997183098592 | 0.857142857143 | 0.937373737374 | [[708,2],[1,9]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 2 | 1 | 2 | 0 | [5.0] | 5 | 5 |

| chb01_16 | 1 | 1 | 0 | 1 | 2 | 1 | 2 | 0 | [5.0] | 5 | 5 |



`chb01_16` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1015.0 | 1066.0 | true | 3 | 5.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 870.0 | 1 | 870.0 | 875.0 | false | "false_alarm" | null |
| 2 | 1010.0 | 1 | 1010.0 | 1015.0 | false | "false_alarm" | null |
| 3 | 1020.0 | 9 | 1060.0 | 1065.0 | false | "detection" | 1 |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_16",
    "label": 0,
    "windows": 710,
    "predicted_positive": 2,
    "mean": 0.002799896489461437,
    "min": 1.5653754235154197e-07,
    "q25": 3.2335273524994044e-06,
    "median": 3.2670385846674065e-06,
    "q75": 3.268826424932374e-06,
    "q95": 4.064272498907833e-06,
    "max": 0.9866670547935502
  },
  {
    "recording_id": "chb01_16",
    "label": 1,
    "windows": 10,
    "predicted_positive": 9,
    "mean": 0.8907243354292339,
    "min": 0.002587820110046144,
    "q25": 0.9875602820571373,
    "median": 0.999964150483797,
    "q75": 0.9999731612141554,
    "q95": 0.9999736177317459,
    "max": 0.9999736177317459
  }
]
```

### results/lightgbm_cv_v1/fold_4_chb01_16/ordinary_lgbm_evaluation.json — root

Source: [results/lightgbm_cv_v1/fold_4_chb01_16/ordinary_lgbm_evaluation.json](../results/lightgbm_cv_v1/fold_4_chb01_16/ordinary_lgbm_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 720 | 10 | 0.994444444444 | 0.875 | 0.7 | 0.998591549296 | 0.777777777778 | 0.926241258741 | [[709,1],[3,7]] |

| chb01_16 | 720 | 10 | 0.994444444444 | 0.875 | 0.7 | 0.998591549296 | 0.777777777778 | 0.926241258741 | [[709,1],[3,7]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | [5.0] | 5 | 5 |

| chb01_16 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | [5.0] | 5 | 5 |



`chb01_16` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1015.0 | 1066.0 | true | 2 | 5.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 870.0 | 1 | 870.0 | 875.0 | false | "false_alarm" | null |
| 2 | 1020.0 | 2 | 1025.0 | 1030.0 | false | "detection" | 1 |
| 3 | 1035.0 | 5 | 1055.0 | 1060.0 | false | "repeat" | 1 |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_16",
    "label": 0,
    "windows": 710,
    "predicted_positive": 1,
    "mean": 0.0017916757883488148,
    "min": 3.2980727835360476e-06,
    "q25": 3.3763019836871884e-06,
    "median": 3.3763019836871884e-06,
    "q75": 3.3763019836871884e-06,
    "q95": 9.030775610605816e-06,
    "max": 0.8109453214783341
  },
  {
    "recording_id": "chb01_16",
    "label": 1,
    "windows": 10,
    "predicted_positive": 7,
    "mean": 0.7175461290578795,
    "min": 0.001618489453845766,
    "q25": 0.430990168071696,
    "median": 0.9839063803983931,
    "q75": 0.9998127573581994,
    "q95": 0.9999194951294677,
    "max": 0.9999832394513312
  }
]
```

### results/lightgbm_cv_v1/fold_4_chb01_16/weighted_rf_evaluation.json — root

Source: [results/lightgbm_cv_v1/fold_4_chb01_16/weighted_rf_evaluation.json](../results/lightgbm_cv_v1/fold_4_chb01_16/weighted_rf_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 720 | 10 | 0.994444444444 | 0.8 | 0.8 | 0.997183098592 | 0.8 | 0.965151515152 | [[708,2],[2,8]] |

| chb01_16 | 720 | 10 | 0.994444444444 | 0.8 | 0.8 | 0.997183098592 | 0.8 | 0.965151515152 | [[708,2],[2,8]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 2 | 1 | 2 | 1 | [5.0] | 5 | 5 |

| chb01_16 | 1 | 1 | 0 | 1 | 2 | 1 | 2 | 1 | [5.0] | 5 | 5 |



`chb01_16` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1015.0 | 1066.0 | true | 3 | 5.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 870.0 | 1 | 870.0 | 875.0 | false | "false_alarm" | null |
| 2 | 1010.0 | 1 | 1010.0 | 1015.0 | false | "false_alarm" | null |
| 3 | 1020.0 | 2 | 1025.0 | 1030.0 | false | "detection" | 1 |
| 4 | 1035.0 | 6 | 1060.0 | 1065.0 | false | "repeat" | 1 |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_16",
    "label": 0,
    "windows": 710,
    "predicted_positive": 2,
    "mean": 0.004619718309859155,
    "min": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 0.0,
    "q95": 0.01,
    "max": 0.625
  },
  {
    "recording_id": "chb01_16",
    "label": 1,
    "windows": 10,
    "predicted_positive": 8,
    "mean": 0.7445,
    "min": 0.395,
    "q25": 0.63875,
    "median": 0.8225,
    "q75": 0.8625,
    "q95": 0.9754999999999999,
    "max": 0.98
  }
]
```

### results/lightgbm_cv_v1/fold_4_chb01_16/ordinary_rf_evaluation.json — root

Source: [results/lightgbm_cv_v1/fold_4_chb01_16/ordinary_rf_evaluation.json](../results/lightgbm_cv_v1/fold_4_chb01_16/ordinary_rf_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 720 | 10 | 0.995833333333 | 1 | 0.7 | 1 | 0.823529411765 | 0.966923076923 | [[710,0],[3,7]] |

| chb01_16 | 720 | 10 | 0.995833333333 | 1 | 0.7 | 1 | 0.823529411765 | 0.966923076923 | [[710,0],[3,7]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | [5.0] | 5 | 5 |

| chb01_16 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | [5.0] | 5 | 5 |



`chb01_16` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1015.0 | 1066.0 | true | 1 | 5.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1020.0 | 2 | 1025.0 | 1030.0 | false | "detection" | 1 |
| 2 | 1035.0 | 5 | 1055.0 | 1060.0 | false | "repeat" | 1 |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_16",
    "label": 0,
    "windows": 710,
    "predicted_positive": 0,
    "mean": 0.005309859154929578,
    "min": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 0.0,
    "q95": 0.025,
    "max": 0.275
  },
  {
    "recording_id": "chb01_16",
    "label": 1,
    "windows": 10,
    "predicted_positive": 7,
    "mean": 0.637,
    "min": 0.2,
    "q25": 0.46625,
    "median": 0.71,
    "q75": 0.85375,
    "q95": 0.90325,
    "max": 0.91
  }
]
```

### results/lightgbm_cv_v1/fold_3_chb01_15/weighted_lgbm_evaluation.json — root

Source: [results/lightgbm_cv_v1/fold_3_chb01_15/weighted_lgbm_evaluation.json](../results/lightgbm_cv_v1/fold_3_chb01_15/weighted_lgbm_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 720 | 8 | 0.9875 | 0 | 0 | 0.998595505618 | 0 | 0.265673144258 | [[711,1],[8,0]] |

| chb01_15 | 720 | 8 | 0.9875 | 0 | 0 | 0.998595505618 | 0 | 0.265673144258 | [[711,1],[8,0]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | [] | undefined | undefined |

| chb01_15 | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | [] | undefined | undefined |



`chb01_15` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1732.0 | 1772.0 | false | null | null |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1785.0 | 1 | 1785.0 | 1790.0 | false | "false_alarm" | null |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_15",
    "label": 0,
    "windows": 712,
    "predicted_positive": 1,
    "mean": 0.0018076457891107796,
    "min": 9.545356254775355e-07,
    "q25": 3.4394820818615265e-06,
    "median": 3.519929777171339e-06,
    "q75": 3.5258457450341693e-06,
    "q95": 2.6503313831876014e-05,
    "max": 0.9977882405768423
  },
  {
    "recording_id": "chb01_15",
    "label": 1,
    "windows": 8,
    "predicted_positive": 0,
    "mean": 0.00044028773537497243,
    "min": 1.972482246284906e-05,
    "q25": 9.150012695168145e-05,
    "median": 0.00014672151841385172,
    "q75": 0.0009133475789227082,
    "q95": 0.0010972540410584108,
    "max": 0.0011502079744589665
  }
]
```

### results/lightgbm_cv_v1/fold_3_chb01_15/ordinary_lgbm_evaluation.json — root

Source: [results/lightgbm_cv_v1/fold_3_chb01_15/ordinary_lgbm_evaluation.json](../results/lightgbm_cv_v1/fold_3_chb01_15/ordinary_lgbm_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 720 | 8 | 0.991666666667 | 0.75 | 0.375 | 0.998595505618 | 0.5 | 0.649176864802 | [[711,1],[5,3]] |

| chb01_15 | 720 | 8 | 0.991666666667 | 0.75 | 0.375 | 0.998595505618 | 0.5 | 0.649176864802 | [[711,1],[5,3]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | [8.0] | 8 | 8 |

| chb01_15 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | [8.0] | 8 | 8 |



`chb01_15` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1732.0 | 1772.0 | true | 1 | 8.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1740.0 | 1 | 1740.0 | 1745.0 | false | "detection" | 1 |
| 2 | 1765.0 | 2 | 1770.0 | 1775.0 | false | "repeat" | 1 |
| 3 | 1785.0 | 1 | 1785.0 | 1790.0 | false | "false_alarm" | null |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_15",
    "label": 0,
    "windows": 712,
    "predicted_positive": 1,
    "mean": 0.0016124589757977252,
    "min": 3.6973390994026746e-06,
    "q25": 3.708225816491311e-06,
    "median": 3.708225816491311e-06,
    "q75": 3.7083858371695566e-06,
    "q95": 6.649616032778236e-06,
    "max": 0.963958583863989
  },
  {
    "recording_id": "chb01_15",
    "label": 1,
    "windows": 8,
    "predicted_positive": 3,
    "mean": 0.3258432252280634,
    "min": 0.0017820856829717452,
    "q25": 0.003835438858552544,
    "median": 0.15723494106297925,
    "q75": 0.7512538193280992,
    "q95": 0.7701099092871914,
    "max": 0.778164417628478
  }
]
```

### results/lightgbm_cv_v1/fold_3_chb01_15/weighted_rf_evaluation.json — root

Source: [results/lightgbm_cv_v1/fold_3_chb01_15/weighted_rf_evaluation.json](../results/lightgbm_cv_v1/fold_3_chb01_15/weighted_rf_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 720 | 8 | 0.9875 | 0 | 0 | 0.998595505618 | 0 | 0.271631000578 | [[711,1],[8,0]] |

| chb01_15 | 720 | 8 | 0.9875 | 0 | 0 | 0.998595505618 | 0 | 0.271631000578 | [[711,1],[8,0]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | [] | undefined | undefined |

| chb01_15 | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | [] | undefined | undefined |



`chb01_15` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1732.0 | 1772.0 | false | null | null |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1785.0 | 1 | 1785.0 | 1790.0 | false | "false_alarm" | null |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_15",
    "label": 0,
    "windows": 712,
    "predicted_positive": 1,
    "mean": 0.0035042134831460674,
    "min": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 0.0,
    "q95": 0.005,
    "max": 0.68
  },
  {
    "recording_id": "chb01_15",
    "label": 1,
    "windows": 8,
    "predicted_positive": 0,
    "mean": 0.06125,
    "min": 0.01,
    "q25": 0.015,
    "median": 0.030000000000000002,
    "q75": 0.07125000000000001,
    "q95": 0.178,
    "max": 0.185
  }
]
```

### results/lightgbm_cv_v1/fold_3_chb01_15/ordinary_rf_evaluation.json — root

Source: [results/lightgbm_cv_v1/fold_3_chb01_15/ordinary_rf_evaluation.json](../results/lightgbm_cv_v1/fold_3_chb01_15/ordinary_rf_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 720 | 8 | 0.9875 | 0 | 0 | 0.998595505618 | 0 | 0.440566378066 | [[711,1],[8,0]] |

| chb01_15 | 720 | 8 | 0.9875 | 0 | 0 | 0.998595505618 | 0 | 0.440566378066 | [[711,1],[8,0]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | [] | undefined | undefined |

| chb01_15 | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | [] | undefined | undefined |



`chb01_15` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 1732.0 | 1772.0 | false | null | null |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1785.0 | 1 | 1785.0 | 1790.0 | false | "false_alarm" | null |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_15",
    "label": 0,
    "windows": 712,
    "predicted_positive": 1,
    "mean": 0.0032865168539325845,
    "min": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 0.0,
    "q95": 0.005,
    "max": 0.52
  },
  {
    "recording_id": "chb01_15",
    "label": 1,
    "windows": 8,
    "predicted_positive": 0,
    "mean": 0.204375,
    "min": 0.0,
    "q25": 0.06625,
    "median": 0.2175,
    "q75": 0.3375,
    "q95": 0.388,
    "max": 0.395
  }
]
```

### results/lightgbm_cv_v1/fold_1_chb01_03/weighted_lgbm_evaluation.json — root

Source: [results/lightgbm_cv_v1/fold_1_chb01_03/weighted_lgbm_evaluation.json](../results/lightgbm_cv_v1/fold_1_chb01_03/weighted_lgbm_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 720 | 8 | 0.995833333333 | 0.777777777778 | 0.875 | 0.997191011236 | 0.823529411765 | 0.766044829768 | [[710,2],[1,7]] |

| chb01_03 | 720 | 8 | 0.995833333333 | 0.777777777778 | 0.875 | 0.997191011236 | 0.823529411765 | 0.766044829768 | [[710,2],[1,7]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 0 | [9.0] | 9 | 9 |

| chb01_03 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 0 | [9.0] | 9 | 9 |



`chb01_03` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 2996.0 | 3036.0 | true | 1 | 9.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3005.0 | 8 | 3040.0 | 3045.0 | false | "detection" | 1 |
| 2 | 3050.0 | 1 | 3050.0 | 3055.0 | false | "false_alarm" | null |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_03",
    "label": 0,
    "windows": 712,
    "predicted_positive": 2,
    "mean": 0.0037425080519334977,
    "min": 6.989236394109283e-07,
    "q25": 3.678876718276149e-06,
    "median": 3.678876718276149e-06,
    "q75": 3.678876718276149e-06,
    "q95": 1.024349707643529e-05,
    "max": 0.999908609751697
  },
  {
    "recording_id": "chb01_03",
    "label": 1,
    "windows": 8,
    "predicted_positive": 7,
    "mean": 0.8627647989649286,
    "min": 3.678876718276149e-06,
    "q25": 0.9741171769455097,
    "median": 0.9981422608649468,
    "q75": 0.9990587749600561,
    "q95": 0.9999520385356395,
    "max": 0.999952040314352
  }
]
```

### results/lightgbm_cv_v1/fold_1_chb01_03/ordinary_lgbm_evaluation.json — root

Source: [results/lightgbm_cv_v1/fold_1_chb01_03/ordinary_lgbm_evaluation.json](../results/lightgbm_cv_v1/fold_1_chb01_03/ordinary_lgbm_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 720 | 8 | 0.997222222222 | 1 | 0.75 | 1 | 0.857142857143 | 0.838974926254 | [[712,0],[2,6]] |

| chb01_03 | 720 | 8 | 0.997222222222 | 1 | 0.75 | 1 | 0.857142857143 | 0.838974926254 | [[712,0],[2,6]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [9.0] | 9 | 9 |

| chb01_03 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [9.0] | 9 | 9 |



`chb01_03` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 2996.0 | 3036.0 | true | 1 | 9.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3005.0 | 6 | 3030.0 | 3035.0 | false | "detection" | 1 |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_03",
    "label": 0,
    "windows": 712,
    "predicted_positive": 0,
    "mean": 3.8915298509860964e-05,
    "min": 1.0325825305472953e-06,
    "q25": 3.695520318817739e-06,
    "median": 3.695520318817739e-06,
    "q75": 3.695520318817739e-06,
    "q95": 3.902028254457448e-06,
    "max": 0.018069491932580926
  },
  {
    "recording_id": "chb01_03",
    "label": 1,
    "windows": 8,
    "predicted_positive": 6,
    "mean": 0.7076121678691704,
    "min": 3.695520318817739e-06,
    "q25": 0.5819421254103027,
    "median": 0.9425855583364593,
    "q75": 0.9998129250465922,
    "q95": 0.9998228344687861,
    "max": 0.9998245939186723
  }
]
```

### results/lightgbm_cv_v1/fold_1_chb01_03/weighted_rf_evaluation.json — root

Source: [results/lightgbm_cv_v1/fold_1_chb01_03/weighted_rf_evaluation.json](../results/lightgbm_cv_v1/fold_1_chb01_03/weighted_rf_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 720 | 8 | 0.997222222222 | 0.875 | 0.875 | 0.998595505618 | 0.875 | 0.876388888889 | [[711,1],[1,7]] |

| chb01_03 | 720 | 8 | 0.997222222222 | 0.875 | 0.875 | 0.998595505618 | 0.875 | 0.876388888889 | [[711,1],[1,7]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 0 | [9.0] | 9 | 9 |

| chb01_03 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 0 | [9.0] | 9 | 9 |



`chb01_03` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 2996.0 | 3036.0 | true | 1 | 9.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3005.0 | 7 | 3035.0 | 3040.0 | false | "detection" | 1 |
| 2 | 3050.0 | 1 | 3050.0 | 3055.0 | false | "false_alarm" | null |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_03",
    "label": 0,
    "windows": 712,
    "predicted_positive": 1,
    "mean": 0.004199438202247191,
    "min": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 0.0,
    "q95": 0.005,
    "max": 0.725
  },
  {
    "recording_id": "chb01_03",
    "label": 1,
    "windows": 8,
    "predicted_positive": 7,
    "mean": 0.78375,
    "min": 0.0,
    "q25": 0.82125,
    "median": 0.855,
    "q75": 0.96375,
    "q95": 0.99,
    "max": 0.99
  }
]
```

### results/lightgbm_cv_v1/fold_1_chb01_03/ordinary_rf_evaluation.json — root

Source: [results/lightgbm_cv_v1/fold_1_chb01_03/ordinary_rf_evaluation.json](../results/lightgbm_cv_v1/fold_1_chb01_03/ordinary_rf_evaluation.json) → `root`.

| Scope | n_windows | n_positive | accuracy | precision | recall | specificity | f1 | average_precision | confusion_matrix |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 720 | 8 | 0.997222222222 | 1 | 0.75 | 1 | 0.857142857143 | 0.860763888889 | [[712,0],[2,6]] |

| chb01_03 | 720 | 8 | 0.997222222222 | 1 | 0.75 | 1 | 0.857142857143 | 0.860763888889 | [[712,0],[2,6]] |



| Scope | n_seizures | n_detected | n_missed | event_sensitivity | n_false_alarms | evaluated_hours | false_alarms_per_hour | n_repeat_alarms | latencies_sec | mean_latency_sec | median_latency_sec |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| pooled | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [9.0] | 9 | 9 |

| chb01_03 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | [9.0] | 9 | 9 |



`chb01_03` event/alarm details and undefined reasons:

**seizures**

| seizure_id | onset_sec | offset_sec | detected | alarm_id | latency_sec |
| --- | --- | --- | --- | --- | --- |
| 1 | 2996.0 | 3036.0 | true | 1 | 9.0 |

**alarms**

| alarm_id | start_sec | positive_windows | last_positive_decision_sec | end_sec | right_censored | status | seizure_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3005.0 | 6 | 3030.0 | 3035.0 | false | "detection" | 1 |


`score_distribution`:

```json
[
  {
    "recording_id": "chb01_03",
    "label": 0,
    "windows": 712,
    "predicted_positive": 0,
    "mean": 0.001734550561797753,
    "min": 0.0,
    "q25": 0.0,
    "median": 0.0,
    "q75": 0.0,
    "q95": 0.005,
    "max": 0.215
  },
  {
    "recording_id": "chb01_03",
    "label": 1,
    "windows": 8,
    "predicted_positive": 6,
    "mean": 0.571875,
    "min": 0.0,
    "q25": 0.45000000000000007,
    "median": 0.6625,
    "q75": 0.81,
    "q95": 0.883,
    "max": 0.89
  }
]
```

## Appendix C. Historical report evidence by phase

The following are attributed transcriptions of the repository reports, in research order. They preserve numerical observations, limitations and contemporaneous recommendations. Statements such as “next phase” describe the time of that report, not current status. Code blocks are omitted to avoid presenting historical execution instructions as new work. Source headings are retained at a lower level; links in the originals remain authoritative.

### Source report: dataset_preparation

Source: [docs/dataset_preparation.md](../docs/dataset_preparation.md); original section names appear below.

#### Phase 2A: Patient 01 dataset and fixed recording splits

This step prepares data only. The test recordings are reserved for final
evaluation: preprocessing and annotation/integrity checks are permitted, but
model training, threshold selection, feature selection, and comparison must not
use them. The manifest is fixed before model development; class counts do not
change its assignments.

#### Files and roles

- `config/chb01_recordings.csv`: one row per recording with filename, assigned
  split (`train`, `validation`, `test`), and annotation source relative to the
  repository root. Duplicate filenames fail, including duplicates in one split.
- `src/preprocess.py`: existing, unchanged causal feature extraction CLI.
- `src/split_dataset.py`: reads only manifest EDFs for integrity checks, verifies
  the expanded features, then adds a `split` column by filename. It neither
  shuffles nor filters rows. It rejects missing/extra recordings, ambiguous IDs,
  duplicate windows, invalid labels/times, nonfinite features, and channel/feature
  mismatches. A new output path is mandatory to preserve previous datasets.
- `tests/test_split_dataset.py`: fixed manifest membership, split safeguards,
  malformed/truncated inputs, independent label checks, and the 465-window short
  recording. An integration test checks the generated expanded artifacts when
  available; it is explicitly skipped if they have not yet been generated.

New outputs:

1. `data/processed/chb01_detection_features_11recordings.parquet` and `.json`:
   expanded unsplit features and existing preprocessing provenance format.
2. `data/processed/chb01_detection_features_11recordings_split.parquet` and `.json`:
   identical rows/features plus `split`; verification report includes raw hashes,
   annotation sources/intervals, recording counts, split counts, feature names,
   warnings, manifest hash, and input/output hashes.
3. `data/processed/chb01_phase2a_preservation.json`: before/after hashes for the
   raw recordings, original datasets, and existing preprocessing/evaluation code
   and tests, captured during the initial Phase 2A run.

#### Exact reproduction commands

Run from the repository root with the existing ARM64 virtual environment:

[Historical code block omitted; see linked source.]

These output names are already present after the initial run. To preserve that
run when reproducing it, choose new `--output` filenames for both commands and
pass the new preprocessing filename to `--features`. The split CLI refuses to
overwrite either an existing Parquet or its JSON report. The preprocessing CLI
retains its existing behavior of replacing the explicitly named detection output;
never point it at a historical dataset you want to keep. All paths in the CSV
are resolved relative to the repository root, independent of the CSV's directory.
No new dependencies are required.

For rechecking this run's stored outputs without rewriting them:

[Historical code block omitted; see linked source.]

#### Verified recording inventory

All 11 files have 256 Hz sampling, 23 raw channels in identical order, and the
same 18 selected bipolar channels. EDF start clocks and duration match the local
`chb01-summary.txt`; EDF byte lengths agree with their declared payload lengths.
Each full selected signal was read by preprocessing. Raw checks additionally
probe the first/last samples, validate annotation ordering and bounds, and compare
SHA-256 hashes to preprocessing provenance. These are local integrity checks,
not a comparison against independently downloaded PhysioNet checksums.

| Recording | Split | Duration (s) | Windows | Class 0 | Class 1 | Annotated seizure interval (s) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| chb01_01 | train | 3600 | 720 | 720 | 0 | none |
| chb01_03 | train | 3600 | 720 | 712 | 8 | 2996–3036 |
| chb01_04 | train | 3600 | 720 | 714 | 6 | 1467–1494 |
| chb01_14 | train | 3600 | 720 | 720 | 0 | none |
| chb01_15 | train | 3600 | 720 | 712 | 8 | 1732–1772 |
| chb01_16 | train | 3600 | 720 | 710 | 10 | 1015–1066 |
| chb01_17 | validation | 3600 | 720 | 720 | 0 | none |
| chb01_18 | validation | 3600 | 720 | 702 | 18 | 1720–1810 |
| chb01_21 | test | 3600 | 720 | 701 | 19 | 327–420 |
| chb01_22 | test | 3600 | 720 | 720 | 0 | none |
| chb01_26 | test | 2325 | 465 | 444 | 21 | 1862–1963 |

`chb01_26` contains 595,200 samples and lasts 38 minutes 45 seconds. Its last
window is `[2320, 2325)`. No selected recording has a partial trailing window;
the existing evaluator's complete-coverage contract therefore needs no change.
Do not concatenate across file boundaries or count acquisition gaps as EEG time.

#### Verified split totals

| Split | Recordings | Class 0 | Class 1 | Total windows | Distinct annotated seizures | EEG duration (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 6 | 4288 | 32 | 4320 | 4 | 21600 |
| Validation | 2 | 1422 | 18 | 1440 | 1 | 7200 |
| Test | 3 | 1865 | 40 | 1905 | 2 | 9525 |
| Total | 11 | 7575 | 90 | 7665 | 7 | 38325 |

The dataset contains 10 hours 38 minutes 45 seconds of EEG. Ictal windows are
1.1742% of the total. Seizure counts come from separate onset/offset annotations,
not runs of positive window labels. Separate events in one patient do not prove
statistical independence, and windows from an event are not independent events.

#### Later model handoff

[Historical code block omitted; see linked source.]

`split`, annotations, quality flags, and recording/timestamp columns are metadata,
not model inputs. Use the existing evaluation handoff with the preprocessing
JSON's exact intervals and per-recording durations. No model development has
been performed in Phase 2A.

#### Warnings and limitations

The only preprocessing warnings are MNE's duplicate `T8-P8` name and the existing
explicit selection of `T8-P8-0`. They are retained in both reports. No NaN/infinite
features, duplicate windows, or near-zero-power channels were found. The first
two windows of each recording (22 total) retain the filter warmup flag; they are
not excluded. Every window retains the original preprocessing metadata.

This is a one-patient study with four training seizures, one validation seizure,
and two final test seizures. Threshold tuning on one event is unstable, and final
event sensitivity will change in 50-percentage-point steps. Test EEG contains
less than three hours, so false-alarm rates will also be imprecise. These limits
must remain visible in later results. The fixed split supplies an honest small
case study, not evidence for generalization to other patients.


### Source report: evaluation

Source: [docs/evaluation.md](../docs/evaluation.md); original section names appear below.

#### Evaluation contract

`src/evaluate.py` evaluates supplied predictions in memory. It performs no file
discovery, EDF access, model fitting, threshold tuning, or temporal filtering.
Tests use synthetic arrays only:

[Historical code block omitted; see linked source.]

#### Public functions

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

#### Window metrics

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

#### Alarm episode and matching rules

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

#### Exposure, recording boundaries, and input validation

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

#### Model handoff

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

[Historical code block omitted; see linked source.]

The preprocessing JSON keys its recording entries by **filename**. For future
model output on an already processed, held-out table `test_df`, build the handoff
as follows (this is illustrative, not a training command):

[Historical code block omitted; see linked source.]

Use only feature columns for model inputs; annotations and labels belong only in
evaluation. The caller is responsible for holding out whole recordings and for
causal prediction generation. The evaluator cannot prove a model avoided future
information. It does not check that supplied labels agree with annotations.
Its decision-time latency includes window accumulation, but does not include
model runtime, transmission, or display delay.

#### Runnable synthetic example

[Historical code block omitted; see linked source.]

These deliberately tiny examples verify arithmetic, not detector performance.
With five-second decisions, a short seizure may end before any decision occurs;
it then counts as missed even if a positive window overlaps it. Window recall
and event sensitivity can therefore disagree. Results depend on the stated
episode/matching rules, and small numbers of independent seizures cannot support
strong generalization claims. Repeated alarms and sustained false-alarm duration
should be inspected in addition to false alarm counts. Temporal filters and
alternative matching tolerances require separate, predeclared comparisons later.


### Source report: baseline_experiment

Source: [docs/baseline_experiment.md](../docs/baseline_experiment.md); original section names appear below.

#### Phase 2B: initial DummyClassifier and Random Forest baselines

The fixed Random Forest detected 11/18 validation seizure windows and the one
validation seizure, with a 10-second onset-to-alarm latency. There were no false
positive windows. This is a small, patient-specific development result, not a
final test result. No threshold or hyperparameter search was performed.

#### Scope, inputs and leakage safeguards

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

#### Fixed model configurations

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

#### Actual validation results

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

[Historical code block omitted; see linked source.]

Dummy accuracy is `(TN + TP)/N = (1422 + 0)/1440 = 98.75%`, despite recall
`0/18 = 0` and event detection `0/1 = 0`. Predicting the majority class is enough
to achieve high accuracy here. Its constant zero scores give AP equal to
validation prevalence, 0.0125. AP is the evaluator's step-weighted precision-recall
summary, not trapezoidal PR area. Undefined precision/latency remain JSON `null`.

The forest improves accuracy by only 0.7639 percentage points, while detecting
11 seizure windows that the dummy misses. Accuracy alone conceals this difference.

#### Failure analysis: observations

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

[Figure: Validation scores and exact seizure interval; see Appendix E.]

The observed misses could reflect sparse training examples, differences in seizure
appearance, or the limits of these features and model. These are **hypotheses**;
this experiment does not identify their cause or demonstrate that class imbalance
alone caused them. In particular, the forest did not collapse to an all-negative
classifier, so the result should not be described that way.

#### Evaluation definitions and limitations

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

#### Artifacts and reproduction

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

[Historical code block omitted; see linked source.]

`MPLCONFIGDIR` just places matplotlib's font cache in a writable location. No
dependencies were installed. The runner refuses to overwrite an existing output
directory. Models are reloaded and their complete validation predictions/scores
compared to the originals during every run. Preserve the saved library versions
for reproduction across environments; sklearn model loading is version-sensitive.

To re-evaluate the saved predictions without fitting any model:

[Historical code block omitted; see linked source.]

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


### Source report: weighted_experiment

Source: [docs/weighted_experiment.md](../docs/weighted_experiment.md); original section names appear below.

#### Phase 2C: controlled class-weight experiment

The weighted forest detected the single validation seizure five seconds earlier,
but detected fewer ictal windows and had lower average precision. No false
positives occurred for either forest. These observations do not establish an
overall improvement from class weighting.

#### Controlled comparison

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

#### Combined validation results

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

[Historical code block omitted; see linked source.]

AP is the unchanged evaluator's step-weighted PR summary, not trapezoidal area.
Pooled AP uses all validation scores and does not average recording APs.

#### Per-recording results

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

#### Prediction changes

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

[Figure: Validation comparison; see Appendix E.]

Markers show window-end decisions. Connecting lines are visual guides, not
additional decisions, interpolation used by the detector, or temporal filtering.

#### Scientific interpretation

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

#### Files, tests and reproduction

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

[Historical code block omitted; see linked source.]

Existing output directories are refused. `MPLCONFIGDIR` places the plot font
cache in a writable location. Reproduce the suite with its temporary exclusion:

[Historical code block omitted; see linked source.]

No threshold search, temporal filtering, LightGBM, EEGNet, dashboard or final test
evaluation was performed. This phase stops at the controlled comparison.


### Source report: rf_cross_validation

Source: [docs/rf_cross_validation.md](../docs/rf_cross_validation.md); original section names appear below.

#### Phase 2D: training-only recording-level RF stability

Both forests detected the same three of four held-out seizures, missed
`chb01_15`, and had identical first detection delays. Weighting increased pooled
window recall from 53.125% to 59.375%, but increased false-alarm episodes from
one to four and reduced macro F1 and AP. This is exploratory recording-level
stability analysis within one patient, not prospective chronological validation.

#### Methodology and scope

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

#### Fold definitions and observed results

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

#### Definitions and interpretation

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

#### Files, verification and reproduction

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

[Historical code block omitted; see linked source.]

Existing output directories are refused. The baseline configuration freezes
settings; previous model files and scores are not loaded. No dependencies were
installed or changes committed/pushed. Reproduce the protected full suite with:

[Historical code block omitted; see linked source.]


### Source report: failure_analysis

Source: [docs/failure_analysis.md](../docs/failure_analysis.md); original section names appear below.

#### Phase 2E: why chb01_15 was missed

Both saved out-of-fold forests assign subthreshold scores throughout the
annotated seizure `[1732,1772)` seconds. Their only positive window is
`[1780,1785)`, fully after offset, and their only alarm starts at its end,
1785 seconds: **13 seconds after offset and 53 seconds after onset**. The
models respond during the seizure but peak afterward. The late response
coincides with a transient high-frequency/line-length increase; this is a clue,
not proof of a causal feature mechanism or of an EEG artifact.

#### Inspected material and verification

Reviewed the Git status, existing CV implementation/report, split metadata,
fold definitions, all eight saved fold models, prediction tables, evaluator
reports and exact annotations. Existing training-only loading reads only the
original six training recordings from the processed dataset. Saved predictions
match original window identities/times/labels exactly, reproduce from all eight
loaded models, and reproduce the original evaluation reports. No models were
fitted, no thresholds or feature definitions changed, and no previous outputs
were rewritten. Original validation/test recordings and new `chb01_23`–`25`
were not opened or used. No EDF was opened for this analysis.

#### Scores and delayed alarms

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

#### Comparison with detected seizures

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

#### EEG feature comparisons

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

#### Postictal feature burst and signal quality

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

#### Figures and limitations

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

#### Reproduction and checks

Added `src/failure_analysis.py`, four focused synthetic tests in
`tests/test_failure_analysis.py`, this report and a README link. Analysis outputs
are separate under `results/failure_analysis_v1/`. Existing experiments and
uncommitted work are preserved. No dependencies installed, commits or pushes.

[Historical code block omitted; see linked source.]

Use a fresh output directory; overwrite is refused. All 29 analysis/evaluation
tests passed (4 new + 25 existing). Training tests were not run in this
analysis-only phase. Eight saved model predictions and evaluator reports were
verified. The analysis fits no model or learned transformation and retains all
existing feature, label, threshold and event definitions. Artifact hashes,
verification, tests and a before/after preservation audit accompany the outputs.


### Source report: raw_eeg_investigation

Source: [docs/raw_eeg_investigation.md](../docs/raw_eeg_investigation.md); original section names appear below.

#### Phase 2F: targeted raw EEG investigation

The delayed-alarm window contains a conspicuous fast burst in the original
EEG, strongest in FP2–F8, F8–T8 and T8–P8. It survives the established causal
filter. Absolute 30–45 Hz power increases sharply, confirming that the relative
gamma spike is not merely a denominator effect. Its physiological versus
artifactual origin remains unresolved. No software or label error was found
in the targeted checks, and the missed seizure remains missed.

#### Scope and methods

Reviewed the previous failure analysis, preprocessing, saved fold-3 definition,
models, predictions and evaluation reports. Only EDFs `chb01_15`, `03`, `04`,
and `16` were opened, as authorized for the target investigation and comparison.
Original validation/test EDFs and supplementary `23`–`25` were not accessed.
Existing processed data were loaded with the training-only filter. No models
were trained or altered, and no cleaning, re-referencing, artifact removal,
notch filtering, feature/label changes or threshold searches were performed.

MNE reads EDF signals in volts; values are multiplied by 1e6 to yield µV.
All four files have 256 Hz sampling and the established 18 bipolar signals.
MNE names the duplicate T8–P8 entries `T8-P8-0` and `T8-P8-1`; canonical
T8–P8 uses the first, exactly as in preprocessing. The saved channel order and
raw-file SHA-256 hashes match preprocessing provenance. The duplicate-name
warnings are expected, documented behavior, not newly discovered mapping errors.

Plots inspect `chb01_15` at 1700–1810 s and a 1775–1790 s zoom. To reproduce
causal filter state, each authorized recording is read from sample zero through
the needed stop time, then filtered forward with the existing fourth-order
Butterworth 0.5–45 Hz SOS design and zero initial state. Earlier samples are
used only for state; they are not part of the visual or quantitative investigation.
Filtering a crop with fresh state would not reproduce the established pipeline.
No samples after the requested stop time are needed. Waveforms show ten fixed
representative channels, including the right lateral chain and left/midline
comparators, with all samples plotted and no per-channel normalization. Each
trace uses the same µV scale, 1200 µV vertical separation and a 500 µV scale bar.
All 18 selected channels are measured numerically.

Each spectral measurement uses a five-second window, Welch density with 2 s
Hann segments, 1 s overlap and constant detrending, exactly as preprocessing.
Frequency spacing is 0.5 Hz. Density units are µV²/Hz; band powers are sums of
included bins multiplied by 0.5 Hz, in µV². Total is [0.5,45] Hz; gamma is
[30,45] Hz; other power is total minus gamma, [0.5,30) Hz. Total here is this
analysis-band total, not all-frequency EEG power. PSD plots display 0.5–100 Hz
with common logarithmic limits 1e-3–1e5 µV²/Hz, while feature calculations still
end at 45 Hz. Values below the plotting floor are simply outside the displayed
range; no data are cleaned or altered. Raw PSDs receive the same detrending
settings but no bandpass. Filtered PSDs include the established filter attenuation.

#### Observations in the original waveform

The annotated interval `[1732,1772)` has visible amplitude and rhythmic waveform
changes relative to nearby pre-onset EEG. In the mid-event exemplar, FP2–F8 and
T8–P8 show relatively slow, rounded oscillations with much less superimposed
fast texture than the selected detected-seizure exemplars. The missed event is
not flat or uniformly low-amplitude. These are descriptive observations, not
an independent clinical confirmation or redefinition of seizure boundaries.

Around 1782–1784 s, within `[1780,1785)`, a short, higher-frequency burst is
visible in FP2–F8, F8–T8 and T8–P8 in both raw and filtered panels. Its amplitude
envelope changes rapidly. Smaller changes occur elsewhere; it is neither
limited to one signal nor comparably strong across all channels. FP2–F8 raw
peak-to-peak amplitude rises from 481.76 µV in the mid-ictal window to 865.84 µV
in the alarm window; filtered values are 490.92 and 676.73 µV. No obvious
recording-wide flat segment or plateau is apparent in these displayed crops,
but this is not an exhaustive clipping or signal-quality test.

#### Absolute versus relative gamma

Selected filtered measurements:

| Channel | Window (s) | Total 0.5–45 (µV²) | Gamma 30–45 (µV²) | Relative gamma | Line length (µV sum) |
| --- | --- | ---: | ---: | ---: | ---: |
| FP2–F8 | pre [1725,1730) | 4837.31 | 1.66 | 0.000343 | 4601.08 |
| FP2–F8 | ictal [1750,1755) | 9970.44 | 2.21 | 0.000222 | 7761.18 |
| FP2–F8 | post [1775,1780) | 5125.76 | 16.97 | 0.003311 | 7629.50 |
| FP2–F8 | alarm [1780,1785) | 7662.54 | 1690.96 | 0.220679 | 26756.50 |
| FP2–F8 | post [1785,1790) | 5733.41 | 2.92 | 0.000509 | 6068.33 |
| T8–P8 | pre [1725,1730) | 2515.39 | 1.43 | 0.000570 | 4622.17 |
| T8–P8 | ictal [1750,1755) | 18317.06 | 6.84 | 0.000374 | 11765.77 |
| T8–P8 | post [1775,1780) | 5206.69 | 26.71 | 0.005129 | 8051.25 |
| T8–P8 | alarm [1780,1785) | 6912.68 | 1633.60 | 0.236319 | 27401.71 |
| T8–P8 | post [1785,1790) | 10306.62 | 152.85 | 0.014830 | 12563.59 |

For FP2–F8, alarm/mid-ictal absolute gamma rises about **764-fold**, while total
power falls about 23%. Other-band power falls from 9968.23 to 5971.58 µV²,
which further raises the gamma fraction, but cannot explain the large increase
in absolute gamma. Compared with the immediately preceding postictal window,
both gamma and total power increase: approximately 100-fold and 49.5%, respectively.
Thus the effect is not solely loss of power in other bands. T8–P8 gamma rises
about 239-fold relative to the mid-ictal exemplar, while total power falls 62.3%.
These ratios use the explicitly selected five-second windows, not whole-event
medians from the previous analysis.

The finding is already present before filtering: raw FP2–F8 gamma is 2.64 µV²
mid-ictally and 2085.54 µV² in the alarm window; raw T8–P8 gamma is 7.68 versus
1976.70 µV². The burst is therefore not created solely by the causal filter.
The raw spectrum shows power spread across a broad higher-frequency range,
including beyond 45 Hz, rather than only a change in the gamma denominator.
No additional spectral measure is used to change model inputs.

The strongest alarm-window filtered absolute gamma powers are FP2–F8 1690.96,
T8–P8 1633.60, F8–T8 1163.92 and FP2–F4 307.36 µV². For comparison, FZ–CZ and
CZ–PZ have 6.54 and 5.88 µV². These are overlapping bipolar derivations sharing
electrodes, not independent spatial measurements or a demonstrated clinical focus.
All channel measurements are retained in `window_measurements.parquet`.

#### Context from detected seizures

Representative windows are selected mechanically: the five-second grid window
containing each annotated event's temporal midpoint. They are not selected by
maximum score or by searching for a favorable waveform. All use identical units,
filter state handling, Welch settings and amplitude display limits.

| Recording, window | FP2–F8 line length / gamma / peak-to-peak | T8–P8 line length / gamma / peak-to-peak |
| --- | --- | --- |
| missed 15, [1750,1755) | 7761.18 / 2.21 / 490.92 | 11765.77 / 6.84 / 586.56 |
| detected 03, [3015,3020) | 45852.62 / 1902.89 / 682.20 | 30163.86 / 530.27 / 628.65 |
| detected 04, [1480,1485) | 25672.50 / 668.51 / 702.47 | 41883.25 / 1961.70 / 612.29 |
| detected 16, [1040,1045) | 29256.31 / 874.50 / 706.76 | 49609.11 / 2597.57 / 665.76 |

All entries are filtered; units are µV sum / µV² / µV. The missed exemplar has
lower line length and gamma, while amplitudes overlap more closely, particularly
T8–P8. Detected examples have more visible rapid components superimposed on
slower activity. One five-second exemplar cannot characterize an entire seizure;
these observations support, rather than prove a mechanism for, the previous
whole-event feature differences. Annotations remain exactly `[2996,3036)`,
`[1467,1494)`, `[1732,1772)`, `[1015,1066)` for 03, 04, 15 and 16.

#### Technical consistency and model behavior

All 126 features reproduce **exactly, maximum absolute error 0.0**, for 22 target
windows at 1700–1810 s and two context windows in each comparison recording:
28 five-second windows total. Checks cover channel mapping, µV conversion,
integer sample starts/ends, causal filter state, window length and Welch ratios.
The original raw hashes and selected channel names match preprocessing provenance.
Synthetic tests verify known absolute powers, amplitude/unit scaling invariance
of ratios, and equivalence of a filtered causal prefix to a full-file crop.

Only the saved fold-3 models, trained without `chb01_15`, are used to verify its
predictions. Full-recording predictions and scores reproduce the saved tables,
and evaluation reproduces the original reports. No other model is substituted.
The alarm is still at 1785 for `[1780,1785)`, 13 seconds after annotated offset.
It remains a false alarm and both models still miss the seizure with undefined detection
latency. There is no overlapping positive window to credit under the existing
rules. The value 13 s is a descriptive post-offset delay, not detected latency.
No technical error requiring correction of previous results was found.

The sampled waveform/PSD can explain why this postictal window looks different
in the feature space: it contains real high-frequency signal energy and larger
line length. It does not establish which feature drove a particular tree decision
or prove that these measurements caused the alarm. The models' uncalibrated scores
0.520 and 0.680 remain model outputs, not clinical probabilities.

#### Unresolved explanations and defensible conclusions

A brief, spatially uneven broadband fast burst is compatible with muscle/movement
or electrode-related contamination, but also potentially with physiological
postictal activity or a mixture. Shared electrodes, bipolar montage, no video,
no EMG and no expert adjudication prevent attribution from these plots alone.
A suspicious waveform does not prove artifact. This review does not justify
extending seizure annotations, calling the burst ongoing seizure, or discarding it.

The missed ictal signal differs from the selected detected examples in fast
spectral content and line length; it is not simply absent or flat. The late alarm
is associated with a distinct post-offset burst already visible in raw EEG.
Sparse training (three other events and 24 ictal windows in fold 3), inter-seizure
variability and this feature representation remain plausible limitations. These
four events belong to one patient and do not establish population or clinical
reliability. Filter delay alone has not been shown to explain a 13-second gap;
reproducing the features only verifies implementation consistency.

A qualified EEG review with clinical/video context would be needed to distinguish
physiology from artifact. If proceeding later to LightGBM, retain the original
annotations, frozen split and predeclared rules; compare it as a separate experiment
rather than assuming a different classifier will resolve this failure. No such
experiment was started here.

#### Outputs and reproduction

Added `src/raw_eeg_investigation.py`, four synthetic tests, this report and a
README link. Existing uncommitted work and all previous artifacts are preserved.
Figures in `results/raw_eeg_investigation_v1/`:

1. `waveforms_overview.png`: 1700–1810 s, raw and filtered panels.
2. `waveforms_alarm_zoom.png`: 1775–1790 s, matching scales, highlighted alarm window.
3. `spectra_comparison.png`: pre, midpoint-ictal and three postictal windows, raw/filtered.
4. `detected_seizure_context.png`: two channels in all four midpoint exemplars.

The output also contains all raw/filtered per-window numerical measurements,
channel/provenance/feature checks, analysis configuration and artifact hashes.
Initial-run test logs and a preservation audit accompany it. Twenty-nine tests
passed: four new synthetic checks and 25 existing evaluator tests. Training tests
were not run during this analysis-only phase. No dependencies were installed,
models fitted, raw files changed, annotations edited, commits made or changes pushed.

From the repository root, using a fresh output directory:

[Historical code block omitted; see linked source.]

The script refuses an existing output directory and opens only the four EDFs
explicitly named in this report. No model modifications, threshold optimization,
new classifier, temporal filtering or dashboard work follows this investigation.


### Source report: lightgbm_experiment

Source: [docs/lightgbm_experiment.md](../docs/lightgbm_experiment.md); original section names appear below.

#### Phase 2G: fixed LightGBM versus Random Forest comparison

Both LightGBM models detect the single validation seizure at 1725 seconds,
5 seconds after onset, and correctly classify the same 10 of 18 ictal windows.
Each produces one false-alarm episode in two hours. Weighting rescues no ictal
windows; it removes one false-positive window and introduces another elsewhere.
Neither LightGBM model establishes an improvement over ordinary RF: ordinary RF
has higher recall, F1 and AP and no false alarm in this validation set. LightGBM
alarms five seconds earlier than ordinary RF, but matches weighted RF's latency.
There is only one evaluated seizure, so these observations cannot establish a
reliable model ranking or clinical performance.

#### Environment and minimum installation

The existing environment is Python 3.14.5 on macOS ARM64, with NumPy 2.5.4,
SciPy 1.18.1, pandas 3.0.6, scikit-learn 1.9.1, joblib 1.6.0, PyArrow 26.0.0,
matplotlib 3.11.2 and MNE 1.13.2. These versions remain unchanged.
LightGBM was absent. The pinned `lightgbm-4.7.0-py3-none-macosx_12_0_arm64.whl`
was installed into the existing venv with no dependency upgrades. Import then
confirmed a missing `libomp.dylib`; only Homebrew's OpenMP runtime, libomp
23.1.4, was added, with automatic Homebrew update/cleanup disabled. No compiler,
GPU runtime or other Python package was installed. Synthetic and actual fits
passed on this environment.

[Historical code block omitted; see linked source.]

These commands describe the minimal additions for this existing environment;
they are already installed for this run. `requirements.txt` gains only the
LightGBM pin. Runtime binary paths/hashes, wheel tag and commands are saved in
`environment.json` with the results. Restoring recorded library/runtime versions
matters for reproduction; deterministic settings are not a cross-version or
cross-platform guarantee.

Sources: [official package and ARM64 wheel](https://pypi.org/project/lightgbm/),
[installation guide](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html).

#### Frozen configurations and data isolation

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

#### Combined and per-recording validation results

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

#### Alarm times and changed predictions

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

[Figure: Four-model validation score comparison; see Appendix E.]

The upper panel shows the complete seizure-containing recording, including the
weighted LightGBM false alert at 2535; the lower panel zooms around onset/offset.
Markers are actual window-end scores. Connecting lines are visual guides and do
not create intervening decisions or temporal filtering. The separate normal
recording's false alert is documented in the ledger and comparison table.

#### Interpretation, limitations and next phase

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

#### Files, verification and reproduction

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

[Historical code block omitted; see linked source.]

Existing output directories are refused. `MPLCONFIGDIR` only chooses a writable
plot-font cache. To reproduce the full suite with the explicit held-out exclusion:

[Historical code block omitted; see linked source.]

No original model, preprocessing implementation, label, manifest or prior result
was changed; no commit or push was performed. Work stops after these two fits
and their validation comparison.


### Source report: lightgbm_cross_validation

Source: [docs/lightgbm_cross_validation.md](../docs/lightgbm_cross_validation.md); original section names appear below.

#### Phase 2H: training-only LightGBM cross-validation and stability

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

#### Scope, fixed models and fold verification

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

#### Per-fold window and event results

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

#### Pooled and macro summaries

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

#### chb01_15: detected by ordinary LightGBM, still problematic

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

#### Continuity, weighting and stability

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

[Figure: Four-model out-of-fold score comparison; see Appendix E.]

Markers show actual decisions; connecting lines are visual guides, not new
predictions between windows or temporal filtering. The figure is a seizure-local
view; complete tables/evaluator exposure include all one-hour recordings and
false alarms outside the plotted ranges, such as the start at 870 in `16`.

#### Interpretation and provisional model recommendation

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

#### Files, tests and reproduction

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

[Historical code block omitted; see linked source.]

Existing output directories are refused. The protected full-suite command is:

[Historical code block omitted; see linked source.]

Work stops after the specified eight LightGBM fits and their cross-validation
comparison. No temporal filtering, final testing, new datasets or dashboard work
was performed.


### Source report: temporal_filtering

Source: [docs/temporal_filtering.md](../docs/temporal_filtering.md); original section names appear below.

#### Phase 3A: fixed causal temporal-filtering experiment

Both confirmation rules remove the observed false-alarm episodes and preserve
the observed event-detection counts. They also delay detection. The critical
case is ordinary LightGBM on `chb01_15`: both postpone the first alarm from
1740 to 1770 seconds, increasing latency from 8 to 38 seconds for a 40-second
seizure. The event is still counted detected, but only two seconds before offset.

**Recommendation: retain no filtering for the provisional primary model and the
RF reference.** The zero-false-alarm result does not by itself justify this delay.
No acceptable latency ceiling was predeclared; treating a 38-second response as
equivalent to an 8-second response would conceal an important degradation.
This is a research recommendation, not a clinical operating policy. Neither
confirmation rule is adopted automatically.

#### Rules frozen before evaluation

The three strategies consume the saved binary model decisions at the existing
0.5 classifier boundary. No model, score threshold or annotation is changed.
For raw decision `p[t]` at actual window end `e[t]`:

- **None:** `a[t] = p[t]`.
- **Two consecutive:** `a[t] = 1` iff `p[t] = p[t-1] = 1`.
- **Two of three:** `a[t] = 1` iff at least two raw decisions in
  `p[max(0,t-2):t+1]` are positive.

Both confirmation rules start with output zero at the first observed window.
At the second window they can fire if both observed decisions are positive;
the two-of-three rule uses available history rather than requiring an invented
third observation. At later windows it uses the current and two previous raw
predictions, not previous filtered outputs. None preserves the first raw value.
There is no future information, centered rolling window, backdating, cooldown
or annotation-dependent override. Output at row t is available only at `e[t]`.
Two-of-three may output 1 when the current raw decision is 0, based on two
previous positives; this is causal persistence, not an earlier timestamp.

State resets for each recording. Complete ordered coverage from time zero to
recording end is required, with contiguous five-second windows and finite,
binary labels and predictions. Missing initial/middle/final windows, duplicates,
out-of-order windows, incomplete or variable-length windows are rejected.
No absent predictions are imputed as negatives, and no state is carried across
gaps. Existing evaluator timeline tolerance (1e-9 s) is retained. The pure
`filter_binary` function needs only the observed binary sequence; annotations
are used for input integrity checks/evaluation, never to generate its decisions.

The evaluator's matching and re-arming definitions remain unchanged: the first
positive **filtered** decision begins an alarm, one negative filtered decision
ends/re-arms it, and state is independent per recording. Only a new start within
exact `[onset,offset)` detects a seizure; further starts inside are repeats,
outside starts are false alarms. Episode ends are first negative decision times,
not retrospective corrections of alarm starts. False alarms/hour divides false
episodes by full evaluated EEG hours, including ictal time. Missed-event latency
is undefined rather than zero. These rules come from unchanged `src/evaluate.py`.
Reports identify the upstream strategy in the policy metadata only.

#### Inputs and separation of assessments

Primary inputs are the saved ordinary RF and ordinary LightGBM out-of-fold
predictions from exactly the four original training holdouts `03`, `04`, `15`,
`16`: 2,880 windows, 32 ictal windows, four distinct annotated seizures and four
hours. Each recording contributes once per model. Fold models were trained
without that recording. No model is loaded, refitted or used to generate fresh
scores here. Saved predictions and original evaluator reports are hash-checked
and the no-filter baseline reproduces before filtering.

Secondary inputs are the saved ordinary RF and ordinary LightGBM validation
predictions on `17` and `18`: 1,440 windows, 18 ictal windows, one seizure and
two hours. They are loaded only after completing the primary assessment, using
the identical three strategies. Validation outcomes do not define or modify a
rule or override the primary conclusion. Do not pool the two assessments as
independent evidence: the underlying training sets overlap, and CV versus
validation models differ in training composition.

No processed feature dataset or raw EDF is read by the experiment. Original
final-test `21`, `22`, `26` and supplementary `23`–`25` predictions or recordings
are never accessed. Only explicitly named CV/validation prediction artifacts
and their metadata/reports are read. Unrelated uncommitted work and previous
experiments are preserved. No dependency installation, model fitting, feature
change or ground-truth relabeling occurs.

#### Primary training-fold results: four hours

| Model | Rule | Detected / missed | False episodes/h | Repeats | Precision | Recall | F1 | FP windows | Raw → filtered positives | Added / removed |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Ordinary RF | None | 3 / 1 | 0.25 | 2 | 0.944444 | 0.531250 | 0.680000 | 1 | 18 → 18 | 0 / 0 |
| Ordinary RF | Two consecutive | 3 / 1 | 0.00 | 1 | 1.000000 | 0.375000 | 0.545455 | 0 | 18 → 12 | 0 / 6 |
| Ordinary RF | Two of three | 3 / 1 | 0.00 | 0 | 0.944444 | 0.531250 | 0.680000 | 1 | 18 → 18 | 4 / 4 |
| Ordinary LightGBM | None | 4 / 0 | 0.50 | 2 | 0.875000 | 0.656250 | 0.750000 | 3 | 24 → 24 | 0 / 0 |
| Ordinary LightGBM | Two consecutive | 4 / 0 | 0.00 | 1 | 0.937500 | 0.468750 | 0.625000 | 1 | 24 → 16 | 0 / 8 |
| Ordinary LightGBM | Two of three | 4 / 0 | 0.00 | 0 | 0.863636 | 0.593750 | 0.703704 | 3 | 24 → 22 | 5 / 7 |

| Event | RF none | RF consecutive | RF two-of-three | LGBM none | LGBM consecutive | LGBM two-of-three |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chb01_03 | 9 s | 14 s | 14 s | 9 s | 14 s | 14 s |
| chb01_04 | 8 s | 23 s | 18 s | 8 s | 13 s | 13 s |
| chb01_15 | missed | missed | missed | 8 s | 38 s | 38 s |
| chb01_16 | 5 s | 10 s | 10 s | 5 s | 10 s | 10 s |

Each delay is first alarm start minus exact onset; no grace period or backdating.

RF's missed `15` remains missed under every strategy; its isolated postictal
false alarm disappears. For LightGBM the `15` first positive at 1740 is isolated
and both filters suppress it. The later raw positives at 1765 and 1770 provide
the first confirmation at **1770**. Nothing is moved back to 1765. Two-consecutive
retains only that ictal decision; two-of-three additionally outputs a positive at
1775, on a nonictal-labelled window, because 1765 and 1770 remain in raw history.
Both still satisfy the unchanged event convention because the start at 1770 is
strictly earlier than offset 1772. The postictal raw alert at 1785 is removed.
Two-consecutive's episode ends at 1775; two-of-three's at 1780.

The mean LightGBM latency increases from 7.5 to 18.75 s with either filter;
individual increments are +5,+5,+30,+5 s. RF mean detected-event latency increases
from 7.333 s to 15.667 s (consecutive) or 14 s (two-of-three). These RF means
exclude its missed event and are not directly comparable to LightGBM's four-event
means. No mean substitutes for the event-specific table.

No additional actual event is missed, but the synthetic short-event tests show
that a single-window detection can be lost entirely and that confirmation
exactly at offset becomes a false alarm. Four observed events cannot establish
that preservation will hold for shorter or differently aligned seizures.

[Figure: Primary positive decision times; see Appendix E.]

Ticks mark actual positive window-end decisions, not continuous seizure coverage.
The annotation is shaded. These are seizure-local views; all metrics still use
full one-hour timelines, including alarms outside the plotted ranges.

#### Window classification versus alarm episodes

Zero false-alarm episodes does **not** mean zero FP windows. Primary LightGBM
two-of-three still has three FP windows: two post-offset windows in `04` and one
in `15`, all within episodes that began during their seizures. Two-consecutive
has one FP window in `04`, also within a valid detection episode. These are not
new false-alarm starts. In contrast, isolated raw starts at 1785 in `15` and
870 in `16` are removed by both filters, reducing LightGBM false episodes from
two to zero over four hours.

Primary RF two-of-three has exactly the same pooled confusion matrix, precision,
recall and F1 as unfiltered RF, yet four positives are removed and four added
at different windows. Its original FP at 1785 in `15` disappears, while a new
FP at 1500 in `04` extends an existing valid episode. Thus its FP-window count
stays one, its false-episode count drops one to zero, and every detected event
starts later. Equal window counts do not imply equal predictions or alert behavior.

The consecutive rule can only remove positives. Two-of-three can remove isolated
positives, bridge gaps and add a trailing positive. It reduces repeat episodes
from two to zero for both models in CV, while consecutive leaves one repeat in
`16`. More sustained alerts do not recover the 30 seconds lost before `15`.
No ground-truth labels or scores change. Original continuous model scores are
retained in prediction files; AP is deliberately not calculated for these binary
strategies, which define no new continuous score. It would not measure the
classification changes induced by filtering.

#### Exact positive-prediction changes

| Assessment | Model | Rule | Recording | Removed positive decision ends | Added positive decision ends |
| --- | --- | --- | --- | --- | --- |
| primary_cv | Ordinary RF | Two consecutive | chb01_03 | 3005 | none |
| primary_cv | Ordinary RF | Two consecutive | chb01_04 | 1475, 1485 | none |
| primary_cv | Ordinary RF | Two consecutive | chb01_15 | 1785 | none |
| primary_cv | Ordinary RF | Two consecutive | chb01_16 | 1020, 1035 | none |
| primary_cv | Ordinary RF | Two of three | chb01_03 | 3005 | 3035 |
| primary_cv | Ordinary RF | Two of three | chb01_04 | 1475 | 1500 |
| primary_cv | Ordinary RF | Two of three | chb01_15 | 1785 | none |
| primary_cv | Ordinary RF | Two of three | chb01_16 | 1020 | 1030, 1060 |
| primary_cv | Ordinary LightGBM | Two consecutive | chb01_03 | 3005 | none |
| primary_cv | Ordinary LightGBM | Two consecutive | chb01_04 | 1475 | none |
| primary_cv | Ordinary LightGBM | Two consecutive | chb01_15 | 1740, 1765, 1785 | none |
| primary_cv | Ordinary LightGBM | Two consecutive | chb01_16 | 870, 1020, 1035 | none |
| primary_cv | Ordinary LightGBM | Two of three | chb01_03 | 3005 | 3035 |
| primary_cv | Ordinary LightGBM | Two of three | chb01_04 | 1475 | 1505 |
| primary_cv | Ordinary LightGBM | Two of three | chb01_15 | 1740, 1765, 1785 | 1775 |
| primary_cv | Ordinary LightGBM | Two of three | chb01_16 | 870, 1020 | 1030, 1060 |
| secondary_validation | Ordinary RF | Two consecutive | chb01_18 | 1730, 1740, 1795 | none |
| secondary_validation | Ordinary RF | Two of three | chb01_18 | 1730, 1795 | 1785 |
| secondary_validation | Ordinary LightGBM | Two consecutive | chb01_17 | 1825 | none |
| secondary_validation | Ordinary LightGBM | Two consecutive | chb01_18 | 1725 | none |
| secondary_validation | Ordinary LightGBM | Two of three | chb01_17 | 1825 | none |
| secondary_validation | Ordinary LightGBM | Two of three | chb01_18 | 1725 | 1775 |

Times in this table are the decision ends of changed rows, in seconds from the
respective recording start. Removed means raw 1 → filtered 0; added means raw
0 → filtered 1. Every changed row, with original start/end, label, score, original
prediction, filtered prediction, model and rule, is saved in
`changed_predictions.parquet`. The complete aligned filtered timelines are also
saved for each strategy. None has no changed rows. Pooled/per-recording CSVs
include raw and filtered positive counts, additions and removals.

#### Secondary assessment: original validation, two hours

| Model | Rule | Detected / missed | False episodes/h | Repeats | Precision | Recall | F1 | FP windows | Raw → filtered positives | Added / removed |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Ordinary RF | None | 1 / 0 | 0.00 | 2 | 1.000000 | 0.611111 | 0.758621 | 0 | 11 → 11 | 0 / 0 |
| Ordinary RF | Two consecutive | 1 / 0 | 0.00 | 0 | 1.000000 | 0.444444 | 0.615385 | 0 | 11 → 8 | 0 / 3 |
| Ordinary RF | Two of three | 1 / 0 | 0.00 | 0 | 1.000000 | 0.555556 | 0.714286 | 0 | 11 → 10 | 1 / 2 |
| Ordinary LightGBM | None | 1 / 0 | 0.50 | 0 | 0.909091 | 0.555556 | 0.689655 | 1 | 11 → 11 | 0 / 0 |
| Ordinary LightGBM | Two consecutive | 1 / 0 | 0.00 | 0 | 1.000000 | 0.500000 | 0.666667 | 0 | 11 → 9 | 0 / 2 |
| Ordinary LightGBM | Two of three | 1 / 0 | 0.00 | 0 | 1.000000 | 0.555556 | 0.714286 | 0 | 11 → 10 | 1 / 2 |

All strategies detect the single `18` event, with zero missed events. RF latency
is 10 s unfiltered, 25 s consecutive and 20 s two-of-three. Its two repeat alarms
disappear with either filter; there was no RF false episode to remove. The filters
therefore add 15 or 10 seconds without reducing RF false alarms in validation.

LightGBM latency increases 5 → 10 s for either confirmation rule. Both remove its
isolated false alarm on `17` at 1825 s, reducing one false episode/two hours to
zero. On `18`, consecutive removes its first ictal positive; two-of-three removes
that positive but adds one at 1775, retaining the same number of true-positive
windows. This explains why two-of-three's secondary F1 improves while its first
detection still becomes later. Neither rule produces a new secondary false alarm.
`17` has no seizures: event sensitivity/latency and window recall are undefined,
not forced to zero or treated as perfect detection. Per-recording JSONs preserve
undefined values as `null` and explain zero denominators.

[Figure: Secondary positive decision times; see Appendix E.]

This relatively benign LightGBM validation outcome cannot erase the primary
38-second latency on `15`. No extra filter was invented to keep its early alert
while suppressing other isolated positives.

#### Decision, limits and reproducibility

The predefined priorities were event preservation, false-alarm reduction,
minimizing delay, then simplicity. Both filters preserve the observed counts and
reduce false episodes, but those coarse event counts conceal the severe onset
response loss in `15`. There was no predeclared numerical latency budget, so
we cannot claim a formally established optimal operating point. Given the
30-second extra delay and only two seconds remaining in that seizure, I do not
recommend replacing the unfiltered provisional LightGBM with either confirmation
rule now. Retain no filtering and record the trade-off transparently. RF
two-of-three is a useful saved alternative for comparison, but adds 5–10 seconds
per detected CV event and does not resolve its missed seizure. It is not adopted
solely because of unchanged F1 or fewer repeats.

Four CV seizures from one patient, overlapping training sets, highly correlated
EEG windows, a repeatedly studied `15`, and one previously examined validation
seizure limit inference. Zero false episodes over four CV hours or two validation
hours does not establish zero future false-alarm risk. Alarm-count reductions
can coexist with longer post-offset positive decisions. No clinical reliability,
population sensitivity or general optimality claim follows. A later decision
would need an agreed latency tolerance and independent evidence, not further
rule invention on these same recordings. No additional experiment is started.

Added `src/temporal_filter.py` (small pure causal module),
`src/temporal_filter_experiment.py` (saved-prediction assessment), twelve tests
in `tests/test_temporal_filter.py`, this report and a README link. No existing
preprocessing, evaluator, manifest, model artifact or annotation was edited.
Earlier uncommitted changes remain intact. Outputs are separate under
`results/temporal_filtering_v1/`, with `primary_cv/` and `secondary_validation/`:
complete predictions, pooled/per-recording CSVs, per-event CSVs, exact changed
rows, JSON alarm ledgers, figures, input/source hashes and rule definitions.
The initial run includes test logs and a preservation audit.

Verification uses 12 new temporal/integration tests and 25 existing evaluator
tests: **37 passed, none skipped**. No model-training tests are needed for this
saved-prediction phase. Tests cover isolated positives, short events, intermittent
positives, causal prefix invariance, startup, boundaries, incomplete/missing/
shuffled/duplicate windows, immutable labels/scores, exact row alignment,
re-arming/repeats, strict offset matching, input preservation and evaluator
integration. Baseline scores/reports reproduce from saved artifacts before
filtering. No models are loaded or refitted by the experiment.

From the repository root, using a fresh output directory:

[Historical code block omitted; see linked source.]

The default inputs are the saved original ordinary RF/CV and ordinary LightGBM/CV
artifacts plus their separate original validation outputs. CLI options permit
explicit source directories, not arbitrary new recordings. Existing output
folders are refused. No final-test or supplementary data, model training,
hyperparameter optimization, new thresholds, EEGNet, dashboard, dependency
installation, commit or push is involved. Work stops after this controlled report.


### Source report: final_evaluation_protocol

Source: [docs/final_evaluation_protocol.md](../docs/final_evaluation_protocol.md); original section names appear below.

#### Final experimental protocol — Phase 3B

Protocol `nyas-p01-final-evaluation-v1`, frozen **2026-11-10 (Asia/Kolkata)**.
Status: **awaiting review and separate authorization for final evaluation**.
No final-test predictions or supplementary evaluation were performed for this freeze.

#### Scientific objective and fixed choices

Estimate recording-held-out seizure detection performance for Patient 01 after
exploratory comparisons of majority prediction, ordinary/cost-sensitive trees,
and two causal temporal persistence rules. Accuracy alone is inadequate: the
majority classifier achieved 98.75% validation accuracy without detecting a
seizure. Weighting did not consistently improve detection. Training-recording
cross-validation detected 4/4 events with ordinary LightGBM versus 3/4 with
ordinary Random Forest, with more false alarms for LightGBM. Temporal filtering
removed observed false alarms but delayed the `chb01_15` detection from 8 to 38
seconds. These small development results motivate the following fixed choices;
they are not estimates of clinical performance.

Use **ordinary LightGBM from Phase 2G as primary** and **ordinary Random Forest
from Phase 2B as reference**. Report both, regardless of the final outcomes.
Use the saved fitted models; do not refit on training plus validation data.
No further weighting, parameter search, resampling, feature selection, scaling,
calibration, temporal persistence, or threshold selection is permitted.

#### Dataset partitions

All IDs below have prefix `chb01_` and EDF extension `.edf`.

| Role | Recording suffixes | Permitted use |
| --- | --- | --- |
| Original training | 01, 03, 04, 14, 15, 16 | Already fitted models; completed development CV |
| Original validation | 17, 18 | Completed secondary development assessment; never added to fitting |
| Primary final test | 21, 22, 26 | One subsequent evaluation of both frozen models |
| Supplementary | 23, 24, 25 | Separate seizure-free false-alarm assessment only |

Original assignments remain in
[`config/chb01_recordings.csv`](../config/chb01_recordings.csv). The primary test
uses the existing
`data/processed/chb01_detection_features_11recordings_split.parquet` and its JSON
integrity audit. Do not recreate or alter this dataset to improve results.
`chb01_26` is shorter than one hour: use its actual duration and window count;
never assume 720 windows per recording. Evaluation requires complete contiguous
coverage of each actual recording.

Supplementary recordings are outside the original manifest and have not been
preprocessed or evaluated in this phase. They must not be used for fitting,
threshold selection, or model comparison. Report their durations, windows,
alarms and false-alarm rate separately; do not pool their exposure into the
primary test. Verify their zero-seizure status from the summary in the later
preparation step.

#### Exact fitted artifacts and configurations

The machine-readable authority is
[`config/final_evaluation_protocol.json`](../config/final_evaluation_protocol.json),
which includes every estimator parameter and the full ordered feature list.
Its SHA-256 is
`cb5b77949755ba0aaf796e6199f388a83a195257e5f3b9459d136bbf2d4c7f3b`.

| Role | Configuration ID | Artifact |
| --- | --- | --- |
| Primary | `ordinary_lightgbm_phase2g` | `results/lightgbm_v1/ordinary_lgbm.joblib` |
| Reference | `ordinary_random_forest_phase2b` | `results/baseline_v1/random_forest.joblib` |

Primary model SHA-256:
`7324d3199850abd2e78d9190ff08946ecfd056e1a32e4b5d0f901d519b7b2b5f`.
Reference model SHA-256:
`05fb53387283e82892aed6885469070fc85782f2155b460ba6b91cc4534d60c0`.

LightGBM uses 150 trees, learning rate 0.05, maximum depth 3, seven leaves,
minimum child samples 10, seed 42, `scale_pos_weight=1.0`, `class_weight=None`,
binary GBDT, four threads, deterministic column-wise training, no regularization,
and full row/column sampling. All other settings are frozen verbatim from
`results/lightgbm_v1/configuration.json` (SHA-256
`c53682449811e2aafb57fd1dc0b11601f78b68288b78854b8a96e2fbee58bb9e`).
The original native LightGBM text archive is also hashed in the protocol.

Random Forest uses 200 trees, seed 42, `class_weight=None`, four threads,
bootstrap sampling, Gini criterion, `max_features="sqrt"`, unrestricted depth,
minimum leaf size 1 and split size 2. All settings are frozen from
`results/baseline_v1/configuration.json` (SHA-256
`562837c804288d0f9c9758ffb965b8a67ad4ad0046b881027a71f9f50991dc56`).

Both selected models reload, have classes `[0, 1]`, and retain exactly the
canonical 126 feature names in order. Outputs reproduce saved historical
validation predictions (classes exact; scores within absolute tolerance 1e-15).
This is a serialization check, not a new validation experiment. Neither selected
model was fitted again.

Training provenance matches the six training IDs and 4,320 windows (4,288
nonictal, 32 ictal), with four events. Original configurations, source/data
hashes and artifact checksums match. LightGBM's original ordered fit-input
fingerprint is
`9e238086a6b2b977a534468748e0616f2b0388d01395fc9439179865d636c699`.
RF tree root weighted sample counts and LightGBM's initial root count agree
with 4,320 training rows. Serialized RF does not embed row identities: exclusion
of reserved recordings is supported by verified training code, configuration
and data provenance, rather than row IDs recoverable from its binary.

#### Features, preprocessing and labels

Keep `detection-v1` unchanged. Five-second nonoverlapping windows contain 1,280
samples at 256 Hz. The 18 bipolar channels, in order, are:

[Historical code block omitted; see linked source.]

Within each channel the ordered statistics are `var`, `line_length`, `delta`,
`theta`, `alpha`, `beta`, `gamma` (18 × 7 = 126), named `<channel>_<stat>`.
Variance is population variance in µV²; line length is the sum of absolute
successive sample differences in µV. No metadata column is a predictor.

Filtering remains forward-only Butterworth SOS (`butter(4, [0.5, 45], ...)`),
with fresh zero state per recording and continuous state across its windows.
No future samples or notch filter are used. The first 10 seconds are flagged
and retained. The first duplicate `T8-P8-0` supplies canonical `T8-P8`.

Relative PSD retains Welch two-second Hann segments, one-second overlap,
constant detrending and 0.5 Hz bin integration. Denominator: integrated power
from 0.5 through 45 Hz. Bands: delta [0.5,4), theta [4,8), alpha [8,13), beta
[13,30), gamma [30,45] Hz. Channels at or below 1e-12 µV² total power receive
zero relative-band values and a quality flag. Quality/warmup/postictal flags
remain metadata only.

Class 1 means at least 50% of the window overlaps the union of validated ictal
intervals; class 0 otherwise. Exactly 50% is positive. Both annotations and
windows are half-open; touching edges contribute no overlap. Keep ground truth
unchanged. Existing preprocessing drops incomplete trailing windows with a
warning. If that causes incomplete evaluation coverage later, stop for review
rather than shortening exposure or changing the frozen evaluator.

#### Decisions, alarms and event matching

Use uncalibrated `predict_proba` class-1 scores and original `model.predict`
decisions at the fixed 0.5 boundary: **score > 0.5** gives class 1; an exact tie
selects class 0. Check this agreement; do not replace it with `>= 0.5`.

The decision timestamp is actual `end_sec`, also retained as `decision_time_sec`.
Never backdate to the window start or use future predictions. Keep one output
row aligned with each original window. Latency includes window acquisition but
excludes measured computation/deployment delay.

Reuse [`src/evaluate.py`](../src/evaluate.py) without changes:

- A positive decision while inactive starts an episode. Consecutive positives
  form one episode. One negative decision ends it and immediately re-arms;
  there is no cooldown, gap merging or persistence filter.
- Episode end is the first negative decision time, or recording end when
  right-censored. This bookkeeping never changes the start. A final-window
  alarm can have equal start and end times and still counts.
- A seizure is detected only by a **new** alarm starting in exact annotated
  `[onset, offset)`: onset inclusive, offset exclusive. No early tolerance or
  late grace. An alarm active before onset cannot detect that event.
- The first matching start detects the event. Later starts in that event are
  repeats, adding neither detected events nor false alarms. Starts outside all
  events are false alarms, including pre-onset and post-offset starts.
- Latency is first matched alarm start minus onset. Misses have `latency_sec=null`.
  Retain every event, including misses, in the ledger.
- State resets per recording. Missing, duplicate, incomplete or unordered
  timelines are errors. Evaluate each whole recording once.

#### Metrics and reporting

Report per recording and pooled, separately for each model and assessment:

| Level | Required outputs |
| --- | --- |
| Windows | Counts/prevalence, accuracy, precision, recall, specificity, F1, average precision, confusion matrix |
| Events | Annotated/detected/missed events, sensitivity, false-alarm count and episodes/hour, repeats, exact alarms, per-event latency |

Confusion matrix is `[[TN, FP], [FN, TP]]`: true rows, predicted columns, class
order `[0,1]`. Accuracy = (TP+TN)/N; precision = TP/(TP+FP); recall = TP/(TP+FN);
specificity = TN/(TN+FP); F1 = 2TP/(2TP+FP+FN). Average precision is sklearn's
step-weighted precision-recall summary on continuous scores, not trapezoidal
PR area or binary decisions.

False alarms/hour = false-alarm starts / evaluated EEG hours, **including ictal
time**, excluding unrecorded gaps. Pool counts and exposure, not an unweighted
mean of recording rates. Pooled latency summaries include detected events only;
the full event ledger must show all misses.

Undefined rates use JSON `null`, with reasons. The existing window evaluator
supplies `undefined_metrics`. Event sensitivity is undefined for zero events;
latency summaries are undefined with no detections. For supplementary seizure-free
data, **seizure sensitivity and detection latency are undefined, not zero**;
window recall and AP are also undefined. Precision is undefined with no positive
predictions; otherwise precision and F1 follow the unchanged formulas.

Report both predetermined models even if the reference performs better. Do not
use final outcomes to change parameters, thresholds, features, filters, labels,
model selection or reporting scope. Later method changes require separate
development and independent evaluation data.

#### Reproducibility and verification

Git branch `main`, HEAD `a1359fa4b64f94dd4eab45c82b1d03952f3d79f6`.
**The tree is dirty:** several completed development scripts/reports are
uncommitted. HEAD alone cannot reproduce this freeze. The JSON records the
pre-freeze status and hashes of every source/test file, requirements, manifest,
dataset/audit, annotation source, models and original configurations. No commit
or push was made; unrelated work was preserved.

Environment: Python 3.14.5, macOS 26.3.2 ARM64; LightGBM 4.7.0, scikit-learn
1.9.1, NumPy 2.5.4, SciPy 1.18.1, pandas 3.0.6, joblib 1.6.0, PyArrow 26.0.0,
Matplotlib 3.11.2, MNE 1.13.2. The JSON pins libomp 23.1.4 and LightGBM native
library hashes. Both models load without installation or repair. Preserve this
environment; do not silently upgrade before evaluation.

Tests: **112 discovered, 111 passed, one explicitly skipped, zero failures**.
Deferred test:
`test_split_dataset.ExpandedArtifactTests.test_all_recordings_labels_features_and_splits`,
which decodes final-test rows. It was skipped to respect this phase's restriction
on test-label access. Some existing tests fit small synthetic estimators; none
retrain the selected artifacts. Test/reload/provenance evidence is saved under
`results/protocol_freeze_v1/`. Preprocessing, labels, evaluation, datasets, models
and earlier experimental outputs retain their pre-phase hashes. Existing Git
rules exclude EDFs, processed datasets, results, model binaries and the virtual
environment; no generated artifacts are tracked.

Reproduce the verification suite without decoding final-test rows:

[Historical code block omitted; see linked source.]

#### Proposed primary final-evaluation command — NOT EXECUTED

Run from the repository root **only after separate explicit authorization**.
The complete command below never fits models. It checks frozen inputs and
environment, selects the three test recordings, then uses the unchanged
evaluator. It refuses an existing output directory. Its Python syntax was checked
in Phase 3B; its final-data path was deliberately not executed. A failed check
must stop for review, not trigger refitting or manual relaxation of the protocol.

[Historical code block omitted; see linked source.]

The reports contain pooled/per-recording metrics, exact alarms and every event's
detection/miss/latency. Convert the unchanged reports into comparison tables;
do not select a new model from the results. Do not rerun development training
commands to perform final testing.

#### Proposed supplementary workflow — NOT EXECUTED

After separate authorization, verify that the explicit EDFs are complete,
readable and annotated with zero seizures in `chb01-summary.txt`. Confirm that
the proposed output does not exist, then use the existing preprocessing CLI:

[Historical code block omitted; see linked source.]

Check and archive raw/summary/source/output hashes, durations, channels, finite
features, exact feature ordering, complete windows, zero ictal labels and zero
events before inference. This artifact does not yet exist, so its checksum
cannot be frozen here. Record its later provenance separately, without changing
models or the original manifest.

Then use the same `select_features` → saved-model `predict`/`predict_proba` →
`evaluate_predictions` handoff in the primary command, confined to supplementary
IDs and their own verified preprocessing metadata. Write only to a fresh
`results/supplementary_false_alarms_v1/`. This is a distinct later preparation and
evaluation step; the primary command intentionally excludes these recordings.
Neither supplementary preparation nor inference is authorized by this freeze.

#### Limitations and unresolved issues

- One patient, few seizures, correlated windows and extensive exploratory reuse
  of development data limit inference. The final split is held out at the
  recording level; it is not patient-independent, prospective or clinical
  validation. Small event counts make sensitivity and false-alarm estimates unstable.
- Five-second windows quantize decisions. Overlap labels can disagree with
  strict alarm-start matching near short seizures and boundaries. Retain this
  established distinction.
- Scores are uncalibrated. AP measures discrimination, not calibration of
  clinical probabilities. Accuracy/specificity alone cannot establish utility.
- Offline latency excludes computation, transport, electrode quality and
  intervention. Zero observed alarms over short exposure does not establish a
  zero underlying false-alarm rate.
- Uncommitted files prevent HEAD-only reproduction: retain hashes and the
  workspace pending a later reviewed commit. There is no serialization problem
  requiring retraining. Supplementary preparation and the deferred final-data
  integration test remain for the separately authorized next phase.


### Source report: final_test_results

Source: [docs/final_test_results.md](../docs/final_test_results.md); original section names appear below.

#### First final held-out evaluation — Phase 3C

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

#### Protocol verification and model identity

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

#### Dataset and exposure

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

#### Pooled results

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

[Figure: Confusion matrices with counts and true-class percentages; see Appendix E.]

#### Per-recording window results

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

#### Seizure outcomes and exact alarm timestamps

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

[Figure: Model scores at actual window-end times around each seizure; see Appendix E.]

Scores are uncalibrated. Dots mark actual decisions; step segments hold the last
available value. The shaded intervals are the unchanged annotations, and the
vertical dotted lines mark the shared first alarm times.

[Figure: Exact first and repeat alarm starts relative to seizure onset; see Appendix E.]

#### Comparison with frozen development findings

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

[Figure: Event, false-alarm and repeat-alarm summary; see Appendix E.]

#### Execution, environment and outputs

The exact frozen shell block is archived as
`results/final_test_v1/execution_command.sh`. It was invoked once using:

[Historical code block omitted; see linked source.]

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

#### Post-evaluation verification

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

[Historical code block omitted; see linked source.]

There was no protocol mismatch or evaluation fault. The first plotting run
reported an unwritable default Matplotlib cache and used a temporary fallback;
final figure rendering used an explicit writable temporary cache. This affected
neither predictions nor metric definitions. Figures were visually checked for
timing, readability and unclipped labels.

#### Defensible interpretation and presentation recommendation

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


## Appendix D. Test evolution and saved verification records

These are historical log summaries, not tests executed while preparing this package. Full-suite and focused-suite scopes differ; skipped tests are included in unittest “Ran” totals.

| Historical log | Ran | Recorded outcome | Elapsed |

| --- | --- | --- | --- |

| [results/final_test_v1/post_evaluation_tests.log](../results/final_test_v1/post_evaluation_tests.log) | 8 | OK | 0.270s |

| [results/final_test_v1/existing_tests.log](../results/final_test_v1/existing_tests.log) | 54 | OK | 0.420s |

| [results/lightgbm_cv_v1/tests.log](../results/lightgbm_cv_v1/tests.log) | 100 | OK (skipped=1) | 8.298s |

| [results/failure_analysis_v1/analysis_tests.log](../results/failure_analysis_v1/analysis_tests.log) | 4 | OK | 0.659s |

| [results/failure_analysis_v1/evaluation_tests.log](../results/failure_analysis_v1/evaluation_tests.log) | 25 | OK | 0.026s |

| [results/temporal_filtering_v1/temporal_tests.log](../results/temporal_filtering_v1/temporal_tests.log) | 12 | OK | 1.278s |

| [results/temporal_filtering_v1/evaluator_tests.log](../results/temporal_filtering_v1/evaluator_tests.log) | 25 | OK | 0.026s |

| [results/baseline_v1/tests.log](../results/baseline_v1/tests.log) | 66 | OK | 20.691s |

| [results/raw_eeg_investigation_v1/analysis_tests.log](../results/raw_eeg_investigation_v1/analysis_tests.log) | 4 | OK | 0.004s |

| [results/raw_eeg_investigation_v1/evaluation_tests.log](../results/raw_eeg_investigation_v1/evaluation_tests.log) | 25 | OK | 0.027s |

| [results/rf_weighted_v1/tests.log](../results/rf_weighted_v1/tests.log) | 72 | OK (skipped=1) | 1.875s |

| [results/rf_cv_v1/tests.log](../results/rf_cv_v1/tests.log) | 79 | OK (skipped=1) | 4.355s |

| [results/lightgbm_v1/tests.log](../results/lightgbm_v1/tests.log) | 94 | OK (skipped=1) | 4.452s |

| [results/protocol_freeze_v1/tests.log](../results/protocol_freeze_v1/tests.log) | 112 | OK (skipped=1) | 8.946s |



Early source coverage: preprocessing includes boundary overlap, multiple-event parsing, finite features/relative-power validity, channel selection and recording length; evaluator covers undefined metrics, strict boundaries, repeats and complete timelines; split tests cover membership, duplication, labels and short 26. Model tests add train-only loading, fixed settings, synthetic fit isolation, reload alignment and output protection. Analysis tests cover period grouping and physical units/causal prefix equivalence. Temporal tests add causality, startup, resets and incomplete streams. The final phase runs 54 existing checks plus eight saved-output checks, without re-predicting. Detailed test method inventory follows.

**[tests/test_baseline_rf.py](../tests/test_baseline_rf.py)**: `test_exact_features_ignore_metadata_and_restore_canonical_order`, `test_missing_duplicate_and_nonfinite_features_fail`, `test_fixed_production_configuration`, `test_fit_rejects_validation_test_or_mixed_rows`, `test_fit_receives_only_train_features`, `test_prediction_alignment_with_nonconsecutive_index`, `test_test_predictions_are_rejected`, `test_serialization_and_standard_tie_rule`, `test_parquet_filter_and_disjoint_recordings`, `test_provenance_mismatch_fails`, `test_split_label_timeline_and_duplicate_errors_fail`, `test_full_runner_artifacts_and_no_overwrite`.

**[tests/test_failure_analysis.py](../tests/test_failure_analysis.py)**: `test_score_period_boundaries_are_disjoint_decision_times`, `test_feature_post_requires_whole_window_and_labels_are_preserved`, `test_score_summary_known_values_and_empty_period`, `test_feature_ratios_distinguish_baseline_scaling_from_relative_change`.

**[tests/test_temporal_filter.py](../tests/test_temporal_filter.py)**: `test_none_is_copy_and_isolated_positives_removed`, `test_startup_and_one_decision_confirmation_delay`, `test_intermittent_positives_and_raw_history`, `test_prefix_invariance_proves_no_future_or_backdating`, `test_invalid_inputs`, `test_metadata_labels_scores_and_row_alignment_unchanged`, `test_state_resets_even_with_interleaved_recordings_and_duplicate_indices`, `test_gaps_partial_missing_duplicate_or_out_of_order_windows_rejected`, `test_isolated_short_seizure_detection_can_be_lost`, `test_offset_boundary_is_not_retroactively_detected`, `test_two_of_three_bridges_interruption_without_extra_event`, `test_full_analysis_uses_only_saved_tables_and_protects_inputs`.

**[tests/test_evaluate.py](../tests/test_evaluate.py)**: `test_hand_calculated_confusion_and_rates`, `test_precision_recall_specificity_have_distinct_denominators`, `test_average_precision_uses_scores`, `test_tied_scores_ap_equals_prevalence`, `test_no_positive_truth_or_predictions_is_not_perfect_recall`, `test_all_missed_and_all_positive_truth`, `test_invalid_inputs`, `test_positive_run_starts_at_end_and_closes_on_negative_decision`, `test_future_predictions_cannot_shift_an_alarm_start`, `test_detection_miss_false_alarm_and_latency`, `test_exact_onset_is_hit_exact_offset_is_not`, `test_short_seizure_ends_before_positive_window_decision`, `test_ongoing_preonset_alarm_does_not_detect_seizure`, `test_repeated_alarms_do_not_inflate_seizure_hits`, `test_one_sustained_alarm_cannot_detect_two_seizures`, `test_adjacent_seizures_boundary_belongs_to_second`, `test_no_seizures_and_rearming_without_cooldown`, `test_no_alarms_all_seizures_missed`, `test_no_seizures_and_no_alarms`, `test_invalid_timeline_cannot_inflate_exposure`, `test_invalid_annotations`, `test_recordings_reset_and_rates_pool_by_exposure`, `test_pooled_latencies_and_recording_identity`, `test_optional_scores_and_exact_decision_times`, `test_metadata_and_table_validation`.

**[tests/test_rf_cross_validation.py](../tests/test_rf_cross_validation.py)**: `test_exact_four_folds_disjoint_and_normals_retained`, `test_forbidden_partitions_and_recordings_rejected`, `test_predictions_align_and_only_eeg_features_enter_fit`, `test_aggregation_counts_events_once_and_preserves_undefined_metrics`, `test_only_training_rows_decoded`, `test_incomplete_timeline_rejected`, `test_independent_models_determinism_and_artifact_reload`.

**[tests/test_weighted_rf.py](../tests/test_weighted_rf.py)**: `test_only_class_weight_differs_and_original_is_untouched`, `test_both_forests_receive_identical_ordered_inputs`, `test_fingerprint_detects_feature_label_or_row_order_changes`, `test_comparison_alignment_and_known_changes`, `test_runner_excludes_test_rows_preserves_baseline_and_reloads`, `test_changed_baseline_artifact_rejected_before_fit`.

**[tests/test_split_dataset.py](../tests/test_split_dataset.py)**: `test_fixed_requested_membership`, `test_conflicts_and_invalid_manifest_fail`, `test_assign_splits_preserves_rows_and_prevents_recording_overlap`, `test_known_labels_and_counts`, `test_incorrect_or_missing_windows_and_metadata_fail`, `test_nonfinite_features_and_wrong_channel_order_fail`, `test_raw_provenance_mismatch_fails`, `test_short_recording_sample_bounds`, `test_edf_payload_size_detects_truncation`, `test_summary_short_duration_and_extended_clock_hours`, `test_all_recordings_labels_features_and_splits`.

**[tests/test_lightgbm_experiment.py](../tests/test_lightgbm_experiment.py)**: `test_exact_configuration_and_training_only_weight`, `test_weight_rejects_nontraining_or_single_class`, `test_fits_receive_same_whitelisted_features_without_validation_or_early_stopping`, `test_prediction_alignment_serialization_and_evaluator`, `test_known_prediction_changes_and_alignment_failure`, `test_full_runner_two_fits_test_exclusion_and_preservation`, `test_changed_reference_artifact_fails_before_training`.

**[tests/test_lightgbm_cross_validation.py](../tests/test_lightgbm_cross_validation.py)**: `test_fold_isolation_and_weights_from_each_subset`, `test_continuity_counts_true_positive_runs_without_counting_fp_as_seizure`, `test_aggregation_unique_events_and_undefined_precision`, `test_runner_eight_independent_fits_and_no_reserved_data`, `test_fixed_setting_reproduction_and_artifact_hashes`, `test_changed_rf_artifact_rejected_before_fit`.

**[tests/test_raw_eeg_investigation.py](../tests/test_raw_eeg_investigation.py)**: `test_known_absolute_power_and_relative_gamma`, `test_amplitude_scaling_changes_absolute_not_relative_power`, `test_causal_prefix_equals_complete_filter_crop`, `test_incomplete_window_rejected`.

**[tests/test_preprocess.py](../tests/test_preprocess.py)**: `test_numbered_multiple_and_seizure_free_files`, `test_real_annotations`, `test_malformed_annotations_fail`, `test_interval_validation`, `test_half_open_boundaries_and_exact_half`, `test_multiple_seizures_overlap_union`, `test_known_boundary_windows_and_counts`, `test_channel_order_and_explicit_duplicate_alias`, `test_band_powers_partition_and_known_alpha_signal`, `test_flat_and_near_flat_channels_remain_finite`, `test_invalid_windows_fail`, `test_filter_does_not_use_future_samples_and_resets`, `test_exact_length_and_partial_tail`, `test_invalid_sampling_rates`, `test_completed_edf_header_lengths`, `test_rows_preserve_full_stream_and_metadata`, `test_out_of_recording_annotations_fail`, `test_legacy_output_is_protected`.

**[results/final_test_v1/test_final_outputs.py](../results/final_test_v1/test_final_outputs.py)**: `test_complete_alignment_metadata_and_natural_distribution`, `test_short_recording_and_exposure`, `test_finite_scores_and_frozen_threshold`, `test_reports_reproduce_from_saved_predictions`, `test_alarm_starts_rearming_matching_and_latency`, `test_undefined_seizure_free_metrics`, `test_original_inputs_sources_models_and_results_preserved`, `test_frozen_command_and_first_outputs_unchanged`.

### Verification and preservation record index

- [results/final_test_v1/verification.json](../results/final_test_v1/verification.json) — top-level fields: `verified_utc`, `git_head`, `protocol_sha256`, `execution_command_exactly_matches_frozen_document`, `primary_command_invocations`, `research_model_fits`, `post_evaluation_research_prediction_calls`, `existing_tests`, `saved_output_tests`, `historical_files_checked`, `changed_historical_files`, `original_final_output_checksums_unchanged`, `table_counts_match_original_reports`, `figures_visually_verified`, `supplementary_edfs_opened_or_evaluated`, `limitations`.

- [results/final_test_v1/pre_execution_verification.json](../results/final_test_v1/pre_execution_verification.json) — top-level fields: `created_utc`, `git_head`, `git_status_before`, `frozen_git_head`, `git_explanation`, `protocol_sha256`, `frozen_document_sha256`, `execution_command_sha256`, `all_frozen_hashes_matched`, `software_versions_matched`, `model_checks`, `training_inputs_sha256`, `previous_prediction_artifacts_checked`, `primary_raw_integrity`, `historical_files_snapshotted`, `new_output_directory_confirmed`, `predictions_generated_during_preflight`, `supplementary_recordings_accessed`.

- [results/lightgbm_cv_v1/continuity.json](../results/lightgbm_cv_v1/continuity.json) — top-level fields: `chb01_03`, `chb01_04`, `chb01_15`, `chb01_16`.

- [results/lightgbm_cv_v1/preservation.json](../results/lightgbm_cv_v1/preservation.json) — top-level fields: `src/raw_eeg_investigation.py`, `src/train_eegnet.py`, `src/preprocess.py`, `src/baseline_rf.py`, `src/__init__.py`, `src/eegnet_model.py`, `src/lightgbm_experiment.py`, `src/split_dataset.py`, `src/evaluate.py`, `src/failure_analysis.py`, `src/rf_cross_validation.py`, `src/weighted_rf.py`, `tests/test_baseline_rf.py`, `tests/test_failure_analysis.py`, `tests/test_evaluate.py`, `tests/test_rf_cross_validation.py`, `tests/test_weighted_rf.py`, `tests/test_split_dataset.py`, `tests/test_lightgbm_experiment.py`, `tests/test_raw_eeg_investigation.py`, `tests/test_preprocess.py`, `data/processed/chb01_detection_features.parquet`, `data/processed/.gitkeep`, `data/processed/chb01_features.parquet`, `data/processed/chb01_detection_features_11recordings_split.json`, `data/processed/chb01_detection_features_11recordings.json`, `data/processed/chb01_phase2a_preservation.json`, `data/processed/chb01_detection_features_11recordings.parquet`, `data/processed/chb01_detection_features.json`, `data/processed/chb01_detection_features_11recordings_split.parquet`, `results/failure_analysis_v1/scores_comparison.png`, `results/failure_analysis_v1/family_distributions.parquet`, `results/failure_analysis_v1/quality.json`, `results/failure_analysis_v1/analysis_tests.log`, `results/failure_analysis_v1/preservation.json`, `results/failure_analysis_v1/feature_timelines.png`, `results/failure_analysis_v1/score_periods.json`, `results/failure_analysis_v1/weighted_local_scores.parquet`, `results/failure_analysis_v1/ordinary_local_scores.parquet`, `results/failure_analysis_v1/scores_missed.png`, `results/failure_analysis_v1/evaluation_tests.log`, `results/failure_analysis_v1/artifact_sha256.json`, `results/failure_analysis_v1/configuration.json`, `results/failure_analysis_v1/feature_differences.parquet`, `results/failure_analysis_v1/feature_families.png`, `results/failure_analysis_v1/verification.json`, `results/failure_analysis_v1/alarms.json`, `results/failure_analysis_v1/local_feature_timeline.parquet`, `results/failure_analysis_v1/all_feature_distributions.parquet`, `results/baseline_v1/dummy_evaluation.json`, `results/baseline_v1/dummy.joblib`, `results/baseline_v1/random_forest.joblib`, `results/baseline_v1/artifact_sha256.json`, `results/baseline_v1/configuration.json`, `results/baseline_v1/random_forest_validation.parquet`, `results/baseline_v1/tests.log`, `results/baseline_v1/verification.json`, `results/baseline_v1/chb01_18_scores.png`, `results/baseline_v1/random_forest_evaluation.json`, `results/baseline_v1/dummy_validation.parquet`, `results/baseline_v1/random_forest_review_windows.parquet`, `results/raw_eeg_investigation_v1/waveforms_overview.png`, `results/raw_eeg_investigation_v1/window_measurements.parquet`, `results/raw_eeg_investigation_v1/waveforms_alarm_zoom.png`, `results/raw_eeg_investigation_v1/analysis_tests.log`, `results/raw_eeg_investigation_v1/preservation.json`, `results/raw_eeg_investigation_v1/evaluation_tests.log`, `results/raw_eeg_investigation_v1/artifact_sha256.json`, `results/raw_eeg_investigation_v1/configuration.json`, `results/raw_eeg_investigation_v1/verification.json`, `results/raw_eeg_investigation_v1/spectra_comparison.png`, `results/raw_eeg_investigation_v1/detected_seizure_context.png`, `results/rf_weighted_v1/ordinary_evaluation.json`, `results/rf_weighted_v1/weighted_validation.parquet`, `results/rf_weighted_v1/validation_comparison.parquet`, `results/rf_weighted_v1/preservation.json`, `results/rf_weighted_v1/weighted_evaluation.json`, `results/rf_weighted_v1/review_windows.parquet`, `results/rf_weighted_v1/chb01_18_comparison.png`, `results/rf_weighted_v1/artifact_sha256.json`, `results/rf_weighted_v1/configuration.json`, `results/rf_weighted_v1/tests.log`, `results/rf_weighted_v1/verification.json`, `results/rf_weighted_v1/weighted_rf.joblib`, `results/rf_cv_v1/summary.json`, `results/rf_cv_v1/preservation.json`, `results/rf_cv_v1/folds.json`, `results/rf_cv_v1/artifact_sha256.json`, `results/rf_cv_v1/configuration.json`, `results/rf_cv_v1/tests.log`, `results/rf_cv_v1/verification.json`, `results/lightgbm_v1/weighted_lgbm_validation.parquet`, `results/lightgbm_v1/window_comparison.parquet`, `results/lightgbm_v1/ordinary_lgbm_validation.parquet`, `results/lightgbm_v1/preservation.json`, `results/lightgbm_v1/ordinary_lgbm.joblib`, `results/lightgbm_v1/comparison.json`, `results/lightgbm_v1/weighted_lgbm_evaluation.json`, `results/lightgbm_v1/ordinary_lgbm_evaluation.json`, `results/lightgbm_v1/artifact_sha256.json`, `results/lightgbm_v1/configuration.json`, `results/lightgbm_v1/ordinary_lgbm.txt`, `results/lightgbm_v1/weighted_rf_evaluation.json`, `results/lightgbm_v1/weighted_lgbm.txt`, `results/lightgbm_v1/tests.log`, `results/lightgbm_v1/verification.json`, `results/lightgbm_v1/ordinary_rf_evaluation.json`, `results/lightgbm_v1/environment.json`, `results/lightgbm_v1/weighted_lgbm.joblib`, `results/lightgbm_v1/warnings.json`, `results/lightgbm_v1/chb01_18_four_model_scores.png`, `results/rf_cv_v1/fold_2_chb01_04/weighted_predictions.parquet`, `results/rf_cv_v1/fold_2_chb01_04/ordinary_evaluation.json`, `results/rf_cv_v1/fold_2_chb01_04/ordinary.joblib`, `results/rf_cv_v1/fold_2_chb01_04/weighted_evaluation.json`, `results/rf_cv_v1/fold_2_chb01_04/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_2_chb01_04/weighted.joblib`, `results/rf_cv_v1/fold_2_chb01_04/definition.json`, `results/rf_cv_v1/fold_4_chb01_16/weighted_predictions.parquet`, `results/rf_cv_v1/fold_4_chb01_16/ordinary_evaluation.json`, `results/rf_cv_v1/fold_4_chb01_16/ordinary.joblib`, `results/rf_cv_v1/fold_4_chb01_16/weighted_evaluation.json`, `results/rf_cv_v1/fold_4_chb01_16/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_4_chb01_16/weighted.joblib`, `results/rf_cv_v1/fold_4_chb01_16/definition.json`, `results/rf_cv_v1/fold_3_chb01_15/weighted_predictions.parquet`, `results/rf_cv_v1/fold_3_chb01_15/ordinary_evaluation.json`, `results/rf_cv_v1/fold_3_chb01_15/ordinary.joblib`, `results/rf_cv_v1/fold_3_chb01_15/weighted_evaluation.json`, `results/rf_cv_v1/fold_3_chb01_15/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_3_chb01_15/weighted.joblib`, `results/rf_cv_v1/fold_3_chb01_15/definition.json`, `results/rf_cv_v1/fold_1_chb01_03/weighted_predictions.parquet`, `results/rf_cv_v1/fold_1_chb01_03/ordinary_evaluation.json`, `results/rf_cv_v1/fold_1_chb01_03/ordinary.joblib`, `results/rf_cv_v1/fold_1_chb01_03/weighted_evaluation.json`, `results/rf_cv_v1/fold_1_chb01_03/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_1_chb01_03/weighted.joblib`, `results/rf_cv_v1/fold_1_chb01_03/definition.json`, `config/chb01_recordings.csv`, `requirements.txt`.

- [results/lightgbm_cv_v1/verification.json](../results/lightgbm_cv_v1/verification.json) — top-level fields: `lightgbm_fits`, `rf_fits`, `saved_lightgbm_reloads_identical`, `rf_reports_and_fold_inputs_unchanged`, `validation_test_supplementary_rows_used`, `weighted_vs_ordinary_changes`, `fitted_lightgbm_models`.

- [results/failure_analysis_v1/quality.json](../results/failure_analysis_v1/quality.json) — top-level fields: `chb01_01`, `chb01_03`, `chb01_04`, `chb01_14`, `chb01_15`, `chb01_16`.

- [results/failure_analysis_v1/preservation.json](../results/failure_analysis_v1/preservation.json) — top-level fields: `results/rf_cv_v1/summary.json`, `results/rf_cv_v1/preservation.json`, `results/rf_cv_v1/folds.json`, `results/rf_cv_v1/artifact_sha256.json`, `results/rf_cv_v1/configuration.json`, `results/rf_cv_v1/tests.log`, `results/rf_cv_v1/verification.json`, `results/rf_cv_v1/fold_2_chb01_04/weighted_predictions.parquet`, `results/rf_cv_v1/fold_2_chb01_04/ordinary_evaluation.json`, `results/rf_cv_v1/fold_2_chb01_04/ordinary.joblib`, `results/rf_cv_v1/fold_2_chb01_04/weighted_evaluation.json`, `results/rf_cv_v1/fold_2_chb01_04/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_2_chb01_04/weighted.joblib`, `results/rf_cv_v1/fold_2_chb01_04/definition.json`, `results/rf_cv_v1/fold_4_chb01_16/weighted_predictions.parquet`, `results/rf_cv_v1/fold_4_chb01_16/ordinary_evaluation.json`, `results/rf_cv_v1/fold_4_chb01_16/ordinary.joblib`, `results/rf_cv_v1/fold_4_chb01_16/weighted_evaluation.json`, `results/rf_cv_v1/fold_4_chb01_16/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_4_chb01_16/weighted.joblib`, `results/rf_cv_v1/fold_4_chb01_16/definition.json`, `results/rf_cv_v1/fold_3_chb01_15/weighted_predictions.parquet`, `results/rf_cv_v1/fold_3_chb01_15/ordinary_evaluation.json`, `results/rf_cv_v1/fold_3_chb01_15/ordinary.joblib`, `results/rf_cv_v1/fold_3_chb01_15/weighted_evaluation.json`, `results/rf_cv_v1/fold_3_chb01_15/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_3_chb01_15/weighted.joblib`, `results/rf_cv_v1/fold_3_chb01_15/definition.json`, `results/rf_cv_v1/fold_1_chb01_03/weighted_predictions.parquet`, `results/rf_cv_v1/fold_1_chb01_03/ordinary_evaluation.json`, `results/rf_cv_v1/fold_1_chb01_03/ordinary.joblib`, `results/rf_cv_v1/fold_1_chb01_03/weighted_evaluation.json`, `results/rf_cv_v1/fold_1_chb01_03/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_1_chb01_03/weighted.joblib`, `results/rf_cv_v1/fold_1_chb01_03/definition.json`, `data/processed/chb01_detection_features.parquet`, `data/processed/.gitkeep`, `data/processed/chb01_features.parquet`, `data/processed/chb01_detection_features_11recordings_split.json`, `data/processed/chb01_detection_features_11recordings.json`, `data/processed/chb01_phase2a_preservation.json`, `data/processed/chb01_detection_features_11recordings.parquet`, `data/processed/chb01_detection_features.json`, `data/processed/chb01_detection_features_11recordings_split.parquet`, `src/train_eegnet.py`, `src/preprocess.py`, `src/baseline_rf.py`, `src/__init__.py`, `src/eegnet_model.py`, `src/split_dataset.py`, `src/evaluate.py`, `src/rf_cross_validation.py`, `src/weighted_rf.py`, `tests/test_baseline_rf.py`, `tests/test_evaluate.py`, `tests/test_rf_cross_validation.py`, `tests/test_weighted_rf.py`, `tests/test_split_dataset.py`, `tests/test_preprocess.py`, `config/chb01_recordings.csv`.

- [results/failure_analysis_v1/verification.json](../results/failure_analysis_v1/verification.json) — top-level fields: `models_trained`, `saved_model_predictions_verified`, `evaluation_reports_reproduced`, `all_126_features_finite`, `protected_recordings_used`.

- [results/temporal_filtering_v1/preservation.json](../results/temporal_filtering_v1/preservation.json) — top-level fields: `src/raw_eeg_investigation.py`, `src/train_eegnet.py`, `src/preprocess.py`, `src/baseline_rf.py`, `src/lightgbm_cross_validation.py`, `src/__init__.py`, `src/eegnet_model.py`, `src/lightgbm_experiment.py`, `src/split_dataset.py`, `src/evaluate.py`, `src/failure_analysis.py`, `src/rf_cross_validation.py`, `src/weighted_rf.py`, `tests/test_baseline_rf.py`, `tests/test_failure_analysis.py`, `tests/test_evaluate.py`, `tests/test_rf_cross_validation.py`, `tests/test_weighted_rf.py`, `tests/test_split_dataset.py`, `tests/test_lightgbm_experiment.py`, `tests/test_lightgbm_cross_validation.py`, `tests/test_raw_eeg_investigation.py`, `tests/test_preprocess.py`, `data/processed/chb01_detection_features.parquet`, `data/processed/.gitkeep`, `data/processed/chb01_features.parquet`, `data/processed/chb01_detection_features_11recordings_split.json`, `data/processed/chb01_detection_features_11recordings.json`, `data/processed/chb01_phase2a_preservation.json`, `data/processed/chb01_detection_features_11recordings.parquet`, `data/processed/chb01_detection_features.json`, `data/processed/chb01_detection_features_11recordings_split.parquet`, `results/lightgbm_cv_v1/continuity.json`, `results/lightgbm_cv_v1/summary.json`, `results/lightgbm_cv_v1/preservation.json`, `results/lightgbm_cv_v1/folds.json`, `results/lightgbm_cv_v1/artifact_sha256.json`, `results/lightgbm_cv_v1/configuration.json`, `results/lightgbm_cv_v1/four_model_seizure_scores.png`, `results/lightgbm_cv_v1/tests.log`, `results/lightgbm_cv_v1/verification.json`, `results/lightgbm_cv_v1/warnings.json`, `results/failure_analysis_v1/scores_comparison.png`, `results/failure_analysis_v1/family_distributions.parquet`, `results/failure_analysis_v1/quality.json`, `results/failure_analysis_v1/analysis_tests.log`, `results/failure_analysis_v1/preservation.json`, `results/failure_analysis_v1/feature_timelines.png`, `results/failure_analysis_v1/score_periods.json`, `results/failure_analysis_v1/weighted_local_scores.parquet`, `results/failure_analysis_v1/ordinary_local_scores.parquet`, `results/failure_analysis_v1/scores_missed.png`, `results/failure_analysis_v1/evaluation_tests.log`, `results/failure_analysis_v1/artifact_sha256.json`, `results/failure_analysis_v1/configuration.json`, `results/failure_analysis_v1/feature_differences.parquet`, `results/failure_analysis_v1/feature_families.png`, `results/failure_analysis_v1/verification.json`, `results/failure_analysis_v1/alarms.json`, `results/failure_analysis_v1/local_feature_timeline.parquet`, `results/failure_analysis_v1/all_feature_distributions.parquet`, `results/baseline_v1/dummy_evaluation.json`, `results/baseline_v1/dummy.joblib`, `results/baseline_v1/random_forest.joblib`, `results/baseline_v1/artifact_sha256.json`, `results/baseline_v1/configuration.json`, `results/baseline_v1/random_forest_validation.parquet`, `results/baseline_v1/tests.log`, `results/baseline_v1/verification.json`, `results/baseline_v1/chb01_18_scores.png`, `results/baseline_v1/random_forest_evaluation.json`, `results/baseline_v1/dummy_validation.parquet`, `results/baseline_v1/random_forest_review_windows.parquet`, `results/raw_eeg_investigation_v1/waveforms_overview.png`, `results/raw_eeg_investigation_v1/window_measurements.parquet`, `results/raw_eeg_investigation_v1/waveforms_alarm_zoom.png`, `results/raw_eeg_investigation_v1/analysis_tests.log`, `results/raw_eeg_investigation_v1/preservation.json`, `results/raw_eeg_investigation_v1/evaluation_tests.log`, `results/raw_eeg_investigation_v1/artifact_sha256.json`, `results/raw_eeg_investigation_v1/configuration.json`, `results/raw_eeg_investigation_v1/verification.json`, `results/raw_eeg_investigation_v1/spectra_comparison.png`, `results/raw_eeg_investigation_v1/detected_seizure_context.png`, `results/rf_weighted_v1/ordinary_evaluation.json`, `results/rf_weighted_v1/weighted_validation.parquet`, `results/rf_weighted_v1/validation_comparison.parquet`, `results/rf_weighted_v1/preservation.json`, `results/rf_weighted_v1/weighted_evaluation.json`, `results/rf_weighted_v1/review_windows.parquet`, `results/rf_weighted_v1/chb01_18_comparison.png`, `results/rf_weighted_v1/artifact_sha256.json`, `results/rf_weighted_v1/configuration.json`, `results/rf_weighted_v1/tests.log`, `results/rf_weighted_v1/verification.json`, `results/rf_weighted_v1/weighted_rf.joblib`, `results/rf_cv_v1/summary.json`, `results/rf_cv_v1/preservation.json`, `results/rf_cv_v1/folds.json`, `results/rf_cv_v1/artifact_sha256.json`, `results/rf_cv_v1/configuration.json`, `results/rf_cv_v1/tests.log`, `results/rf_cv_v1/verification.json`, `results/lightgbm_v1/weighted_lgbm_validation.parquet`, `results/lightgbm_v1/window_comparison.parquet`, `results/lightgbm_v1/ordinary_lgbm_validation.parquet`, `results/lightgbm_v1/preservation.json`, `results/lightgbm_v1/ordinary_lgbm.joblib`, `results/lightgbm_v1/comparison.json`, `results/lightgbm_v1/weighted_lgbm_evaluation.json`, `results/lightgbm_v1/ordinary_lgbm_evaluation.json`, `results/lightgbm_v1/artifact_sha256.json`, `results/lightgbm_v1/configuration.json`, `results/lightgbm_v1/ordinary_lgbm.txt`, `results/lightgbm_v1/weighted_rf_evaluation.json`, `results/lightgbm_v1/weighted_lgbm.txt`, `results/lightgbm_v1/tests.log`, `results/lightgbm_v1/verification.json`, `results/lightgbm_v1/ordinary_rf_evaluation.json`, `results/lightgbm_v1/environment.json`, `results/lightgbm_v1/weighted_lgbm.joblib`, `results/lightgbm_v1/warnings.json`, `results/lightgbm_v1/chb01_18_four_model_scores.png`, `results/rf_cv_v1/fold_2_chb01_04/weighted_predictions.parquet`, `results/rf_cv_v1/fold_2_chb01_04/ordinary_evaluation.json`, `results/rf_cv_v1/fold_2_chb01_04/ordinary.joblib`, `results/rf_cv_v1/fold_2_chb01_04/weighted_evaluation.json`, `results/rf_cv_v1/fold_2_chb01_04/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_2_chb01_04/weighted.joblib`, `results/rf_cv_v1/fold_2_chb01_04/definition.json`, `results/rf_cv_v1/fold_4_chb01_16/weighted_predictions.parquet`, `results/rf_cv_v1/fold_4_chb01_16/ordinary_evaluation.json`, `results/rf_cv_v1/fold_4_chb01_16/ordinary.joblib`, `results/rf_cv_v1/fold_4_chb01_16/weighted_evaluation.json`, `results/rf_cv_v1/fold_4_chb01_16/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_4_chb01_16/weighted.joblib`, `results/rf_cv_v1/fold_4_chb01_16/definition.json`, `results/rf_cv_v1/fold_3_chb01_15/weighted_predictions.parquet`, `results/rf_cv_v1/fold_3_chb01_15/ordinary_evaluation.json`, `results/rf_cv_v1/fold_3_chb01_15/ordinary.joblib`, `results/rf_cv_v1/fold_3_chb01_15/weighted_evaluation.json`, `results/rf_cv_v1/fold_3_chb01_15/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_3_chb01_15/weighted.joblib`, `results/rf_cv_v1/fold_3_chb01_15/definition.json`, `results/rf_cv_v1/fold_1_chb01_03/weighted_predictions.parquet`, `results/rf_cv_v1/fold_1_chb01_03/ordinary_evaluation.json`, `results/rf_cv_v1/fold_1_chb01_03/ordinary.joblib`, `results/rf_cv_v1/fold_1_chb01_03/weighted_evaluation.json`, `results/rf_cv_v1/fold_1_chb01_03/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_1_chb01_03/weighted.joblib`, `results/rf_cv_v1/fold_1_chb01_03/definition.json`, `results/lightgbm_cv_v1/fold_2_chb01_04/weighted_lgbm_predictions.parquet`, `results/lightgbm_cv_v1/fold_2_chb01_04/ordinary_lgbm_predictions.parquet`, `results/lightgbm_cv_v1/fold_2_chb01_04/ordinary_lgbm.joblib`, `results/lightgbm_cv_v1/fold_2_chb01_04/weighted_lgbm_evaluation.json`, `results/lightgbm_cv_v1/fold_2_chb01_04/four_model_predictions.parquet`, `results/lightgbm_cv_v1/fold_2_chb01_04/ordinary_lgbm_evaluation.json`, `results/lightgbm_cv_v1/fold_2_chb01_04/ordinary_lgbm.txt`, `results/lightgbm_cv_v1/fold_2_chb01_04/weighted_rf_evaluation.json`, `results/lightgbm_cv_v1/fold_2_chb01_04/weighted_lgbm.txt`, `results/lightgbm_cv_v1/fold_2_chb01_04/ordinary_rf_evaluation.json`, `results/lightgbm_cv_v1/fold_2_chb01_04/weighted_lgbm.joblib`, `results/lightgbm_cv_v1/fold_2_chb01_04/definition.json`, `results/lightgbm_cv_v1/fold_4_chb01_16/weighted_lgbm_predictions.parquet`, `results/lightgbm_cv_v1/fold_4_chb01_16/ordinary_lgbm_predictions.parquet`, `results/lightgbm_cv_v1/fold_4_chb01_16/ordinary_lgbm.joblib`, `results/lightgbm_cv_v1/fold_4_chb01_16/weighted_lgbm_evaluation.json`, `results/lightgbm_cv_v1/fold_4_chb01_16/four_model_predictions.parquet`, `results/lightgbm_cv_v1/fold_4_chb01_16/ordinary_lgbm_evaluation.json`, `results/lightgbm_cv_v1/fold_4_chb01_16/ordinary_lgbm.txt`, `results/lightgbm_cv_v1/fold_4_chb01_16/weighted_rf_evaluation.json`, `results/lightgbm_cv_v1/fold_4_chb01_16/weighted_lgbm.txt`, `results/lightgbm_cv_v1/fold_4_chb01_16/ordinary_rf_evaluation.json`, `results/lightgbm_cv_v1/fold_4_chb01_16/weighted_lgbm.joblib`, `results/lightgbm_cv_v1/fold_4_chb01_16/definition.json`, `results/lightgbm_cv_v1/fold_3_chb01_15/weighted_lgbm_predictions.parquet`, `results/lightgbm_cv_v1/fold_3_chb01_15/ordinary_lgbm_predictions.parquet`, `results/lightgbm_cv_v1/fold_3_chb01_15/ordinary_lgbm.joblib`, `results/lightgbm_cv_v1/fold_3_chb01_15/weighted_lgbm_evaluation.json`, `results/lightgbm_cv_v1/fold_3_chb01_15/four_model_predictions.parquet`, `results/lightgbm_cv_v1/fold_3_chb01_15/ordinary_lgbm_evaluation.json`, `results/lightgbm_cv_v1/fold_3_chb01_15/ordinary_lgbm.txt`, `results/lightgbm_cv_v1/fold_3_chb01_15/weighted_rf_evaluation.json`, `results/lightgbm_cv_v1/fold_3_chb01_15/weighted_lgbm.txt`, `results/lightgbm_cv_v1/fold_3_chb01_15/ordinary_rf_evaluation.json`, `results/lightgbm_cv_v1/fold_3_chb01_15/weighted_lgbm.joblib`, `results/lightgbm_cv_v1/fold_3_chb01_15/definition.json`, `results/lightgbm_cv_v1/fold_1_chb01_03/weighted_lgbm_predictions.parquet`, `results/lightgbm_cv_v1/fold_1_chb01_03/ordinary_lgbm_predictions.parquet`, `results/lightgbm_cv_v1/fold_1_chb01_03/ordinary_lgbm.joblib`, `results/lightgbm_cv_v1/fold_1_chb01_03/weighted_lgbm_evaluation.json`, `results/lightgbm_cv_v1/fold_1_chb01_03/four_model_predictions.parquet`, `results/lightgbm_cv_v1/fold_1_chb01_03/ordinary_lgbm_evaluation.json`, `results/lightgbm_cv_v1/fold_1_chb01_03/ordinary_lgbm.txt`, `results/lightgbm_cv_v1/fold_1_chb01_03/weighted_rf_evaluation.json`, `results/lightgbm_cv_v1/fold_1_chb01_03/weighted_lgbm.txt`, `results/lightgbm_cv_v1/fold_1_chb01_03/ordinary_rf_evaluation.json`, `results/lightgbm_cv_v1/fold_1_chb01_03/weighted_lgbm.joblib`, `results/lightgbm_cv_v1/fold_1_chb01_03/definition.json`, `config/chb01_recordings.csv`, `requirements.txt`.

- [results/temporal_filtering_v1/verification.json](../results/temporal_filtering_v1/verification.json) — top-level fields: `model_fits`, `baseline_reports_reproduced`, `labels_metadata_scores_preserved`, `final_test_or_supplementary_rows_used`, `primary_recordings`, `secondary_recordings`, `independent_filter_formula_checks`, `tests_passed`, `tests_skipped`.

- [results/baseline_v1/verification.json](../results/baseline_v1/verification.json) — top-level fields: `models`, `tests_passed`, `tests_skipped`, `protected_files`, `reserved_test_predictions`.

- [results/raw_eeg_investigation_v1/preservation.json](../results/raw_eeg_investigation_v1/preservation.json) — top-level fields: `src/train_eegnet.py`, `src/preprocess.py`, `src/baseline_rf.py`, `src/__init__.py`, `src/eegnet_model.py`, `src/split_dataset.py`, `src/evaluate.py`, `src/failure_analysis.py`, `src/rf_cross_validation.py`, `src/weighted_rf.py`, `tests/test_baseline_rf.py`, `tests/test_failure_analysis.py`, `tests/test_evaluate.py`, `tests/test_rf_cross_validation.py`, `tests/test_weighted_rf.py`, `tests/test_split_dataset.py`, `tests/test_preprocess.py`, `data/processed/chb01_detection_features.parquet`, `data/processed/.gitkeep`, `data/processed/chb01_features.parquet`, `data/processed/chb01_detection_features_11recordings_split.json`, `data/processed/chb01_detection_features_11recordings.json`, `data/processed/chb01_phase2a_preservation.json`, `data/processed/chb01_detection_features_11recordings.parquet`, `data/processed/chb01_detection_features.json`, `data/processed/chb01_detection_features_11recordings_split.parquet`, `results/rf_cv_v1/summary.json`, `results/rf_cv_v1/preservation.json`, `results/rf_cv_v1/folds.json`, `results/rf_cv_v1/artifact_sha256.json`, `results/rf_cv_v1/configuration.json`, `results/rf_cv_v1/tests.log`, `results/rf_cv_v1/verification.json`, `results/rf_cv_v1/fold_2_chb01_04/weighted_predictions.parquet`, `results/rf_cv_v1/fold_2_chb01_04/ordinary_evaluation.json`, `results/rf_cv_v1/fold_2_chb01_04/ordinary.joblib`, `results/rf_cv_v1/fold_2_chb01_04/weighted_evaluation.json`, `results/rf_cv_v1/fold_2_chb01_04/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_2_chb01_04/weighted.joblib`, `results/rf_cv_v1/fold_2_chb01_04/definition.json`, `results/rf_cv_v1/fold_4_chb01_16/weighted_predictions.parquet`, `results/rf_cv_v1/fold_4_chb01_16/ordinary_evaluation.json`, `results/rf_cv_v1/fold_4_chb01_16/ordinary.joblib`, `results/rf_cv_v1/fold_4_chb01_16/weighted_evaluation.json`, `results/rf_cv_v1/fold_4_chb01_16/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_4_chb01_16/weighted.joblib`, `results/rf_cv_v1/fold_4_chb01_16/definition.json`, `results/rf_cv_v1/fold_3_chb01_15/weighted_predictions.parquet`, `results/rf_cv_v1/fold_3_chb01_15/ordinary_evaluation.json`, `results/rf_cv_v1/fold_3_chb01_15/ordinary.joblib`, `results/rf_cv_v1/fold_3_chb01_15/weighted_evaluation.json`, `results/rf_cv_v1/fold_3_chb01_15/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_3_chb01_15/weighted.joblib`, `results/rf_cv_v1/fold_3_chb01_15/definition.json`, `results/rf_cv_v1/fold_1_chb01_03/weighted_predictions.parquet`, `results/rf_cv_v1/fold_1_chb01_03/ordinary_evaluation.json`, `results/rf_cv_v1/fold_1_chb01_03/ordinary.joblib`, `results/rf_cv_v1/fold_1_chb01_03/weighted_evaluation.json`, `results/rf_cv_v1/fold_1_chb01_03/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_1_chb01_03/weighted.joblib`, `results/rf_cv_v1/fold_1_chb01_03/definition.json`, `results/failure_analysis_v1/scores_comparison.png`, `results/failure_analysis_v1/family_distributions.parquet`, `results/failure_analysis_v1/quality.json`, `results/failure_analysis_v1/analysis_tests.log`, `results/failure_analysis_v1/preservation.json`, `results/failure_analysis_v1/feature_timelines.png`, `results/failure_analysis_v1/score_periods.json`, `results/failure_analysis_v1/weighted_local_scores.parquet`, `results/failure_analysis_v1/ordinary_local_scores.parquet`, `results/failure_analysis_v1/scores_missed.png`, `results/failure_analysis_v1/evaluation_tests.log`, `results/failure_analysis_v1/artifact_sha256.json`, `results/failure_analysis_v1/configuration.json`, `results/failure_analysis_v1/feature_differences.parquet`, `results/failure_analysis_v1/feature_families.png`, `results/failure_analysis_v1/verification.json`, `results/failure_analysis_v1/alarms.json`, `results/failure_analysis_v1/local_feature_timeline.parquet`, `results/failure_analysis_v1/all_feature_distributions.parquet`, `config/chb01_recordings.csv`, `data/raw/chb01_03.edf`, `data/raw/chb01_04.edf`, `data/raw/chb01_15.edf`, `data/raw/chb01_16.edf`.

- [results/raw_eeg_investigation_v1/verification.json](../results/raw_eeg_investigation_v1/verification.json) — top-level fields: `chb01_15`, `chb01_03`, `chb01_04`, `chb01_16`.

- [results/rf_weighted_v1/preservation.json](../results/rf_weighted_v1/preservation.json) — top-level fields: `results/baseline_v1/dummy_evaluation.json`, `results/baseline_v1/dummy.joblib`, `results/baseline_v1/random_forest.joblib`, `results/baseline_v1/artifact_sha256.json`, `results/baseline_v1/configuration.json`, `results/baseline_v1/random_forest_validation.parquet`, `results/baseline_v1/tests.log`, `results/baseline_v1/verification.json`, `results/baseline_v1/chb01_18_scores.png`, `results/baseline_v1/random_forest_evaluation.json`, `results/baseline_v1/dummy_validation.parquet`, `results/baseline_v1/random_forest_review_windows.parquet`, `data/processed/chb01_detection_features.parquet`, `data/processed/.gitkeep`, `data/processed/chb01_features.parquet`, `data/processed/chb01_detection_features_11recordings_split.json`, `data/processed/chb01_detection_features_11recordings.json`, `data/processed/chb01_phase2a_preservation.json`, `data/processed/chb01_detection_features_11recordings.parquet`, `data/processed/chb01_detection_features.json`, `data/processed/chb01_detection_features_11recordings_split.parquet`, `src/train_eegnet.py`, `src/preprocess.py`, `src/baseline_rf.py`, `src/__init__.py`, `src/eegnet_model.py`, `src/split_dataset.py`, `src/evaluate.py`, `config/chb01_recordings.csv`.

- [results/rf_weighted_v1/verification.json](../results/rf_weighted_v1/verification.json) — top-level fields: `baseline_predictions_and_evaluation_unchanged`, `weighted_reload_identical`, `training_inputs_unchanged`, `test_rows_used`, `changed_windows`, `corrected_windows`, `regressed_windows`, `previous_misses_corrected`.

- [results/rf_cv_v1/preservation.json](../results/rf_cv_v1/preservation.json) — top-level fields: `src/train_eegnet.py`, `src/preprocess.py`, `src/baseline_rf.py`, `src/__init__.py`, `src/eegnet_model.py`, `src/split_dataset.py`, `src/evaluate.py`, `src/weighted_rf.py`, `tests/test_baseline_rf.py`, `tests/test_evaluate.py`, `tests/test_weighted_rf.py`, `tests/test_split_dataset.py`, `tests/test_preprocess.py`, `data/processed/chb01_detection_features.parquet`, `data/processed/.gitkeep`, `data/processed/chb01_features.parquet`, `data/processed/chb01_detection_features_11recordings_split.json`, `data/processed/chb01_detection_features_11recordings.json`, `data/processed/chb01_phase2a_preservation.json`, `data/processed/chb01_detection_features_11recordings.parquet`, `data/processed/chb01_detection_features.json`, `data/processed/chb01_detection_features_11recordings_split.parquet`, `results/baseline_v1/dummy_evaluation.json`, `results/baseline_v1/dummy.joblib`, `results/baseline_v1/random_forest.joblib`, `results/baseline_v1/artifact_sha256.json`, `results/baseline_v1/configuration.json`, `results/baseline_v1/random_forest_validation.parquet`, `results/baseline_v1/tests.log`, `results/baseline_v1/verification.json`, `results/baseline_v1/chb01_18_scores.png`, `results/baseline_v1/random_forest_evaluation.json`, `results/baseline_v1/dummy_validation.parquet`, `results/baseline_v1/random_forest_review_windows.parquet`, `results/rf_weighted_v1/ordinary_evaluation.json`, `results/rf_weighted_v1/weighted_validation.parquet`, `results/rf_weighted_v1/validation_comparison.parquet`, `results/rf_weighted_v1/preservation.json`, `results/rf_weighted_v1/weighted_evaluation.json`, `results/rf_weighted_v1/review_windows.parquet`, `results/rf_weighted_v1/chb01_18_comparison.png`, `results/rf_weighted_v1/artifact_sha256.json`, `results/rf_weighted_v1/configuration.json`, `results/rf_weighted_v1/tests.log`, `results/rf_weighted_v1/verification.json`, `results/rf_weighted_v1/weighted_rf.joblib`, `config/chb01_recordings.csv`.

- [results/rf_cv_v1/verification.json](../results/rf_cv_v1/verification.json) — top-level fields: `independent_models_fitted`, `reloads_identical`, `unique_evaluated_recordings_per_model`, `validation_test_or_new_recordings_used`, `training_inputs_unchanged`.

- [results/lightgbm_v1/preservation.json](../results/lightgbm_v1/preservation.json) — top-level fields: `src/raw_eeg_investigation.py`, `src/train_eegnet.py`, `src/preprocess.py`, `src/baseline_rf.py`, `src/__init__.py`, `src/eegnet_model.py`, `src/split_dataset.py`, `src/evaluate.py`, `src/failure_analysis.py`, `src/rf_cross_validation.py`, `src/weighted_rf.py`, `tests/test_baseline_rf.py`, `tests/test_failure_analysis.py`, `tests/test_evaluate.py`, `tests/test_rf_cross_validation.py`, `tests/test_weighted_rf.py`, `tests/test_split_dataset.py`, `tests/test_raw_eeg_investigation.py`, `tests/test_preprocess.py`, `data/processed/chb01_detection_features.parquet`, `data/processed/.gitkeep`, `data/processed/chb01_features.parquet`, `data/processed/chb01_detection_features_11recordings_split.json`, `data/processed/chb01_detection_features_11recordings.json`, `data/processed/chb01_phase2a_preservation.json`, `data/processed/chb01_detection_features_11recordings.parquet`, `data/processed/chb01_detection_features.json`, `data/processed/chb01_detection_features_11recordings_split.parquet`, `results/failure_analysis_v1/scores_comparison.png`, `results/failure_analysis_v1/family_distributions.parquet`, `results/failure_analysis_v1/quality.json`, `results/failure_analysis_v1/analysis_tests.log`, `results/failure_analysis_v1/preservation.json`, `results/failure_analysis_v1/feature_timelines.png`, `results/failure_analysis_v1/score_periods.json`, `results/failure_analysis_v1/weighted_local_scores.parquet`, `results/failure_analysis_v1/ordinary_local_scores.parquet`, `results/failure_analysis_v1/scores_missed.png`, `results/failure_analysis_v1/evaluation_tests.log`, `results/failure_analysis_v1/artifact_sha256.json`, `results/failure_analysis_v1/configuration.json`, `results/failure_analysis_v1/feature_differences.parquet`, `results/failure_analysis_v1/feature_families.png`, `results/failure_analysis_v1/verification.json`, `results/failure_analysis_v1/alarms.json`, `results/failure_analysis_v1/local_feature_timeline.parquet`, `results/failure_analysis_v1/all_feature_distributions.parquet`, `results/baseline_v1/dummy_evaluation.json`, `results/baseline_v1/dummy.joblib`, `results/baseline_v1/random_forest.joblib`, `results/baseline_v1/artifact_sha256.json`, `results/baseline_v1/configuration.json`, `results/baseline_v1/random_forest_validation.parquet`, `results/baseline_v1/tests.log`, `results/baseline_v1/verification.json`, `results/baseline_v1/chb01_18_scores.png`, `results/baseline_v1/random_forest_evaluation.json`, `results/baseline_v1/dummy_validation.parquet`, `results/baseline_v1/random_forest_review_windows.parquet`, `results/raw_eeg_investigation_v1/waveforms_overview.png`, `results/raw_eeg_investigation_v1/window_measurements.parquet`, `results/raw_eeg_investigation_v1/waveforms_alarm_zoom.png`, `results/raw_eeg_investigation_v1/analysis_tests.log`, `results/raw_eeg_investigation_v1/preservation.json`, `results/raw_eeg_investigation_v1/evaluation_tests.log`, `results/raw_eeg_investigation_v1/artifact_sha256.json`, `results/raw_eeg_investigation_v1/configuration.json`, `results/raw_eeg_investigation_v1/verification.json`, `results/raw_eeg_investigation_v1/spectra_comparison.png`, `results/raw_eeg_investigation_v1/detected_seizure_context.png`, `results/rf_weighted_v1/ordinary_evaluation.json`, `results/rf_weighted_v1/weighted_validation.parquet`, `results/rf_weighted_v1/validation_comparison.parquet`, `results/rf_weighted_v1/preservation.json`, `results/rf_weighted_v1/weighted_evaluation.json`, `results/rf_weighted_v1/review_windows.parquet`, `results/rf_weighted_v1/chb01_18_comparison.png`, `results/rf_weighted_v1/artifact_sha256.json`, `results/rf_weighted_v1/configuration.json`, `results/rf_weighted_v1/tests.log`, `results/rf_weighted_v1/verification.json`, `results/rf_weighted_v1/weighted_rf.joblib`, `results/rf_cv_v1/summary.json`, `results/rf_cv_v1/preservation.json`, `results/rf_cv_v1/folds.json`, `results/rf_cv_v1/artifact_sha256.json`, `results/rf_cv_v1/configuration.json`, `results/rf_cv_v1/tests.log`, `results/rf_cv_v1/verification.json`, `results/rf_cv_v1/fold_2_chb01_04/weighted_predictions.parquet`, `results/rf_cv_v1/fold_2_chb01_04/ordinary_evaluation.json`, `results/rf_cv_v1/fold_2_chb01_04/ordinary.joblib`, `results/rf_cv_v1/fold_2_chb01_04/weighted_evaluation.json`, `results/rf_cv_v1/fold_2_chb01_04/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_2_chb01_04/weighted.joblib`, `results/rf_cv_v1/fold_2_chb01_04/definition.json`, `results/rf_cv_v1/fold_4_chb01_16/weighted_predictions.parquet`, `results/rf_cv_v1/fold_4_chb01_16/ordinary_evaluation.json`, `results/rf_cv_v1/fold_4_chb01_16/ordinary.joblib`, `results/rf_cv_v1/fold_4_chb01_16/weighted_evaluation.json`, `results/rf_cv_v1/fold_4_chb01_16/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_4_chb01_16/weighted.joblib`, `results/rf_cv_v1/fold_4_chb01_16/definition.json`, `results/rf_cv_v1/fold_3_chb01_15/weighted_predictions.parquet`, `results/rf_cv_v1/fold_3_chb01_15/ordinary_evaluation.json`, `results/rf_cv_v1/fold_3_chb01_15/ordinary.joblib`, `results/rf_cv_v1/fold_3_chb01_15/weighted_evaluation.json`, `results/rf_cv_v1/fold_3_chb01_15/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_3_chb01_15/weighted.joblib`, `results/rf_cv_v1/fold_3_chb01_15/definition.json`, `results/rf_cv_v1/fold_1_chb01_03/weighted_predictions.parquet`, `results/rf_cv_v1/fold_1_chb01_03/ordinary_evaluation.json`, `results/rf_cv_v1/fold_1_chb01_03/ordinary.joblib`, `results/rf_cv_v1/fold_1_chb01_03/weighted_evaluation.json`, `results/rf_cv_v1/fold_1_chb01_03/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_1_chb01_03/weighted.joblib`, `results/rf_cv_v1/fold_1_chb01_03/definition.json`, `config/chb01_recordings.csv`.

- [results/lightgbm_v1/verification.json](../results/lightgbm_v1/verification.json) — top-level fields: `model_fits`, `positive_class_weight`, `rf_predictions_and_reports_unchanged`, `lightgbm_reloads_identical`, `test_or_supplementary_predictions`, `training_inputs_unchanged`, `prediction_changes`, `fitted_models`.

- [results/protocol_freeze_v1/verification.json](../results/protocol_freeze_v1/verification.json) — top-level fields: `protocol_sha256`, `selected_models`, `git`, `verification`.

- [results/protocol_freeze_v1/preservation_verification.json](../results/protocol_freeze_v1/preservation_verification.json) — top-level fields: `prior_artifacts_checked`, `changed_prior_artifacts`, `documented_python_blocks_syntax_checked_only`, `final_test_command_executed`, `supplementary_command_executed`, `protocol_sha256`, `reserved_raw_edf_files_opened_in_this_phase`, `training_edf_headers_read_by_existing_tests`.

- [data/processed/chb01_phase2a_preservation.json](../data/processed/chb01_phase2a_preservation.json) — top-level fields: `all_unchanged`, `files`.

## Appendix E. Figure inventory

| Figure path | Format | Explanation |

| --- | --- | --- |

| [results/baseline_v1/chb01_18_scores.png](../results/baseline_v1/chb01_18_scores.png) | .png | Baseline RF scores over the complete validation seizure recording and onset/offset zoom. |

| [results/failure_analysis_v1/feature_families.png](../results/failure_analysis_v1/feature_families.png) | .png | Remote, ictal and postictal distributions of seven feature families; descriptive channel medians. |

| [results/failure_analysis_v1/feature_timelines.png](../results/failure_analysis_v1/feature_timelines.png) | .png | Illustrative channel features around missed 15 and its postictal burst. |

| [results/failure_analysis_v1/scores_comparison.png](../results/failure_analysis_v1/scores_comparison.png) | .png | Ordinary/weighted RF scores around all four training-fold seizures. |

| [results/failure_analysis_v1/scores_missed.png](../results/failure_analysis_v1/scores_missed.png) | .png | Missed 15: full recording and local RF score/alarm view. |

| [results/final_test_v1/figures/alarm_timing.png](../results/final_test_v1/figures/alarm_timing.png) | .png | Exact final first/repeat alarm starts relative to onset; equal first-detection delays. |

| [results/final_test_v1/figures/alarm_timing.svg](../results/final_test_v1/figures/alarm_timing.svg) | .svg | Exact final first/repeat alarm starts relative to onset; equal first-detection delays. |

| [results/final_test_v1/figures/confusion_matrices.png](../results/final_test_v1/figures/confusion_matrices.png) | .png | Final test matrices: counts and within-true-class percentages. |

| [results/final_test_v1/figures/confusion_matrices.svg](../results/final_test_v1/figures/confusion_matrices.svg) | .svg | Final test matrices: counts and within-true-class percentages. |

| [results/final_test_v1/figures/event_false_alarm_summary.png](../results/final_test_v1/figures/event_false_alarm_summary.png) | .png | Final detected/missed events, observed false episodes and repeat alarms. |

| [results/final_test_v1/figures/event_false_alarm_summary.svg](../results/final_test_v1/figures/event_false_alarm_summary.svg) | .svg | Final detected/missed events, observed false episodes and repeat alarms. |

| [results/final_test_v1/figures/seizure_score_timelines.png](../results/final_test_v1/figures/seizure_score_timelines.png) | .png | Final model scores at actual decision times around both test events. |

| [results/final_test_v1/figures/seizure_score_timelines.svg](../results/final_test_v1/figures/seizure_score_timelines.svg) | .svg | Final model scores at actual decision times around both test events. |

| [results/lightgbm_cv_v1/four_model_seizure_scores.png](../results/lightgbm_cv_v1/four_model_seizure_scores.png) | .png | Four-model OOF scores around each of the four held-out training events. |

| [results/lightgbm_v1/chb01_18_four_model_scores.png](../results/lightgbm_v1/chb01_18_four_model_scores.png) | .png | Validation scores for ordinary/weighted RF and LightGBM, including late false alert. |

| [results/raw_eeg_investigation_v1/detected_seizure_context.png](../results/raw_eeg_investigation_v1/detected_seizure_context.png) | .png | Fixed midpoint examples from detected training events compared with 15. |

| [results/raw_eeg_investigation_v1/spectra_comparison.png](../results/raw_eeg_investigation_v1/spectra_comparison.png) | .png | Raw and filtered spectra for selected pre/ictal/postictal windows; absolute versus relative power context. |

| [results/raw_eeg_investigation_v1/waveforms_alarm_zoom.png](../results/raw_eeg_investigation_v1/waveforms_alarm_zoom.png) | .png | Raw/filtered 1775–1790-second zoom containing the delayed alarm window. |

| [results/raw_eeg_investigation_v1/waveforms_overview.png](../results/raw_eeg_investigation_v1/waveforms_overview.png) | .png | Raw and causally filtered EEG in 15 at 1700–1810 seconds; common physical scales. |

| [results/rf_weighted_v1/chb01_18_comparison.png](../results/rf_weighted_v1/chb01_18_comparison.png) | .png | Ordinary versus weighted RF validation scores and timing. |

| [results/temporal_filtering_v1/primary_cv/positive_decisions.png](../results/temporal_filtering_v1/primary_cv/positive_decisions.png) | .png | Raw and filtered positive window-end decisions; primary CV or secondary validation as named in parent directory. |

| [results/temporal_filtering_v1/secondary_validation/positive_decisions.png](../results/temporal_filtering_v1/secondary_validation/positive_decisions.png) | .png | Raw and filtered positive window-end decisions; primary CV or secondary validation as named in parent directory. |



PNG and SVG siblings are alternate exports of one figure, not separate experiments. Inventory is based on existing files and documented plotting code/captions; no figures were regenerated. Full numerical tables include observations outside plot zooms. Visual juxtaposition does not establish physiology, causality or calibrated probability.

## Appendix F. Artifact and implementation inventory

### Numerical data products

These linked tables preserve the complete row-level numerical evidence (including thousands of per-window/per-feature values that would be impractical to duplicate in narrative). The recorded row/column counts below are read from Parquet metadata or CSVs; no features or scores were recomputed. Source reports in Appendix C give descriptive numerical claims and period definitions.

| Saved table | Rows | Columns | Schema preview |

| --- | --- | --- | --- |

| [config/chb01_recordings.csv](../config/chb01_recordings.csv) | 11 | 3 | filename, split, annotation_source |

| [data/processed/chb01_detection_features.parquet](../data/processed/chb01_detection_features.parquet) | 2160 | 144 | FP1-F7_var, FP1-F7_line_length, FP1-F7_delta, FP1-F7_theta, FP1-F7_alpha, FP1-F7_beta, FP1-F7_gamma, F7-T7_var … |

| [data/processed/chb01_detection_features_11recordings.parquet](../data/processed/chb01_detection_features_11recordings.parquet) | 7665 | 144 | FP1-F7_var, FP1-F7_line_length, FP1-F7_delta, FP1-F7_theta, FP1-F7_alpha, FP1-F7_beta, FP1-F7_gamma, F7-T7_var … |

| [data/processed/chb01_detection_features_11recordings_split.parquet](../data/processed/chb01_detection_features_11recordings_split.parquet) | 7665 | 145 | FP1-F7_var, FP1-F7_line_length, FP1-F7_delta, FP1-F7_theta, FP1-F7_alpha, FP1-F7_beta, FP1-F7_gamma, F7-T7_var … |

| [data/processed/chb01_features.parquet](../data/processed/chb01_features.parquet) | 1612 | 130 | FP1-F7_var, FP1-F7_line_length, FP1-F7_delta, FP1-F7_theta, FP1-F7_alpha, FP1-F7_beta, FP1-F7_gamma, F7-T7_var … |

| [results/baseline_v1/dummy_validation.parquet](../results/baseline_v1/dummy_validation.parquet) | 1440 | 10 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/baseline_v1/random_forest_review_windows.parquet](../results/baseline_v1/random_forest_review_windows.parquet) | 18 | 10 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/baseline_v1/random_forest_validation.parquet](../results/baseline_v1/random_forest_validation.parquet) | 1440 | 10 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/failure_analysis_v1/all_feature_distributions.parquet](../results/failure_analysis_v1/all_feature_distributions.parquet) | 1512 | 10 | recording_id, period, feature, family, n, median, q25, q75 … |

| [results/failure_analysis_v1/family_distributions.parquet](../results/failure_analysis_v1/family_distributions.parquet) | 84 | 7 | recording_id, period, family, n, median, q25, q75 |

| [results/failure_analysis_v1/feature_differences.parquet](../results/failure_analysis_v1/feature_differences.parquet) | 126 | 10 | feature, target_ictal, other_event_median_ictal, target_remote, other_event_median_remote, target_ictal_minus_remote, other_event_median_change, target_ictal_over_remote … |

| [results/failure_analysis_v1/local_feature_timeline.parquet](../results/failure_analysis_v1/local_feature_timeline.parquet) | 32 | 12 | start_sec, end_sec, label, family_var, family_line_length, family_delta, family_theta, family_alpha … |

| [results/failure_analysis_v1/ordinary_local_scores.parquet](../results/failure_analysis_v1/ordinary_local_scores.parquet) | 32 | 10 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/failure_analysis_v1/weighted_local_scores.parquet](../results/failure_analysis_v1/weighted_local_scores.parquet) | 32 | 10 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/final_test_v1/alarm_episodes.csv](../results/final_test_v1/alarm_episodes.csv) | 8 | 11 | model_role, model, recording_id, alarm_id, start_sec, positive_windows, last_positive_decision_sec, end_sec, right_censored, status, seizure_id |

| [results/final_test_v1/dataset_counts.csv](../results/final_test_v1/dataset_counts.csv) | 3 | 7 | recording_id, duration_sec, hours, windows, class_0, class_1, seizures |

| [results/final_test_v1/development_comparison.csv](../results/final_test_v1/development_comparison.csv) | 6 | 18 | assessment, model, n_windows, n_positive, accuracy, precision, recall, specificity, f1, average_precision, n_seizures, n_detected, n_missed, n_false_alarms, false_alarms_per_hour, mean_latency_sec, n_repeat_alarms, evaluated_hours |

| [results/final_test_v1/event_metrics.csv](../results/final_test_v1/event_metrics.csv) | 8 | 17 | model_role, model, recording_id, n_seizures, n_detected, n_missed, event_sensitivity, n_false_alarms, evaluated_duration_sec, evaluated_hours, false_alarms_per_hour, latencies_sec, mean_latency_sec, median_latency_sec, n_alarms, n_repeat_alarms, undefined_metrics |

| [results/final_test_v1/prediction_differences.csv](../results/final_test_v1/prediction_differences.csv) | 4 | 8 | recording_id, start_sec, end_sec, label, lightgbm_prediction, random_forest_prediction, lightgbm_score, random_forest_score |

| [results/final_test_v1/primary_predictions.parquet](../results/final_test_v1/primary_predictions.parquet) | 1905 | 10 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/final_test_v1/reference_predictions.parquet](../results/final_test_v1/reference_predictions.parquet) | 1905 | 10 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/final_test_v1/seizure_outcomes.csv](../results/final_test_v1/seizure_outcomes.csv) | 4 | 10 | model_role, model, recording_id, seizure_id, onset_sec, offset_sec, detected, alarm_id, latency_sec, first_alarm_start_sec |

| [results/final_test_v1/window_metrics.csv](../results/final_test_v1/window_metrics.csv) | 8 | 18 | model_role, model, recording_id, n_windows, n_positive, positive_prevalence, tn, fp, fn, tp, confusion_matrix, accuracy, precision, recall, specificity, f1, average_precision, undefined_metrics |

| [results/lightgbm_cv_v1/fold_1_chb01_03/four_model_predictions.parquet](../results/lightgbm_cv_v1/fold_1_chb01_03/four_model_predictions.parquet) | 720 | 16 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/lightgbm_cv_v1/fold_1_chb01_03/ordinary_lgbm_predictions.parquet](../results/lightgbm_cv_v1/fold_1_chb01_03/ordinary_lgbm_predictions.parquet) | 720 | 10 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/lightgbm_cv_v1/fold_1_chb01_03/weighted_lgbm_predictions.parquet](../results/lightgbm_cv_v1/fold_1_chb01_03/weighted_lgbm_predictions.parquet) | 720 | 10 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/lightgbm_cv_v1/fold_2_chb01_04/four_model_predictions.parquet](../results/lightgbm_cv_v1/fold_2_chb01_04/four_model_predictions.parquet) | 720 | 16 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/lightgbm_cv_v1/fold_2_chb01_04/ordinary_lgbm_predictions.parquet](../results/lightgbm_cv_v1/fold_2_chb01_04/ordinary_lgbm_predictions.parquet) | 720 | 10 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/lightgbm_cv_v1/fold_2_chb01_04/weighted_lgbm_predictions.parquet](../results/lightgbm_cv_v1/fold_2_chb01_04/weighted_lgbm_predictions.parquet) | 720 | 10 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/lightgbm_cv_v1/fold_3_chb01_15/four_model_predictions.parquet](../results/lightgbm_cv_v1/fold_3_chb01_15/four_model_predictions.parquet) | 720 | 16 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/lightgbm_cv_v1/fold_3_chb01_15/ordinary_lgbm_predictions.parquet](../results/lightgbm_cv_v1/fold_3_chb01_15/ordinary_lgbm_predictions.parquet) | 720 | 10 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/lightgbm_cv_v1/fold_3_chb01_15/weighted_lgbm_predictions.parquet](../results/lightgbm_cv_v1/fold_3_chb01_15/weighted_lgbm_predictions.parquet) | 720 | 10 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/lightgbm_cv_v1/fold_4_chb01_16/four_model_predictions.parquet](../results/lightgbm_cv_v1/fold_4_chb01_16/four_model_predictions.parquet) | 720 | 16 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/lightgbm_cv_v1/fold_4_chb01_16/ordinary_lgbm_predictions.parquet](../results/lightgbm_cv_v1/fold_4_chb01_16/ordinary_lgbm_predictions.parquet) | 720 | 10 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/lightgbm_cv_v1/fold_4_chb01_16/weighted_lgbm_predictions.parquet](../results/lightgbm_cv_v1/fold_4_chb01_16/weighted_lgbm_predictions.parquet) | 720 | 10 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/lightgbm_v1/ordinary_lgbm_validation.parquet](../results/lightgbm_v1/ordinary_lgbm_validation.parquet) | 1440 | 10 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/lightgbm_v1/weighted_lgbm_validation.parquet](../results/lightgbm_v1/weighted_lgbm_validation.parquet) | 1440 | 10 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/lightgbm_v1/window_comparison.parquet](../results/lightgbm_v1/window_comparison.parquet) | 1440 | 16 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/raw_eeg_investigation_v1/window_measurements.parquet](../results/raw_eeg_investigation_v1/window_measurements.parquet) | 396 | 14 | recording_id, period, mode, channel, start_sec, end_sec, total_0p5_45_uv2, gamma_30_45_uv2 … |

| [results/rf_cv_v1/fold_1_chb01_03/ordinary_predictions.parquet](../results/rf_cv_v1/fold_1_chb01_03/ordinary_predictions.parquet) | 720 | 10 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/rf_cv_v1/fold_1_chb01_03/weighted_predictions.parquet](../results/rf_cv_v1/fold_1_chb01_03/weighted_predictions.parquet) | 720 | 10 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/rf_cv_v1/fold_2_chb01_04/ordinary_predictions.parquet](../results/rf_cv_v1/fold_2_chb01_04/ordinary_predictions.parquet) | 720 | 10 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/rf_cv_v1/fold_2_chb01_04/weighted_predictions.parquet](../results/rf_cv_v1/fold_2_chb01_04/weighted_predictions.parquet) | 720 | 10 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/rf_cv_v1/fold_3_chb01_15/ordinary_predictions.parquet](../results/rf_cv_v1/fold_3_chb01_15/ordinary_predictions.parquet) | 720 | 10 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/rf_cv_v1/fold_3_chb01_15/weighted_predictions.parquet](../results/rf_cv_v1/fold_3_chb01_15/weighted_predictions.parquet) | 720 | 10 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/rf_cv_v1/fold_4_chb01_16/ordinary_predictions.parquet](../results/rf_cv_v1/fold_4_chb01_16/ordinary_predictions.parquet) | 720 | 10 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/rf_cv_v1/fold_4_chb01_16/weighted_predictions.parquet](../results/rf_cv_v1/fold_4_chb01_16/weighted_predictions.parquet) | 720 | 10 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/rf_weighted_v1/review_windows.parquet](../results/rf_weighted_v1/review_windows.parquet) | 10 | 16 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/rf_weighted_v1/validation_comparison.parquet](../results/rf_weighted_v1/validation_comparison.parquet) | 1440 | 16 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/rf_weighted_v1/weighted_validation.parquet](../results/rf_weighted_v1/weighted_validation.parquet) | 1440 | 10 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/temporal_filtering_v1/primary_cv/changed_predictions.parquet](../results/temporal_filtering_v1/primary_cv/changed_predictions.parquet) | 34 | 14 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/temporal_filtering_v1/primary_cv/events.csv](../results/temporal_filtering_v1/primary_cv/events.csv) | 24 | 9 | model, rule, recording_id, seizure_id, onset_sec, offset_sec, detected, alarm_id, latency_sec |

| [results/temporal_filtering_v1/primary_cv/ordinary_lgbm_none_predictions.parquet](../results/temporal_filtering_v1/primary_cv/ordinary_lgbm_none_predictions.parquet) | 2880 | 11 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/temporal_filtering_v1/primary_cv/ordinary_lgbm_two_consecutive_predictions.parquet](../results/temporal_filtering_v1/primary_cv/ordinary_lgbm_two_consecutive_predictions.parquet) | 2880 | 11 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/temporal_filtering_v1/primary_cv/ordinary_lgbm_two_of_three_predictions.parquet](../results/temporal_filtering_v1/primary_cv/ordinary_lgbm_two_of_three_predictions.parquet) | 2880 | 11 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/temporal_filtering_v1/primary_cv/ordinary_rf_none_predictions.parquet](../results/temporal_filtering_v1/primary_cv/ordinary_rf_none_predictions.parquet) | 2880 | 11 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/temporal_filtering_v1/primary_cv/ordinary_rf_two_consecutive_predictions.parquet](../results/temporal_filtering_v1/primary_cv/ordinary_rf_two_consecutive_predictions.parquet) | 2880 | 11 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/temporal_filtering_v1/primary_cv/ordinary_rf_two_of_three_predictions.parquet](../results/temporal_filtering_v1/primary_cv/ordinary_rf_two_of_three_predictions.parquet) | 2880 | 11 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/temporal_filtering_v1/primary_cv/per_recording_metrics.csv](../results/temporal_filtering_v1/primary_cv/per_recording_metrics.csv) | 24 | 35 | model, rule, recording_id, n_windows, n_positive, positive_prevalence, tn, fp, fn, tp, confusion_matrix, accuracy, precision, recall, specificity, f1, average_precision, undefined_metrics, n_seizures, n_detected, n_missed, event_sensitivity, n_false_alarms, evaluated_duration_sec, evaluated_hours, false_alarms_per_hour, latencies_sec, mean_latency_sec, median_latency_sec, n_alarms, n_repeat_alarms, raw_positives, filtered_positives, added_positives, removed_positives |

| [results/temporal_filtering_v1/primary_cv/pooled_metrics.csv](../results/temporal_filtering_v1/primary_cv/pooled_metrics.csv) | 6 | 34 | model, rule, n_windows, n_positive, positive_prevalence, tn, fp, fn, tp, confusion_matrix, accuracy, precision, recall, specificity, f1, average_precision, undefined_metrics, n_seizures, n_detected, n_missed, event_sensitivity, n_false_alarms, evaluated_duration_sec, evaluated_hours, false_alarms_per_hour, latencies_sec, mean_latency_sec, median_latency_sec, n_alarms, n_repeat_alarms, raw_positives, filtered_positives, added_positives, removed_positives |

| [results/temporal_filtering_v1/secondary_validation/changed_predictions.parquet](../results/temporal_filtering_v1/secondary_validation/changed_predictions.parquet) | 11 | 14 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/temporal_filtering_v1/secondary_validation/events.csv](../results/temporal_filtering_v1/secondary_validation/events.csv) | 6 | 9 | model, rule, recording_id, seizure_id, onset_sec, offset_sec, detected, alarm_id, latency_sec |

| [results/temporal_filtering_v1/secondary_validation/ordinary_lgbm_none_predictions.parquet](../results/temporal_filtering_v1/secondary_validation/ordinary_lgbm_none_predictions.parquet) | 1440 | 11 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/temporal_filtering_v1/secondary_validation/ordinary_lgbm_two_consecutive_predictions.parquet](../results/temporal_filtering_v1/secondary_validation/ordinary_lgbm_two_consecutive_predictions.parquet) | 1440 | 11 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/temporal_filtering_v1/secondary_validation/ordinary_lgbm_two_of_three_predictions.parquet](../results/temporal_filtering_v1/secondary_validation/ordinary_lgbm_two_of_three_predictions.parquet) | 1440 | 11 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/temporal_filtering_v1/secondary_validation/ordinary_rf_none_predictions.parquet](../results/temporal_filtering_v1/secondary_validation/ordinary_rf_none_predictions.parquet) | 1440 | 11 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/temporal_filtering_v1/secondary_validation/ordinary_rf_two_consecutive_predictions.parquet](../results/temporal_filtering_v1/secondary_validation/ordinary_rf_two_consecutive_predictions.parquet) | 1440 | 11 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/temporal_filtering_v1/secondary_validation/ordinary_rf_two_of_three_predictions.parquet](../results/temporal_filtering_v1/secondary_validation/ordinary_rf_two_of_three_predictions.parquet) | 1440 | 11 | patient_id, recording_id, source_file, split, start_sec, end_sec, decision_time_sec, label … |

| [results/temporal_filtering_v1/secondary_validation/per_recording_metrics.csv](../results/temporal_filtering_v1/secondary_validation/per_recording_metrics.csv) | 12 | 35 | model, rule, recording_id, n_windows, n_positive, positive_prevalence, tn, fp, fn, tp, confusion_matrix, accuracy, precision, recall, specificity, f1, average_precision, undefined_metrics, n_seizures, n_detected, n_missed, event_sensitivity, n_false_alarms, evaluated_duration_sec, evaluated_hours, false_alarms_per_hour, latencies_sec, mean_latency_sec, median_latency_sec, n_alarms, n_repeat_alarms, raw_positives, filtered_positives, added_positives, removed_positives |

| [results/temporal_filtering_v1/secondary_validation/pooled_metrics.csv](../results/temporal_filtering_v1/secondary_validation/pooled_metrics.csv) | 6 | 34 | model, rule, n_windows, n_positive, positive_prevalence, tn, fp, fn, tp, confusion_matrix, accuracy, precision, recall, specificity, f1, average_precision, undefined_metrics, n_seizures, n_detected, n_missed, event_sensitivity, n_false_alarms, evaluated_duration_sec, evaluated_hours, false_alarms_per_hour, latencies_sec, mean_latency_sec, median_latency_sec, n_alarms, n_repeat_alarms, raw_positives, filtered_positives, added_positives, removed_positives |



### Model artifacts (read as bytes for hashes; never loaded)

| Artifact | Bytes | Current SHA-256 |

| --- | --- | --- |

| [results/baseline_v1/dummy.joblib](../results/baseline_v1/dummy.joblib) | 1003 | 3163817d5a9541c401affa64fce9d35b9c48600725f1d18b1fffa3e84e139e5c |

| [results/baseline_v1/random_forest.joblib](../results/baseline_v1/random_forest.joblib) | 95814 | 05fb53387283e82892aed6885469070fc85782f2155b460ba6b91cc4534d60c0 |

| [results/lightgbm_cv_v1/fold_1_chb01_03/ordinary_lgbm.joblib](../results/lightgbm_cv_v1/fold_1_chb01_03/ordinary_lgbm.joblib) | 49701 | a9835e324ff69359915fb60d017ce84662e7085e40bcbc96d92e8e47ab473652 |

| [results/lightgbm_cv_v1/fold_1_chb01_03/ordinary_lgbm.txt](../results/lightgbm_cv_v1/fold_1_chb01_03/ordinary_lgbm.txt) | 144246 | 27411c83f55ac41c119d3ce4eb9f1ec6792d154e3e8e2a0ac9dd9f1e4347462d |

| [results/lightgbm_cv_v1/fold_1_chb01_03/weighted_lgbm.joblib](../results/lightgbm_cv_v1/fold_1_chb01_03/weighted_lgbm.joblib) | 49445 | 4bdc3090b5880ccf82c367de5d39c88fd5ca0d17021280b4554b8dd943cb55c0 |

| [results/lightgbm_cv_v1/fold_1_chb01_03/weighted_lgbm.txt](../results/lightgbm_cv_v1/fold_1_chb01_03/weighted_lgbm.txt) | 146050 | 1feeaf22c85677b114552953aa58dac9435c43d6d071bccedc7b3cc80b70a203 |

| [results/lightgbm_cv_v1/fold_2_chb01_04/ordinary_lgbm.joblib](../results/lightgbm_cv_v1/fold_2_chb01_04/ordinary_lgbm.joblib) | 49124 | c82f6a26f483fd99370cd9a00a1d0eca6ca093712117c8eb9794540e1a21d053 |

| [results/lightgbm_cv_v1/fold_2_chb01_04/ordinary_lgbm.txt](../results/lightgbm_cv_v1/fold_2_chb01_04/ordinary_lgbm.txt) | 146227 | 8a00017b0514e563768d9535e93444f3153ad2797b7d6eaf4f913c0909f1dc6c |

| [results/lightgbm_cv_v1/fold_2_chb01_04/weighted_lgbm.joblib](../results/lightgbm_cv_v1/fold_2_chb01_04/weighted_lgbm.joblib) | 48442 | a5d727cb41154984a7e46994f2924d5abf9bc8f866e18bfb9a22a1bd980e210f |

| [results/lightgbm_cv_v1/fold_2_chb01_04/weighted_lgbm.txt](../results/lightgbm_cv_v1/fold_2_chb01_04/weighted_lgbm.txt) | 145014 | 50612b29b12ad2a80333e6106f8c36f54d71539bc6764e1333ffc01286889bdc |

| [results/lightgbm_cv_v1/fold_3_chb01_15/ordinary_lgbm.joblib](../results/lightgbm_cv_v1/fold_3_chb01_15/ordinary_lgbm.joblib) | 49124 | 47e47a811e6db8681406186013c9997dceec1eb29602199c1fbd8cd4ff02dc14 |

| [results/lightgbm_cv_v1/fold_3_chb01_15/ordinary_lgbm.txt](../results/lightgbm_cv_v1/fold_3_chb01_15/ordinary_lgbm.txt) | 146041 | 08e15cf9374db6c83f1fdeb25837ad78b232fbe561d7df967e697719c4cca395 |

| [results/lightgbm_cv_v1/fold_3_chb01_15/weighted_lgbm.joblib](../results/lightgbm_cv_v1/fold_3_chb01_15/weighted_lgbm.joblib) | 48374 | dbfec3898dc90aa02f21045ac590fbccf3746bd0b61ff1ba03468663e88d4d54 |

| [results/lightgbm_cv_v1/fold_3_chb01_15/weighted_lgbm.txt](../results/lightgbm_cv_v1/fold_3_chb01_15/weighted_lgbm.txt) | 143690 | a39e092234eb7314a480c809b1d798a3639790f81840dd2e3ef5e33553ae4999 |

| [results/lightgbm_cv_v1/fold_4_chb01_16/ordinary_lgbm.joblib](../results/lightgbm_cv_v1/fold_4_chb01_16/ordinary_lgbm.joblib) | 49649 | c5687a3a9369510740b0882b7cfd978589799a89ff0599d6649dec5a8fc53f3c |

| [results/lightgbm_cv_v1/fold_4_chb01_16/ordinary_lgbm.txt](../results/lightgbm_cv_v1/fold_4_chb01_16/ordinary_lgbm.txt) | 146847 | 7b0588ec83bcf67e472a66becc3c8890583da64c8450d7ba35e99217e8634aac |

| [results/lightgbm_cv_v1/fold_4_chb01_16/weighted_lgbm.joblib](../results/lightgbm_cv_v1/fold_4_chb01_16/weighted_lgbm.joblib) | 48264 | 73793721a34c051e99dd6d44552ade11f0bfa5ec6d251f88a6677abc67a429e3 |

| [results/lightgbm_cv_v1/fold_4_chb01_16/weighted_lgbm.txt](../results/lightgbm_cv_v1/fold_4_chb01_16/weighted_lgbm.txt) | 145333 | a747989a95c9df7cc771207bc3065f5000328338b94213f13617f04b473c0506 |

| [results/lightgbm_v1/ordinary_lgbm.joblib](../results/lightgbm_v1/ordinary_lgbm.joblib) | 48507 | 7324d3199850abd2e78d9190ff08946ecfd056e1a32e4b5d0f901d519b7b2b5f |

| [results/lightgbm_v1/ordinary_lgbm.txt](../results/lightgbm_v1/ordinary_lgbm.txt) | 144256 | 59cf62e6e74316288fcff37dfdfd58761202b522093313ee239df22f8d2e0183 |

| [results/lightgbm_v1/weighted_lgbm.joblib](../results/lightgbm_v1/weighted_lgbm.joblib) | 49215 | e51b9533115ee69c17b1db825a0f6be6f41f770c13bc3701a8b39dd0a017386c |

| [results/lightgbm_v1/weighted_lgbm.txt](../results/lightgbm_v1/weighted_lgbm.txt) | 145084 | ce12083b7893a94dccdb67170d6a218cd8a01f419c2183d14bd50ecce370992c |

| [results/rf_cv_v1/fold_1_chb01_03/ordinary.joblib](../results/rf_cv_v1/fold_1_chb01_03/ordinary.joblib) | 76409 | f9f50ed66684c8970716a62eca0492cd770771f687e1a6861e26103d5f9f455a |

| [results/rf_cv_v1/fold_1_chb01_03/weighted.joblib](../results/rf_cv_v1/fold_1_chb01_03/weighted.joblib) | 71706 | 8fedf0c8caf54d129e6387df0e06f83b93fe3e0c69b2035fc95abb3f14652660 |

| [results/rf_cv_v1/fold_2_chb01_04/ordinary.joblib](../results/rf_cv_v1/fold_2_chb01_04/ordinary.joblib) | 87883 | 694c23e011bc6599b41e226716bd8bdfca818ac4a7b1f810e417c1ce45819094 |

| [results/rf_cv_v1/fold_2_chb01_04/weighted.joblib](../results/rf_cv_v1/fold_2_chb01_04/weighted.joblib) | 91413 | fea8a435ff7fa6cbc2006ff70b7820ee0e009e87404943e268da2bc68b23ba93 |

| [results/rf_cv_v1/fold_3_chb01_15/ordinary.joblib](../results/rf_cv_v1/fold_3_chb01_15/ordinary.joblib) | 64388 | 848663bc9162d07414ccbf146a2a24161652da50554bf3484581135b26be0020 |

| [results/rf_cv_v1/fold_3_chb01_15/weighted.joblib](../results/rf_cv_v1/fold_3_chb01_15/weighted.joblib) | 64669 | fe2e32a5d912169c3feaf8f2b9c4dd10ff6c332274fb8d21f39d7946a3908fb9 |

| [results/rf_cv_v1/fold_4_chb01_16/ordinary.joblib](../results/rf_cv_v1/fold_4_chb01_16/ordinary.joblib) | 83876 | 531b75ed1087550de0c4adc36d14fe81e4b7fb76c10ac6b87fa3d770132f3adc |

| [results/rf_cv_v1/fold_4_chb01_16/weighted.joblib](../results/rf_cv_v1/fold_4_chb01_16/weighted.joblib) | 82378 | 02a7ebf7791df059a1dc66016e8f27c1d38966945d1d0663459a206c3d0b3477 |

| [results/rf_weighted_v1/weighted_rf.joblib](../results/rf_weighted_v1/weighted_rf.joblib) | 100493 | 54bfa11d7c4502bed5879adfda4fe8a48eea1499e3908de2e9a53244725f6cef |



### Source and scaffold inventory

- [src/raw_eeg_investigation.py](../src/raw_eeg_investigation.py) — Read-only targeted raw EEG investigation; no model fitting or signal cleaning. Functions/classes: window_spectrum, waveform_plot, run_analysis, main.

- [src/train_eegnet.py](../src/train_eegnet.py) — Empty placeholder; no implementation. Functions/classes: .

- [src/preprocess.py](../src/preprocess.py) — Extract patient-specific seizure DETECTION features from explicitly selected EDFs. Functions/classes: validate_seizures, parse_summary_file, select_channels, causal_bandpass, load_and_filter_edf, extract_epoch_features, window_bounds, window_annotation, slice_and_extract, sha256, main.

- [src/baseline_rf.py](../src/baseline_rf.py) — Phase 2B: fixed, unweighted baselines; train -> validation only. Functions/classes: select_features, load_development_data, make_models, fit_baselines, validation_predictions, score_summary, plot_scores, write_json, run_experiment, main.

- [src/lightgbm_cross_validation.py](../src/lightgbm_cross_validation.py) — Phase 2H: frozen LightGBM configurations on the four training-only RF folds. Functions/classes: ictal_continuity, load_rf_fold, plot_scores, run_experiment, main.

- [src/__init__.py](../src/__init__.py) — Empty placeholder; no implementation. Functions/classes: .

- [src/eegnet_model.py](../src/eegnet_model.py) — Empty placeholder; no implementation. Functions/classes: .

- [src/temporal_filter_experiment.py](../src/temporal_filter_experiment.py) — Phase 3A: saved-prediction filtering; primary CV then secondary validation. Functions/classes: ArtifactReader, verified_baseline, load_primary, load_secondary, evaluate_strategies, plot_events, run_experiment, main.

- [src/lightgbm_experiment.py](../src/lightgbm_experiment.py) — Phase 2G: two fixed LightGBM fits, validation-only comparison with saved RFs. Functions/classes: make_models, load_rf_reference, comparison_table, prediction_changes, plot_comparison, run_experiment, main.

- [src/split_dataset.py](../src/split_dataset.py) — Verify raw/processed EEG integrity and attach explicit recording-level splits. Functions/classes: load_manifest, _summary_metadata, _check_edf_size, verify_raw_recordings, assign_splits, verify_processed, main.

- [src/evaluate.py](../src/evaluate.py) — Offline evaluation of binary window decisions and seizure alarm episodes. Functions/classes: _vector, _binary, _ratio, window_metrics, _timeline, alarm_episodes, _seizure_intervals, _event_summary, event_metrics, evaluate_predictions.

- [src/failure_analysis.py](../src/failure_analysis.py) — Analysis only: saved out-of-fold scores and existing training EEG features. Functions/classes: score_periods, feature_periods, describe_scores, feature_descriptions, figures, run_analysis, main.

- [src/rf_cross_validation.py](../src/rf_cross_validation.py) — Four fixed training-only leave-one-seizure-recording-out RF comparisons. Functions/classes: check_training_scope, load_training_data, fold_data, predict_holdout, aggregate_predictions, run_experiment, main.

- [src/weighted_rf.py](../src/weighted_rf.py) — One controlled class-weight experiment, using Phase 2B artifacts read-only. Functions/classes: weighted_clone, training_fingerprint, compare_predictions, plot_comparison, run_experiment, main.

- [src/temporal_filter.py](../src/temporal_filter.py) — Fixed causal binary alert rules. No file access, fitting or label changes. Functions/classes: filter_binary, filter_predictions.

- [tests/test_baseline_rf.py](../tests/test_baseline_rf.py) — Leakage, alignment and reproducibility checks on tiny synthetic recordings. Functions/classes: synthetic_frame, small_models, BaselineTests, DataAndRunnerTests.

- [tests/test_failure_analysis.py](../tests/test_failure_analysis.py) — Hand-calculated descriptive periods and feature contrasts; no model fitting. Functions/classes: FailureAnalysisTests.

- [tests/test_temporal_filter.py](../tests/test_temporal_filter.py) — Hand-calculated causal rules and saved-prediction integration; no fitting. Functions/classes: recording, CausalRuleTests, TimelineAndEventTests, SavedArtifactTests.

- [tests/test_evaluate.py](../tests/test_evaluate.py) — Hand-calculated evaluation examples; no datasets, models, or EDF access. Functions/classes: events, WindowMetricTests, EventMetricTests, RecordingAggregationTests.

- [tests/test_rf_cross_validation.py](../tests/test_rf_cross_validation.py) — Training-only fold isolation, complete timelines and aggregation safeguards. Functions/classes: synthetic_data, tiny_models, FoldTests, LoaderAndRunnerTests.

- [tests/test_weighted_rf.py](../tests/test_weighted_rf.py) — Controlled comparison tests; only synthetic EEG-like feature tables. Functions/classes: ControlledComparisonTests, WeightedRunnerTests.

- [tests/test_split_dataset.py](../tests/test_split_dataset.py) — Split safeguards and expanded-data checks; no model training. Functions/classes: ManifestTests, IntegrityTests, ExpandedArtifactTests.

- [tests/test_lightgbm_experiment.py](../tests/test_lightgbm_experiment.py) — Synthetic-only checks for the fixed two-model LightGBM comparison. Functions/classes: synthetic_data, small_rf_models, LightGBMUnitTests, LightGBMRunnerTests.

- [tests/test_lightgbm_cross_validation.py](../tests/test_lightgbm_cross_validation.py) — Synthetic fold isolation, fold-local weights, alignment and aggregation checks. Functions/classes: synthetic_data, tiny_rf_models, CVUnitTests, CVRunnerTests.

- [tests/test_raw_eeg_investigation.py](../tests/test_raw_eeg_investigation.py) — Unit and band-integration checks on synthetic signals; no EDF/model access. Functions/classes: SpectrumTests.

- [tests/test_preprocess.py](../tests/test_preprocess.py) — Scientific invariants; unittest avoids adding a test dependency. Functions/classes: AnnotationTests, SignalTests.

- [app/dashboard.py](../app/dashboard.py) — Empty placeholder; no implementation. Functions/classes: .

- [results/final_test_v1/pre_execution_checks.py](../results/final_test_v1/pre_execution_checks.py) — Integrity checks only. No calls to model prediction or fitting methods. Functions/classes: .

- [results/final_test_v1/test_final_outputs.py](../results/final_test_v1/test_final_outputs.py) — Post-evaluation checks on saved outputs only; never load or predict with models. Functions/classes: FinalOutputTests.

- [results/final_test_v1/create_figures_and_tables.py](../results/final_test_v1/create_figures_and_tables.py) — Render the once-generated final predictions; no model loading or inference. Functions/classes: save, scopes, flat.

### Other numerical diagnostics, configuration and integrity sources

- [config/final_evaluation_protocol.json](../config/final_evaluation_protocol.json) — fields: `protocol_id`, `protocol_date`, `timezone`, `status`, `git`, `models`, `development`, `data`, `assessments`, `features`, `preprocessing`, `decisions`, `evaluation`, `software`, `integrity`, `verification`, `reporting`.

- [results/final_test_v1/execution_record.json](../results/final_test_v1/execution_record.json) — fields: `started_utc`, `finished_utc`, `returncode`, `command`, `command_sha256`, `extracted_verbatim_from`, `primary_command_invocations`, `research_model_fits`, `supplementary_evaluated`.

- [results/final_test_v1/final_evaluation_protocol.json](../results/final_test_v1/final_evaluation_protocol.json) — fields: `protocol_id`, `protocol_date`, `timezone`, `status`, `git`, `models`, `development`, `data`, `assessments`, `features`, `preprocessing`, `decisions`, `evaluation`, `software`, `integrity`, `verification`, `reporting`.

- [results/final_test_v1/verification.json](../results/final_test_v1/verification.json) — fields: `verified_utc`, `git_head`, `protocol_sha256`, `execution_command_exactly_matches_frozen_document`, `primary_command_invocations`, `research_model_fits`, `post_evaluation_research_prediction_calls`, `existing_tests`, `saved_output_tests`, `historical_files_checked`, `changed_historical_files`, `original_final_output_checksums_unchanged`, `table_counts_match_original_reports`, `figures_visually_verified`, `supplementary_edfs_opened_or_evaluated`, `limitations`.

- [results/final_test_v1/pre_execution_verification.json](../results/final_test_v1/pre_execution_verification.json) — fields: `created_utc`, `git_head`, `git_status_before`, `frozen_git_head`, `git_explanation`, `protocol_sha256`, `frozen_document_sha256`, `execution_command_sha256`, `all_frozen_hashes_matched`, `software_versions_matched`, `model_checks`, `training_inputs_sha256`, `previous_prediction_artifacts_checked`, `primary_raw_integrity`, `historical_files_snapshotted`, `new_output_directory_confirmed`, `predictions_generated_during_preflight`, `supplementary_recordings_accessed`.

- [results/lightgbm_cv_v1/continuity.json](../results/lightgbm_cv_v1/continuity.json) — fields: `chb01_03`, `chb01_04`, `chb01_15`, `chb01_16`.

- [results/lightgbm_cv_v1/preservation.json](../results/lightgbm_cv_v1/preservation.json) — fields: `src/raw_eeg_investigation.py`, `src/train_eegnet.py`, `src/preprocess.py`, `src/baseline_rf.py`, `src/__init__.py`, `src/eegnet_model.py`, `src/lightgbm_experiment.py`, `src/split_dataset.py`, `src/evaluate.py`, `src/failure_analysis.py`, `src/rf_cross_validation.py`, `src/weighted_rf.py`, `tests/test_baseline_rf.py`, `tests/test_failure_analysis.py`, `tests/test_evaluate.py`, `tests/test_rf_cross_validation.py`, `tests/test_weighted_rf.py`, `tests/test_split_dataset.py`, `tests/test_lightgbm_experiment.py`, `tests/test_raw_eeg_investigation.py`, `tests/test_preprocess.py`, `data/processed/chb01_detection_features.parquet`, `data/processed/.gitkeep`, `data/processed/chb01_features.parquet`, `data/processed/chb01_detection_features_11recordings_split.json`, `data/processed/chb01_detection_features_11recordings.json`, `data/processed/chb01_phase2a_preservation.json`, `data/processed/chb01_detection_features_11recordings.parquet`, `data/processed/chb01_detection_features.json`, `data/processed/chb01_detection_features_11recordings_split.parquet`, `results/failure_analysis_v1/scores_comparison.png`, `results/failure_analysis_v1/family_distributions.parquet`, `results/failure_analysis_v1/quality.json`, `results/failure_analysis_v1/analysis_tests.log`, `results/failure_analysis_v1/preservation.json`, `results/failure_analysis_v1/feature_timelines.png`, `results/failure_analysis_v1/score_periods.json`, `results/failure_analysis_v1/weighted_local_scores.parquet`, `results/failure_analysis_v1/ordinary_local_scores.parquet`, `results/failure_analysis_v1/scores_missed.png`, `results/failure_analysis_v1/evaluation_tests.log`, `results/failure_analysis_v1/artifact_sha256.json`, `results/failure_analysis_v1/configuration.json`, `results/failure_analysis_v1/feature_differences.parquet`, `results/failure_analysis_v1/feature_families.png`, `results/failure_analysis_v1/verification.json`, `results/failure_analysis_v1/alarms.json`, `results/failure_analysis_v1/local_feature_timeline.parquet`, `results/failure_analysis_v1/all_feature_distributions.parquet`, `results/baseline_v1/dummy_evaluation.json`, `results/baseline_v1/dummy.joblib`, `results/baseline_v1/random_forest.joblib`, `results/baseline_v1/artifact_sha256.json`, `results/baseline_v1/configuration.json`, `results/baseline_v1/random_forest_validation.parquet`, `results/baseline_v1/tests.log`, `results/baseline_v1/verification.json`, `results/baseline_v1/chb01_18_scores.png`, `results/baseline_v1/random_forest_evaluation.json`, `results/baseline_v1/dummy_validation.parquet`, `results/baseline_v1/random_forest_review_windows.parquet`, `results/raw_eeg_investigation_v1/waveforms_overview.png`, `results/raw_eeg_investigation_v1/window_measurements.parquet`, `results/raw_eeg_investigation_v1/waveforms_alarm_zoom.png`, `results/raw_eeg_investigation_v1/analysis_tests.log`, `results/raw_eeg_investigation_v1/preservation.json`, `results/raw_eeg_investigation_v1/evaluation_tests.log`, `results/raw_eeg_investigation_v1/artifact_sha256.json`, `results/raw_eeg_investigation_v1/configuration.json`, `results/raw_eeg_investigation_v1/verification.json`, `results/raw_eeg_investigation_v1/spectra_comparison.png`, `results/raw_eeg_investigation_v1/detected_seizure_context.png`, `results/rf_weighted_v1/ordinary_evaluation.json`, `results/rf_weighted_v1/weighted_validation.parquet`, `results/rf_weighted_v1/validation_comparison.parquet`, `results/rf_weighted_v1/preservation.json`, `results/rf_weighted_v1/weighted_evaluation.json`, `results/rf_weighted_v1/review_windows.parquet`, `results/rf_weighted_v1/chb01_18_comparison.png`, `results/rf_weighted_v1/artifact_sha256.json`, `results/rf_weighted_v1/configuration.json`, `results/rf_weighted_v1/tests.log`, `results/rf_weighted_v1/verification.json`, `results/rf_weighted_v1/weighted_rf.joblib`, `results/rf_cv_v1/summary.json`, `results/rf_cv_v1/preservation.json`, `results/rf_cv_v1/folds.json`, `results/rf_cv_v1/artifact_sha256.json`, `results/rf_cv_v1/configuration.json`, `results/rf_cv_v1/tests.log`, `results/rf_cv_v1/verification.json`, `results/lightgbm_v1/weighted_lgbm_validation.parquet`, `results/lightgbm_v1/window_comparison.parquet`, `results/lightgbm_v1/ordinary_lgbm_validation.parquet`, `results/lightgbm_v1/preservation.json`, `results/lightgbm_v1/ordinary_lgbm.joblib`, `results/lightgbm_v1/comparison.json`, `results/lightgbm_v1/weighted_lgbm_evaluation.json`, `results/lightgbm_v1/ordinary_lgbm_evaluation.json`, `results/lightgbm_v1/artifact_sha256.json`, `results/lightgbm_v1/configuration.json`, `results/lightgbm_v1/ordinary_lgbm.txt`, `results/lightgbm_v1/weighted_rf_evaluation.json`, `results/lightgbm_v1/weighted_lgbm.txt`, `results/lightgbm_v1/tests.log`, `results/lightgbm_v1/verification.json`, `results/lightgbm_v1/ordinary_rf_evaluation.json`, `results/lightgbm_v1/environment.json`, `results/lightgbm_v1/weighted_lgbm.joblib`, `results/lightgbm_v1/warnings.json`, `results/lightgbm_v1/chb01_18_four_model_scores.png`, `results/rf_cv_v1/fold_2_chb01_04/weighted_predictions.parquet`, `results/rf_cv_v1/fold_2_chb01_04/ordinary_evaluation.json`, `results/rf_cv_v1/fold_2_chb01_04/ordinary.joblib`, `results/rf_cv_v1/fold_2_chb01_04/weighted_evaluation.json`, `results/rf_cv_v1/fold_2_chb01_04/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_2_chb01_04/weighted.joblib`, `results/rf_cv_v1/fold_2_chb01_04/definition.json`, `results/rf_cv_v1/fold_4_chb01_16/weighted_predictions.parquet`, `results/rf_cv_v1/fold_4_chb01_16/ordinary_evaluation.json`, `results/rf_cv_v1/fold_4_chb01_16/ordinary.joblib`, `results/rf_cv_v1/fold_4_chb01_16/weighted_evaluation.json`, `results/rf_cv_v1/fold_4_chb01_16/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_4_chb01_16/weighted.joblib`, `results/rf_cv_v1/fold_4_chb01_16/definition.json`, `results/rf_cv_v1/fold_3_chb01_15/weighted_predictions.parquet`, `results/rf_cv_v1/fold_3_chb01_15/ordinary_evaluation.json`, `results/rf_cv_v1/fold_3_chb01_15/ordinary.joblib`, `results/rf_cv_v1/fold_3_chb01_15/weighted_evaluation.json`, `results/rf_cv_v1/fold_3_chb01_15/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_3_chb01_15/weighted.joblib`, `results/rf_cv_v1/fold_3_chb01_15/definition.json`, `results/rf_cv_v1/fold_1_chb01_03/weighted_predictions.parquet`, `results/rf_cv_v1/fold_1_chb01_03/ordinary_evaluation.json`, `results/rf_cv_v1/fold_1_chb01_03/ordinary.joblib`, `results/rf_cv_v1/fold_1_chb01_03/weighted_evaluation.json`, `results/rf_cv_v1/fold_1_chb01_03/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_1_chb01_03/weighted.joblib`, `results/rf_cv_v1/fold_1_chb01_03/definition.json`, `config/chb01_recordings.csv`, `requirements.txt`.

- [results/lightgbm_cv_v1/folds.json](../results/lightgbm_cv_v1/folds.json) — fields: `list records: 4`.

- [results/lightgbm_cv_v1/verification.json](../results/lightgbm_cv_v1/verification.json) — fields: `lightgbm_fits`, `rf_fits`, `saved_lightgbm_reloads_identical`, `rf_reports_and_fold_inputs_unchanged`, `validation_test_supplementary_rows_used`, `weighted_vs_ordinary_changes`, `fitted_lightgbm_models`.

- [results/lightgbm_cv_v1/warnings.json](../results/lightgbm_cv_v1/warnings.json) — fields: `list records: 0`.

- [results/failure_analysis_v1/quality.json](../results/failure_analysis_v1/quality.json) — fields: `chb01_01`, `chb01_03`, `chb01_04`, `chb01_14`, `chb01_15`, `chb01_16`.

- [results/failure_analysis_v1/preservation.json](../results/failure_analysis_v1/preservation.json) — fields: `results/rf_cv_v1/summary.json`, `results/rf_cv_v1/preservation.json`, `results/rf_cv_v1/folds.json`, `results/rf_cv_v1/artifact_sha256.json`, `results/rf_cv_v1/configuration.json`, `results/rf_cv_v1/tests.log`, `results/rf_cv_v1/verification.json`, `results/rf_cv_v1/fold_2_chb01_04/weighted_predictions.parquet`, `results/rf_cv_v1/fold_2_chb01_04/ordinary_evaluation.json`, `results/rf_cv_v1/fold_2_chb01_04/ordinary.joblib`, `results/rf_cv_v1/fold_2_chb01_04/weighted_evaluation.json`, `results/rf_cv_v1/fold_2_chb01_04/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_2_chb01_04/weighted.joblib`, `results/rf_cv_v1/fold_2_chb01_04/definition.json`, `results/rf_cv_v1/fold_4_chb01_16/weighted_predictions.parquet`, `results/rf_cv_v1/fold_4_chb01_16/ordinary_evaluation.json`, `results/rf_cv_v1/fold_4_chb01_16/ordinary.joblib`, `results/rf_cv_v1/fold_4_chb01_16/weighted_evaluation.json`, `results/rf_cv_v1/fold_4_chb01_16/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_4_chb01_16/weighted.joblib`, `results/rf_cv_v1/fold_4_chb01_16/definition.json`, `results/rf_cv_v1/fold_3_chb01_15/weighted_predictions.parquet`, `results/rf_cv_v1/fold_3_chb01_15/ordinary_evaluation.json`, `results/rf_cv_v1/fold_3_chb01_15/ordinary.joblib`, `results/rf_cv_v1/fold_3_chb01_15/weighted_evaluation.json`, `results/rf_cv_v1/fold_3_chb01_15/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_3_chb01_15/weighted.joblib`, `results/rf_cv_v1/fold_3_chb01_15/definition.json`, `results/rf_cv_v1/fold_1_chb01_03/weighted_predictions.parquet`, `results/rf_cv_v1/fold_1_chb01_03/ordinary_evaluation.json`, `results/rf_cv_v1/fold_1_chb01_03/ordinary.joblib`, `results/rf_cv_v1/fold_1_chb01_03/weighted_evaluation.json`, `results/rf_cv_v1/fold_1_chb01_03/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_1_chb01_03/weighted.joblib`, `results/rf_cv_v1/fold_1_chb01_03/definition.json`, `data/processed/chb01_detection_features.parquet`, `data/processed/.gitkeep`, `data/processed/chb01_features.parquet`, `data/processed/chb01_detection_features_11recordings_split.json`, `data/processed/chb01_detection_features_11recordings.json`, `data/processed/chb01_phase2a_preservation.json`, `data/processed/chb01_detection_features_11recordings.parquet`, `data/processed/chb01_detection_features.json`, `data/processed/chb01_detection_features_11recordings_split.parquet`, `src/train_eegnet.py`, `src/preprocess.py`, `src/baseline_rf.py`, `src/__init__.py`, `src/eegnet_model.py`, `src/split_dataset.py`, `src/evaluate.py`, `src/rf_cross_validation.py`, `src/weighted_rf.py`, `tests/test_baseline_rf.py`, `tests/test_evaluate.py`, `tests/test_rf_cross_validation.py`, `tests/test_weighted_rf.py`, `tests/test_split_dataset.py`, `tests/test_preprocess.py`, `config/chb01_recordings.csv`.

- [results/failure_analysis_v1/score_periods.json](../results/failure_analysis_v1/score_periods.json) — fields: `list records: 32`.

- [results/failure_analysis_v1/verification.json](../results/failure_analysis_v1/verification.json) — fields: `models_trained`, `saved_model_predictions_verified`, `evaluation_reports_reproduced`, `all_126_features_finite`, `protected_recordings_used`.

- [results/failure_analysis_v1/alarms.json](../results/failure_analysis_v1/alarms.json) — fields: `list records: 15`.

- [results/temporal_filtering_v1/preservation.json](../results/temporal_filtering_v1/preservation.json) — fields: `src/raw_eeg_investigation.py`, `src/train_eegnet.py`, `src/preprocess.py`, `src/baseline_rf.py`, `src/lightgbm_cross_validation.py`, `src/__init__.py`, `src/eegnet_model.py`, `src/lightgbm_experiment.py`, `src/split_dataset.py`, `src/evaluate.py`, `src/failure_analysis.py`, `src/rf_cross_validation.py`, `src/weighted_rf.py`, `tests/test_baseline_rf.py`, `tests/test_failure_analysis.py`, `tests/test_evaluate.py`, `tests/test_rf_cross_validation.py`, `tests/test_weighted_rf.py`, `tests/test_split_dataset.py`, `tests/test_lightgbm_experiment.py`, `tests/test_lightgbm_cross_validation.py`, `tests/test_raw_eeg_investigation.py`, `tests/test_preprocess.py`, `data/processed/chb01_detection_features.parquet`, `data/processed/.gitkeep`, `data/processed/chb01_features.parquet`, `data/processed/chb01_detection_features_11recordings_split.json`, `data/processed/chb01_detection_features_11recordings.json`, `data/processed/chb01_phase2a_preservation.json`, `data/processed/chb01_detection_features_11recordings.parquet`, `data/processed/chb01_detection_features.json`, `data/processed/chb01_detection_features_11recordings_split.parquet`, `results/lightgbm_cv_v1/continuity.json`, `results/lightgbm_cv_v1/summary.json`, `results/lightgbm_cv_v1/preservation.json`, `results/lightgbm_cv_v1/folds.json`, `results/lightgbm_cv_v1/artifact_sha256.json`, `results/lightgbm_cv_v1/configuration.json`, `results/lightgbm_cv_v1/four_model_seizure_scores.png`, `results/lightgbm_cv_v1/tests.log`, `results/lightgbm_cv_v1/verification.json`, `results/lightgbm_cv_v1/warnings.json`, `results/failure_analysis_v1/scores_comparison.png`, `results/failure_analysis_v1/family_distributions.parquet`, `results/failure_analysis_v1/quality.json`, `results/failure_analysis_v1/analysis_tests.log`, `results/failure_analysis_v1/preservation.json`, `results/failure_analysis_v1/feature_timelines.png`, `results/failure_analysis_v1/score_periods.json`, `results/failure_analysis_v1/weighted_local_scores.parquet`, `results/failure_analysis_v1/ordinary_local_scores.parquet`, `results/failure_analysis_v1/scores_missed.png`, `results/failure_analysis_v1/evaluation_tests.log`, `results/failure_analysis_v1/artifact_sha256.json`, `results/failure_analysis_v1/configuration.json`, `results/failure_analysis_v1/feature_differences.parquet`, `results/failure_analysis_v1/feature_families.png`, `results/failure_analysis_v1/verification.json`, `results/failure_analysis_v1/alarms.json`, `results/failure_analysis_v1/local_feature_timeline.parquet`, `results/failure_analysis_v1/all_feature_distributions.parquet`, `results/baseline_v1/dummy_evaluation.json`, `results/baseline_v1/dummy.joblib`, `results/baseline_v1/random_forest.joblib`, `results/baseline_v1/artifact_sha256.json`, `results/baseline_v1/configuration.json`, `results/baseline_v1/random_forest_validation.parquet`, `results/baseline_v1/tests.log`, `results/baseline_v1/verification.json`, `results/baseline_v1/chb01_18_scores.png`, `results/baseline_v1/random_forest_evaluation.json`, `results/baseline_v1/dummy_validation.parquet`, `results/baseline_v1/random_forest_review_windows.parquet`, `results/raw_eeg_investigation_v1/waveforms_overview.png`, `results/raw_eeg_investigation_v1/window_measurements.parquet`, `results/raw_eeg_investigation_v1/waveforms_alarm_zoom.png`, `results/raw_eeg_investigation_v1/analysis_tests.log`, `results/raw_eeg_investigation_v1/preservation.json`, `results/raw_eeg_investigation_v1/evaluation_tests.log`, `results/raw_eeg_investigation_v1/artifact_sha256.json`, `results/raw_eeg_investigation_v1/configuration.json`, `results/raw_eeg_investigation_v1/verification.json`, `results/raw_eeg_investigation_v1/spectra_comparison.png`, `results/raw_eeg_investigation_v1/detected_seizure_context.png`, `results/rf_weighted_v1/ordinary_evaluation.json`, `results/rf_weighted_v1/weighted_validation.parquet`, `results/rf_weighted_v1/validation_comparison.parquet`, `results/rf_weighted_v1/preservation.json`, `results/rf_weighted_v1/weighted_evaluation.json`, `results/rf_weighted_v1/review_windows.parquet`, `results/rf_weighted_v1/chb01_18_comparison.png`, `results/rf_weighted_v1/artifact_sha256.json`, `results/rf_weighted_v1/configuration.json`, `results/rf_weighted_v1/tests.log`, `results/rf_weighted_v1/verification.json`, `results/rf_weighted_v1/weighted_rf.joblib`, `results/rf_cv_v1/summary.json`, `results/rf_cv_v1/preservation.json`, `results/rf_cv_v1/folds.json`, `results/rf_cv_v1/artifact_sha256.json`, `results/rf_cv_v1/configuration.json`, `results/rf_cv_v1/tests.log`, `results/rf_cv_v1/verification.json`, `results/lightgbm_v1/weighted_lgbm_validation.parquet`, `results/lightgbm_v1/window_comparison.parquet`, `results/lightgbm_v1/ordinary_lgbm_validation.parquet`, `results/lightgbm_v1/preservation.json`, `results/lightgbm_v1/ordinary_lgbm.joblib`, `results/lightgbm_v1/comparison.json`, `results/lightgbm_v1/weighted_lgbm_evaluation.json`, `results/lightgbm_v1/ordinary_lgbm_evaluation.json`, `results/lightgbm_v1/artifact_sha256.json`, `results/lightgbm_v1/configuration.json`, `results/lightgbm_v1/ordinary_lgbm.txt`, `results/lightgbm_v1/weighted_rf_evaluation.json`, `results/lightgbm_v1/weighted_lgbm.txt`, `results/lightgbm_v1/tests.log`, `results/lightgbm_v1/verification.json`, `results/lightgbm_v1/ordinary_rf_evaluation.json`, `results/lightgbm_v1/environment.json`, `results/lightgbm_v1/weighted_lgbm.joblib`, `results/lightgbm_v1/warnings.json`, `results/lightgbm_v1/chb01_18_four_model_scores.png`, `results/rf_cv_v1/fold_2_chb01_04/weighted_predictions.parquet`, `results/rf_cv_v1/fold_2_chb01_04/ordinary_evaluation.json`, `results/rf_cv_v1/fold_2_chb01_04/ordinary.joblib`, `results/rf_cv_v1/fold_2_chb01_04/weighted_evaluation.json`, `results/rf_cv_v1/fold_2_chb01_04/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_2_chb01_04/weighted.joblib`, `results/rf_cv_v1/fold_2_chb01_04/definition.json`, `results/rf_cv_v1/fold_4_chb01_16/weighted_predictions.parquet`, `results/rf_cv_v1/fold_4_chb01_16/ordinary_evaluation.json`, `results/rf_cv_v1/fold_4_chb01_16/ordinary.joblib`, `results/rf_cv_v1/fold_4_chb01_16/weighted_evaluation.json`, `results/rf_cv_v1/fold_4_chb01_16/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_4_chb01_16/weighted.joblib`, `results/rf_cv_v1/fold_4_chb01_16/definition.json`, `results/rf_cv_v1/fold_3_chb01_15/weighted_predictions.parquet`, `results/rf_cv_v1/fold_3_chb01_15/ordinary_evaluation.json`, `results/rf_cv_v1/fold_3_chb01_15/ordinary.joblib`, `results/rf_cv_v1/fold_3_chb01_15/weighted_evaluation.json`, `results/rf_cv_v1/fold_3_chb01_15/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_3_chb01_15/weighted.joblib`, `results/rf_cv_v1/fold_3_chb01_15/definition.json`, `results/rf_cv_v1/fold_1_chb01_03/weighted_predictions.parquet`, `results/rf_cv_v1/fold_1_chb01_03/ordinary_evaluation.json`, `results/rf_cv_v1/fold_1_chb01_03/ordinary.joblib`, `results/rf_cv_v1/fold_1_chb01_03/weighted_evaluation.json`, `results/rf_cv_v1/fold_1_chb01_03/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_1_chb01_03/weighted.joblib`, `results/rf_cv_v1/fold_1_chb01_03/definition.json`, `results/lightgbm_cv_v1/fold_2_chb01_04/weighted_lgbm_predictions.parquet`, `results/lightgbm_cv_v1/fold_2_chb01_04/ordinary_lgbm_predictions.parquet`, `results/lightgbm_cv_v1/fold_2_chb01_04/ordinary_lgbm.joblib`, `results/lightgbm_cv_v1/fold_2_chb01_04/weighted_lgbm_evaluation.json`, `results/lightgbm_cv_v1/fold_2_chb01_04/four_model_predictions.parquet`, `results/lightgbm_cv_v1/fold_2_chb01_04/ordinary_lgbm_evaluation.json`, `results/lightgbm_cv_v1/fold_2_chb01_04/ordinary_lgbm.txt`, `results/lightgbm_cv_v1/fold_2_chb01_04/weighted_rf_evaluation.json`, `results/lightgbm_cv_v1/fold_2_chb01_04/weighted_lgbm.txt`, `results/lightgbm_cv_v1/fold_2_chb01_04/ordinary_rf_evaluation.json`, `results/lightgbm_cv_v1/fold_2_chb01_04/weighted_lgbm.joblib`, `results/lightgbm_cv_v1/fold_2_chb01_04/definition.json`, `results/lightgbm_cv_v1/fold_4_chb01_16/weighted_lgbm_predictions.parquet`, `results/lightgbm_cv_v1/fold_4_chb01_16/ordinary_lgbm_predictions.parquet`, `results/lightgbm_cv_v1/fold_4_chb01_16/ordinary_lgbm.joblib`, `results/lightgbm_cv_v1/fold_4_chb01_16/weighted_lgbm_evaluation.json`, `results/lightgbm_cv_v1/fold_4_chb01_16/four_model_predictions.parquet`, `results/lightgbm_cv_v1/fold_4_chb01_16/ordinary_lgbm_evaluation.json`, `results/lightgbm_cv_v1/fold_4_chb01_16/ordinary_lgbm.txt`, `results/lightgbm_cv_v1/fold_4_chb01_16/weighted_rf_evaluation.json`, `results/lightgbm_cv_v1/fold_4_chb01_16/weighted_lgbm.txt`, `results/lightgbm_cv_v1/fold_4_chb01_16/ordinary_rf_evaluation.json`, `results/lightgbm_cv_v1/fold_4_chb01_16/weighted_lgbm.joblib`, `results/lightgbm_cv_v1/fold_4_chb01_16/definition.json`, `results/lightgbm_cv_v1/fold_3_chb01_15/weighted_lgbm_predictions.parquet`, `results/lightgbm_cv_v1/fold_3_chb01_15/ordinary_lgbm_predictions.parquet`, `results/lightgbm_cv_v1/fold_3_chb01_15/ordinary_lgbm.joblib`, `results/lightgbm_cv_v1/fold_3_chb01_15/weighted_lgbm_evaluation.json`, `results/lightgbm_cv_v1/fold_3_chb01_15/four_model_predictions.parquet`, `results/lightgbm_cv_v1/fold_3_chb01_15/ordinary_lgbm_evaluation.json`, `results/lightgbm_cv_v1/fold_3_chb01_15/ordinary_lgbm.txt`, `results/lightgbm_cv_v1/fold_3_chb01_15/weighted_rf_evaluation.json`, `results/lightgbm_cv_v1/fold_3_chb01_15/weighted_lgbm.txt`, `results/lightgbm_cv_v1/fold_3_chb01_15/ordinary_rf_evaluation.json`, `results/lightgbm_cv_v1/fold_3_chb01_15/weighted_lgbm.joblib`, `results/lightgbm_cv_v1/fold_3_chb01_15/definition.json`, `results/lightgbm_cv_v1/fold_1_chb01_03/weighted_lgbm_predictions.parquet`, `results/lightgbm_cv_v1/fold_1_chb01_03/ordinary_lgbm_predictions.parquet`, `results/lightgbm_cv_v1/fold_1_chb01_03/ordinary_lgbm.joblib`, `results/lightgbm_cv_v1/fold_1_chb01_03/weighted_lgbm_evaluation.json`, `results/lightgbm_cv_v1/fold_1_chb01_03/four_model_predictions.parquet`, `results/lightgbm_cv_v1/fold_1_chb01_03/ordinary_lgbm_evaluation.json`, `results/lightgbm_cv_v1/fold_1_chb01_03/ordinary_lgbm.txt`, `results/lightgbm_cv_v1/fold_1_chb01_03/weighted_rf_evaluation.json`, `results/lightgbm_cv_v1/fold_1_chb01_03/weighted_lgbm.txt`, `results/lightgbm_cv_v1/fold_1_chb01_03/ordinary_rf_evaluation.json`, `results/lightgbm_cv_v1/fold_1_chb01_03/weighted_lgbm.joblib`, `results/lightgbm_cv_v1/fold_1_chb01_03/definition.json`, `config/chb01_recordings.csv`, `requirements.txt`.

- [results/temporal_filtering_v1/verification.json](../results/temporal_filtering_v1/verification.json) — fields: `model_fits`, `baseline_reports_reproduced`, `labels_metadata_scores_preserved`, `final_test_or_supplementary_rows_used`, `primary_recordings`, `secondary_recordings`, `independent_filter_formula_checks`, `tests_passed`, `tests_skipped`.

- [results/baseline_v1/verification.json](../results/baseline_v1/verification.json) — fields: `models`, `tests_passed`, `tests_skipped`, `protected_files`, `reserved_test_predictions`.

- [results/raw_eeg_investigation_v1/preservation.json](../results/raw_eeg_investigation_v1/preservation.json) — fields: `src/train_eegnet.py`, `src/preprocess.py`, `src/baseline_rf.py`, `src/__init__.py`, `src/eegnet_model.py`, `src/split_dataset.py`, `src/evaluate.py`, `src/failure_analysis.py`, `src/rf_cross_validation.py`, `src/weighted_rf.py`, `tests/test_baseline_rf.py`, `tests/test_failure_analysis.py`, `tests/test_evaluate.py`, `tests/test_rf_cross_validation.py`, `tests/test_weighted_rf.py`, `tests/test_split_dataset.py`, `tests/test_preprocess.py`, `data/processed/chb01_detection_features.parquet`, `data/processed/.gitkeep`, `data/processed/chb01_features.parquet`, `data/processed/chb01_detection_features_11recordings_split.json`, `data/processed/chb01_detection_features_11recordings.json`, `data/processed/chb01_phase2a_preservation.json`, `data/processed/chb01_detection_features_11recordings.parquet`, `data/processed/chb01_detection_features.json`, `data/processed/chb01_detection_features_11recordings_split.parquet`, `results/rf_cv_v1/summary.json`, `results/rf_cv_v1/preservation.json`, `results/rf_cv_v1/folds.json`, `results/rf_cv_v1/artifact_sha256.json`, `results/rf_cv_v1/configuration.json`, `results/rf_cv_v1/tests.log`, `results/rf_cv_v1/verification.json`, `results/rf_cv_v1/fold_2_chb01_04/weighted_predictions.parquet`, `results/rf_cv_v1/fold_2_chb01_04/ordinary_evaluation.json`, `results/rf_cv_v1/fold_2_chb01_04/ordinary.joblib`, `results/rf_cv_v1/fold_2_chb01_04/weighted_evaluation.json`, `results/rf_cv_v1/fold_2_chb01_04/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_2_chb01_04/weighted.joblib`, `results/rf_cv_v1/fold_2_chb01_04/definition.json`, `results/rf_cv_v1/fold_4_chb01_16/weighted_predictions.parquet`, `results/rf_cv_v1/fold_4_chb01_16/ordinary_evaluation.json`, `results/rf_cv_v1/fold_4_chb01_16/ordinary.joblib`, `results/rf_cv_v1/fold_4_chb01_16/weighted_evaluation.json`, `results/rf_cv_v1/fold_4_chb01_16/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_4_chb01_16/weighted.joblib`, `results/rf_cv_v1/fold_4_chb01_16/definition.json`, `results/rf_cv_v1/fold_3_chb01_15/weighted_predictions.parquet`, `results/rf_cv_v1/fold_3_chb01_15/ordinary_evaluation.json`, `results/rf_cv_v1/fold_3_chb01_15/ordinary.joblib`, `results/rf_cv_v1/fold_3_chb01_15/weighted_evaluation.json`, `results/rf_cv_v1/fold_3_chb01_15/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_3_chb01_15/weighted.joblib`, `results/rf_cv_v1/fold_3_chb01_15/definition.json`, `results/rf_cv_v1/fold_1_chb01_03/weighted_predictions.parquet`, `results/rf_cv_v1/fold_1_chb01_03/ordinary_evaluation.json`, `results/rf_cv_v1/fold_1_chb01_03/ordinary.joblib`, `results/rf_cv_v1/fold_1_chb01_03/weighted_evaluation.json`, `results/rf_cv_v1/fold_1_chb01_03/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_1_chb01_03/weighted.joblib`, `results/rf_cv_v1/fold_1_chb01_03/definition.json`, `results/failure_analysis_v1/scores_comparison.png`, `results/failure_analysis_v1/family_distributions.parquet`, `results/failure_analysis_v1/quality.json`, `results/failure_analysis_v1/analysis_tests.log`, `results/failure_analysis_v1/preservation.json`, `results/failure_analysis_v1/feature_timelines.png`, `results/failure_analysis_v1/score_periods.json`, `results/failure_analysis_v1/weighted_local_scores.parquet`, `results/failure_analysis_v1/ordinary_local_scores.parquet`, `results/failure_analysis_v1/scores_missed.png`, `results/failure_analysis_v1/evaluation_tests.log`, `results/failure_analysis_v1/artifact_sha256.json`, `results/failure_analysis_v1/configuration.json`, `results/failure_analysis_v1/feature_differences.parquet`, `results/failure_analysis_v1/feature_families.png`, `results/failure_analysis_v1/verification.json`, `results/failure_analysis_v1/alarms.json`, `results/failure_analysis_v1/local_feature_timeline.parquet`, `results/failure_analysis_v1/all_feature_distributions.parquet`, `config/chb01_recordings.csv`, `data/raw/chb01_03.edf`, `data/raw/chb01_04.edf`, `data/raw/chb01_15.edf`, `data/raw/chb01_16.edf`.

- [results/raw_eeg_investigation_v1/verification.json](../results/raw_eeg_investigation_v1/verification.json) — fields: `chb01_15`, `chb01_03`, `chb01_04`, `chb01_16`.

- [results/rf_weighted_v1/preservation.json](../results/rf_weighted_v1/preservation.json) — fields: `results/baseline_v1/dummy_evaluation.json`, `results/baseline_v1/dummy.joblib`, `results/baseline_v1/random_forest.joblib`, `results/baseline_v1/artifact_sha256.json`, `results/baseline_v1/configuration.json`, `results/baseline_v1/random_forest_validation.parquet`, `results/baseline_v1/tests.log`, `results/baseline_v1/verification.json`, `results/baseline_v1/chb01_18_scores.png`, `results/baseline_v1/random_forest_evaluation.json`, `results/baseline_v1/dummy_validation.parquet`, `results/baseline_v1/random_forest_review_windows.parquet`, `data/processed/chb01_detection_features.parquet`, `data/processed/.gitkeep`, `data/processed/chb01_features.parquet`, `data/processed/chb01_detection_features_11recordings_split.json`, `data/processed/chb01_detection_features_11recordings.json`, `data/processed/chb01_phase2a_preservation.json`, `data/processed/chb01_detection_features_11recordings.parquet`, `data/processed/chb01_detection_features.json`, `data/processed/chb01_detection_features_11recordings_split.parquet`, `src/train_eegnet.py`, `src/preprocess.py`, `src/baseline_rf.py`, `src/__init__.py`, `src/eegnet_model.py`, `src/split_dataset.py`, `src/evaluate.py`, `config/chb01_recordings.csv`.

- [results/rf_weighted_v1/verification.json](../results/rf_weighted_v1/verification.json) — fields: `baseline_predictions_and_evaluation_unchanged`, `weighted_reload_identical`, `training_inputs_unchanged`, `test_rows_used`, `changed_windows`, `corrected_windows`, `regressed_windows`, `previous_misses_corrected`.

- [results/rf_cv_v1/preservation.json](../results/rf_cv_v1/preservation.json) — fields: `src/train_eegnet.py`, `src/preprocess.py`, `src/baseline_rf.py`, `src/__init__.py`, `src/eegnet_model.py`, `src/split_dataset.py`, `src/evaluate.py`, `src/weighted_rf.py`, `tests/test_baseline_rf.py`, `tests/test_evaluate.py`, `tests/test_weighted_rf.py`, `tests/test_split_dataset.py`, `tests/test_preprocess.py`, `data/processed/chb01_detection_features.parquet`, `data/processed/.gitkeep`, `data/processed/chb01_features.parquet`, `data/processed/chb01_detection_features_11recordings_split.json`, `data/processed/chb01_detection_features_11recordings.json`, `data/processed/chb01_phase2a_preservation.json`, `data/processed/chb01_detection_features_11recordings.parquet`, `data/processed/chb01_detection_features.json`, `data/processed/chb01_detection_features_11recordings_split.parquet`, `results/baseline_v1/dummy_evaluation.json`, `results/baseline_v1/dummy.joblib`, `results/baseline_v1/random_forest.joblib`, `results/baseline_v1/artifact_sha256.json`, `results/baseline_v1/configuration.json`, `results/baseline_v1/random_forest_validation.parquet`, `results/baseline_v1/tests.log`, `results/baseline_v1/verification.json`, `results/baseline_v1/chb01_18_scores.png`, `results/baseline_v1/random_forest_evaluation.json`, `results/baseline_v1/dummy_validation.parquet`, `results/baseline_v1/random_forest_review_windows.parquet`, `results/rf_weighted_v1/ordinary_evaluation.json`, `results/rf_weighted_v1/weighted_validation.parquet`, `results/rf_weighted_v1/validation_comparison.parquet`, `results/rf_weighted_v1/preservation.json`, `results/rf_weighted_v1/weighted_evaluation.json`, `results/rf_weighted_v1/review_windows.parquet`, `results/rf_weighted_v1/chb01_18_comparison.png`, `results/rf_weighted_v1/artifact_sha256.json`, `results/rf_weighted_v1/configuration.json`, `results/rf_weighted_v1/tests.log`, `results/rf_weighted_v1/verification.json`, `results/rf_weighted_v1/weighted_rf.joblib`, `config/chb01_recordings.csv`.

- [results/rf_cv_v1/folds.json](../results/rf_cv_v1/folds.json) — fields: `list records: 4`.

- [results/rf_cv_v1/verification.json](../results/rf_cv_v1/verification.json) — fields: `independent_models_fitted`, `reloads_identical`, `unique_evaluated_recordings_per_model`, `validation_test_or_new_recordings_used`, `training_inputs_unchanged`.

- [results/lightgbm_v1/preservation.json](../results/lightgbm_v1/preservation.json) — fields: `src/raw_eeg_investigation.py`, `src/train_eegnet.py`, `src/preprocess.py`, `src/baseline_rf.py`, `src/__init__.py`, `src/eegnet_model.py`, `src/split_dataset.py`, `src/evaluate.py`, `src/failure_analysis.py`, `src/rf_cross_validation.py`, `src/weighted_rf.py`, `tests/test_baseline_rf.py`, `tests/test_failure_analysis.py`, `tests/test_evaluate.py`, `tests/test_rf_cross_validation.py`, `tests/test_weighted_rf.py`, `tests/test_split_dataset.py`, `tests/test_raw_eeg_investigation.py`, `tests/test_preprocess.py`, `data/processed/chb01_detection_features.parquet`, `data/processed/.gitkeep`, `data/processed/chb01_features.parquet`, `data/processed/chb01_detection_features_11recordings_split.json`, `data/processed/chb01_detection_features_11recordings.json`, `data/processed/chb01_phase2a_preservation.json`, `data/processed/chb01_detection_features_11recordings.parquet`, `data/processed/chb01_detection_features.json`, `data/processed/chb01_detection_features_11recordings_split.parquet`, `results/failure_analysis_v1/scores_comparison.png`, `results/failure_analysis_v1/family_distributions.parquet`, `results/failure_analysis_v1/quality.json`, `results/failure_analysis_v1/analysis_tests.log`, `results/failure_analysis_v1/preservation.json`, `results/failure_analysis_v1/feature_timelines.png`, `results/failure_analysis_v1/score_periods.json`, `results/failure_analysis_v1/weighted_local_scores.parquet`, `results/failure_analysis_v1/ordinary_local_scores.parquet`, `results/failure_analysis_v1/scores_missed.png`, `results/failure_analysis_v1/evaluation_tests.log`, `results/failure_analysis_v1/artifact_sha256.json`, `results/failure_analysis_v1/configuration.json`, `results/failure_analysis_v1/feature_differences.parquet`, `results/failure_analysis_v1/feature_families.png`, `results/failure_analysis_v1/verification.json`, `results/failure_analysis_v1/alarms.json`, `results/failure_analysis_v1/local_feature_timeline.parquet`, `results/failure_analysis_v1/all_feature_distributions.parquet`, `results/baseline_v1/dummy_evaluation.json`, `results/baseline_v1/dummy.joblib`, `results/baseline_v1/random_forest.joblib`, `results/baseline_v1/artifact_sha256.json`, `results/baseline_v1/configuration.json`, `results/baseline_v1/random_forest_validation.parquet`, `results/baseline_v1/tests.log`, `results/baseline_v1/verification.json`, `results/baseline_v1/chb01_18_scores.png`, `results/baseline_v1/random_forest_evaluation.json`, `results/baseline_v1/dummy_validation.parquet`, `results/baseline_v1/random_forest_review_windows.parquet`, `results/raw_eeg_investigation_v1/waveforms_overview.png`, `results/raw_eeg_investigation_v1/window_measurements.parquet`, `results/raw_eeg_investigation_v1/waveforms_alarm_zoom.png`, `results/raw_eeg_investigation_v1/analysis_tests.log`, `results/raw_eeg_investigation_v1/preservation.json`, `results/raw_eeg_investigation_v1/evaluation_tests.log`, `results/raw_eeg_investigation_v1/artifact_sha256.json`, `results/raw_eeg_investigation_v1/configuration.json`, `results/raw_eeg_investigation_v1/verification.json`, `results/raw_eeg_investigation_v1/spectra_comparison.png`, `results/raw_eeg_investigation_v1/detected_seizure_context.png`, `results/rf_weighted_v1/ordinary_evaluation.json`, `results/rf_weighted_v1/weighted_validation.parquet`, `results/rf_weighted_v1/validation_comparison.parquet`, `results/rf_weighted_v1/preservation.json`, `results/rf_weighted_v1/weighted_evaluation.json`, `results/rf_weighted_v1/review_windows.parquet`, `results/rf_weighted_v1/chb01_18_comparison.png`, `results/rf_weighted_v1/artifact_sha256.json`, `results/rf_weighted_v1/configuration.json`, `results/rf_weighted_v1/tests.log`, `results/rf_weighted_v1/verification.json`, `results/rf_weighted_v1/weighted_rf.joblib`, `results/rf_cv_v1/summary.json`, `results/rf_cv_v1/preservation.json`, `results/rf_cv_v1/folds.json`, `results/rf_cv_v1/artifact_sha256.json`, `results/rf_cv_v1/configuration.json`, `results/rf_cv_v1/tests.log`, `results/rf_cv_v1/verification.json`, `results/rf_cv_v1/fold_2_chb01_04/weighted_predictions.parquet`, `results/rf_cv_v1/fold_2_chb01_04/ordinary_evaluation.json`, `results/rf_cv_v1/fold_2_chb01_04/ordinary.joblib`, `results/rf_cv_v1/fold_2_chb01_04/weighted_evaluation.json`, `results/rf_cv_v1/fold_2_chb01_04/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_2_chb01_04/weighted.joblib`, `results/rf_cv_v1/fold_2_chb01_04/definition.json`, `results/rf_cv_v1/fold_4_chb01_16/weighted_predictions.parquet`, `results/rf_cv_v1/fold_4_chb01_16/ordinary_evaluation.json`, `results/rf_cv_v1/fold_4_chb01_16/ordinary.joblib`, `results/rf_cv_v1/fold_4_chb01_16/weighted_evaluation.json`, `results/rf_cv_v1/fold_4_chb01_16/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_4_chb01_16/weighted.joblib`, `results/rf_cv_v1/fold_4_chb01_16/definition.json`, `results/rf_cv_v1/fold_3_chb01_15/weighted_predictions.parquet`, `results/rf_cv_v1/fold_3_chb01_15/ordinary_evaluation.json`, `results/rf_cv_v1/fold_3_chb01_15/ordinary.joblib`, `results/rf_cv_v1/fold_3_chb01_15/weighted_evaluation.json`, `results/rf_cv_v1/fold_3_chb01_15/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_3_chb01_15/weighted.joblib`, `results/rf_cv_v1/fold_3_chb01_15/definition.json`, `results/rf_cv_v1/fold_1_chb01_03/weighted_predictions.parquet`, `results/rf_cv_v1/fold_1_chb01_03/ordinary_evaluation.json`, `results/rf_cv_v1/fold_1_chb01_03/ordinary.joblib`, `results/rf_cv_v1/fold_1_chb01_03/weighted_evaluation.json`, `results/rf_cv_v1/fold_1_chb01_03/ordinary_predictions.parquet`, `results/rf_cv_v1/fold_1_chb01_03/weighted.joblib`, `results/rf_cv_v1/fold_1_chb01_03/definition.json`, `config/chb01_recordings.csv`.

- [results/lightgbm_v1/verification.json](../results/lightgbm_v1/verification.json) — fields: `model_fits`, `positive_class_weight`, `rf_predictions_and_reports_unchanged`, `lightgbm_reloads_identical`, `test_or_supplementary_predictions`, `training_inputs_unchanged`, `prediction_changes`, `fitted_models`.

- [results/lightgbm_v1/environment.json](../results/lightgbm_v1/environment.json) — fields: `python_executable`, `python`, `architecture`, `lightgbm`, `wheel_metadata`, `native_library`, `libomp`, `installation_commands`.

- [results/lightgbm_v1/warnings.json](../results/lightgbm_v1/warnings.json) — fields: `list records: 0`.

- [results/protocol_freeze_v1/verification.json](../results/protocol_freeze_v1/verification.json) — fields: `protocol_sha256`, `selected_models`, `git`, `verification`.

- [results/protocol_freeze_v1/preservation_verification.json](../results/protocol_freeze_v1/preservation_verification.json) — fields: `prior_artifacts_checked`, `changed_prior_artifacts`, `documented_python_blocks_syntax_checked_only`, `final_test_command_executed`, `supplementary_command_executed`, `protocol_sha256`, `reserved_raw_edf_files_opened_in_this_phase`, `training_edf_headers_read_by_existing_tests`.

- [results/rf_cv_v1/fold_2_chb01_04/definition.json](../results/rf_cv_v1/fold_2_chb01_04/definition.json) — fields: `fold`, `held_out`, `train_recordings`, `training_windows`, `training_ictal`, `evaluated_windows`, `evaluated_ictal`, `training_inputs_sha256`, `balanced_weights`.

- [results/rf_cv_v1/fold_4_chb01_16/definition.json](../results/rf_cv_v1/fold_4_chb01_16/definition.json) — fields: `fold`, `held_out`, `train_recordings`, `training_windows`, `training_ictal`, `evaluated_windows`, `evaluated_ictal`, `training_inputs_sha256`, `balanced_weights`.

- [results/rf_cv_v1/fold_3_chb01_15/definition.json](../results/rf_cv_v1/fold_3_chb01_15/definition.json) — fields: `fold`, `held_out`, `train_recordings`, `training_windows`, `training_ictal`, `evaluated_windows`, `evaluated_ictal`, `training_inputs_sha256`, `balanced_weights`.

- [results/rf_cv_v1/fold_1_chb01_03/definition.json](../results/rf_cv_v1/fold_1_chb01_03/definition.json) — fields: `fold`, `held_out`, `train_recordings`, `training_windows`, `training_ictal`, `evaluated_windows`, `evaluated_ictal`, `training_inputs_sha256`, `balanced_weights`.

- [results/lightgbm_cv_v1/fold_2_chb01_04/definition.json](../results/lightgbm_cv_v1/fold_2_chb01_04/definition.json) — fields: `fold`, `held_out`, `train_recordings`, `training_windows`, `training_ictal`, `evaluated_windows`, `evaluated_ictal`, `training_inputs_sha256`, `training_class_0`, `scale_pos_weight`, `models`.

- [results/lightgbm_cv_v1/fold_4_chb01_16/definition.json](../results/lightgbm_cv_v1/fold_4_chb01_16/definition.json) — fields: `fold`, `held_out`, `train_recordings`, `training_windows`, `training_ictal`, `evaluated_windows`, `evaluated_ictal`, `training_inputs_sha256`, `training_class_0`, `scale_pos_weight`, `models`.

- [results/lightgbm_cv_v1/fold_3_chb01_15/definition.json](../results/lightgbm_cv_v1/fold_3_chb01_15/definition.json) — fields: `fold`, `held_out`, `train_recordings`, `training_windows`, `training_ictal`, `evaluated_windows`, `evaluated_ictal`, `training_inputs_sha256`, `training_class_0`, `scale_pos_weight`, `models`.

- [results/lightgbm_cv_v1/fold_1_chb01_03/definition.json](../results/lightgbm_cv_v1/fold_1_chb01_03/definition.json) — fields: `fold`, `held_out`, `train_recordings`, `training_windows`, `training_ictal`, `evaluated_windows`, `evaluated_ictal`, `training_inputs_sha256`, `training_class_0`, `scale_pos_weight`, `models`.

- [data/processed/chb01_detection_features_11recordings_split.json](../data/processed/chb01_detection_features_11recordings_split.json) — fields: `manifest`, `manifest_sha256`, `input_sha256`, `preprocessing_provenance_sha256`, `raw_recordings`, `per_recording`, `per_split`, `warnings`, `preprocessing_warnings`, `feature_columns`, `test_policy`, `checks`, `output_sha256`.

- [data/processed/chb01_detection_features_11recordings.json](../data/processed/chb01_detection_features_11recordings.json) — fields: `processing_version`, `label_rule`, `window_sec`, `channels`, `bands_hz`, `filter`, `filter_warmup_flag_sec`, `power_floor_uv2`, `postictal_flag_sec`, `postictal_policy`, `annotation_sha256`, `recordings`, `preprocessing_source_sha256`, `python_version`, `versions`, `warnings`, `counts`.

- [data/processed/chb01_phase2a_preservation.json](../data/processed/chb01_phase2a_preservation.json) — fields: `all_unchanged`, `files`.

- [data/processed/chb01_detection_features.json](../data/processed/chb01_detection_features.json) — fields: `processing_version`, `label_rule`, `window_sec`, `channels`, `bands_hz`, `filter`, `filter_warmup_flag_sec`, `power_floor_uv2`, `postictal_flag_sec`, `postictal_policy`, `annotation_sha256`, `recordings`, `preprocessing_source_sha256`, `python_version`, `versions`, `warnings`, `counts`.

Environment source: [requirements.txt](../requirements.txt) and experiment `versions` fields; installed packages do not demonstrate use. EDFs under `data/raw/` are source inputs, not regenerated evidence. Models/results/data are intentionally ignored by Git; preserve this workspace or an artifact archive together with the Markdown. Source code alone cannot reproduce the saved history without those inputs.

## Appendix G. Evidence-package integrity and remaining verification

Read-only checksum review: 251 references checked across existing artifact/delivery checksum manifests. This is an integrity comparison, not a rerun of research tests or experiments.

All checked references matched their recorded hashes. This does not prove completeness of external history or validate clinical annotations.

Every pre-existing inspected project file was hashed before and after compilation of this document; no differences were found. Raw EDF contents were not read for this evidence package. No repository source, configuration or result was edited; no commit/push was made. Numeric report quotations retain historical rounding; use linked machine outputs for full precision. If the later narrative assigns personal motivations, exact daily timing or expert conclusions, obtain those from the original conversations and explicitly distinguish them from repository evidence.
