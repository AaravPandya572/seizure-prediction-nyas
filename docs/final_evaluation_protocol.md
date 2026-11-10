# Final experimental protocol — Phase 3B

Protocol `nyas-p01-final-evaluation-v1`, frozen **2026-11-10 (Asia/Kolkata)**.
Status: **awaiting review and separate authorization for final evaluation**.
No final-test predictions or supplementary evaluation were performed for this freeze.

## Scientific objective and fixed choices

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

## Dataset partitions

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

## Exact fitted artifacts and configurations

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

## Features, preprocessing and labels

Keep `detection-v1` unchanged. Five-second nonoverlapping windows contain 1,280
samples at 256 Hz. The 18 bipolar channels, in order, are:

```text
FP1-F7, F7-T7, T7-P7, P7-O1, FP1-F3, F3-C3, C3-P3, P3-O1,
FP2-F4, F4-C4, C4-P4, P4-O2, FP2-F8, F8-T8, T8-P8, P8-O2,
FZ-CZ, CZ-PZ
```

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

## Decisions, alarms and event matching

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

## Metrics and reporting

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

## Reproducibility and verification

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

```sh
venv/bin/python - <<'PY'
import unittest
suite = unittest.defaultTestLoader.discover('tests')
deferred = 'test_split_dataset.ExpandedArtifactTests.test_all_recordings_labels_features_and_splits'
def defer(test):
    if isinstance(test, unittest.TestSuite):
        for child in test:
            defer(child)
    elif test.id() == deferred:
        method = getattr(type(test), test._testMethodName)
        setattr(type(test), test._testMethodName,
                unittest.skip('Phase 3B: do not decode final-test rows')(method))
defer(suite)
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
PY
```

## Proposed primary final-evaluation command — NOT EXECUTED

Run from the repository root **only after separate explicit authorization**.
The complete command below never fits models. It checks frozen inputs and
environment, selects the three test recordings, then uses the unchanged
evaluator. It refuses an existing output directory. Its Python syntax was checked
in Phase 3B; its final-data path was deliberately not executed. A failed check
must stop for review, not trigger refitting or manual relaxation of the protocol.

```sh
shasum -a 256 -c config/final_evaluation_protocol.sha256
NYAS_FINAL_TEST_AUTHORIZED=YES venv/bin/python - <<'PY'
import os
if os.environ.get('NYAS_FINAL_TEST_AUTHORIZED') != 'YES':
    raise SystemExit('Final evaluation requires separate explicit authorization')
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import joblib
import numpy as np
import pandas as pd
from src.baseline_rf import PREDICTION_METADATA, select_features
from src.evaluate import evaluate_predictions
from src.preprocess import FEATURE_COLUMNS, window_annotation

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def require(condition, message):
    if not condition:
        raise SystemExit(message)

protocol_path = Path('config/final_evaluation_protocol.json')
protocol_hash = digest(protocol_path)
require(protocol_hash == 'cb5b77949755ba0aaf796e6199f388a83a195257e5f3b9459d136bbf2d4c7f3b',
        'Protocol changed: stop for review')
cfg = json.loads(protocol_path.read_text())
for section in ('source_sha256', 'test_source_sha256'):
    for path, expected in cfg['integrity'][section].items():
        require(digest(path) == expected, f'Changed source: {path}')
files = list(cfg['data'].values()) + [cfg['integrity']['requirements'],
                                     cfg['integrity']['gitignore']]
for model_cfg in cfg['models'].values():
    files += [model_cfg['artifact'], model_cfg['training_configuration']]
    if 'native_archive' in model_cfg:
        files.append(model_cfg['native_archive'])
files += [cfg['software']['libomp'], cfg['software']['lightgbm_native_library']]
for item in files:
    require(digest(item['path']) == item['sha256'], f"Changed artifact: {item['path']}")
require(platform.python_version() == cfg['software']['python'], 'Python changed')
require(platform.machine() == cfg['software']['architecture'], 'Architecture changed')
for package, expected in cfg['software']['packages'].items():
    require(version(package) == expected, f'Package changed: {package}')
require(FEATURE_COLUMNS == cfg['features']['columns'], 'Feature ordering changed')

plan = cfg['assessments']['primary']
output = Path(plan['output_directory'])
require(not output.exists(), 'Output already exists: do not overwrite or rerun')
audit = json.loads(Path(cfg['data']['split_audit']['path']).read_text())
require(audit['feature_columns'] == FEATURE_COLUMNS, 'Audit feature ordering changed')
manifest = pd.read_csv(cfg['data']['manifest']['path'])
ids = plan['recordings']
require(set(manifest.loc[manifest.split == 'test', 'filename']) ==
        {rid + '.edf' for rid in ids}, 'Test manifest differs')
frame = pd.read_parquet(cfg['data']['features']['path'], filters=[
    ('split', '==', plan['required_split']), ('recording_id', 'in', ids)])
require(set(frame.recording_id) == set(ids) and frame.split.eq('test').all(),
        'Missing or unexpected test recordings')
require(frame.patient_id.eq('chb01').all(), 'Unexpected patient')
require(frame.source_file.eq(frame.recording_id + '.edf').all(), 'Source ID mismatch')
require(not frame.duplicated(['recording_id', 'start_sec', 'end_sec']).any(),
        'Duplicate windows')
require(np.array_equal(frame.decision_time_sec, frame.end_sec), 'Decision time changed')
require(np.allclose(frame.end_sec - frame.start_sec, 5, rtol=0, atol=1e-9),
        'Unexpected window length')
features = select_features(frame)  # Fixed whitelist and finite-value check.
metadata = {}
for rid, group in frame.groupby('recording_id', sort=False):
    raw = audit['raw_recordings'][rid + '.edf']
    metadata[rid] = {key: raw[key] for key in ('duration_sec', 'seizures_sec')}
    require(group.recording_duration_sec.eq(raw['duration_sec']).all(), 'Duration changed')
    require(group.sampling_rate_hz.eq(cfg['features']['sampling_rate_hz']).all(),
            'Sampling frequency changed')
    expected = [window_annotation(a, b, raw['seizures_sec'])[0]
                for a, b in zip(group.start_sec, group.end_sec)]
    require(np.array_equal(group.label, expected), f'Annotation mismatch: {rid}')
# Check complete timelines and annotations before model inference.
checked = frame[PREDICTION_METADATA].copy()
checked['prediction'] = 0
evaluate_predictions(checked, metadata)

results = {}
for role, model_cfg in cfg['models'].items():
    model = joblib.load(model_cfg['artifact']['path'])
    require(model.get_params() == model_cfg['parameters'], f'Parameters changed: {role}')
    require(list(model.feature_names_in_) == FEATURE_COLUMNS, f'Features changed: {role}')
    require(np.array_equal(model.classes_, [0, 1]), f'Classes changed: {role}')
    predictions = frame[PREDICTION_METADATA].copy()
    predictions['prediction'] = model.predict(features)
    predictions['score'] = model.predict_proba(features)[:, 1]
    require(np.array_equal(predictions.prediction, (predictions.score > 0.5).astype(int)),
            f'Decision boundary changed: {role}')
    report = evaluate_predictions(predictions, metadata)
    for events in [report['event_metrics']] + [
            value['event_metrics'] for value in report['per_recording'].values()]:
        events['undefined_metrics'] = {}
        if events['n_seizures'] == 0:
            events['undefined_metrics']['event_sensitivity'] = 'no annotated seizures'
        if not events['latencies_sec']:
            for metric in ('mean_latency_sec', 'median_latency_sec'):
                events['undefined_metrics'][metric] = 'no detected seizures'
    report.update(protocol_id=cfg['protocol_id'], protocol_sha256=protocol_hash,
                  model_id=model_cfg['id'], assessment='primary_final_test')
    results[role] = (predictions, report)

output.mkdir(parents=True, exist_ok=False)
shutil.copy2(protocol_path, output / protocol_path.name)
shutil.copy2('config/final_evaluation_protocol.sha256', output / 'protocol.sha256')
for role, (predictions, report) in results.items():
    predictions.to_parquet(output / f'{role}_predictions.parquet', index=False)
    (output / f'{role}_evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
checksums = {path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()}
(output / 'artifact_sha256.json').write_text(json.dumps(checksums, indent=2) + '\n')
print(f'Wrote both fixed-model reports and aligned predictions to {output}')
PY
```

The reports contain pooled/per-recording metrics, exact alarms and every event's
detection/miss/latency. Convert the unchanged reports into comparison tables;
do not select a new model from the results. Do not rerun development training
commands to perform final testing.

## Proposed supplementary workflow — NOT EXECUTED

After separate authorization, verify that the explicit EDFs are complete,
readable and annotated with zero seizures in `chb01-summary.txt`. Confirm that
the proposed output does not exist, then use the existing preprocessing CLI:

```sh
venv/bin/python src/preprocess.py \
  --recordings chb01_23.edf chb01_24.edf chb01_25.edf \
  --output data/processed/chb01_supplementary_features_v1.parquet
```

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

## Limitations and unresolved issues

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
