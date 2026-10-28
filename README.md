# EEG seizure detection: Week 1 preprocessing

Offline research project using CHB-MIT clinical scalp EEG. The preprocessing
produces **ictal versus nonictal** features. Phase 2B adds offline baseline models.
The original preictal pipeline is preserved in Git commit `4454197`.

## Run on the existing Apple Silicon environment

From the repository root (no additional packages required):

```sh
venv/bin/python -m unittest discover -s tests -v
venv/bin/python src/preprocess.py
```

The default explicitly opens **only** `chb01_01.edf`, `chb01_03.edf`, and
`chb01_04.edf`. There is no EDF directory scan. Downloading files are not inspected.
The old `data/processed/chb01_features.parquet` is protected from overwriting.
Outputs are `data/processed/chb01_detection_features.parquet` and a matching
`.json` report. Re-running replaces these detection outputs.

After additional downloads have **finished**, select completed files explicitly:

```sh
venv/bin/python src/preprocess.py --recordings chb01_01.edf chb01_03.edf chb01_04.edf chb01_14.edf chb01_15.edf chb01_16.edf chb01_17.edf chb01_18.edf
```

`--raw-dir`, `--summary`, and `--output` support other local inputs. A summary
must contain every selected filename, including an explicit zero seizure count
for seizure-free files. For another patient, supply that patient's summary.
Do not select a file while it is downloading.

## Scientific definitions

- Each recording is tiled in nonoverlapping five-second windows using integer
  sample indices. Window and seizure intervals are half-open `[start, end)`.
  Any final incomplete window is dropped with a warning; none is expected for
  the three complete one-hour files.
- A window is positive when at least 50% overlaps the union of validated seizure
  intervals. Touching an onset/offset without overlap is not positive. All events
  in a recording are considered. Invalid, incomplete, overlapping, unsorted, or
  out-of-recording annotations fail clearly.
- All postictal and other nonictal EEG is retained. `postictal_60s` marks windows
  overlapping the first 60 seconds after any seizure offset for later analysis;
  this descriptive flag does not change labels or exclude data.
- A fixed ordered list of 18 existing bipolar signals is required. MNE's first
  duplicate `T8-P8-0` is explicitly mapped to canonical `T8-P8`, with a warning.
  Other missing or ambiguous required channels fail rather than shift features.
- A fourth-order Butterworth design (bandpass transformation) at 0.5–45 Hz is
  applied forward once using SciPy SOS filtering over each complete recording.
  State persists across windows and starts at zero for each EDF. There is no
  backward filtering or use of future samples. The 60 Hz notch is omitted because
  the analysis ends at 45 Hz.
- `filter_warmup` flags windows starting within the first ten seconds. This is a
  pragmatic quality-control flag, not proof that every transient has settled.
  These windows remain in the continuous dataset. Causal filtering changes phase
  and may affect eventual detection delay; that delay still needs evaluation.
- Per channel: variance (µV²), sum of absolute sample differences (µV; fixed
  sample rate/window length), and five relative Welch powers. Welch uses two-second
  Hann segments, one-second overlap, and constant detrending. Frequency-bin sums
  partition 0.5–45 Hz without gaps: `[0.5,4)`, `[4,8)`, `[8,13)`, `[13,30)`,
  `[30,45]`. The same bin integration defines the denominator.
- Channels with band power <=1e-12 µV² receive zero spectral ratios and are
  counted in `low_power_channel_count`, with a warning. Other channels' relative
  powers sum to one. This checks near-flat signals, not all possible EEG artifacts.

## Output and reproducibility

The Parquet has 126 canonical feature columns plus patient/recording/file IDs,
start/end samples and seconds, decision time (window end), sampling rate,
recording duration, ictal overlap, overlapping event IDs (one-based within file),
quality flags, label, and processing version. There are **no train/test assignments**;
use whole recording IDs for future splits. Import `FEATURE_COLUMNS` from
`src.preprocess` to select model inputs; annotation and timing metadata must not
be included as predictors.

The JSON preserves exact seizure intervals, original EDF channel selections,
raw EDF and summary SHA-256 hashes, preprocessing source hash, filter/feature
settings, dependency versions, counts, and captured warnings. Its raw filenames
are relative to the selected raw directory. Preserve the JSON with the Parquet.

The initial three recordings should yield 720 windows each: 2,160 total,
2,146 nonictal and 14 ictal (8 in `_03`, 6 in `_04`). This is a one-patient,
two-seizure development dataset. It cannot establish generalization.
Recording-level validation/test splits and model comparisons are later work.
Exact annotated event timing must be used for latency evaluation,
rather than rounded window labels. EDFs are never concatenated across gaps.

Annotation source: [PhysioNet CHB-MIT 1.0.0](https://physionet.org/content/chbmit/1.0.0/).
The local environment is a Python 3.14 ARM64 `venv`; installed package versions
are recorded in each output report and the existing requirements file.

## Evaluation framework

`src/evaluate.py` provides window metrics, positive-run alarm episodes, and
seizure-event evaluation from supplied predictions and exact annotations.
See [the evaluation contract and model handoff examples](docs/evaluation.md).
Run its synthetic tests without opening EDF recordings:

```sh
venv/bin/python -m unittest discover -s tests -p 'test_evaluate.py' -v
```

## Expanded Patient 01 dataset (Phase 2A)

The 11-recording dataset and fixed recording-level splits are described in
[dataset preparation and verification](docs/dataset_preparation.md), including
the exact CLI commands, per-recording counts, warnings, and limitations.
The reproducible assignments are in `config/chb01_recordings.csv`.
Use `chb01_detection_features_11recordings_split.parquet` for later development;
its `split` column reserves `chb01_21`, `chb01_22`, and `chb01_26` for final testing.
The older three-recording and prediction outputs are preserved.

## Initial baseline experiment (Phase 2B)

The majority-class DummyClassifier and unweighted 200-tree Random Forest train
only on the fixed training recordings and evaluate validation only. See the
[baseline experiment report](docs/baseline_experiment.md) for exact feature
selection, model settings, measured results, saved artifacts and reproduction
commands. The test recordings remain reserved for final evaluation.
