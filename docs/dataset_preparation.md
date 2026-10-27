# Phase 2A: Patient 01 dataset and fixed recording splits

This step prepares data only. The test recordings are reserved for final
evaluation: preprocessing and annotation/integrity checks are permitted, but
model training, threshold selection, feature selection, and comparison must not
use them. The manifest is fixed before model development; class counts do not
change its assignments.

## Files and roles

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

## Exact reproduction commands

Run from the repository root with the existing ARM64 virtual environment:

```sh
venv/bin/python src/preprocess.py \
  --recordings chb01_01.edf chb01_03.edf chb01_04.edf chb01_14.edf \
    chb01_15.edf chb01_16.edf chb01_17.edf chb01_18.edf \
    chb01_21.edf chb01_22.edf chb01_26.edf \
  --output data/processed/chb01_detection_features_11recordings.parquet

venv/bin/python -m src.split_dataset \
  --manifest config/chb01_recordings.csv \
  --features data/processed/chb01_detection_features_11recordings.parquet \
  --output data/processed/chb01_detection_features_11recordings_split.parquet

venv/bin/python -m unittest discover -s tests -v
```

These output names are already present after the initial run. To preserve that
run when reproducing it, choose new `--output` filenames for both commands and
pass the new preprocessing filename to `--features`. The split CLI refuses to
overwrite either an existing Parquet or its JSON report. The preprocessing CLI
retains its existing behavior of replacing the explicitly named detection output;
never point it at a historical dataset you want to keep. All paths in the CSV
are resolved relative to the repository root, independent of the CSV's directory.
No new dependencies are required.

For rechecking this run's stored outputs without rewriting them:

```sh
venv/bin/python -m unittest discover -s tests -v
```

## Verified recording inventory

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

## Verified split totals

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

## Later model handoff

```python
import pandas as pd
from src.preprocess import FEATURE_COLUMNS

df = pd.read_parquet("data/processed/chb01_detection_features_11recordings_split.parquet")
train = df.loc[df["split"] == "train"].copy()
validation = df.loc[df["split"] == "validation"].copy()
# Keep test rows reserved until the experiment and thresholds are frozen.
X_train = train[FEATURE_COLUMNS]
y_train = train["label"]
```

`split`, annotations, quality flags, and recording/timestamp columns are metadata,
not model inputs. Use the existing evaluation handoff with the preprocessing
JSON's exact intervals and per-recording durations. No model development has
been performed in Phase 2A.

## Warnings and limitations

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
