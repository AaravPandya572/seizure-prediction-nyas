# Phase 2F: targeted raw EEG investigation

The delayed-alarm window contains a conspicuous fast burst in the original
EEG, strongest in FP2–F8, F8–T8 and T8–P8. It survives the established causal
filter. Absolute 30–45 Hz power increases sharply, confirming that the relative
gamma spike is not merely a denominator effect. Its physiological versus
artifactual origin remains unresolved. No software or label error was found
in the targeted checks, and the missed seizure remains missed.

## Scope and methods

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

## Observations in the original waveform

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

## Absolute versus relative gamma

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

## Context from detected seizures

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

## Technical consistency and model behavior

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

## Unresolved explanations and defensible conclusions

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

## Outputs and reproduction

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

```sh
MPLCONFIGDIR=/tmp/nyas-matplotlib venv/bin/python -m src.raw_eeg_investigation \
  --output results/raw_eeg_investigation_reproduction
venv/bin/python -m unittest discover -s tests -p test_raw_eeg_investigation.py -v
venv/bin/python -m unittest discover -s tests -p test_evaluate.py -v
```

The script refuses an existing output directory and opens only the four EDFs
explicitly named in this report. No model modifications, threshold optimization,
new classifier, temporal filtering or dashboard work follows this investigation.
