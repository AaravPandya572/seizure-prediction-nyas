# Phase 3A: fixed causal temporal-filtering experiment

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

## Rules frozen before evaluation

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

## Inputs and separation of assessments

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

## Primary training-fold results: four hours

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

![Primary positive decision times](../results/temporal_filtering_v1/primary_cv/positive_decisions.png)

Ticks mark actual positive window-end decisions, not continuous seizure coverage.
The annotation is shaded. These are seizure-local views; all metrics still use
full one-hour timelines, including alarms outside the plotted ranges.

## Window classification versus alarm episodes

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

## Exact positive-prediction changes

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

## Secondary assessment: original validation, two hours

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

![Secondary positive decision times](../results/temporal_filtering_v1/secondary_validation/positive_decisions.png)

This relatively benign LightGBM validation outcome cannot erase the primary
38-second latency on `15`. No extra filter was invented to keep its early alert
while suppressing other isolated positives.

## Decision, limits and reproducibility

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

```sh
MPLCONFIGDIR=/tmp/nyas-matplotlib venv/bin/python -m src.temporal_filter_experiment \
  --output results/temporal_filtering_reproduction

MPLCONFIGDIR=/tmp/nyas-matplotlib venv/bin/python -m unittest discover -s tests \
  -p test_temporal_filter.py -v
venv/bin/python -m unittest discover -s tests -p test_evaluate.py -v
```

The default inputs are the saved original ordinary RF/CV and ordinary LightGBM/CV
artifacts plus their separate original validation outputs. CLI options permit
explicit source directories, not arbitrary new recordings. Existing output
folders are refused. No final-test or supplementary data, model training,
hyperparameter optimization, new thresholds, EEGNet, dashboard, dependency
installation, commit or push is involved. Work stops after this controlled report.
