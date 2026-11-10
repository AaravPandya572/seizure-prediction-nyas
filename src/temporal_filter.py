"""Fixed causal binary alert rules. No file access, fitting or label changes."""

import numpy as np

from .evaluate import evaluate_predictions

RULES = ('none', 'two_consecutive', 'two_of_three')
RULE_DESCRIPTIONS = {
    'none': 'output current raw decision',
    'two_consecutive': 'output 1 iff current and immediately previous raw decisions are 1',
    'two_of_three': 'output 1 iff at least 2 of current and up to 2 previous raw decisions are 1',
}


def filter_binary(predictions, rule):
    """Process one recording in order, using raw history only.

    Startup uses available history, with no invented positives: first decision
    is zero for both filters; second can be positive if both observed values
    are positive. Output is available now, never shifted back to an earlier row.
    """
    if rule not in RULES:
        raise ValueError(f'Unknown rule {rule}; expected one of {RULES}')
    values = np.asarray(predictions)
    if values.ndim != 1 or not len(values) or not np.isin(values, [0, 1]).all():
        raise ValueError('Predictions must be a nonempty binary vector')
    values = values.astype(int)
    if rule == 'none': return values.copy()
    result = np.zeros(len(values), dtype=int)
    for i in range(1, len(values)):
        if rule == 'two_consecutive':
            result[i] = int(values[i] == 1 and values[i-1] == 1)
        else:
            result[i] = int(values[max(0, i-2):i+1].sum() >= 2)
    return result


def filter_predictions(frame, recording_metadata, rule):
    """Return an aligned copy, with independent state for every recording.

    The existing evaluator validates binary labels/predictions, exact decision
    times, annotation bounds and complete, chronological coverage per recording.
    Additionally require fixed five-second windows. Missing/incomplete windows
    are rejected; do not infer negative predictions or carry state across gaps.
    Only prediction is changed. Scores remain original model scores. The new
    unfiltered_prediction column preserves the decision before filtering.
    """
    if 'unfiltered_prediction' in frame:
        raise ValueError('Use original model predictions, not an already filtered table')
    evaluate_predictions(frame, recording_metadata)
    if not np.allclose(frame.end_sec - frame.start_sec, 5., rtol=0, atol=1e-9):
        raise ValueError('Temporal rules require complete five-second windows')
    result = frame.copy()
    result['unfiltered_prediction'] = frame.prediction.to_numpy().copy()
    prediction_column = result.columns.get_loc('prediction')
    # indices are row positions, so duplicate/nonconsecutive DataFrame indices
    # cannot change which window receives a prediction. No sorting or dropping.
    for positions in frame.groupby('recording_id', sort=False).indices.values():
        result.iloc[positions, prediction_column] = filter_binary(frame.iloc[positions].prediction, rule)
    return result
