"""Phase 2G: two fixed LightGBM fits, validation-only comparison with saved RFs."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import warnings

import joblib
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.base import clone

from .baseline_rf import (DEFAULT_DATA, DEFAULT_MANIFEST, PREDICTION_METADATA,
                          fit_baselines, load_development_data, score_summary,
                          validation_predictions, write_json)
from .evaluate import evaluate_predictions
from .preprocess import FEATURE_COLUMNS, PROJECT_ROOT, sha256
from .weighted_rf import training_fingerprint


def make_models(train):
    """The only pairwise parameter difference is weight, estimated on train."""
    if train.empty or not train.split.eq('train').all() or set(train.label) != {0, 1}:
        raise ValueError('Weights require nonempty training rows and both classes')
    ratio = float(train.label.eq(0).sum() / train.label.eq(1).sum())
    ordinary = LGBMClassifier(objective='binary', n_estimators=150, learning_rate=.05,
                             max_depth=3, num_leaves=7, min_child_samples=10,
                             random_state=42, n_jobs=4, scale_pos_weight=1.0,
                             deterministic=True, force_col_wise=True, verbosity=-1)
    return {'ordinary_lgbm': ordinary, 'weighted_lgbm': clone(ordinary).set_params(scale_pos_weight=ratio)}


def load_rf_reference(folder, stem, model_file, validation, metadata):
    """Recompute metrics from saved validation tables and reproduce saved scores."""
    folder = Path(folder)
    for name, digest in json.loads((folder / 'artifact_sha256.json').read_text()).items():
        if sha256(folder / name) != digest:
            raise ValueError(f'Historical RF artifact changed: {name}')
    saved = pd.read_parquet(folder / f'{stem}_validation.parquet')
    pd.testing.assert_frame_equal(saved[PREDICTION_METADATA], validation[PREDICTION_METADATA].reset_index(drop=True))
    reproduced = validation_predictions(joblib.load(folder / model_file), validation).reset_index(drop=True)
    pd.testing.assert_frame_equal(saved, reproduced)
    report = evaluate_predictions(saved, metadata)
    old = json.loads((folder / f'{stem}_evaluation.json').read_text())
    for key, value in report.items():
        if value != old[key]: raise ValueError(f'RF evaluation changed: {key}')
    report['score_distribution'] = score_summary(saved)
    return saved, report


def comparison_table(tables):
    """Positional comparison only after exact identity/timing/label checks."""
    first = next(iter(tables.values()))
    result = first[PREDICTION_METADATA].reset_index(drop=True).copy()
    for name, table in tables.items():
        table = table.reset_index(drop=True)
        pd.testing.assert_frame_equal(result[PREDICTION_METADATA], table[PREDICTION_METADATA])
        result[name + '_prediction'] = table.prediction.to_numpy()
        result[name + '_score'] = table.score.to_numpy()
    return result


def prediction_changes(reference, candidate):
    pd.testing.assert_frame_equal(reference[PREDICTION_METADATA].reset_index(drop=True),
                                  candidate[PREDICTION_METADATA].reset_index(drop=True))
    truth = reference.label.to_numpy()
    a, b = reference.prediction.to_numpy(), candidate.prediction.to_numpy()
    return {'changed': int((a != b).sum()),
            'misses_rescued': int(((truth == 1) & (a == 0) & (b == 1)).sum()),
            'new_ictal_misses': int(((truth == 1) & (a == 1) & (b == 0)).sum()),
            'new_fp_windows': int(((truth == 0) & (a == 0) & (b == 1)).sum()),
            'fp_windows_removed': int(((truth == 0) & (a == 1) & (b == 0)).sum())}


def plot_comparison(tables, metadata, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for rid, record in metadata.items():
        if not record['seizures_sec']: continue
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout='constrained')
        colors = ('#1768ac', '#799f0c', '#b23a48', '#8c55b2')
        for ax in axes:
            for (name, table), color in zip(tables.items(), colors):
                group = table[table.recording_id.eq(rid)]
                ax.plot(group.end_sec, group.score, marker='.', markersize=3, linewidth=1,
                        label=name, color=color)
            for i, (on, off) in enumerate(record['seizures_sec']):
                ax.axvspan(on, off, color='#e9b44c', alpha=.2, label='Annotated seizure' if i == 0 else None)
            ax.axhline(.5, linestyle='--', color='gray', label='Default boundary; ties select 0')
            ax.set(ylabel='Model score', ylim=(-.02, 1.02))
            ax.grid(alpha=.2)
        on, off = record['seizures_sec'][0]
        axes[0].set(title=f'{rid}: four fixed models, validation only', xlim=(0, record['duration_sec']))
        axes[0].legend(fontsize=8, loc='upper right')
        axes[1].set(xlim=(on-60, off+60), xlabel='Actual window-end decision time (seconds)')
        fig.savefig(Path(output) / f'{rid}_four_model_scores.png', dpi=160)
        plt.close(fig)


def run_experiment(data, manifest, rf_dir, weighted_rf_dir, output):
    output, rf_dir, weighted_rf_dir = Path(output), Path(rf_dir), Path(weighted_rf_dir)
    if output.exists(): raise FileExistsError(f'Use a new output directory; preserving {output}')
    rf_config = json.loads((rf_dir / 'configuration.json').read_text())
    weighted_config = json.loads((weighted_rf_dir / 'configuration.json').read_text())
    for config in (rf_config, weighted_config):
        if sha256(data) != config['data_sha256'] or sha256(manifest) != config['manifest_sha256']:
            raise ValueError('Dataset or frozen assignments changed from RF experiments')
        if config['feature_columns'] != FEATURE_COLUMNS:
            raise ValueError('Canonical feature ordering changed')
    for name, expected in rf_config['versions'].items():
        if version(name) != expected: raise ValueError(f'Existing dependency version changed: {name}')
    for name, expected in rf_config['source_sha256'].items():
        if sha256(PROJECT_ROOT / 'src' / name) != expected: raise ValueError(f'Existing methodology changed: {name}')
    partitions, metadata = load_development_data(data, manifest)
    train, validation = partitions['train'], partitions['validation']
    metadata = {rid: metadata[rid] for rid in validation.recording_id.unique()}
    if metadata != rf_config['validation_metadata']:
        raise ValueError('Validation annotations changed')
    tables, reports = {}, {}
    for name, folder, stem, model_file in (
            ('ordinary_rf', rf_dir, 'random_forest', 'random_forest.joblib'),
            ('weighted_rf', weighted_rf_dir, 'weighted', 'weighted_rf.joblib')):
        tables[name], reports[name] = load_rf_reference(folder, stem, model_file, validation, metadata)
    models = make_models(train)
    different = [key for key in models['ordinary_lgbm'].get_params()
                 if models['ordinary_lgbm'].get_params()[key] != models['weighted_lgbm'].get_params()[key]]
    if different != ['scale_pos_weight']:
        raise ValueError('Only positive-class weight may differ')
    signature = training_fingerprint(train)
    output.mkdir(parents=True, exist_ok=False)
    config = {'created_utc': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
              'platform': platform.platform(), 'data': str(Path(data).resolve()), 'data_sha256': sha256(data),
              'manifest_sha256': sha256(manifest), 'feature_columns': FEATURE_COLUMNS,
              'training_inputs_sha256': signature, 'models': {name: model.get_params() for name, model in models.items()},
              'changed_parameters': different, 'validation_metadata': metadata,
              'counts': rf_config['counts'],
              'versions': {name: version(name) for name in ('lightgbm', *rf_config['versions'])},
              'reference_configuration_sha256': {'ordinary_rf': sha256(rf_dir / 'configuration.json'),
                                                 'weighted_rf': sha256(weighted_rf_dir / 'configuration.json')},
              'source_sha256': {name: sha256(PROJECT_ROOT / 'src' / name) for name in
                               ('lightgbm_experiment.py', 'baseline_rf.py', 'weighted_rf.py', 'evaluate.py', 'preprocess.py')},
              'fit_policy': 'train only; no eval_set, callbacks, early stopping, sample weighting or resampling',
              'decision_rule': 'sklearn-style predict; positive score >0.5, exact tie selects class 0'}
    write_json(output / 'configuration.json', config)  # Frozen before fitting.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        # Existing helper supplies only the training feature whitelist and labels.
        fit_baselines(train, models)
        if training_fingerprint(train) != signature: raise ValueError('Training inputs mutated')
        for name, model in models.items():
            prediction = validation_predictions(model, validation).reset_index(drop=True)
            joblib.dump(model, output / f'{name}.joblib', compress=3)
            restored = validation_predictions(joblib.load(output / f'{name}.joblib'), validation).reset_index(drop=True)
            pd.testing.assert_frame_equal(prediction, restored)
            model.booster_.save_model(str(output / f'{name}.txt'))
            prediction.to_parquet(output / f'{name}_validation.parquet', index=False)
            report = evaluate_predictions(prediction, metadata)
            report['score_distribution'] = score_summary(prediction)
            tables[name], reports[name] = prediction, report
    for name, report in reports.items(): write_json(output / f'{name}_evaluation.json', report)
    write_json(output / 'comparison.json', reports)
    comparison_table(tables).to_parquet(output / 'window_comparison.parquet', index=False)
    changes = {f'{a}_to_{b}': prediction_changes(tables[a], tables[b]) for a, b in
               (('ordinary_lgbm','weighted_lgbm'), ('ordinary_rf','ordinary_lgbm'), ('ordinary_rf','weighted_lgbm'))}
    write_json(output / 'verification.json', {'model_fits': 2, 'positive_class_weight': models['weighted_lgbm'].scale_pos_weight,
        'rf_predictions_and_reports_unchanged': True, 'lightgbm_reloads_identical': True,
        'test_or_supplementary_predictions': False, 'training_inputs_unchanged': True, 'prediction_changes': changes,
        'fitted_models': {name: {'actual_trees': model.booster_.num_trees(), 'actual_iterations': model.n_estimators_,
                               'booster_parameters': model.booster_.params} for name, model in models.items()}})
    write_json(output / 'warnings.json', list(dict.fromkeys(str(w.message) for w in caught)))
    plot_comparison(tables, metadata, output)
    write_json(output / 'artifact_sha256.json', {p.name: sha256(p) for p in sorted(output.iterdir())})
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--rf-dir', type=Path, default=PROJECT_ROOT / 'results/baseline_v1')
    parser.add_argument('--weighted-rf-dir', type=Path, default=PROJECT_ROOT / 'results/rf_weighted_v1')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'results/lightgbm_v1')
    args = parser.parse_args()
    reports = run_experiment(args.data, args.manifest, args.rf_dir, args.weighted_rf_dir, args.output)
    print(json.dumps({name: {k: report[k] for k in ('window_metrics','event_metrics')} for name, report in reports.items()}, indent=2))


if __name__ == '__main__': main()
