"""Phase 2H: frozen LightGBM configurations on the four training-only RF folds."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import warnings

import joblib
import numpy as np
import pandas as pd

from .baseline_rf import (DEFAULT_DATA, DEFAULT_MANIFEST, PREDICTION_METADATA,
                          fit_baselines, score_summary, write_json)
from .evaluate import evaluate_predictions
from .lightgbm_experiment import make_models, comparison_table, prediction_changes
from .preprocess import FEATURE_COLUMNS, PROJECT_ROOT, sha256
from .rf_cross_validation import HOLDOUT_IDS, load_training_data, fold_data, predict_holdout, aggregate_predictions
from .weighted_rf import training_fingerprint


def ictal_continuity(table):
    """Consecutive true-positive windows, distinct from alarm/event definitions.

    Input is one complete, chronologically ordered recording already validated
    by the evaluator. Runs use the existing >=50% ictal labels; no smoothing.
    """
    if table.empty or table.recording_id.nunique() != 1:
        raise ValueError('Continuity requires one evaluated recording')
    runs, current = [], None
    for row in table.itertuples():
        if row.label == 1 and row.prediction == 1:
            if current is None:
                current = {'start_sec': float(row.start_sec), 'end_sec': float(row.end_sec), 'windows': 1}
            else:
                current['end_sec'], current['windows'] = float(row.end_sec), current['windows'] + 1
        elif current is not None:
            runs.append(current)
            current = None
    if current is not None: runs.append(current)
    return {'ictal_windows': int(table.label.eq(1).sum()),
            'true_positive_windows': sum(r['windows'] for r in runs),
            'true_positive_runs': len(runs),
            'longest_run_windows': max((r['windows'] for r in runs), default=0),
            'longest_run_sec': max((r['end_sec']-r['start_sec'] for r in runs), default=0.),
            'runs': runs}


def load_rf_fold(folder, name, held, metadata):
    table = pd.read_parquet(Path(folder) / f'{name}_predictions.parquet')
    pd.testing.assert_frame_equal(table[PREDICTION_METADATA], held[PREDICTION_METADATA].reset_index(drop=True))
    report = evaluate_predictions(table, metadata)
    saved = json.loads((Path(folder) / f'{name}_evaluation.json').read_text())
    for key, value in report.items():
        if value != saved[key]: raise ValueError(f'Saved RF metrics changed: {name}/{key}')
    report['score_distribution'] = score_summary(table)
    return table, report


def plot_scores(tables, metadata, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), layout='constrained')
    for ax, rid in zip(axes, HOLDOUT_IDS):
        on, off = metadata[rid]['seizures_sec'][0]
        for (name, table), color in zip(tables.items(), ('#1768ac','#799f0c','#b23a48','#8c55b2')):
            group = table[table.recording_id.eq(rid)]
            ax.plot(group.end_sec, group.score, marker='.', markersize=3, linewidth=1, label=name, color=color)
        ax.axvspan(on, off, color='#e9b44c', alpha=.2, label='Annotated seizure')
        ax.axhline(.5, color='gray', linestyle='--', label='Default boundary; ties select 0')
        ax.set(title=f'{rid}: [{on:g},{off:g})', xlim=(on-30, off+40), ylim=(-.02,1.02), ylabel='Model score')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8, loc='upper right')
    axes[-1].set_xlabel('Actual window-end decision time (seconds)')
    fig.savefig(Path(output) / 'four_model_seizure_scores.png', dpi=160)
    plt.close(fig)


def run_experiment(data, manifest, rf_dir, lgbm_config, output):
    rf_dir, output = Path(rf_dir), Path(output)
    if output.exists(): raise FileExistsError(f'Use a new output directory; preserving {output}')
    frozen = json.loads(Path(lgbm_config).read_text())  # Settings only, no validation predictions/reports.
    rf_config = json.loads((rf_dir / 'configuration.json').read_text())
    for config in (frozen, rf_config):
        if sha256(data) != config['data_sha256'] or sha256(manifest) != config['manifest_sha256']:
            raise ValueError('Dataset/manifest differ from original experiments')
        if config['feature_columns'] != FEATURE_COLUMNS: raise ValueError('Feature ordering changed')
        for name, expected in config['source_sha256'].items():
            if sha256(PROJECT_ROOT / 'src' / name) != expected: raise ValueError(f'Methodology changed: {name}')
        for name, expected in config['versions'].items():
            if version(name) != expected: raise ValueError(f'Library version changed: {name}')
    for name, digest in json.loads((rf_dir / 'artifact_sha256.json').read_text()).items():
        if sha256(rf_dir / name) != digest: raise ValueError(f'RF artifact changed: {name}')
    frame, metadata = load_training_data(data, manifest)
    old_folds = json.loads((rf_dir / 'folds.json').read_text())
    if [f['held_out'] for f in old_folds] != list(HOLDOUT_IDS): raise ValueError('RF fold order changed')
    output.mkdir(parents=True, exist_ok=False)
    all_tables = {n: [] for n in ('ordinary_rf','weighted_rf','ordinary_lgbm','weighted_lgbm')}
    definitions, diagnostics, changes, fitted = [], {}, {}, {}
    config = {'created_utc': datetime.now(timezone.utc).isoformat(), 'data_sha256': sha256(data),
              'manifest_sha256': sha256(manifest), 'feature_columns': FEATURE_COLUMNS, 'metadata': metadata,
              'versions': frozen['versions'], 'python': platform.python_version(), 'platform': platform.platform(),
              'frozen_lightgbm_configuration_sha256': sha256(lgbm_config),
              'rf_cv_configuration_sha256': sha256(rf_dir / 'configuration.json'),
              'source_sha256': {n: sha256(PROJECT_ROOT/'src'/n) for n in
                   ('lightgbm_cross_validation.py','lightgbm_experiment.py','rf_cross_validation.py','evaluate.py','preprocess.py')},
              'ordinary_parameters': frozen['models']['ordinary_lgbm'],
              'weighted_parameter_rule': 'same parameters, scale_pos_weight=fold-training nonictal/ictal',
              'decision_rule': frozen['decision_rule'], 'policy': 'four frozen training-only folds; no tuning or early stopping'}
    write_json(output/'configuration.json', config)
    captured = []
    for number, rid in enumerate(HOLDOUT_IDS, 1):
        train, held = fold_data(frame, rid)
        signature = training_fingerprint(train)
        old = old_folds[number-1]
        if (list(train.recording_id.unique()) != old['train_recordings'] or signature != old['training_inputs_sha256']
                or len(train) != old['training_windows'] or int(train.label.sum()) != old['training_ictal']
                or len(held) != old['evaluated_windows'] or int(held.label.sum()) != old['evaluated_ictal']):
            raise ValueError('Fold inputs differ from RF experiment')
        models = make_models(train)  # Fresh estimators and training-derived weights in every fold.
        ratio = float(train.label.eq(0).sum()/train.label.eq(1).sum())
        expected_weighted = dict(frozen['models']['weighted_lgbm'], scale_pos_weight=ratio)
        if (models['ordinary_lgbm'].get_params() != frozen['models']['ordinary_lgbm'] or
                models['weighted_lgbm'].get_params() != expected_weighted):
            raise ValueError('LightGBM parameters differ from frozen Phase 2G settings')
        folder = output/f'fold_{number}_{rid}';folder.mkdir()
        definition = {**{key:value for key,value in old.items() if key != 'balanced_weights'},
                      'training_class_0': int(train.label.eq(0).sum()), 'scale_pos_weight': ratio,
                      'models': {name:m.get_params() for name,m in models.items()}}
        definitions.append(definition);write_json(folder/'definition.json', definition)
        tables, reports = {}, {}
        for name in ('ordinary','weighted'):
            tables[name+'_rf'],reports[name+'_rf'] = load_rf_fold(rf_dir/f'fold_{number}_{rid}',name,held,{rid:metadata[rid]})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fit_baselines(train, models)
            if training_fingerprint(train) != signature: raise ValueError('Training inputs mutated')
            for name, model in models.items():
                table = predict_holdout(model,held,rid).reset_index(drop=True)
                joblib.dump(model,folder/f'{name}.joblib',compress=3)
                restored = predict_holdout(joblib.load(folder/f'{name}.joblib'),held,rid).reset_index(drop=True)
                np.testing.assert_array_equal(table.prediction,restored.prediction)
                np.testing.assert_array_equal(table.score,restored.score)
                table.to_parquet(folder/f'{name}_predictions.parquet',index=False)
                model.booster_.save_model(str(folder/f'{name}.txt'))
                report = evaluate_predictions(table,{rid:metadata[rid]})
                report['score_distribution'] = score_summary(table)
                tables[name],reports[name] = table,report
                fitted[f'{rid}/{name}'] = {'trees':model.booster_.num_trees(),'iterations':model.n_estimators_}
        captured.extend(str(w.message) for w in caught)
        for name,table in tables.items():
            all_tables[name].append(table)
            write_json(folder/f'{name}_evaluation.json',reports[name])
        comparison_table(tables).to_parquet(folder/'four_model_predictions.parquet',index=False)
        diagnostics[rid] = {name:ictal_continuity(table) for name,table in tables.items()}
        changes[rid] = prediction_changes(tables['ordinary_lgbm'],tables['weighted_lgbm'])
    write_json(output/'folds.json',definitions)
    summaries = {name:aggregate_predictions(values,metadata) for name,values in all_tables.items()}
    old_summary = json.loads((rf_dir/'summary.json').read_text())
    for name in ('ordinary','weighted'):
        if summaries[name+'_rf'] != old_summary[name]: raise ValueError('RF aggregate changed')
    write_json(output/'summary.json',summaries)
    write_json(output/'continuity.json',diagnostics)
    write_json(output/'verification.json',{'lightgbm_fits':8,'rf_fits':0,'saved_lightgbm_reloads_identical':True,
        'rf_reports_and_fold_inputs_unchanged':True,'validation_test_supplementary_rows_used':False,
        'weighted_vs_ordinary_changes':changes,'fitted_lightgbm_models':fitted})
    write_json(output/'warnings.json',list(dict.fromkeys(captured)))
    plot_scores({name:pd.concat(values,ignore_index=True) for name,values in all_tables.items()},metadata,output)
    write_json(output/'artifact_sha256.json',{str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*')) if p.is_file()})
    return summaries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST)
    parser.add_argument('--rf-dir',type=Path,default=PROJECT_ROOT/'results/rf_cv_v1')
    parser.add_argument('--lightgbm-config',type=Path,default=PROJECT_ROOT/'results/lightgbm_v1/configuration.json')
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'results/lightgbm_cv_v1')
    args=parser.parse_args()
    summaries=run_experiment(args.data,args.manifest,args.rf_dir,args.lightgbm_config,args.output)
    print(json.dumps({name:{k:r[k] for k in ('window_metrics','event_metrics','macro_window_metrics')}
                      for name,r in summaries.items()},indent=2))


if __name__=='__main__':main()
