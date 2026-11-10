"""Phase 3A: saved-prediction filtering; primary CV then secondary validation."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline_rf import PREDICTION_METADATA, write_json
from .evaluate import evaluate_predictions
from .preprocess import PROJECT_ROOT, sha256
from .temporal_filter import RULES, RULE_DESCRIPTIONS, filter_predictions

HOLDOUTS = ('chb01_03', 'chb01_04', 'chb01_15', 'chb01_16')


class ArtifactReader:
    """Read only explicitly named artifacts and verify their stored checksums."""
    def __init__(self, folder, used):
        self.folder, self.used = Path(folder), used
        self.hashes = json.loads((self.folder/'artifact_sha256.json').read_text())
        used[str(self.folder/'artifact_sha256.json')] = sha256(self.folder/'artifact_sha256.json')

    def path(self, relative):
        path = self.folder/relative
        digest = sha256(path)
        if digest != self.hashes[relative]: raise ValueError(f'Saved artifact changed: {path}')
        self.used[str(path)] = digest
        return path

    def json(self, relative): return json.loads(self.path(relative).read_text())
    def table(self, relative): return pd.read_parquet(self.path(relative))


def verified_baseline(table, metadata, saved):
    result = evaluate_predictions(table, metadata)
    for key, value in result.items():
        if value != saved[key]: raise ValueError(f'Original baseline no longer matches: {key}')
    return result


def load_primary(rf_dir, lgbm_dir, used):
    tables, reports, shared_metadata = {}, {}, None
    for model, folder, prefix in (('ordinary_rf', rf_dir, 'ordinary'), ('ordinary_lgbm', lgbm_dir, 'ordinary_lgbm')):
        reader = ArtifactReader(folder, used)
        config = reader.json('configuration.json')
        metadata = {rid: config['metadata'][rid] for rid in HOLDOUTS}
        if shared_metadata is not None and metadata != shared_metadata: raise ValueError('CV annotations differ')
        shared_metadata = metadata
        frames = []
        for i, rid in enumerate(HOLDOUTS, 1):
            sub = f'fold_{i}_{rid}'
            frame = reader.table(f'{sub}/{prefix}_predictions.parquet')
            if set(frame.recording_id) != {rid} or not frame.split.eq('train').all():
                raise ValueError('Expected original training-partition CV holdout only')
            verified_baseline(frame, {rid: metadata[rid]}, reader.json(f'{sub}/{prefix}_evaluation.json'))
            frames.append(frame)
        table = pd.concat(frames, ignore_index=True)
        saved_summary = reader.json('summary.json')
        report = verified_baseline(table, metadata, saved_summary[prefix])
        tables[model], reports[model] = table, report
    pd.testing.assert_frame_equal(tables['ordinary_rf'][PREDICTION_METADATA], tables['ordinary_lgbm'][PREDICTION_METADATA])
    return tables, shared_metadata


def load_secondary(rf_dir, lgbm_dir, used):
    tables, shared_metadata = {}, None
    for model, folder, stem in (('ordinary_rf', rf_dir, 'random_forest'), ('ordinary_lgbm', lgbm_dir, 'ordinary_lgbm')):
        reader = ArtifactReader(folder, used)
        metadata = reader.json('configuration.json')['validation_metadata']
        if set(metadata) != {'chb01_17', 'chb01_18'}: raise ValueError('Expected original validation IDs only')
        if shared_metadata is not None and metadata != shared_metadata: raise ValueError('Validation annotations differ')
        shared_metadata = metadata
        table = reader.table(stem+'_validation.parquet')
        if not table.split.eq('validation').all(): raise ValueError('Expected validation predictions only')
        verified_baseline(table, metadata, reader.json(stem+'_evaluation.json'))
        tables[model] = table
    pd.testing.assert_frame_equal(tables['ordinary_rf'][PREDICTION_METADATA], tables['ordinary_lgbm'][PREDICTION_METADATA])
    return tables, shared_metadata


def evaluate_strategies(tables, metadata, output):
    output = Path(output);output.mkdir()
    reports, metric_rows, event_rows, changes = {}, [], [], []
    for model, original in tables.items():
        reports[model] = {}
        for rule in RULES:
            filtered = filter_predictions(original, metadata, rule)
            pd.testing.assert_frame_equal(filtered.drop(columns=['prediction','unfiltered_prediction']), original.drop(columns='prediction'))
            filtered.to_parquet(output/f'{model}_{rule}_predictions.parquet', index=False)
            # A binary alert strategy defines no new continuous score. Retain the
            # original score in output tables, but do not present its AP as a
            # property of the temporal filter (it would be unchanged).
            report = evaluate_predictions(filtered.drop(columns='score', errors='ignore'), metadata)
            report['evaluation_policy']['temporal_filter'] = rule  # Describes upstream processing; evaluator code unchanged.
            report['alert_rule'] = RULE_DESCRIPTIONS[rule]
            report['score_policy'] = 'Original scores retained in prediction files; AP not evaluated for alert strategies'
            reports[model][rule] = report
            write_json(output/f'{model}_{rule}_evaluation.json', report)
            for rid, group in filtered.groupby('recording_id', sort=False):
                a, b = group.unfiltered_prediction.to_numpy(), group.prediction.to_numpy()
                delta = group.loc[a != b].copy()
                delta['model'], delta['rule'] = model, rule
                delta['change'] = np.where(delta.prediction.eq(1), 'added_positive', 'removed_positive')
                changes.append(delta)
                single = report['per_recording'][rid]
                metric_rows.append({'model':model,'rule':rule,'recording_id':rid,**single['window_metrics'],**{
                    key:value for key,value in single['event_metrics'].items() if key not in ('alarms','seizures')},
                    'raw_positives':int(a.sum()),'filtered_positives':int(b.sum()),
                    'added_positives':int(((a==0)&(b==1)).sum()),'removed_positives':int(((a==1)&(b==0)).sum())})
                for event in single['event_metrics']['seizures']:
                    event_rows.append({'model':model,'rule':rule,'recording_id':rid,**event})
    pd.concat(changes,ignore_index=True).to_parquet(output/'changed_predictions.parquet',index=False)
    pd.DataFrame(event_rows).to_csv(output/'events.csv',index=False)
    pd.DataFrame(metric_rows).to_csv(output/'per_recording_metrics.csv',index=False)
    pooled = []
    for model, rules in reports.items():
        for rule, report in rules.items():
            counts = [row for row in metric_rows if row['model']==model and row['rule']==rule]
            pooled.append({'model':model,'rule':rule,**report['window_metrics'],**report['event_metrics'],
                           **{key:sum(c[key] for c in counts) for key in ('raw_positives','filtered_positives','added_positives','removed_positives')}})
    pd.DataFrame(pooled).to_csv(output/'pooled_metrics.csv',index=False)
    write_json(output/'comparison.json',reports)
    return reports


def plot_events(tables, metadata, output, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ids = [rid for rid,m in metadata.items() if m['seizures_sec']]
    fig, axes = plt.subplots(len(ids),2,figsize=(12,2.6*len(ids)+1),layout='constrained',squeeze=False)
    for row,rid in enumerate(ids):
        on,off = metadata[rid]['seizures_sec'][0]
        for col,(model,table) in enumerate(tables.items()):
            ax=axes[row,col]
            group=table[table.recording_id.eq(rid)]
            for j,rule in enumerate(RULES):
                filtered=filter_predictions(group,{rid:metadata[rid]},rule)
                positives=filtered[filtered.prediction.eq(1)]
                ax.scatter(positives.end_sec,np.full(len(positives),j),marker='|',s=180,label=rule)
            ax.axvspan(on,off,color='#e9b44c',alpha=.2)
            ax.axvline(on,color='gray',linewidth=.6);ax.axvline(off,color='gray',linewidth=.6)
            ax.set(title=f'{rid}: {model}',yticks=range(3),yticklabels=RULES,ylim=(-.5,2.5),xlim=(on-20,off+30))
            ax.grid(axis='x',alpha=.2)
    for ax in axes[-1]:ax.set_xlabel('Actual window-end decision time (seconds)')
    fig.suptitle(title+'; ticks mark positive decisions, shading is annotation')
    fig.savefig(Path(output)/'positive_decisions.png',dpi=160);plt.close(fig)


def run_experiment(rf_cv, lgbm_cv, rf_validation, lgbm_validation, output):
    output=Path(output)
    if output.exists(): raise FileExistsError(f'Preserving existing output {output}')
    output.mkdir(parents=True,exist_ok=False)
    config={'created_utc':datetime.now(timezone.utc).isoformat(),'rules':RULE_DESCRIPTIONS,
            'startup':'For confirmation rules only: first decision 0; second may fire after two observed positives',
            'missing_windows':'Reject missing/incomplete/duplicate/shuffled/variable-duration windows; no imputation',
            'timing':'Current window end; no future values, backdating or within-recording state carry across gaps',
            'rearm':'Existing evaluator: one negative filtered decision closes and re-arms an episode',
            'primary':'four original training CV holdouts','secondary':'original validation, descriptive only; rules unchanged',
            'decision_priorities':['preserve event detection','reduce false alarms','minimize delay','simplicity'],
            'versions':{name:version(name) for name in ('numpy','pandas','scikit-learn','matplotlib')},
            'source_sha256':{n:sha256(PROJECT_ROOT/'src'/n) for n in ('temporal_filter.py','temporal_filter_experiment.py','evaluate.py')}}
    write_json(output/'configuration.json',config)  # Fixed before either assessment.
    used={}
    primary,cv_metadata=load_primary(rf_cv,lgbm_cv,used)
    primary_report=evaluate_strategies(primary,cv_metadata,output/'primary_cv')
    plot_events(primary,cv_metadata,output/'primary_cv','Primary out-of-fold alert strategies')
    # Only after completing the primary assessment, load the designated secondary split.
    secondary,val_metadata=load_secondary(rf_validation,lgbm_validation,used)
    secondary_report=evaluate_strategies(secondary,val_metadata,output/'secondary_validation')
    plot_events(secondary,val_metadata,output/'secondary_validation','Secondary validation alert strategies')
    write_json(output/'input_sha256.json',used)
    write_json(output/'verification.json',{'model_fits':0,'baseline_reports_reproduced':True,
        'labels_metadata_scores_preserved':True,'final_test_or_supplementary_rows_used':False,
        'primary_recordings':list(cv_metadata),'secondary_recordings':list(val_metadata)})
    write_json(output/'artifact_sha256.json',{str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*')) if p.is_file()})
    return {'primary_cv':primary_report,'secondary_validation':secondary_report}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rf-cv',type=Path,default=PROJECT_ROOT/'results/rf_cv_v1')
    parser.add_argument('--lgbm-cv',type=Path,default=PROJECT_ROOT/'results/lightgbm_cv_v1')
    parser.add_argument('--rf-validation',type=Path,default=PROJECT_ROOT/'results/baseline_v1')
    parser.add_argument('--lgbm-validation',type=Path,default=PROJECT_ROOT/'results/lightgbm_v1')
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'results/temporal_filtering_v1')
    args=parser.parse_args()
    run_experiment(args.rf_cv,args.lgbm_cv,args.rf_validation,args.lgbm_validation,args.output)


if __name__=='__main__':main()
