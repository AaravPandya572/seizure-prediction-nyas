"""Read-only targeted raw EEG investigation; no model fitting or signal cleaning."""

import argparse
import json
from pathlib import Path

import joblib
import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

from .baseline_rf import DEFAULT_DATA, DEFAULT_MANIFEST, write_json
from .evaluate import evaluate_predictions
from .preprocess import (CHANNELS, FEATURE_COLUMNS, PROJECT_ROOT, causal_bandpass,
                         extract_epoch_features, parse_summary_file, select_channels, sha256)
from .rf_cross_validation import load_training_data, predict_holdout

RECORDINGS = ("chb01_15", "chb01_03", "chb01_04", "chb01_16")
DISPLAY_CHANNELS = ("FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FZ-CZ", "CZ-PZ")


def window_spectrum(epoch, sfreq):
    """Exact preprocessing Welch settings; input µV, density µV²/Hz."""
    if epoch.ndim != 2 or epoch.shape[1] != round(5 * sfreq) or not np.isfinite(epoch).all():
        raise ValueError("Expected finite channels x five-second samples")
    freq, psd = welch(epoch, fs=sfreq, window="hann", nperseg=round(2 * sfreq),
                      noverlap=round(sfreq), detrend="constant", axis=1)
    width = freq[1] - freq[0]
    total = psd[:, (freq >= .5) & (freq <= 45)].sum(axis=1) * width
    gamma = psd[:, (freq >= 30) & (freq <= 45)].sum(axis=1) * width
    return freq, psd, total, gamma


def waveform_plot(raw, filtered, sfreq, start, end, onset, offset, output, title):
    import matplotlib.pyplot as plt
    indices = [CHANNELS.index(c) for c in DISPLAY_CHANNELS]
    times = np.arange(round(start * sfreq), round(end * sfreq)) / sfreq
    fig, axes = plt.subplots(1, 2, figsize=(18, 9), layout="constrained", sharey=True)
    spacing = 1200.0  # µV; identical scale, no channel normalization or clipping.
    baselines = np.arange(len(indices))[::-1] * spacing
    for ax, source, name in zip(axes, (raw, filtered), ("Original EDF (unfiltered)", "Established causal 0.5–45 Hz filter")):
        segment = source[indices, round(start * sfreq):round(end * sfreq)]
        for i, baseline in enumerate(baselines):
            ax.plot(times, segment[i] + baseline, linewidth=.45, color="#193d69")
        ax.axvspan(onset, offset, color="#e9b44c", alpha=.2, label="Annotated seizure")
        ax.axvline(onset, color="#946c10", linewidth=.8)
        ax.axvline(offset, color="#946c10", linewidth=.8)
        ax.axvspan(1780, 1785, color="#b23a48", alpha=.15, label="Delayed-alarm window")
        ax.axvline(1785, color="#b23a48", linestyle="--", linewidth=.8)
        ax.set(title=name, xlabel="Seconds from recording start", xlim=(start, end),
               yticks=baselines, yticklabels=DISPLAY_CHANNELS, ylim=(-650, baselines[0]+650))
        xbar = end - .02*(end-start)
        ax.plot([xbar, xbar], [0, 500], color="black", linewidth=2)
        ax.text(xbar, 550, "500 µV", ha="right", fontsize=8)
        ax.grid(axis="x", alpha=.15)
    axes[0].set_ylabel("Bipolar signal (µV); vertical separation 1200 µV")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(title)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def run_analysis(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    if output.exists(): raise FileExistsError(f"Preserving existing output {output}")
    frame, metadata = load_training_data(DEFAULT_DATA, DEFAULT_MANIFEST)
    provenance = json.loads(DEFAULT_DATA.with_suffix('.json').read_text())['raw_recordings']
    summary = parse_summary_file(PROJECT_ROOT / "data/raw/chb01-summary.txt")
    cv = PROJECT_ROOT / "results/rf_cv_v1"
    fold = cv / "fold_3_chb01_15"
    definition = json.loads((fold / "definition.json").read_text())
    if definition["held_out"] != "chb01_15" or "chb01_15" in definition["train_recordings"]:
        raise ValueError("Expected the fold that excluded chb01_15")
    for name in ("ordinary", "weighted"):
        model = joblib.load(fold / f"{name}.joblib")
        predicted = predict_holdout(model, frame[frame.recording_id.eq("chb01_15")], "chb01_15").reset_index(drop=True)
        pd.testing.assert_frame_equal(predicted, pd.read_parquet(fold / f"{name}_predictions.parquet"))
        report = evaluate_predictions(predicted, {"chb01_15": metadata["chb01_15"]})
        old = json.loads((fold / f"{name}_evaluation.json").read_text())
        for key,value in report.items():
            if value != old[key]: raise ValueError("Original fold report mismatch")
    output.mkdir(parents=True)
    rows, checks, spectra, exemplars = [], {}, {}, {}
    for rid in RECORDINGS:
        path = PROJECT_ROOT / "data/raw" / f"{rid}.edf"
        on, off = metadata[rid]["seizures_sec"][0]
        if summary[path.name] != [(on, off)]: raise ValueError("Summary/CV annotation mismatch")
        windows = {"pre": (int(on//5)*5-5, int(on//5)*5),
                   "ictal_mid": (int(((on+off)/2)//5)*5, int(((on+off)/2)//5)*5+5)}
        if rid == "chb01_15":
            windows.update(post_pre=(1775,1780), alarm=(1780,1785), post_next=(1785,1790))
        stop = 1810 if rid == "chb01_15" else max(b for a,b in windows.values())
        with mne.io.read_raw_edf(path, preload=False, verbose="WARNING") as edf:
            chosen = select_channels(edf.ch_names)
            sfreq = float(edf.info['sfreq'])
            raw = edf.get_data(picks=chosen, start=0, stop=round(stop*sfreq))*1e6
            checks[rid] = {"raw_sha256":sha256(path), "canonical_channels":list(CHANNELS),
                           "selected_edf_channels":chosen, "sampling_rate_hz":sfreq,
                           "original_units":edf._orig_units, "annotation_sec":[on,off],
                           "read_prefix_stop_sec":stop, "compared_windows":0, "max_absolute_feature_error":0.0}
        filtered = causal_bandpass(raw, sfreq)  # Full prefix from t=0, not restarted at crop.
        if (checks[rid]['raw_sha256'] != provenance[path.name]['sha256'] or
                chosen != provenance[path.name]['selected_edf_channels']):
            raise ValueError('Raw EDF or channel mapping differs from preprocessing provenance')
        held = frame[frame.recording_id.eq(rid)]
        # Check every complete window in the target crop, plus context exemplars.
        starts = list(range(1700,1810,5)) if rid == "chb01_15" else [a for a,b in windows.values()]
        for start in starts:
            features,_ = extract_epoch_features(filtered[:,round(start*sfreq):round((start+5)*sfreq)], sfreq)
            saved = held.loc[held.start_sec.eq(start),FEATURE_COLUMNS].iloc[0].to_numpy(dtype=float)
            if not np.allclose(features,saved,rtol=1e-10,atol=1e-10):
                raise ValueError(f"Feature mismatch: {rid} at {start}; original results require review")
            checks[rid]['compared_windows'] += 1
            checks[rid]['max_absolute_feature_error'] = max(checks[rid]['max_absolute_feature_error'],float(abs(features-saved).max()))
        for label,(start,end) in windows.items():
            for mode,signal in (("raw",raw),("filtered",filtered)):
                epoch = signal[:,round(start*sfreq):round(end*sfreq)]
                freq,psd,total,gamma = window_spectrum(epoch,sfreq)
                if rid == 'chb01_15': spectra[label,mode]=(freq,psd)
                for i,ch in enumerate(CHANNELS):
                    rows.append({"recording_id":rid,"period":label,"mode":mode,"channel":ch,
                                 "start_sec":start,"end_sec":end,"total_0p5_45_uv2":float(total[i]),
                                 "gamma_30_45_uv2":float(gamma[i]), "other_0p5_30_uv2":float(total[i]-gamma[i]),
                                 "relative_gamma":float(gamma[i]/total[i]),"variance_uv2":float(np.var(epoch[i])),
                                 "line_length_uv":float(abs(np.diff(epoch[i])).sum()),
                                 "peak_to_peak_uv":float(np.ptp(epoch[i])),"max_abs_uv":float(abs(epoch[i]).max())})
        exemplars[rid]=(raw[:,round(windows['ictal_mid'][0]*sfreq):round(windows['ictal_mid'][1]*sfreq)],
                        filtered[:,round(windows['ictal_mid'][0]*sfreq):round(windows['ictal_mid'][1]*sfreq)],sfreq,windows['ictal_mid'])
        if rid == 'chb01_15':
            waveform_plot(raw,filtered,sfreq,1700,1810,on,off,output/'waveforms_overview.png', 'chb01_15: onset 1732, offset 1772; identical scaling in both panels')
            waveform_plot(raw,filtered,sfreq,1775,1790,on,off,output/'waveforms_alarm_zoom.png','chb01_15: shaded alarm window [1780,1785); decision at 1785')
    table=pd.DataFrame(rows)
    table.to_parquet(output/'window_measurements.parquet',index=False)
    write_json(output/'verification.json',checks)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for row,ch in enumerate(('FP2-F8','T8-P8')):
        for col,mode in enumerate(('raw','filtered')):
            ax=axes[row,col]
            for period in ('pre','ictal_mid','post_pre','alarm','post_next'):
                freq,psd=spectra[period,mode]
                ax.semilogy(freq[1:],psd[CHANNELS.index(ch),1:],label=period,linewidth=1)
            ax.axvspan(30,45,color='#e9b44c',alpha=.2)
            ax.set(title=f'{ch}: {mode}',xlim=(.5,100),ylim=(1e-3,1e5),xlabel='Frequency (Hz)',ylabel='PSD (µV²/Hz)')
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Five-second windows; exact preprocessing Welch settings; yellow = 30–45 Hz')
    fig.savefig(output/'spectra_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(12,10),layout='constrained',sharex=True,sharey=True)
    for row,rid in enumerate(RECORDINGS):
        raw,filtered,sfreq,bounds=exemplars[rid]
        for col,ch in enumerate(('FP2-F8','T8-P8')):
            i=CHANNELS.index(ch);t=np.arange(raw.shape[1])/sfreq
            axes[row,col].plot(t,raw[i],color='#888888',linewidth=.7,label='raw')
            axes[row,col].plot(t,filtered[i],color='#1768ac',linewidth=.7,label='causal filtered')
            axes[row,col].set(title=f'{rid} {ch}: [{bounds[0]},{bounds[1]})',ylim=(-650,650),ylabel='µV',xlabel='Seconds within window')
            axes[row,col].grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Mid-event five-second exemplars, common amplitude scale')
    fig.savefig(output/'detected_seizure_context.png',dpi=170);plt.close(fig)
    write_json(output/'configuration.json',{'source_sha256':sha256(Path(__file__)),'raw_files':list(RECORDINGS),
        'filter':'existing causal_bandpass; prefix from t=0, zero initial state; no cleaning/notch',
        'psd':'Welch 2s Hann, 1s overlap, constant detrending; 0.5Hz bin sums times 0.5Hz; gamma[30,45], total[0.5,45]',
        'waveform_units':'MNE volts multiplied by 1e6 => µV; no trace normalization or display decimation',
        'raw_prefix_note':'earlier samples read only for correct causal state; interpretation restricted to target crops/exemplars',
        'model':'saved CV fold_3_chb01_15 only; verified predictions/evaluation, no training'})
    write_json(output/'artifact_sha256.json',{p.name:sha256(p) for p in sorted(output.iterdir())})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'results/raw_eeg_investigation_v1')
    run_analysis(parser.parse_args().output)


if __name__=='__main__': main()
