"""
================================================================================
MODULE: src/preprocess.py
PROJECT: Seizure Prediction via Failure Analysis (NYAS BrainTech Challenge)
DESCRIPTION:
    End-to-end signal processing and feature engineering pipeline for the
    CHB-MIT Scalp EEG Database. This script:
      1. Parses 'chb01-summary.txt' to automatically locate seizure timestamps.
      2. Loads raw .edf brainwave recordings using MNE-Python.
      3. Applies a 60 Hz Notch filter and a 0.5-45 Hz Butterworth Bandpass filter.
      4. Slices recordings into 5-second epochs (with configurable stride).
      5. Extracts 7 temporal & spectral features across 18 EEG channels (126 total).
      6. Saves clean, structured training and testing datasets to 'data/processed/'.
================================================================================
"""

# ==============================================================================
# 1. IMPORTS & LIBRARY SETUP
# ==============================================================================
import os                   # Interacting with Mac file paths (data/raw, data/processed)
import re                   # Regular expressions for parsing the doctor's summary text file
import warnings             # Suppressing non-critical MNE channel naming warnings
import numpy as np          # High-speed linear algebra and matrix manipulation
import pandas as pd         # Structuring our 126 features into tabular DataFrames
import mne                  # Neuroimaging library for reading and filtering .edf files
from scipy.signal import welch  # Welch's method for Power Spectral Density (FFT)

# Silence standard MNE console clutter so we only see our custom progress messages
warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")


# ==============================================================================
# 2. GLOBAL SIGNAL PROCESSING CONSTANTS
# ==============================================================================
RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")

NUM_CHANNELS = 18           # Standardize to 18 common bipolar channels across recordings
WINDOW_SEC = 5.0            # Length of each brainwave snapshot (5 seconds = 1,280 samples at 256 Hz)
PREICTAL_HORIZON_SEC = 1800 # 30 minutes (1,800 seconds) prior to seizure onset = Warning Zone (Class 1)

# Canonical neurological EEG frequency bands (in Hertz)
FREQ_BANDS = {
    "delta": (0.5, 4.0),    # Deep sleep / slow-wave abnormalities
    "theta": (4.0, 8.0),    # Drowsiness / early focal slowing
    "alpha": (8.0, 13.0),   # Relaxed wakefulness
    "beta":  (13.0, 30.0),  # Active thinking / motor cortex activity
    "gamma": (30.0, 45.0),  # High-frequency neural synchronization (key preictal marker)
}


# ==============================================================================
# 3. SUMMARY FILE PARSER
# ==============================================================================
def parse_summary_file(summary_path):
    """
    Reads the hospital 'chb01-summary.txt' file and extracts exact seizure start
    and end times (in seconds) for every .edf recording.

    Parameters:
        summary_path (str): Path to the *-summary.txt file.

    Returns:
        seizure_dict (dict): Maps filename -> list of (start_sec, end_sec) tuples.
                             Example: {'chb01_03.edf': [(2996, 3036)], 'chb01_01.edf': []}
    """
    seizure_dict = {}
    current_file = None
    start_time = None

    with open(summary_path, "r") as f:
        for line in f:
            line = line.strip()

            # Match lines like: "File Name: chb01_03.edf"
            if line.startswith("File Name:"):
                current_file = line.split(":")[1].strip()
                seizure_dict[current_file] = []

            # Match lines like: "Seizure Start Time: 2996 seconds" or "Seizure 1 Start Time: ..."
            elif "Start Time:" in line and "Seizure" in line:
                # Extract the integer number of seconds using regex
                match = re.search(r"(\d+)\s*seconds", line)
                if match:
                    start_time = int(match.group(1))

            # Match lines like: "Seizure End Time: 3036 seconds"
            elif "End Time:" in line and "Seizure" in line:
                match = re.search(r"(\d+)\s*seconds", line)
                if match and current_file is not None and start_time is not None:
                    end_time = int(match.group(1))
                    seizure_dict[current_file].append((start_time, end_time))
                    start_time = None

    return seizure_dict


# ==============================================================================
# 4. SIGNAL INGESTION & DIGITAL FILTERING
# ==============================================================================
def load_and_filter_edf(edf_path):
    """
    Loads a raw .edf file into memory, standardizes the 18 bipolar channels,
    and removes electromagnetic + muscle noise using digital filters.

    Parameters:
        edf_path (str): Full path to the .edf file.

    Returns:
        data (np.ndarray): Filtered 2D voltage matrix of shape (18_channels, N_samples).
        sfreq (float): Sampling frequency in Hz (256.0 Hz for CHB-MIT).
        ch_names (list): List of the 18 selected channel names.
    """
    # preload=True loads the full voltage matrix into RAM so we can filter it fast
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # Clean up duplicate channel names that MNE appends '-0' or '-1' to (like T8-P8-0)
    # and pick the first 18 valid EEG channels so every file has identical dimensions
    clean_channels = [ch for ch in raw.ch_names if ch != "-" and not ch.startswith("STI")]
    selected_channels = clean_channels[:NUM_CHANNELS]
    raw.pick(selected_channels)

    # 1. Notch Filter at 60 Hz: Removes US hospital electrical wall-outlet hum
    raw.notch_filter(freqs=60.0, verbose=False)

    # 2. Butterworth Bandpass Filter (0.5 Hz to 45.0 Hz):
    #    - Cuts < 0.5 Hz to remove slow sweat/breathing baseline drift
    #    - Cuts > 45.0 Hz to remove high-frequency jaw/muscle (EMG) clenching noise
    raw.filter(l_freq=0.5, h_freq=45.0, method="iir", verbose=False)

    # Extract the underlying 2D NumPy matrix: shape (18, total_time_samples)
    # Multiply by 1e6 to convert Volts (V) to Microvolts (uV) so numbers aren't tiny (10^-12)
    data = raw.get_data() * 1e6
    sfreq = float(raw.info["sfreq"])

    return data, sfreq, selected_channels


# ==============================================================================
# 5. VECTORIZED MATHEMATICAL FEATURE EXTRACTION (126 FEATURES)
# ==============================================================================
def extract_epoch_features(epoch_matrix, sfreq):
    """
    Converts a single 5-second multi-channel brainwave clip into a 1D vector
    of 126 mathematical features (18 channels x 7 features per channel).

    Parameters:
        epoch_matrix (np.ndarray): Shape (18, 1280) representing 5s of 18-channel EEG.
        sfreq (float): Sampling rate (256.0 Hz).

    Returns:
        feature_vector (np.ndarray): 1D array of 126 floats.
    """
    # --- FEATURE 1: Signal Variance (Hjorth Activity / "Loudness") ---
    # Computed simultaneously across all 18 channels along the time axis (axis=1)
    variance = np.var(epoch_matrix, axis=1)  # Shape: (18,)

    # --- FEATURE 2: Line Length ("Squiggliness" / Waveform Complexity) ---
    # Sum of absolute voltage jumps between adjacent time points: sum(|x[i] - x[i-1]|)
    line_length = np.sum(np.abs(np.diff(epoch_matrix, axis=1)), axis=1)  # Shape: (18,)

    # --- FEATURES 3 to 7: Relative Spectral Band Powers (Welch's PSD) ---
    # nperseg = 2 seconds (512 samples) gives 0.5 Hz frequency resolution
    freqs, psd = welch(epoch_matrix, fs=sfreq, nperseg=int(sfreq * 2), axis=1)

    # Calculate total spectral power between 0.5 and 45 Hz for normalization
    total_mask = (freqs >= 0.5) & (freqs <= 45.0)
    total_power = np.trapezoid(psd[:, total_mask], freqs[total_mask], axis=1)
    # Prevent division by zero if a sensor flatlines
    total_power = np.where(total_power == 0, 1e-12, total_power)

    band_powers = []
    for band_name, (low_f, high_f) in FREQ_BANDS.items():
        band_mask = (freqs >= low_f) & (freqs < high_f)
        # Integrate area under the PSD curve for this specific frequency band
        abs_power = np.trapezoid(psd[:, band_mask], freqs[band_mask], axis=1)
        # Convert to Relative Power (0.0 to 1.0 ratio of total brainwave energy)
        rel_power = abs_power / total_power
        band_powers.append(rel_power)  # Each is shape (18,)

    # Stack all 7 feature arrays per channel: shape (18_channels, 7_features)
    # Order per channel: [Variance, LineLength, Delta, Theta, Alpha, Beta, Gamma]
    channel_features = np.column_stack([variance, line_length] + band_powers)

    # Flatten into a single row of 126 numbers: [Ch1_Var, Ch1_LL, ..., Ch18_Gamma]
    return channel_features.flatten()


# ==============================================================================
# 6. TIME-AWARE SLIDING WINDOW SLICER
# ==============================================================================
def slice_and_extract(edf_filename, seizure_times, stride_sec=5.0, role="train"):
    """
    Slices a continuous recording into 5-second windows, assigns ground-truth
    clinical labels (0 = Interictal Normal, 1 = Preictal Warning), and extracts
    the 126 features for each window.

    Scientific Labeling Logic:
      - If the file has NO seizures (e.g., chb01_01.edf):
          Every 5-second window is labeled Class 0 (Interictal).
      - If the file HAS a seizure starting at t_onset (e.g., chb01_03.edf at 2996s):
          1. Windows before (t_onset - 1800s) -> Labeled Class 0 (Interictal).
          2. Windows inside [t_onset - 1800s, t_onset] -> Labeled Class 1 (Preictal).
          3. Windows during/after t_onset -> DISCARDED (Ictal/Postictal excluded
             so the AI learns to predict future seizures, not detect active ones).

    Parameters:
        edf_filename (str): Name of the file in data/raw/ (e.g., 'chb01_03.edf').
        seizure_times (list): List of (start_sec, end_sec) tuples from summary parser.
        stride_sec (float): Step size in seconds for Preictal windows.
                            Set to 5.0 for standard non-overlapping windows,
                            or 1.0 for 5x overlapping minority augmentation.
        role (str): Tag identifying whether this file belongs to 'train' or 'test'.

    Returns:
        df (pd.DataFrame): DataFrame where each row is a 5s window with 126 features + metadata.
    """
    edf_path = os.path.join(RAW_DATA_DIR, edf_filename)
    print(f"  -> Loading & filtering '{edf_filename}'...")
    data, sfreq, ch_names = load_and_filter_edf(edf_path)

    total_samples = data.shape[1]
    total_duration_sec = total_samples / sfreq
    window_samples = int(WINDOW_SEC * sfreq)  # 5.0 * 256 = 1,280 samples

    rows = []

    # CASE A: Seizure-Free Baseline Recording (All Interictal Class 0)
    if len(seizure_times) == 0:
        # Step through the entire recording in non-overlapping 5-second jumps
        for start_sec in np.arange(0, total_duration_sec - WINDOW_SEC + 0.01, WINDOW_SEC):
            start_idx = int(start_sec * sfreq)
            end_idx = start_idx + window_samples
            epoch = data[:, start_idx:end_idx]

            feats = extract_epoch_features(epoch, sfreq)
            rows.append(np.append(feats, [0, start_sec, edf_filename, role]))

    # CASE B: Recording Contains a Seizure
    else:
        onset_sec, _ = seizure_times[0]
        preictal_start_sec = max(0.0, onset_sec - PREICTAL_HORIZON_SEC)

        # 1. Extract Normal Interictal windows prior to the 30-min warning zone (5s stride)
        for start_sec in np.arange(0, preictal_start_sec - WINDOW_SEC + 0.01, WINDOW_SEC):
            start_idx = int(start_sec * sfreq)
            end_idx = start_idx + window_samples
            epoch = data[:, start_idx:end_idx]

            feats = extract_epoch_features(epoch, sfreq)
            rows.append(np.append(feats, [0, start_sec, edf_filename, role]))

        # 2. Extract Preictal Warning windows inside [onset - 30 min, onset]
        #    Uses configurable 'stride_sec' (5.0s for baseline, 1.0s for balanced training)
        for start_sec in np.arange(preictal_start_sec, onset_sec - WINDOW_SEC + 0.01, stride_sec):
            start_idx = int(start_sec * sfreq)
            end_idx = start_idx + window_samples
            epoch = data[:, start_idx:end_idx]

            feats = extract_epoch_features(epoch, sfreq)
            rows.append(np.append(feats, [1, start_sec, edf_filename, role]))

    # Build human-readable column names for all 126 features
    # Format: ChName_Feature (e.g., 'FP1-F7_var', 'FP1-F7_gamma', etc.)
    stat_names = ["var", "line_length", "delta", "theta", "alpha", "beta", "gamma"]
    feature_cols = [f"{ch}_{stat}" for ch in ch_names for stat in stat_names]
    all_cols = feature_cols + ["label", "time_sec", "source_file", "split_role"]

    df = pd.DataFrame(rows, columns=all_cols)

    # Cast numeric columns back from string/object to float64/int64
    for col in feature_cols + ["time_sec"]:
        df[col] = df[col].astype(float)
    df["label"] = df["label"].astype(int)

    return df


# ==============================================================================
# 7. MAIN EXECUTION PIPELINE
# ==============================================================================
if __name__ == "__main__":
    print("=================================================================")
    print(" NYAS BrainTech: Starting CHB-MIT Preprocessing Pipeline (chb01)")
    print("=================================================================\n")

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Step 1: Parse the clinical summary text file to get ground-truth onset times
    summary_file = os.path.join(RAW_DATA_DIR, "chb01-summary.txt")
    seizure_map = parse_summary_file(summary_file)

    print("1. Discovered Seizure Annotations from chb01-summary.txt:")
    for fname in ["chb01_01.edf", "chb01_03.edf", "chb01_04.edf"]:
        print(f"   - {fname}: {seizure_map.get(fname, [])}")
    print()

    # Step 2: Process our 3 recordings
    print("2. Extracting 126 Spectral/Temporal Features per 5-second Window...")
    
    # Extract all 720 normal windows from chb01_01.edf
    df_normal_all = slice_and_extract("chb01_01.edf", seizure_map["chb01_01.edf"], stride_sec=5.0, role="train")
    
    # Split chb01_01.edf chronologically: first 50% (360 windows) to Train, last 50% (360 windows) to Test
    # This guarantees our Test set has plenty of normal brainwaves to measure False Alarms!
    half_idx = len(df_normal_all) // 2
    df_train_normal = df_normal_all.iloc[:half_idx].copy()
    df_test_normal = df_normal_all.iloc[half_idx:].copy()
    df_test_normal["split_role"] = "test"

    # Extract Seizure #1 (chb01_03.edf) for Training, and Seizure #2 (chb01_04.edf) for Blind Testing
    df_train_seizure = slice_and_extract("chb01_03.edf", seizure_map["chb01_03.edf"], stride_sec=5.0, role="train")
    df_test_seizure = slice_and_extract("chb01_04.edf", seizure_map["chb01_04.edf"], stride_sec=5.0, role="test")

    # Combine all splits into the master Parquet table
    master_df = pd.concat([df_train_normal, df_test_normal, df_train_seizure, df_test_seizure], ignore_index=True)
    output_path = os.path.join(PROCESSED_DATA_DIR, "chb01_features.parquet")
    master_df.to_parquet(output_path, index=False)

    # Step 3: Print Diagnostic Summary for the Technical Lead
    print("\n=================================================================")
    print(f" SUCCESS! Saved Master Feature Matrix to: {output_path}")
    print("=================================================================")
    print(f" Total 5-Second Windows Extracted : {len(master_df)}")
    print(f" Total Features per Window        : 126 (18 channels x 7 stats)")
    print("\n Class Breakdown by Split:")
    print(master_df.groupby(["split_role", "label"]).size().unstack(fill_value=0).rename(
        columns={0: "Class 0 (Interictal Normal)", 1: "Class 1 (Preictal Warning)"}
    ))
    print("=================================================================\n")