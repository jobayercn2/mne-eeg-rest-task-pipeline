from pathlib import Path
import re
import warnings

import mne
import pandas as pd
import numpy as np


# ============================================================
# Project paths
# ============================================================

RAW_DIR = Path(r"I:\EEG_Python_Project\data\raw\resting_state")
OUT_DIR = Path(r"I:\EEG_Python_Project\data\processed\resting_state")

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Preprocessing settings
# ============================================================

TARGET_SFREQ = 256          # Hz
L_FREQ = 1.0                # High-pass filter
H_FREQ = 40.0               # Low-pass filter
NOTCH_FREQS = [50, 100]     # European line noise + harmonic

EPOCH_LENGTH = 2.0          # seconds
AMPLITUDE_REJECT = 120e-6   # 120 microvolts

# Important:
# Keep False unless EOG channels are properly identified.
RUN_ICA = False

ICA_RANDOM_STATE = 42
ICA_MAX_ITER = "auto"


# ============================================================
# File handling
# ============================================================

def find_eeg_files(raw_dir):
    """
    Recursively find EEG files.
    Supported formats: .bdf, .edf, .set, .fif
    """
    extensions = ["*.bdf", "*.edf", "*.set", "*.fif"]
    files = []

    for ext in extensions:
        files.extend(raw_dir.rglob(ext))

    return sorted(files)


def get_subject_id(file_path):
    """
    Extract subject ID such as S10, S11, S12.
    """
    for part in reversed(file_path.parts):
        if re.match(r"^S\d+$", part, re.IGNORECASE):
            return part.upper()

    match = re.search(r"S\d+", file_path.stem, re.IGNORECASE)

    if match:
        return match.group(0).upper()

    return file_path.stem


def get_session_label(file_path):
    """
    Detect session label from filename.
    """
    name = file_path.stem.upper()

    if "PRE" in name:
        return "PRE"
    elif "POST" in name:
        return "POST"
    else:
        return "REST"


def load_raw_eeg(file_path):
    """
    Load EEG file depending on extension.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".bdf":
        raw = mne.io.read_raw_bdf(file_path, preload=True, verbose="ERROR")
    elif suffix == ".edf":
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose="ERROR")
    elif suffix == ".set":
        raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose="ERROR")
    elif suffix == ".fif":
        raw = mne.io.read_raw_fif(file_path, preload=True, verbose="ERROR")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return raw


# ============================================================
# Channel handling
# ============================================================

def clean_channel_names(raw):
    """
    Clean channel names for consistency.
    """
    rename_dict = {}

    for ch in raw.ch_names:
        new_name = ch.strip()
        new_name = new_name.replace(" ", "")
        new_name = new_name.replace("-Ref", "")
        new_name = new_name.replace("_Ref", "")
        rename_dict[ch] = new_name

    raw.rename_channels(rename_dict)

    return raw


def drop_external_channels(raw):
    """
    Drop BioSemi EXG channels from resting-state EEG analysis.

    EXG1-EXG8 are external channels. They should not be treated as scalp EEG
    unless explicitly used as EOG/EMG channels.
    """
    exg_chs = [ch for ch in raw.ch_names if ch.upper().startswith("EXG")]

    if exg_chs:
        raw.drop_channels(exg_chs)
        print(f"Dropped external EXG channels: {exg_chs}")

    return raw, exg_chs


def mark_eog_channels_if_present(raw):
    """
    Mark EOG channels only if they are explicitly named as EOG/VEOG/HEOG.
    """
    eog_candidates = [
        ch for ch in raw.ch_names
        if any(label in ch.upper() for label in ["EOG", "VEOG", "HEOG"])
    ]

    if eog_candidates:
        raw.set_channel_types({ch: "eog" for ch in eog_candidates})
        print(f"Detected EOG channels: {eog_candidates}")

    return raw, eog_candidates


# ============================================================
# Main preprocessing
# ============================================================

def basic_cleaning(raw):
    """
    Resting-state EEG preprocessing:
    1. Clean channel names
    2. Drop EXG channels
    3. Mark EOG channels if explicitly present
    4. Apply 10-20 montage
    5. Keep EEG channels only
    6. Band-pass filter 1-40 Hz
    7. Notch filter 50/100 Hz
    8. Resample to 256 Hz
    9. Average reference
    """

    raw = raw.copy()

    raw = clean_channel_names(raw)
    raw, dropped_exg = drop_external_channels(raw)
    raw, eog_candidates = mark_eog_channels_if_present(raw)

    # Set standard montage
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing="ignore")

    # Keep EEG channels only
    raw.pick("eeg")

    if len(raw.ch_names) == 0:
        raise RuntimeError("No EEG channels found after channel selection.")

    # Filter
    raw.filter(
        l_freq=L_FREQ,
        h_freq=H_FREQ,
        fir_design="firwin",
        verbose="ERROR"
    )

    # Notch filter only valid frequencies below Nyquist
    nyquist = raw.info["sfreq"] / 2
    valid_notches = [freq for freq in NOTCH_FREQS if freq < nyquist]

    if valid_notches:
        raw.notch_filter(
            freqs=valid_notches,
            fir_design="firwin",
            verbose="ERROR"
        )

    # Resample
    if not np.isclose(raw.info["sfreq"], TARGET_SFREQ):
        raw.resample(TARGET_SFREQ, verbose="ERROR")

    # Average reference
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    return raw, dropped_exg, eog_candidates


# ============================================================
# Optional ICA
# ============================================================

def run_ica_cleaning(raw):
    """
    Optional ICA.

    Current recommendation:
    Use only if real EOG channels are identified.
    Otherwise, keep RUN_ICA = False.
    """

    raw = raw.copy()

    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False)

    if len(eeg_picks) < 2:
        warnings.warn("Not enough EEG channels for ICA. Skipping ICA.")
        return raw, 0

    n_components = min(20, len(eeg_picks) - 1)

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        random_state=ICA_RANDOM_STATE,
        max_iter=ICA_MAX_ITER,
        method="fastica"
    )

    ica.fit(raw, picks=eeg_picks, verbose="ERROR")

    eog_channels = [
        ch for ch in raw.ch_names
        if raw.get_channel_types(picks=[ch])[0] == "eog"
    ]

    eog_inds = []

    if eog_channels:
        try:
            eog_inds, _ = ica.find_bads_eog(
                raw,
                ch_name=eog_channels[0],
                verbose="ERROR"
            )
            ica.exclude = eog_inds
            raw = ica.apply(raw, verbose="ERROR")

        except Exception as e:
            warnings.warn(f"ICA EOG detection failed: {e}")
            eog_inds = []

    else:
        warnings.warn(
            "No EOG channel found. ICA was fitted but no components were removed."
        )

    return raw, len(eog_inds)


# ============================================================
# Epoching
# ============================================================

def create_resting_epochs(raw):
    """
    Create 2-second non-overlapping resting-state epochs.
    Reject epochs exceeding the EEG amplitude threshold.
    """

    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=EPOCH_LENGTH,
        overlap=0.0,
        preload=True,
        verbose="ERROR"
    )

    n_epochs_before = len(epochs)

    epochs.drop_bad(
        reject=dict(eeg=AMPLITUDE_REJECT),
        verbose="ERROR"
    )

    n_epochs_after = len(epochs)
    n_epochs_removed = n_epochs_before - n_epochs_after

    return epochs, n_epochs_before, n_epochs_after, n_epochs_removed


# ============================================================
# Saving
# ============================================================

def save_outputs(raw, epochs, out_subject_dir, session_label, subject_id):
    """
    Save cleaned raw and cleaned epochs.
    """
    out_subject_dir.mkdir(parents=True, exist_ok=True)

    raw_out = out_subject_dir / f"{session_label}_{subject_id}_cleaned_raw.fif"
    epochs_out = out_subject_dir / f"{session_label}_{subject_id}_cleaned_epochs-epo.fif"

    raw.save(raw_out, overwrite=True, verbose="ERROR")
    epochs.save(epochs_out, overwrite=True, verbose="ERROR")

    return raw_out, epochs_out


# ============================================================
# Main loop
# ============================================================

def main():
    eeg_files = find_eeg_files(RAW_DIR)

    if len(eeg_files) == 0:
        raise FileNotFoundError(f"No EEG files found in: {RAW_DIR}")

    print(f"Found {len(eeg_files)} EEG files.")

    logs = []

    for file_path in eeg_files:
        subject_id = get_subject_id(file_path)
        session_label = get_session_label(file_path)

        print("\n" + "=" * 60)
        print(f"Processing: {subject_id} | {session_label}")
        print(f"File: {file_path.name}")

        out_subject_dir = OUT_DIR / subject_id

        try:
            raw = load_raw_eeg(file_path)

            original_sfreq = raw.info["sfreq"]
            original_channels = len(raw.ch_names)
            duration_sec = raw.n_times / raw.info["sfreq"]

            raw, dropped_exg, eog_candidates = basic_cleaning(raw)

            if RUN_ICA:
                raw, n_ica_removed = run_ica_cleaning(raw)
            else:
                n_ica_removed = 0

            epochs, n_epochs_before, n_epochs_after, n_epochs_removed = create_resting_epochs(raw)

            raw_out, epochs_out = save_outputs(
                raw=raw,
                epochs=epochs,
                out_subject_dir=out_subject_dir,
                session_label=session_label,
                subject_id=subject_id
            )

            percent_epochs_kept = (
                round((n_epochs_after / n_epochs_before) * 100, 2)
                if n_epochs_before > 0 else 0
            )

            qc_flag = "good"

            if percent_epochs_kept < 40:
                qc_flag = "problematic"
            elif percent_epochs_kept < 60:
                qc_flag = "check"
            elif percent_epochs_kept < 80:
                qc_flag = "acceptable"

            logs.append({
                "subject": subject_id,
                "session": session_label,
                "input_file": str(file_path),
                "original_sfreq": original_sfreq,
                "final_sfreq": raw.info["sfreq"],
                "original_channels": original_channels,
                "final_eeg_channels": len(raw.ch_names),
                "dropped_exg_channels": ", ".join(dropped_exg),
                "detected_eog_channels": ", ".join(eog_candidates),
                "duration_sec": round(duration_sec, 2),
                "epochs_before_rejection": n_epochs_before,
                "epochs_after_rejection": n_epochs_after,
                "epochs_removed": n_epochs_removed,
                "percent_epochs_kept": percent_epochs_kept,
                "ica_enabled": RUN_ICA,
                "ica_components_removed": n_ica_removed,
                "qc_flag": qc_flag,
                "cleaned_raw_file": str(raw_out),
                "cleaned_epochs_file": str(epochs_out),
                "status": "success"
            })

            print(f"Saved cleaned raw: {raw_out.name}")
            print(f"Saved cleaned epochs: {epochs_out.name}")
            print(f"Epochs kept: {n_epochs_after}/{n_epochs_before} ({percent_epochs_kept}%)")
            print(f"QC flag: {qc_flag}")

        except Exception as e:
            print(f"Failed: {file_path.name}")
            print(f"Reason: {e}")

            logs.append({
                "subject": subject_id,
                "session": session_label,
                "input_file": str(file_path),
                "status": "failed",
                "error": str(e)
            })

    log_df = pd.DataFrame(logs)
    log_path = OUT_DIR / "resting_preprocessing_log.csv"
    log_df.to_csv(log_path, index=False)

    print("\n" + "=" * 60)
    print("Preprocessing complete.")
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()