from pathlib import Path
import re
import warnings

import mne
import pandas as pd
import numpy as np


# ============================================================
# Project paths
# ============================================================

RAW_DIR = Path(r"I:\EEG_Python_Project\data\raw\task")
OUT_DIR = Path(r"I:\EEG_Python_Project\data\processed\task")

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Preprocessing settings
# ============================================================

TASK_LABEL = "RDK"

TARGET_SFREQ = 256

# Task EEG filter settings
L_FREQ = 0.1
H_FREQ = 40.0
NOTCH_FREQS = [50, 100]

# Epoch settings
EPOCH_TMIN = -0.5
EPOCH_TMAX = 1.0
BASELINE = (-0.2, 0.0)

# Bad epoch rejection threshold
AMPLITUDE_REJECT = 120e-6  # 120 µV

# Keep ICA off unless EOG channels are properly identified
RUN_ICA = False
ICA_RANDOM_STATE = 42
ICA_MAX_ITER = "auto"

# Manual exclusion
EXCLUDE_SUBJECTS = ["S17"]

# Real RDK task event codes from your Status channel
NUMERIC_EVENT_ID = {
    "interval1_random": 10,
    "interval1_coherent": 18,
    "interval2_random": 30,
    "interval2_coherent": 38,
}


# ============================================================
# File handling
# ============================================================

def find_eeg_files(raw_dir):
    """
    Recursively find EEG files.
    Supports .bdf, .edf, .set, .fif
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
    Drop BioSemi EXG channels from scalp EEG analysis.
    """
    exg_chs = [ch for ch in raw.ch_names if ch.upper().startswith("EXG")]

    if exg_chs:
        raw.drop_channels(exg_chs)
        print(f"Dropped EXG channels: {exg_chs}")

    return raw, exg_chs


def mark_eog_channels_if_present(raw):
    """
    Mark EOG channels only if explicitly named.
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
# Event handling
# ============================================================

def find_rdk_events(raw):
    """
    Find RDK task events from the Status channel.

    Expected real event codes:
        10 = interval1_random
        18 = interval1_coherent
        30 = interval2_random
        38 = interval2_coherent
    """

    stim_candidates = [
        ch for ch in raw.ch_names
        if ch.upper() in ["STATUS", "STI 014", "STI014", "TRIGGER", "TRIG"]
        or "STATUS" in ch.upper()
        or "TRIG" in ch.upper()
        or "STI" in ch.upper()
    ]

    if len(stim_candidates) == 0:
        raise RuntimeError(
            "No stim/status channel found. Expected a channel like 'Status'."
        )

    stim_channel = stim_candidates[0]
    print(f"Using stim channel: {stim_channel}")

    events = mne.find_events(
        raw,
        stim_channel=stim_channel,
        shortest_event=1,
        verbose="ERROR"
    )

    if len(events) == 0:
        raise RuntimeError("No events found in the Status channel.")

    unique_codes, counts = np.unique(events[:, 2], return_counts=True)
    unique_codes = unique_codes.astype(int).tolist()
    counts = counts.astype(int).tolist()

    print("Detected event codes:")
    for code, count in zip(unique_codes, counts):
        print(f"  Code {code}: {count} events")

    available_codes = set(unique_codes)
    required_codes = set(NUMERIC_EVENT_ID.values())

    if not required_codes.issubset(available_codes):
        raise RuntimeError(
            f"RDK event mapping does not match detected codes.\n"
            f"Current mapping: {NUMERIC_EVENT_ID}\n"
            f"Detected event codes: {unique_codes}\n"
            f"Edit NUMERIC_EVENT_ID."
        )

    print(f"Using RDK event mapping: {NUMERIC_EVENT_ID}")

    return events, NUMERIC_EVENT_ID, "stim"


# ============================================================
# Basic preprocessing
# ============================================================

def preprocess_raw(raw):
    """
    RDK task preprocessing:
    - Clean channel names
    - Drop EXG channels
    - Mark EOG if present
    - Set montage
    - Filter EEG 0.1–40 Hz
    - Notch EEG 50/100 Hz
    - Resample to 256 Hz
    - Average reference
    """

    raw = raw.copy()

    raw = clean_channel_names(raw)
    raw, dropped_exg = drop_external_channels(raw)
    raw, eog_candidates = mark_eog_channels_if_present(raw)

    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing="ignore")

    eeg_chs = [
        ch for ch in raw.ch_names
        if raw.get_channel_types(picks=[ch])[0] == "eeg"
    ]

    if len(eeg_chs) == 0:
        raise RuntimeError("No EEG channels found.")

    raw.filter(
        l_freq=L_FREQ,
        h_freq=H_FREQ,
        picks="eeg",
        fir_design="firwin",
        verbose="ERROR"
    )

    nyquist = raw.info["sfreq"] / 2
    valid_notches = [freq for freq in NOTCH_FREQS if freq < nyquist]

    if valid_notches:
        raw.notch_filter(
            freqs=valid_notches,
            picks="eeg",
            fir_design="firwin",
            verbose="ERROR"
        )

    if not np.isclose(raw.info["sfreq"], TARGET_SFREQ):
        raw.resample(TARGET_SFREQ, verbose="ERROR")

    raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    return raw, dropped_exg, eog_candidates


# ============================================================
# Optional ICA
# ============================================================

def run_ica_cleaning(raw):
    """
    Optional ICA. Keep RUN_ICA=False unless EOG channels are defined.
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
            "No EOG channel found. ICA fitted but no components were removed."
        )

    return raw, len(eog_inds)


# ============================================================
# Epoching
# ============================================================

def create_task_epochs(raw, events, event_id):
    """
    Create RDK task epochs around all four event types.
    """
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        baseline=BASELINE,
        preload=True,
        reject=dict(eeg=AMPLITUDE_REJECT),
        picks="eeg",
        verbose="ERROR"
    )

    return epochs


def get_condition_counts(epochs):
    """
    Count epochs for each RDK event condition.
    """
    counts = {}

    for condition in epochs.event_id.keys():
        counts[condition] = len(epochs[condition])

    return counts


def get_detected_event_counts(events):
    """
    Count raw detected event codes before epoch rejection.
    """
    codes, counts = np.unique(events[:, 2], return_counts=True)
    return {int(code): int(count) for code, count in zip(codes, counts)}


# ============================================================
# Saving
# ============================================================

def save_outputs(raw, epochs, subject_id):
    """
    Save cleaned raw and task epochs.
    """
    out_subject_dir = OUT_DIR / subject_id
    out_subject_dir.mkdir(parents=True, exist_ok=True)

    raw_out = out_subject_dir / f"{TASK_LABEL}_{subject_id}_cleaned_raw.fif"
    epochs_out = out_subject_dir / f"{TASK_LABEL}_{subject_id}_cleaned_epochs-epo.fif"

    raw.save(raw_out, overwrite=True, verbose="ERROR")
    epochs.save(epochs_out, overwrite=True, verbose="ERROR")

    return raw_out, epochs_out


# ============================================================
# Main loop
# ============================================================

def main():
    eeg_files = find_eeg_files(RAW_DIR)

    eeg_files = [
        f for f in eeg_files
        if get_subject_id(f) not in EXCLUDE_SUBJECTS
    ]

    if len(eeg_files) == 0:
        raise FileNotFoundError(f"No EEG files found in: {RAW_DIR}")

    print(f"Found {len(eeg_files)} task EEG files.")
    print(f"Excluding subjects: {EXCLUDE_SUBJECTS}")

    logs = []

    for file_path in eeg_files:
        subject_id = get_subject_id(file_path)

        print("\n" + "=" * 60)
        print(f"Processing RDK task: {subject_id}")
        print(f"File: {file_path.name}")

        try:
            raw = load_raw_eeg(file_path)

            original_sfreq = raw.info["sfreq"]
            original_channels = len(raw.ch_names)
            duration_sec = raw.n_times / raw.info["sfreq"]

            raw, dropped_exg, eog_candidates = preprocess_raw(raw)

            events, event_id, event_source = find_rdk_events(raw)

            if RUN_ICA:
                raw, n_ica_removed = run_ica_cleaning(raw)
            else:
                n_ica_removed = 0

            n_events_before = len(events)
            detected_event_counts = get_detected_event_counts(events)

            epochs = create_task_epochs(
                raw=raw,
                events=events,
                event_id=event_id
            )

            n_epochs_after = len(epochs)
            condition_counts = get_condition_counts(epochs)

            raw_out, epochs_out = save_outputs(
                raw=raw,
                epochs=epochs,
                subject_id=subject_id
            )

            percent_epochs_kept = (
                round((n_epochs_after / n_events_before) * 100, 2)
                if n_events_before > 0 else 0
            )

            qc_flag = "good"

            if percent_epochs_kept < 40:
                qc_flag = "problematic"
            elif percent_epochs_kept < 60:
                qc_flag = "check"
            elif percent_epochs_kept < 80:
                qc_flag = "acceptable"

            logs.append({
                "task": TASK_LABEL,
                "subject": subject_id,
                "input_file": str(file_path),
                "original_sfreq": original_sfreq,
                "final_sfreq": raw.info["sfreq"],
                "original_channels": original_channels,
                "final_eeg_channels": len(epochs.ch_names),
                "dropped_exg_channels": ", ".join(dropped_exg),
                "detected_eog_channels": ", ".join(eog_candidates),
                "duration_sec": round(duration_sec, 2),
                "event_source": event_source,

                "event_id_interval1_random": event_id.get("interval1_random", np.nan),
                "event_id_interval1_coherent": event_id.get("interval1_coherent", np.nan),
                "event_id_interval2_random": event_id.get("interval2_random", np.nan),
                "event_id_interval2_coherent": event_id.get("interval2_coherent", np.nan),

                "detected_code_10_count": detected_event_counts.get(10, 0),
                "detected_code_18_count": detected_event_counts.get(18, 0),
                "detected_code_30_count": detected_event_counts.get(30, 0),
                "detected_code_38_count": detected_event_counts.get(38, 0),

                "events_before_epoching": n_events_before,
                "epochs_after_rejection": n_epochs_after,

                "epochs_interval1_random": condition_counts.get("interval1_random", 0),
                "epochs_interval1_coherent": condition_counts.get("interval1_coherent", 0),
                "epochs_interval2_random": condition_counts.get("interval2_random", 0),
                "epochs_interval2_coherent": condition_counts.get("interval2_coherent", 0),

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
            print(f"Events found: {n_events_before}")
            print(f"Epochs kept: {n_epochs_after}/{n_events_before} ({percent_epochs_kept}%)")

            for condition, count in condition_counts.items():
                print(f"{condition} epochs: {count}")

            print(f"QC flag: {qc_flag}")

        except Exception as e:
            print(f"Failed: {file_path.name}")
            print(f"Reason: {e}")

            logs.append({
                "task": TASK_LABEL,
                "subject": subject_id,
                "input_file": str(file_path),
                "status": "failed",
                "error": str(e)
            })

    log_df = pd.DataFrame(logs)
    log_path = OUT_DIR / "rdk_task_preprocessing_log.csv"
    log_df.to_csv(log_path, index=False)

    print("\n" + "=" * 60)
    print("RDK task preprocessing complete.")
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()