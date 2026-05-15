from pathlib import Path
import re

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path(r"I:\EEG_Python_Project")

PROCESSED_DIR = PROJECT_DIR / "data" / "processed" / "task"
OUT_DIR = PROJECT_DIR / "results" / "task_features"

TABLES_DIR = OUT_DIR / "tables"
FIGURES_DIR = OUT_DIR / "figures"
PSD_DIR = FIGURES_DIR / "psd_by_subject"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
PSD_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Subject inclusion / exclusion
# ============================================================

INCLUDE_SUBJECTS = ["S10", "S11", "S12", "S13", "S15", "S18"]
EXCLUDE_SUBJECTS = ["S14", "S17", "S19", "S20"]


# ============================================================
# RDK condition labels
# ============================================================

RDK_CONDITIONS = [
    "interval1_random",
    "interval1_coherent",
    "interval2_random",
    "interval2_coherent"
]

CONDITION_GROUPS = {
    "interval1_random": {
        "interval": "interval1",
        "motion_type": "random"
    },
    "interval1_coherent": {
        "interval": "interval1",
        "motion_type": "coherent"
    },
    "interval2_random": {
        "interval": "interval2",
        "motion_type": "random"
    },
    "interval2_coherent": {
        "interval": "interval2",
        "motion_type": "coherent"
    }
}


# ============================================================
# Feature settings
# ============================================================

# Use post-stimulus/task response window
FEATURE_TMIN = 0.0
FEATURE_TMAX = 1.0

PSD_METHOD = "multitaper"
PSD_FMIN = 1.0
PSD_FMAX = 40.0
BANDWIDTH = 2.0

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}

TOTAL_POWER_RANGE = (1.0, 40.0)

ROIS = {
    "posterior": ["O1", "O2", "Oz", "PO3", "PO4", "PO7", "PO8"],
    "occipital": ["O1", "O2", "Oz"],
    "global": None
}

PSD_FLOOR = 1e-20

mne.set_log_level("WARNING")


# ============================================================
# Helper functions
# ============================================================

def normalize_label(label):
    """
    Normalize EEG channel labels.
    Example: PO-7 -> PO7
    """
    return re.sub(r"[^A-Z0-9]", "", str(label).upper())


def find_epoch_file(subject):
    """
    Find cleaned RDK epochs file for one subject.
    Expected:
    data/processed/task/S10/RDK_S10_cleaned_epochs-epo.fif
    """
    subject_dir = PROCESSED_DIR / subject

    possible_patterns = [
        f"RDK_{subject}_cleaned_epochs-epo.fif",
        f"*{subject}*_cleaned_epochs-epo.fif",
        "*cleaned_epochs-epo.fif"
    ]

    files = []

    if subject_dir.exists():
        for pattern in possible_patterns:
            files.extend(sorted(subject_dir.glob(pattern)))

    if len(files) == 0:
        files = sorted(PROCESSED_DIR.rglob(f"*{subject}*_cleaned_epochs-epo.fif"))

    files = list(dict.fromkeys(files))

    if len(files) == 0:
        raise FileNotFoundError(f"No cleaned task epochs file found for {subject}")

    if len(files) > 1:
        print(f"[WARNING] Multiple files found for {subject}. Using: {files[0]}")

    return files[0]


def get_roi_indices(ch_names, roi_channels):
    """
    Return channel indices for ROI.
    If roi_channels is None, return all channels.
    """
    if roi_channels is None:
        return list(range(len(ch_names)))

    norm_ch = [normalize_label(ch) for ch in ch_names]
    norm_roi = [normalize_label(ch) for ch in roi_channels]

    roi_idx = [i for i, ch in enumerate(norm_ch) if ch in norm_roi]

    return roi_idx


def compute_psd_for_epochs(epochs):
    """
    Compute PSD from epochs within the feature time window.
    Returns:
        freqs
        psd_mean: n_channels x n_freqs
        ch_names
    """
    epochs_crop = epochs.copy().crop(tmin=FEATURE_TMIN, tmax=FEATURE_TMAX)

    spectrum = epochs_crop.compute_psd(
        method=PSD_METHOD,
        fmin=PSD_FMIN,
        fmax=PSD_FMAX,
        bandwidth=BANDWIDTH,
        verbose=False
    )

    psds = spectrum.get_data()  # n_epochs x n_channels x n_freqs
    psd_mean = psds.mean(axis=0)
    psd_mean = np.maximum(psd_mean, PSD_FLOOR)

    freqs = spectrum.freqs
    ch_names = spectrum.ch_names

    return freqs, psd_mean, ch_names


def bandpower(freqs, psd, fmin, fmax):
    """
    Compute band power using trapezoidal integration.
    Compatible with newer NumPy versions.
    """
    mask = (freqs >= fmin) & (freqs < fmax)

    if not np.any(mask):
        return np.nan

    return np.trapezoid(psd[..., mask], freqs[mask], axis=-1)


def extract_condition_roi_features(subject, condition, epochs_condition, roi_name, roi_idx):
    """
    Extract bandpower features for one subject, one condition, one ROI.
    """
    freqs, psd_mean, ch_names = compute_psd_for_epochs(epochs_condition)

    if len(roi_idx) == 0:
        return [], None

    roi_psd = psd_mean[roi_idx, :]
    roi_channels = [ch_names[i] for i in roi_idx]

    total_power_per_channel = bandpower(
        freqs,
        roi_psd,
        TOTAL_POWER_RANGE[0],
        TOTAL_POWER_RANGE[1]
    )

    channel_rows = []

    for ch_name, ch_psd, total_power in zip(roi_channels, roi_psd, total_power_per_channel):
        row = {
            "subject": subject,
            "condition": condition,
            "interval": CONDITION_GROUPS[condition]["interval"],
            "motion_type": CONDITION_GROUPS[condition]["motion_type"],
            "roi": roi_name,
            "channel": ch_name,
            "n_epochs": len(epochs_condition),
            "feature_tmin": FEATURE_TMIN,
            "feature_tmax": FEATURE_TMAX,
            "total_power_1_40": total_power,
        }

        for band_name, (fmin, fmax) in BANDS.items():
            bp = bandpower(freqs, ch_psd, fmin, fmax)
            row[f"{band_name}_power"] = bp
            row[f"{band_name}_relative_power"] = bp / total_power if total_power > 0 else np.nan

        channel_rows.append(row)

    df_ch = pd.DataFrame(channel_rows)

    roi_summary = {
        "subject": subject,
        "condition": condition,
        "interval": CONDITION_GROUPS[condition]["interval"],
        "motion_type": CONDITION_GROUPS[condition]["motion_type"],
        "roi": roi_name,
        "n_epochs": len(epochs_condition),
        "n_channels": len(roi_channels),
        "feature_tmin": FEATURE_TMIN,
        "feature_tmax": FEATURE_TMAX,
        "total_power_1_40_mean": df_ch["total_power_1_40"].mean(),
        "total_power_1_40_median": df_ch["total_power_1_40"].median(),
    }

    for band_name in BANDS.keys():
        roi_summary[f"{band_name}_power_mean"] = df_ch[f"{band_name}_power"].mean()
        roi_summary[f"{band_name}_power_median"] = df_ch[f"{band_name}_power"].median()
        roi_summary[f"{band_name}_relative_power_mean"] = df_ch[f"{band_name}_relative_power"].mean()
        roi_summary[f"{band_name}_relative_power_median"] = df_ch[f"{band_name}_relative_power"].median()

    return channel_rows, roi_summary


def plot_condition_psd(subject, condition, epochs_condition):
    """
    Save global PSD plot for subject-condition.
    """
    freqs, psd_mean, _ = compute_psd_for_epochs(epochs_condition)
    global_psd = psd_mean.mean(axis=0)

    plt.figure(figsize=(7, 4))
    plt.plot(freqs, global_psd)
    plt.title(f"RDK PSD - {subject} - {condition}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.xlim(1, 40)
    plt.tight_layout()

    out_path = PSD_DIR / f"{subject}_{condition}_global_psd.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


def process_subject(subject):
    """
    Process one subject:
    - load cleaned RDK epochs
    - extract condition-wise bandpower features
    """
    print("\n" + "=" * 60)
    print(f"Processing subject: {subject}")

    epoch_file = find_epoch_file(subject)
    print(f"Epoch file: {epoch_file.name}")

    epochs = mne.read_epochs(epoch_file, preload=True, verbose="ERROR")

    print(f"Total epochs: {len(epochs)}")
    print(f"Channels: {len(epochs.ch_names)}")
    print(f"Available event IDs: {epochs.event_id}")

    all_channel_rows = []
    all_roi_rows = []

    ch_names = epochs.ch_names

    roi_indices = {
        roi_name: get_roi_indices(ch_names, roi_channels)
        for roi_name, roi_channels in ROIS.items()
    }

    for roi_name, roi_idx in roi_indices.items():
        if len(roi_idx) == 0:
            print(f"[WARNING] No channels found for ROI: {roi_name}")
        else:
            roi_chs = [ch_names[i] for i in roi_idx]
            print(f"ROI {roi_name}: {roi_chs}")

    for condition in RDK_CONDITIONS:
        if condition not in epochs.event_id:
            print(f"[WARNING] Condition not found for {subject}: {condition}")
            continue

        epochs_condition = epochs[condition]
        print(f"{condition}: {len(epochs_condition)} epochs")

        if len(epochs_condition) == 0:
            continue

        plot_condition_psd(subject, condition, epochs_condition)

        for roi_name, roi_idx in roi_indices.items():
            channel_rows, roi_summary = extract_condition_roi_features(
                subject=subject,
                condition=condition,
                epochs_condition=epochs_condition,
                roi_name=roi_name,
                roi_idx=roi_idx
            )

            all_channel_rows.extend(channel_rows)

            if roi_summary is not None:
                roi_summary["epoch_file"] = str(epoch_file)
                all_roi_rows.append(roi_summary)

    return all_channel_rows, all_roi_rows


def create_collapsed_features(df_roi):
    """
    Create collapsed features:
    - coherent vs random
    - interval1 vs interval2
    """
    collapsed_rows = []

    feature_cols = [
        col for col in df_roi.columns
        if col.endswith("_mean") or col.endswith("_median")
    ]

    groupings = [
        ("motion_type", "coherent_random"),
        ("interval", "interval1_interval2")
    ]

    for (subject, roi), df_sub in df_roi.groupby(["subject", "roi"]):
        for group_col, contrast_family in groupings:
            for group_value, df_group in df_sub.groupby(group_col):
                row = {
                    "subject": subject,
                    "roi": roi,
                    "contrast_family": contrast_family,
                    "group_variable": group_col,
                    "group_value": group_value,
                    "n_conditions_collapsed": df_group["condition"].nunique(),
                    "n_epochs_total": df_group["n_epochs"].sum()
                }

                for col in feature_cols:
                    row[col] = df_group[col].mean()

                collapsed_rows.append(row)

    return pd.DataFrame(collapsed_rows)


def create_difference_features(df_collapsed):
    """
    Create simple difference scores:
    - coherent minus random
    - interval2 minus interval1
    """
    difference_rows = []

    feature_cols = [
        col for col in df_collapsed.columns
        if col.endswith("_mean") or col.endswith("_median")
    ]

    for (subject, roi, contrast_family), df_sub in df_collapsed.groupby(
        ["subject", "roi", "contrast_family"]
    ):
        values = df_sub["group_value"].tolist()

        if contrast_family == "coherent_random":
            if "coherent" in values and "random" in values:
                coherent = df_sub[df_sub["group_value"] == "coherent"].iloc[0]
                random = df_sub[df_sub["group_value"] == "random"].iloc[0]

                row = {
                    "subject": subject,
                    "roi": roi,
                    "contrast": "coherent_minus_random"
                }

                for col in feature_cols:
                    row[col] = coherent[col] - random[col]

                difference_rows.append(row)

        elif contrast_family == "interval1_interval2":
            if "interval1" in values and "interval2" in values:
                interval2 = df_sub[df_sub["group_value"] == "interval2"].iloc[0]
                interval1 = df_sub[df_sub["group_value"] == "interval1"].iloc[0]

                row = {
                    "subject": subject,
                    "roi": roi,
                    "contrast": "interval2_minus_interval1"
                }

                for col in feature_cols:
                    row[col] = interval2[col] - interval1[col]

                difference_rows.append(row)

    return pd.DataFrame(difference_rows)


# ============================================================
# Main
# ============================================================

def main():
    print("RDK task feature extraction")
    print(f"Included subjects: {INCLUDE_SUBJECTS}")
    print(f"Excluded subjects: {EXCLUDE_SUBJECTS}")

    all_channel_rows = []
    all_roi_rows = []

    for subject in INCLUDE_SUBJECTS:
        try:
            channel_rows, roi_rows = process_subject(subject)
            all_channel_rows.extend(channel_rows)
            all_roi_rows.extend(roi_rows)

        except Exception as e:
            print(f"[ERROR] Failed subject {subject}: {type(e).__name__}: {e}")

    df_channel = pd.DataFrame(all_channel_rows)
    df_roi = pd.DataFrame(all_roi_rows)

    if df_channel.empty or df_roi.empty:
        raise RuntimeError("No task features were extracted. Check input files and event labels.")

    df_collapsed = create_collapsed_features(df_roi)
    df_differences = create_difference_features(df_collapsed)

    channel_out = TABLES_DIR / "rdk_task_features_per_channel.csv"
    roi_out = TABLES_DIR / "rdk_task_features_ROI_summary.csv"
    collapsed_out = TABLES_DIR / "rdk_task_features_collapsed.csv"
    differences_out = TABLES_DIR / "rdk_task_features_differences.csv"

    df_channel.to_csv(channel_out, index=False)
    df_roi.to_csv(roi_out, index=False)
    df_collapsed.to_csv(collapsed_out, index=False)
    df_differences.to_csv(differences_out, index=False)

    print("\n" + "=" * 60)
    print("RDK task feature extraction complete.")
    print(f"Per-channel features saved to: {channel_out}")
    print(f"ROI summary features saved to: {roi_out}")
    print(f"Collapsed features saved to: {collapsed_out}")
    print(f"Difference features saved to: {differences_out}")

    print("\nROI summary preview:")
    preview_cols = [
        "subject",
        "condition",
        "roi",
        "n_epochs",
        "theta_power_mean",
        "alpha_power_mean",
        "beta_power_mean",
        "gamma_power_mean"
    ]

    available_cols = [col for col in preview_cols if col in df_roi.columns]
    print(df_roi[available_cols].head(30).to_string(index=False))

    print("\nDifference preview:")
    if not df_differences.empty:
        diff_preview_cols = [
            "subject",
            "roi",
            "contrast",
            "theta_power_mean",
            "alpha_power_mean",
            "beta_power_mean"
        ]
        available_diff_cols = [col for col in diff_preview_cols if col in df_differences.columns]
        print(df_differences[available_diff_cols].head(30).to_string(index=False))

    print("\nPSD figures saved in:")
    print(PSD_DIR)


if __name__ == "__main__":
    main()