from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne


# ============================================================
# Try importing specparam
# ============================================================

try:
    from specparam import SpectralModel
except ImportError:
    raise ImportError(
        "specparam is not installed. Install it with:\n"
        "pip install specparam"
    )


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path(r"I:\EEG_Python_Project")

PROCESSED_DIR = PROJECT_DIR / "data" / "processed" / "resting_state"
QC_TABLE = PROJECT_DIR / "results" / "resting_state_qc" / "tables" / "included_subjects.csv"

OUT_DIR = PROJECT_DIR / "results" / "resting_state_features"
TABLES_DIR = OUT_DIR / "tables"
FIGURES_DIR = OUT_DIR / "figures"
PSD_DIR = FIGURES_DIR / "psd"
SPECPARAM_FIT_DIR = FIGURES_DIR / "specparam_fits"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
PSD_DIR.mkdir(parents=True, exist_ok=True)
SPECPARAM_FIT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Settings
# ============================================================

EXCLUDE_SUBJECTS = ["S17"]

SESSION_LABEL = "REST"

# PSD settings
PSD_METHOD = "multitaper"
PSD_FMIN = 1.0
PSD_FMAX = 40.0
BANDWIDTH = 2.0

# Specparam settings
FREQ_RANGE = [2.0, 40.0]
APERIODIC_MODE = "fixed"
PEAK_WIDTH_LIMITS = (1.0, 12.0)
MAX_N_PEAKS = 4
MIN_PEAK_HEIGHT = 0.001
PEAK_THRESHOLD = 0.05

PSD_FLOOR = 1e-20

# ROI channels
POSTERIOR_ROI = ["O1", "O2", "Oz", "PO3", "PO4", "PO7", "PO8"]

ROIS = {
    "posterior": POSTERIOR_ROI,
    "global": None
}

# Frequency bands
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}

TOTAL_POWER_RANGE = (1.0, 40.0)
ALPHA_BAND = (8.0, 13.0)

# Specparam quality thresholds
R2_EXCLUDE_THRESHOLD = 0.85
R2_FLAG_THRESHOLD = 0.90
R2_MIN_THRESHOLD = 0.80

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")


# ============================================================
# Helper functions
# ============================================================

def normalize_label(label):
    """
    Normalize EEG channel labels.
    Example: 'PO-7' -> 'PO7'
    """
    return re.sub(r"[^A-Z0-9]", "", str(label).upper())


def get_subject_id_from_path(path):
    """
    Extract subject ID such as S10 from filename or folder.
    """
    path = Path(path)

    for part in reversed(path.parts):
        if re.match(r"^S\d+$", part, re.IGNORECASE):
            return part.upper()

    match = re.search(r"S\d+", path.stem, re.IGNORECASE)

    if match:
        return match.group(0).upper()

    return path.stem


def load_included_subjects():
    """
    Load included subjects from QC table if available.
    Otherwise, find all cleaned epoch files and exclude S17 manually.
    """
    if QC_TABLE.exists():
        df = pd.read_csv(QC_TABLE)

        if "subject" not in df.columns:
            raise ValueError(f"'subject' column not found in {QC_TABLE}")

        subjects = sorted(df["subject"].dropna().astype(str).unique().tolist())
        subjects = [s for s in subjects if s not in EXCLUDE_SUBJECTS]

        print(f"Loaded included subjects from QC table: {QC_TABLE}")
        print(f"Included subjects: {subjects}")

        return subjects

    print("QC included_subjects.csv not found. Falling back to cleaned epoch files.")

    epoch_files = sorted(PROCESSED_DIR.rglob("*_cleaned_epochs-epo.fif"))
    subjects = sorted(set(get_subject_id_from_path(f) for f in epoch_files))
    subjects = [s for s in subjects if s not in EXCLUDE_SUBJECTS]

    print(f"Included subjects: {subjects}")

    return subjects


def find_epoch_file(subject):
    """
    Find cleaned epochs file for one subject.
    """
    pattern = f"*{subject}*_cleaned_epochs-epo.fif"
    files = sorted((PROCESSED_DIR / subject).glob(pattern))

    if len(files) == 0:
        files = sorted(PROCESSED_DIR.rglob(pattern))

    if len(files) == 0:
        raise FileNotFoundError(f"No cleaned epochs file found for {subject}")

    if len(files) > 1:
        print(f"[WARNING] Multiple files found for {subject}. Using first: {files[0]}")

    return files[0]


def compute_psd_epochs(epochs):
    """
    Compute average PSD across epochs using MNE multitaper.
    Returns:
        freqs: shape n_freqs
        psd_mean: shape n_channels x n_freqs
        ch_names: channel names
    """
    spectrum = epochs.compute_psd(
        method=PSD_METHOD,
        fmin=PSD_FMIN,
        fmax=PSD_FMAX,
        bandwidth=BANDWIDTH,
        verbose=False
    )

    psds = spectrum.get_data()  # n_epochs x n_channels x n_freqs
    freqs = spectrum.freqs
    psd_mean = psds.mean(axis=0)  # n_channels x n_freqs
    ch_names = spectrum.ch_names

    psd_mean = np.maximum(psd_mean, PSD_FLOOR)

    return freqs, psd_mean, ch_names


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


def bandpower(freqs, psd, fmin, fmax):
    """
    Compute absolute band power using trapezoidal integration.
    Compatible with newer NumPy versions.
    psd shape: n_channels x n_freqs or n_freqs
    """
    mask = (freqs >= fmin) & (freqs < fmax)

    if not np.any(mask):
        return np.nan

    return np.trapezoid(psd[..., mask], freqs[mask], axis=-1)


def compute_bandpower_features(freqs, psd_mean, ch_names, subject, roi_name, roi_idx):
    """
    Compute absolute and relative bandpower for one ROI.
    """
    rows = []

    if len(roi_idx) == 0:
        print(f"[WARNING] No channels found for ROI: {roi_name}, subject: {subject}")
        return rows

    roi_psd = psd_mean[roi_idx, :]
    roi_ch_names = [ch_names[i] for i in roi_idx]

    total_power_per_channel = bandpower(
        freqs,
        roi_psd,
        TOTAL_POWER_RANGE[0],
        TOTAL_POWER_RANGE[1]
    )

    for ch_name, ch_psd, total_power in zip(roi_ch_names, roi_psd, total_power_per_channel):
        row = {
            "subject": subject,
            "session": SESSION_LABEL,
            "roi": roi_name,
            "channel": ch_name,
            "total_power_1_40": total_power
        }

        for band_name, (fmin, fmax) in BANDS.items():
            bp = bandpower(freqs, ch_psd, fmin, fmax)
            row[f"{band_name}_power"] = bp

            if total_power > 0:
                row[f"{band_name}_relative_power"] = bp / total_power
            else:
                row[f"{band_name}_relative_power"] = np.nan

        rows.append(row)

    return rows


def get_alpha_peak(peak_params):
    """
    Return alpha peak CF, PW, BW from specparam peak parameters.
    """
    if peak_params is None:
        return np.nan, np.nan, np.nan

    peaks = np.asarray(peak_params)

    if peaks.size == 0:
        return np.nan, np.nan, np.nan

    peaks = np.atleast_2d(peaks)

    if np.all(np.isnan(peaks)):
        return np.nan, np.nan, np.nan

    alpha_mask = (peaks[:, 0] >= ALPHA_BAND[0]) & (peaks[:, 0] <= ALPHA_BAND[1])

    if not np.any(alpha_mask):
        return np.nan, np.nan, np.nan

    alpha_peaks = peaks[alpha_mask]
    best_peak = alpha_peaks[np.argmax(alpha_peaks[:, 1])]

    alpha_cf = float(best_peak[0])
    alpha_pw = float(best_peak[1])
    alpha_bw = float(best_peak[2])

    return alpha_cf, alpha_pw, alpha_bw


def fit_specparam_channel(freqs, psd_channel):
    """
    Fit specparam model to one channel.
    """
    try:
        fm = SpectralModel(
            peak_width_limits=PEAK_WIDTH_LIMITS,
            max_n_peaks=MAX_N_PEAKS,
            min_peak_height=MIN_PEAK_HEIGHT,
            peak_threshold=PEAK_THRESHOLD,
            aperiodic_mode=APERIODIC_MODE,
            verbose=False
        )

        fm.fit(freqs, psd_channel, FREQ_RANGE)

        ap = fm.get_params("aperiodic")
        peaks = fm.get_params("peak")

        offset = float(ap[0])
        exponent = float(ap[1])

        r2 = float(fm.get_metrics("gof"))
        error = float(fm.get_metrics("error"))

        if peaks is None or len(peaks) == 0:
            n_peaks = 0
        else:
            peaks_arr = np.atleast_2d(peaks)
            if np.all(np.isnan(peaks_arr)):
                n_peaks = 0
            else:
                n_peaks = peaks_arr.shape[0]

        alpha_cf, alpha_pw, alpha_bw = get_alpha_peak(peaks)

        result = {
            "offset": offset,
            "exponent": exponent,
            "r2": r2,
            "error": error,
            "n_peaks": n_peaks,
            "alpha_cf": alpha_cf,
            "alpha_pw": alpha_pw,
            "alpha_bw": alpha_bw,
            "fit_success": True,
            "fit_error": "",
            "_model": fm
        }

    except Exception as e:
        result = {
            "offset": np.nan,
            "exponent": np.nan,
            "r2": np.nan,
            "error": np.nan,
            "n_peaks": 0,
            "alpha_cf": np.nan,
            "alpha_pw": np.nan,
            "alpha_bw": np.nan,
            "fit_success": False,
            "fit_error": f"{type(e).__name__}: {e}",
            "_model": None
        }

    return result


def quality_flag(r2_mean, r2_min):
    """
    Quality decision for specparam fit.
    """
    if np.isnan(r2_mean):
        return "EXCLUDE", True

    if r2_mean < R2_EXCLUDE_THRESHOLD or r2_min < R2_MIN_THRESHOLD:
        return "EXCLUDE", True

    if r2_mean < R2_FLAG_THRESHOLD:
        return "FLAG", False

    return "OK", False


def plot_subject_psd(subject, freqs, psd_mean, ch_names):
    """
    Plot average global PSD for one subject.
    """
    global_psd = psd_mean.mean(axis=0)

    plt.figure(figsize=(7, 4))
    plt.plot(freqs, global_psd)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.title(f"Global PSD - {subject}")
    plt.xlim(1, 40)
    plt.tight_layout()

    out_path = PSD_DIR / f"{subject}_global_psd.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_specparam_roi_fits(subject, roi_name, freqs, psd_mean, ch_names, roi_idx, fit_results):
    """
    Save specparam fit plots for ROI channels.
    """
    if len(roi_idx) == 0:
        return

    n_channels = len(roi_idx)
    ncols = min(4, n_channels)
    nrows = int(np.ceil(n_channels / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 3 * nrows),
        squeeze=False
    )

    for i, ch_idx in enumerate(roi_idx):
        ax = axes[i // ncols, i % ncols]
        ch_name = ch_names[ch_idx]
        model = fit_results[ch_idx].get("_model")

        if model is None:
            ax.text(
                0.5,
                0.5,
                f"{ch_name}\nFIT FAILED",
                ha="center",
                va="center",
                transform=ax.transAxes
            )
            ax.set_axis_off()
            continue

        try:
            model.plot(ax=ax)
        except Exception:
            ax.plot(freqs, psd_mean[ch_idx, :])
            ax.set_title(f"{ch_name} - fallback PSD")

        r2 = fit_results[ch_idx]["r2"]
        exponent = fit_results[ch_idx]["exponent"]

        r2_text = f"{r2:.3f}" if not np.isnan(r2) else "NaN"
        exp_text = f"{exponent:.2f}" if not np.isnan(exponent) else "NaN"

        ax.set_title(f"{ch_name} | R²={r2_text} | Exp={exp_text}", fontsize=9)

    for j in range(n_channels, nrows * ncols):
        axes[j // ncols, j % ncols].set_axis_off()

    fig.suptitle(f"{subject} - {roi_name} specparam fits", fontsize=12)
    fig.tight_layout()

    out_path = SPECPARAM_FIT_DIR / f"{subject}_{roi_name}_specparam_fits.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def aggregate_roi_features(df_channel, subject, roi_name):
    """
    Aggregate channel-level features into one subject-level ROI row.
    """
    df_roi = df_channel[
        (df_channel["subject"] == subject) &
        (df_channel["roi"] == roi_name)
    ].copy()

    if df_roi.empty:
        return None

    row = {
        "subject": subject,
        "session": SESSION_LABEL,
        "roi": roi_name,
        "n_channels": df_roi["channel"].nunique()
    }

    # Bandpower features
    for band in BANDS.keys():
        row[f"{band}_power_mean"] = df_roi[f"{band}_power"].mean()
        row[f"{band}_power_median"] = df_roi[f"{band}_power"].median()
        row[f"{band}_relative_power_mean"] = df_roi[f"{band}_relative_power"].mean()
        row[f"{band}_relative_power_median"] = df_roi[f"{band}_relative_power"].median()

    row["total_power_1_40_mean"] = df_roi["total_power_1_40"].mean()
    row["total_power_1_40_median"] = df_roi["total_power_1_40"].median()

    # Specparam features
    spec_cols = [
        "offset",
        "exponent",
        "r2",
        "error",
        "n_peaks",
        "alpha_cf",
        "alpha_pw",
        "alpha_bw"
    ]

    for col in spec_cols:
        if col in df_roi.columns:
            row[f"{col}_mean"] = df_roi[col].mean()
            row[f"{col}_median"] = df_roi[col].median()

    r2_mean = row.get("r2_mean", np.nan)
    r2_min = df_roi["r2"].min() if "r2" in df_roi.columns else np.nan

    quality, exclude = quality_flag(r2_mean, r2_min)

    row["r2_min"] = r2_min
    row["specparam_quality"] = quality
    row["specparam_exclude"] = exclude

    return row


# ============================================================
# Main feature extraction
# ============================================================

def process_subject(subject):
    """
    Process one subject:
    - load cleaned epochs
    - compute PSD
    - extract bandpower
    - fit specparam
    - save plots
    """
    print("\n" + "=" * 60)
    print(f"Processing subject: {subject}")

    epoch_file = find_epoch_file(subject)
    print(f"Epoch file: {epoch_file.name}")

    epochs = mne.read_epochs(epoch_file, preload=True, verbose="ERROR")

    n_epochs = len(epochs)
    n_channels = len(epochs.ch_names)

    print(f"Epochs: {n_epochs}")
    print(f"Channels: {n_channels}")

    freqs, psd_mean, ch_names = compute_psd_epochs(epochs)

    plot_subject_psd(subject, freqs, psd_mean, ch_names)

    all_channel_rows = []
    all_roi_summary_rows = []

    # First fit specparam on all channels once
    fit_results = []

    print("Fitting specparam channel-by-channel...")

    for ch_idx, ch_name in enumerate(ch_names):
        result = fit_specparam_channel(freqs, psd_mean[ch_idx, :])
        fit_results.append(result)

    # Extract features for each ROI
    for roi_name, roi_channels in ROIS.items():
        roi_idx = get_roi_indices(ch_names, roi_channels)

        if len(roi_idx) == 0:
            print(f"[WARNING] No channels found for ROI: {roi_name}")
            continue

        print(f"ROI: {roi_name} | Channels: {[ch_names[i] for i in roi_idx]}")

        band_rows = compute_bandpower_features(
            freqs=freqs,
            psd_mean=psd_mean,
            ch_names=ch_names,
            subject=subject,
            roi_name=roi_name,
            roi_idx=roi_idx
        )

        for row in band_rows:
            ch_name = row["channel"]
            ch_idx = ch_names.index(ch_name)
            spec = fit_results[ch_idx]

            row.update({
                "n_epochs": n_epochs,
                "n_total_channels": n_channels,
                "offset": spec["offset"],
                "exponent": spec["exponent"],
                "r2": spec["r2"],
                "error": spec["error"],
                "n_peaks": spec["n_peaks"],
                "alpha_cf": spec["alpha_cf"],
                "alpha_pw": spec["alpha_pw"],
                "alpha_bw": spec["alpha_bw"],
                "fit_success": spec["fit_success"],
                "fit_error": spec["fit_error"]
            })

            all_channel_rows.append(row)

        df_subject_channel = pd.DataFrame(all_channel_rows)

        roi_summary = aggregate_roi_features(
            df_channel=df_subject_channel,
            subject=subject,
            roi_name=roi_name
        )

        if roi_summary is not None:
            roi_summary["n_epochs"] = n_epochs
            roi_summary["epoch_file"] = str(epoch_file)
            all_roi_summary_rows.append(roi_summary)

        if roi_name == "posterior":
            plot_specparam_roi_fits(
                subject=subject,
                roi_name=roi_name,
                freqs=freqs,
                psd_mean=psd_mean,
                ch_names=ch_names,
                roi_idx=roi_idx,
                fit_results=fit_results
            )

    # Clean models before returning
    for result in fit_results:
        result.pop("_model", None)

    return all_channel_rows, all_roi_summary_rows


def main():
    subjects = load_included_subjects()

    # Double protection: exclude S17
    subjects = [s for s in subjects if s not in EXCLUDE_SUBJECTS]

    print("\nFinal subjects for feature extraction:")
    print(subjects)
    print(f"Final N = {len(subjects)}")

    all_channel_rows = []
    all_roi_summary_rows = []

    for subject in subjects:
        try:
            channel_rows, roi_rows = process_subject(subject)
            all_channel_rows.extend(channel_rows)
            all_roi_summary_rows.extend(roi_rows)

        except Exception as e:
            print(f"[ERROR] Failed subject {subject}: {type(e).__name__}: {e}")

    df_channel = pd.DataFrame(all_channel_rows)
    df_roi = pd.DataFrame(all_roi_summary_rows)

    channel_out = TABLES_DIR / "resting_features_per_channel.csv"
    roi_out = TABLES_DIR / "resting_features_ROI_summary.csv"

    df_channel.to_csv(channel_out, index=False)
    df_roi.to_csv(roi_out, index=False)

    print("\n" + "=" * 60)
    print("Feature extraction complete.")
    print(f"Per-channel features saved to: {channel_out}")
    print(f"ROI summary features saved to: {roi_out}")

    if not df_roi.empty:
        print("\nROI summary preview:")
        preview_cols = [
            "subject",
            "roi",
            "n_epochs",
            "alpha_power_mean",
            "theta_power_mean",
            "beta_power_mean",
            "offset_mean",
            "exponent_mean",
            "r2_mean",
            "r2_min",
            "specparam_quality"
        ]

        available_cols = [col for col in preview_cols if col in df_roi.columns]
        print(df_roi[available_cols].to_string(index=False))

    print("\nFigures saved in:")
    print(PSD_DIR)
    print(SPECPARAM_FIT_DIR)


if __name__ == "__main__":
    main()