from pathlib import Path

import mne
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PROCESSED_DIR = Path(r"I:\EEG_Python_Project\data\processed\resting_state")
RESULTS_DIR = Path(r"I:\EEG_Python_Project\results\resting_state_qc")

FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
PSD_DIR = FIGURES_DIR / "psd_by_subject"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
PSD_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Settings
# ============================================================

LOG_FILE = PROCESSED_DIR / "resting_preprocessing_log.csv"

# Manual exclusion based on preprocessing result
EXCLUDE_SUBJECTS = ["S17"]

# QC thresholds
GOOD_THRESHOLD = 80
ACCEPTABLE_THRESHOLD = 60
CHECK_THRESHOLD = 40


# ============================================================
# Helper functions
# ============================================================

def assign_qc_decision(percent_kept):
    """
    Assign QC category based on percentage of epochs kept.
    """
    if percent_kept >= GOOD_THRESHOLD:
        return "good"
    elif percent_kept >= ACCEPTABLE_THRESHOLD:
        return "acceptable"
    elif percent_kept >= CHECK_THRESHOLD:
        return "check"
    else:
        return "problematic"


def load_qc_log():
    """
    Load preprocessing log and add QC decision.
    """
    if not LOG_FILE.exists():
        raise FileNotFoundError(f"Log file not found: {LOG_FILE}")

    df = pd.read_csv(LOG_FILE)

    required_cols = [
        "subject",
        "epochs_before_rejection",
        "epochs_after_rejection",
        "percent_epochs_kept",
        "cleaned_epochs_file",
        "status"
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns in log file: {missing_cols}")

    df["qc_decision"] = df["percent_epochs_kept"].apply(assign_qc_decision)
    df["manual_exclusion"] = df["subject"].isin(EXCLUDE_SUBJECTS)
    df["final_include"] = (df["status"] == "success") & (~df["manual_exclusion"])

    return df


def save_qc_tables(df):
    """
    Save full QC table, included subjects, and excluded subjects.
    """
    full_qc_path = TABLES_DIR / "resting_qc_summary.csv"
    included_path = TABLES_DIR / "included_subjects.csv"
    excluded_path = TABLES_DIR / "excluded_subjects.csv"

    included_df = df[df["final_include"]].copy()
    excluded_df = df[~df["final_include"]].copy()

    df.to_csv(full_qc_path, index=False)
    included_df.to_csv(included_path, index=False)
    excluded_df.to_csv(excluded_path, index=False)

    print(f"Saved QC summary: {full_qc_path}")
    print(f"Saved included subjects: {included_path}")
    print(f"Saved excluded subjects: {excluded_path}")

    return included_df, excluded_df


def plot_epoch_retention(df):
    """
    Plot percentage of epochs kept per subject.
    """
    plot_df = df.sort_values("subject").copy()

    plt.figure(figsize=(10, 5))
    plt.bar(plot_df["subject"], plot_df["percent_epochs_kept"])

    plt.axhline(GOOD_THRESHOLD, linestyle="--", linewidth=1, label="Good threshold: 80%")
    plt.axhline(ACCEPTABLE_THRESHOLD, linestyle="--", linewidth=1, label="Acceptable threshold: 60%")
    plt.axhline(CHECK_THRESHOLD, linestyle="--", linewidth=1, label="Check threshold: 40%")

    for _, row in plot_df.iterrows():
        label = f"{row['percent_epochs_kept']:.1f}%"
        plt.text(
            row["subject"],
            row["percent_epochs_kept"] + 1,
            label,
            ha="center",
            fontsize=8
        )

    plt.title("Resting-State EEG Quality Control: Percentage of Epochs Kept")
    plt.xlabel("Subject")
    plt.ylabel("Epochs kept (%)")
    plt.ylim(0, 110)
    plt.legend()
    plt.tight_layout()

    output_path = FIGURES_DIR / "epoch_retention_percentage.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved epoch retention plot: {output_path}")


def plot_epochs_count(df):
    """
    Plot number of clean epochs per subject.
    """
    plot_df = df.sort_values("subject").copy()

    plt.figure(figsize=(10, 5))
    plt.bar(plot_df["subject"], plot_df["epochs_after_rejection"])

    for _, row in plot_df.iterrows():
        plt.text(
            row["subject"],
            row["epochs_after_rejection"] + 1,
            str(int(row["epochs_after_rejection"])),
            ha="center",
            fontsize=8
        )

    plt.title("Resting-State EEG Quality Control: Clean Epochs per Subject")
    plt.xlabel("Subject")
    plt.ylabel("Number of clean 2-second epochs")
    plt.tight_layout()

    output_path = FIGURES_DIR / "clean_epochs_count.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved clean epochs count plot: {output_path}")


def generate_psd_plots(included_df):
    """
    Generate PSD plots for each included subject using cleaned epochs.
    """
    for _, row in included_df.iterrows():
        subject = row["subject"]
        epochs_file = Path(row["cleaned_epochs_file"])

        if not epochs_file.exists():
            print(f"Skipping {subject}: epochs file not found")
            continue

        print(f"Generating PSD plot for {subject}")

        epochs = mne.read_epochs(epochs_file, preload=True, verbose="ERROR")

        psd = epochs.compute_psd(
            method="welch",
            fmin=1,
            fmax=40,
            n_fft=512,
            n_overlap=128,
            verbose="ERROR"
        )

        fig = psd.plot(
            average=True,
            spatial_colors=False,
            show=False
        )

        fig.suptitle(f"PSD - {subject}", fontsize=14)

        output_path = PSD_DIR / f"{subject}_psd.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

    print(f"Saved PSD plots in: {PSD_DIR}")


def print_summary(df, included_df, excluded_df):
    """
    Print readable QC summary.
    """
    print("\n" + "=" * 60)
    print("RESTING-STATE QC SUMMARY")
    print("=" * 60)

    print("\nAll subjects:")
    print(df[[
        "subject",
        "epochs_before_rejection",
        "epochs_after_rejection",
        "percent_epochs_kept",
        "qc_decision",
        "manual_exclusion",
        "final_include"
    ]].to_string(index=False))

    print("\nIncluded subjects:")
    print(", ".join(included_df["subject"].tolist()))

    print("\nExcluded subjects:")
    print(", ".join(excluded_df["subject"].tolist()))

    print("\nFinal N:")
    print(f"N included = {len(included_df)}")
    print(f"N excluded = {len(excluded_df)}")


# ============================================================
# Main
# ============================================================

def main():
    df = load_qc_log()

    included_df, excluded_df = save_qc_tables(df)

    plot_epoch_retention(df)
    plot_epochs_count(df)
    generate_psd_plots(included_df)

    print_summary(df, included_df, excluded_df)

    print("\nQC complete.")


if __name__ == "__main__":
    main()