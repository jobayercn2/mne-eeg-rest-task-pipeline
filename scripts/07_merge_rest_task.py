from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path(r"I:\EEG_Python_Project")

REST_FEATURES_FILE = (
    PROJECT_DIR
    / "results"
    / "resting_state_features"
    / "tables"
    / "resting_features_ROI_summary.csv"
)

TASK_FEATURES_FILE = (
    PROJECT_DIR
    / "results"
    / "task_features"
    / "tables"
    / "rdk_task_features_differences.csv"
)

OUT_DIR = PROJECT_DIR / "results" / "merged_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Settings
# ============================================================

FINAL_SUBJECTS = ["S10", "S11", "S12", "S13", "S15", "S18"]

REST_ROI = "posterior"
TASK_ROI = "posterior"
TASK_CONTRAST = "coherent_minus_random"


# ============================================================
# Load data
# ============================================================

def load_rest_features():
    if not REST_FEATURES_FILE.exists():
        raise FileNotFoundError(f"Resting features file not found: {REST_FEATURES_FILE}")

    df = pd.read_csv(REST_FEATURES_FILE)

    df = df[
        (df["subject"].isin(FINAL_SUBJECTS)) &
        (df["roi"] == REST_ROI)
    ].copy()

    if "specparam_quality" in df.columns:
        df = df[df["specparam_quality"] == "OK"].copy()

    return df


def load_task_features():
    if not TASK_FEATURES_FILE.exists():
        raise FileNotFoundError(f"Task features file not found: {TASK_FEATURES_FILE}")

    df = pd.read_csv(TASK_FEATURES_FILE)

    df = df[
        (df["subject"].isin(FINAL_SUBJECTS)) &
        (df["roi"] == TASK_ROI) &
        (df["contrast"] == TASK_CONTRAST)
    ].copy()

    return df


# ============================================================
# Prepare columns
# ============================================================

def prepare_rest_features(df):
    """
    Keep and rename important resting-state features.
    """

    keep_cols = [
        "subject",
        "roi",
        "n_epochs",

        "delta_power_mean",
        "theta_power_mean",
        "alpha_power_mean",
        "beta_power_mean",
        "gamma_power_mean",

        "delta_relative_power_mean",
        "theta_relative_power_mean",
        "alpha_relative_power_mean",
        "beta_relative_power_mean",
        "gamma_relative_power_mean",

        "offset_mean",
        "exponent_mean",
        "alpha_cf_mean",
        "alpha_pw_mean",
        "alpha_bw_mean",

        "r2_mean",
        "r2_min",
        "specparam_quality",
    ]

    available_cols = [col for col in keep_cols if col in df.columns]
    df = df[available_cols].copy()

    rename_dict = {
        "roi": "rest_roi",
        "n_epochs": "rest_n_epochs",

        "delta_power_mean": "rest_delta_power",
        "theta_power_mean": "rest_theta_power",
        "alpha_power_mean": "rest_alpha_power",
        "beta_power_mean": "rest_beta_power",
        "gamma_power_mean": "rest_gamma_power",

        "delta_relative_power_mean": "rest_delta_relative_power",
        "theta_relative_power_mean": "rest_theta_relative_power",
        "alpha_relative_power_mean": "rest_alpha_relative_power",
        "beta_relative_power_mean": "rest_beta_relative_power",
        "gamma_relative_power_mean": "rest_gamma_relative_power",

        "offset_mean": "rest_offset",
        "exponent_mean": "rest_exponent",
        "alpha_cf_mean": "rest_alpha_cf",
        "alpha_pw_mean": "rest_alpha_pw",
        "alpha_bw_mean": "rest_alpha_bw",

        "r2_mean": "rest_specparam_r2_mean",
        "r2_min": "rest_specparam_r2_min",
        "specparam_quality": "rest_specparam_quality",
    }

    df = df.rename(columns=rename_dict)

    return df


def prepare_task_features(df):
    """
    Keep and rename important RDK task difference features.
    """

    keep_cols = [
        "subject",
        "roi",
        "contrast",

        "theta_power_mean",
        "alpha_power_mean",
        "beta_power_mean",
        "gamma_power_mean",

        "theta_relative_power_mean",
        "alpha_relative_power_mean",
        "beta_relative_power_mean",
        "gamma_relative_power_mean",

        "total_power_1_40_mean",
    ]

    available_cols = [col for col in keep_cols if col in df.columns]
    df = df[available_cols].copy()

    rename_dict = {
        "roi": "task_roi",
        "contrast": "task_contrast",

        "theta_power_mean": "task_theta_power_diff",
        "alpha_power_mean": "task_alpha_power_diff",
        "beta_power_mean": "task_beta_power_diff",
        "gamma_power_mean": "task_gamma_power_diff",

        "theta_relative_power_mean": "task_theta_relative_power_diff",
        "alpha_relative_power_mean": "task_alpha_relative_power_diff",
        "beta_relative_power_mean": "task_beta_relative_power_diff",
        "gamma_relative_power_mean": "task_gamma_relative_power_diff",

        "total_power_1_40_mean": "task_total_power_diff",
    }

    df = df.rename(columns=rename_dict)

    return df


# ============================================================
# Merge
# ============================================================

def merge_features(rest_df, task_df):
    merged = pd.merge(
        rest_df,
        task_df,
        on="subject",
        how="inner"
    )

    merged = merged.sort_values("subject").reset_index(drop=True)

    return merged


def save_outputs(rest_df, task_df, merged_df):
    rest_out = OUT_DIR / "rest_features_for_merge.csv"
    task_out = OUT_DIR / "task_features_for_merge.csv"
    merged_out = OUT_DIR / "merged_rest_task_features.csv"

    rest_df.to_csv(rest_out, index=False)
    task_df.to_csv(task_out, index=False)
    merged_df.to_csv(merged_out, index=False)

    print(f"Saved rest merge table: {rest_out}")
    print(f"Saved task merge table: {task_out}")
    print(f"Saved final merged table: {merged_out}")


# ============================================================
# Main
# ============================================================

def main():
    print("Loading resting-state features...")
    rest_df_raw = load_rest_features()

    print("Loading RDK task features...")
    task_df_raw = load_task_features()

    print("\nPreparing resting-state features...")
    rest_df = prepare_rest_features(rest_df_raw)

    print("Preparing task features...")
    task_df = prepare_task_features(task_df_raw)

    print("\nMerging rest + task features...")
    merged_df = merge_features(rest_df, task_df)

    save_outputs(rest_df, task_df, merged_df)

    print("\n" + "=" * 60)
    print("MERGE SUMMARY")
    print("=" * 60)

    print(f"Rest subjects: {sorted(rest_df['subject'].tolist())}")
    print(f"Task subjects: {sorted(task_df['subject'].tolist())}")
    print(f"Merged subjects: {sorted(merged_df['subject'].tolist())}")
    print(f"Final merged N = {len(merged_df)}")

    print("\nMerged table preview:")
    preview_cols = [
        "subject",
        "rest_exponent",
        "rest_offset",
        "rest_alpha_power",
        "rest_alpha_relative_power",
        "task_theta_power_diff",
        "task_alpha_power_diff",
        "task_beta_power_diff",
        "task_gamma_power_diff",
    ]

    available_preview_cols = [
        col for col in preview_cols
        if col in merged_df.columns
    ]

    print(merged_df[available_preview_cols].to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()