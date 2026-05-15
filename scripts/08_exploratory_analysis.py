from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path(r"I:\EEG_Python_Project")

MERGED_FILE = (
    PROJECT_DIR
    / "results"
    / "merged_features"
    / "merged_rest_task_features.csv"
)

OUT_DIR = PROJECT_DIR / "results" / "merged_analysis"
TABLES_DIR = OUT_DIR / "tables"
FIGURES_DIR = OUT_DIR / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Analysis settings
# ============================================================

REST_PREDICTORS = [
    "rest_exponent",
    "rest_offset",
    "rest_alpha_power",
    "rest_alpha_relative_power",
    "rest_theta_power",
    "rest_beta_power",
]

TASK_OUTCOMES = [
    "task_theta_power_diff",
    "task_alpha_power_diff",
    "task_beta_power_diff",
    "task_gamma_power_diff",
]

PRIMARY_PAIRS = [
    ("rest_exponent", "task_alpha_power_diff"),
    ("rest_exponent", "task_beta_power_diff"),
    ("rest_offset", "task_alpha_power_diff"),
    ("rest_offset", "task_beta_power_diff"),
    ("rest_alpha_power", "task_alpha_power_diff"),
    ("rest_alpha_relative_power", "task_alpha_power_diff"),
]


# ============================================================
# Helper functions
# ============================================================

def load_data():
    """
    Load final merged rest-task feature table.
    """
    if not MERGED_FILE.exists():
        raise FileNotFoundError(f"Merged file not found: {MERGED_FILE}")

    df = pd.read_csv(MERGED_FILE)

    if "subject" not in df.columns:
        raise ValueError("The merged file must contain a 'subject' column.")

    print(f"Loaded merged data: {MERGED_FILE}")
    print(f"N subjects = {len(df)}")
    print(f"Subjects: {df['subject'].tolist()}")

    return df


def safe_corr(x, y, method="pearson"):
    """
    Compute correlation safely.
    Returns r, p, n.
    """
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(data)

    if n < 3:
        return np.nan, np.nan, n

    if data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return np.nan, np.nan, n

    if method == "pearson":
        r, p = pearsonr(data["x"], data["y"])
    elif method == "spearman":
        r, p = spearmanr(data["x"], data["y"])
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")

    return float(r), float(p), n


def correlation_table(df):
    """
    Compute Pearson and Spearman correlations for all rest-task pairs.
    """
    rows = []

    for predictor in REST_PREDICTORS:
        if predictor not in df.columns:
            print(f"[WARNING] Missing predictor column: {predictor}")
            continue

        for outcome in TASK_OUTCOMES:
            if outcome not in df.columns:
                print(f"[WARNING] Missing outcome column: {outcome}")
                continue

            pearson_r, pearson_p, n = safe_corr(df[predictor], df[outcome], method="pearson")
            spearman_r, spearman_p, _ = safe_corr(df[predictor], df[outcome], method="spearman")

            rows.append({
                "predictor": predictor,
                "outcome": outcome,
                "n": n,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_r,
                "spearman_p": spearman_p,
            })

    corr_df = pd.DataFrame(rows)

    if not corr_df.empty:
        corr_df["abs_pearson_r"] = corr_df["pearson_r"].abs()
        corr_df = corr_df.sort_values("abs_pearson_r", ascending=False)

    return corr_df


def plot_scatter(df, x_col, y_col, output_name=None):
    """
    Create scatterplot with regression line and subject labels.
    """
    if x_col not in df.columns or y_col not in df.columns:
        print(f"[WARNING] Cannot plot missing columns: {x_col}, {y_col}")
        return

    data = df[["subject", x_col, y_col]].dropna()

    if len(data) < 3:
        print(f"[WARNING] Not enough data to plot {x_col} vs {y_col}")
        return

    x = data[x_col].values
    y = data[y_col].values

    pearson_r, pearson_p, n = safe_corr(x, y, method="pearson")
    spearman_r, spearman_p, _ = safe_corr(x, y, method="spearman")

    plt.figure(figsize=(6.5, 5))

    plt.scatter(x, y)

    # Regression line
    if len(np.unique(x)) > 1:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(np.min(x), np.max(x), 100)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, linestyle="--", linewidth=1)

    # Subject labels
    for _, row in data.iterrows():
        plt.annotate(
            row["subject"],
            (row[x_col], row[y_col]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8
        )

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(
        f"{x_col} vs {y_col}\n"
        f"Pearson r={pearson_r:.2f}, p={pearson_p:.3f}; "
        f"Spearman ρ={spearman_r:.2f}, p={spearman_p:.3f}; N={n}"
    )

    plt.tight_layout()

    if output_name is None:
        output_name = f"{x_col}_vs_{y_col}.png"

    out_path = FIGURES_DIR / output_name
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved plot: {out_path}")


def plot_all_primary_pairs(df):
    """
    Plot selected theoretically meaningful relationships.
    """
    for x_col, y_col in PRIMARY_PAIRS:
        output_name = f"{x_col}_vs_{y_col}.png"
        plot_scatter(df, x_col, y_col, output_name)


def plot_correlation_heatmap(corr_df):
    """
    Plot heatmap-like correlation matrix using matplotlib.
    """
    if corr_df.empty:
        print("[WARNING] Correlation table is empty. Skipping heatmap.")
        return

    pivot = corr_df.pivot(
        index="predictor",
        columns="outcome",
        values="pearson_r"
    )

    plt.figure(figsize=(9, 5))
    im = plt.imshow(pivot.values, aspect="auto", vmin=-1, vmax=1)

    plt.xticks(
        ticks=np.arange(len(pivot.columns)),
        labels=pivot.columns,
        rotation=45,
        ha="right"
    )
    plt.yticks(
        ticks=np.arange(len(pivot.index)),
        labels=pivot.index
    )

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            if not np.isnan(value):
                plt.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

    plt.colorbar(im, label="Pearson r")
    plt.title("Resting EEG Features vs RDK Task Modulation")
    plt.tight_layout()

    out_path = FIGURES_DIR / "correlation_heatmap.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved heatmap: {out_path}")


def create_descriptive_table(df):
    """
    Create descriptive statistics for key variables.
    """
    key_cols = ["subject"] + REST_PREDICTORS + TASK_OUTCOMES
    available_cols = [col for col in key_cols if col in df.columns]

    numeric_cols = [col for col in available_cols if col != "subject"]

    desc = df[numeric_cols].describe().T.reset_index()
    desc = desc.rename(columns={"index": "variable"})

    return desc


def save_outputs(df, corr_df, desc_df):
    """
    Save final tables.
    """
    clean_data_out = TABLES_DIR / "analysis_dataset.csv"
    corr_out = TABLES_DIR / "rest_task_correlations.csv"
    desc_out = TABLES_DIR / "descriptive_statistics.csv"

    df.to_csv(clean_data_out, index=False)
    corr_df.to_csv(corr_out, index=False)
    desc_df.to_csv(desc_out, index=False)

    print(f"Saved analysis dataset: {clean_data_out}")
    print(f"Saved correlation table: {corr_out}")
    print(f"Saved descriptive statistics: {desc_out}")


def print_summary(df, corr_df, desc_df):
    """
    Print a compact analysis summary.
    """
    print("\n" + "=" * 70)
    print("MERGED REST-TASK ANALYSIS SUMMARY")
    print("=" * 70)

    print("\nSubjects:")
    print(", ".join(df["subject"].tolist()))
    print(f"\nFinal N = {len(df)}")

    print("\nDescriptive statistics:")
    print(desc_df[["variable", "count", "mean", "std", "min", "max"]].to_string(index=False))

    print("\nTop correlations by absolute Pearson r:")
    if corr_df.empty:
        print("No correlations computed.")
    else:
        cols = [
            "predictor",
            "outcome",
            "n",
            "pearson_r",
            "pearson_p",
            "spearman_rho",
            "spearman_p"
        ]
        print(corr_df[cols].head(12).to_string(index=False))

    print("\nInterpretation note:")
    print(
        "The final matched sample is small (N=6). "
        "Treat these results as exploratory and pipeline-demonstration level, "
        "not confirmatory inferential evidence."
    )


# ============================================================
# Main
# ============================================================

def main():
    df = load_data()

    desc_df = create_descriptive_table(df)
    corr_df = correlation_table(df)

    save_outputs(df, corr_df, desc_df)

    plot_all_primary_pairs(df)
    plot_correlation_heatmap(corr_df)

    print_summary(df, corr_df, desc_df)

    print("\nAnalysis complete.")
    print(f"Tables saved in: {TABLES_DIR}")
    print(f"Figures saved in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()