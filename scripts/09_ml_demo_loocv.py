from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path(r"I:\EEG_Python_Project")

DATA_FILE = (
    PROJECT_DIR
    / "results"
    / "merged_features"
    / "merged_rest_task_features.csv"
)

OUT_DIR = PROJECT_DIR / "results" / "basic_ml"
TABLES_DIR = OUT_DIR / "tables"
FIGURES_DIR = OUT_DIR / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ML settings
# ============================================================

PREDICTORS = [
    "rest_exponent",
    "rest_offset",
    "rest_alpha_power",
    "rest_alpha_relative_power",
    "rest_theta_power",
    "rest_beta_power",
]

TARGETS = [
    "task_alpha_power_diff",
    "task_beta_power_diff",
    "task_gamma_power_diff",
]

MODELS = {
    "linear_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "ridge_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),

    "random_forest": Pipeline([
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=2
        ))
    ])
}


# ============================================================
# Helper functions
# ============================================================

def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    print(f"Loaded data: {DATA_FILE}")
    print(f"N subjects = {len(df)}")
    print(f"Subjects: {df['subject'].tolist()}")

    return df


def prepare_xy(df, target):
    cols = ["subject"] + PREDICTORS + [target]
    missing = [col for col in cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data = df[cols].dropna().copy()

    X = data[PREDICTORS].values
    y = data[target].values
    subjects = data["subject"].values

    return data, X, y, subjects


def evaluate_predictions(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # R2 can be unstable with N=6, but we calculate it for demonstration.
    try:
        r2 = r2_score(y_true, y_pred)
    except Exception:
        r2 = np.nan

    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 2 else np.nan

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "prediction_correlation": corr
    }


def run_loocv_model(df, target, model_name, model):
    """
    Leave-one-subject-out prediction.
    """
    data, X, y, subjects = prepare_xy(df, target)

    loo = LeaveOneOut()

    y_pred = cross_val_predict(
        model,
        X,
        y,
        cv=loo
    )

    metrics = evaluate_predictions(y, y_pred)

    prediction_rows = []

    for subject, true_value, predicted_value in zip(subjects, y, y_pred):
        prediction_rows.append({
            "subject": subject,
            "target": target,
            "model": model_name,
            "true_value": true_value,
            "predicted_value": predicted_value,
            "error": predicted_value - true_value,
            "absolute_error": abs(predicted_value - true_value)
        })

    result_row = {
        "target": target,
        "model": model_name,
        "n": len(y),
        **metrics
    }

    return result_row, prediction_rows


def plot_predictions(pred_df, target, model_name):
    """
    Plot predicted vs observed values.
    """
    sub = pred_df[
        (pred_df["target"] == target) &
        (pred_df["model"] == model_name)
    ].copy()

    if sub.empty:
        return

    y_true = sub["true_value"].values
    y_pred = sub["predicted_value"].values

    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred)

    # Identity line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1)

    for _, row in sub.iterrows():
        plt.annotate(
            row["subject"],
            (row["true_value"], row["predicted_value"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8
        )

    plt.xlabel("Observed value")
    plt.ylabel("Predicted value")
    plt.title(f"{model_name}: predicted vs observed\nTarget: {target}")
    plt.tight_layout()

    out_path = FIGURES_DIR / f"{model_name}_{target}_predicted_vs_observed.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved plot: {out_path}")


def fit_final_model_for_feature_importance(df, target):
    """
    Fit a simple Ridge model on full data to inspect coefficient direction.
    This is descriptive only, not confirmatory.
    """
    data, X, y, subjects = prepare_xy(df, target)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])

    model.fit(X, y)

    coefs = model.named_steps["model"].coef_

    coef_df = pd.DataFrame({
        "target": target,
        "predictor": PREDICTORS,
        "ridge_coefficient": coefs
    })

    coef_df["abs_coefficient"] = coef_df["ridge_coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False)

    return coef_df


# ============================================================
# Main
# ============================================================

def main():
    df = load_data()

    all_results = []
    all_predictions = []
    all_coefficients = []

    print("\nRunning basic ML with Leave-One-Out Cross-Validation...")
    print("Important: N=6, so this is pipeline demonstration only.\n")

    for target in TARGETS:
        print("=" * 70)
        print(f"Target: {target}")
        print("=" * 70)

        for model_name, model in MODELS.items():
            print(f"Running model: {model_name}")

            result_row, prediction_rows = run_loocv_model(
                df=df,
                target=target,
                model_name=model_name,
                model=model
            )

            all_results.append(result_row)
            all_predictions.extend(prediction_rows)

            print(
                f"MAE={result_row['mae']:.3e}, "
                f"RMSE={result_row['rmse']:.3e}, "
                f"R2={result_row['r2']:.3f}, "
                f"corr={result_row['prediction_correlation']:.3f}"
            )

        coef_df = fit_final_model_for_feature_importance(df, target)
        all_coefficients.append(coef_df)

    results_df = pd.DataFrame(all_results)
    predictions_df = pd.DataFrame(all_predictions)
    coefficients_df = pd.concat(all_coefficients, ignore_index=True)

    # Save outputs
    results_out = TABLES_DIR / "ml_model_performance.csv"
    predictions_out = TABLES_DIR / "ml_predictions.csv"
    coefficients_out = TABLES_DIR / "ridge_coefficients.csv"

    results_df.to_csv(results_out, index=False)
    predictions_df.to_csv(predictions_out, index=False)
    coefficients_df.to_csv(coefficients_out, index=False)

    print("\nSaved ML performance table:")
    print(results_out)

    print("Saved ML predictions:")
    print(predictions_out)

    print("Saved Ridge coefficients:")
    print(coefficients_out)

    # Plot predictions for Ridge model only
    for target in TARGETS:
        plot_predictions(predictions_df, target, "ridge_regression")

    print("\n" + "=" * 70)
    print("BASIC ML SUMMARY")
    print("=" * 70)

    print("\nModel performance:")
    print(results_df.to_string(index=False))

    print("\nTop Ridge coefficients:")
    print(coefficients_df.groupby("target").head(3).to_string(index=False))

    print("\nInterpretation note:")
    print(
        "This ML analysis is not intended as a reliable predictive model because "
        "the matched dataset contains only six participants. It is included as a "
        "demonstration of a reproducible machine-learning workflow: feature table "
        "loading, preprocessing, leave-one-subject-out validation, prediction, "
        "performance reporting, and coefficient inspection."
    )

    print("\nDone.")


if __name__ == "__main__":
    main()