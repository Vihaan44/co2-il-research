"""
plot_tp_residuals.py
---------------------
PURPOSE: Diagnose whether the model's prediction error depends on temperature (T_K)
         or pressure (P_kPa) -- a check that the model generalizes across conditions,
         not just across ILs.

WHY THIS MATTERS:
  Our model uses T_K and P_kPa as features, so in principle it should work at
  any T and P in its training range. But if T_K or P_kPa are poorly covered
  in the training data, the model may have learned spurious correlations that
  only hold at commonly-seen conditions.

  If residuals (errors) show a systematic trend with T or P, the model's T/P
  generalization is broken and virtual screening at 298.15 K / 101.325 kPa
  may give biased predictions.

  If residuals are flat (no trend with T or P), the model generalizes well
  across conditions -- this is what we want to see and report to judges.

WHAT IT PRODUCES:
  A 2-row figure:
    Row 1: Residual (predicted - actual) vs T_K  [scatter + trend line]
    Row 2: Residual (predicted - actual) vs P_kPa [scatter + trend line]
  Annotated with Pearson R and p-value for each trend.
  Prints a verdict: 'No systematic trend' or 'WARNING: systematic trend found'.

OUTPUTS:
  figures/tp_residual_diagnostics.png

INPUTS:
  data/processed/test_set.csv       -- test set with T_K, P_kPa, log_x2_CO2
  results/test_predictions.csv      -- predicted values (from train_model.py)

Run from project root:
    python src/plot_tp_residuals.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# -- Constants -----------------------------------------------------------------
TEST_CSV         = os.path.join("data", "processed", "test_set.csv")
PREDICTIONS_CSV  = os.path.join("results", "test_predictions.csv")
FIGURES_DIR      = "figures"
FIGURE_PATH      = os.path.join(FIGURES_DIR, "tp_residual_diagnostics.png")

# A Pearson |r| above this threshold = noteworthy systematic trend
CORRELATION_CONCERN_THRESHOLD = 0.20


def load_and_merge(test_csv: str, predictions_csv: str) -> pd.DataFrame:
    """
    Load the test set (which has T_K, P_kPa) and the prediction CSV (which has
    y_pred_log and y_true_log), and merge them on il_smiles.

    We use the best model's predictions only -- if multiple models are in
    predictions_csv, we pick the one with the lowest RMSE on test.
    """
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test set not found at {test_csv}.")
    if not os.path.exists(predictions_csv):
        raise FileNotFoundError(
            f"Predictions CSV not found at {predictions_csv}. "
            f"Run train_model.py first."
        )

    test_df = pd.read_csv(test_csv)
    pred_df = pd.read_csv(predictions_csv)
    print(f"[load_and_merge] Test set: {len(test_df)} rows")
    print(f"[load_and_merge] Predictions: {len(pred_df)} rows, "
          f"models: {pred_df['model'].unique().tolist()}")

    # If multiple models in pred_df, pick the one with the best RMSE
    if pred_df["model"].nunique() > 1:
        rmse_by_model = pred_df.groupby("model").apply(
            lambda g: np.sqrt(((g["y_pred_log"] - g["y_true_log"]) ** 2).mean())
        )
        best_model_name = rmse_by_model.idxmin()
        pred_df = pred_df[pred_df["model"] == best_model_name].copy()
        print(f"[load_and_merge] Using predictions from: {best_model_name}")

    # Merge test conditions (T_K, P_kPa) with predictions (y_pred_log, y_true_log)
    # test_df may have multiple rows per IL (different T/P); pred_df has one row
    # per test observation aligned by row order from the test split.
    # The safest merge: if both have the same row count, align by index.
    if len(test_df) == len(pred_df):
        merged_df = test_df[["il_smiles", "T_K", "P_kPa", "log_x2_CO2"]].copy()
        merged_df["y_pred_log"]   = pred_df["y_pred_log"].values
        merged_df["residual_log"] = pred_df["residual_log"].values
        print(f"[load_and_merge] Row-aligned merge OK ({len(merged_df)} rows)")
    else:
        # Fallback: merge on il_smiles -- may produce duplicates if an IL appears
        # at multiple T/P conditions in both files
        print(f"[load_and_merge] Row counts differ ({len(test_df)} vs {len(pred_df)}) -- "
              f"merging on il_smiles (may produce duplicates)")
        merged_df = test_df[["il_smiles", "T_K", "P_kPa", "log_x2_CO2"]].merge(
            pred_df[["il_smiles", "y_pred_log", "residual_log"]],
            on="il_smiles", how="inner"
        )
        print(f"[load_and_merge] Merged: {len(merged_df)} rows")

    return merged_df


def compute_trend(x: np.ndarray, residuals: np.ndarray,
                  variable_name: str) -> dict:
    """
    Compute Pearson correlation between a condition variable and the residuals.
    Also fits a linear regression line for plotting.

    Returns dict with: r, p_value, slope, intercept, verdict.
    A significant positive r means the model underpredicts at high conditions.
    A significant negative r means the model overpredicts at high conditions.
    """
    r, p_value = stats.pearsonr(x, residuals)
    slope, intercept, _, _, _ = stats.linregress(x, residuals)

    # Interpret the trend for judges
    if abs(r) < CORRELATION_CONCERN_THRESHOLD:
        verdict = f"No systematic trend (|r|={abs(r):.3f} < {CORRELATION_CONCERN_THRESHOLD})"
    else:
        direction = "increases" if slope > 0 else "decreases"
        verdict   = (
            f"WARNING: systematic trend found (|r|={abs(r):.3f}). "
            f"Residual {direction} with {variable_name}. "
            f"Model may not generalize well across {variable_name}."
        )

    print(f"[compute_trend] {variable_name}: r={r:.3f}, p={p_value:.4f}, "
          f"slope={slope:.5f}")
    print(f"  Verdict: {verdict}")
    return {
        "variable": variable_name,
        "r":        r,
        "p_value":  p_value,
        "slope":    slope,
        "intercept": intercept,
        "verdict":  verdict,
    }


def plot_residuals_vs_tp(merged_df: pd.DataFrame,
                         trend_T: dict, trend_P: dict):
    """
    Plot residuals (predicted - actual, in log10 units) vs T_K and P_kPa.
    Each subplot shows a scatter of test points and a linear trend line.
    Annotated with r and p-value.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Residual Diagnostics: Do Prediction Errors Depend on T or P?",
                 fontsize=13)

    plot_configs = [
        (axes[0], "T_K",    trend_T, "Temperature (K)"),
        (axes[1], "P_kPa",  trend_P, "Pressure (kPa)"),
    ]

    for ax, col, trend, xlabel in plot_configs:
        x         = merged_df[col].values
        residuals = merged_df["residual_log"].values

        # Scatter of residuals
        ax.scatter(x, residuals, alpha=0.4, s=18, color="#3498db", label="Test points")

        # Zero-residual reference line
        ax.axhline(0, color="gray", linewidth=1, linestyle="--")

        # Trend line over the observed range
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = trend["slope"] * x_line + trend["intercept"]
        line_color = "#e74c3c" if abs(trend["r"]) >= CORRELATION_CONCERN_THRESHOLD else "#27ae60"
        ax.plot(x_line, y_line, color=line_color, linewidth=2,
                label=f"Trend: r={trend['r']:.3f}, p={trend['p_value']:.3f}")

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Residual: predicted − actual (log10 x2)", fontsize=10)
        ax.legend(fontsize=9)

        # Annotate verdict in corner
        verdict_color = "red" if "WARNING" in trend["verdict"] else "green"
        short_verdict = "No systematic trend" if "WARNING" not in trend["verdict"] \
                        else "Systematic trend - check carefully"
        ax.text(0.02, 0.97, short_verdict, transform=ax.transAxes, fontsize=9,
                color=verdict_color, va="top", ha="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved -> {FIGURE_PATH}")


def print_t_coverage(merged_df: pd.DataFrame):
    """
    Print T_K and P_kPa coverage in the test set so Vihaan can see what
    conditions the model is actually being evaluated at.
    If coverage is very narrow (e.g. only 290-310 K), the residual analysis
    may not reveal real generalization failure.
    """
    print("\n[coverage] T_K  distribution in test set:")
    print(f"  min={merged_df['T_K'].min():.1f}, max={merged_df['T_K'].max():.1f}, "
          f"median={merged_df['T_K'].median():.1f}, "
          f"n_unique={merged_df['T_K'].nunique()}")
    print("[coverage] P_kPa distribution in test set:")
    print(f"  min={merged_df['P_kPa'].min():.1f}, max={merged_df['P_kPa'].max():.1f}, "
          f"median={merged_df['P_kPa'].median():.1f}, "
          f"n_unique={merged_df['P_kPa'].nunique()}")

    if merged_df["T_K"].nunique() < 5:
        print("  WARNING: fewer than 5 unique T_K values in test set -- "
              "T/P residual analysis has low statistical power.")
    if merged_df["P_kPa"].nunique() < 5:
        print("  WARNING: fewer than 5 unique P_kPa values in test set -- "
              "T/P residual analysis has low statistical power.")


def main():
    """Main pipeline: load -> merge -> compute trends -> plot -> print verdicts."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # -- Step 1: Load and merge test set with predictions --------------------
    merged_df = load_and_merge(TEST_CSV, PREDICTIONS_CSV)

    # -- Step 2: Show T/P coverage -------------------------------------------
    print_t_coverage(merged_df)

    # -- Step 3: Compute trends ----------------------------------------------
    print("\n[main] Computing residual trends ...")
    residuals = merged_df["residual_log"].values
    trend_T   = compute_trend(merged_df["T_K"].values,   residuals, "T_K")
    trend_P   = compute_trend(merged_df["P_kPa"].values, residuals, "P_kPa")

    # -- Step 4: Plot --------------------------------------------------------
    plot_residuals_vs_tp(merged_df, trend_T, trend_P)

    # -- Step 5: Final summary -----------------------------------------------
    print("\n=== T/P GENERALIZATION SUMMARY ===")
    print(f"  T_K  verdict: {trend_T['verdict']}")
    print(f"  P_kPa verdict: {trend_P['verdict']}")
    print("\n  Interpretation for competition:")
    if abs(trend_T["r"]) < CORRELATION_CONCERN_THRESHOLD and \
       abs(trend_P["r"]) < CORRELATION_CONCERN_THRESHOLD:
        print("  GOOD: Residuals show no systematic dependence on T or P.")
        print("  The model generalizes across the tested T/P range.")
        print("  Virtual screening at 298.15 K / 101.325 kPa is supported.")
    else:
        print("  CAUTION: Systematic residual trend detected.")
        print("  Predictions at 298.15 K / 101.325 kPa may be biased.")
        print("  Consider: (1) adding more training data at target conditions,")
        print("            (2) restricting screening to conditions well-covered in training.")


if __name__ == "__main__":
    main()
