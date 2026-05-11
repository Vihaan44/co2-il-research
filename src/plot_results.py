"""
plot_results.py
---------------
PURPOSE: Generate three publication-quality diagnostic figures from Phase 3 model results.

  Figure 1 — Predicted vs Actual scatter (test set)
    Shows how well the XGBoost model's predictions match experimental values.
    Points near the diagonal = accurate predictions.

  Figure 2 — Feature Importance bar chart (top 20)
    Shows which structural bits / condition variables most influence predictions.
    T_K and P_kPa dominating is physically expected (Henry's law).
    Molecular bits that rank highly identify structural motifs that drive CO2 absorption.

  Figure 3 — Residual distribution histogram
    Residual = predicted − actual (log10 units).
    Symmetric distribution centered near 0 = no systematic bias.
    Heavy tails = outlier ILs where the model struggles.

INPUTS:
  results/test_predictions.csv      (from train_model.py)
  results/feature_importances.csv   (from train_model.py)
  models/best_model_name.txt        (from train_model.py)

OUTPUTS:
  figures/predicted_vs_actual.png
  figures/feature_importance.png
  figures/residual_distribution.png

Run from project root:
    python src/plot_results.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Constants ──────────────────────────────────────────────────────────────────
RESULTS_DIR   = "results"
FIGURES_DIR   = "figures"
PREDICTIONS_CSV      = os.path.join(RESULTS_DIR, "test_predictions.csv")
FEATURE_IMP_CSV      = os.path.join(RESULTS_DIR, "feature_importances.csv")
BEST_MODEL_NAME_FILE = os.path.join("models", "best_model_name.txt")

FIG_DPI         = 300    # print-quality resolution
FIG_WIDTH       = 7      # inches — fits single-column journal layout
FIG_HEIGHT      = 5
TOP_N_FEATURES  = 20     # how many features to show in importance chart

# Consistent color palette
SCATTER_COLOR   = "#2563EB"   # blue for test points
IMPORTANCE_COLOR_CONDITION  = "#DC2626"  # red  — T_K / P_kPa bars
IMPORTANCE_COLOR_MOLECULAR  = "#16A34A"  # green — molecular fingerprint bars
RESIDUAL_COLOR  = "#7C3AED"  # purple for histogram


def load_results() -> tuple:
    """
    Load test predictions and feature importances CSVs.
    Filters to the best model only (in case both RF and XGBoost results are present).
    Returns: predictions_df, importance_df, best_model_name (str)
    """
    if not os.path.exists(PREDICTIONS_CSV):
        raise FileNotFoundError(
            f"Missing {PREDICTIONS_CSV}. Run src/train_model.py first."
        )
    if not os.path.exists(FEATURE_IMP_CSV):
        raise FileNotFoundError(
            f"Missing {FEATURE_IMP_CSV}. Run src/train_model.py first."
        )

    # Read which model was selected as best
    if os.path.exists(BEST_MODEL_NAME_FILE):
        with open(BEST_MODEL_NAME_FILE) as f:
            best_model_name = f.read().strip()
    else:
        best_model_name = None   # fall back to using all rows

    predictions_df = pd.read_csv(PREDICTIONS_CSV)
    importance_df  = pd.read_csv(FEATURE_IMP_CSV)

    # Filter to best model only if the 'model' column exists
    if best_model_name and "model" in predictions_df.columns:
        predictions_df = predictions_df[predictions_df["model"] == best_model_name].copy()
        importance_df  = importance_df[importance_df["model"] == best_model_name].copy()
        print(f"[load_results] Using best model: {best_model_name}")
    else:
        print("[load_results] No model filter applied — using all rows")
        best_model_name = best_model_name or "Model"

    print(f"[load_results] {len(predictions_df)} test predictions loaded")
    print(f"[load_results] {len(importance_df)} feature importance rows loaded")
    return predictions_df, importance_df, best_model_name


def plot_predicted_vs_actual(predictions_df: pd.DataFrame,
                              best_model_name: str) -> None:
    """
    Scatter plot: x = actual log10(x2_CO2), y = predicted log10(x2_CO2).
    A dashed diagonal (perfect prediction line) is overlaid.
    R² and RMSE are annotated on the figure so judges can read them at a glance.
    """
    y_true = predictions_df["y_true_log"].values
    y_pred = predictions_df["y_pred_log"].values

    # Compute metrics for annotation
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2   = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    ax.scatter(y_true, y_pred,
               color=SCATTER_COLOR, alpha=0.7, edgecolors="white",
               linewidths=0.4, s=60, zorder=3, label="Test IL")

    # Diagonal perfect-prediction line
    axis_min = min(y_true.min(), y_pred.min()) - 0.2
    axis_max = max(y_true.max(), y_pred.max()) + 0.2
    ax.plot([axis_min, axis_max], [axis_min, axis_max],
            color="black", linestyle="--", linewidth=1.2,
            label="Perfect prediction", zorder=2)

    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_xlabel(r"Actual log$_{10}$(x$_2^{\mathrm{CO}_2}$)", fontsize=12)
    ax.set_ylabel(r"Predicted log$_{10}$(x$_2^{\mathrm{CO}_2}$)", fontsize=12)
    ax.set_title(f"{best_model_name} — Predicted vs. Actual (Test Set)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_aspect("equal", adjustable="box")

    # R² and RMSE annotation box
    annotation_text = f"$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f} log$_{{10}}$ units"
    ax.text(0.05, 0.93, annotation_text,
            transform=ax.transAxes,
            fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.8))

    ax.grid(True, linestyle=":", alpha=0.5, zorder=1)

    out_path = os.path.join(FIGURES_DIR, "predicted_vs_actual.png")
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_predicted_vs_actual] Saved → {out_path}")
    print(f"  R² = {r2:.4f} | RMSE = {rmse:.4f} log10 units")


def plot_feature_importance(importance_df: pd.DataFrame,
                             best_model_name: str) -> None:
    """
    Horizontal bar chart of top N features by importance score.

    Bars are color-coded:
      - Red  = condition features (T_K, P_kPa) — expected to dominate due to Henry's law
      - Green = molecular features (Morgan FP bits and RDKit descriptors) — differentiate ILs

    This color distinction is a key talking point: "After controlling for temperature
    and pressure, these molecular bits determine which ILs absorb more CO2."
    """
    condition_feature_names = {"T_K", "P_kPa"}  # from CONDITION_FEATURES in train_model.py

    top_features = (
        importance_df
        .sort_values("importance", ascending=False)
        .head(TOP_N_FEATURES)
        .sort_values("importance", ascending=True)   # ascending for horizontal bar (bottom = highest)
    )

    # Assign color per bar based on whether it's a condition or molecular feature
    colors = [
        IMPORTANCE_COLOR_CONDITION if feat in condition_feature_names else IMPORTANCE_COLOR_MOLECULAR
        for feat in top_features["feature"]
    ]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT + 1))  # taller for many labels

    bars = ax.barh(top_features["feature"], top_features["importance"],
                   color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Feature Importance (Gain)", fontsize=12)
    ax.set_title(f"{best_model_name} — Top {TOP_N_FEATURES} Feature Importances", fontsize=13)

    # Custom legend for color coding
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=IMPORTANCE_COLOR_CONDITION, label="Condition (T, P)"),
        Patch(facecolor=IMPORTANCE_COLOR_MOLECULAR,  label="Molecular fingerprint / descriptor"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)

    out_path = os.path.join(FIGURES_DIR, "feature_importance.png")
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_feature_importance] Saved → {out_path}")
    print(f"  Top feature: {top_features.iloc[-1]['feature']} "
          f"(importance = {top_features.iloc[-1]['importance']:.5f})")


def plot_residual_distribution(predictions_df: pd.DataFrame,
                                best_model_name: str) -> None:
    """
    Histogram of residuals (predicted − actual, in log10 units).

    A well-calibrated model produces:
      - Symmetric distribution centered near 0 (no systematic over/under-prediction)
      - Most residuals within ±0.3 log10 units (factor of 2x in mole fraction)

    Outliers in the tails identify ILs where the model extrapolates — useful for
    honest limitation discussion with judges.
    """
    residuals = predictions_df["residual_log"].values  # positive = overpredicted

    mean_resid = residuals.mean()
    std_resid  = residuals.std()
    n_outliers = np.sum(np.abs(residuals) > 0.5)  # >0.5 log10 = >3x error in x2

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    ax.hist(residuals, bins=20, color=RESIDUAL_COLOR, alpha=0.8,
            edgecolor="white", linewidth=0.5)

    # Vertical line at zero
    ax.axvline(0, color="black", linestyle="--", linewidth=1.3, label="Zero bias")
    # Vertical line at mean residual — indicates systematic bias direction
    ax.axvline(mean_resid, color="red", linestyle="-", linewidth=1.3,
               label=f"Mean = {mean_resid:+.4f}")

    ax.set_xlabel(r"Residual: Predicted $-$ Actual (log$_{10}$ units)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"{best_model_name} — Residual Distribution (Test Set)", fontsize=13)
    ax.legend(fontsize=10)

    # Annotation: mean, std, outlier count
    annotation_text = (
        f"Mean = {mean_resid:+.4f}\n"
        f"Std  = {std_resid:.4f}\n"
        f"|residual| > 0.5: {n_outliers} ILs"
    )
    ax.text(0.97, 0.95, annotation_text,
            transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.8))

    ax.grid(True, linestyle=":", alpha=0.5)

    out_path = os.path.join(FIGURES_DIR, "residual_distribution.png")
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_residual_distribution] Saved → {out_path}")
    print(f"  Mean residual = {mean_resid:+.4f} | Std = {std_resid:.4f}")
    if n_outliers > 0:
        print(f"  ⚠  {n_outliers} test ILs have |residual| > 0.5 log10 — check for structural novelty")


def main():
    """Load results, generate all three figures, report paths."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    predictions_df, importance_df, best_model_name = load_results()

    print("\n=== FIGURE 1: Predicted vs Actual ===")
    plot_predicted_vs_actual(predictions_df, best_model_name)

    print("\n=== FIGURE 2: Feature Importance ===")
    plot_feature_importance(importance_df, best_model_name)

    print("\n=== FIGURE 3: Residual Distribution ===")
    plot_residual_distribution(predictions_df, best_model_name)

    print("\n✓ All figures saved to figures/")
    print("  figures/predicted_vs_actual.png")
    print("  figures/feature_importance.png")
    print("  figures/residual_distribution.png")


if __name__ == "__main__":
    main()
