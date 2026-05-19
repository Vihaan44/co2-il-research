"""
eval_tuned_model.py
-------------------
PURPOSE: Evaluate the Optuna-tuned model (forward_model_tuned.pkl) on the
         held-out test set and compare against the untuned baseline.

WHY A SEPARATE SCRIPT:
  tune_hyperparameters.py only uses CV -- it never touches the test set (by
  design: looking at test during tuning would leak information into
  hyperparameter selection). This script is the clean post-tuning eval step.

WHAT TO LOOK FOR:
  1. Test R² vs untuned XGBoost (0.4244) -- should be higher.
  2. CV RMSE (0.3409) vs test RMSE -- gap should be smaller than untuned
     (CV R²=0.675 vs test R²=0.424). min_child_weight=10 should close it.

INPUTS:
  models/forward_model_tuned.pkl       (from tune_hyperparameters.py)
  data/processed/test_set.csv          (from build_dataset.py)
  results/model_performance.csv        (untuned baseline, for comparison)

OUTPUTS:
  results/tuned_model_performance.csv        -- RMSE, R², MAE vs baseline
  results/tuned_model_test_predictions.csv   -- per-row predictions + residuals

Run from project root:
    python src/eval_tuned_model.py
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# -- Constants -----------------------------------------------------------------
TUNED_MODEL_PATH   = os.path.join("models", "forward_model_tuned.pkl")
TEST_CSV           = os.path.join("data", "processed", "test_set.csv")
BASELINE_PERF_CSV  = os.path.join("results", "model_performance.csv")
RESULTS_DIR        = "results"

TARGET_COL         = "log_x2_CO2"
CONDITION_FEATURES = ["T_K", "P_kPa"]


def load_tuned_model() -> tuple:
    """Load the tuned model bundle; return model, feature_cols, model_name."""
    if not os.path.exists(TUNED_MODEL_PATH):
        raise FileNotFoundError(
            f"Tuned model not found at {TUNED_MODEL_PATH}. "
            "Run tune_hyperparameters.py first."
        )
    bundle = joblib.load(TUNED_MODEL_PATH)
    print(f"[load_tuned_model] Loaded: {bundle['model_name']}", flush=True)
    return bundle["model"], bundle["feature_cols"], bundle["model_name"]


def load_test_set(feature_cols: list) -> tuple:
    """
    Load test CSV, align columns to model's expected feature order.
    Drops NaN T/P rows defensively; raises if feature mismatch.
    """
    df = pd.read_csv(TEST_CSV)
    df = df.dropna(subset=CONDITION_FEATURES).copy()

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Test set missing {len(missing)} expected features: {missing[:5]}..."
        )

    X      = df[feature_cols].values
    y      = df[TARGET_COL].values
    smiles = df["il_smiles"]

    print(f"[load_test_set] {len(df)} rows, {smiles.nunique()} unique ILs", flush=True)
    print(f"[load_test_set] Target range: [{y.min():.3f}, {y.max():.3f}]", flush=True)
    return X, y, smiles


def evaluate(model, X: np.ndarray, y: np.ndarray,
             smiles: pd.Series, name: str) -> tuple:
    """
    Predict on test set; compute RMSE, R², MAE in log10 and mole fraction units.
    Also prints per-IL MAE to show which IL families are hardest to predict.
    """
    y_pred   = model.predict(X)
    rmse_log = np.sqrt(mean_squared_error(y, y_pred))
    r2       = r2_score(y, y_pred)
    mae_log  = mean_absolute_error(y, y_pred)
    # Back-transform: x2 = 10^(log_x2) for interpretable mole fraction error
    rmse_x2  = np.sqrt(mean_squared_error(10**y, 10**y_pred))

    print(f"\n[evaluate] {name} -- HELD-OUT TEST SET:", flush=True)
    print(f"  RMSE (log10 x2) : {rmse_log:.4f}", flush=True)
    print(f"  R²              : {r2:.4f}", flush=True)
    print(f"  MAE  (log10 x2) : {mae_log:.4f}", flush=True)
    print(f"  RMSE (x2 units) : {rmse_x2:.6f}   <- mole fraction error", flush=True)

    preds_df = pd.DataFrame({
        "il_smiles":    smiles.values,
        "y_true_log":   y,
        "y_pred_log":   y_pred,
        "residual_log": y_pred - y,
        "x2_true":      10**y,
        "x2_pred":      10**y_pred,
    })

    # Per-IL mean absolute error -- highlights structurally hard IL families
    per_il_mae = (
        preds_df.groupby("il_smiles")["residual_log"]
        .apply(lambda r: r.abs().mean())
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"residual_log": "mean_abs_residual"})
    )
    print(f"\n[evaluate] Top 5 hardest ILs (highest MAE):", flush=True)
    print(per_il_mae.head(5).to_string(index=False), flush=True)

    perf = {
        "model":         name,
        "test_rmse_log": rmse_log,
        "test_r2":       r2,
        "test_mae_log":  mae_log,
        "test_rmse_x2":  rmse_x2,
    }
    return perf, preds_df


def compare_to_baseline(tuned: dict) -> None:
    """Load untuned baseline from CSV and print side-by-side comparison."""
    if not os.path.exists(BASELINE_PERF_CSV):
        print("\n[compare] Baseline CSV not found -- skipping comparison.", flush=True)
        return

    base = pd.read_csv(BASELINE_PERF_CSV)
    best = base.loc[base["test_r2"].idxmax()]  # best untuned model by R²

    print("\n=== TUNED vs UNTUNED COMPARISON ===", flush=True)
    print(f"  {'Metric':<20} {'Untuned ' + best['model']:<28} {'Tuned ' + tuned['model']:<28}",
          flush=True)
    print(f"  {'Test R²':<20} {best['test_r2']:<28.4f} {tuned['test_r2']:<28.4f}",
          flush=True)
    print(f"  {'RMSE (log10 x2)':<20} {best['test_rmse_log']:<28.4f} {tuned['test_rmse_log']:<28.4f}",
          flush=True)
    print(f"  {'MAE (log10 x2)':<20} {best['test_mae_log']:<28.4f} {tuned['test_mae_log']:<28.4f}",
          flush=True)

    r2_delta   = tuned["test_r2"]       - best["test_r2"]
    rmse_delta = tuned["test_rmse_log"] - best["test_rmse_log"]
    print(f"\n  R² change:   {r2_delta:+.4f}  ({'improvement' if r2_delta > 0 else 'regression'})",
          flush=True)
    print(f"  RMSE change: {rmse_delta:+.4f}  ({'improvement' if rmse_delta < 0 else 'regression'})",
          flush=True)


def main():
    """Load tuned model -> evaluate on test set -> compare to baseline -> save."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model, feature_cols, name = load_tuned_model()
    X, y, smiles              = load_test_set(feature_cols)
    perf, preds               = evaluate(model, X, y, smiles, name)

    compare_to_baseline(perf)

    pd.DataFrame([perf]).to_csv(
        os.path.join(RESULTS_DIR, "tuned_model_performance.csv"), index=False)
    preds.to_csv(
        os.path.join(RESULTS_DIR, "tuned_model_test_predictions.csv"), index=False)

    print("\n[main] Saved:", flush=True)
    print("  results/tuned_model_performance.csv", flush=True)
    print("  results/tuned_model_test_predictions.csv", flush=True)
    print("[main] DONE.", flush=True)


if __name__ == "__main__":
    main()
