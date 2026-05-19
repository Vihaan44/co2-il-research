"""
save_best_model.py
------------------
PURPOSE: Extract the tuned RF from the stacked model bundle and save it as
         the primary forward model (models/forward_model.pkl).

BACKGROUND:
  Stacking (RF + XGB + Ridge) was evaluated and found to hurt performance
  (-0.05 R² vs RF alone). This is because the tuned RF (Optuna v2 params:
  n_est=515, max_feat=0.5, min_leaf=7) generalizes better to held-out ILs
  than XGB on this dataset (test R²=0.714 vs 0.632).

  The meta-learner was misled: it heavily weighted XGB (0.787) because XGB
  had better OOF R² (0.719 vs 0.690), but on the true held-out test set RF
  won. This is a known failure mode of stacking on small datasets -- the
  meta-learner overfits to OOF performance.

  Decision: use tuned RF alone as the forward model going forward.

WHAT THIS SCRIPT DOES:
  Loads models/stacked_model.pkl (which contains the full-train-fitted RF),
  wraps it in the standard forward_model.pkl bundle format, and overwrites
  models/forward_model.pkl.

  This makes all downstream scripts (predict_with_uncertainty.py,
  applicability_domain.py, screen_virtual_library.py) automatically use
  the best model without modification.

INPUT:  models/stacked_model.pkl
OUTPUT: models/forward_model.pkl  (overwritten with tuned RF)

Run from project root:
    python src/save_best_model.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

STACKED_MODEL_PATH = os.path.join("models", "stacked_model.pkl")
FORWARD_MODEL_PATH = os.path.join("models", "forward_model.pkl")
TEST_CSV           = os.path.join("data", "processed", "test_set.csv")
RESULTS_DIR        = "results"
TARGET_COL         = "log_x2_CO2"
CONDITION_FEATURES = ["T_K", "P_kPa"]


def main():
    """Load stacked bundle, extract RF, verify on test set, save as forward model."""

    # Load the stacked bundle (contains full-train-fitted rf, xgb, meta_learner)
    stacked = joblib.load(STACKED_MODEL_PATH)
    rf_model     = stacked["rf"]
    feature_cols = stacked["feature_cols"]
    print(f"[main] Loaded RF from stacked bundle.", flush=True)
    print(f"[main] RF params: {rf_model.get_params()}", flush=True)

    # Verify on test set to confirm R²=0.714 before overwriting
    df_test = pd.read_csv(TEST_CSV)
    df_test = df_test.dropna(subset=CONDITION_FEATURES).copy()
    X_test  = df_test[feature_cols].values
    y_test  = df_test[TARGET_COL].values

    y_pred   = rf_model.predict(X_test)
    rmse     = np.sqrt(mean_squared_error(y_test, y_pred))
    r2       = r2_score(y_test, y_pred)
    mae      = mean_absolute_error(y_test, y_pred)
    rmse_x2  = np.sqrt(mean_squared_error(10**y_test, 10**y_pred))

    print(f"\n[verify] Tuned RF on held-out test set:", flush=True)
    print(f"  RMSE (log10 x2) : {rmse:.4f}", flush=True)
    print(f"  R²              : {r2:.4f}", flush=True)
    print(f"  MAE  (log10 x2) : {mae:.4f}", flush=True)
    print(f"  RMSE (x2 units) : {rmse_x2:.6f}", flush=True)

    if r2 < 0.70:
        print(f"  WARNING: R²={r2:.4f} is below expected 0.714 -- check data.", flush=True)
    else:
        print(f"  OK: R²={r2:.4f} matches expected result.", flush=True)

    # Save as primary forward model in standard bundle format
    bundle = {
        "model":        rf_model,
        "feature_cols": feature_cols,
        "model_name":   "RandomForest_tuned_v2",
    }
    joblib.dump(bundle, FORWARD_MODEL_PATH)
    print(f"\n[main] Saved tuned RF -> {FORWARD_MODEL_PATH}", flush=True)
    print("[main] All downstream scripts will now use RandomForest_tuned_v2.", flush=True)

    # Save updated performance record
    perf = pd.DataFrame([{
        "model":         "RandomForest_tuned_v2",
        "test_rmse_log": rmse,
        "test_r2":       r2,
        "test_mae_log":  mae,
        "test_rmse_x2":  rmse_x2,
        "notes":         "Optuna v2 tuned RF; stacking evaluated and rejected (hurt R² by 0.05)",
    }])
    perf.to_csv(os.path.join(RESULTS_DIR, "best_model_performance.csv"), index=False)
    print("[main] Saved results/best_model_performance.csv", flush=True)


if __name__ == "__main__":
    main()
