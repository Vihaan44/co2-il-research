"""
train_stacked_model.py
----------------------
PURPOSE: Build a stacking ensemble that blends Random Forest and tuned XGBoost
         predictions using a Ridge regression meta-learner.

WHY STACKING:
  RF and XGBoost make structurally different errors:
    - RF averages many shallow decisions -> smooth, slightly underfit predictions
    - XGBoost builds sequential corrections -> sharper but occasionally overfit
  A meta-learner trained on their combined out-of-fold (OOF) predictions learns
  when to trust each model, typically gaining 0.02-0.05 R² over either alone.

HOW STACKING WORKS (judge-friendly explanation):
  Step 1 — OOF predictions:
    For each CV fold, train RF and XGB on the fold's training rows, then predict
    on the fold's validation rows. Do this for all folds so every training row
    gets a prediction from a model that never saw it. This gives us a "meta-feature"
    matrix: [rf_pred, xgb_pred] for every training row.

  Step 2 — Meta-learner:
    Train a Ridge regression on [rf_pred, xgb_pred] -> y_true. Ridge learns the
    optimal linear blend weight for each base model. We use Ridge (not plain
    linear regression) because with only 2 meta-features it makes no practical
    difference, but it's more numerically stable and the regularization is
    explicit and explainable.

  Step 3 — Final base models:
    Refit RF and XGB on the FULL training set (not fold subsets). These are the
    models used at inference time.

  Step 4 — Inference:
    For a new IL: get RF prediction, get XGB prediction, feed both into Ridge
    -> stacked prediction.

WHY GroupKFold FOR OOF:
  Same reason as CV: if the same IL appears in both the fold's train and val,
  the OOF predictions are optimistically biased and the meta-learner learns on
  leaked data. GroupKFold by il_smiles prevents this.

BASE MODEL HYPERPARAMETERS:
  RF:  best params from Optuna v2 tuning (n_est=515, max_feat=0.5, min_leaf=7)
  XGB: best params from Optuna v2 tuning (n_est=440, lr=0.075, depth=5,
       mcw=10, alpha=0.646, subsample=0.776, colsample=0.460)
  These produced test R²=0.632 for XGB alone; stacking should improve on this.

INPUTS:
  data/processed/train_set.csv   (from build_dataset.py)
  data/processed/test_set.csv    (from build_dataset.py)

OUTPUTS:
  models/stacked_model.pkl            -- full stacking bundle (rf, xgb, ridge, feature_cols)
  results/stacked_model_performance.csv
  results/stacked_model_predictions.csv
  results/stacking_weights.csv        -- ridge coefficients (how much each base model contributes)

Run from project root:
    python src/train_stacked_model.py
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle
from xgboost import XGBRegressor

# -- Constants -----------------------------------------------------------------
TRAIN_CSV   = os.path.join("data", "processed", "train_set.csv")
TEST_CSV    = os.path.join("data", "processed", "test_set.csv")
MODEL_DIR   = "models"
RESULTS_DIR = "results"
STACKED_MODEL_PATH = os.path.join(MODEL_DIR, "stacked_model.pkl")

TARGET_COL         = "log_x2_CO2"
CONDITION_FEATURES = ["T_K", "P_kPa"]
CV_FOLDS           = 5     # folds for OOF generation
RANDOM_SEED        = 42

# RF best params from Optuna v2 tuning
RF_N_ESTIMATORS     = 515
RF_MAX_FEATURES     = 0.5
RF_MIN_SAMPLES_LEAF = 7
RF_MAX_DEPTH        = None  # unlimited depth was best

# XGB best params from Optuna v2 tuning (produced test R²=0.632 alone)
XGB_N_ESTIMATORS   = 440
XGB_LEARNING_RATE  = 0.07481496458680137
XGB_MAX_DEPTH      = 5
XGB_SUBSAMPLE      = 0.7760131200377596
XGB_COLSAMPLE      = 0.45987026313463525
XGB_REG_ALPHA      = 0.6457288433886444   # L1 regularization
XGB_REG_LAMBDA     = 3.5815949001230535e-07  # L2 regularization
XGB_MIN_CHILD_WEIGHT = 10  # key regularizer: min instances per leaf

# Ridge meta-learner regularization -- small alpha since we only have 2 meta-features
RIDGE_ALPHA = 1.0


def load_split(path: str, label: str) -> tuple:
    """Load train or test CSV; return X, y, il_smiles, feature_cols."""
    df = pd.read_csv(path)
    df = df.dropna(subset=CONDITION_FEATURES).copy()

    molecular_cols = [c for c in df.columns if c.startswith("cat_") or c.startswith("an_")]
    feature_cols   = molecular_cols + CONDITION_FEATURES

    X      = df[feature_cols].values
    y      = df[TARGET_COL].values
    smiles = df["il_smiles"]

    print(f"[load] {label}: {len(df)} rows, {smiles.nunique()} unique ILs, "
          f"{len(feature_cols)} features", flush=True)
    return X, y, smiles, feature_cols


def make_base_models() -> tuple:
    """
    Instantiate RF and XGB with Optuna v2 best hyperparameters.
    Returns (rf_model, xgb_model) -- unfitted.
    """
    rf = RandomForestRegressor(
        n_estimators     = RF_N_ESTIMATORS,
        max_features     = RF_MAX_FEATURES,
        min_samples_leaf = RF_MIN_SAMPLES_LEAF,
        max_depth        = RF_MAX_DEPTH,
        random_state     = RANDOM_SEED,
        n_jobs           = -1,
    )
    xgb = XGBRegressor(
        n_estimators     = XGB_N_ESTIMATORS,
        learning_rate    = XGB_LEARNING_RATE,
        max_depth        = XGB_MAX_DEPTH,
        subsample        = XGB_SUBSAMPLE,
        colsample_bytree = XGB_COLSAMPLE,
        reg_alpha        = XGB_REG_ALPHA,
        reg_lambda       = XGB_REG_LAMBDA,
        min_child_weight = XGB_MIN_CHILD_WEIGHT,
        random_state     = RANDOM_SEED,
        n_jobs           = 2,       # limit cores to keep memory under control
        verbosity        = 0,
        tree_method      = "hist",
    )
    return rf, xgb


def generate_oof_predictions(X_train: np.ndarray, y_train: np.ndarray,
                              il_smiles_train: pd.Series) -> np.ndarray:
    """
    Generate out-of-fold (OOF) predictions for every training row using GroupKFold.

    For each fold:
      - Train RF and XGB on the fold's training rows (ILs not in this fold's val set)
      - Predict on the fold's val rows (ILs never seen by this fold's models)

    After all folds, every training row has a [rf_pred, xgb_pred] pair from a
    model that never saw its IL during training. This is the meta-feature matrix
    that the Ridge meta-learner trains on.

    Returns: oof_meta_X shape (n_train_rows, 2) -- columns are [rf_oof, xgb_oof]
    """
    groups     = il_smiles_train.values
    gkf        = GroupKFold(n_splits=CV_FOLDS)
    oof_rf     = np.zeros(len(y_train))   # OOF predictions from RF
    oof_xgb    = np.zeros(len(y_train))   # OOF predictions from XGB

    print(f"\n[oof] Generating OOF predictions with {CV_FOLDS}-fold GroupKFold...",
          flush=True)

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        # Confirm no IL appears in both train and val for this fold
        train_ils = set(groups[train_idx])
        val_ils   = set(groups[val_idx])
        assert len(train_ils & val_ils) == 0, f"IL overlap in fold {fold_idx}!"

        rf_fold, xgb_fold = make_base_models()

        rf_fold.fit(X_train[train_idx], y_train[train_idx])
        xgb_fold.fit(X_train[train_idx], y_train[train_idx])

        oof_rf[val_idx]  = rf_fold.predict(X_train[val_idx])
        oof_xgb[val_idx] = xgb_fold.predict(X_train[val_idx])

        fold_r2_rf  = r2_score(y_train[val_idx], oof_rf[val_idx])
        fold_r2_xgb = r2_score(y_train[val_idx], oof_xgb[val_idx])
        n_val_ils   = len(val_ils)
        print(f"  Fold {fold_idx+1}/{CV_FOLDS}: {n_val_ils} val ILs | "
              f"RF R²={fold_r2_rf:.3f}, XGB R²={fold_r2_xgb:.3f}", flush=True)

    # OOF R² -- how well each base model predicts across all held-out folds
    oof_r2_rf  = r2_score(y_train, oof_rf)
    oof_r2_xgb = r2_score(y_train, oof_xgb)
    print(f"\n[oof] OOF R² (all folds combined): RF={oof_r2_rf:.4f}, XGB={oof_r2_xgb:.4f}",
          flush=True)

    # Stack OOF predictions as meta-features: shape (n_rows, 2)
    oof_meta_X = np.column_stack([oof_rf, oof_xgb])
    return oof_meta_X


def train_meta_learner(oof_meta_X: np.ndarray, y_train: np.ndarray) -> Ridge:
    """
    Train Ridge regression on OOF meta-features [rf_pred, xgb_pred] -> y_true.

    Ridge learns optimal blend weights. Coefficients > 0 for both models means
    both contribute; a coefficient near 0 means that model adds little.
    We use Ridge over plain LinearRegression for numerical stability, even though
    with only 2 meta-features the regularization effect is minimal.
    """
    meta_learner = Ridge(alpha=RIDGE_ALPHA)
    meta_learner.fit(oof_meta_X, y_train)

    print(f"\n[meta] Ridge meta-learner trained:", flush=True)
    print(f"  RF  weight : {meta_learner.coef_[0]:.4f}", flush=True)
    print(f"  XGB weight : {meta_learner.coef_[1]:.4f}", flush=True)
    print(f"  Intercept  : {meta_learner.intercept_:.4f}", flush=True)

    # Warn if one model gets near-zero weight -- stacking isn't helping
    if abs(meta_learner.coef_[0]) < 0.05:
        print("  WARNING: RF weight near zero -- RF may not be contributing.", flush=True)
    if abs(meta_learner.coef_[1]) < 0.05:
        print("  WARNING: XGB weight near zero -- XGB may not be contributing.", flush=True)

    return meta_learner


def evaluate_stacked(rf_final, xgb_final, meta_learner: Ridge,
                     X_test: np.ndarray, y_test: np.ndarray,
                     smiles_test: pd.Series) -> tuple:
    """
    Evaluate the stacked model on the held-out test set.
    Inference: rf_pred + xgb_pred -> Ridge -> stacked_pred.
    Also reports base model test performance individually for comparison.
    """
    # Base model test predictions (on full train-fitted models)
    rf_test_pred  = rf_final.predict(X_test)
    xgb_test_pred = xgb_final.predict(X_test)

    # Stacked prediction: meta-learner blends both
    test_meta_X    = np.column_stack([rf_test_pred, xgb_test_pred])
    stacked_pred   = meta_learner.predict(test_meta_X)

    def metrics(y_true, y_pred, name):
        """Compute and print RMSE, R², MAE for a set of predictions."""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)
        mae  = mean_absolute_error(y_true, y_pred)
        rmse_x2 = np.sqrt(mean_squared_error(10**y_true, 10**y_pred))
        print(f"\n[eval] {name}:", flush=True)
        print(f"  RMSE (log10 x2) : {rmse:.4f}", flush=True)
        print(f"  R²              : {r2:.4f}", flush=True)
        print(f"  MAE  (log10 x2) : {mae:.4f}", flush=True)
        print(f"  RMSE (x2 units) : {rmse_x2:.6f}", flush=True)
        return {"model": name, "test_rmse_log": rmse, "test_r2": r2,
                "test_mae_log": mae, "test_rmse_x2": rmse_x2}

    perf_rf      = metrics(y_test, rf_test_pred,  "RF (final, full train)")
    perf_xgb     = metrics(y_test, xgb_test_pred, "XGB (final, full train)")
    perf_stacked = metrics(y_test, stacked_pred,  "Stacked (RF + XGB + Ridge)")

    predictions_df = pd.DataFrame({
        "il_smiles":      smiles_test.values,
        "y_true_log":     y_test,
        "rf_pred_log":    rf_test_pred,
        "xgb_pred_log":   xgb_test_pred,
        "stacked_pred_log": stacked_pred,
        "residual_log":   stacked_pred - y_test,
    })

    perf_df = pd.DataFrame([perf_rf, perf_xgb, perf_stacked])
    return perf_df, predictions_df


def main():
    """
    Full stacking pipeline:
    1. Load data
    2. Generate OOF predictions (GroupKFold, no IL leakage)
    3. Train Ridge meta-learner on OOF predictions
    4. Refit base models on full training set
    5. Evaluate stacked model on test set
    6. Save stacked model bundle
    """
    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=== STACKING ENSEMBLE: RF + XGB + Ridge ===\n", flush=True)

    # Step 1: Load data
    X_train, y_train, smiles_train, feature_cols = load_split(TRAIN_CSV, "Train")
    X_test,  y_test,  smiles_test,  _            = load_split(TEST_CSV,  "Test")

    # Step 2: OOF predictions for meta-learner training
    oof_meta_X = generate_oof_predictions(X_train, y_train, smiles_train)

    # Step 3: Train Ridge meta-learner on OOF predictions
    meta_learner = train_meta_learner(oof_meta_X, y_train)

    # Step 4: Refit base models on FULL training set (these are used at inference)
    print("\n[main] Fitting final RF on full training set...", flush=True)
    rf_final, xgb_final = make_base_models()
    rf_final.fit(X_train, y_train)
    print("[main] Fitting final XGB on full training set...", flush=True)
    xgb_final.fit(X_train, y_train)

    # Step 5: Evaluate on held-out test set
    print("\n=== TEST SET EVALUATION ===", flush=True)
    perf_df, predictions_df = evaluate_stacked(
        rf_final, xgb_final, meta_learner, X_test, y_test, smiles_test
    )

    print("\n=== SUMMARY ===", flush=True)
    print(perf_df[["model", "test_r2", "test_rmse_log", "test_mae_log"]].to_string(index=False),
          flush=True)

    # Report stacking gain vs best single model
    best_single_r2 = perf_df[perf_df["model"] != "Stacked (RF + XGB + Ridge)"]["test_r2"].max()
    stacked_r2     = perf_df[perf_df["model"] == "Stacked (RF + XGB + Ridge)"]["test_r2"].values[0]
    print(f"\n  Stacking gain vs best single model: {stacked_r2 - best_single_r2:+.4f} R²",
          flush=True)

    # Step 6: Save stacked model bundle
    stacking_weights = pd.DataFrame({
        "base_model": ["RF", "XGB"],
        "ridge_weight": meta_learner.coef_,
        "intercept": [meta_learner.intercept_, ""],
    })

    bundle = {
        "rf":           rf_final,
        "xgb":          xgb_final,
        "meta_learner": meta_learner,
        "feature_cols": feature_cols,
        "model_name":   "Stacked_RF_XGB_Ridge",
    }
    joblib.dump(bundle, STACKED_MODEL_PATH)

    perf_df.to_csv(
        os.path.join(RESULTS_DIR, "stacked_model_performance.csv"), index=False)
    predictions_df.to_csv(
        os.path.join(RESULTS_DIR, "stacked_model_predictions.csv"), index=False)
    stacking_weights.to_csv(
        os.path.join(RESULTS_DIR, "stacking_weights.csv"), index=False)

    print("\n[main] Saved:", flush=True)
    print("  models/stacked_model.pkl", flush=True)
    print("  results/stacked_model_performance.csv", flush=True)
    print("  results/stacked_model_predictions.csv", flush=True)
    print("  results/stacking_weights.csv", flush=True)
    print("[main] DONE.", flush=True)


if __name__ == "__main__":
    main()
