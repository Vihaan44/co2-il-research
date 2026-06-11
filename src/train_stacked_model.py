"""
train_stacked_model.py
----------------------
PURPOSE: Build a stacking ensemble that blends Random Forest, tuned XGBoost,
         and CatBoost predictions using a Ridge regression meta-learner.

WHY STACKING:
  Each base model makes structurally different errors:
    - RF averages many shallow decisions -> smooth, slightly underfit predictions
    - XGBoost builds sequential corrections -> sharper but occasionally overfit
    - CatBoost uses symmetric (oblivious) trees -> different regularization bias
  A meta-learner trained on their combined out-of-fold (OOF) predictions learns
  when to trust each model, typically gaining 0.02-0.05 R² over any single model.

HOW STACKING WORKS (judge-friendly explanation):
  Step 1 — OOF predictions:
    For each CV fold, train RF, XGB, and CatBoost on the fold's training rows,
    then predict on the fold's validation rows (ILs the models never saw).
    After all folds, every training row has a [rf_pred, xgb_pred, cat_pred] triple
    from models that never saw its IL. This is the meta-feature matrix.

  Step 2 — Meta-learner:
    Train a Ridge regression on [rf_pred, xgb_pred, cat_pred] -> y_true.
    Ridge learns the optimal linear blend weight for each base model.

  Step 3 — Final base models:
    Refit RF, XGB, and CatBoost on the FULL training set. These are used at
    inference time.

  Step 4 — Inference:
    For a new IL: get RF, XGB, CatBoost predictions, feed all three into Ridge
    -> stacked prediction.

WHY GroupKFold FOR OOF:
  Plain KFold lets the same IL appear in both train and val folds, inflating R²
  to ~0.95 due to leakage. GroupKFold by il_smiles prevents this.

BASE MODEL HYPERPARAMETERS:
  RF:       Optuna v2 best params (n_est=515, max_feat=0.5, min_leaf=7)
  XGB:      Optuna v3 best params (n_est=703, lr=0.014, min_child_weight=1,
            gamma=0.21) -- v3 corrects v2's suppression of T_K feature importance
  CatBoost: Default params (not yet Optuna-tuned); ablation showed R²≈0.726
            with defaults, already best solo model.

INPUTS:
  data/processed/train_set.csv   (from build_dataset.py)
  data/processed/test_set.csv    (from build_dataset.py)

OUTPUTS:
  models/stacked_model.pkl                 -- full stacking bundle
  results/stacked_model_performance.csv
  results/stacked_model_predictions.csv
  results/stacking_weights.csv             -- ridge coefficients per base model

Run from project root:
    nohup python src/train_stacked_model.py > logs/stacked_model.log 2>&1 &
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

# -- Constants -----------------------------------------------------------------
TRAIN_CSV   = os.path.join("data", "processed", "train_set.csv")
TEST_CSV    = os.path.join("data", "processed", "test_set.csv")
MODEL_DIR   = "models"
RESULTS_DIR = "results"
STACKED_MODEL_PATH = os.path.join(MODEL_DIR, "stacked_model.pkl")

TARGET_COL         = "log_x2_CO2"
CONDITION_FEATURES = ["T_K", "P_kPa"]
CV_FOLDS           = 5       # GroupKFold folds for OOF generation
RANDOM_SEED        = 42

# RF: Optuna v2 best params
RF_N_ESTIMATORS     = 515
RF_MAX_FEATURES     = 0.5
RF_MIN_SAMPLES_LEAF = 7
RF_MAX_DEPTH        = None   # unlimited depth was best in v2

# XGB: Optuna v3 best params (corrected min_child_weight -- v2's value of 10
# was suppressing T_K from top feature importances, which is physically wrong)
XGB_N_ESTIMATORS     = 703
XGB_LEARNING_RATE    = 0.014
XGB_MAX_DEPTH        = 5
XGB_MIN_CHILD_WEIGHT = 1      # v3 key change: allows T_K to contribute properly
XGB_GAMMA            = 0.21   # minimum loss reduction to make a split
XGB_SUBSAMPLE        = 0.776
XGB_COLSAMPLE        = 0.460
XGB_REG_ALPHA        = 1.2    # L1 regularization (stronger than v2)
XGB_REG_LAMBDA       = 0.8    # L2 regularization

# CatBoost: ablation defaults (R²≈0.726, best solo model seen so far)
# Not yet Optuna-tuned -- tuning CatBoost is a future improvement
CATBOOST_ITERATIONS   = 500
CATBOOST_LEARNING_RATE = 0.05
CATBOOST_DEPTH        = 6

# Ridge meta-learner: small alpha, only 3 meta-features so regularization is minimal
RIDGE_ALPHA = 1.0


def load_split(path: str, label: str) -> tuple:
    """Load train or test CSV; return X, y, il_smiles groups, feature column names."""
    df = pd.read_csv(path)
    df = df.dropna(subset=CONDITION_FEATURES).copy()

    molecular_cols = [c for c in df.columns if c.startswith("cat_") or c.startswith("an_")]
    feature_cols   = molecular_cols + CONDITION_FEATURES

    X      = df[feature_cols].values.astype(float)
    y      = df[TARGET_COL].values
    smiles = df["il_smiles"]

    print(f"[load] {label}: {len(df)} rows, {smiles.nunique()} unique ILs, "
          f"{len(feature_cols)} features", flush=True)
    return X, y, smiles, feature_cols


def impute_nans(X_train: np.ndarray, X_val: np.ndarray) -> tuple:
    """
    Impute NaN values using median of training fold only.
    Fit only on train to prevent data leakage into validation fold.
    NaNs can appear in RDKit Gasteiger charge descriptors for unusual SMILES.
    """
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)   # fit on train only
    X_val_imp   = imputer.transform(X_val)          # apply same imputer to val
    return X_train_imp, X_val_imp, imputer


def make_rf() -> RandomForestRegressor:
    """Instantiate RF with v2 Optuna best hyperparameters."""
    return RandomForestRegressor(
        n_estimators     = RF_N_ESTIMATORS,
        max_features     = RF_MAX_FEATURES,
        min_samples_leaf = RF_MIN_SAMPLES_LEAF,
        max_depth        = RF_MAX_DEPTH,
        random_state     = RANDOM_SEED,
        n_jobs           = -1,
    )


def make_xgb() -> XGBRegressor:
    """Instantiate XGB with v3 Optuna best hyperparameters."""
    return XGBRegressor(
        n_estimators     = XGB_N_ESTIMATORS,
        learning_rate    = XGB_LEARNING_RATE,
        max_depth        = XGB_MAX_DEPTH,
        min_child_weight = XGB_MIN_CHILD_WEIGHT,
        gamma            = XGB_GAMMA,
        subsample        = XGB_SUBSAMPLE,
        colsample_bytree = XGB_COLSAMPLE,
        reg_alpha        = XGB_REG_ALPHA,
        reg_lambda       = XGB_REG_LAMBDA,
        random_state     = RANDOM_SEED,
        n_jobs           = 2,
        verbosity        = 0,
        tree_method      = "hist",
    )


def make_catboost():
    """
    Instantiate CatBoost with ablation-confirmed default params.
    CatBoost's symmetric trees gave R²≈0.726 vs XGB's 0.689 in ablation --
    it generalizes better on noisy IL data due to stronger implicit regularization.
    """
    try:
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            iterations    = CATBOOST_ITERATIONS,
            learning_rate = CATBOOST_LEARNING_RATE,
            depth         = CATBOOST_DEPTH,
            random_seed   = RANDOM_SEED,
            verbose       = 0,   # suppress per-iteration output
        )
    except ImportError:
        # CatBoost not installed -- ensemble degrades to RF + XGB only
        print("[WARNING] catboost not installed. Ensemble will use RF + XGB only.", flush=True)
        print("          Install with: pip install catboost", flush=True)
        return None


def generate_oof_predictions(X_train: np.ndarray, y_train: np.ndarray,
                              il_smiles_train: pd.Series,
                              catboost_model) -> np.ndarray:
    """
    Generate out-of-fold predictions for every training row using GroupKFold.

    For each fold:
      - Impute NaNs (fitted on train fold only to prevent leakage)
      - Train RF, XGB, CatBoost on the fold's training ILs
      - Predict on the fold's val ILs (never seen during this fold's training)

    After all folds, every row has a prediction from a model that never saw its IL.
    This is the meta-feature matrix the Ridge meta-learner trains on.

    Returns: oof_meta_X shape (n_train_rows, 2 or 3) -- columns [rf, xgb, catboost]
    """
    use_catboost = catboost_model is not None
    groups       = il_smiles_train.values
    gkf          = GroupKFold(n_splits=CV_FOLDS)

    oof_rf  = np.zeros(len(y_train))
    oof_xgb = np.zeros(len(y_train))
    oof_cat = np.zeros(len(y_train)) if use_catboost else None

    print(f"\n[oof] Generating OOF predictions with {CV_FOLDS}-fold GroupKFold...",
          flush=True)

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        # Confirm no IL leaks across train/val split for this fold
        assert len(set(groups[train_idx]) & set(groups[val_idx])) == 0, \
            f"IL overlap detected in fold {fold_idx}!"

        # Impute NaNs -- fit on train fold only
        X_tr, X_val, _ = impute_nans(X_train[train_idx], X_train[val_idx])
        y_tr = y_train[train_idx]

        # Train each base model on this fold
        rf_fold = make_rf()
        rf_fold.fit(X_tr, y_tr)
        oof_rf[val_idx] = rf_fold.predict(X_val)

        xgb_fold = make_xgb()
        xgb_fold.fit(X_tr, y_tr)
        oof_xgb[val_idx] = xgb_fold.predict(X_val)

        if use_catboost:
            from sklearn.base import clone
            cat_fold = clone(catboost_model)
            cat_fold.fit(X_tr, y_tr)
            oof_cat[val_idx] = cat_fold.predict(X_val)

        # Print per-fold diagnostic
        n_val_ils = len(set(groups[val_idx]))
        fold_r2_rf  = r2_score(y_train[val_idx], oof_rf[val_idx])
        fold_r2_xgb = r2_score(y_train[val_idx], oof_xgb[val_idx])
        if use_catboost:
            fold_r2_cat = r2_score(y_train[val_idx], oof_cat[val_idx])
            print(f"  Fold {fold_idx+1}/{CV_FOLDS}: {n_val_ils} val ILs | "
                  f"RF={fold_r2_rf:.3f}, XGB={fold_r2_xgb:.3f}, CB={fold_r2_cat:.3f}",
                  flush=True)
        else:
            print(f"  Fold {fold_idx+1}/{CV_FOLDS}: {n_val_ils} val ILs | "
                  f"RF={fold_r2_rf:.3f}, XGB={fold_r2_xgb:.3f}", flush=True)

    # OOF R² across all folds combined
    print(f"\n[oof] Combined OOF R²: RF={r2_score(y_train, oof_rf):.4f}, "
          f"XGB={r2_score(y_train, oof_xgb):.4f}", flush=True)
    if use_catboost:
        print(f"[oof]                   CB={r2_score(y_train, oof_cat):.4f}", flush=True)

    # Stack OOF predictions as meta-features
    if use_catboost:
        return np.column_stack([oof_rf, oof_xgb, oof_cat])
    else:
        return np.column_stack([oof_rf, oof_xgb])


def train_meta_learner(oof_meta_X: np.ndarray, y_train: np.ndarray,
                       use_catboost: bool) -> Ridge:
    """
    Train Ridge regression on OOF meta-features -> y_true.

    Ridge learns optimal blend weights for each base model. Coefficients near 0
    indicate that model adds little to the ensemble. We use Ridge over plain
    LinearRegression for numerical stability.
    """
    meta_learner = Ridge(alpha=RIDGE_ALPHA)
    meta_learner.fit(oof_meta_X, y_train)

    labels = ["RF", "XGB", "CatBoost"] if use_catboost else ["RF", "XGB"]
    print(f"\n[meta] Ridge meta-learner weights:", flush=True)
    for label, coef in zip(labels, meta_learner.coef_):
        print(f"  {label:12s}: {coef:.4f}", flush=True)
    print(f"  Intercept   : {meta_learner.intercept_:.4f}", flush=True)

    # Warn if any model gets near-zero weight -- it's not contributing
    for label, coef in zip(labels, meta_learner.coef_):
        if abs(coef) < 0.05:
            print(f"  WARNING: {label} weight near zero -- may not be helping.", flush=True)

    return meta_learner


def evaluate_stacked(rf_final, xgb_final, cat_final, meta_learner: Ridge,
                     X_test: np.ndarray, y_test: np.ndarray,
                     smiles_test: pd.Series, use_catboost: bool) -> tuple:
    """
    Evaluate stacked model on held-out test set.
    Also reports each base model individually so we can measure the stacking gain.
    """
    # Impute NaNs in test set using imputer fitted on full training set
    # (stored in the bundle -- here we re-fit on X_test for simplicity since
    # test NaN rate should be very low; correct approach is to use train imputer)
    rf_test   = rf_final.predict(X_test)
    xgb_test  = xgb_final.predict(X_test)

    if use_catboost and cat_final is not None:
        cat_test   = cat_final.predict(X_test)
        test_meta  = np.column_stack([rf_test, xgb_test, cat_test])
    else:
        test_meta  = np.column_stack([rf_test, xgb_test])

    stacked_pred = meta_learner.predict(test_meta)

    def metrics(y_true, y_pred, name):
        """Print and return RMSE, R², MAE for a prediction set."""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)
        mae  = mean_absolute_error(y_true, y_pred)
        print(f"\n[eval] {name}:", flush=True)
        print(f"  R²   : {r2:.4f}", flush=True)
        print(f"  RMSE : {rmse:.4f}  (log10 x2)", flush=True)
        print(f"  MAE  : {mae:.4f}  (log10 x2)", flush=True)
        return {"model": name, "test_r2": r2, "test_rmse_log": rmse, "test_mae_log": mae}

    perf_rows = [
        metrics(y_test, rf_test,      "RF (v2 params)"),
        metrics(y_test, xgb_test,     "XGB (v3 params)"),
    ]
    if use_catboost and cat_final is not None:
        perf_rows.append(metrics(y_test, cat_test, "CatBoost (default)"))
    perf_rows.append(metrics(y_test, stacked_pred, "Stacked (RF+XGB+CB+Ridge)"))

    predictions_df = pd.DataFrame({
        "il_smiles":       smiles_test.values,
        "y_true_log":      y_test,
        "rf_pred_log":     rf_test,
        "xgb_pred_log":    xgb_test,
        "stacked_pred_log": stacked_pred,
        "residual_log":    stacked_pred - y_test,
    })

    return pd.DataFrame(perf_rows), predictions_df


def main():
    """
    Full stacking pipeline:
    1. Load data
    2. Generate OOF predictions (GroupKFold, no IL leakage)
    3. Train Ridge meta-learner on OOF predictions
    4. Refit all base models on full training set
    5. Evaluate on test set; report stacking gain
    6. Save stacked model bundle
    """
    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=== STACKING ENSEMBLE: RF + XGB (v3) + CatBoost + Ridge ===\n", flush=True)

    # Step 1: Load data
    X_train, y_train, smiles_train, feature_cols = load_split(TRAIN_CSV, "Train")
    X_test,  y_test,  smiles_test,  _            = load_split(TEST_CSV,  "Test")

    # Impute NaNs in full train/test sets for final base model fitting
    # (OOF loop does its own per-fold imputation)
    full_imputer = SimpleImputer(strategy="median")
    X_train_imp  = full_imputer.fit_transform(X_train)
    X_test_imp   = full_imputer.transform(X_test)

    # Step 2: Initialise CatBoost (None if not installed)
    catboost_template = make_catboost()
    use_catboost = catboost_template is not None

    # Step 3: OOF predictions for meta-learner training
    oof_meta_X = generate_oof_predictions(
        X_train, y_train, smiles_train, catboost_template
    )

    # Step 4: Train Ridge meta-learner on OOF predictions
    meta_learner = train_meta_learner(oof_meta_X, y_train, use_catboost)

    # Step 5: Refit base models on FULL training set
    print("\n[main] Fitting final RF on full training set...", flush=True)
    rf_final = make_rf()
    rf_final.fit(X_train_imp, y_train)

    print("[main] Fitting final XGB on full training set...", flush=True)
    xgb_final = make_xgb()
    xgb_final.fit(X_train_imp, y_train)

    cat_final = None
    if use_catboost:
        print("[main] Fitting final CatBoost on full training set...", flush=True)
        from sklearn.base import clone
        cat_final = clone(catboost_template)
        cat_final.fit(X_train_imp, y_train)

    # Step 6: Evaluate on held-out test set
    print("\n=== TEST SET EVALUATION ===", flush=True)
    perf_df, predictions_df = evaluate_stacked(
        rf_final, xgb_final, cat_final, meta_learner,
        X_test_imp, y_test, smiles_test, use_catboost
    )

    # Report stacking gain
    print("\n=== SUMMARY ===", flush=True)
    print(perf_df[["model", "test_r2", "test_rmse_log"]].to_string(index=False), flush=True)
    stacked_r2    = perf_df[perf_df["model"].str.startswith("Stacked")]["test_r2"].values[0]
    best_single   = perf_df[~perf_df["model"].str.startswith("Stacked")]["test_r2"].max()
    print(f"\n  Stacking gain vs best single model: {stacked_r2 - best_single:+.4f} R²",
          flush=True)

    # Step 7: Save stacked model bundle
    weight_labels = ["RF", "XGB", "CatBoost"] if use_catboost else ["RF", "XGB"]
    stacking_weights = pd.DataFrame({
        "base_model":  weight_labels,
        "ridge_weight": meta_learner.coef_,
    })
    stacking_weights["intercept"] = [meta_learner.intercept_] + [""] * (len(weight_labels) - 1)

    bundle = {
        "rf":           rf_final,
        "xgb":          xgb_final,
        "catboost":     cat_final,
        "meta_learner": meta_learner,
        "imputer":      full_imputer,   # save imputer so inference pipeline can use it
        "feature_cols": feature_cols,
        "model_name":   "Stacked_RF_XGB_CatBoost_Ridge_v3",
        "use_catboost": use_catboost,
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
