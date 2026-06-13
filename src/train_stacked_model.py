"""
train_stacked_model.py
----------------------
PURPOSE: Build a stacking ensemble that blends RF, tuned XGBoost, tuned CatBoost,
         and optionally LightGBM predictions using a Ridge meta-learner.

WHY STACKING:
  Each base model makes structurally different errors:
    - RF (bagging): averages many decision trees -> smooth, slightly underfit
    - XGBoost (level-wise boosting): sharp corrections but occasionally overfit
    - CatBoost (symmetric trees): different regularization, empirically best solo
    - LightGBM (leaf-wise boosting): optional 4th member if residual correlation
      vs XGB is below 0.85 after running measure_residual_correlations.py
  A Ridge meta-learner trained on OOF predictions learns when to trust each model,
  typically gaining 0.02-0.05 R² over any single model.

HOW STACKING WORKS:
  Step 1 -- OOF predictions:
    For each CV fold, train all base models on the fold's training ILs, then
    predict on the fold's validation ILs (ILs they never saw). After all folds,
    every training row has a [rf, xgb, cat, lgb] tuple from models that never
    saw its IL. This is the meta-feature matrix.

  Step 2 -- Meta-learner:
    Train Ridge regression on the meta-feature matrix -> y_true.
    Ridge learns the optimal linear blend weights for each base model.

  Step 3 -- Final base models:
    Refit all base models on the FULL training set.

  Step 4 -- Inference:
    For new IL: get predictions from each base model, feed into Ridge -> final pred.

GROUP-AWARE CV:
  GroupKFold by il_smiles -- the same IL never appears in both train and val
  within any fold. Plain KFold would inflate OOF R² to ~0.95 due to leakage.

BASE MODEL HYPERPARAMETERS:
  RF:       Optuna v2 best params
  XGB:      Optuna v3 best params (min_child_weight=1 corrects v2's T_K suppression)
  CatBoost: >>> UPDATE FROM tune_catboost.py OUTPUT <<<
            Defaults below. Run tune_catboost.py, then replace the CATBOOST_* constants.
  LightGBM: Optional 4th model. Set INCLUDE_LGB = True after checking
            residual correlation vs XGB via measure_residual_correlations.py.
            Only include if r_vs_xgb < 0.85.

INPUTS:
  data/processed/train_set.csv   (from build_dataset.py)
  data/processed/test_set.csv    (from build_dataset.py)

OUTPUTS:
  models/stacked_model.pkl                 -- full stacking bundle
  results/stacked_model_performance.csv    -- R², RMSE per model + stacked
  results/stacked_model_predictions.csv    -- per-row predictions on test set
  results/stacking_weights.csv             -- ridge coefficients per base model

Run from project root:
  nohup python src/train_stacked_model.py > logs/stacked_model.log 2>&1 &
Then monitor:
  tail -f logs/stacked_model.log
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
CV_FOLDS           = 5
RANDOM_SEED        = 42

# ---- LGB toggle ----
# Set True only if measure_residual_correlations.py showed r_vs_xgb < 0.85 for LGB.
# If uncertain, leave False -- a weakly diverse 4th model adds noise, not signal.
INCLUDE_LGB = False

# ---- RF: Optuna v2 best params ----
RF_N_ESTIMATORS     = 515
RF_MAX_FEATURES     = 0.5
RF_MIN_SAMPLES_LEAF = 7
RF_MAX_DEPTH        = None   # unlimited depth was best in v2

# ---- XGB: Optuna v3 best params ----
# Key change from v2: min_child_weight=1 (v2 was 10, which suppressed T_K importance)
XGB_N_ESTIMATORS     = 703
XGB_LEARNING_RATE    = 0.014
XGB_MAX_DEPTH        = 5
XGB_MIN_CHILD_WEIGHT = 1
XGB_GAMMA            = 0.21
XGB_SUBSAMPLE        = 0.776
XGB_COLSAMPLE        = 0.460
XGB_REG_ALPHA        = 1.2
XGB_REG_LAMBDA       = 0.8

# ---- CatBoost: UPDATE THESE after running tune_catboost.py ----
# Run:  nohup python src/tune_catboost.py > logs/tune_catboost.log 2>&1 &
# Then paste the printed BEST PARAMS block here:
# -----------------------------------------------------------------------
CATBOOST_ITERATIONS      = 500    # <-- replace with tuned value
CATBOOST_LEARNING_RATE   = 0.05   # <-- replace with tuned value
CATBOOST_DEPTH           = 6      # <-- replace with tuned value
CATBOOST_L2_LEAF_REG     = 3.0    # <-- replace with tuned value (default=3)
CATBOOST_BORDER_COUNT    = 254    # <-- replace with tuned value
CATBOOST_BAGGING_TEMP    = 1.0    # <-- replace with tuned value (default=1.0)
# -----------------------------------------------------------------------

# ---- LGB best params ----
# Update from results/tuning_results_lgb.csv after tune_hyperparameters.py finishes.
# Only used if INCLUDE_LGB = True.
LGB_N_ESTIMATORS      = 500
LGB_LEARNING_RATE     = 0.05
LGB_NUM_LEAVES        = 31
LGB_MIN_CHILD_SAMPLES = 10
LGB_FEATURE_FRACTION  = 0.7
LGB_BAGGING_FRACTION  = 0.8
LGB_BAGGING_FREQ      = 5
LGB_REG_ALPHA         = 0.1
LGB_REG_LAMBDA        = 0.1

# Ridge meta-learner: small alpha, only 3-4 meta-features
RIDGE_ALPHA = 1.0


# -- Data loading ---------------------------------------------------------------

def load_split(path: str, label: str) -> tuple:
    """
    Load train or test CSV. Returns X, y, il_smiles groups, and feature column names.
    Fills NaN descriptors with column median (occurs for unusual SMILES Gasteiger charges).
    """
    df = pd.read_csv(path)
    df = df.dropna(subset=CONDITION_FEATURES).copy()

    molecular_cols = [c for c in df.columns if c.startswith("cat_") or c.startswith("an_")]
    feature_cols   = molecular_cols + CONDITION_FEATURES

    # Fill NaN descriptors to avoid imputer-fit-inside-CV issues for some edge cases
    desc_cols = [c for c in molecular_cols if "_fp_" not in c]
    for col in desc_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    X      = df[feature_cols].values.astype(float)
    y      = df[TARGET_COL].values
    smiles = df["il_smiles"]

    print(f"[load] {label}: {len(df)} rows, {smiles.nunique()} ILs, "
          f"{len(feature_cols)} features", flush=True)
    return X, y, smiles, feature_cols


# -- Model constructors ---------------------------------------------------------

def make_rf() -> RandomForestRegressor:
    """Instantiate RF with Optuna v2 best hyperparameters."""
    return RandomForestRegressor(
        n_estimators     = RF_N_ESTIMATORS,
        max_features     = RF_MAX_FEATURES,
        min_samples_leaf = RF_MIN_SAMPLES_LEAF,
        max_depth        = RF_MAX_DEPTH,
        random_state     = RANDOM_SEED,
        n_jobs           = -1,
    )


def make_xgb() -> XGBRegressor:
    """Instantiate XGB with Optuna v3 best hyperparameters."""
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
    Instantiate CatBoost with Optuna-tuned hyperparameters.
    CatBoost's symmetric trees gave R²≈0.726 (best solo model in ablation).
    Returns None if catboost is not installed -- ensemble degrades gracefully.
    """
    try:
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            iterations          = CATBOOST_ITERATIONS,
            learning_rate       = CATBOOST_LEARNING_RATE,
            depth               = CATBOOST_DEPTH,
            l2_leaf_reg         = CATBOOST_L2_LEAF_REG,
            border_count        = CATBOOST_BORDER_COUNT,
            bagging_temperature = CATBOOST_BAGGING_TEMP,
            random_seed         = RANDOM_SEED,
            verbose             = 0,
        )
    except ImportError:
        print("[WARNING] catboost not installed. Ensemble will use RF + XGB only.", flush=True)
        print("          Install with: pip install catboost", flush=True)
        return None


def make_lgb():
    """
    Instantiate LightGBM with best-available params.
    Returns None if lightgbm is not installed or INCLUDE_LGB=False.
    Only add LGB to ensemble after confirming r_vs_xgb < 0.85 via
    measure_residual_correlations.py.
    """
    if not INCLUDE_LGB:
        return None
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators      = LGB_N_ESTIMATORS,
            learning_rate     = LGB_LEARNING_RATE,
            num_leaves        = LGB_NUM_LEAVES,
            min_child_samples = LGB_MIN_CHILD_SAMPLES,
            feature_fraction  = LGB_FEATURE_FRACTION,
            bagging_fraction  = LGB_BAGGING_FRACTION,
            bagging_freq      = LGB_BAGGING_FREQ,
            reg_alpha         = LGB_REG_ALPHA,
            reg_lambda        = LGB_REG_LAMBDA,
            random_state      = RANDOM_SEED,
            n_jobs            = 2,
            verbose           = -1,
        )
    except ImportError:
        print("[WARNING] lightgbm not installed. Skipping LGB in ensemble.", flush=True)
        return None


# -- OOF prediction generation -------------------------------------------------

def generate_oof_predictions(X_train: np.ndarray, y_train: np.ndarray,
                              il_smiles_train: pd.Series,
                              catboost_template, lgb_template) -> tuple:
    """
    Generate out-of-fold predictions for every training row using GroupKFold.

    For each fold:
      - Impute NaNs (fitted on train fold only, no leakage)
      - Train all base models on fold's training ILs
      - Predict on fold's val ILs (ILs never seen in this fold)

    After all folds, every training row has predictions from models that never
    saw its IL. This is the meta-feature matrix the Ridge meta-learner trains on.

    Returns (oof_meta_X, model_labels) where oof_meta_X is shape
    (n_train_rows, n_base_models).
    """
    from sklearn.base import clone
    groups = il_smiles_train.values
    gkf    = GroupKFold(n_splits=CV_FOLDS)

    oof_rf  = np.zeros(len(y_train))
    oof_xgb = np.zeros(len(y_train))
    oof_cat = np.zeros(len(y_train)) if catboost_template is not None else None
    oof_lgb = np.zeros(len(y_train)) if lgb_template is not None else None

    print(f"\n[oof] Generating OOF predictions with {CV_FOLDS}-fold GroupKFold...",
          flush=True)

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        # Confirm no IL leaks across this fold's split
        assert len(set(groups[train_idx]) & set(groups[val_idx])) == 0, \
            f"IL overlap in fold {fold_idx}!"

        # Impute NaNs -- fit only on train fold to prevent leakage
        imputer = SimpleImputer(strategy="median")
        X_tr  = imputer.fit_transform(X_train[train_idx])
        X_val = imputer.transform(X_train[val_idx])
        y_tr  = y_train[train_idx]

        # RF fold
        rf_fold = make_rf()
        rf_fold.fit(X_tr, y_tr)
        oof_rf[val_idx] = rf_fold.predict(X_val)

        # XGB fold
        xgb_fold = make_xgb()
        xgb_fold.fit(X_tr, y_tr)
        oof_xgb[val_idx] = xgb_fold.predict(X_val)

        # CatBoost fold
        if catboost_template is not None:
            cat_fold = clone(catboost_template)
            cat_fold.fit(X_tr, y_tr)
            oof_cat[val_idx] = cat_fold.predict(X_val)

        # LGB fold (optional)
        if lgb_template is not None:
            lgb_fold = clone(lgb_template)
            lgb_fold.fit(X_tr, y_tr)
            oof_lgb[val_idx] = lgb_fold.predict(X_val)

        # Per-fold diagnostic
        n_val_ils = len(set(groups[val_idx]))
        line = (f"  Fold {fold_idx+1}/{CV_FOLDS}: {n_val_ils} val ILs | "
                f"RF={r2_score(y_train[val_idx], oof_rf[val_idx]):.3f}, "
                f"XGB={r2_score(y_train[val_idx], oof_xgb[val_idx]):.3f}")
        if catboost_template is not None:
            line += f", CB={r2_score(y_train[val_idx], oof_cat[val_idx]):.3f}"
        if lgb_template is not None:
            line += f", LGB={r2_score(y_train[val_idx], oof_lgb[val_idx]):.3f}"
        print(line, flush=True)

    # Summarise combined OOF R²
    print(f"\n[oof] Combined OOF R²: "
          f"RF={r2_score(y_train, oof_rf):.4f}, "
          f"XGB={r2_score(y_train, oof_xgb):.4f}", flush=True)
    if catboost_template is not None:
        print(f"[oof]                   CB={r2_score(y_train, oof_cat):.4f}", flush=True)
    if lgb_template is not None:
        print(f"[oof]                   LGB={r2_score(y_train, oof_lgb):.4f}", flush=True)

    # Build meta-feature matrix and label list
    meta_cols  = [oof_rf, oof_xgb]
    meta_labels = ["RF", "XGB"]
    if catboost_template is not None:
        meta_cols.append(oof_cat)
        meta_labels.append("CatBoost")
    if lgb_template is not None:
        meta_cols.append(oof_lgb)
        meta_labels.append("LightGBM")

    return np.column_stack(meta_cols), meta_labels


def r2_score(y_true, y_pred):
    """Convenience wrapper for sklearn r2_score (avoids star-import)."""
    from sklearn.metrics import r2_score as _r2
    return _r2(y_true, y_pred)


# -- Meta-learner training ------------------------------------------------------

def train_meta_learner(oof_meta_X: np.ndarray, y_train: np.ndarray,
                       meta_labels: list) -> Ridge:
    """
    Train Ridge regression on OOF meta-features -> y_true.
    The coefficients are the ensemble blend weights -- positive coefficient means
    the model is trusted. A near-zero coefficient means that model adds nothing.
    """
    meta_learner = Ridge(alpha=RIDGE_ALPHA)
    meta_learner.fit(oof_meta_X, y_train)

    print(f"\n[meta] Ridge meta-learner weights:", flush=True)
    for label, coef in zip(meta_labels, meta_learner.coef_):
        print(f"  {label:12s}: {coef:.4f}", flush=True)
    print(f"  Intercept   : {meta_learner.intercept_:.4f}", flush=True)

    # Warn if any model gets near-zero weight -- it's not contributing
    for label, coef in zip(meta_labels, meta_learner.coef_):
        if abs(coef) < 0.05:
            print(f"  WARNING: {label} weight near zero ({coef:.4f}) -- "
                  f"this model is not contributing to the ensemble.", flush=True)
            print(f"           Consider removing it and rerunning.", flush=True)

    return meta_learner


# -- Test evaluation ------------------------------------------------------------

def evaluate_stacked(rf_final, xgb_final, cat_final, lgb_final,
                      meta_learner: Ridge, meta_labels: list,
                      X_test: np.ndarray, y_test: np.ndarray,
                      smiles_test: pd.Series) -> tuple:
    """
    Evaluate all base models and the stacked ensemble on the held-out test set.
    Reports individual base model performance so we can measure stacking gain.
    """
    from sklearn.metrics import r2_score as _r2

    # Collect base model test predictions
    base_preds = {
        "RF":  rf_final.predict(X_test),
        "XGB": xgb_final.predict(X_test),
    }
    if cat_final is not None:
        base_preds["CatBoost"] = cat_final.predict(X_test)
    if lgb_final is not None:
        base_preds["LightGBM"] = lgb_final.predict(X_test)

    # Build meta-feature matrix for stacked prediction
    test_meta = np.column_stack([base_preds[label] for label in meta_labels])
    stacked_pred = meta_learner.predict(test_meta)

    def metrics(y_true, y_pred, name):
        """Compute and print RMSE, R², MAE for one model."""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = _r2(y_true, y_pred)
        mae  = mean_absolute_error(y_true, y_pred)
        print(f"\n[eval] {name}:", flush=True)
        print(f"  R²   : {r2:.4f}", flush=True)
        print(f"  RMSE : {rmse:.4f}  (log10 x2)", flush=True)
        print(f"  MAE  : {mae:.4f}  (log10 x2)", flush=True)
        return {"model": name, "test_r2": r2, "test_rmse_log": rmse, "test_mae_log": mae}

    perf_rows = [metrics(y_test, pred, name) for name, pred in base_preds.items()]
    perf_rows.append(metrics(y_test, stacked_pred, f"Stacked ({'+'.join(meta_labels)}+Ridge)"))

    # Build predictions DataFrame
    pred_df = pd.DataFrame({
        "il_smiles":    smiles_test.values,
        "y_true_log":   y_test,
        "stacked_pred_log": stacked_pred,
        "residual_log": stacked_pred - y_test,
    })
    for name, pred in base_preds.items():
        pred_df[f"{name.lower()}_pred_log"] = pred

    return pd.DataFrame(perf_rows), pred_df


# -- Main -----------------------------------------------------------------------

def main():
    """
    Full stacking pipeline:
    1. Load data
    2. Generate OOF predictions (GroupKFold, no IL leakage)
    3. Train Ridge meta-learner on OOF meta-features
    4. Refit all base models on full training set
    5. Evaluate on test set; report stacking gain vs best single model
    6. Save stacking bundle
    """
    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=== STACKING ENSEMBLE ===\n", flush=True)
    print(f"Base models: RF + XGB + CatBoost" +
          (" + LightGBM" if INCLUDE_LGB else "") +
          " (INCLUDE_LGB=" + str(INCLUDE_LGB) + ")", flush=True)
    print(f"Meta-learner: Ridge (alpha={RIDGE_ALPHA})\n", flush=True)

    # Step 1: Load data
    X_train, y_train, smiles_train, feature_cols = load_split(TRAIN_CSV, "Train")
    X_test,  y_test,  smiles_test,  _            = load_split(TEST_CSV,  "Test")

    # Impute NaNs in full train/test for final base model fitting
    full_imputer  = SimpleImputer(strategy="median")
    X_train_imp   = full_imputer.fit_transform(X_train)
    X_test_imp    = full_imputer.transform(X_test)

    # Step 2: Initialise model templates
    catboost_template = make_catboost()
    lgb_template      = make_lgb()

    use_catboost = catboost_template is not None
    use_lgb      = lgb_template is not None

    # Step 3: OOF predictions for meta-learner
    oof_meta_X, meta_labels = generate_oof_predictions(
        X_train, y_train, smiles_train, catboost_template, lgb_template
    )

    # Step 4: Train Ridge meta-learner on OOF predictions
    meta_learner = train_meta_learner(oof_meta_X, y_train, meta_labels)

    # Step 5: Refit all base models on FULL training set
    print("\n[main] Fitting final base models on full training set...", flush=True)

    rf_final = make_rf()
    rf_final.fit(X_train_imp, y_train)
    print("[main] RF done.", flush=True)

    xgb_final = make_xgb()
    xgb_final.fit(X_train_imp, y_train)
    print("[main] XGB done.", flush=True)

    cat_final = None
    if use_catboost:
        from sklearn.base import clone
        cat_final = clone(catboost_template)
        cat_final.fit(X_train_imp, y_train)
        print("[main] CatBoost done.", flush=True)

    lgb_final = None
    if use_lgb:
        from sklearn.base import clone
        lgb_final = clone(lgb_template)
        lgb_final.fit(X_train_imp, y_train)
        print("[main] LightGBM done.", flush=True)

    # Step 6: Evaluate on held-out test set
    print("\n=== TEST SET EVALUATION ===", flush=True)
    perf_df, pred_df = evaluate_stacked(
        rf_final, xgb_final, cat_final, lgb_final,
        meta_learner, meta_labels,
        X_test_imp, y_test, smiles_test
    )

    # Report stacking gain
    from sklearn.metrics import r2_score as _r2
    print("\n=== SUMMARY ===", flush=True)
    print(perf_df[["model", "test_r2", "test_rmse_log"]].to_string(index=False), flush=True)
    stacked_r2  = perf_df[perf_df["model"].str.startswith("Stacked")]["test_r2"].values[0]
    best_single = perf_df[~perf_df["model"].str.startswith("Stacked")]["test_r2"].max()
    print(f"\n  Stacking gain vs best single model: {stacked_r2 - best_single:+.4f} R²",
          flush=True)
    if stacked_r2 - best_single < 0.005:
        print("  NOTE: Stacking gain < 0.005 R² -- base models may be too correlated.",
              flush=True)
        print("  Check results/residual_correlations.csv.", flush=True)

    # Step 7: Save stacking bundle
    stacking_weights = pd.DataFrame({
        "base_model":    meta_labels,
        "ridge_weight":  meta_learner.coef_,
    })
    stacking_weights["intercept"] = [
        meta_learner.intercept_
    ] + [""] * (len(meta_labels) - 1)

    bundle = {
        "rf":           rf_final,
        "xgb":          xgb_final,
        "catboost":     cat_final,
        "lgb":          lgb_final,
        "meta_learner": meta_learner,
        "meta_labels":  meta_labels,
        "imputer":      full_imputer,
        "feature_cols": feature_cols,
        "model_name":   f"Stacked_{'_'.join(meta_labels)}_Ridge_v4",
        "include_lgb":  use_lgb,
    }
    joblib.dump(bundle, STACKED_MODEL_PATH)

    perf_df.to_csv(
        os.path.join(RESULTS_DIR, "stacked_model_performance.csv"), index=False)
    pred_df.to_csv(
        os.path.join(RESULTS_DIR, "stacked_model_predictions.csv"), index=False)
    stacking_weights.to_csv(
        os.path.join(RESULTS_DIR, "stacking_weights.csv"), index=False)

    print("\n[main] Saved:", flush=True)
    print(f"  {STACKED_MODEL_PATH}", flush=True)
    print("  results/stacked_model_performance.csv", flush=True)
    print("  results/stacked_model_predictions.csv", flush=True)
    print("  results/stacking_weights.csv", flush=True)
    print("[main] DONE.", flush=True)


if __name__ == "__main__":
    main()
