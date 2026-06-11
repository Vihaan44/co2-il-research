"""
compare_models.py
-----------------
PURPOSE: Head-to-head comparison of every candidate model on BOTH the
         held-out test set (true generalisation) AND GroupKFold CV
         (in-distribution estimate). Comparing both reveals whether a model
         is overfit to CV folds vs genuinely better on new ILs.

WHY THIS MATTERS NOW:
  Stacking run showed RF test R²=0.704 while XGB v3 collapsed to 0.591.
  This script checks whether that's a v3-specific problem or a general
  pattern, by also testing XGB v2 params (min_child_weight=10) and seeing
  if tuning actually helped or hurt test generalisation.

MODELS COMPARED:
  - RF (v2 Optuna params)
  - XGB v2 (Optuna v2: min_child_weight=10, n_est=440, lr=0.075)
  - XGB v3 (Optuna v3: min_child_weight=1, n_est=703, lr=0.014, gamma=0.21)
  - CatBoost (default: iterations=500, lr=0.05, depth=6)
  - LightGBM (default: n_est=500, lr=0.05, num_leaves=31)

OUTPUT:
  results/model_comparison.csv  -- test R², CV R², RMSE for every model
  (also printed to stdout)

INPUTS:
  data/processed/train_set.csv
  data/processed/test_set.csv

Run from project root:
  python src/compare_models.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

# -- Constants -----------------------------------------------------------------
TRAIN_CSV   = os.path.join("data", "processed", "train_set.csv")
TEST_CSV    = os.path.join("data", "processed", "test_set.csv")
RESULTS_DIR = "results"

TARGET_COL         = "log_x2_CO2"
CONDITION_FEATURES = ["T_K", "P_kPa"]
CV_FOLDS           = 5
RANDOM_SEED        = 42

# RF: Optuna v2 best params
RF_N_ESTIMATORS     = 515
RF_MAX_FEATURES     = 0.5
RF_MIN_SAMPLES_LEAF = 7

# XGB v2: Optuna v2 best params (min_child_weight=10 was the key setting)
XGB_V2_N_ESTIMATORS     = 440
XGB_V2_LEARNING_RATE    = 0.07481496458680137
XGB_V2_MAX_DEPTH        = 5
XGB_V2_MIN_CHILD_WEIGHT = 10    # high value suppresses T_K in importances
XGB_V2_SUBSAMPLE        = 0.7760131200377596
XGB_V2_COLSAMPLE        = 0.45987026313463525
XGB_V2_REG_ALPHA        = 0.6457288433886444
XGB_V2_REG_LAMBDA       = 3.5815949001230535e-07

# XGB v3: Optuna v3 best params (min_child_weight=1 corrects T_K suppression)
XGB_V3_N_ESTIMATORS     = 703
XGB_V3_LEARNING_RATE    = 0.014
XGB_V3_MAX_DEPTH        = 5
XGB_V3_MIN_CHILD_WEIGHT = 1
XGB_V3_GAMMA            = 0.21
XGB_V3_SUBSAMPLE        = 0.776
XGB_V3_COLSAMPLE        = 0.460
XGB_V3_REG_ALPHA        = 1.2
XGB_V3_REG_LAMBDA       = 0.8

# CatBoost defaults (not yet Optuna-tuned)
CB_ITERATIONS    = 500
CB_LEARNING_RATE = 0.05
CB_DEPTH         = 6

# LightGBM defaults
LGB_N_ESTIMATORS = 500
LGB_LEARNING_RATE = 0.05
LGB_NUM_LEAVES   = 31
LGB_MIN_CHILD_SAMPLES = 10


def load_data() -> tuple:
    """Load train and test sets; return X, y, groups for each."""
    def _load(path, label):
        df = pd.read_csv(path).dropna(subset=CONDITION_FEATURES).copy()
        mol_cols  = [c for c in df.columns if c.startswith("cat_") or c.startswith("an_")]
        feat_cols = mol_cols + CONDITION_FEATURES
        X = df[feat_cols].values.astype(float)
        y = df[TARGET_COL].values
        print(f"[load] {label}: {len(df)} rows, {df['il_smiles'].nunique()} ILs", flush=True)
        return X, y, df["il_smiles"]

    X_tr, y_tr, sm_tr = _load(TRAIN_CSV, "Train")
    X_te, y_te, sm_te = _load(TEST_CSV,  "Test")
    return X_tr, y_tr, sm_tr, X_te, y_te, sm_te


def impute(X_train, X_test):
    """Fit median imputer on train only; apply to both."""
    imp = SimpleImputer(strategy="median")
    return imp.fit_transform(X_train), imp.transform(X_test)


def cv_r2(model, X, y, groups) -> float:
    """GroupKFold CV R² -- in-distribution estimate."""
    gkf  = GroupKFold(n_splits=CV_FOLDS)
    oof  = np.zeros(len(y))
    for tr, va in gkf.split(X, y, groups.values):
        from sklearn.base import clone
        m = clone(model)
        # Per-fold imputation to avoid leakage
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X[tr])
        Xva = imp.transform(X[va])
        m.fit(Xtr, y[tr])
        oof[va] = m.predict(Xva)
    return r2_score(y, oof)


def evaluate(name, model, X_train, y_train, X_test, y_test, groups_train):
    """Train on full train set; report CV R² and test R²."""
    print(f"\n[eval] {name}...", flush=True)

    # CV R² (in-distribution)
    cv = cv_r2(model, X_train, y_train, groups_train)

    # Test R² (out-of-distribution -- held-out ILs)
    from sklearn.base import clone
    m = clone(model)
    m.fit(X_train, y_train)
    test_pred = m.predict(X_test)
    test_r2   = r2_score(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

    print(f"  CV R²   : {cv:.4f}  (GroupKFold, in-distribution)", flush=True)
    print(f"  Test R² : {test_r2:.4f}  (held-out ILs, true generalisation)", flush=True)
    print(f"  Test RMSE: {test_rmse:.4f}", flush=True)
    print(f"  CV-Test gap: {cv - test_r2:+.4f}  "
          f"({'overfit signal' if cv - test_r2 > 0.15 else 'ok'})", flush=True)

    return {"model": name, "cv_r2": round(cv, 4),
            "test_r2": round(test_r2, 4), "test_rmse": round(test_rmse, 4),
            "cv_test_gap": round(cv - test_r2, 4)}


def build_models() -> list:
    """Return list of (name, model) tuples to evaluate."""
    models = []

    models.append(("RF (v2 params)", RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS, max_features=RF_MAX_FEATURES,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF, random_state=RANDOM_SEED, n_jobs=-1,
    )))

    models.append(("XGB v2 (mcw=10)", XGBRegressor(
        n_estimators=XGB_V2_N_ESTIMATORS, learning_rate=XGB_V2_LEARNING_RATE,
        max_depth=XGB_V2_MAX_DEPTH, min_child_weight=XGB_V2_MIN_CHILD_WEIGHT,
        subsample=XGB_V2_SUBSAMPLE, colsample_bytree=XGB_V2_COLSAMPLE,
        reg_alpha=XGB_V2_REG_ALPHA, reg_lambda=XGB_V2_REG_LAMBDA,
        random_state=RANDOM_SEED, n_jobs=2, verbosity=0, tree_method="hist",
    )))

    models.append(("XGB v3 (mcw=1)", XGBRegressor(
        n_estimators=XGB_V3_N_ESTIMATORS, learning_rate=XGB_V3_LEARNING_RATE,
        max_depth=XGB_V3_MAX_DEPTH, min_child_weight=XGB_V3_MIN_CHILD_WEIGHT,
        gamma=XGB_V3_GAMMA, subsample=XGB_V3_SUBSAMPLE,
        colsample_bytree=XGB_V3_COLSAMPLE,
        reg_alpha=XGB_V3_REG_ALPHA, reg_lambda=XGB_V3_REG_LAMBDA,
        random_state=RANDOM_SEED, n_jobs=2, verbosity=0, tree_method="hist",
    )))

    try:
        from catboost import CatBoostRegressor
        models.append(("CatBoost (default)", CatBoostRegressor(
            iterations=CB_ITERATIONS, learning_rate=CB_LEARNING_RATE,
            depth=CB_DEPTH, random_seed=RANDOM_SEED, verbose=0,
        )))
    except ImportError:
        print("[build_models] CatBoost not installed -- skipping.", flush=True)

    try:
        from lightgbm import LGBMRegressor
        models.append(("LightGBM (default)", LGBMRegressor(
            n_estimators=LGB_N_ESTIMATORS, learning_rate=LGB_LEARNING_RATE,
            num_leaves=LGB_NUM_LEAVES, min_child_samples=LGB_MIN_CHILD_SAMPLES,
            random_state=RANDOM_SEED, n_jobs=2, verbose=-1,
        )))
    except ImportError:
        print("[build_models] LightGBM not installed -- skipping.", flush=True)

    return models


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=== MODEL COMPARISON: CV R² vs Test R² ===\n", flush=True)

    X_tr, y_tr, sm_tr, X_te, y_te, sm_te = load_data()
    X_tr_imp, X_te_imp = impute(X_tr, X_te)

    models  = build_models()
    results = []
    for name, model in models:
        row = evaluate(name, model, X_tr_imp, y_tr, X_te_imp, y_te, sm_tr)
        results.append(row)

    results_df = pd.DataFrame(results).sort_values("test_r2", ascending=False)

    print("\n" + "=" * 65, flush=True)
    print("FINAL RANKING (by test R²):", flush=True)
    print(results_df.to_string(index=False), flush=True)
    print("=" * 65, flush=True)

    # Flag the large CV-test gaps -- these indicate overfitting to CV folds
    print("\nOVERFIT FLAGS (CV-Test gap > 0.15):", flush=True)
    overfit = results_df[results_df["cv_test_gap"] > 0.15]
    if len(overfit):
        for _, row in overfit.iterrows():
            print(f"  {row['model']}: gap={row['cv_test_gap']:+.4f} -- "
                  f"CV R²={row['cv_r2']:.4f} vs Test R²={row['test_r2']:.4f}", flush=True)
    else:
        print("  None -- all models generalise consistently.", flush=True)

    results_df.to_csv(
        os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
    print(f"\n[main] Saved: results/model_comparison.csv", flush=True)
    print("[main] DONE.", flush=True)


if __name__ == "__main__":
    main()
