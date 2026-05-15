"""
tune_hyperparameters.py
-----------------------
PURPOSE: Find better hyperparameters for Random Forest and XGBoost using
         Optuna (Bayesian optimization), evaluated with GroupKFold CV.

WHY THIS MATTERS:
  The defaults in train_model.py (e.g. XGBoost max_depth=6, lr=0.05) are
  reasonable starting points but not optimized for our specific dataset
  (~200 ILs, ~1500 rows). Tuning with Optuna can realistically improve
  R² by 0.05-0.10 on this size dataset.

WHY GroupKFold INSIDE TUNING:
  Every candidate hyperparameter set is evaluated by GroupKFold (grouped by
  il_smiles). This prevents the tuner from overfitting to measurement repeats
  of the same IL -- without this, the "best" parameters would be those that
  best memorize repeated T/P conditions, not those that best generalize to
  new ILs.

OUTPUTS:
  results/tuning_results_rf.csv    -- all RF trials with their CV RMSE
  results/tuning_results_xgb.csv   -- all XGBoost trials with their CV RMSE
  results/best_hyperparams.csv     -- best params for each model
  models/forward_model_tuned.pkl   -- retrained best model on full train set

INPUT:
  data/processed/train_set.csv  (from build_dataset.py)

INSTALL (if needed):
  pip install optuna

Run from project root:
    python src/tune_hyperparameters.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

# Suppress Optuna's verbose per-trial logging -- we print our own summaries
optuna.logging.set_verbosity(optuna.logging.WARNING)

# -- Constants -----------------------------------------------------------------
TRAIN_CSV   = os.path.join("data", "processed", "train_set.csv")
MODEL_DIR   = "models"
RESULTS_DIR = "results"
TUNED_MODEL_PATH = os.path.join(MODEL_DIR, "forward_model_tuned.pkl")

TARGET_COL         = "log_x2_CO2"
CONDITION_FEATURES = ["T_K", "P_kPa"]
CV_FOLDS           = 5
RANDOM_SEED        = 42

# How many hyperparameter combinations Optuna will try.
# 50 trials each is enough for this dataset size without taking too long (~5-10 min).
N_TRIALS_RF  = 50
N_TRIALS_XGB = 50


def load_train_data() -> tuple:
    """
    Load the training set and return X (features), y (target), il_smiles (groups),
    and feature_cols (column names). Same logic as train_model.py.
    """
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(
            f"Train set not found at {TRAIN_CSV}. Run build_dataset.py first."
        )
    df = pd.read_csv(TRAIN_CSV)
    print(f"[load_train_data] {df.shape[0]} rows, {df['il_smiles'].nunique()} unique ILs")

    molecular_cols = [c for c in df.columns if c.startswith("cat_") or c.startswith("an_")]
    feature_cols   = molecular_cols + CONDITION_FEATURES

    X         = df[feature_cols].values
    y         = df[TARGET_COL].values
    il_smiles = df["il_smiles"]  # group label for GroupKFold
    return X, y, il_smiles, feature_cols


def grouped_cv_rmse(model, X: np.ndarray, y: np.ndarray,
                    il_smiles: pd.Series) -> float:
    """
    Compute mean RMSE across GroupKFold folds.

    This is the objective Optuna minimizes: we want the lowest average RMSE
    across 5 folds where each fold's val set contains ILs not in that fold's
    training split. Lower RMSE = better generalization to new ILs.
    """
    group_kfold = GroupKFold(n_splits=CV_FOLDS)
    groups      = il_smiles.values
    rmse_scores = []

    for train_idx, val_idx in group_kfold.split(X, y, groups):
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[val_idx])
        rmse   = np.sqrt(mean_squared_error(y[val_idx], y_pred))
        rmse_scores.append(rmse)

    return float(np.mean(rmse_scores))


def rf_objective(trial, X: np.ndarray, y: np.ndarray,
                 il_smiles: pd.Series) -> float:
    """
    Optuna objective for Random Forest.
    Suggests hyperparameter values and returns GroupKFold RMSE.
    Optuna's Bayesian optimizer will use past trial results to
    suggest better values in future trials.
    """
    # Optuna suggests values within the ranges we specify
    n_estimators     = trial.suggest_int("n_estimators", 100, 600)
    max_features     = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5])
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    max_depth        = trial.suggest_categorical("max_depth", [None, 10, 20, 30])

    model = RandomForestRegressor(
        n_estimators     = n_estimators,
        max_features     = max_features,
        min_samples_leaf = min_samples_leaf,
        max_depth        = max_depth,
        random_state     = RANDOM_SEED,
        n_jobs           = -1,
    )
    return grouped_cv_rmse(model, X, y, il_smiles)


def xgb_objective(trial, X: np.ndarray, y: np.ndarray,
                  il_smiles: pd.Series) -> float:
    """
    Optuna objective for XGBoost.
    Suggests hyperparameter values and returns GroupKFold RMSE.
    """
    n_estimators  = trial.suggest_int("n_estimators", 100, 800)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    max_depth     = trial.suggest_int("max_depth", 3, 10)
    subsample     = trial.suggest_float("subsample", 0.5, 1.0)
    colsample     = trial.suggest_float("colsample_bytree", 0.3, 1.0)
    reg_alpha     = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)  # L1 regularization
    reg_lambda    = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True) # L2 regularization

    model = XGBRegressor(
        n_estimators     = n_estimators,
        learning_rate    = learning_rate,
        max_depth        = max_depth,
        subsample        = subsample,
        colsample_bytree = colsample,
        reg_alpha        = reg_alpha,
        reg_lambda       = reg_lambda,
        random_state     = RANDOM_SEED,
        n_jobs           = -1,
        verbosity        = 0,
    )
    return grouped_cv_rmse(model, X, y, il_smiles)


def run_study(model_name: str, objective_fn, n_trials: int,
             X: np.ndarray, y: np.ndarray,
             il_smiles: pd.Series) -> tuple:
    """
    Run an Optuna study for one model type.
    Returns (best_params, trials_df) where trials_df has all trial results.
    """
    print(f"\n[run_study] Tuning {model_name}: {n_trials} trials ...")

    # Use TPE sampler: Bayesian optimizer that learns from past trials
    study = optuna.create_study(
        direction = "minimize",  # minimize RMSE
        sampler   = optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )

    # Lambda wraps the objective so we can pass extra args (X, y, il_smiles)
    study.optimize(
        lambda trial: objective_fn(trial, X, y, il_smiles),
        n_trials = n_trials,
        n_jobs   = 1,  # 1 trial at a time -- avoids memory issues with RF's n_jobs=-1
    )

    best_params = study.best_params
    best_rmse   = study.best_value
    print(f"[run_study] {model_name} best CV RMSE: {best_rmse:.4f}")
    print(f"[run_study] Best params: {best_params}")

    # Collect all trial results for inspection
    trials_df = pd.DataFrame([
        {"trial": t.number, "cv_rmse": t.value, **t.params}
        for t in study.trials if t.value is not None
    ])
    trials_df = trials_df.sort_values("cv_rmse").reset_index(drop=True)

    return best_params, trials_df


def retrain_best_model(best_params_rf: dict, best_cv_rf: float,
                       best_params_xgb: dict, best_cv_xgb: float,
                       X_train: np.ndarray, y_train: np.ndarray,
                       feature_cols: list) -> tuple:
    """
    Retrain whichever model had the better CV RMSE on the full training set.
    Returns (best_model, model_name).
    """
    if best_cv_rf <= best_cv_xgb:
        print(f"\n[retrain] RF wins (CV RMSE {best_cv_rf:.4f} vs XGB {best_cv_xgb:.4f})")
        model_name = "RandomForest_tuned"
        best_model = RandomForestRegressor(
            **best_params_rf,
            random_state = RANDOM_SEED,
            n_jobs       = -1,
        )
    else:
        print(f"\n[retrain] XGB wins (CV RMSE {best_cv_xgb:.4f} vs RF {best_cv_rf:.4f})")
        model_name = "XGBoost_tuned"
        best_model = XGBRegressor(
            **best_params_xgb,
            random_state = RANDOM_SEED,
            n_jobs       = -1,
            verbosity    = 0,
        )

    print(f"[retrain] Fitting {model_name} on full train set ...")
    best_model.fit(X_train, y_train)
    return best_model, model_name


def main():
    """Main pipeline: load -> tune RF -> tune XGB -> retrain best -> save."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # -- Step 1: Load training data ------------------------------------------
    X_train, y_train, il_smiles_train, feature_cols = load_train_data()

    # -- Step 2: Tune Random Forest ------------------------------------------
    best_params_rf, trials_rf = run_study(
        "RandomForest", rf_objective, N_TRIALS_RF,
        X_train, y_train, il_smiles_train
    )
    trials_rf["model"] = "RandomForest"
    trials_rf.to_csv(os.path.join(RESULTS_DIR, "tuning_results_rf.csv"), index=False)
    print(f"[main] RF tuning results saved -> results/tuning_results_rf.csv")

    # -- Step 3: Tune XGBoost ------------------------------------------------
    best_params_xgb, trials_xgb = run_study(
        "XGBoost", xgb_objective, N_TRIALS_XGB,
        X_train, y_train, il_smiles_train
    )
    trials_xgb["model"] = "XGBoost"
    trials_xgb.to_csv(os.path.join(RESULTS_DIR, "tuning_results_xgb.csv"), index=False)
    print(f"[main] XGB tuning results saved -> results/tuning_results_xgb.csv")

    # -- Step 4: Build best-hyperparameters summary --------------------------
    best_cv_rf  = trials_rf.iloc[0]["cv_rmse"]   # already sorted ascending
    best_cv_xgb = trials_xgb.iloc[0]["cv_rmse"]

    best_hyperparams = pd.DataFrame([
        {"model": "RandomForest", "best_cv_rmse": best_cv_rf,  **best_params_rf},
        {"model": "XGBoost",      "best_cv_rmse": best_cv_xgb, **best_params_xgb},
    ])
    best_hyperparams.to_csv(
        os.path.join(RESULTS_DIR, "best_hyperparams.csv"), index=False)
    print("[main] Best hyperparams saved -> results/best_hyperparams.csv")
    print("\n=== BEST HYPERPARAMETERS ===")
    print(best_hyperparams.to_string(index=False))

    # -- Step 5: Retrain winner on full training set -------------------------
    best_model, model_name = retrain_best_model(
        best_params_rf, best_cv_rf,
        best_params_xgb, best_cv_xgb,
        X_train, y_train, feature_cols
    )

    model_bundle = {
        "model":        best_model,
        "feature_cols": feature_cols,
        "model_name":   model_name,
    }
    joblib.dump(model_bundle, TUNED_MODEL_PATH)
    print(f"[main] Tuned model saved -> {TUNED_MODEL_PATH}")
    print("\n[main] DONE. To use the tuned model in inverse design:")
    print("  Edit inverse_design.py MODEL_PATH to point at models/forward_model_tuned.pkl")
    print("  Or rename it: mv models/forward_model_tuned.pkl models/forward_model.pkl")


if __name__ == "__main__":
    main()
