"""
tune_hyperparameters.py
-----------------------
PURPOSE: Find the best hyperparameters for all THREE base models that will
         form our stacking ensemble (XGBoost, LightGBM, Random Forest).
         Each model gets its own Optuna study backed by the same SQLite DB.
         Best params are saved to results/best_hyperparams.csv and loaded
         by train_ensemble.py -- no need to re-run tuning before ensembling.

WHY THREE MODELS:
  Stacking works best when the base models make different kinds of errors.
  XGBoost and LightGBM are both gradient boosters but differ in how they
  grow trees (level-wise vs leaf-wise). RF is a bagging method that errors
  differently from both. The meta-learner (Ridge) learns which model to
  trust more for each region of feature space.

CRASH RECOVERY:
  Optuna saves every completed trial to logs/optuna.db.
  If interrupted, re-run -- each study resumes from where it stopped.
  To start completely fresh:
    rm logs/optuna.db
    python src/tune_hyperparameters.py

WHY GroupKFold INSIDE TUNING:
  Every candidate hyperparameter set is evaluated with GroupKFold (grouped
  by il_smiles). This prevents tuning from finding params that memorize
  repeated T/P measurements of the same IL -- without this, the "best"
  params would generalize to repeated conditions, not new ILs.

VERSION: v4 = 211-IL dataset + all three base models + ensemble prep.
  Delete logs/optuna.db before running (previous study names were v3).

OUTPUTS:
  results/tuning_results_xgb.csv    -- all XGB trials
  results/tuning_results_lgb.csv    -- all LGB trials
  results/tuning_results_rf.csv     -- all RF trials
  results/best_hyperparams.csv      -- best params for all three models
  models/best_xgb.pkl               -- best XGB retrained on full train set
  models/best_lgb.pkl               -- best LGB retrained on full train set
  models/best_rf.pkl                -- best RF retrained on full train set
  logs/optuna.db                    -- SQLite checkpoint

INPUT:
  data/processed/train_set.csv

Run from project root:
    rm logs/optuna.db       # required if upgrading from v3 study
    python src/tune_hyperparameters.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Force line-buffered stdout so nohup logs update in real time
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# -- Constants -----------------------------------------------------------------
TRAIN_CSV   = os.path.join("data", "processed", "train_set.csv")
MODEL_DIR   = "models"
RESULTS_DIR = "results"
LOGS_DIR    = "logs"

OPTUNA_DB_PATH = os.path.join(LOGS_DIR, "optuna.db")
OPTUNA_DB_URL  = f"sqlite:///{OPTUNA_DB_PATH}"

TARGET_COL         = "log_x2_CO2"
CONDITION_FEATURES = ["T_K", "P_kPa"]
CV_FOLDS           = 5
RANDOM_SEED        = 42

# Number of Optuna trials per model.
# XGB/LGB get more because their search spaces are larger.
N_TRIALS_XGB = 75
N_TRIALS_LGB = 75
N_TRIALS_RF  = 50

# Study names -- bump version string when restarting on a new dataset/features.
STUDY_NAME_XGB = "xgb_co2_il_v4"
STUDY_NAME_LGB = "lgb_co2_il_v4"
STUDY_NAME_RF  = "rf_co2_il_v4"

# XGB search bounds (same as v3)
XGB_MAX_DEPTH_MAX        = 9
XGB_N_EST_MAX            = 800
XGB_MIN_CHILD_WEIGHT_MAX = 20
XGB_GAMMA_MAX            = 5.0
XGB_MAX_DELTA_STEP_MAX   = 10

# LGB search bounds
# num_leaves controls model complexity -- LGB grows leaf-wise, so this is more
# important than max_depth. At 211 ILs, keep <= 63 to avoid overfitting.
LGB_N_EST_MAX      = 800
LGB_NUM_LEAVES_MAX = 63   # 2^(max_depth=6) - 1; keeps model from memorizing
LGB_MIN_DATA_MAX   = 30   # min samples in a leaf -- key regularizer for small data
LGB_FEATURE_FRAC_MAX = 1.0

# RF search bounds
RF_N_EST_MAX      = 500
RF_MAX_DEPTH_MAX  = 20    # None (unlimited) is also valid but harder to tune
RF_MIN_SAMPLES_MAX = 10


# -- Data loading --------------------------------------------------------------

def load_train_data() -> tuple:
    """
    Load the training set and return feature matrix X, target y,
    IL SMILES group labels, and the list of feature column names.
    Drops rows with missing T_K or P_kPa defensively.
    """
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(
            f"Train set not found at {TRAIN_CSV}. Run build_dataset.py first."
        )
    df = pd.read_csv(TRAIN_CSV)
    df = df.dropna(subset=CONDITION_FEATURES).copy()

    # All columns starting with cat_ or an_ are molecular features (fingerprints + descriptors)
    molecular_cols = [c for c in df.columns if c.startswith("cat_") or c.startswith("an_")]
    feature_cols   = molecular_cols + CONDITION_FEATURES

    print(f"[load_train_data] {df.shape[0]} rows, {df['il_smiles'].nunique()} unique ILs",
          flush=True)
    print(f"[load_train_data] {len(feature_cols)} features ({len(molecular_cols)} molecular "
          f"+ {len(CONDITION_FEATURES)} condition)", flush=True)

    X         = df[feature_cols].values
    y         = df[TARGET_COL].values
    il_smiles = df["il_smiles"]
    return X, y, il_smiles, feature_cols


# -- CV helper -----------------------------------------------------------------

def grouped_cv_rmse(model, X: np.ndarray, y: np.ndarray,
                    il_smiles: pd.Series) -> float:
    """
    Evaluate a model via GroupKFold CV grouped by IL identity.
    Returns mean RMSE across folds. This is what every Optuna objective
    minimizes -- lower RMSE = better generalisation to new ILs.
    """
    group_kfold = GroupKFold(n_splits=CV_FOLDS)
    groups      = il_smiles.values
    rmse_scores = []

    for train_idx, val_idx in group_kfold.split(X, y, groups):
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[val_idx])
        rmse_scores.append(np.sqrt(mean_squared_error(y[val_idx], y_pred)))

    return float(np.mean(rmse_scores))


# -- Optuna objectives ---------------------------------------------------------

def xgb_objective(trial, X: np.ndarray, y: np.ndarray,
                   il_smiles: pd.Series) -> float:
    """
    Optuna objective for XGBoost. Searches over 10 hyperparameters.
    Returns GroupKFold CV RMSE (lower = better).

    Key parameters:
      gamma: minimum loss reduction to make a split -- prevents splits on
        fingerprint bits that appear in only 1-2 ILs (sparse bit problem).
      min_child_weight: minimum samples per leaf -- main guard against overfitting
        at our dataset size.
    """
    params = dict(
        n_estimators     = trial.suggest_int("n_estimators", 100, XGB_N_EST_MAX),
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth        = trial.suggest_int("max_depth", 3, XGB_MAX_DEPTH_MAX),
        subsample        = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.3, 1.0),
        reg_alpha        = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda       = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        min_child_weight = trial.suggest_int("min_child_weight", 1, XGB_MIN_CHILD_WEIGHT_MAX),
        gamma            = trial.suggest_float("gamma", 0.0, XGB_GAMMA_MAX),
        max_delta_step   = trial.suggest_int("max_delta_step", 0, XGB_MAX_DELTA_STEP_MAX),
    )
    model = XGBRegressor(
        **params, random_state=RANDOM_SEED, n_jobs=2,
        verbosity=0, tree_method="hist",
    )
    rmse = grouped_cv_rmse(model, X, y, il_smiles)
    print(f"  XGB  Trial {trial.number:3d}: RMSE={rmse:.4f} | "
          f"n_est={params['n_estimators']}, lr={params['learning_rate']:.4f}, "
          f"depth={params['max_depth']}, mcw={params['min_child_weight']}, "
          f"gamma={params['gamma']:.2f}", flush=True)
    return rmse


def lgb_objective(trial, X: np.ndarray, y: np.ndarray,
                   il_smiles: pd.Series) -> float:
    """
    Optuna objective for LightGBM. LGB grows trees leaf-wise (deepest leaf
    first) instead of level-wise like XGBoost, which often finds better splits
    on small/medium tabular datasets. The key regularisers here are num_leaves
    (controls total model capacity) and min_child_samples (minimum samples per
    leaf -- essential guard against memorising individual ILs).

    feature_fraction: subsample of features per tree (like colsample_bytree in XGB).
      On 4000+ binary fingerprint features, this also speeds up training.
    bagging_fraction + bagging_freq: row subsampling (like subsample in XGB).
    """
    params = dict(
        n_estimators       = trial.suggest_int("n_estimators", 100, LGB_N_EST_MAX),
        learning_rate      = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        num_leaves         = trial.suggest_int("num_leaves", 15, LGB_NUM_LEAVES_MAX),
        min_child_samples  = trial.suggest_int("min_child_samples", 5, LGB_MIN_DATA_MAX),
        feature_fraction   = trial.suggest_float("feature_fraction", 0.3, LGB_FEATURE_FRAC_MAX),
        bagging_fraction   = trial.suggest_float("bagging_fraction", 0.5, 1.0),
        bagging_freq       = trial.suggest_int("bagging_freq", 1, 10),
        reg_alpha          = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda         = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    )
    model = LGBMRegressor(
        **params, random_state=RANDOM_SEED, n_jobs=2,
        verbose=-1,  # suppress LGB training logs
    )
    rmse = grouped_cv_rmse(model, X, y, il_smiles)
    print(f"  LGB  Trial {trial.number:3d}: RMSE={rmse:.4f} | "
          f"n_est={params['n_estimators']}, lr={params['learning_rate']:.4f}, "
          f"leaves={params['num_leaves']}, min_child={params['min_child_samples']}",
          flush=True)
    return rmse


def rf_objective(trial, X: np.ndarray, y: np.ndarray,
                  il_smiles: pd.Series) -> float:
    """
    Optuna objective for Random Forest. RF is a bagging method -- it trains
    many decision trees on bootstrap samples of the data and averages their
    predictions. It errors very differently from gradient boosters (XGB/LGB)
    which is why it's a valuable third member of the ensemble.

    max_features: fraction of features considered at each split.
      Lower values make trees more diverse (less correlated), which improves
      the ensemble average. 'sqrt' is a common default for regression.
    min_samples_leaf: minimum samples in any leaf node.
      The most important regularizer for RF at our dataset size.
    """
    params = dict(
        n_estimators     = trial.suggest_int("n_estimators", 100, RF_N_EST_MAX),
        max_depth        = trial.suggest_int("max_depth", 5, RF_MAX_DEPTH_MAX),
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, RF_MIN_SAMPLES_MAX),
        max_features     = trial.suggest_float("max_features", 0.1, 1.0),
    )
    model = RandomForestRegressor(
        **params, random_state=RANDOM_SEED, n_jobs=2,
    )
    rmse = grouped_cv_rmse(model, X, y, il_smiles)
    print(f"  RF   Trial {trial.number:3d}: RMSE={rmse:.4f} | "
          f"n_est={params['n_estimators']}, depth={params['max_depth']}, "
          f"min_leaf={params['min_samples_leaf']}, max_feat={params['max_features']:.2f}",
          flush=True)
    return rmse


# -- Study runner --------------------------------------------------------------

def run_study(study_name: str, objective_fn,
              n_trials: int, X: np.ndarray, y: np.ndarray,
              il_smiles: pd.Series) -> tuple:
    """
    Create or resume an Optuna study backed by SQLite.
    load_if_exists=True means crashes are safe -- just re-run.
    Returns (best_params dict, all_trials DataFrame).
    """
    os.makedirs(LOGS_DIR, exist_ok=True)

    study = optuna.create_study(
        study_name     = study_name,
        storage        = OPTUNA_DB_URL,
        direction      = "minimize",
        sampler        = optuna.samplers.TPESampler(seed=RANDOM_SEED),
        load_if_exists = True,
    )

    n_completed = len([t for t in study.trials
                       if t.state == optuna.trial.TrialState.COMPLETE])
    n_remaining = max(0, n_trials - n_completed)

    print(f"\n[run_study] {study_name}: {n_completed} trials done, "
          f"{n_remaining} remaining (target={n_trials})", flush=True)

    if n_remaining > 0:
        study.optimize(
            lambda trial: objective_fn(trial, X, y, il_smiles),
            n_trials = n_remaining,
            n_jobs   = 1,
        )
    else:
        print(f"[run_study] {study_name}: already complete -- loading from DB.", flush=True)

    best_params = study.best_params
    best_rmse   = study.best_value
    print(f"\n[run_study] {study_name} best CV RMSE: {best_rmse:.4f}", flush=True)
    print(f"[run_study] Best params: {best_params}", flush=True)

    trials_df = pd.DataFrame([
        {"trial": t.number, "cv_rmse": t.value, **t.params}
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]).sort_values("cv_rmse").reset_index(drop=True)

    return best_params, trials_df, best_rmse


# -- Model retraining ----------------------------------------------------------

def retrain_and_save(model_name: str, model, X_train: np.ndarray,
                     y_train: np.ndarray, feature_cols: list,
                     save_path: str):
    """
    Retrain a model with best tuned params on the full training set and save.
    The saved dict includes feature_cols so train_ensemble.py can verify
    it's using the same feature set.
    """
    print(f"\n[retrain] Fitting {model_name} on full train set ({X_train.shape[0]} rows)...",
          flush=True)
    model.fit(X_train, y_train)
    joblib.dump({"model": model, "feature_cols": feature_cols,
                 "model_name": model_name}, save_path)
    print(f"[retrain] Saved -> {save_path}", flush=True)


# -- Main ----------------------------------------------------------------------

def main():
    """
    Tune XGBoost, LightGBM, and Random Forest sequentially.
    Save best params and retrained models for use by train_ensemble.py.
    """
    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,    exist_ok=True)

    print("=" * 60, flush=True)
    print("tune_hyperparameters.py  v4 -- three-model tuning", flush=True)
    print(f"Optuna DB: {OPTUNA_DB_PATH}", flush=True)
    print(f"Studies: {STUDY_NAME_XGB}, {STUDY_NAME_LGB}, {STUDY_NAME_RF}", flush=True)
    print("REMINDER: rm logs/optuna.db if upgrading from v3 study.", flush=True)
    print("=" * 60, flush=True)

    X_train, y_train, il_smiles_train, feature_cols = load_train_data()

    # ---- XGBoost ----
    print("\n>>> TUNING XGBoost <<<", flush=True)
    best_xgb, trials_xgb, rmse_xgb = run_study(
        STUDY_NAME_XGB, xgb_objective, N_TRIALS_XGB,
        X_train, y_train, il_smiles_train
    )
    trials_xgb["model"] = "XGBoost"
    trials_xgb.to_csv(os.path.join(RESULTS_DIR, "tuning_results_xgb.csv"), index=False)

    xgb_model = XGBRegressor(
        **best_xgb, random_state=RANDOM_SEED, n_jobs=2,
        verbosity=0, tree_method="hist",
    )
    retrain_and_save("XGBoost_v4", xgb_model, X_train, y_train,
                     feature_cols, os.path.join(MODEL_DIR, "best_xgb.pkl"))

    # ---- LightGBM ----
    print("\n>>> TUNING LightGBM <<<", flush=True)
    best_lgb, trials_lgb, rmse_lgb = run_study(
        STUDY_NAME_LGB, lgb_objective, N_TRIALS_LGB,
        X_train, y_train, il_smiles_train
    )
    trials_lgb["model"] = "LightGBM"
    trials_lgb.to_csv(os.path.join(RESULTS_DIR, "tuning_results_lgb.csv"), index=False)

    lgb_model = LGBMRegressor(
        **best_lgb, random_state=RANDOM_SEED, n_jobs=2, verbose=-1,
    )
    retrain_and_save("LightGBM_v4", lgb_model, X_train, y_train,
                     feature_cols, os.path.join(MODEL_DIR, "best_lgb.pkl"))

    # ---- Random Forest ----
    print("\n>>> TUNING Random Forest <<<", flush=True)
    best_rf, trials_rf, rmse_rf = run_study(
        STUDY_NAME_RF, rf_objective, N_TRIALS_RF,
        X_train, y_train, il_smiles_train
    )
    trials_rf["model"] = "RandomForest"
    trials_rf.to_csv(os.path.join(RESULTS_DIR, "tuning_results_rf.csv"), index=False)

    rf_model = RandomForestRegressor(
        **best_rf, random_state=RANDOM_SEED, n_jobs=2,
    )
    retrain_and_save("RandomForest_v4", rf_model, X_train, y_train,
                     feature_cols, os.path.join(MODEL_DIR, "best_rf.pkl"))

    # ---- Summary ----
    best_hyperparams = pd.DataFrame([
        {"model": "XGBoost_v4",     "best_cv_rmse": rmse_xgb, **best_xgb},
        {"model": "LightGBM_v4",    "best_cv_rmse": rmse_lgb, **best_lgb},
        {"model": "RandomForest_v4","best_cv_rmse": rmse_rf,  **best_rf},
    ])
    best_hyperparams.to_csv(
        os.path.join(RESULTS_DIR, "best_hyperparams.csv"), index=False
    )

    print("\n" + "=" * 60, flush=True)
    print("=== TUNING COMPLETE ===", flush=True)
    print(f"  XGBoost  best CV RMSE: {rmse_xgb:.4f}", flush=True)
    print(f"  LightGBM best CV RMSE: {rmse_lgb:.4f}", flush=True)
    print(f"  RF       best CV RMSE: {rmse_rf:.4f}", flush=True)
    print("Saved: results/best_hyperparams.csv", flush=True)
    print("=" * 60, flush=True)
    print("\nNEXT STEP:", flush=True)
    print("  python src/train_ensemble.py", flush=True)
    print("  (loads best_xgb.pkl, best_lgb.pkl, best_rf.pkl from models/)", flush=True)


if __name__ == "__main__":
    main()
