"""
tune_catboost.py
----------------
PURPOSE: Run Optuna hyperparameter search for CatBoost, the current strongest
         solo model (ablation OOF R²≈0.726 with defaults, vs XGB≈0.689).

         CatBoost's defaults are already strong due to its symmetric-tree
         architecture, but Optuna tuning XGB gained ~0.03-0.05 R² in past
         studies. Similar gain is plausible here. This is the single
         highest-value tuning action before finalising the ensemble.

KEY PARAMETERS BEING TUNED:
  iterations    : number of boosting rounds (equiv. to n_estimators in XGB)
  depth         : depth of symmetric trees (6-8 typical; higher = more
                  expressive but CatBoost constrains splits symmetrically)
  learning_rate : step size per round; low lr + high iterations generalizes better
  l2_leaf_reg   : L2 regularization on leaf weights (key for small datasets)
  border_count  : number of bins for numeric feature splits (higher = slower
                  but more precise; important for continuous T and P features)
  bagging_temperature: controls randomness of bootstrap sampling (Bayesian
                  bootstrap style, 0=deterministic, 1=full randomness)

CRASH RECOVERY:
  Optuna saves every completed trial to logs/optuna_catboost.db.
  If interrupted, just re-run -- the study resumes automatically.
  To start fresh: rm logs/optuna_catboost.db

WHY GroupKFold INSIDE TUNING:
  Every hyperparameter set is evaluated with GroupKFold (grouped by il_smiles).
  This prevents tuning from finding params that memorise repeated T/P measurements
  of the same IL -- without this, the best params overfit to known ILs.

INPUT:
  data/processed/train_set.csv   (from build_dataset.py)

OUTPUTS:
  results/tuning_results_catboost.csv  -- all trials
  results/best_catboost_params.txt     -- best params (copy into train_stacked_model.py)
  models/best_catboost.pkl             -- CatBoost retrained on full train set
  logs/optuna_catboost.db              -- SQLite checkpoint

Run from project root:
  nohup python src/tune_catboost.py > logs/tune_catboost.log 2>&1 &
Then monitor with:
  tail -f logs/tune_catboost.log
When done, paste the printed BEST PARAMS block into tune_stacked_model.py constants.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

# Force line-buffered stdout so nohup logs update in real time
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# -- Constants -----------------------------------------------------------------
TRAIN_CSV   = os.path.join("data", "processed", "train_set.csv")
MODEL_DIR   = "models"
RESULTS_DIR = "results"
LOGS_DIR    = "logs"

OPTUNA_DB_PATH = os.path.join(LOGS_DIR, "optuna_catboost.db")
OPTUNA_DB_URL  = f"sqlite:///{OPTUNA_DB_PATH}"

TARGET_COL         = "log_x2_CO2"
CONDITION_FEATURES = ["T_K", "P_kPa"]
CV_FOLDS           = 5
RANDOM_SEED        = 42
N_TRIALS           = 75   # 75 trials balances search quality vs runtime (~30-60 min)

# Optuna study name -- bump version if rerunning on new dataset
STUDY_NAME = "catboost_co2_il_v1"

# CatBoost hyperparameter search bounds
# iterations: 300-1000 (low lr needs more rounds; let Optuna find the pair)
# depth: 4-8 (CatBoost symmetric trees; deeper = more expressive but slower)
# learning_rate: 0.01-0.3 log scale (low lr generalizes better with more iters)
# l2_leaf_reg: 1-30 (L2 regularization; key guard against overfitting at 211 ILs)
# border_count: 32-255 (bins for numeric features; 64-128 usually enough)
# bagging_temperature: 0-1 (0=no bootstrap noise, 1=full Bayesian bootstrap)
CAT_ITERATIONS_MAX    = 1000
CAT_ITERATIONS_MIN    = 300
CAT_DEPTH_MAX         = 8
CAT_DEPTH_MIN         = 4
CAT_L2_MAX            = 30.0
CAT_L2_MIN            = 1.0
CAT_BORDER_COUNT_MAX  = 255
CAT_BORDER_COUNT_MIN  = 32


def load_train_data() -> tuple:
    """
    Load training set. Returns X (feature matrix), y (target), il_smiles groups,
    and feature column names.
    NaN rows in T_K / P_kPa are dropped (can't use incomplete condition data).
    """
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(
            f"Train set not found at {TRAIN_CSV}. Run build_dataset.py first."
        )
    df = pd.read_csv(TRAIN_CSV)
    df = df.dropna(subset=CONDITION_FEATURES).copy()

    # All cat_* and an_* columns are molecular features (fingerprints + descriptors)
    molecular_cols = [c for c in df.columns if c.startswith("cat_") or c.startswith("an_")]
    feature_cols   = molecular_cols + CONDITION_FEATURES

    # Fill any remaining NaN in descriptors with column median
    # (can occur for Gasteiger charges on unusual SMILES)
    desc_cols = [c for c in molecular_cols if "_fp_" not in c]
    for col in desc_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    print(f"[load] {df.shape[0]} rows, {df['il_smiles'].nunique()} unique ILs, "
          f"{len(feature_cols)} features", flush=True)

    X         = df[feature_cols].values.astype(float)
    y         = df[TARGET_COL].values
    il_smiles = df["il_smiles"]
    return X, y, il_smiles, feature_cols


def grouped_cv_rmse_catboost(params: dict, X: np.ndarray, y: np.ndarray,
                              il_smiles: pd.Series) -> float:
    """
    Evaluate a CatBoost configuration via GroupKFold CV grouped by IL identity.
    Returns mean RMSE across folds. Lower is better.

    GroupKFold prevents the same IL appearing in train and val within any fold,
    ensuring we measure generalisation to new ILs rather than interpolation.
    """
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        raise RuntimeError(
            "catboost not installed. Run: pip install catboost"
        )

    group_kfold = GroupKFold(n_splits=CV_FOLDS)
    groups      = il_smiles.values
    rmse_scores = []

    for train_idx, val_idx in group_kfold.split(X, y, groups):
        model = CatBoostRegressor(**params, random_seed=RANDOM_SEED, verbose=0)
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[val_idx])
        rmse_scores.append(np.sqrt(mean_squared_error(y[val_idx], y_pred)))

    return float(np.mean(rmse_scores))


def catboost_objective(trial, X: np.ndarray, y: np.ndarray,
                        il_smiles: pd.Series) -> float:
    """
    Optuna objective function for CatBoost.
    Samples one hyperparameter configuration, evaluates it with GroupKFold CV,
    and returns the mean RMSE (Optuna minimizes this).

    Note on border_count: higher values give finer-grained splits for
    continuous features like T_K and P_kPa. The default (254) is already high;
    we search down to 32 to see if a coarser grid regularizes better.
    """
    params = dict(
        iterations        = trial.suggest_int("iterations", CAT_ITERATIONS_MIN, CAT_ITERATIONS_MAX),
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        depth             = trial.suggest_int("depth", CAT_DEPTH_MIN, CAT_DEPTH_MAX),
        l2_leaf_reg       = trial.suggest_float("l2_leaf_reg", CAT_L2_MIN, CAT_L2_MAX, log=True),
        border_count      = trial.suggest_int("border_count", CAT_BORDER_COUNT_MIN,
                                               CAT_BORDER_COUNT_MAX),
        bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 1.0),
    )

    rmse = grouped_cv_rmse_catboost(params, X, y, il_smiles)

    print(
        f"  Trial {trial.number:3d}: RMSE={rmse:.4f} | "
        f"iters={params['iterations']}, "
        f"lr={params['learning_rate']:.4f}, "
        f"depth={params['depth']}, "
        f"l2={params['l2_leaf_reg']:.2f}, "
        f"border={params['border_count']}, "
        f"bag_t={params['bagging_temperature']:.2f}",
        flush=True
    )
    return rmse


def main():
    """
    Run Optuna CatBoost tuning study, retrain best model on full train set, save.
    """
    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,    exist_ok=True)

    # Check CatBoost is available before starting
    try:
        from catboost import CatBoostRegressor
        print("[main] CatBoost detected.", flush=True)
    except ImportError:
        print("[main] ERROR: catboost not installed.", flush=True)
        print("       Run: pip install catboost", flush=True)
        return

    print("=" * 60, flush=True)
    print(f"tune_catboost.py  |  Study: {STUDY_NAME}", flush=True)
    print(f"Trials: {N_TRIALS}  |  CV folds: {CV_FOLDS}  |  DB: {OPTUNA_DB_PATH}", flush=True)
    print("To restart fresh: rm logs/optuna_catboost.db", flush=True)
    print("=" * 60, flush=True)

    X_train, y_train, il_smiles_train, feature_cols = load_train_data()

    # Create or resume Optuna study (load_if_exists=True enables crash recovery)
    study = optuna.create_study(
        study_name     = STUDY_NAME,
        storage        = OPTUNA_DB_URL,
        direction      = "minimize",
        sampler        = optuna.samplers.TPESampler(seed=RANDOM_SEED),
        load_if_exists = True,
    )

    n_completed = len([t for t in study.trials
                       if t.state == optuna.trial.TrialState.COMPLETE])
    n_remaining = max(0, N_TRIALS - n_completed)
    print(f"[main] {n_completed} trials done, {n_remaining} remaining.", flush=True)

    if n_remaining > 0:
        study.optimize(
            lambda trial: catboost_objective(trial, X_train, y_train, il_smiles_train),
            n_trials = n_remaining,
            n_jobs   = 1,   # CatBoost is already multithreaded internally
        )
    else:
        print("[main] All trials complete -- loading from DB.", flush=True)

    best_params = study.best_params
    best_rmse   = study.best_value

    # Save all trial results
    trials_df = pd.DataFrame([
        {"trial": t.number, "cv_rmse": t.value, **t.params}
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]).sort_values("cv_rmse").reset_index(drop=True)
    trials_df.to_csv(
        os.path.join(RESULTS_DIR, "tuning_results_catboost.csv"), index=False
    )
    print(f"[main] All trials saved -> results/tuning_results_catboost.csv", flush=True)

    # Retrain best CatBoost on full training set
    print(f"\n[main] Retraining best CatBoost on full train set ({len(X_train)} rows)...",
          flush=True)
    best_model = CatBoostRegressor(**best_params, random_seed=RANDOM_SEED, verbose=0)
    best_model.fit(X_train, y_train)
    joblib.dump(
        {"model": best_model, "feature_cols": feature_cols,
         "model_name": "CatBoost_v1_tuned"},
        os.path.join(MODEL_DIR, "best_catboost.pkl")
    )
    print(f"[main] Saved -> models/best_catboost.pkl", flush=True)

    # Print best params block to paste into train_stacked_model.py
    print("\n" + "=" * 60, flush=True)
    print("TUNING COMPLETE", flush=True)
    print(f"Best CV RMSE: {best_rmse:.4f}", flush=True)
    print("\n>>> PASTE THIS BLOCK INTO train_stacked_model.py CONSTANTS: <<<",
          flush=True)
    print("-" * 60, flush=True)
    print(f"CATBOOST_ITERATIONS        = {best_params['iterations']}", flush=True)
    print(f"CATBOOST_LEARNING_RATE     = {best_params['learning_rate']}", flush=True)
    print(f"CATBOOST_DEPTH             = {best_params['depth']}", flush=True)
    print(f"CATBOOST_L2_LEAF_REG       = {best_params['l2_leaf_reg']}", flush=True)
    print(f"CATBOOST_BORDER_COUNT      = {best_params['border_count']}", flush=True)
    print(f"CATBOOST_BAGGING_TEMP      = {best_params['bagging_temperature']}", flush=True)
    print("-" * 60, flush=True)

    # Also save to a text file so you can find it in results/
    params_txt_path = os.path.join(RESULTS_DIR, "best_catboost_params.txt")
    with open(params_txt_path, "w") as f:
        f.write(f"Best CV RMSE: {best_rmse:.4f}\n")
        f.write(f"Study: {STUDY_NAME}\n")
        f.write(f"N trials: {N_TRIALS}\n\n")
        f.write("# Paste into train_stacked_model.py:\n")
        f.write(f"CATBOOST_ITERATIONS        = {best_params['iterations']}\n")
        f.write(f"CATBOOST_LEARNING_RATE     = {best_params['learning_rate']}\n")
        f.write(f"CATBOOST_DEPTH             = {best_params['depth']}\n")
        f.write(f"CATBOOST_L2_LEAF_REG       = {best_params['l2_leaf_reg']}\n")
        f.write(f"CATBOOST_BORDER_COUNT      = {best_params['border_count']}\n")
        f.write(f"CATBOOST_BAGGING_TEMP      = {best_params['bagging_temperature']}\n")
    print(f"\n[main] Best params also saved -> {params_txt_path}", flush=True)


if __name__ == "__main__":
    main()
