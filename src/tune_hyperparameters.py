"""
tune_hyperparameters.py
-----------------------
PURPOSE: Find better hyperparameters for XGBoost using Optuna (Bayesian
         optimization), evaluated with GroupKFold CV.

         RF tuning was removed — XGBoost consistently outperforms RF
         (CV R² 0.67 vs 0.41 on 211-IL dataset) and no hyperparameter
         setting realistically closes that gap on our feature matrix.

CRASH RECOVERY:
  Optuna saves every completed trial to a SQLite database (logs/optuna.db).
  If the process is interrupted (laptop sleep, codespace timeout), simply
  re-run this script -- it will load completed trials from the database and
  continue from where it left off. No trials are re-run.

  To start completely fresh (discard all prior trials):
    rm logs/optuna.db
    python src/tune_hyperparameters.py

WHY GroupKFold INSIDE TUNING:
  Every candidate hyperparameter set is evaluated by GroupKFold (grouped by
  il_smiles). This prevents the tuner from overfitting to measurement repeats
  of the same IL -- without this, the "best" parameters would be those that
  best memorize repeated T/P conditions, not those that best generalize to
  new ILs.

XGB MEMORY CONSTRAINTS:
  The codespace has ~8GB RAM. We allow max_depth up to 9 and n_estimators
  up to 800 -- prior tuning on the 168-IL dataset found depth=9, n_est=783
  was optimal without OOM. tree_method=hist is used throughout to reduce
  memory vs the exact method.

KEY REGULARIZATION PARAMS:
  min_child_weight: minimum sum of instance weight in a leaf. Higher values
  prevent the tree from learning splits that cover very few ILs -- the most
  direct way to combat overfitting on our sparse 4114-feature / 211-IL matrix.
  Searched over 1-20.

  reg_alpha (L1) and reg_lambda (L2): penalize large leaf weights. Both
  searched log-uniformly over [1e-8, 10.0].

STUDY NAME:
  v2 = 211-IL expanded dataset (delete logs/optuna.db before running).
  v1 = old 168-IL dataset (obsolete).

OUTPUTS:
  results/tuning_results_xgb.csv   -- all XGBoost trials with their CV RMSE
  results/best_hyperparams.csv     -- best params
  models/forward_model_tuned.pkl   -- retrained best model on full train set
  logs/optuna.db                   -- SQLite checkpoint (auto-resume on crash)

INPUT:
  data/processed/train_set.csv  (from build_dataset.py)

Run from project root:
    rm logs/optuna.db   # required if switching from v1 to v2
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
from xgboost import XGBRegressor

sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# -- Constants -----------------------------------------------------------------
TRAIN_CSV        = os.path.join("data", "processed", "train_set.csv")
MODEL_DIR        = "models"
RESULTS_DIR      = "results"
LOGS_DIR         = "logs"
TUNED_MODEL_PATH = os.path.join(MODEL_DIR, "forward_model_tuned.pkl")

OPTUNA_DB_PATH = os.path.join(LOGS_DIR, "optuna.db")
OPTUNA_DB_URL  = f"sqlite:///{OPTUNA_DB_PATH}"

TARGET_COL         = "log_x2_CO2"
CONDITION_FEATURES = ["T_K", "P_kPa"]
CV_FOLDS           = 5
RANDOM_SEED        = 42

N_TRIALS_XGB = 50

# v2 = 211-IL expanded dataset. Bump version when restarting on a new dataset.
STUDY_NAME_XGB = "xgb_co2_il_v2"

# XGB search bounds (v2: raised from depth=6/n_est=500 after confirming
# depth=9, n_est=783 ran without OOM on 168-IL dataset).
XGB_MAX_DEPTH_MAX        = 9
XGB_N_EST_MAX            = 800
XGB_MIN_CHILD_WEIGHT_MAX = 20   # higher = more regularization against IL-level overfitting


def load_train_data() -> tuple:
    """
    Load the training set and return X, y, il_smiles groups, and feature_cols.
    Drops rows with missing T_K or P_kPa defensively.
    """
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(
            f"Train set not found at {TRAIN_CSV}. Run build_dataset.py first."
        )
    df = pd.read_csv(TRAIN_CSV)
    df = df.dropna(subset=CONDITION_FEATURES).copy()

    molecular_cols = [c for c in df.columns if c.startswith("cat_") or c.startswith("an_")]
    feature_cols   = molecular_cols + CONDITION_FEATURES

    print(f"[load_train_data] {df.shape[0]} rows, {df['il_smiles'].nunique()} unique ILs",
          flush=True)
    print(f"[load_train_data] {len(feature_cols)} features", flush=True)

    X         = df[feature_cols].values
    y         = df[TARGET_COL].values
    il_smiles = df["il_smiles"]
    return X, y, il_smiles, feature_cols


def grouped_cv_rmse(model, X: np.ndarray, y: np.ndarray,
                    il_smiles: pd.Series) -> float:
    """
    Compute mean RMSE across GroupKFold folds (grouped by il_smiles).
    This is the objective Optuna minimizes.
    """
    group_kfold = GroupKFold(n_splits=CV_FOLDS)
    groups      = il_smiles.values
    rmse_scores = []

    for train_idx, val_idx in group_kfold.split(X, y, groups):
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[val_idx])
        rmse_scores.append(np.sqrt(mean_squared_error(y[val_idx], y_pred)))

    return float(np.mean(rmse_scores))


def xgb_objective(trial, X: np.ndarray, y: np.ndarray,
                  il_smiles: pd.Series) -> float:
    """
    Optuna objective for XGBoost -- returns GroupKFold CV RMSE.

    v2 search space additions vs v1:
      min_child_weight: 1-20 (new) -- key regularizer for IL-level overfitting.
        Minimum sum of instance weight required to create a new leaf split.
        Higher values = fewer, more general splits = less memorization of
        individual IL structure patterns. This directly addresses the 0.25
        CV/test R2 gap observed after training on the 211-IL dataset.

      max_depth: up to 9 (was 6) -- prior tuning confirmed depth=9 is safe.
      n_estimators: up to 800 (was 500) -- prior tuning found n_est=783 optimal.
      reg_alpha, reg_lambda: unchanged (L1/L2 leaf weight penalties).
    """
    n_estimators      = trial.suggest_int("n_estimators", 100, XGB_N_EST_MAX)
    learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    max_depth         = trial.suggest_int("max_depth", 3, XGB_MAX_DEPTH_MAX)
    subsample         = trial.suggest_float("subsample", 0.5, 1.0)
    colsample         = trial.suggest_float("colsample_bytree", 0.3, 1.0)
    reg_alpha         = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)
    reg_lambda        = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
    min_child_weight  = trial.suggest_int("min_child_weight", 1, XGB_MIN_CHILD_WEIGHT_MAX)

    model = XGBRegressor(
        n_estimators=n_estimators, learning_rate=learning_rate,
        max_depth=max_depth, subsample=subsample,
        colsample_bytree=colsample, reg_alpha=reg_alpha, reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        random_state=RANDOM_SEED,
        n_jobs=2,          # limit to 2 cores (not -1) to cap memory use per trial
        verbosity=0,
        tree_method="hist",  # histogram method uses less memory than exact
    )
    rmse = grouped_cv_rmse(model, X, y, il_smiles)
    print(f"  XGB Trial {trial.number:3d}: RMSE={rmse:.4f} | "
          f"n_est={n_estimators}, lr={learning_rate:.4f}, depth={max_depth}, "
          f"mcw={min_child_weight}, alpha={reg_alpha:.2e}, lambda={reg_lambda:.2e}",
          flush=True)
    return rmse


def run_study(study_name: str, objective_fn,
              n_trials: int, X: np.ndarray, y: np.ndarray,
              il_smiles: pd.Series) -> tuple:
    """
    Run (or resume) an Optuna study backed by SQLite.
    load_if_exists=True resumes from DB after crashes.
    """
    os.makedirs(LOGS_DIR, exist_ok=True)

    study = optuna.create_study(
        study_name    = study_name,
        storage       = OPTUNA_DB_URL,
        direction     = "minimize",
        sampler       = optuna.samplers.TPESampler(seed=RANDOM_SEED),
        load_if_exists= True,
    )

    n_completed = len([t for t in study.trials
                       if t.state == optuna.trial.TrialState.COMPLETE])
    n_remaining = max(0, n_trials - n_completed)

    print(f"\n[run_study] XGBoost: {n_completed} trials already done, "
          f"{n_remaining} remaining (target={n_trials})", flush=True)

    if n_remaining == 0:
        print(f"[run_study] XGBoost: already complete -- loading from DB.",
              flush=True)
    else:
        study.optimize(
            lambda trial: objective_fn(trial, X, y, il_smiles),
            n_trials = n_remaining,
            n_jobs   = 1,
        )

    best_params = study.best_params
    best_rmse   = study.best_value
    print(f"\n[run_study] XGBoost best CV RMSE: {best_rmse:.4f}", flush=True)
    print(f"[run_study] Best params: {best_params}", flush=True)

    trials_df = pd.DataFrame([
        {"trial": t.number, "cv_rmse": t.value, **t.params}
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]).sort_values("cv_rmse").reset_index(drop=True)

    return best_params, trials_df


def retrain_best_model(best_params_xgb: dict,
                       X_train: np.ndarray, y_train: np.ndarray,
                       feature_cols: list) -> tuple:
    """Retrain XGBoost with best tuned params on the full training set."""
    model_name = "XGBoost_tuned"
    print(f"\n[retrain] Fitting {model_name} on full train set ...", flush=True)
    best_model = XGBRegressor(
        **best_params_xgb, random_state=RANDOM_SEED,
        n_jobs=2, verbosity=0, tree_method="hist",
    )
    best_model.fit(X_train, y_train)
    return best_model, model_name


def main():
    """Main: load -> tune XGB -> retrain -> save."""
    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,    exist_ok=True)

    print(f"[main] Optuna checkpoint DB: {OPTUNA_DB_PATH}", flush=True)
    print(f"[main] Study: {STUDY_NAME_XGB} (v2 = 211-IL dataset)", flush=True)
    print(f"[main] XGB search bounds: max_depth<={XGB_MAX_DEPTH_MAX}, "
          f"n_est<={XGB_N_EST_MAX}, min_child_weight<=20, tree_method=hist",
          flush=True)
    print(f"[main] RF tuning removed -- XGB CV R2 0.67 vs RF 0.41; gap is not closable by tuning.",
          flush=True)
    print(f"[main] If interrupted, re-run to resume from checkpoint.", flush=True)

    X_train, y_train, il_smiles_train, feature_cols = load_train_data()

    # XGB tuning (resumes from DB if partially done)
    best_params_xgb, trials_xgb = run_study(
        STUDY_NAME_XGB, xgb_objective, N_TRIALS_XGB,
        X_train, y_train, il_smiles_train
    )
    trials_xgb["model"] = "XGBoost"
    trials_xgb.to_csv(os.path.join(RESULTS_DIR, "tuning_results_xgb.csv"), index=False)
    print(f"[main] XGB results saved -> results/tuning_results_xgb.csv", flush=True)

    best_cv_xgb = trials_xgb.iloc[0]["cv_rmse"]

    best_hyperparams = pd.DataFrame([
        {"model": "XGBoost", "best_cv_rmse": best_cv_xgb, **best_params_xgb},
    ])
    best_hyperparams.to_csv(os.path.join(RESULTS_DIR, "best_hyperparams.csv"), index=False)
    print("\n=== BEST HYPERPARAMETERS ===", flush=True)
    print(best_hyperparams.to_string(index=False), flush=True)

    best_model, model_name = retrain_best_model(
        best_params_xgb, X_train, y_train, feature_cols
    )
    joblib.dump({"model": best_model, "feature_cols": feature_cols,
                 "model_name": model_name}, TUNED_MODEL_PATH)
    print(f"[main] Tuned model saved -> {TUNED_MODEL_PATH}", flush=True)
    print("[main] DONE.", flush=True)


if __name__ == "__main__":
    main()
