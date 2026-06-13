"""
measure_residual_correlations.py
--------------------------------
PURPOSE: After the ablation study finishes, compute pairwise Pearson correlation
         between every candidate model's OOF residuals. This is the empirical
         gate before finalising ensemble composition.

WHY THIS MATTERS:
  Adding a correlated model to an ensemble gives near-zero variance reduction.
  If XGB and CatBoost make the same mistakes on the same ILs (r > 0.90), stacking
  them gives almost no R² gain. We want models with r < 0.80 vs XGB.

INPUT:
  This script re-runs GroupKFold OOF predictions on the live dataset for all
  confirmed candidates (XGB, CatBoost, RF, LGB if available). It does NOT rely
  on ablation_study.py output files, so it can be run independently.

  Uses hyperparameters from:
    - XGB: Optuna v3 best (same as train_stacked_model.py)
    - CatBoost: read from results/best_catboost_params.txt if it exists,
                otherwise uses defaults
    - RF: Optuna v2 best (same as train_stacked_model.py)
    - LGB: uses defaults (or results/tuning_results_lgb.csv best if available)

OUTPUTS:
  results/residual_correlations.csv   -- pairwise Pearson r matrix
  figures/residual_correlation_heatmap.png  -- visual matrix

Run from project root after ablation + tune_catboost finish:
  python src/measure_residual_correlations.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

# -- Constants -----------------------------------------------------------------
TRAIN_CSV   = os.path.join("data", "processed", "train_set.csv")
RESULTS_DIR = "results"
FIGURES_DIR = "figures"

TARGET_COL         = "log_x2_CO2"
CONDITION_FEATURES = ["T_K", "P_kPa"]
CV_FOLDS           = 5
RANDOM_SEED        = 42

# Thresholds from ablation_study.py (kept consistent)
INCLUDE_THRESHOLD = 0.80   # r below this -> genuine diversity, include
EXCLUDE_THRESHOLD = 0.90   # r above this -> correlated copy, exclude

# XGB: Optuna v3 best params
XGB_PARAMS = dict(
    n_estimators     = 703,
    learning_rate    = 0.014,
    max_depth        = 5,
    min_child_weight = 1,
    gamma            = 0.21,
    subsample        = 0.776,
    colsample_bytree = 0.460,
    reg_alpha        = 1.2,
    reg_lambda       = 0.8,
    random_state     = RANDOM_SEED,
    n_jobs           = 2,
    verbosity        = 0,
    tree_method      = "hist",
)

# RF: Optuna v2 best params
RF_PARAMS = dict(
    n_estimators     = 515,
    max_features     = 0.5,
    min_samples_leaf = 7,
    max_depth        = None,
    random_state     = RANDOM_SEED,
    n_jobs           = -1,
)

# CatBoost defaults (overridden if results/best_catboost_params.txt exists)
CATBOOST_DEFAULT_PARAMS = dict(
    iterations    = 500,
    learning_rate = 0.05,
    depth         = 6,
    random_seed   = RANDOM_SEED,
    verbose       = 0,
)


def load_train_data() -> tuple:
    """
    Load training set and return X (features), y (target), il_smiles groups,
    and feature column names. NaN descriptor columns filled with median.
    """
    df = pd.read_csv(TRAIN_CSV)
    df = df.dropna(subset=CONDITION_FEATURES).copy()

    molecular_cols = [c for c in df.columns if c.startswith("cat_") or c.startswith("an_")]
    feature_cols   = molecular_cols + CONDITION_FEATURES

    # Fill NaN descriptors with median
    desc_cols = [c for c in molecular_cols if "_fp_" not in c]
    for col in desc_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    X         = df[feature_cols].values.astype(float)
    y         = df[TARGET_COL].values
    il_smiles = df["il_smiles"]

    print(f"[load] {len(df)} rows, {il_smiles.nunique()} unique ILs, "
          f"{len(feature_cols)} features", flush=True)
    return X, y, il_smiles, feature_cols


def load_catboost_params() -> dict:
    """
    Try to load tuned CatBoost params from results/best_catboost_params.txt.
    Falls back to defaults if the file doesn't exist.

    The params file is written by tune_catboost.py at the end of its run.
    """
    params_path = os.path.join(RESULTS_DIR, "best_catboost_params.txt")
    if not os.path.exists(params_path):
        print("[catboost_params] best_catboost_params.txt not found -- "
              "using defaults. Run tune_catboost.py first for tuned params.", flush=True)
        return CATBOOST_DEFAULT_PARAMS.copy()

    params = CATBOOST_DEFAULT_PARAMS.copy()
    with open(params_path, "r") as f:
        for line in f:
            line = line.strip()
            # Lines look like: CATBOOST_ITERATIONS = 742
            if "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                # Map param file keys to CatBoost API param names
                key_map = {
                    "CATBOOST_ITERATIONS":    "iterations",
                    "CATBOOST_LEARNING_RATE": "learning_rate",
                    "CATBOOST_DEPTH":         "depth",
                    "CATBOOST_L2_LEAF_REG":   "l2_leaf_reg",
                    "CATBOOST_BORDER_COUNT":  "border_count",
                    "CATBOOST_BAGGING_TEMP":  "bagging_temperature",
                }
                if key in key_map:
                    # Cast to correct type: int or float
                    api_key = key_map[key]
                    if api_key in ("iterations", "depth", "border_count"):
                        params[api_key] = int(float(val))
                    else:
                        params[api_key] = float(val)
    print(f"[catboost_params] Loaded tuned params: {params}", flush=True)
    return params


def oof_predictions(model, X: np.ndarray, y: np.ndarray,
                    il_smiles: pd.Series, model_name: str) -> np.ndarray:
    """
    Generate out-of-fold predictions for all training rows using GroupKFold.
    Each row's prediction comes from a model that never saw its IL during training.
    Returns array of OOF predictions, same length as y.
    """
    from sklearn.base import clone
    groups = il_smiles.values
    gkf    = GroupKFold(n_splits=CV_FOLDS)
    oof    = np.zeros(len(y))

    print(f"\n[oof] {model_name}: {CV_FOLDS}-fold GroupKFold OOF...", flush=True)

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        # Impute NaNs using train-fold median only (prevent leakage)
        imputer = SimpleImputer(strategy="median")
        X_tr = imputer.fit_transform(X[train_idx])
        X_val = imputer.transform(X[val_idx])

        fold_model = clone(model)
        fold_model.fit(X_tr, y[train_idx])
        oof[val_idx] = fold_model.predict(X_val)

        n_val_ils = len(set(groups[val_idx]))
        fold_r2   = r2_score(y[val_idx], oof[val_idx])
        print(f"  Fold {fold_idx+1}/{CV_FOLDS}: {n_val_ils} val ILs | R²={fold_r2:.3f}",
              flush=True)

    combined_r2 = r2_score(y, oof)
    print(f"  Combined OOF R²: {combined_r2:.4f}", flush=True)
    return oof


def compute_correlation_matrix(oof_dict: dict, y: np.ndarray) -> pd.DataFrame:
    """
    Compute pairwise Pearson correlation between every model's OOF residuals.
    A residual is (y_true - y_pred): large positive = model underestimated.
    High residual correlation between two models means they make the same errors
    on the same ILs -- adding the second model to the ensemble provides no gain.
    """
    model_names = list(oof_dict.keys())
    n = len(model_names)
    corr_matrix = np.zeros((n, n))

    for i, name_i in enumerate(model_names):
        res_i = y - oof_dict[name_i]
        for j, name_j in enumerate(model_names):
            res_j = y - oof_dict[name_j]
            r, _ = pearsonr(res_i, res_j)
            corr_matrix[i, j] = r

    return pd.DataFrame(corr_matrix, index=model_names, columns=model_names)


def print_ensemble_verdict(corr_df: pd.DataFrame, oof_dict: dict, y: np.ndarray):
    """
    Print a clear include/exclude recommendation for each non-XGB model,
    based on its residual correlation with XGB and its solo OOF R².
    """
    print("\n" + "=" * 65, flush=True)
    print("RESIDUAL CORRELATION ANALYSIS", flush=True)
    print(f"Include threshold: r < {INCLUDE_THRESHOLD} | Exclude threshold: r > {EXCLUDE_THRESHOLD}",
          flush=True)
    print("=" * 65, flush=True)

    # Print full matrix first
    print("\nFull pairwise residual correlation matrix:", flush=True)
    print(corr_df.round(3).to_string(), flush=True)

    # Per-model verdict vs XGB
    print("\nVerdicts (vs XGB as reference):", flush=True)
    xgb_name = "XGB"
    if xgb_name not in corr_df.columns:
        # Try partial match
        xgb_name = next((n for n in corr_df.columns if "XGB" in n), None)

    if xgb_name is None:
        print("  WARNING: XGB not found in OOF dict -- skipping verdict.", flush=True)
        return

    for model_name in corr_df.columns:
        if model_name == xgb_name:
            continue
        r = corr_df.loc[xgb_name, model_name]
        solo_r2 = r2_score(y, oof_dict[model_name])

        if r < INCLUDE_THRESHOLD:
            verdict = f"INCLUDE (genuine diversity)"
        elif r > EXCLUDE_THRESHOLD:
            verdict = f"EXCLUDE (correlated copy of XGB)"
        else:
            verdict = f"MARGINAL (include only if squeezing last R²)"

        print(f"  {model_name:30s} | solo R²={solo_r2:.4f}, r_vs_xgb={r:.3f} | {verdict}",
              flush=True)

    print("=" * 65, flush=True)


def plot_heatmap(corr_df: pd.DataFrame):
    """Save pairwise residual correlation heatmap to figures/."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    model_names = list(corr_df.columns)
    n = len(model_names)

    fig, ax = plt.subplots(figsize=(max(6, n + 1), max(5, n)))
    im = ax.imshow(corr_df.values, vmin=0, vmax=1, cmap="RdYlGn_r")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(model_names, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = corr_df.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if val > 0.7 else "black")

    plt.colorbar(im, ax=ax, label="Pearson r (residual correlation)")
    ax.set_title(
        "OOF Residual Correlation Matrix\n"
        "Low off-diagonal = diverse ensemble = better stacking gain",
        fontsize=10
    )
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "residual_correlation_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[figures] Saved: {out_path}", flush=True)


def main():
    """
    Load training data, run OOF predictions for all confirmed candidates,
    compute pairwise residual correlations, print verdicts.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 65, flush=True)
    print("RESIDUAL CORRELATION CHECK -- Ensemble Composition Gate", flush=True)
    print("=" * 65 + "\n", flush=True)

    X, y, il_smiles, feature_cols = load_train_data()

    # Build candidate models
    oof_dict = {}

    # XGB (reference)
    xgb_model = XGBRegressor(**XGB_PARAMS)
    oof_dict["XGB"] = oof_predictions(xgb_model, X, y, il_smiles, "XGB")

    # RF
    rf_model = RandomForestRegressor(**RF_PARAMS)
    oof_dict["RF"] = oof_predictions(rf_model, X, y, il_smiles, "RF")

    # CatBoost
    try:
        from catboost import CatBoostRegressor
        cat_params = load_catboost_params()
        cat_model  = CatBoostRegressor(**cat_params)
        oof_dict["CatBoost"] = oof_predictions(cat_model, X, y, il_smiles, "CatBoost")
    except ImportError:
        print("[main] CatBoost not installed -- skipping.", flush=True)

    # LightGBM (optional)
    try:
        from lightgbm import LGBMRegressor

        # Try loading tuned LGB params from the v4 tuning results
        lgb_params_path = os.path.join(RESULTS_DIR, "tuning_results_lgb.csv")
        if os.path.exists(lgb_params_path):
            lgb_trials = pd.read_csv(lgb_params_path).sort_values("cv_rmse")
            best_lgb_row = lgb_trials.iloc[0]
            # Extract only the model param columns (exclude trial, cv_rmse, model)
            exclude_cols = {"trial", "cv_rmse", "model"}
            lgb_best_params = {
                k: int(v) if k in ("n_estimators", "num_leaves", "min_child_samples",
                                    "bagging_freq")
                else float(v)
                for k, v in best_lgb_row.items()
                if k not in exclude_cols and not pd.isna(v)
            }
            print(f"[lgb] Using tuned params from {lgb_params_path}", flush=True)
            print(f"[lgb] Best params: {lgb_best_params}", flush=True)
        else:
            lgb_best_params = dict(
                n_estimators=500, learning_rate=0.05,
                num_leaves=31, min_child_samples=10,
                random_state=RANDOM_SEED, n_jobs=2, verbose=-1,
            )
            print("[lgb] No tuning results found -- using defaults.", flush=True)

        lgb_model = LGBMRegressor(
            **lgb_best_params,
            random_state=RANDOM_SEED, n_jobs=2, verbose=-1,
        )
        oof_dict["LightGBM"] = oof_predictions(lgb_model, X, y, il_smiles, "LightGBM")
    except ImportError:
        print("[main] LightGBM not installed -- skipping.", flush=True)

    # Compute correlation matrix and print verdicts
    corr_df = compute_correlation_matrix(oof_dict, y)
    print_ensemble_verdict(corr_df, oof_dict, y)

    # Save
    corr_df.round(4).to_csv(
        os.path.join(RESULTS_DIR, "residual_correlations.csv")
    )
    print(f"[main] Saved -> results/residual_correlations.csv", flush=True)

    plot_heatmap(corr_df)
    print("\n[main] DONE.", flush=True)


if __name__ == "__main__":
    main()
