"""
nested_cv_model_comparison.py
------------------------------
PURPOSE: Determine whether RF's win over XGB/CatBoost on the single train/test
         split (RF test R²=0.697 vs CatBoost 0.554) is a STRUCTURAL difference
         in OOD generalization, or an ARTIFACT of one particular 60-IL split.

WHY THIS MATTERS:
  With only 239 unique ILs, a single train/test split is a fragile basis for
  picking a final model -- a different random split could easily flip the
  ranking. This script runs 5 independent "outer" splits (5-fold GroupKFold,
  grouped by il_smiles) and reports the DISTRIBUTION of test R² for each model
  across all 5 splits, not just one number.

  If RF wins consistently across all 5 outer folds -> structural, ship RF.
  If the ranking flips between folds -> the single-split result was noise,
  and we need a different decision process (e.g. average over folds, or
  pick based on which model has the smallest variance).

METHOD -- Nested GroupKFold:
  Outer loop (5 folds): rotates which ILs are held out as the "test" set for
  that fold. This mimics re-running the original train/test split 5 different
  ways, all grouped by IL so no IL ever appears in both outer-train and
  outer-test within a fold.

  For each outer fold:
    1. Take the outer-train ILs, fit RF / XGB / CatBoost (current tuned params)
    2. Predict on the held-out outer-test ILs
    3. Record R², RMSE per model
    4. Record per-(IL, row) residuals for every model -> used later for
       per-IL error analysis (which ILs are hard, and for which model)

  This reuses the exact same hyperparameters already tuned via Optuna --
  this script does NOT re-tune. It only asks "how stable is the generalization
  ranking across different OOD test sets."

OUTPUTS:
  results/nested_cv_comparison.csv       -- per-fold, per-model R²/RMSE
  results/nested_cv_per_il_residuals.csv -- per-IL, per-model residuals (for
                                             the next-step error analysis)
  figures/nested_cv_boxplot.png          -- boxplot of test R² spread per model

Run from project root:
  nohup python src/nested_cv_model_comparison.py > logs/nested_cv.log 2>&1 &
Then monitor:
  tail -f logs/nested_cv.log
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

# -- Constants -----------------------------------------------------------------
# We combine train_set.csv and test_set.csv into one pool, then re-split it
# 5 ways. This is necessary because nested CV needs to rotate which ILs are
# "held out" -- if we only used the original train_set.csv, we'd never test
# on the 60 ILs that were already set aside as the fixed test set.
TRAIN_CSV   = os.path.join("data", "processed", "train_set.csv")
TEST_CSV    = os.path.join("data", "processed", "test_set.csv")
RESULTS_DIR = "results"
FIGURES_DIR = "figures"

TARGET_COL         = "log_x2_CO2"
CONDITION_FEATURES = ["T_K", "P_kPa"]
N_OUTER_FOLDS       = 5    # 5 independent train/test splits, grouped by IL
RANDOM_SEED         = 42

# -- RF: Optuna v2 best params (same as train_stacked_model.py) ----------------
RF_PARAMS = dict(
    n_estimators     = 515,
    max_features     = 0.5,
    min_samples_leaf = 7,
    max_depth        = None,
    random_state     = RANDOM_SEED,
    n_jobs           = -1,
)

# -- XGB: Optuna v3 best params (same as train_stacked_model.py) ---------------
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

# -- CatBoost: Optuna v1 best params (same as train_stacked_model.py) ----------
CATBOOST_PARAMS = dict(
    iterations          = 432,
    learning_rate       = 0.13184562661431584,
    depth               = 7,
    l2_leaf_reg          = 10.894623708254054,
    border_count         = 121,
    bagging_temperature  = 0.9280224125290069,
    random_seed          = RANDOM_SEED,
    verbose              = 0,
)


def load_full_pool() -> tuple:
    """
    Load and combine train_set.csv + test_set.csv into one pool of all
    available ILs. We need the full pool because nested CV must rotate
    through different held-out ILs -- the original fixed test set is only
    one possible 60-IL holdout among many.

    Returns X (features), y (target), il_smiles groups, and feature column names.
    NaN descriptor columns are filled with the column median computed over
    the FULL pool here (acceptable because this script is for model comparison
    diagnostics, not the final production model -- the final model still uses
    fold-local imputation, see train_stacked_model.py).
    """
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)
    full_df  = pd.concat([train_df, test_df], ignore_index=True)
    full_df  = full_df.dropna(subset=CONDITION_FEATURES).copy()

    molecular_cols = [c for c in full_df.columns if c.startswith("cat_") or c.startswith("an_")]
    feature_cols   = molecular_cols + CONDITION_FEATURES

    # Fill NaN descriptors with column median (Gasteiger charge edge cases)
    desc_cols = [c for c in molecular_cols if "_fp_" not in c]
    for col in desc_cols:
        if full_df[col].isna().any():
            full_df[col] = full_df[col].fillna(full_df[col].median())

    X         = full_df[feature_cols].values.astype(float)
    y         = full_df[TARGET_COL].values
    il_smiles = full_df["il_smiles"]

    print(f"[load] Combined pool: {len(full_df)} rows, "
          f"{il_smiles.nunique()} unique ILs, {len(feature_cols)} features", flush=True)
    return X, y, il_smiles, feature_cols


def make_rf() -> RandomForestRegressor:
    """Instantiate RF with the same tuned hyperparameters used in the final ensemble."""
    return RandomForestRegressor(**RF_PARAMS)


def make_xgb() -> XGBRegressor:
    """Instantiate XGB with the same tuned hyperparameters used in the final ensemble."""
    return XGBRegressor(**XGB_PARAMS)


def make_catboost():
    """
    Instantiate CatBoost with the same tuned hyperparameters used in the final
    ensemble. Returns None if catboost is not installed.
    """
    try:
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**CATBOOST_PARAMS)
    except ImportError:
        print("[WARNING] catboost not installed -- skipping CatBoost in comparison.",
              flush=True)
        return None


def run_outer_fold(fold_idx: int, X: np.ndarray, y: np.ndarray,
                    il_smiles: pd.Series, train_idx: np.ndarray,
                    test_idx: np.ndarray, use_catboost: bool) -> tuple:
    """
    Train RF, XGB, and (if available) CatBoost on one outer-train split,
    evaluate on the corresponding outer-test split.

    Returns:
      fold_metrics: list of dicts with {fold, model, test_r2, test_rmse}
      per_il_rows:  list of dicts with {fold, model, il_smiles, y_true, y_pred,
                    residual} for EVERY row in the outer-test set -- this is
                    the raw material for the next step's per-IL error analysis.
    """
    # Impute NaNs using the outer-train fold only (no leakage from outer-test ILs)
    imputer = SimpleImputer(strategy="median")
    X_train_fold = imputer.fit_transform(X[train_idx])
    X_test_fold  = imputer.transform(X[test_idx])
    y_train_fold = y[train_idx]
    y_test_fold  = y[test_idx]
    smiles_test_fold = il_smiles.values[test_idx]

    fold_metrics  = []
    per_il_rows   = []

    models_to_run = {
        "RF":  make_rf(),
        "XGB": make_xgb(),
    }
    if use_catboost:
        cat_model = make_catboost()
        if cat_model is not None:
            models_to_run["CatBoost"] = cat_model

    for model_name, model in models_to_run.items():
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_test_fold)

        test_r2   = r2_score(y_test_fold, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test_fold, y_pred))

        fold_metrics.append({
            "fold": fold_idx, "model": model_name,
            "test_r2": test_r2, "test_rmse": test_rmse,
            "n_test_rows": len(test_idx),
            "n_test_ils": len(set(smiles_test_fold)),
        })

        # Record every row's prediction for later per-IL error analysis
        for row_idx in range(len(test_idx)):
            per_il_rows.append({
                "fold":      fold_idx,
                "model":     model_name,
                "il_smiles": smiles_test_fold[row_idx],
                "y_true":    y_test_fold[row_idx],
                "y_pred":    y_pred[row_idx],
                "residual":  y_pred[row_idx] - y_test_fold[row_idx],
            })

        print(f"  Fold {fold_idx+1}/{N_OUTER_FOLDS} | {model_name:10s} | "
              f"test R²={test_r2:.4f}, RMSE={test_rmse:.4f} | "
              f"{len(set(smiles_test_fold))} test ILs", flush=True)

    return fold_metrics, per_il_rows


def summarize_results(all_fold_metrics: list) -> tuple:
    """
    Aggregate per-fold metrics into mean ± std test R² per model.
    This is the key robustness statistic: a model whose test R² varies wildly
    across folds is unreliable, even if its mean looks good.
    """
    metrics_df = pd.DataFrame(all_fold_metrics)

    summary = metrics_df.groupby("model").agg(
        mean_test_r2   = ("test_r2", "mean"),
        std_test_r2    = ("test_r2", "std"),
        min_test_r2    = ("test_r2", "min"),
        max_test_r2    = ("test_r2", "max"),
        mean_test_rmse = ("test_rmse", "mean"),
    ).reset_index().sort_values("mean_test_r2", ascending=False)

    print("\n" + "=" * 70, flush=True)
    print("NESTED CV SUMMARY (5 independent outer folds, GroupKFold by IL)", flush=True)
    print("=" * 70, flush=True)
    print(summary.to_string(index=False), flush=True)
    print("=" * 70, flush=True)

    # Robustness interpretation
    best_model = summary.iloc[0]
    print(f"\n[interpretation] Best mean test R²: {best_model['model']} "
          f"({best_model['mean_test_r2']:.4f} ± {best_model['std_test_r2']:.4f})",
          flush=True)
    if best_model["std_test_r2"] > 0.10:
        print(f"  WARNING: std > 0.10 -- this model's performance varies a lot "
              f"across folds. The single-split result may not be representative.",
              flush=True)
    else:
        print(f"  std < 0.10 -- this model's ranking is reasonably stable across folds.",
              flush=True)

    return metrics_df, summary


def plot_boxplot(metrics_df: pd.DataFrame):
    """
    Plot a boxplot of test R² distribution per model across the 5 outer folds.
    This is the visual answer to "is RF's win structural or a fluke" -- if the
    boxes don't overlap much, the ranking is robust; heavy overlap means the
    single-split result could easily have gone differently.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    model_order = metrics_df.groupby("model")["test_r2"].mean().sort_values(
        ascending=False).index.tolist()
    data_by_model = [metrics_df[metrics_df["model"] == m]["test_r2"].values
                      for m in model_order]

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(data_by_model, labels=model_order, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#3498db")
        patch.set_alpha(0.6)

    # Overlay individual fold points so small sample size (n=5) is visible
    for i, scores in enumerate(data_by_model):
        x_jitter = np.random.normal(i + 1, 0.04, size=len(scores))
        ax.scatter(x_jitter, scores, color="black", alpha=0.7, zorder=3, s=25)

    ax.set_ylabel("Test R² (held-out ILs)", fontsize=12)
    ax.set_title(
        "Model Generalization Across 5 Independent Outer Folds\n"
        "(GroupKFold by il_smiles -- each fold holds out different ILs)",
        fontsize=11
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "nested_cv_boxplot.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[figures] Saved: {out_path}", flush=True)


def main():
    """
    Full nested CV comparison pipeline:
    1. Load combined pool (train + test ILs together)
    2. Run 5-fold outer GroupKFold; for each fold, train+evaluate RF/XGB/CatBoost
    3. Aggregate mean ± std test R² per model
    4. Save per-fold metrics, per-IL residuals, and boxplot
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 70, flush=True)
    print("NESTED CV MODEL COMPARISON", flush=True)
    print(f"Outer folds: {N_OUTER_FOLDS} (GroupKFold by il_smiles)", flush=True)
    print("Goal: determine if RF's single-split test win is structural or noise", flush=True)
    print("=" * 70 + "\n", flush=True)

    X, y, il_smiles, feature_cols = load_full_pool()

    use_catboost = make_catboost() is not None
    if not use_catboost:
        print("[main] CatBoost unavailable -- comparison will only cover RF and XGB.",
              flush=True)

    outer_gkf = GroupKFold(n_splits=N_OUTER_FOLDS)
    groups    = il_smiles.values

    all_fold_metrics = []
    all_per_il_rows  = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_gkf.split(X, y, groups)):
        # Sanity check: no IL leaks across the outer split
        assert len(set(groups[train_idx]) & set(groups[test_idx])) == 0, \
            f"IL overlap in outer fold {fold_idx}!"

        fold_metrics, per_il_rows = run_outer_fold(
            fold_idx, X, y, il_smiles, train_idx, test_idx, use_catboost
        )
        all_fold_metrics.extend(fold_metrics)
        all_per_il_rows.extend(per_il_rows)

    # Aggregate and report
    metrics_df, summary_df = summarize_results(all_fold_metrics)

    # Save outputs
    metrics_df.to_csv(
        os.path.join(RESULTS_DIR, "nested_cv_comparison.csv"), index=False)
    print(f"\n[main] Saved -> results/nested_cv_comparison.csv", flush=True)

    per_il_df = pd.DataFrame(all_per_il_rows)
    per_il_df.to_csv(
        os.path.join(RESULTS_DIR, "nested_cv_per_il_residuals.csv"), index=False)
    print(f"[main] Saved -> results/nested_cv_per_il_residuals.csv "
          f"({len(per_il_df)} rows -- use this for per-IL error analysis)", flush=True)

    plot_boxplot(metrics_df)

    print("\n[main] DONE. Review results/nested_cv_comparison.csv and the boxplot", flush=True)
    print("       before deciding on a final model.", flush=True)


if __name__ == "__main__":
    main()
