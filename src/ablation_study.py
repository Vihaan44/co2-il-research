"""
ablation_study.py
-----------------
PURPOSE:
    Before expanding the stacking ensemble, test every candidate model to see
    if it genuinely earns its place. Two criteria must both be met:
      1. Solo GroupKFold CV R² is competitive (within ~0.10 of XGB baseline)
      2. Out-of-fold residuals are weakly correlated with XGB residuals (Pearson r < 0.85)
         -- if errors are highly correlated, the model is just a noisier copy of XGB
         and adding it to the ensemble provides almost no variance reduction.

CANDIDATE MODELS TESTED:
    - Extra Trees         (genuinely decorrelated from RF via random splits)
    - LightGBM            (leaf-wise GBDT, different convergence from XGB)
    - CatBoost            (symmetric trees, different regularization)
    - MLP (small)         (can learn feature interactions fingerprints miss)
    - Ridge on RDKit only (linear model in a completely different feature space)

    The existing ensemble members (RF, XGB) are included as reference baselines.

DECISION RULES (printed automatically):
    - Solo R² < (XGB_R2 - 0.10)  -> EXCLUDE: too weak to help even if decorrelated
    - Residual correlation > 0.90 -> EXCLUDE: correlated copy of XGB, no diversity gain
    - Residual correlation < 0.80 -> INCLUDE: genuine diversity
    - 0.80 <= r <= 0.90           -> MARGINAL: include only if solo R² is strong

OUTPUT:
    results/ablation_solo_cv.csv           -- solo CV R² and RMSE per model
    results/ablation_residual_corr.csv     -- pairwise residual correlations vs XGB
    figures/ablation_residual_heatmap.png  -- visual correlation matrix

INPUTS:
    data/processed/train_set.csv  (from build_dataset.py)
    data/processed/test_set.csv   (from build_dataset.py)

Run from project root:
    python src/ablation_study.py

NOTE ON CATBOOST / MLP:
    CatBoost and MLP are optional installs. If either is not installed,
    the script skips that model and prints a warning rather than crashing.
    Install with:
        pip install catboost
        pip install scikit-learn  (MLP is in sklearn, already installed)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for Codespaces
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Suppress sklearn convergence warnings during MLP CV -- expected on some folds
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# -- Constants ------------------------------------------------------------------
TRAIN_CSV   = os.path.join("data", "processed", "train_set.csv")
TEST_CSV    = os.path.join("data", "processed", "test_set.csv")
RESULTS_DIR = "results"
FIGURES_DIR = "figures"

TARGET_COL         = "log_x2_CO2"
CONDITION_FEATURES = ["T_K", "P_kPa"]
CV_FOLDS           = 5    # GroupKFold folds for OOF generation
RANDOM_SEED        = 42

# Decision thresholds -- see module docstring for rationale
RESIDUAL_CORR_INCLUDE_THRESHOLD  = 0.80   # below this -> include (genuine diversity)
RESIDUAL_CORR_EXCLUDE_THRESHOLD  = 0.90   # above this -> exclude (correlated copy)
SOLO_R2_MIN_GAP                  = 0.10   # must be within this gap of XGB R²

# XGB best hyperparameters from Optuna v2 (kept consistent with train_stacked_model.py)
XGB_N_ESTIMATORS    = 440
XGB_LEARNING_RATE   = 0.07481496458680137
XGB_MAX_DEPTH       = 5
XGB_SUBSAMPLE       = 0.7760131200377596
XGB_COLSAMPLE       = 0.45987026313463525
XGB_REG_ALPHA       = 0.6457288433886444
XGB_REG_LAMBDA      = 3.5815949001230535e-07
XGB_MIN_CHILD_WEIGHT = 10

# RF best hyperparameters from Optuna v2
RF_N_ESTIMATORS     = 515
RF_MAX_FEATURES     = 0.5
RF_MIN_SAMPLES_LEAF = 7


# -- Data loading ---------------------------------------------------------------

def load_data() -> tuple:
    """
    Load train set and return X (all features), y (target), il_smiles groups,
    X_desc (RDKit descriptor columns only), and full feature column list.

    We need X_desc separately because Ridge-on-descriptors uses only those ~40
    physicochemical features, not the 2048-bit Morgan fingerprints.
    """
    df = pd.read_csv(TRAIN_CSV)
    df = df.dropna(subset=CONDITION_FEATURES + [TARGET_COL]).copy()

    # All molecular features + T/P conditions
    cat_cols  = [c for c in df.columns if c.startswith("cat_")]
    an_cols   = [c for c in df.columns if c.startswith("an_")]
    all_mol   = cat_cols + an_cols
    all_feat  = all_mol + CONDITION_FEATURES

    # RDKit descriptor columns only (no fingerprint bits) for Ridge baseline
    # Descriptors are named things like cat_MolWt, an_TPSA etc -- not bit integers
    # Fingerprint bit columns are named cat_fp_0, cat_fp_1 ... an_fp_0 ...
    desc_cols = [c for c in all_mol
                 if not ("_fp_" in c)]
    desc_cols = desc_cols + CONDITION_FEATURES

    X       = df[all_feat].values.astype(float)
    X_desc  = df[desc_cols].values.astype(float) if desc_cols else None
    y       = df[TARGET_COL].values
    smiles  = df["il_smiles"]

    n_fp_bits = len([c for c in all_mol if "_fp_" in c])
    print(f"[load] Train rows: {len(df)} | Unique ILs: {smiles.nunique()}", flush=True)
    print(f"[load] Total features: {len(all_feat)} | "
          f"Fingerprint bits: {n_fp_bits} | "
          f"Descriptor cols: {len(desc_cols)}", flush=True)

    if X_desc is None or X_desc.shape[1] == 0:
        print("  WARNING: No descriptor columns found (no cat_/an_ non-fp columns). "
              "Ridge-on-descriptors will be skipped.", flush=True)

    return X, y, smiles, X_desc, desc_cols, all_feat


# -- Model definitions ----------------------------------------------------------

def build_candidate_models(X_desc_available: bool) -> list:
    """
    Return list of (name, model, use_descriptors_only) tuples.

    use_descriptors_only=True means this model trains on X_desc (RDKit features
    only) instead of full X (fingerprints + descriptors). This is intentional for
    Ridge -- it gives the ensemble a genuinely different feature-space signal.
    """
    models = []

    # --- Reference baselines (already in ensemble) ---
    models.append(("XGB (baseline)", XGBRegressor(
        n_estimators     = XGB_N_ESTIMATORS,
        learning_rate    = XGB_LEARNING_RATE,
        max_depth        = XGB_MAX_DEPTH,
        subsample        = XGB_SUBSAMPLE,
        colsample_bytree = XGB_COLSAMPLE,
        reg_alpha        = XGB_REG_ALPHA,
        reg_lambda       = XGB_REG_LAMBDA,
        min_child_weight = XGB_MIN_CHILD_WEIGHT,
        random_state     = RANDOM_SEED,
        n_jobs           = 2,
        verbosity        = 0,
        tree_method      = "hist",
    ), False))

    models.append(("RF (baseline)", RandomForestRegressor(
        n_estimators     = RF_N_ESTIMATORS,
        max_features     = RF_MAX_FEATURES,
        min_samples_leaf = RF_MIN_SAMPLES_LEAF,
        random_state     = RANDOM_SEED,
        n_jobs           = -1,
    ), False))

    # --- Candidates ---

    # Extra Trees: same API as RF but splits chosen randomly (not optimally)
    # This produces higher bias but lower variance, making errors decorrelated from RF
    models.append(("Extra Trees", ExtraTreesRegressor(
        n_estimators     = RF_N_ESTIMATORS,   # same tree count as RF for fair comparison
        max_features     = RF_MAX_FEATURES,
        min_samples_leaf = RF_MIN_SAMPLES_LEAF,
        random_state     = RANDOM_SEED,
        n_jobs           = -1,
    ), False))

    # LightGBM: leaf-wise tree growth (vs XGB's level-wise) -- genuinely different
    try:
        from lightgbm import LGBMRegressor
        models.append(("LightGBM", LGBMRegressor(
            n_estimators  = 500,
            learning_rate = 0.05,
            num_leaves    = 31,       # default; controls model complexity
            min_child_samples = 10,   # like XGB min_child_weight, prevents overfitting
            random_state  = RANDOM_SEED,
            n_jobs        = 2,
            verbose       = -1,
        ), False))
        print("[models] LightGBM found and added.", flush=True)
    except ImportError:
        print("[models] WARNING: lightgbm not installed. Skipping. "
              "Install with: pip install lightgbm", flush=True)

    # CatBoost: symmetric (oblivious) trees -- very different regularization from XGB/LGBM
    try:
        from catboost import CatBoostRegressor
        models.append(("CatBoost", CatBoostRegressor(
            iterations    = 500,
            learning_rate = 0.05,
            depth         = 6,
            random_seed   = RANDOM_SEED,
            verbose       = 0,        # suppress per-iteration output
        ), False))
        print("[models] CatBoost found and added.", flush=True)
    except ImportError:
        print("[models] WARNING: catboost not installed. Skipping. "
              "Install with: pip install catboost", flush=True)

    # MLP: small neural net -- can learn feature interactions tree models miss
    # Scaled inputs are critical; StandardScaler applied inside CV loop
    # Risk: may underfit on 211 ILs; solo R² will tell us if it's worth keeping
    models.append(("MLP", MLPRegressor(
        hidden_layer_sizes = (256, 128, 64),   # 3-layer network
        activation         = "relu",
        max_iter           = 500,
        early_stopping     = True,             # prevents overfitting; uses 10% val set
        validation_fraction = 0.1,
        n_iter_no_change   = 20,
        random_state       = RANDOM_SEED,
        learning_rate_init = 0.001,
    ), False))

    # Ridge on RDKit descriptors only (NOT fingerprints)
    # This is the only model operating in a fundamentally different feature space
    # -- the diversity comes from feature space difference, not model architecture
    if X_desc_available:
        models.append(("Ridge (descriptors only)", Ridge(alpha=1.0), True))
    else:
        print("[models] No descriptor columns detected -- Ridge-on-descriptors skipped.",
              flush=True)

    return models


# -- CV evaluation --------------------------------------------------------------

def run_oof_cv(model, X: np.ndarray, y: np.ndarray,
               il_smiles: pd.Series, model_name: str,
               scale_inputs: bool = False) -> tuple:
    """
    Run GroupKFold CV and return (oof_predictions, cv_r2_scores, cv_rmse_scores).

    out-of-fold predictions cover every training row exactly once, from a model
    that never saw that row's IL during training -- honest generalization estimate.

    scale_inputs=True applies StandardScaler per fold (required for MLP/Ridge).
    """
    groups = il_smiles.values
    gkf    = GroupKFold(n_splits=CV_FOLDS)
    oof    = np.zeros(len(y))     # collects OOF predictions for all rows
    fold_r2_scores   = []
    fold_rmse_scores = []

    print(f"\n[cv] {model_name} -- {CV_FOLDS}-fold GroupKFold...", flush=True)

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]

        # Scale if needed (MLP and Ridge are sensitive to feature scale;
        # tree models are invariant and don't need this)
        if scale_inputs:
            scaler = StandardScaler()
            X_fold_train = scaler.fit_transform(X_fold_train)
            X_fold_val   = scaler.transform(X_fold_val)   # fit only on train!

        # Clone the model for each fold to avoid state leakage between folds
        from sklearn.base import clone
        fold_model = clone(model)
        fold_model.fit(X_fold_train, y_fold_train)

        fold_preds = fold_model.predict(X_fold_val)
        oof[val_idx] = fold_preds

        fold_r2   = r2_score(y_fold_val, fold_preds)
        fold_rmse = np.sqrt(mean_squared_error(y_fold_val, fold_preds))
        fold_r2_scores.append(fold_r2)
        fold_rmse_scores.append(fold_rmse)

        n_val_ils = len(set(groups[val_idx]))
        print(f"  Fold {fold_idx+1}/{CV_FOLDS}: {n_val_ils} val ILs | "
              f"R²={fold_r2:.3f}, RMSE={fold_rmse:.3f}", flush=True)

    # Combined OOF metrics across all folds
    oof_r2   = r2_score(y, oof)
    oof_rmse = np.sqrt(mean_squared_error(y, oof))
    print(f"  OOF combined: R²={oof_r2:.4f}, RMSE={oof_rmse:.4f}", flush=True)

    return oof, oof_r2, oof_rmse


# -- Residual correlation analysis ----------------------------------------------

def compute_residual_correlations(oof_dict: dict, y: np.ndarray) -> pd.DataFrame:
    """
    Compute pairwise Pearson correlation between each model's OOF residuals and
    XGB's OOF residuals.

    High correlation (r > 0.90) means models make the same mistakes on the same ILs
    -- adding such a model to the ensemble provides almost no benefit.

    Returns a DataFrame with one row per non-XGB model showing its correlation
    with XGB residuals.
    """
    xgb_residuals = y - oof_dict["XGB (baseline)"]   # XGB errors on each row
    rows = []

    for model_name, oof_preds in oof_dict.items():
        model_residuals = y - oof_preds
        r, p_val = pearsonr(xgb_residuals, model_residuals)
        rows.append({
            "model":        model_name,
            "corr_with_xgb": round(r, 4),
            "p_value":       round(p_val, 4),
        })
        print(f"  {model_name:35s} vs XGB residuals: r={r:.4f}", flush=True)

    return pd.DataFrame(rows)


def compute_full_correlation_matrix(oof_dict: dict, y: np.ndarray) -> pd.DataFrame:
    """
    Compute the full pairwise residual correlation matrix across all models.
    Used for the heatmap figure.
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


# -- Verdict printing -----------------------------------------------------------

def print_verdicts(solo_results: list, corr_df: pd.DataFrame) -> None:
    """
    Print a clear include/exclude recommendation for each candidate model.

    Decision logic:
      1. If solo R² < (XGB_R2 - SOLO_R2_MIN_GAP) -> EXCLUDE (too weak)
      2. If residual_corr > RESIDUAL_CORR_EXCLUDE_THRESHOLD -> EXCLUDE (no diversity)
      3. If residual_corr < RESIDUAL_CORR_INCLUDE_THRESHOLD -> INCLUDE (genuine diversity)
      4. Otherwise -> MARGINAL (include only if you want to squeeze everything)
    """
    # Get XGB solo R² as reference
    xgb_r2 = next(r["oof_r2"] for r in solo_results if r["model"] == "XGB (baseline)")
    r2_threshold = xgb_r2 - SOLO_R2_MIN_GAP

    corr_lookup = {
        row["model"]: row["corr_with_xgb"]
        for _, row in corr_df.iterrows()
    }

    print("\n" + "="*65, flush=True)
    print("ENSEMBLE INCLUSION VERDICT", flush=True)
    print(f"  Reference XGB OOF R²: {xgb_r2:.4f}", flush=True)
    print(f"  Min R² to qualify:    {r2_threshold:.4f}  (XGB - {SOLO_R2_MIN_GAP})", flush=True)
    print(f"  Residual corr thresholds: INCLUDE < {RESIDUAL_CORR_INCLUDE_THRESHOLD}, "
          f"EXCLUDE > {RESIDUAL_CORR_EXCLUDE_THRESHOLD}", flush=True)
    print("="*65, flush=True)

    for result in solo_results:
        name    = result["model"]
        solo_r2 = result["oof_r2"]
        corr    = corr_lookup.get(name, 1.0)   # default 1.0 if not found (XGB itself)

        # Skip the baselines -- they're already in the ensemble
        if name in ("XGB (baseline)", "RF (baseline)"):
            print(f"  {name:35s} | ALREADY IN ENSEMBLE", flush=True)
            continue

        # Apply decision rules
        if solo_r2 < r2_threshold:
            verdict = "EXCLUDE -- solo R² too weak"
        elif corr > RESIDUAL_CORR_EXCLUDE_THRESHOLD:
            verdict = "EXCLUDE -- residuals too correlated with XGB"
        elif corr < RESIDUAL_CORR_INCLUDE_THRESHOLD:
            verdict = f"INCLUDE -- genuine diversity (r={corr:.3f})"
        else:
            verdict = f"MARGINAL -- include if squeezing last R² (r={corr:.3f})"

        print(f"  {name:35s} | R²={solo_r2:.4f}, corr={corr:.3f} | {verdict}",
              flush=True)

    print("="*65 + "\n", flush=True)


# -- Figures --------------------------------------------------------------------

def plot_residual_heatmap(corr_matrix_df: pd.DataFrame) -> None:
    """
    Save a heatmap of pairwise residual correlations across all models.
    Low off-diagonal values = diverse ensemble; high values = redundant models.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_matrix_df.values, vmin=0, vmax=1, cmap="RdYlGn_r")

    model_names = list(corr_matrix_df.columns)
    ax.set_xticks(range(len(model_names)))
    ax.set_yticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(model_names, fontsize=8)

    # Annotate cells with correlation value
    for i in range(len(model_names)):
        for j in range(len(model_names)):
            val = corr_matrix_df.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if val > 0.7 else "black")

    plt.colorbar(im, ax=ax, label="Pearson r (residual correlation)")
    ax.set_title("OOF Residual Correlation Matrix\n"
                 "(lower off-diagonal = more diverse ensemble)", fontsize=10)
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "ablation_residual_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[figures] Saved: {out_path}", flush=True)


def plot_solo_r2_bar(solo_results: list) -> None:
    """Bar chart of each model's solo OOF R² -- quick visual comparison."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    names = [r["model"] for r in solo_results]
    r2s   = [r["oof_r2"] for r in solo_results]

    # Color baselines differently
    colors = ["#4CAF50" if "baseline" in n else "#2196F3" for n in names]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(names, r2s, color=colors)
    ax.set_xlabel("OOF R² (GroupKFold CV)")
    ax.set_title("Solo Model Performance — Ablation Study\n"
                 "(green = existing ensemble members, blue = candidates)")
    ax.set_xlim(0, 1.0)

    # Annotate bars with R² value
    for bar, val in zip(bars, r2s):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "ablation_solo_r2.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[figures] Saved: {out_path}", flush=True)


# -- Main -----------------------------------------------------------------------

def main():
    """
    Full ablation pipeline:
    1. Load training data
    2. Build all candidate models
    3. Run GroupKFold OOF CV for each model, collecting residuals
    4. Compute residual correlations vs XGB
    5. Print verdicts and save results/figures
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 65, flush=True)
    print("ABLATION STUDY: Which models deserve a place in the ensemble?", flush=True)
    print("=" * 65 + "\n", flush=True)

    # Step 1: Load data
    X, y, il_smiles, X_desc, desc_cols, all_feat = load_data()
    X_desc_available = X_desc is not None and X_desc.shape[1] > len(CONDITION_FEATURES)

    print(f"\n[main] Feature matrix shape: {X.shape}", flush=True)
    if X_desc_available:
        print(f"[main] Descriptor-only matrix shape: {X_desc.shape}", flush=True)

    # Step 2: Build candidate list
    candidates = build_candidate_models(X_desc_available)
    print(f"\n[main] {len(candidates)} models to evaluate: "
          f"{[n for n, _, _ in candidates]}", flush=True)

    # Step 3: OOF CV for each model
    oof_dict     = {}   # model_name -> oof predictions array
    solo_results = []   # list of {model, oof_r2, oof_rmse}

    for model_name, model, use_desc_only in candidates:
        # Select feature matrix
        if use_desc_only and X_desc_available:
            X_input = X_desc
            print(f"\n[cv] {model_name} using descriptor-only features "
                  f"({X_desc.shape[1]} cols)", flush=True)
        else:
            X_input = X

        # MLP and Ridge need scaled inputs; tree models don't
        needs_scaling = isinstance(model, (MLPRegressor, Ridge))

        oof_preds, oof_r2, oof_rmse = run_oof_cv(
            model, X_input, y, il_smiles, model_name,
            scale_inputs=needs_scaling
        )

        oof_dict[model_name] = oof_preds
        solo_results.append({
            "model":    model_name,
            "oof_r2":   round(oof_r2, 4),
            "oof_rmse": round(oof_rmse, 4),
        })

    # Step 4: Residual correlations vs XGB
    print("\n[corr] Computing OOF residual correlations vs XGB baseline...", flush=True)
    corr_df = compute_residual_correlations(oof_dict, y)

    # Full pairwise matrix for heatmap
    corr_matrix_df = compute_full_correlation_matrix(oof_dict, y)

    # Step 5: Print verdicts
    print_verdicts(solo_results, corr_df)

    # Step 6: Save results
    solo_df = pd.DataFrame(solo_results)
    solo_df = solo_df.sort_values("oof_r2", ascending=False)

    solo_df.to_csv(
        os.path.join(RESULTS_DIR, "ablation_solo_cv.csv"), index=False)
    corr_df.to_csv(
        os.path.join(RESULTS_DIR, "ablation_residual_corr.csv"), index=False)
    corr_matrix_df.to_csv(
        os.path.join(RESULTS_DIR, "ablation_residual_corr_matrix.csv"))

    print("\n[main] Solo CV results:", flush=True)
    print(solo_df.to_string(index=False), flush=True)

    print("\n[main] Residual correlations vs XGB:", flush=True)
    print(corr_df.to_string(index=False), flush=True)

    # Step 7: Figures
    plot_solo_r2_bar(solo_results)
    plot_residual_heatmap(corr_matrix_df)

    print("\n[main] Saved:", flush=True)
    print("  results/ablation_solo_cv.csv", flush=True)
    print("  results/ablation_residual_corr.csv", flush=True)
    print("  results/ablation_residual_corr_matrix.csv", flush=True)
    print("  figures/ablation_solo_r2.png", flush=True)
    print("  figures/ablation_residual_heatmap.png", flush=True)
    print("\n[main] DONE. Read the ENSEMBLE INCLUSION VERDICT above.", flush=True)


if __name__ == "__main__":
    main()
