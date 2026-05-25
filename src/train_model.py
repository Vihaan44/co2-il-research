"""
train_model.py
--------------
PURPOSE: Train an XGBoost regressor to predict log10(CO2 mole fraction
         solubility) from IL molecular features + temperature + pressure.

         RF was removed from the training loop. XGBoost consistently
         outperforms RF (CV R² 0.67 vs 0.41 on 211-IL dataset) and the gap
         is not closable by tuning. Running RF wastes time and adds no value.

XGB HYPERPARAMETERS:
  These come directly from src/tune_hyperparameters.py (Optuna Bayesian search,
  50 trials, GroupKFold CV on 239-IL dataset). Best CV RMSE = 0.3498.
  Key insight from tuning: min_child_weight=10 (high regularization) and
  max_depth=5 (shallow trees) were optimal -- confirms the model was overfitting
  on the sparse 4114-feature / 239-IL matrix with default settings.

WHY T_K AND P_kPa ARE INCLUDED AS FEATURES:
  The first run (structure-only) produced R² ≈ -0.10 (worse than predicting the mean).
  This happened because the same IL measured at 298K vs 350K has very different x2 values --
  molecular fingerprints alone cannot explain that variance.
  T_K and P_kPa are direct physical predictors of solubility (Henry's law: x2 ∝ P/H(T)).
  Adding them is scientifically correct -- real process models always condition on T and P.

  For competition framing: "Our model predicts CO2 solubility given the IL structure,
  temperature, and pressure -- mirroring real industrial process conditions."

CROSS-VALIDATION STRATEGY — Repeated GroupKFold:
  We use REPEATED GroupKFold: CV_FOLDS folds × CV_REPEATS different random IL orderings.
  This averages out the variance from unlucky fold composition.

  WHY REPEATED:
    With only ~239 ILs and 5 folds, each fold contains ~48 ILs. One fold might by
    chance contain a structurally unusual IL family with no training precedent (e.g.
    all phosphonium ILs in one fold), giving a very bad fold score that drags down
    the mean and inflates std. A single GroupKFold run (5 folds) can give std ≈ 0.44
    in R² purely from this sampling accident.
    Repeating 5 times (25 total folds) averages over different random IL assignments,
    giving a stable, trustworthy CV estimate. This is standard practice for small datasets
    in cheminformatics (Sheridan 2013, J. Chem. Inf. Model.).

  WHY GroupKFold (not plain KFold):
    Plain KFold can put measurements of the SAME IL in both train and val folds.
    Since one IL appears at many (T, P) conditions, this leaks structural information
    and makes CV scores optimistically biased. GroupKFold by il_smiles ensures every
    fold's val set contains only ILs not seen during that fold's training.

WHAT EACH STEP PRODUCES:
  1. Cross-validation results on train set → results/cv_results.csv
  2. Test set evaluation (RMSE, R², MAE)  → results/model_performance.csv
  3. Predicted vs actual values on test   → results/test_predictions.csv
  4. Feature importances (top 30)         → results/feature_importances.csv
  5. Best model + feature_cols saved      → models/forward_model.pkl

INPUTS:
  data/processed/train_set.csv   (from build_dataset.py)
  data/processed/test_set.csv    (from build_dataset.py)

Run from project root:
    python src/train_model.py
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle
from xgboost import XGBRegressor

# -- Constants -----------------------------------------------------------------
TRAIN_CSV   = os.path.join("data", "processed", "train_set.csv")
TEST_CSV    = os.path.join("data", "processed", "test_set.csv")
MODEL_DIR   = "models"
RESULTS_DIR = "results"
MODEL_PATH  = os.path.join(MODEL_DIR, "forward_model.pkl")

TARGET_COL         = "log_x2_CO2"       # log10-transformed mole fraction solubility
CONDITION_FEATURES = ["T_K", "P_kPa"]   # temperature (K) and pressure (kPa) from ILThermo
CV_FOLDS           = 5    # folds per repeat
CV_REPEATS         = 5    # number of times to repeat with different random shuffles
                           # total CV evaluations = CV_FOLDS * CV_REPEATS = 25
RANDOM_SEED        = 42

# XGBoost hyperparameters — from Optuna tuning (tune_hyperparameters.py),
# 50 trials, GroupKFold CV on 239-IL / 11033-row dataset. Best CV RMSE = 0.3498.
# min_child_weight=10 and max_depth=5 are the key regularization params that
# prevent overfitting on the sparse 4114-feature / 239-IL matrix.
XGB_N_ESTIMATORS    = 200
XGB_LEARNING_RATE   = 0.08252578137355096
XGB_MAX_DEPTH       = 5
XGB_SUBSAMPLE       = 0.7240833769291847
XGB_COLSAMPLE       = 0.4778802788064814
XGB_REG_ALPHA       = 0.08067465044762985
XGB_REG_LAMBDA      = 8.226130315805552e-05
XGB_MIN_CHILD_WEIGHT = 10


def load_split(path: str, label: str) -> tuple:
    """
    Load a train or test CSV and build the feature matrix X and target vector y.

    Features = Morgan fingerprint bits (cat_fp_* / an_fp_*) + RDKit descriptors
    (cat_mol_weight, cat_num_hbd, etc.) + T_K + P_kPa.
    Target   = log10(x2_CO2).

    Returns X (numpy array), y (numpy array), il_smiles (Series), feature_cols (list).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found at {path}. Run build_dataset.py first.")

    df = pd.read_csv(path)
    print(f"[load_split] {label}: {df.shape[0]} rows x {df.shape[1]} cols")

    molecular_cols = [c for c in df.columns if c.startswith("cat_") or c.startswith("an_")]
    fp_cols   = [c for c in molecular_cols if "_fp_" in c]
    desc_cols = [c for c in molecular_cols if "_fp_" not in c]
    print(f"[load_split] {label}: {len(fp_cols)} Morgan FP bits + "
          f"{len(desc_cols)} RDKit descriptors + {len(CONDITION_FEATURES)} T/P = "
          f"{len(molecular_cols) + len(CONDITION_FEATURES)} total features")

    missing_conditions = [c for c in CONDITION_FEATURES if c not in df.columns]
    if missing_conditions:
        raise ValueError(f"Missing condition columns {missing_conditions} in {path}.")

    feature_cols = molecular_cols + CONDITION_FEATURES
    print(f"[load_split] {label}: target range "
          f"[{df[TARGET_COL].min():.2f}, {df[TARGET_COL].max():.2f}]")

    # Drop NaN condition rows defensively (build_dataset.py should have cleaned these,
    # but if running against older data files, drop here rather than crash silently)
    n_missing_t = df["T_K"].isna().sum()
    n_missing_p = df["P_kPa"].isna().sum()
    if n_missing_t > 0 or n_missing_p > 0:
        print(f"[load_split] WARNING: {n_missing_t} rows missing T_K, "
              f"{n_missing_p} rows missing P_kPa -- dropping these rows.")
        df = df.dropna(subset=CONDITION_FEATURES).copy()
        print(f"[load_split] After drop: {len(df)} rows remain.")

    X         = df[feature_cols].values
    y         = df[TARGET_COL].values
    il_smiles = df["il_smiles"]
    return X, y, il_smiles, feature_cols


def repeated_group_kfold_cv(model, X_train: np.ndarray, y_train: np.ndarray,
                             il_smiles_train: pd.Series, model_name: str) -> dict:
    """
    Run repeated GroupKFold cross-validation (CV_FOLDS x CV_REPEATS).

    Each repeat shuffles the IL ordering before splitting into folds, so different
    ILs end up in the val fold each time. This averages out the variance from any
    single fold containing an unusually hard or easy IL family.

    Total evaluations = CV_FOLDS * CV_REPEATS (e.g. 5 x 5 = 25).
    We report both the per-repeat mean R² and the overall mean/std across all folds,
    so you can see how much the variance shrinks with repetition.

    WHY NOT sklearn's RepeatedGroupKFold:
      sklearn doesn't have RepeatedGroupKFold. We implement it manually by shuffling
      the unique IL list with a different seed each repeat, then re-assigning groups.
    """
    groups        = il_smiles_train.values
    unique_ils    = np.unique(groups)            # all unique IL SMILES in train set
    all_rmse      = []
    all_r2        = []
    repeat_r2_means = []  # mean R² per repeat, for reporting trend

    print(f"[cv] {model_name}: {CV_REPEATS} repeats x {CV_FOLDS} folds = "
          f"{CV_REPEATS * CV_FOLDS} total evaluations")

    for repeat_idx in range(CV_REPEATS):
        repeat_seed = RANDOM_SEED + repeat_idx  # different seed per repeat

        # Shuffle the unique ILs with this repeat's seed, then re-map to row indices.
        # This changes which ILs fall into which fold without changing the training data.
        shuffled_ils = shuffle(unique_ils, random_state=repeat_seed)

        # Build a new group array where each IL gets a fold assignment based on
        # its position in the shuffled list. This is equivalent to GroupKFold
        # on a freshly shuffled IL ordering.
        il_to_fold = {il: i % CV_FOLDS for i, il in enumerate(shuffled_ils)}
        fold_assignment = np.array([il_to_fold[il] for il in groups])

        repeat_rmse = []
        repeat_r2   = []

        for fold_idx in range(CV_FOLDS):
            # Val = rows where this IL is assigned to this fold
            val_mask   = fold_assignment == fold_idx
            train_mask = ~val_mask

            X_fold_train = X_train[train_mask]
            y_fold_train = y_train[train_mask]
            X_fold_val   = X_train[val_mask]
            y_fold_val   = y_train[val_mask]

            # Verify no IL overlap between fold train and val
            train_ils_fold = set(groups[train_mask])
            val_ils_fold   = set(groups[val_mask])
            assert len(train_ils_fold & val_ils_fold) == 0, "IL overlap in fold!"

            model.fit(X_fold_train, y_fold_train)
            y_fold_pred = model.predict(X_fold_val)

            fold_rmse = np.sqrt(mean_squared_error(y_fold_val, y_fold_pred))
            fold_r2   = r2_score(y_fold_val, y_fold_pred)
            repeat_rmse.append(fold_rmse)
            repeat_r2.append(fold_r2)

        repeat_mean_r2 = np.mean(repeat_r2)
        repeat_r2_means.append(repeat_mean_r2)
        all_rmse.extend(repeat_rmse)
        all_r2.extend(repeat_r2)

        print(f"  Repeat {repeat_idx+1}/{CV_REPEATS}: "
              f"mean R²={repeat_mean_r2:.4f}, "
              f"fold R² range=[{min(repeat_r2):.3f}, {max(repeat_r2):.3f}]")

    rmse_arr = np.array(all_rmse)
    r2_arr   = np.array(all_r2)

    print(f"\n[cv] {model_name} — Repeated GroupKFold ({CV_REPEATS}x{CV_FOLDS}) summary:")
    print(f"  CV RMSE  (log10 x2): {rmse_arr.mean():.4f} +/- {rmse_arr.std():.4f}")
    print(f"  CV R²              : {r2_arr.mean():.4f} +/- {r2_arr.std():.4f}")
    print(f"  Per-repeat R² means: {[f'{x:.3f}' for x in repeat_r2_means]}")
    print(f"  NOTE: std here reflects true fold-to-fold variability across 25 evaluations,")
    print(f"  not just one lucky/unlucky 5-fold split. This is the honest CV estimate.")

    return {
        "model":           model_name,
        "cv_rmse_mean":    float(rmse_arr.mean()),
        "cv_rmse_std":     float(rmse_arr.std()),
        "cv_r2_mean":      float(r2_arr.mean()),
        "cv_r2_std":       float(r2_arr.std()),
        "cv_protocol":     f"RepeatedGroupKFold({CV_REPEATS}x{CV_FOLDS}, grouped by il_smiles)",
        "n_cv_evaluations": CV_FOLDS * CV_REPEATS,
    }


def evaluate_on_test(model, X_test: np.ndarray, y_test: np.ndarray,
                     il_smiles_test: pd.Series, model_name: str) -> tuple:
    """
    Predict on the held-out test set and compute RMSE, R², and MAE.

    Reports error in both log10 units (what the model predicts) and mole fraction
    units (what chemists care about), via back-transform: x2 = 10^y.
    """
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)

    x2_true = 10 ** y_test
    x2_pred = 10 ** y_pred
    x2_rmse = np.sqrt(mean_squared_error(x2_true, x2_pred))

    print(f"\n[evaluate_on_test] {model_name} -- HELD-OUT TEST SET:")
    print(f"  RMSE  (log10 x2) : {rmse:.4f}")
    print(f"  R²               : {r2:.4f}")
    print(f"  MAE   (log10 x2) : {mae:.4f}")
    print(f"  RMSE  (x2 units) : {x2_rmse:.6f}   <- mole fraction error")

    predictions_df = pd.DataFrame({
        "il_smiles":    il_smiles_test.values,
        "y_true_log":   y_test,
        "y_pred_log":   y_pred,
        "residual_log": y_pred - y_test,
        "x2_true":      x2_true,
        "x2_pred":      x2_pred,
    })

    perf = {
        "model":         model_name,
        "test_rmse_log": rmse,
        "test_r2":       r2,
        "test_mae_log":  mae,
        "test_rmse_x2":  x2_rmse,
    }
    return perf, predictions_df


def get_feature_importances(model, feature_cols: list, model_name: str,
                            top_n: int = 30) -> pd.DataFrame:
    """
    Extract and rank feature importances from a fitted XGBoost model (gain metric).

    The appearance of T_K / P_kPa near the top is expected and physically meaningful.
    Morgan fingerprint bits that appear here represent structural motifs that
    correlate most strongly with CO2 solubility across the training ILs.
    """
    importance_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
        "model":      model_name,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print(f"\n[feature_importances] Top 10 features for {model_name}:")
    print(importance_df.head(10).to_string(index=False))
    return importance_df.head(top_n)


def main():
    """
    Main pipeline: load -> repeated GroupKFold CV -> fit on full train -> test evaluation -> save.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # -- Step 1: Load data ---------------------------------------------------
    X_train, y_train, il_smiles_train, feature_cols = load_split(TRAIN_CSV, "Train")
    X_test,  y_test,  il_smiles_test,  _            = load_split(TEST_CSV,  "Test")

    n_unique_train_ils = il_smiles_train.nunique()
    print(f"\n[main] {X_train.shape[1]} total features | "
          f"{X_train.shape[0]} train rows ({n_unique_train_ils} unique ILs) | "
          f"{X_test.shape[0]} test rows")

    # -- Step 2: Define tuned XGBoost model ----------------------------------
    # Hyperparameters from Optuna tuning (tune_hyperparameters.py, 50 trials).
    # min_child_weight=10 is the key regularizer: requires at least 10 samples
    # to justify a new split, preventing memorization of individual ILs.
    xgboost_model = XGBRegressor(
        n_estimators     = XGB_N_ESTIMATORS,
        learning_rate    = XGB_LEARNING_RATE,
        max_depth        = XGB_MAX_DEPTH,
        subsample        = XGB_SUBSAMPLE,
        colsample_bytree = XGB_COLSAMPLE,
        reg_alpha        = XGB_REG_ALPHA,
        reg_lambda       = XGB_REG_LAMBDA,
        min_child_weight = XGB_MIN_CHILD_WEIGHT,
        random_state     = RANDOM_SEED,
        n_jobs           = -1,
        verbosity        = 0,
        tree_method      = "hist",  # memory-efficient histogram method
    )

    models = {"XGBoost": xgboost_model}

    # -- Step 3: Repeated GroupKFold CV --------------------------------------
    print(f"\n=== CROSS-VALIDATION ({CV_REPEATS}x{CV_FOLDS} Repeated GroupKFold by IL) ===")
    print("[main] Each repeat uses a different random IL-to-fold assignment.")
    print("[main] This averages out variance from unlucky fold composition.\n")
    cv_results = []
    for name, model in models.items():
        print(f"--- {name} ---")
        cv_result = repeated_group_kfold_cv(model, X_train, y_train, il_smiles_train, name)
        cv_results.append(cv_result)

    pd.DataFrame(cv_results).to_csv(
        os.path.join(RESULTS_DIR, "cv_results.csv"), index=False)
    print(f"\n[main] CV results saved -> results/cv_results.csv")

    # -- Step 4: Fit on full train set, evaluate on test ---------------------
    print("\n=== TEST SET EVALUATION ===")
    test_performances = []
    all_predictions   = []
    all_importances   = []

    for name, model in models.items():
        print(f"\n[main] Fitting {name} on full training set...")
        model.fit(X_train, y_train)

        perf, predictions_df = evaluate_on_test(model, X_test, y_test, il_smiles_test, name)
        test_performances.append(perf)

        predictions_df["model"] = name
        all_predictions.append(predictions_df)

        imp_df = get_feature_importances(model, feature_cols, name, top_n=30)
        all_importances.append(imp_df)

    perf_df = pd.DataFrame(test_performances)
    print("\n=== FINAL PERFORMANCE COMPARISON ===")
    print(perf_df.to_string(index=False))

    # -- Step 5: Save model --------------------------------------------------
    best_name  = "XGBoost"
    best_model = xgboost_model
    best_r2    = perf_df.loc[0, "test_r2"]
    best_rmse  = perf_df.loc[0, "test_rmse_log"]

    print(f"\n[main] Model: {best_name}  (R2 = {best_r2:.4f}, RMSE = {best_rmse:.4f})")

    model_bundle = {
        "model":        best_model,
        "feature_cols": feature_cols,
        "model_name":   best_name,
    }
    joblib.dump(model_bundle, MODEL_PATH)
    print(f"[main] Model bundle saved -> {MODEL_PATH}")

    with open(os.path.join(MODEL_DIR, "best_model_name.txt"), "w") as f:
        f.write(best_name)

    # -- Step 6: Save all result tables --------------------------------------
    perf_df.to_csv(os.path.join(RESULTS_DIR, "model_performance.csv"), index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(
        os.path.join(RESULTS_DIR, "test_predictions.csv"), index=False)
    pd.concat(all_importances, ignore_index=True).to_csv(
        os.path.join(RESULTS_DIR, "feature_importances.csv"), index=False)

    print("\n[main] All results saved:")
    print("  results/cv_results.csv")
    print("  results/model_performance.csv")
    print("  results/test_predictions.csv")
    print("  results/feature_importances.csv")
    print("  models/forward_model.pkl")
    print("  models/best_model_name.txt")

    # -- Final diagnostic ----------------------------------------------------
    cv_xgb = cv_results[0]
    if best_r2 < 0.5:
        print(f"\nWARNING: Best R2 = {best_r2:.4f} -- below 0.5.")
        print("   Consider stacking ensemble or PCA on fingerprints.")
    else:
        print(f"\nModel R2 = {best_r2:.4f}. Phase 3 complete.")
        print(f"  Repeated GroupKFold CV R² = {cv_xgb['cv_r2_mean']:.3f} "
              f"+/- {cv_xgb['cv_r2_std']:.3f}  "
              f"({cv_xgb['n_cv_evaluations']} evaluations)")
        print("  Next: src/predict_with_uncertainty.py, src/applicability_domain.py,")
        print("        src/plot_tp_residuals.py -> Phase 5 DFT inputs")


if __name__ == "__main__":
    main()
