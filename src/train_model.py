"""
train_model.py
--------------
PURPOSE: Train a Random Forest and XGBoost regressor to predict log10(CO2 mole
         fraction solubility) from IL molecular features + temperature + pressure.

WHY T_K AND P_kPa ARE INCLUDED AS FEATURES:
  The first run (structure-only) produced R² ≈ -0.10 (worse than predicting the mean).
  This happened because the same IL measured at 298K vs 350K has very different x2 values --
  molecular fingerprints alone cannot explain that variance.
  T_K and P_kPa are direct physical predictors of solubility (Henry's law: x2 ∝ P/H(T)).
  Adding them is scientifically correct -- real process models always condition on T and P.

  For competition framing: "Our model predicts CO2 solubility given the IL structure,
  temperature, and pressure -- mirroring real industrial process conditions."

CROSS-VALIDATION STRATEGY — GroupKFold:
  We use GroupKFold (grouped by il_smiles) instead of plain KFold.
  WHY THIS MATTERS: Plain KFold can put measurements of the SAME IL in both
  training and validation folds. Since one IL can appear at many (T, P) conditions,
  this leaks structural information from train into val -- the CV score looks
  artificially better than generalization to new ILs actually is.
  GroupKFold ensures every fold's validation set contains ONLY ILs not seen in
  that fold's training set, giving a fair estimate of performance on novel ILs.
  This is the correct protocol for IL property prediction (Sistla et al. 2023).

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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, GroupKFold
from xgboost import XGBRegressor

# -- Constants -----------------------------------------------------------------
TRAIN_CSV   = os.path.join("data", "processed", "train_set.csv")
TEST_CSV    = os.path.join("data", "processed", "test_set.csv")
MODEL_DIR   = "models"
RESULTS_DIR = "results"
MODEL_PATH  = os.path.join(MODEL_DIR, "forward_model.pkl")

TARGET_COL         = "log_x2_CO2"       # log10-transformed mole fraction solubility
CONDITION_FEATURES = ["T_K", "P_kPa"]   # temperature (K) and pressure (kPa) from ILThermo
CV_FOLDS           = 5
RANDOM_SEED        = 42

# Random Forest hyperparameters
RF_N_ESTIMATORS     = 300
RF_MAX_FEATURES     = "sqrt"   # sqrt(n_features) per split -- standard for RF
RF_MIN_SAMPLES_LEAF = 3        # prevents overfitting on sparse IL clusters

# XGBoost hyperparameters
XGB_N_ESTIMATORS    = 500
XGB_LEARNING_RATE   = 0.05
XGB_MAX_DEPTH       = 6
XGB_SUBSAMPLE       = 0.8
XGB_COLSAMPLE       = 0.8


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

    # Molecular structure features: Morgan FP bits (cat_fp_*, an_fp_*) AND
    # RDKit physicochemical descriptors (cat_mol_weight, an_num_hbd, etc.)
    # Both are prefixed cat_ or an_ in featurize.py.
    molecular_cols = [c for c in df.columns if c.startswith("cat_") or c.startswith("an_")]

    # Count FP bits vs descriptors for transparency
    fp_cols   = [c for c in molecular_cols if "_fp_" in c]
    desc_cols = [c for c in molecular_cols if "_fp_" not in c]
    print(f"[load_split] {label}: {len(fp_cols)} Morgan FP bits + "
          f"{len(desc_cols)} RDKit descriptors + {len(CONDITION_FEATURES)} T/P = "
          f"{len(molecular_cols) + len(CONDITION_FEATURES)} total features")

    # Verify condition columns exist before proceeding
    missing_conditions = [c for c in CONDITION_FEATURES if c not in df.columns]
    if missing_conditions:
        raise ValueError(
            f"Missing condition columns {missing_conditions} in {path}.\n"
            f"Available columns include: {list(df.columns[:20])}"
        )

    feature_cols = molecular_cols + CONDITION_FEATURES  # structure + T + P
    print(f"[load_split] {label}: target range "
          f"[{df[TARGET_COL].min():.2f}, {df[TARGET_COL].max():.2f}]")

    # Warn if any condition values are missing -- NaN rows cause silent errors
    n_missing_t = df["T_K"].isna().sum()
    n_missing_p = df["P_kPa"].isna().sum()
    if n_missing_t > 0 or n_missing_p > 0:
        # DATA QUALITY FLAG: missing T or P means those rows can't be used
        print(f"[load_split] WARNING: {n_missing_t} rows missing T_K, "
              f"{n_missing_p} rows missing P_kPa -- these will cause NaN predictions")

    X         = df[feature_cols].values
    y         = df[TARGET_COL].values
    il_smiles = df["il_smiles"]  # used as group label in GroupKFold CV
    return X, y, il_smiles, feature_cols


def cross_validate_model(model, X_train: np.ndarray, y_train: np.ndarray,
                         il_smiles_train: pd.Series, model_name: str) -> dict:
    """
    Run 5-fold cross-validation on the training set using GroupKFold.

    GroupKFold groups rows by il_smiles so that all measurements from
    the same IL go into the same fold. This prevents data leakage: without
    grouping, the same IL measured at 10 temperatures could appear in both
    train and val, making CV scores optimistically biased.

    The CV RMSE here reflects true generalization to new ILs -- a harder and
    more honest metric than plain KFold would give.
    """
    # GroupKFold requires groups: same IL always in same fold
    group_kfold = GroupKFold(n_splits=CV_FOLDS)
    groups = il_smiles_train.values  # one group label per row

    rmse_scores = []
    r2_scores   = []

    # Manual loop because GroupKFold requires explicit groups -- can't use cross_val_score
    for fold_idx, (train_idx, val_idx) in enumerate(group_kfold.split(X_train, y_train, groups)):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

        model.fit(X_fold_train, y_fold_train)
        y_fold_pred = model.predict(X_fold_val)

        fold_rmse = np.sqrt(mean_squared_error(y_fold_val, y_fold_pred))
        fold_r2   = r2_score(y_fold_val, y_fold_pred)
        rmse_scores.append(fold_rmse)
        r2_scores.append(fold_r2)

        # Count unique ILs in val fold to show the grouping is working
        n_val_ils = len(np.unique(groups[val_idx]))
        print(f"    Fold {fold_idx+1}: {n_val_ils} ILs in val | RMSE={fold_rmse:.4f} | R²={fold_r2:.4f}")

    rmse_arr = np.array(rmse_scores)
    r2_arr   = np.array(r2_scores)

    print(f"\n[cross_validate] {model_name} (GroupKFold, grouped by IL):")
    print(f"  CV RMSE  (log10 x2): {rmse_arr.mean():.4f} +/- {rmse_arr.std():.4f}")
    print(f"  CV R2              : {r2_arr.mean():.4f} +/- {r2_arr.std():.4f}")
    print(f"  NOTE: GroupKFold CV is stricter than plain KFold -- "
          f"each fold's val set contains ILs never seen during that fold's training.")

    return {
        "model":        model_name,
        "cv_rmse_mean": rmse_arr.mean(),
        "cv_rmse_std":  rmse_arr.std(),
        "cv_r2_mean":   r2_arr.mean(),
        "cv_r2_std":    r2_arr.std(),
        "cv_protocol":  "GroupKFold(il_smiles)",  # document the protocol used
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

    x2_true = 10 ** y_test    # back-transform from log10 to mole fraction
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
        "residual_log": y_pred - y_test,   # positive = model overpredicted
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
    Extract and rank feature importances from a fitted tree-based model.

    RF uses mean decrease in impurity; XGBoost uses gain.
    Both reflect how much each feature reduces prediction error across all trees.
    The appearance of T_K / P_kPa near the top is expected and physically meaningful.
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
    Main pipeline: load -> cross-validate (GroupKFold) -> fit -> test evaluation -> save.
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

    # Warn if too few ILs for GroupKFold -- each fold needs at least 1 IL
    if n_unique_train_ils < CV_FOLDS * 2:
        print(f"[main] WARNING: only {n_unique_train_ils} unique ILs in train set -- "
              f"GroupKFold with {CV_FOLDS} folds may have very uneven folds.")

    # -- Step 2: Define models -----------------------------------------------
    random_forest = RandomForestRegressor(
        n_estimators     = RF_N_ESTIMATORS,
        max_features     = RF_MAX_FEATURES,
        min_samples_leaf = RF_MIN_SAMPLES_LEAF,
        random_state     = RANDOM_SEED,
        n_jobs           = -1,
    )
    xgboost_model = XGBRegressor(
        n_estimators     = XGB_N_ESTIMATORS,
        learning_rate    = XGB_LEARNING_RATE,
        max_depth        = XGB_MAX_DEPTH,
        subsample        = XGB_SUBSAMPLE,
        colsample_bytree = XGB_COLSAMPLE,
        random_state     = RANDOM_SEED,
        n_jobs           = -1,
        verbosity        = 0,
    )

    models = {
        "RandomForest": random_forest,
        "XGBoost":      xgboost_model,
    }

    # -- Step 3: Cross-validate on training set (GroupKFold) ----------------
    print("\n=== CROSS-VALIDATION (train set, 5-fold GroupKFold by IL) ===")
    print("[main] Using GroupKFold -- each val fold contains ILs absent from train.")
    print("[main] This gives a realistic estimate of generalization to novel ILs.\n")
    cv_results = []
    for name, model in models.items():
        print(f"--- {name} ---")
        cv_result = cross_validate_model(model, X_train, y_train, il_smiles_train, name)
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

    # -- Step 5: Select and save best model ----------------------------------
    best_idx   = perf_df["test_r2"].idxmax()
    best_name  = perf_df.loc[best_idx, "model"]
    best_model = models[best_name]
    best_r2    = perf_df.loc[best_idx, "test_r2"]
    best_rmse  = perf_df.loc[best_idx, "test_rmse_log"]

    print(f"\n[main] Best model: {best_name}  (R2 = {best_r2:.4f}, RMSE = {best_rmse:.4f})")

    # Save model + feature_cols together so inverse_design.py can align columns.
    # WHY: XGBoost trained on numpy arrays does not store feature names internally.
    # Without feature_cols, inverse_design.py cannot detect column order mismatches.
    model_bundle = {
        "model":        best_model,
        "feature_cols": feature_cols,   # exact list of column names in training order
        "model_name":   best_name,
    }
    joblib.dump(model_bundle, MODEL_PATH)
    print(f"[main] Model bundle (model + feature_cols) saved -> {MODEL_PATH}")

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
    if best_r2 < 0.5:
        print(f"\nWARNING: Best R2 = {best_r2:.4f} -- still below 0.5.")
        print("   Next steps: run src/tune_hyperparameters.py or check T/P coverage.")
        print("   Check P_kPa coverage: python -c \"import pandas as pd; "
              "print(pd.read_csv('data/processed/train_set.csv')[['T_K','P_kPa']].describe())\"")
    else:
        print(f"\nModel R2 = {best_r2:.4f}. Phase 3 complete.")
        print("  Next: run src/tune_hyperparameters.py, src/predict_with_uncertainty.py,")
        print("        or src/applicability_domain.py for robustness improvements.")


if __name__ == "__main__":
    main()
