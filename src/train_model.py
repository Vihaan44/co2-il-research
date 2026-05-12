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

WHAT EACH STEP PRODUCES:
  1. Cross-validation results on train set → results/cv_results.csv
  2. Test set evaluation (RMSE, R², MAE)  → results/model_performance.csv
  3. Predicted vs actual values on test   → results/test_predictions.csv
  4. Feature importances (top 30)         → results/feature_importances.csv
  5. Best model + feature_cols saved      → models/forward_model.pkl
     (feature_cols are saved inside the pkl so inverse_design.py can align
      the virtual library to the exact same column order used during training)

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
from sklearn.model_selection import cross_val_score, KFold
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

    Features = Morgan fingerprint bits (cat_* / an_*) + RDKit descriptors + T_K + P_kPa.
    Target   = log10(x2_CO2).

    Returns X (numpy array), y (numpy array), il_smiles (Series), feature_cols (list).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found at {path}. Run build_dataset.py first.")

    df = pd.read_csv(path)
    print(f"[load_split] {label}: {df.shape[0]} rows x {df.shape[1]} cols")

    # Molecular structure features (Morgan FP bits + RDKit descriptors)
    molecular_cols = [c for c in df.columns if c.startswith("cat_") or c.startswith("an_")]

    # Verify condition columns exist before proceeding
    missing_conditions = [c for c in CONDITION_FEATURES if c not in df.columns]
    if missing_conditions:
        raise ValueError(
            f"Missing condition columns {missing_conditions} in {path}.\n"
            f"Available columns include: {list(df.columns[:20])}"
        )

    feature_cols = molecular_cols + CONDITION_FEATURES  # structure + T + P
    print(f"[load_split] {label}: {len(molecular_cols)} molecular features "
          f"+ {len(CONDITION_FEATURES)} condition features = {len(feature_cols)} total")
    print(f"[load_split] {label}: target range "
          f"[{df[TARGET_COL].min():.2f}, {df[TARGET_COL].max():.2f}]")

    # Warn if any condition values are missing -- NaN rows would cause silent issues
    n_missing_t = df["T_K"].isna().sum()
    n_missing_p = df["P_kPa"].isna().sum()
    if n_missing_t > 0 or n_missing_p > 0:
        # DATA QUALITY FLAG: missing T or P means those rows can't be used
        print(f"[load_split] WARNING: {n_missing_t} rows missing T_K, "
              f"{n_missing_p} rows missing P_kPa -- these will cause NaN predictions")

    X         = df[feature_cols].values
    y         = df[TARGET_COL].values
    il_smiles = df["il_smiles"]
    return X, y, il_smiles, feature_cols


def cross_validate_model(model, X_train: np.ndarray, y_train: np.ndarray,
                         model_name: str) -> dict:
    """
    Run 5-fold cross-validation on the training set, reporting mean +/- std RMSE and R².

    Plain KFold (not GroupKFold) is used here: CV is for hyperparameter sanity-checking.
    The authoritative generalization estimate is the held-out test set, which was already
    split by IL identity (no IL overlap) in build_dataset.py.
    """
    kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    rmse_scores = -cross_val_score(model, X_train, y_train,
                                   cv=kfold, scoring="neg_root_mean_squared_error",
                                   n_jobs=-1)
    r2_scores   =  cross_val_score(model, X_train, y_train,
                                   cv=kfold, scoring="r2",
                                   n_jobs=-1)

    print(f"\n[cross_validate] {model_name}:")
    print(f"  CV RMSE  (log10 x2): {rmse_scores.mean():.4f} +/- {rmse_scores.std():.4f}")
    print(f"  CV R2              : {r2_scores.mean():.4f} +/- {r2_scores.std():.4f}")

    return {
        "model":        model_name,
        "cv_rmse_mean": rmse_scores.mean(),
        "cv_rmse_std":  rmse_scores.std(),
        "cv_r2_mean":   r2_scores.mean(),
        "cv_r2_std":    r2_scores.std(),
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
    Main pipeline: load -> cross-validate -> fit -> test evaluation -> save.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # -- Step 1: Load data ---------------------------------------------------
    X_train, y_train, _, feature_cols       = load_split(TRAIN_CSV, "Train")
    X_test,  y_test,  il_smiles_test, _     = load_split(TEST_CSV,  "Test")

    print(f"\n[main] {X_train.shape[1]} total features | "
          f"{X_train.shape[0]} train rows | {X_test.shape[0]} test rows")

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

    # -- Step 3: Cross-validate on training set ------------------------------
    print("\n=== CROSS-VALIDATION (train set, 5-fold) ===")
    cv_results = []
    for name, model in models.items():
        cv_result = cross_validate_model(model, X_train, y_train, name)
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
        print("   Check P_kPa coverage: python -c \"import pandas as pd; "
              "print(pd.read_csv('data/processed/train_set.csv')[['T_K','P_kPa']].describe())\"")
    else:
        print(f"\nModel R2 = {best_r2:.4f}. Phase 3 complete.")
        print("  Ready for src/plot_results.py (figures) or Phase 4 inverse design.")


if __name__ == "__main__":
    main()
