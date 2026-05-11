"""
train_model.py
--------------
PURPOSE: Train a Random Forest and XGBoost regressor on the ML-ready dataset
         to predict log10(CO2 mole fraction solubility) from IL Morgan fingerprints
         and RDKit descriptors. Evaluates with 5-fold cross-validation on train set
         and final RMSE/R² on the held-out test set.

WHAT EACH STEP PRODUCES:
  1. Cross-validation results on train set → results/cv_results.csv
  2. Test set evaluation (RMSE, R², MAE) → results/model_performance.csv
  3. Predicted vs actual values on test → results/test_predictions.csv
  4. Feature importances (top 30) → results/feature_importances.csv
  5. Best model saved → models/forward_model.pkl

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

# ── Constants ──────────────────────────────────────────────────────────────────
TRAIN_CSV     = os.path.join("data", "processed", "train_set.csv")
TEST_CSV      = os.path.join("data", "processed", "test_set.csv")
MODEL_DIR     = "models"
RESULTS_DIR   = "results"
MODEL_PATH    = os.path.join(MODEL_DIR, "forward_model.pkl")

TARGET_COL    = "log_x2_CO2"    # what we're predicting
CV_FOLDS      = 5               # number of cross-validation folds
RANDOM_SEED   = 42              # for reproducibility

# Random Forest hyperparameters — reasonable defaults for ~7,000 training rows
RF_N_ESTIMATORS  = 300         # more trees = more stable, diminishing returns after ~300
RF_MAX_FEATURES  = "sqrt"      # sqrt(4112) ≈ 64 features per split (standard for RF)
RF_MIN_SAMPLES_LEAF = 3        # avoids overfitting on sparse IL clusters

# XGBoost hyperparameters — conservative settings to avoid overfitting
XGB_N_ESTIMATORS   = 500
XGB_LEARNING_RATE  = 0.05      # small LR with many trees generalizes better
XGB_MAX_DEPTH      = 6         # standard depth; deeper risks overfitting
XGB_SUBSAMPLE      = 0.8       # row subsampling adds regularization
XGB_COLSAMPLE      = 0.8       # feature subsampling per tree


def load_split(path: str, label: str) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """
    Load a train or test CSV and separate features (cat_* / an_* columns)
    from the target column.

    Returns:
        X       — feature matrix as numpy array
        y       — target vector (log10 x2_CO2) as numpy array
        il_smiles — IL identity column (kept for error analysis)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found at {path}. Run build_dataset.py first.")

    df = pd.read_csv(path)
    print(f"[load_split] {label}: {df.shape[0]} rows × {df.shape[1]} cols")

    # Feature columns: all Morgan fingerprint bits and RDKit descriptors
    # They are prefixed cat_ (cation features) or an_ (anion features)
    feature_cols = [c for c in df.columns if c.startswith("cat_") or c.startswith("an_")]
    print(f"[load_split] {label}: {len(feature_cols)} features, "
          f"target range [{df[TARGET_COL].min():.2f}, {df[TARGET_COL].max():.2f}]")

    X         = df[feature_cols].values
    y         = df[TARGET_COL].values
    il_smiles = df["il_smiles"]
    return X, y, il_smiles, feature_cols


def cross_validate_model(model, X_train: np.ndarray, y_train: np.ndarray,
                         model_name: str) -> dict:
    """
    Run 5-fold cross-validation on the training set and report mean ± std RMSE and R².

    NOTE: We use plain KFold here (not GroupKFold) because the GroupShuffleSplit
    in build_dataset.py already guarantees the test set has no IL overlap.
    Cross-val is for hyperparameter sanity-checking, not final evaluation.
    Final unbiased performance comes from the held-out test set.
    """
    kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # neg_root_mean_squared_error returns negative RMSE — sklearn convention
    rmse_scores = -cross_val_score(model, X_train, y_train,
                                   cv=kfold, scoring="neg_root_mean_squared_error",
                                   n_jobs=-1)  # use all CPU cores
    r2_scores   =  cross_val_score(model, X_train, y_train,
                                   cv=kfold, scoring="r2",
                                   n_jobs=-1)

    print(f"\n[cross_validate] {model_name}:")
    print(f"  CV RMSE  (log10 x2): {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")
    print(f"  CV R²              : {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")

    return {
        "model":       model_name,
        "cv_rmse_mean": rmse_scores.mean(),
        "cv_rmse_std":  rmse_scores.std(),
        "cv_r2_mean":   r2_scores.mean(),
        "cv_r2_std":    r2_scores.std(),
    }


def evaluate_on_test(model, X_test: np.ndarray, y_test: np.ndarray,
                     il_smiles_test: pd.Series, model_name: str) -> tuple[dict, pd.DataFrame]:
    """
    Fit the model on the full training set (already done upstream), then
    evaluate RMSE, R², and MAE on the completely held-out test set.

    Also computes back-transformed error: we predicted log10(x2), so
    actual prediction error in mole fraction units = 10^y_pred vs 10^y_true.

    Returns the performance dict and a DataFrame of per-row predictions.
    """
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)

    # Convert log10 predictions back to mole fraction for interpretability
    x2_true = 10 ** y_test
    x2_pred = 10 ** y_pred
    x2_rmse = np.sqrt(mean_squared_error(x2_true, x2_pred))

    print(f"\n[evaluate_on_test] {model_name} — HELD-OUT TEST SET:")
    print(f"  RMSE  (log10 x2) : {rmse:.4f}")
    print(f"  R²               : {r2:.4f}")
    print(f"  MAE   (log10 x2) : {mae:.4f}")
    print(f"  RMSE  (x2 units) : {x2_rmse:.6f}   ← mole fraction error")

    # Per-row predictions table for plotting and error analysis
    predictions_df = pd.DataFrame({
        "il_smiles":    il_smiles_test.values,
        "y_true_log":   y_test,
        "y_pred_log":   y_pred,
        "residual_log": y_pred - y_test,    # positive = overpredicted
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
    Extract feature importances from a fitted tree-based model and return
    the top N most important features sorted descending.

    For Random Forest: mean decrease in impurity.
    For XGBoost: gain-based importance.
    Both measure how much each feature reduces prediction error across all trees.
    """
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": importances,
        "model":      model_name,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print(f"\n[feature_importances] Top 10 features for {model_name}:")
    print(importance_df.head(10).to_string(index=False))
    return importance_df.head(top_n)


def main():
    """
    Main pipeline:
      1. Load train/test splits
      2. Cross-validate RF and XGBoost on training set
      3. Fit best model on full training set
      4. Evaluate on held-out test set
      5. Extract feature importances
      6. Save model + all result CSVs
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Step 1: Load data ───────────────────────────────────────────────────
    X_train, y_train, _, feature_cols = load_split(TRAIN_CSV, "Train")
    X_test,  y_test,  il_smiles_test, _ = load_split(TEST_CSV,  "Test")

    print(f"\n[main] Feature matrix: {X_train.shape[1]} features, "
          f"{X_train.shape[0]} train rows, {X_test.shape[0]} test rows")

    # ── Step 2: Define models ───────────────────────────────────────────────
    random_forest = RandomForestRegressor(
        n_estimators     = RF_N_ESTIMATORS,
        max_features     = RF_MAX_FEATURES,
        min_samples_leaf = RF_MIN_SAMPLES_LEAF,
        random_state     = RANDOM_SEED,
        n_jobs           = -1,    # use all CPU cores
    )

    xgboost_model = XGBRegressor(
        n_estimators    = XGB_N_ESTIMATORS,
        learning_rate   = XGB_LEARNING_RATE,
        max_depth       = XGB_MAX_DEPTH,
        subsample       = XGB_SUBSAMPLE,
        colsample_bytree= XGB_COLSAMPLE,
        random_state    = RANDOM_SEED,
        n_jobs          = -1,
        verbosity       = 0,   # suppress XGBoost internal logging
    )

    models = [
        (random_forest,  "RandomForest"),
        (xgboost_model,  "XGBoost"),
    ]

    # ── Step 3: Cross-validate on training set ──────────────────────────────
    print("\n=== CROSS-VALIDATION (train set, 5-fold) ===")
    cv_results = []
    for model, name in models:
        cv_result = cross_validate_model(model, X_train, y_train, name)
        cv_results.append(cv_result)

    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(os.path.join(RESULTS_DIR, "cv_results.csv"), index=False)
    print(f"\n[main] CV results saved → results/cv_results.csv")

    # ── Step 4: Fit both models on full train set, evaluate on test ─────────
    print("\n=== TEST SET EVALUATION ===")
    test_performances = []
    all_predictions   = []
    all_importances   = []

    for model, name in models:
        print(f"\n[main] Fitting {name} on full training set…")
        model.fit(X_train, y_train)   # fit on ALL training rows (not CV fold)

        perf, predictions_df = evaluate_on_test(
            model, X_test, y_test, il_smiles_test, name)
        test_performances.append(perf)

        predictions_df["model"] = name  # tag rows with model name
        all_predictions.append(predictions_df)

        imp_df = get_feature_importances(model, feature_cols, name, top_n=30)
        all_importances.append(imp_df)

    perf_df = pd.DataFrame(test_performances)
    print("\n=== FINAL PERFORMANCE COMPARISON ===")
    print(perf_df.to_string(index=False))

    # ── Step 5: Select best model by test R² ────────────────────────────────
    best_row   = perf_df.loc[perf_df["test_r2"].idxmax()]
    best_name  = best_row["model"]
    best_model = dict(models)[best_name]

    print(f"\n[main] Best model: {best_name}  "
          f"(R² = {best_row['test_r2']:.4f}, RMSE = {best_row['test_rmse_log']:.4f})")

    # ── Step 6: Save everything ─────────────────────────────────────────────
    joblib.dump(best_model, MODEL_PATH)
    print(f"[main] Best model saved → {MODEL_PATH}")

    # Save a label so downstream scripts know which model was selected
    with open(os.path.join(MODEL_DIR, "best_model_name.txt"), "w") as f:
        f.write(best_name)

    perf_df.to_csv(os.path.join(RESULTS_DIR, "model_performance.csv"), index=False)

    all_preds_df = pd.concat(all_predictions, ignore_index=True)
    all_preds_df.to_csv(os.path.join(RESULTS_DIR, "test_predictions.csv"), index=False)

    all_imps_df = pd.concat(all_importances, ignore_index=True)
    all_imps_df.to_csv(os.path.join(RESULTS_DIR, "feature_importances.csv"), index=False)

    print("\n[main] All results saved:")
    print("  results/cv_results.csv")
    print("  results/model_performance.csv")
    print("  results/test_predictions.csv")
    print("  results/feature_importances.csv")
    print("  models/forward_model.pkl")
    print("  models/best_model_name.txt")

    # ── Limitation note for paper ────────────────────────────────────────────
    # DATA QUALITY NOTE: If R² < 0.6 on the test set, the most likely causes are:
    #   1. Too few unique ILs (211 is borderline for generalization)
    #   2. Morgan fingerprints don't capture conformational or ionic effects
    #   3. T/P variation in the dataset adds noise not captured by structure alone
    # In that case: (a) add T_K and P_MPa as explicit features, (b) expand dataset.

    if min(r["test_r2"] for r in test_performances) < 0.6:
        print("\n⚠️  WARNING: Best R² < 0.6 — consider adding T_K/P_MPa as features "
              "or expanding dataset. See limitation note in code.")

    print("\n[main] Phase 3 complete. Ready for src/plot_results.py (figures) "
          "or Phase 4 inverse design.")


if __name__ == "__main__":
    main()
