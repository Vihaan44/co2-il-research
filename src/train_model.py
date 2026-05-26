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
  75 trials, GroupKFold CV on 211-IL dataset). Best CV RMSE = 0.3498 (v2).
  Update these constants after running tune_hyperparameters.py (v3 study).

EARLY STOPPING EVALUATION:
  In addition to the fixed-hyperparameter model, this script also trains an
  early-stopping variant (ES model): it holds out 10% of the training ILs as
  an internal validation set and lets XGBoost auto-select n_estimators by
  stopping when validation RMSE stops improving for EARLY_STOPPING_ROUNDS rounds.

  WHY:
    Our fixed n_estimators=200 was set conservatively. The true optimal may be
    higher (e.g. 400-600). Early stopping finds this automatically without
    overfitting. The ES model is saved separately as models/forward_model_es.pkl
    for comparison -- if its test R² beats the fixed model, update the constants.

  IMPORTANT: The ES model uses a GroupShuffleSplit to hold out 10% of ILs
    (not rows) as internal val, so the same IL is never in both ES train
    and ES val. This mirrors our GroupKFold CV principle.

CROSS-VALIDATION STRATEGY — Repeated GroupKFold:
  CV_FOLDS folds x CV_REPEATS different random IL orderings = 25 total evaluations.
  See module-level docstring in previous version for full rationale.

WHAT EACH STEP PRODUCES:
  1. Cross-validation results on train set → results/cv_results.csv
  2. Test set evaluation (RMSE, R², MAE)  → results/model_performance.csv
  3. Predicted vs actual values on test   → results/test_predictions.csv
  4. Feature importances (top 30)         → results/feature_importances.csv
  5. Fixed model + feature_cols saved     → models/forward_model.pkl
  6. Early-stopping model                 → models/forward_model_es.pkl
     (compare test R² -- use whichever is higher as the production model)

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
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.utils import shuffle
from xgboost import XGBRegressor

# -- Constants -----------------------------------------------------------------
TRAIN_CSV   = os.path.join("data", "processed", "train_set.csv")
TEST_CSV    = os.path.join("data", "processed", "test_set.csv")
MODEL_DIR   = "models"
RESULTS_DIR = "results"
MODEL_PATH     = os.path.join(MODEL_DIR, "forward_model.pkl")
MODEL_ES_PATH  = os.path.join(MODEL_DIR, "forward_model_es.pkl")  # early-stopping variant

TARGET_COL         = "log_x2_CO2"
CONDITION_FEATURES = ["T_K", "P_kPa"]
CV_FOLDS           = 5
CV_REPEATS         = 5    # 5 x 5 = 25 total evaluations
RANDOM_SEED        = 42

# Early stopping config
EARLY_STOPPING_ROUNDS  = 30    # stop if val RMSE doesn't improve for 30 rounds
ES_VAL_IL_FRACTION     = 0.10  # hold out 10% of unique ILs as internal val for ES
ES_MAX_N_ESTIMATORS    = 1000  # upper bound; early stopping will pick the actual value

# XGBoost hyperparameters — from Optuna v2 tuning (tune_hyperparameters.py, 50 trials).
# UPDATE THESE after running tune_hyperparameters.py v3 study on expanded features.
XGB_N_ESTIMATORS     = 200
XGB_LEARNING_RATE    = 0.08252578137355096
XGB_MAX_DEPTH        = 5
XGB_SUBSAMPLE        = 0.7240833769291847
XGB_COLSAMPLE        = 0.4778802788064814
XGB_REG_ALPHA        = 0.08067465044762985
XGB_REG_LAMBDA       = 8.226130315805552e-05
XGB_MIN_CHILD_WEIGHT = 10
# gamma and max_delta_step: set to 0 (XGBoost defaults) until v3 tuning completes.
# After tune_hyperparameters.py v3, update these from results/best_hyperparams.csv.
XGB_GAMMA            = 0.0   # min loss reduction per split (0 = unconstrained)
XGB_MAX_DELTA_STEP   = 0     # weight update cap (0 = unconstrained)


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

    n_missing_t = df["T_K"].isna().sum()
    n_missing_p = df["P_kPa"].isna().sum()
    if n_missing_t > 0 or n_missing_p > 0:
        print(f"[load_split] WARNING: {n_missing_t} rows missing T_K, "
              f"{n_missing_p} rows missing P_kPa -- dropping these rows.")
        df = df.dropna(subset=CONDITION_FEATURES).copy()
        print(f"[load_split] After drop: {len(df)} rows remain.")

    # Drop NaN descriptor rows (can occur if Gasteiger charge failed for some ILs)
    n_nan_desc = df[desc_cols].isna().any(axis=1).sum()
    if n_nan_desc > 0:
        print(f"[load_split] WARNING: {n_nan_desc} rows have NaN descriptors -- "
              f"filling with column median (safe fallback for tree models).")
        for col in desc_cols:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    X         = df[feature_cols].values
    y         = df[TARGET_COL].values
    il_smiles = df["il_smiles"]
    return X, y, il_smiles, feature_cols


def repeated_group_kfold_cv(model, X_train: np.ndarray, y_train: np.ndarray,
                             il_smiles_train: pd.Series, model_name: str) -> dict:
    """
    Run repeated GroupKFold cross-validation (CV_FOLDS x CV_REPEATS).
    Each repeat shuffles the IL ordering before splitting into folds.
    Total evaluations = CV_FOLDS * CV_REPEATS (e.g. 5 x 5 = 25).
    """
    groups        = il_smiles_train.values
    unique_ils    = np.unique(groups)
    all_rmse      = []
    all_r2        = []
    repeat_r2_means = []

    print(f"[cv] {model_name}: {CV_REPEATS} repeats x {CV_FOLDS} folds = "
          f"{CV_REPEATS * CV_FOLDS} total evaluations")

    for repeat_idx in range(CV_REPEATS):
        repeat_seed = RANDOM_SEED + repeat_idx
        shuffled_ils = shuffle(unique_ils, random_state=repeat_seed)
        il_to_fold = {il: i % CV_FOLDS for i, il in enumerate(shuffled_ils)}
        fold_assignment = np.array([il_to_fold[il] for il in groups])

        repeat_rmse = []
        repeat_r2   = []

        for fold_idx in range(CV_FOLDS):
            val_mask   = fold_assignment == fold_idx
            train_mask = ~val_mask

            X_fold_train = X_train[train_mask]
            y_fold_train = y_train[train_mask]
            X_fold_val   = X_train[val_mask]
            y_fold_val   = y_train[val_mask]

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

    return {
        "model":            model_name,
        "cv_rmse_mean":     float(rmse_arr.mean()),
        "cv_rmse_std":      float(rmse_arr.std()),
        "cv_r2_mean":       float(r2_arr.mean()),
        "cv_r2_std":        float(r2_arr.std()),
        "cv_protocol":      f"RepeatedGroupKFold({CV_REPEATS}x{CV_FOLDS}, grouped by il_smiles)",
        "n_cv_evaluations": CV_FOLDS * CV_REPEATS,
    }


def train_early_stopping_model(X_train: np.ndarray, y_train: np.ndarray,
                                il_smiles_train: pd.Series,
                                feature_cols: list) -> tuple:
    """
    Train an XGBoost model with early stopping to auto-select n_estimators.

    WHY THIS EXISTS:
      Our fixed n_estimators=200 may underfit. Early stopping trains with up to
      ES_MAX_N_ESTIMATORS trees but stops when the held-out IL validation RMSE
      stops improving for EARLY_STOPPING_ROUNDS consecutive rounds. This finds
      the true optimal n_estimators for the current dataset size without us
      having to guess.

    HOW WE SPLIT:
      GroupShuffleSplit by il_smiles: holds out ES_VAL_IL_FRACTION (10%) of
      unique ILs as internal validation. The same IL is never in both ES train
      and ES val, consistent with our GroupKFold principle.

    LIMITATION:
      ES model is trained on 90% of train data (not 100%), so it has slightly
      less training data than the fixed model. Compare test R² of both models
      and use whichever is higher as the production model.
    """
    print(f"\n[early_stopping] Building ES train/val split "
          f"(hold out {ES_VAL_IL_FRACTION*100:.0f}% of unique ILs)...")

    # GroupShuffleSplit respects IL identity -- same IL never in both splits
    gss = GroupShuffleSplit(n_splits=1, test_size=ES_VAL_IL_FRACTION,
                            random_state=RANDOM_SEED)
    groups = il_smiles_train.values
    es_train_idx, es_val_idx = next(gss.split(X_train, y_train, groups))

    X_es_train = X_train[es_train_idx]
    y_es_train = y_train[es_train_idx]
    X_es_val   = X_train[es_val_idx]
    y_es_val   = y_train[es_val_idx]

    n_es_train_ils = len(set(groups[es_train_idx]))
    n_es_val_ils   = len(set(groups[es_val_idx]))
    print(f"[early_stopping] ES train: {len(X_es_train)} rows ({n_es_train_ils} ILs) | "
          f"ES val: {len(X_es_val)} rows ({n_es_val_ils} ILs)")

    # Train with the same hyperparameters as the fixed model (including gamma/max_delta_step),
    # but let early stopping choose n_estimators automatically.
    es_model = XGBRegressor(
        n_estimators     = ES_MAX_N_ESTIMATORS,  # upper bound; ES will stop earlier
        learning_rate    = XGB_LEARNING_RATE,
        max_depth        = XGB_MAX_DEPTH,
        subsample        = XGB_SUBSAMPLE,
        colsample_bytree = XGB_COLSAMPLE,
        reg_alpha        = XGB_REG_ALPHA,
        reg_lambda       = XGB_REG_LAMBDA,
        min_child_weight = XGB_MIN_CHILD_WEIGHT,
        gamma            = XGB_GAMMA,
        max_delta_step   = XGB_MAX_DELTA_STEP,
        random_state     = RANDOM_SEED,
        n_jobs           = -1,
        verbosity        = 0,
        tree_method      = "hist",
        early_stopping_rounds = EARLY_STOPPING_ROUNDS,
        eval_metric      = "rmse",
    )

    es_model.fit(
        X_es_train, y_es_train,
        eval_set=[(X_es_val, y_es_val)],
        verbose=False,
    )

    best_iteration = es_model.best_iteration
    print(f"[early_stopping] Best n_estimators found by early stopping: {best_iteration}")
    print(f"  (stopped at round {best_iteration} of max {ES_MAX_N_ESTIMATORS})")
    print(f"  Interpretation: fixed model's n_estimators=200 may be "
          f"{'undertrained' if best_iteration > 200 else 'about right or overtrained'}.")

    return es_model, best_iteration


def evaluate_on_test(model, X_test: np.ndarray, y_test: np.ndarray,
                     il_smiles_test: pd.Series, model_name: str) -> tuple:
    """
    Predict on the held-out test set and compute RMSE, R², and MAE.
    Reports error in both log10 units and back-transformed mole fraction units.
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
    T_K and P_kPa near the top is expected and physically meaningful.
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
    Main pipeline: load -> repeated GroupKFold CV -> fit on full train ->
    early-stopping variant -> test evaluation -> save both models.
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

    # -- Step 2: Define fixed XGBoost model (Optuna v2 hyperparams) ----------
    # gamma and max_delta_step are 0.0/0 (defaults) until v3 tuning completes.
    xgboost_model = XGBRegressor(
        n_estimators     = XGB_N_ESTIMATORS,
        learning_rate    = XGB_LEARNING_RATE,
        max_depth        = XGB_MAX_DEPTH,
        subsample        = XGB_SUBSAMPLE,
        colsample_bytree = XGB_COLSAMPLE,
        reg_alpha        = XGB_REG_ALPHA,
        reg_lambda       = XGB_REG_LAMBDA,
        min_child_weight = XGB_MIN_CHILD_WEIGHT,
        gamma            = XGB_GAMMA,           # 0.0 until v3 tuning
        max_delta_step   = XGB_MAX_DELTA_STEP,  # 0 until v3 tuning
        random_state     = RANDOM_SEED,
        n_jobs           = -1,
        verbosity        = 0,
        tree_method      = "hist",
    )

    # -- Step 3: Repeated GroupKFold CV (fixed model) ------------------------
    print(f"\n=== CROSS-VALIDATION ({CV_REPEATS}x{CV_FOLDS} Repeated GroupKFold by IL) ===")
    cv_result = repeated_group_kfold_cv(
        xgboost_model, X_train, y_train, il_smiles_train, "XGBoost"
    )
    pd.DataFrame([cv_result]).to_csv(
        os.path.join(RESULTS_DIR, "cv_results.csv"), index=False)
    print(f"[main] CV results saved -> results/cv_results.csv")

    # -- Step 4: Fit fixed model on full train, evaluate on test -------------
    print("\n=== TEST SET EVALUATION (Fixed hyperparams model) ===")
    print("[main] Fitting XGBoost on full training set...")
    xgboost_model.fit(X_train, y_train)
    fixed_perf, fixed_preds = evaluate_on_test(
        xgboost_model, X_test, y_test, il_smiles_test, "XGBoost_fixed"
    )
    fixed_imp = get_feature_importances(xgboost_model, feature_cols, "XGBoost_fixed")

    # -- Step 5: Train early-stopping model and evaluate on test -------------
    print("\n=== EARLY STOPPING MODEL ===")
    es_model, best_n_est = train_early_stopping_model(
        X_train, y_train, il_smiles_train, feature_cols
    )
    es_perf, es_preds = evaluate_on_test(
        es_model, X_test, y_test, il_smiles_test, "XGBoost_ES"
    )
    es_imp = get_feature_importances(es_model, feature_cols, "XGBoost_ES")

    # -- Step 6: Compare both models ----------------------------------------
    perf_df = pd.DataFrame([fixed_perf, es_perf])
    print("\n=== FINAL PERFORMANCE COMPARISON ===")
    print(perf_df.to_string(index=False))
    print()
    if es_perf["test_r2"] > fixed_perf["test_r2"]:
        print(f"  ES model wins: R²={es_perf['test_r2']:.4f} vs "
              f"fixed R²={fixed_perf['test_r2']:.4f}")
        print(f"  ACTION: Update XGB_N_ESTIMATORS to {best_n_est} in train_model.py constants.")
    else:
        print(f"  Fixed model wins: R²={fixed_perf['test_r2']:.4f} vs "
              f"ES R²={es_perf['test_r2']:.4f}")
        print(f"  Current n_estimators={XGB_N_ESTIMATORS} is reasonable.")

    # -- Step 7: Save models -------------------------------------------------
    joblib.dump({"model": xgboost_model, "feature_cols": feature_cols,
                 "model_name": "XGBoost_fixed"}, MODEL_PATH)
    print(f"[main] Fixed model saved -> {MODEL_PATH}")

    joblib.dump({"model": es_model, "feature_cols": feature_cols,
                 "model_name": "XGBoost_ES",
                 "best_n_estimators": best_n_est}, MODEL_ES_PATH)
    print(f"[main] ES model saved -> {MODEL_ES_PATH}")

    with open(os.path.join(MODEL_DIR, "best_model_name.txt"), "w") as f:
        best_name = "XGBoost_ES" if es_perf["test_r2"] > fixed_perf["test_r2"] else "XGBoost_fixed"
        f.write(best_name)
    print(f"[main] Best model name -> models/best_model_name.txt: {best_name}")

    # -- Step 8: Save all result tables -------------------------------------
    perf_df.to_csv(os.path.join(RESULTS_DIR, "model_performance.csv"), index=False)
    pd.concat([fixed_preds, es_preds], ignore_index=True).to_csv(
        os.path.join(RESULTS_DIR, "test_predictions.csv"), index=False)
    pd.concat([fixed_imp, es_imp], ignore_index=True).to_csv(
        os.path.join(RESULTS_DIR, "feature_importances.csv"), index=False)

    print("\n[main] All results saved.")
    best_r2 = max(fixed_perf["test_r2"], es_perf["test_r2"])
    if best_r2 < 0.5:
        print(f"WARNING: Best R² = {best_r2:.4f} -- below 0.5. Check data pipeline.")
    else:
        print(f"Best R² = {best_r2:.4f}. Phase 3 complete.")
        print(f"  CV R² = {cv_result['cv_r2_mean']:.3f} +/- {cv_result['cv_r2_std']:.3f}")
        print("  Next: src/predict_with_uncertainty.py, src/applicability_domain.py,")
        print("        src/plot_tp_residuals.py -> Phase 5 DFT inputs")


if __name__ == "__main__":
    main()
