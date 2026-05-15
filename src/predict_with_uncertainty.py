"""
predict_with_uncertainty.py
----------------------------
PURPOSE: Generate predictions with uncertainty estimates (confidence intervals)
         for virtual IL candidates using a trained Random Forest model.

WHY UNCERTAINTY QUANTIFICATION MATTERS:
  The forward model outputs a single number for each IL. But how confident
  is that prediction? A candidate predicted at x2=0.072 with uncertainty
  ±0.002 is very different from one predicted at 0.072 ±0.020 (which could
  plausibly be anywhere from 0.032 to 0.11 in real units).

  For competition: uncertainty tells you which predictions MOST NEED DFT
  validation -- the high-uncertainty ones are where the model is guessing.
  This is how professional cheminformatics models are deployed.

HOW IT WORKS (Random Forest prediction variance):
  A Random Forest is an ensemble of many decision trees. Each tree produces
  its own prediction. The MEAN of tree predictions = final prediction.
  The STD of tree predictions = uncertainty estimate.
  High std = trees disagree = model is uncertain = structure is unusual.

  This only works for Random Forest (not XGBoost, which uses boosting not
  bagging). If the best model is XGBoost, we fall back to an RF trained
  on the same data just for uncertainty estimation.

OUTPUTS:
  results/predictions_with_uncertainty.csv  -- predictions + uncertainty for each candidate

INPUTS:
  models/forward_model.pkl                 -- trained model bundle (from train_model.py)
  results/virtual_library_predictions.csv  -- or any predictions CSV with il_smiles column
  data/processed/train_set.csv             -- needed to check if RF is the best model

Run from project root:
    python src/predict_with_uncertainty.py
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

# -- Constants -----------------------------------------------------------------
MODEL_PATH       = os.path.join("models", "forward_model.pkl")
TRAIN_CSV         = os.path.join("data", "processed", "train_set.csv")
PREDICTIONS_CSV   = os.path.join("results", "virtual_library_predictions.csv")
OUTPUT_CSV        = os.path.join("results", "predictions_with_uncertainty.csv")

TARGET_COL         = "log_x2_CO2"
CONDITION_FEATURES = ["T_K", "P_kPa"]

# Standard screening conditions (same as used in inverse_design.py)
SCREENING_T_K    = 298.15    # 25°C -- near-ambient CO2 capture
SCREENING_P_kPa  = 101.325   # 1 atm

# RF hyperparameters for fallback uncertainty model (if best model is XGBoost)
RF_N_ESTIMATORS     = 300
RF_MAX_FEATURES     = "sqrt"
RF_MIN_SAMPLES_LEAF = 3
RANDOM_SEED         = 42

# Threshold for flagging high uncertainty (in log10 x2 units)
# A std > 0.15 in log10 space means predictions span roughly a factor of 2 in x2
HIGH_UNCERTAINTY_THRESHOLD = 0.15


def load_model_bundle(model_path: str) -> tuple:
    """
    Load the trained model bundle from disk.
    Returns (model, feature_cols, model_name).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train_model.py first."
        )
    bundle       = joblib.load(model_path)
    model        = bundle["model"]
    feature_cols = bundle["feature_cols"]
    model_name   = bundle["model_name"]
    print(f"[load_model_bundle] Loaded {model_name} with {len(feature_cols)} features")
    return model, feature_cols, model_name


def get_rf_for_uncertainty(model, model_name: str,
                           X_train: np.ndarray, y_train: np.ndarray,
                           il_smiles_train: pd.Series,
                           feature_cols: list) -> RandomForestRegressor:
    """
    Return an RF model suitable for uncertainty estimation.

    If the best model is already a Random Forest, use it directly.
    If it's XGBoost, train a separate RF on the same data just for
    uncertainty -- its predictions won't be used, only its tree variance.

    WHY A SEPARATE RF FOR XGBOOST:
      XGBoost uses boosting: trees are sequential, not independent.
      Tree-to-tree variance in XGBoost doesn't reflect prediction uncertainty
      in the same way RF bagging variance does. RF's independent trees
      are the correct tool for this type of ensemble variance.
    """
    if "RandomForest" in model_name or isinstance(model, RandomForestRegressor):
        print("[get_rf] Best model is RF -- using it directly for uncertainty.")
        return model
    else:
        print("[get_rf] Best model is not RF -- training auxiliary RF for uncertainty.")
        print("[get_rf] (Auxiliary RF used ONLY for uncertainty, not for final predictions)")
        aux_rf = RandomForestRegressor(
            n_estimators     = RF_N_ESTIMATORS,
            max_features     = RF_MAX_FEATURES,
            min_samples_leaf = RF_MIN_SAMPLES_LEAF,
            random_state     = RANDOM_SEED,
            n_jobs           = -1,
        )
        aux_rf.fit(X_train, y_train)
        # Quick sanity check: print CV RMSE of aux RF
        groups      = il_smiles_train.values
        group_kfold = GroupKFold(n_splits=5)
        rmse_folds  = []
        for train_idx, val_idx in group_kfold.split(X_train, y_train, groups):
            aux_rf.fit(X_train[train_idx], y_train[train_idx])
            y_pred = aux_rf.predict(X_train[val_idx])
            rmse_folds.append(np.sqrt(mean_squared_error(y_train[val_idx], y_pred)))
        # Refit on full train set after CV
        aux_rf.fit(X_train, y_train)
        print(f"[get_rf] Auxiliary RF GroupKFold CV RMSE: {np.mean(rmse_folds):.4f}")
        return aux_rf


def predict_with_tree_variance(rf_model: RandomForestRegressor,
                               X: np.ndarray) -> tuple:
    """
    Use individual tree predictions to compute mean and std per sample.

    rf_model.estimators_ is the list of trained decision trees.
    We collect each tree's prediction for each sample, then compute:
      - mean: the ensemble prediction (same as rf_model.predict(X))
      - std:  across-tree standard deviation = uncertainty estimate

    A high std means the trees disagree => the model is unsure.
    In log10 x2 space, std ≈ 0.15 means the 95% interval spans roughly
    ±0.3 log10 units, i.e. about a factor of 2 in mole fraction.
    """
    # Stack all trees' predictions: shape (n_trees, n_samples)
    tree_predictions = np.array([
        tree.predict(X) for tree in rf_model.estimators_
    ])

    pred_mean = tree_predictions.mean(axis=0)   # average across trees
    pred_std  = tree_predictions.std(axis=0)    # std across trees = uncertainty
    return pred_mean, pred_std


def compute_confidence_intervals(pred_mean: np.ndarray,
                                 pred_std: np.ndarray) -> tuple:
    """
    Compute 95% prediction intervals from mean and std.

    We use ±1.96 * std as an approximate 95% interval, assuming tree
    predictions are roughly normally distributed around the mean.
    This is a common approximation for RF uncertainty -- the true
    distribution is not Gaussian, so treat these as indicative ranges.

    Returns (lower_bound, upper_bound) in log10 x2 units.
    """
    lower_log = pred_mean - 1.96 * pred_std   # 95% lower bound (log10)
    upper_log = pred_mean + 1.96 * pred_std   # 95% upper bound (log10)
    return lower_log, upper_log


def build_feature_matrix(predictions_df: pd.DataFrame,
                          feature_cols: list) -> np.ndarray:
    """
    Build the feature matrix X for candidates in predictions_df.

    predictions_df must have columns matching feature_cols.
    We add T_K and P_kPa at screening conditions if they're missing.
    """
    # Add screening conditions if the prediction CSV doesn't have them
    if "T_K" not in predictions_df.columns:
        predictions_df["T_K"] = SCREENING_T_K
        print(f"[build_feature_matrix] T_K not found -- using {SCREENING_T_K} K")
    if "P_kPa" not in predictions_df.columns:
        predictions_df["P_kPa"] = SCREENING_P_kPa
        print(f"[build_feature_matrix] P_kPa not found -- using {SCREENING_P_kPa} kPa")

    # Verify all feature columns are present
    missing_cols = [c for c in feature_cols if c not in predictions_df.columns]
    if missing_cols:
        raise ValueError(
            f"{len(missing_cols)} feature columns missing from predictions CSV.\n"
            f"Example missing: {missing_cols[:5]}\n"
            f"Make sure to use a predictions CSV that was produced by inverse_design.py."
        )

    return predictions_df[feature_cols].values


def main():
    """
    Main pipeline: load model -> load predictions -> compute uncertainty -> save.
    """
    os.makedirs("results", exist_ok=True)

    # -- Step 1: Load model and training data --------------------------------
    model, feature_cols, model_name = load_model_bundle(MODEL_PATH)

    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(
            f"Train set not found at {TRAIN_CSV}. Needed to fit auxiliary RF."
        )
    train_df = pd.read_csv(TRAIN_CSV)
    print(f"[main] Train set: {train_df.shape[0]} rows, "
          f"{train_df['il_smiles'].nunique()} unique ILs")

    X_train        = train_df[feature_cols].values
    y_train        = train_df[TARGET_COL].values
    il_smiles_train = train_df["il_smiles"]

    # -- Step 2: Get an RF model for uncertainty estimation ------------------
    rf_model = get_rf_for_uncertainty(
        model, model_name, X_train, y_train, il_smiles_train, feature_cols
    )

    # -- Step 3: Load candidate predictions ----------------------------------
    if not os.path.exists(PREDICTIONS_CSV):
        raise FileNotFoundError(
            f"Predictions CSV not found at {PREDICTIONS_CSV}.\n"
            f"Run src/inverse_design.py or src/build_virtual_library_expanded.py first."
        )
    predictions_df = pd.read_csv(PREDICTIONS_CSV)
    print(f"[main] Loaded {len(predictions_df)} candidates from {PREDICTIONS_CSV}")

    # -- Step 4: Build feature matrix for candidates -------------------------
    X_candidates = build_feature_matrix(predictions_df, feature_cols)

    # -- Step 5: Compute predictions + uncertainty ---------------------------
    print("[main] Computing per-tree predictions (this takes ~10-30 seconds for large RF)...")
    pred_mean, pred_std = predict_with_tree_variance(rf_model, X_candidates)
    lower_log, upper_log = compute_confidence_intervals(pred_mean, pred_std)

    # Back-transform to mole fraction units for interpretability
    x2_pred_mean  = 10 ** pred_mean
    x2_pred_lower = 10 ** lower_log
    x2_pred_upper = 10 ** upper_log

    # -- Step 6: Assemble output DataFrame -----------------------------------
    output_df = predictions_df[["il_smiles"]].copy()

    # Carry over useful ID columns if they exist
    for col in ["cation_smiles", "anion_smiles", "cation_name", "anion_name",
                "pred_log_x2", "pred_x2"]:
        if col in predictions_df.columns:
            output_df[col] = predictions_df[col].values

    # Add uncertainty columns
    output_df["pred_log_x2_mean"]  = pred_mean
    output_df["pred_log_x2_std"]   = pred_std     # key uncertainty metric
    output_df["pred_log_x2_lower"] = lower_log    # 95% CI lower bound (log10)
    output_df["pred_log_x2_upper"] = upper_log    # 95% CI upper bound (log10)
    output_df["pred_x2_mean"]      = x2_pred_mean
    output_df["pred_x2_lower"]     = x2_pred_lower  # 95% CI lower (mole fraction)
    output_df["pred_x2_upper"]     = x2_pred_upper  # 95% CI upper (mole fraction)

    # Flag high-uncertainty predictions -- these need DFT validation most
    output_df["high_uncertainty"] = pred_std > HIGH_UNCERTAINTY_THRESHOLD

    # Sort by predicted x2 (highest = best CO2 absorption)
    output_df = output_df.sort_values("pred_x2_mean", ascending=False).reset_index(drop=True)

    # -- Step 7: Print summary -----------------------------------------------
    n_high_unc = output_df["high_uncertainty"].sum()
    print(f"\n[main] Results summary:")
    print(f"  Total candidates: {len(output_df)}")
    print(f"  High uncertainty (std > {HIGH_UNCERTAINTY_THRESHOLD}): {n_high_unc} "
          f"({100*n_high_unc/len(output_df):.1f}%)")
    print(f"\n  Top 10 candidates by predicted x2 (with uncertainty):")
    top10_cols = ["il_smiles", "pred_x2_mean", "pred_log_x2_std", "high_uncertainty"]
    available  = [c for c in top10_cols if c in output_df.columns]
    print(output_df.head(10)[available].to_string(index=False))
    print(f"\n  Uncertainty note: std is in log10(x2) units.")
    print(f"  std=0.10 means 95% CI spans roughly ±0.20 log10 units (~1.6x in x2).")
    print(f"  std=0.15 means 95% CI spans roughly ±0.30 log10 units (~2x in x2).")

    # -- Step 8: Save --------------------------------------------------------
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[main] Saved -> {OUTPUT_CSV}")
    print("[main] Use 'high_uncertainty=True' candidates as priority DFT targets.")


if __name__ == "__main__":
    main()
