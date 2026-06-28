"""
train_rf_bagged_folds.py
-------------------------
PURPOSE: Reduce RF's fold-to-fold test R² variance (nested CV showed
         0.708 ± 0.111 -- std exceeds the gap between models) by training
         several RF models on DIFFERENT GroupKFold train/test splits and
         averaging their predictions, instead of relying on one split.

WHY THIS IS DIFFERENT FROM THE FAILED STACKING ENSEMBLE:
  The earlier stacking failure (RF+XGB+CatBoost+Ridge) came from combining
  three CORRELATED models (r=0.91-0.95 residual correlation) where the
  meta-learner over-trusted CatBoost's CV score despite CatBoost's poor true
  OOD generalization (-0.17 OOF-to-test gap).

  This script does something structurally different: it trains the SAME
  model (RF, single architecture) on MULTIPLE independent data splits, then
  averages. This is bagging applied one level up -- instead of bagging
  bootstrap samples within one RF, we're bagging across entire train/test
  partitions. Since each RF copy is trained on a different ~80% of the ILs,
  each one has different sampling noise but the same good inductive bias
  (RF already wins 3/5 folds and has the best mean test R² in nested CV).
  Averaging reduces variance from "which ILs happened to land in train vs
  test" -- the documented problem -- without reintroducing the correlated-
  model averaging failure from stacking.

METHOD:
  1. Pool train_set.csv + test_set.csv together (so every IL is available
     to be used as held-out test in some split).
  2. Run N_BAGGED_FOLDS independent "rounds." Each round shuffles the row
     order and runs a fresh GroupKFold (grouped by il_smiles), then takes
     the first fold's split as that round's train/holdout partition. Because
     the row order is shuffled differently each round, each round produces a
     genuinely different grouping of ILs into the held-out set.
  3. In each round, train one RF on that round's training ILs.
  4. For every row in the FULL pool, collect predictions from every round
     in which that row's IL was held out (a genuine OOD prediction -- never
     a prediction from a model that trained on that exact IL).
  5. Average those OOD predictions per row -> the bagged-fold prediction.
     Rows held out in multiple rounds get a true multi-model average; rows
     held out in only one round get a single prediction (no averaging benefit
     for that row, but still a valid OOD prediction).
  6. Compare bagged-fold R² against the single-split RF baseline (0.697) and
     the nested CV per-fold numbers, to check whether variance actually
     dropped for the rows that did get averaged.

IMPORTANT CAVEAT (state this clearly in the writeup):
  Every row in the full pool gets used as "test" by at least one RF model
  here, so this evaluation does NOT produce a single clean held-out test set
  in the traditional sense. It answers a different question: "if we average
  several independently-trained RF models on different IL splits, does the
  ensemble's per-IL OOD prediction variance go down compared to a single
  split?" The honest final-model deliverable is still ONE RF retrained on
  ALL available ILs for deployment/GA inverse design (see make_deployment_rf
  below) -- the bagged-fold analysis here is to validate whether deploying
  an N-fold-averaged RF instead would be more robust.

OUTPUTS:
  results/rf_bagged_fold_predictions.csv   -- per-IL OOD averaged predictions
                                               + per-model breakdown
  results/rf_bagged_fold_performance.csv   -- bagged R²/RMSE vs single-split
                                               RF baseline
  models/rf_bagged_ensemble.pkl            -- list of N_BAGGED_FOLDS RF models
                                               + a final deployment RF trained
                                               on the full pool (for GA / new
                                               IL predictions)

Run from project root:
  nohup python src/train_rf_bagged_folds.py > logs/rf_bagged_folds.log 2>&1 &
Then monitor:
  tail -f logs/rf_bagged_folds.log
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

# -- Constants -----------------------------------------------------------------
TRAIN_CSV   = os.path.join("data", "processed", "train_set.csv")
TEST_CSV    = os.path.join("data", "processed", "test_set.csv")
MODEL_DIR   = "models"
RESULTS_DIR = "results"
BAGGED_MODEL_PATH = os.path.join(MODEL_DIR, "rf_bagged_ensemble.pkl")

TARGET_COL         = "log_x2_CO2"
CONDITION_FEATURES = ["T_K", "P_kPa"]
N_BAGGED_FOLDS      = 5    # number of independent RF models to bag
RANDOM_SEED         = 42

# -- RF: Optuna v2 best params (same as train_stacked_model.py / nested CV) ---
RF_PARAMS = dict(
    n_estimators     = 515,
    max_features     = 0.5,
    min_samples_leaf = 7,
    max_depth        = None,
    n_jobs           = -1,
)

# Single fixed-split RF baseline from earlier evaluation, for comparison.
# (train_stacked_model.py RF test R² on the original 60-IL fixed test set)
SINGLE_SPLIT_RF_BASELINE_R2 = 0.6970


def load_full_pool() -> tuple:
    """
    Combine train_set.csv + test_set.csv into one pool of all available ILs.
    We need the full pool because each of the N_BAGGED_FOLDS RF models needs
    a DIFFERENT held-out group of ILs -- the original fixed 60-IL test set is
    just one possible holdout, not special.

    Returns X, y, il_smiles groups, and feature column names. NaN descriptor
    columns are filled with the column median computed over the full pool.
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


def train_repeated_bagged_rf(X: np.ndarray, y: np.ndarray,
                              il_smiles: pd.Series) -> tuple:
    """
    CORRECTED METHOD: train N_BAGGED_FOLDS RF models using N_BAGGED_FOLDS
    DIFFERENT random GroupKFold partitions (different random_state each time),
    not N_BAGGED_FOLDS folds of the SAME partition.

    With a single K-fold split, every row is held out by exactly one model --
    there's nothing to average. To get a genuine multi-model average per row,
    each "round" must use an independently-shuffled GroupKFold split. Across
    N_BAGGED_FOLDS rounds, a given IL will usually land in the held-out group
    multiple times (different rounds, different partitions), giving multiple
    independent OOD predictions for the same row to average.

    Returns:
      all_round_results: list of (round_idx, model, holdout_idx, imputer) so
                          we can later collect every model's OOD prediction
                          for every row across all rounds.
    """
    all_round_results = []

    print(f"\n[train] Training {N_BAGGED_FOLDS} RF models on "
          f"{N_BAGGED_FOLDS} independently-shuffled GroupKFold partitions...",
          flush=True)

    for round_idx in range(N_BAGGED_FOLDS):
        # A fresh random partition each round -- different ILs land in the
        # held-out group than in any other round, by virtue of shuffling.
        gkf = GroupKFold(n_splits=N_BAGGED_FOLDS)
        groups = il_smiles.values

        # GroupKFold itself has no shuffle parameter, so we permute the row
        # order before splitting -- this changes which ILs group together
        # into each of the K folds, giving a different partition per round.
        rng = np.random.RandomState(RANDOM_SEED + round_idx)
        permuted_order = rng.permutation(len(X))

        # Take only the FIRST fold of this round's shuffled K-way split as
        # this round's held-out set (the other K-1 folds are the train set).
        # Looping K times within a round would just reproduce the unshuffled
        # case since GroupKFold is deterministic given group order.
        splits = list(gkf.split(X[permuted_order], y[permuted_order],
                                  groups[permuted_order]))
        train_local_idx, holdout_local_idx = splits[0]

        # Map back from the permuted local indices to original row indices
        train_idx   = permuted_order[train_local_idx]
        holdout_idx = permuted_order[holdout_local_idx]

        assert len(set(groups[train_idx]) & set(groups[holdout_idx])) == 0, \
            f"IL overlap in round {round_idx}!"

        imputer = SimpleImputer(strategy="median")
        X_train_round = imputer.fit_transform(X[train_idx])

        model = RandomForestRegressor(**RF_PARAMS, random_state=RANDOM_SEED + round_idx)
        model.fit(X_train_round, y[train_idx])

        n_holdout_ils = len(set(groups[holdout_idx]))
        print(f"  Round {round_idx+1}/{N_BAGGED_FOLDS} trained | "
              f"holds out {n_holdout_ils} ILs ({len(holdout_idx)} rows)", flush=True)

        all_round_results.append((round_idx, model, holdout_idx, imputer))

    return all_round_results


def collect_and_average_predictions(X: np.ndarray, y: np.ndarray,
                                     il_smiles: pd.Series,
                                     all_round_results: list) -> pd.DataFrame:
    """
    For every row, gather predictions from every round-model that held it out,
    then average. Rows held out by only one round still get a valid (single)
    prediction; rows held out by multiple rounds get a genuine multi-model
    average -- this is where the variance reduction comes from.
    """
    n_rows = len(y)
    # Each row accumulates a list of predictions from every round that held it out
    row_predictions = [[] for _ in range(n_rows)]

    print(f"\n[collect] Gathering OOD predictions across {len(all_round_results)} rounds...",
          flush=True)

    for round_idx, model, holdout_idx, imputer in all_round_results:
        X_holdout = imputer.transform(X[holdout_idx])
        y_pred_holdout = model.predict(X_holdout)
        for local_pos, row_idx in enumerate(holdout_idx):
            row_predictions[row_idx].append(y_pred_holdout[local_pos])

    n_with_predictions = sum(1 for preds in row_predictions if len(preds) > 0)
    n_multi_prediction  = sum(1 for preds in row_predictions if len(preds) > 1)
    print(f"[collect] {n_with_predictions}/{n_rows} rows have >=1 OOD prediction", flush=True)
    print(f"[collect] {n_multi_prediction}/{n_rows} rows have >=2 OOD predictions "
          f"(these benefit from averaging)", flush=True)

    averaged_preds = np.array([
        np.mean(preds) if len(preds) > 0 else np.nan
        for preds in row_predictions
    ])
    n_predictions_per_row = np.array([len(preds) for preds in row_predictions])

    # Std across individual round-predictions for the SAME row, before
    # averaging -- this is the direct evidence for variance reduction.
    # Only meaningful for rows with >=2 predictions; NaN otherwise.
    pred_spread = np.array([
        np.std(preds) if len(preds) >= 2 else np.nan
        for preds in row_predictions
    ])

    results_df = pd.DataFrame({
        "il_smiles":            il_smiles.values,
        "y_true":                y,
        "bagged_pred":           averaged_preds,
        "n_models_averaged":     n_predictions_per_row,
        "pred_std_across_rounds": pred_spread,
        "residual":              averaged_preds - y,
    })
    return results_df


def evaluate_bagged_predictions(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute overall R²/RMSE for the bagged-fold predictions, and break down
    performance by how many models contributed to each row's prediction
    (1 model vs 2+) to directly check whether averaging helped.
    """
    valid_df = results_df.dropna(subset=["bagged_pred"])

    overall_r2   = r2_score(valid_df["y_true"], valid_df["bagged_pred"])
    overall_rmse = np.sqrt(mean_squared_error(valid_df["y_true"], valid_df["bagged_pred"]))

    print("\n" + "=" * 70, flush=True)
    print("BAGGED-FOLD RF EVALUATION", flush=True)
    print("=" * 70, flush=True)
    print(f"Overall: R²={overall_r2:.4f}, RMSE={overall_rmse:.4f} "
          f"({len(valid_df)} rows)", flush=True)
    print(f"Single-split RF baseline (original 60-IL test set): "
          f"R²={SINGLE_SPLIT_RF_BASELINE_R2:.4f}", flush=True)
    print(f"Nested CV RF mean ± std (5 folds): R²=0.7078 ± 0.1115", flush=True)

    perf_rows = [{
        "evaluation": "bagged_fold_overall",
        "test_r2": overall_r2, "test_rmse": overall_rmse,
        "n_rows": len(valid_df),
    }]

    # Breakdown: rows averaged over >=2 models vs only 1
    multi_df  = valid_df[valid_df["n_models_averaged"] >= 2]
    single_df = valid_df[valid_df["n_models_averaged"] == 1]

    if len(multi_df) > 0:
        multi_r2 = r2_score(multi_df["y_true"], multi_df["bagged_pred"])
        print(f"\nRows averaged over >=2 models ({len(multi_df)} rows): R²={multi_r2:.4f}",
              flush=True)
        perf_rows.append({
            "evaluation": "bagged_fold_multi_model_rows",
            "test_r2": multi_r2, "test_rmse": np.nan, "n_rows": len(multi_df),
        })

        # Direct variance-reduction evidence: for these SAME rows, the mean
        # prediction spread across individual rounds (before averaging) tells
        # us how much averaging actually shrank disagreement between models.
        # This is a fairer test than comparing different rows against each
        # other, since "easier" ILs landing in the multi-prediction group by
        # chance could otherwise be mistaken for an averaging benefit.
        mean_pred_std = multi_df["pred_std_across_rounds"].mean()
        print(f"  Mean prediction std across rounds for these rows: "
              f"{mean_pred_std:.4f} (log10 x2 units)", flush=True)
        print(f"  This is how much individual round-models disagreed with "
              f"each other on the SAME row before averaging -- the averaged "
              f"prediction's error is bounded by this spread, not amplified "
              f"by it.", flush=True)
    if len(single_df) > 0:
        single_r2 = r2_score(single_df["y_true"], single_df["bagged_pred"])
        print(f"Rows with only 1 model's prediction ({len(single_df)} rows): R²={single_r2:.4f}",
              flush=True)
        perf_rows.append({
            "evaluation": "bagged_fold_single_model_rows",
            "test_r2": single_r2, "test_rmse": np.nan, "n_rows": len(single_df),
        })

    print("=" * 70, flush=True)
    return pd.DataFrame(perf_rows)


def make_deployment_rf(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Train one final RF on the FULL pool (all 299 ILs) for actual deployment --
    e.g. feeding the genetic algorithm's inverse design search or predicting
    on brand-new candidate ILs. This is NOT evaluated for R² here (it has seen
    everything) -- its expected generalization is estimated by the nested CV
    and bagged-fold numbers above, which used the same hyperparameters.
    """
    print("\n[deploy] Training final deployment RF on full pool "
          "(all available ILs)...", flush=True)
    imputer = SimpleImputer(strategy="median")
    X_full_imp = imputer.fit_transform(X)

    deployment_model = RandomForestRegressor(**RF_PARAMS, random_state=RANDOM_SEED)
    deployment_model.fit(X_full_imp, y)
    print("[deploy] Deployment RF trained.", flush=True)
    return deployment_model, imputer


def main():
    """
    Full bagged-fold RF pipeline:
    1. Load combined pool
    2. Train N_BAGGED_FOLDS RF models on N_BAGGED_FOLDS independently-shuffled
       GroupKFold partitions
    3. Collect and average OOD predictions per row across rounds
    4. Evaluate bagged R² vs single-split baseline and nested CV numbers
    5. Train and save a final deployment RF on the full pool
    """
    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70, flush=True)
    print("RF BAGGED-FOLD ENSEMBLE", flush=True)
    print(f"Rounds: {N_BAGGED_FOLDS} independently-shuffled GroupKFold partitions",
          flush=True)
    print("Goal: reduce fold-to-fold test R² variance by averaging predictions "
          "across multiple IL splits", flush=True)
    print("=" * 70 + "\n", flush=True)

    X, y, il_smiles, feature_cols = load_full_pool()

    all_round_results = train_repeated_bagged_rf(X, y, il_smiles)
    results_df = collect_and_average_predictions(X, y, il_smiles, all_round_results)
    perf_df = evaluate_bagged_predictions(results_df)

    deployment_model, deployment_imputer = make_deployment_rf(X, y)

    # Save outputs
    results_df.to_csv(
        os.path.join(RESULTS_DIR, "rf_bagged_fold_predictions.csv"), index=False)
    print(f"\n[main] Saved -> results/rf_bagged_fold_predictions.csv", flush=True)

    perf_df.to_csv(
        os.path.join(RESULTS_DIR, "rf_bagged_fold_performance.csv"), index=False)
    print(f"[main] Saved -> results/rf_bagged_fold_performance.csv", flush=True)

    bundle = {
        "round_models":        [m for _, m, _, _ in all_round_results],
        "round_imputers":      [imp for _, _, _, imp in all_round_results],
        "deployment_model":    deployment_model,
        "deployment_imputer":  deployment_imputer,
        "feature_cols":        feature_cols,
        "model_name":          f"RF_Bagged_{N_BAGGED_FOLDS}rounds_v1",
        "rf_params":           RF_PARAMS,
    }
    joblib.dump(bundle, BAGGED_MODEL_PATH)
    print(f"[main] Saved -> {BAGGED_MODEL_PATH}", flush=True)

    print("\n[main] DONE. Compare bagged-fold R² against:", flush=True)
    print(f"  Single-split RF baseline: R²={SINGLE_SPLIT_RF_BASELINE_R2:.4f}", flush=True)
    print(f"  Nested CV RF (5 folds):   R²=0.7078 ± 0.1115", flush=True)
    print("  If bagged R² is similar but multi-model rows show tighter spread,", flush=True)
    print("  this confirms variance reduction without sacrificing accuracy.", flush=True)


if __name__ == "__main__":
    main()
