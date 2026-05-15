"""
build_dataset.py
----------------
PURPOSE: Merge (T, P, x2) measurement rows with IL structural features to produce
         one ML-ready dataset, then split into train and test sets.

KEY DESIGN DECISION — STRATIFIED SPLIT BY IL IDENTITY:
  We use GroupShuffleSplit with IL SMILES as the group key, NOT a random row split.

  Why this matters: if we split rows randomly, measurements of the same IL land in
  both train and test. The model sees [BMIM][BF4] at 298K in training, then gets
  tested on [BMIM][BF4] at 320K — that's not generalization, that's interpolation.
  Our scientific goal is predicting CO2 absorption for NOVEL ILs we've never seen.
  GroupShuffleSplit guarantees every IL is entirely in train OR test, never both.

LOG TRANSFORM:
  x2_CO2 values span multiple orders of magnitude (0.001 to 0.8).
  We predict log10(x2) instead of x2 directly. This spreads the target distribution
  more evenly and makes the regression task easier for tree-based models.
  We inverse-transform (10^y) predictions before reporting results.

MISSING P_kPa HANDLING:
  A small number of ILThermo entries report x2 without a pressure value (typically
  measurements implicitly assumed to be at atmospheric pressure ~101.325 kPa).
  We drop these rows rather than imputing, because:
    1. P_kPa is a strong predictor (Henry's law: x2 ∝ P); imputing wrongly biases predictions.
    2. 65 rows out of ~9000 is <1% of the dataset — negligible data loss.
  The affected ILs are reported so they can be manually checked against ILThermo.

INPUT:
  data/raw/ilthermo_mole_fraction_datapoints.csv  (from fetch_datapoints.py)
  data/processed/il_features.csv                  (from featurize.py)

OUTPUT:
  data/processed/ml_dataset.csv          — full merged dataset (NaN-clean)
  data/processed/train_set.csv           — training rows (80% of ILs)
  data/processed/test_set.csv            — held-out test rows (20% of ILs)
  results/dataset_summary.csv            — statistics for the paper
  data/processed/train_test_split_info.json — which ILs are in which split

Run from project root:
    python src/build_dataset.py
"""

import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import GroupShuffleSplit

# ── Constants ──────────────────────────────────────────────────────────────────
DATAPOINTS_CSV  = os.path.join("data", "raw",       "ilthermo_mole_fraction_datapoints.csv")
FEATURES_CSV    = os.path.join("data", "processed", "il_features.csv")
ML_DATASET_CSV  = os.path.join("data", "processed", "ml_dataset.csv")
TRAIN_CSV       = os.path.join("data", "processed", "train_set.csv")
TEST_CSV        = os.path.join("data", "processed", "test_set.csv")
SUMMARY_CSV     = os.path.join("results",           "dataset_summary.csv")
SPLIT_JSON      = os.path.join("data", "processed", "train_test_split_info.json")

TEST_FRACTION   = 0.20   # 20% of unique ILs go to test set
RANDOM_SEED     = 42     # fixed seed for reproducibility
TARGET_COL      = "log_x2_CO2"   # ML target (log-transformed)
RAW_TARGET_COL  = "x2_CO2"       # original measurement

# Condition columns that must be non-null for a row to be usable
REQUIRED_CONDITION_COLS = ["T_K", "P_kPa"]


def load_csv(path: str, label: str) -> pd.DataFrame:
    """Load a CSV and print its shape. Raises if missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found at {path}. Run prior step first.")
    df = pd.read_csv(path)
    print(f"[load] {label}: {df.shape[0]} rows × {df.shape[1]} cols")
    return df


def drop_missing_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where T_K or P_kPa is missing.

    These rows cannot be used for training or evaluation because T and P are
    required model inputs. XGBoost handles NaN internally via learned split
    directions, but this is not principled for physical features with known
    causal roles (Henry's law). We prefer to drop the 65 affected rows (<1%)
    and report which ILs are affected for transparency.

    If an IL loses ALL its rows here, it will be absent from training entirely.
    That IL is reported separately so it can be manually investigated.
    """
    before = len(df)
    # Find which ILs are affected before dropping
    affected_mask = df[REQUIRED_CONDITION_COLS].isna().any(axis=1)
    affected_ils  = df.loc[affected_mask, "il_name"].unique().tolist()

    df_clean = df.dropna(subset=REQUIRED_CONDITION_COLS).copy()
    dropped  = before - len(df_clean)

    if dropped > 0:
        # DATA QUALITY FLAG: document exactly what was removed and why
        print(f"[drop_missing_conditions] Dropped {dropped} rows with missing T_K or P_kPa")
        print(f"  Affected ILs ({len(affected_ils)}): {affected_ils[:10]}"
              f"{'...' if len(affected_ils) > 10 else ''}")
        print(f"  Rationale: T and P are causal predictors (Henry's law); "
              f"imputing would introduce systematic bias.")
        print(f"  Data loss: {dropped}/{before} = {100*dropped/before:.1f}% of rows")

        # Warn if any IL lost all its rows (would vanish from training)
        ils_after  = set(df_clean["il_name"].unique())
        ils_before = set(df["il_name"].unique())
        fully_lost = ils_before - ils_after
        if fully_lost:
            print(f"  WARNING: {len(fully_lost)} IL(s) lost ALL rows and will be absent from training:")
            for il in sorted(fully_lost):
                print(f"    - {il}")
    else:
        print(f"[drop_missing_conditions] No missing T_K or P_kPa — all rows clean.")

    return df_clean


def add_log_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a log10-transformed CO2 mole fraction column as the ML regression target.
    Rows with x2 <= 0 are dropped (can't take log of zero/negative).

    log10 chosen over ln because: log10(0.01) = -2 and log10(0.1) = -1,
    making the scale intuitive when discussing results with judges.
    """
    before = len(df)
    df = df[df[RAW_TARGET_COL] > 0].copy()  # drop non-positive x2
    dropped = before - len(df)
    if dropped > 0:
        # DATA QUALITY FLAG
        print(f"[add_log_target] Dropped {dropped} rows with x2_CO2 <= 0")

    df[TARGET_COL] = np.log10(df[RAW_TARGET_COL])
    print(f"[add_log_target] {TARGET_COL} range: "
          f"{df[TARGET_COL].min():.3f} to {df[TARGET_COL].max():.3f}")
    return df


def merge_features(datapoints_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join measurement rows with IL features on il_smiles (the canonical IL identity).
    Uses inner join — rows without a matching featurized IL are dropped and reported.

    We merge on SMILES not il_name because names can have inconsistent spacing/
    capitalization across ILThermo entries. SMILES is the reliable structural key.
    """
    # Only bring feature columns across (not duplicate metadata)
    feature_cols = [c for c in features_df.columns
                    if c.startswith("cat_") or c.startswith("an_")]
    smiles_plus_features = features_df[["il_smiles"] + feature_cols]

    before = len(datapoints_df)
    merged_df = datapoints_df.merge(smiles_plus_features, on="il_smiles", how="inner")
    dropped = before - len(merged_df)

    if dropped > 0:
        # DATA QUALITY FLAG: rows dropped = ILs in datapoints but not in features
        print(f"[merge_features] WARNING: {dropped} rows dropped during merge")
        print(f"  This means some ILs have no features (likely missing/invalid SMILES).")

    print(f"[merge_features] Merged: {len(merged_df)} rows × {merged_df.shape[1]} cols")
    return merged_df


def stratified_il_split(merged_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by IL identity so all rows for a given IL stay in the same split.
    Uses sklearn's GroupShuffleSplit with il_smiles as the group key.

    This is the correct evaluation protocol for generalizing to novel ILs.
    A random row split would leak IL identity into the test set and give
    artificially high R² values.
    """
    groups = merged_df["il_smiles"]  # group key: unique IL identity
    n_unique = groups.nunique()
    print(f"\n[stratified_il_split] {n_unique} unique ILs → "
          f"~{int(n_unique * TEST_FRACTION)} in test, ~{int(n_unique * (1-TEST_FRACTION))} in train")

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=RANDOM_SEED)
    train_idx, test_idx = next(splitter.split(merged_df, groups=groups))

    train_df = merged_df.iloc[train_idx].reset_index(drop=True)
    test_df  = merged_df.iloc[test_idx].reset_index(drop=True)

    # Verify zero IL overlap — this must always pass
    train_smiles = set(train_df["il_smiles"])
    test_smiles  = set(test_df["il_smiles"])
    overlap = train_smiles & test_smiles
    if overlap:
        print(f"[ERROR] {len(overlap)} ILs appear in BOTH splits — data leakage detected!")
    else:
        print(f"[stratified_il_split] ✓ Zero IL overlap confirmed between train and test")

    print(f"  Train: {len(train_df)} rows ({train_df['il_name'].nunique()} unique ILs)")
    print(f"  Test : {len(test_df)} rows ({test_df['il_name'].nunique()} unique ILs)")
    return train_df, test_df


def save_split_info(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Save JSON record of which ILs are in train vs test.
    Lets us verify reproducibility if features are ever regenerated.
    """
    split_info = {
        "random_seed":      RANDOM_SEED,
        "test_fraction":    TEST_FRACTION,
        "train_il_names":   sorted(train_df["il_name"].unique().tolist()),
        "test_il_names":    sorted(test_df["il_name"].unique().tolist()),
        "train_row_count":  int(len(train_df)),
        "test_row_count":   int(len(test_df)),
    }
    with open(SPLIT_JSON, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"[save_split_info] Saved split metadata → {SPLIT_JSON}")


def generate_summary(merged_df: pd.DataFrame,
                     train_df: pd.DataFrame,
                     test_df: pd.DataFrame) -> pd.DataFrame:
    """Generate dataset statistics table for the paper methods section."""
    feature_dim = sum(1 for c in merged_df.columns if c.startswith("cat_") or c.startswith("an_"))
    summary_rows = [
        {"metric": "total_measurement_rows",  "value": len(merged_df)},
        {"metric": "unique_ILs",              "value": int(merged_df["il_name"].nunique())},
        {"metric": "T_K_min",                 "value": float(merged_df["T_K"].min())},
        {"metric": "T_K_max",                 "value": float(merged_df["T_K"].max())},
        {"metric": "P_kPa_min",               "value": float(merged_df["P_kPa"].min())},
        {"metric": "P_kPa_max",               "value": float(merged_df["P_kPa"].max())},
        {"metric": "x2_CO2_min",              "value": float(merged_df["x2_CO2"].min())},
        {"metric": "x2_CO2_max",              "value": float(merged_df["x2_CO2"].max())},
        {"metric": "log_x2_mean",             "value": float(merged_df[TARGET_COL].mean())},
        {"metric": "log_x2_std",              "value": float(merged_df[TARGET_COL].std())},
        {"metric": "train_rows",              "value": int(len(train_df))},
        {"metric": "train_unique_ILs",        "value": int(train_df["il_name"].nunique())},
        {"metric": "test_rows",               "value": int(len(test_df))},
        {"metric": "test_unique_ILs",         "value": int(test_df["il_name"].nunique())},
        {"metric": "feature_vector_dim",      "value": feature_dim},
    ]
    summary_df = pd.DataFrame(summary_rows)
    print("\n── DATASET SUMMARY ──────────────────────────────────────────────────")
    print(summary_df.to_string(index=False))
    print("─────────────────────────────────────────────────────────────────────\n")
    return summary_df


def main():
    """Main: load → drop NaN conditions → log-transform → merge features → split → save."""
    datapoints_df = load_csv(DATAPOINTS_CSV, "Datapoints")
    features_df   = load_csv(FEATURES_CSV,   "Features")

    # Drop rows with missing T_K or P_kPa BEFORE log-transform and merge
    # so the row count printed by add_log_target reflects only usable rows
    datapoints_df = drop_missing_conditions(datapoints_df)
    datapoints_df = add_log_target(datapoints_df)
    merged_df     = merge_features(datapoints_df, features_df)
    train_df, test_df = stratified_il_split(merged_df)

    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    os.makedirs("results", exist_ok=True)

    merged_df.to_csv(ML_DATASET_CSV, index=False)
    train_df.to_csv(TRAIN_CSV,       index=False)
    test_df.to_csv(TEST_CSV,         index=False)
    save_split_info(train_df, test_df)

    summary_df = generate_summary(merged_df, train_df, test_df)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print(f"[main] All outputs saved:")
    print(f"  {ML_DATASET_CSV}  ({len(merged_df)} rows)")
    print(f"  {TRAIN_CSV}  ({len(train_df)} rows)")
    print(f"  {TEST_CSV}  ({len(test_df)} rows)")
    print(f"  {SPLIT_JSON}")
    print(f"  {SUMMARY_CSV}")
    print("\n[main] Phase 2 complete. Ready for src/train_model.py (Phase 3).")


if __name__ == "__main__":
    main()
