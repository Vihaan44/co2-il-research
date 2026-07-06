"""
build_dataset_v2.py
--------------------
PURPOSE: Merge il_features_v2.csv (all feature blocks) with the CO2 datapoints
         CSV and produce five versioned train/test splits -- one per feature
         variant being ablated. Each variant adds exactly ONE new block on top
         of the Morgan+18desc baseline so ablation results are clean.

VARIANTS PRODUCED:
  v1_baseline   -- Morgan 2048-bit x2 + 18 RDKit descriptors x2
  v2_maccs      -- v1 + MACCS keys (166 bits x2 ions)
  v3_anion_elec -- v1 + anion electronic features (6 features)
  v4_cross      -- v1 + ion-pair cross features (4 features)
  v5_mordred    -- v1 + Mordred 2D descriptors (up to 200 x2 ions)

OUTPUTS (data/processed/):
  train_set_v1_baseline.csv / test_set_v1_baseline.csv
  train_set_v2_maccs.csv    / test_set_v2_maccs.csv
  train_set_v3_anion_elec.csv / test_set_v3_anion_elec.csv
  train_set_v4_cross.csv    / test_set_v4_cross.csv
  train_set_v5_mordred.csv  / test_set_v5_mordred.csv

INPUTS:
  data/raw/all_co2_datapoints_merged.csv
  data/processed/il_features_v2.csv  (from featurize_v2.py)

Run from project root:
  python src/build_dataset_v2.py
"""

import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# -- Constants -----------------------------------------------------------------
DATAPOINTS_CSV  = os.path.join("data", "raw",       "all_co2_datapoints_merged.csv")
FEATURES_V2_CSV = os.path.join("data", "processed", "il_features_v2.csv")
OUTPUT_DIR      = os.path.join("data", "processed")

TARGET_COL         = "log_x2_CO2"
CONDITION_FEATURES = ["T_K", "P_kPa"]
TRAIN_FRACTION      = 0.80
RANDOM_SEED         = 42


def get_variant_columns(all_feature_cols: list) -> dict:
    """
    Return a dict mapping variant name to list of feature column names.
    Each variant selects the baseline columns plus at most one additional block.
    Prefix conventions from featurize_v2.py:
      cat_fp_{i}, an_fp_{i}               -- Morgan FP bits
      cat_{desc}, an_{desc}               -- RDKit 18 descriptors
      cat_maccs_{i}, an_maccs_{i}         -- MACCS key bits
      an_elec_{feat}                      -- anion electronic features
      pair_{feat}                         -- ion-pair cross features
      cat_mordred_{desc}, an_mordred_{desc} -- Mordred descriptors
    """
    # Baseline: Morgan FP bits + RDKit 18 descriptors only
    baseline_cols = [
        c for c in all_feature_cols
        if ("_fp_" in c)
        or (
            (c.startswith("cat_") or c.startswith("an_"))
            and "_fp_" not in c
            and "_maccs_" not in c
            and "_elec_" not in c
            and "_mordred_" not in c
        )
    ]

    maccs_cols   = [c for c in all_feature_cols if "_maccs_" in c]
    elec_cols    = [c for c in all_feature_cols if "_elec_" in c]
    cross_cols   = [c for c in all_feature_cols if c.startswith("pair_")]
    mordred_cols = [c for c in all_feature_cols if "_mordred_" in c]

    return {
        "v1_baseline":   baseline_cols,
        "v2_maccs":      baseline_cols + maccs_cols,
        "v3_anion_elec": baseline_cols + elec_cols,
        "v4_cross":      baseline_cols + cross_cols,
        "v5_mordred":    baseline_cols + mordred_cols,
    }


def merge_datapoints_with_features(datapoints_df: pd.DataFrame,
                                    features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join CO2 datapoints onto IL features on il_smiles.
    Drops rows where the IL had no matching features and rows missing T_K/P_kPa.
    """
    merged = datapoints_df.merge(features_df, on="il_smiles", how="left")
    before = len(merged)
    merged = merged.dropna(subset=CONDITION_FEATURES).copy()
    print(f"  Merged: {before} -> {len(merged)} rows after dropping missing conditions",
          flush=True)
    return merged


def split_by_il(merged_df: pd.DataFrame) -> tuple:
    """
    Split into train/test by IL identity using GroupShuffleSplit on il_smiles.
    No IL appears in both train and test -- same protocol as original pipeline.
    Returns (train_df, test_df).
    """
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=1 - TRAIN_FRACTION, random_state=RANDOM_SEED
    )
    groups = merged_df["il_smiles"].values
    train_idx, test_idx = next(splitter.split(merged_df, groups=groups))

    train_df = merged_df.iloc[train_idx].copy()
    test_df  = merged_df.iloc[test_idx].copy()

    print(f"  Train: {len(train_df)} rows, {train_df['il_smiles'].nunique()} ILs", flush=True)
    print(f"  Test:  {len(test_df)} rows,  {test_df['il_smiles'].nunique()} ILs", flush=True)

    # Sanity check: confirm no IL appears in both splits
    overlap = set(train_df["il_smiles"]) & set(test_df["il_smiles"])
    assert len(overlap) == 0, f"IL LEAKAGE: {len(overlap)} ILs in both train and test!"
    return train_df, test_df


def save_variant(train_df: pd.DataFrame, test_df: pd.DataFrame,
                  variant_name: str, variant_cols: list, output_dir: str):
    """
    Select only the relevant feature columns for this variant and save
    train/test CSVs. Always includes il_smiles, TARGET_COL, CONDITION_FEATURES.
    """
    keep_cols = ["il_smiles", TARGET_COL] + CONDITION_FEATURES + variant_cols
    existing  = [c for c in keep_cols if c in train_df.columns]
    missing   = [c for c in keep_cols if c not in train_df.columns]
    if missing:
        print(f"  [DATA QUALITY] {len(missing)} columns missing for {variant_name}",
              flush=True)

    train_out = os.path.join(output_dir, f"train_set_{variant_name}.csv")
    test_out  = os.path.join(output_dir, f"test_set_{variant_name}.csv")
    train_df[existing].to_csv(train_out, index=False)
    test_df[existing].to_csv(test_out,  index=False)

    n_feat = len(existing) - 3  # subtract il_smiles, target, 2 conditions
    print(f"  Saved {variant_name}: {n_feat} features", flush=True)


def main():
    """Load datapoints + features, merge, split by IL, write all 5 variant CSVs."""
    print("=" * 65, flush=True)
    print("build_dataset_v2.py -- producing all feature variant datasets", flush=True)
    print("=" * 65, flush=True)

    if not os.path.exists(DATAPOINTS_CSV):
        raise FileNotFoundError(f"Datapoints not found: {DATAPOINTS_CSV}")
    if not os.path.exists(FEATURES_V2_CSV):
        raise FileNotFoundError(
            f"Features not found: {FEATURES_V2_CSV}. Run featurize_v2.py first."
        )

    print(f"\n[load] Datapoints: {DATAPOINTS_CSV}", flush=True)
    datapoints_df = pd.read_csv(DATAPOINTS_CSV)
    print(f"  {len(datapoints_df)} rows, {datapoints_df['il_smiles'].nunique()} unique ILs",
          flush=True)

    print(f"\n[load] Features v2: {FEATURES_V2_CSV}", flush=True)
    features_df = pd.read_csv(FEATURES_V2_CSV)
    print(f"  {len(features_df)} ILs, {features_df.shape[1]} columns", flush=True)

    print("\n[merge] Joining datapoints onto IL features...", flush=True)
    merged_df = merge_datapoints_with_features(datapoints_df, features_df)

    print("\n[split] Splitting by IL identity...", flush=True)
    train_df, test_df = split_by_il(merged_df)

    # Identify feature columns per variant (exclude metadata columns)
    all_feature_cols = [
        c for c in features_df.columns
        if c not in ("il_smiles", "il_name", "cation_smiles", "anion_smiles")
    ]
    variants = get_variant_columns(all_feature_cols)

    print("\n[variants] Feature counts per variant:", flush=True)
    for name, cols in variants.items():
        print(f"  {name:20s}: {len(cols)} features", flush=True)

    print("\n[save] Writing variant CSVs...", flush=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for variant_name, variant_cols in variants.items():
        save_variant(train_df, test_df, variant_name, variant_cols, OUTPUT_DIR)

    print("\n[main] All variants saved.", flush=True)
    print("NEXT STEP: nohup python src/ablate_feature_sets.py > logs/feature_ablation.log 2>&1 &",
          flush=True)


if __name__ == "__main__":
    main()
