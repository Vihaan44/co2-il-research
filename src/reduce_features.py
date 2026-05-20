"""
reduce_features.py
------------------
PURPOSE: Remove near-zero-variance Morgan fingerprint bits from the feature
         matrix to reduce noise and improve model generalization.

THE PROBLEM:
  We have 4096 Morgan FP bits (2048 per ion) across 211 training ILs.
  Many bits are set in only 1-2 ILs out of 211, or in all but 1-2 ILs.
  These near-constant bits carry almost no discriminative information and
  add noise that random forests must waste splits on.

  A feature with variance=0 is literally constant -- the model can never
  use it. Features with variance < threshold are nearly constant and
  statistically unreliable.

WHY VARIANCE AT THE IL LEVEL, NOT THE ROW LEVEL:
  Each IL appears in many (T, P) rows. If we compute variance over all rows,
  [BMIM] at 300 temperatures inflates the count for [BMIM]-related bits,
  making them look more common than they are.
  We compute variance over unique IL SMILES in the training set -- 211 values
  per bit, one per unique IL. This is the honest measure of how much a bit
  discriminates between different IL structures.

THRESHOLD SELECTION:
  A bit set in k out of N ILs has variance = (k/N)(1 - k/N).
  VARIANCE_THRESHOLD = 0.01 corresponds to bits set in fewer than ~2/211
  ILs or more than ~209/211 ILs. These are essentially constant.

  We also print the threshold sensitivity table so you can see how many
  features survive at different cutoffs -- useful for tuning.

WHAT THIS SCRIPT DOES:
  1. Loads train_set.csv and extracts unique-IL feature matrix (211 rows)
  2. Computes per-bit variance at the IL level
  3. Drops FP bits below VARIANCE_THRESHOLD (keeps all RDKit descriptors)
  4. Saves filtered_il_features.csv with only surviving bits
  5. Reports how many bits were dropped and from which ion

  After running this script, you must re-run build_dataset.py to rebuild
  train_set.csv and test_set.csv with the reduced feature set, then retrain.

INPUTS:
  data/processed/train_set.csv      (to compute IL-level variance)
  data/processed/il_features.csv    (full feature matrix to filter)

OUTPUTS:
  data/processed/filtered_il_features.csv   -- reduced feature matrix
  results/feature_variance_report.csv       -- per-bit variance + kept/dropped
  figures/feature_variance_histogram.png    -- distribution of bit variances

Run from project root:
    python src/reduce_features.py
    # Then re-run the pipeline:
    python src/build_dataset.py
    python src/train_stacked_model.py  (or train_model.py)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for codespace
import matplotlib.pyplot as plt

# -- Constants -----------------------------------------------------------------
TRAIN_CSV            = os.path.join("data", "processed", "train_set.csv")
IL_FEATURES_CSV      = os.path.join("data", "processed", "il_features.csv")
FILTERED_FEATURES_CSV = os.path.join("data", "processed", "filtered_il_features.csv")
VARIANCE_REPORT_CSV  = os.path.join("results", "feature_variance_report.csv")
VARIANCE_HIST_PNG    = os.path.join("figures", "feature_variance_histogram.png")
RESULTS_DIR          = "results"
FIGURES_DIR          = "figures"

# Variance threshold for Morgan FP bits (computed at IL level).
# Bits with variance < this are dropped.
# 0.01 = bits set in <2 or >209 of 211 training ILs.
VARIANCE_THRESHOLD = 0.01

# RDKit descriptor columns are never dropped -- they're always kept regardless
# of variance, since they encode physically meaningful quantities (MW, logP, etc.)
RDKIT_DESC_SUFFIXES = [
    "mol_weight", "num_hbd", "num_hba", "tpsa",
    "num_rotatable_bonds", "num_aromatic_rings", "num_heavy_atoms", "log_p"
]


def is_fp_bit(col_name: str) -> bool:
    """Return True if this column is a Morgan fingerprint bit (cat_fp_N or an_fp_N)."""
    return ("_fp_" in col_name and
            (col_name.startswith("cat_") or col_name.startswith("an_")))


def load_il_level_features(train_csv: str, il_features_csv: str) -> tuple:
    """
    Load the full feature matrix at the IL level (one row per unique IL).

    We use train_set.csv to get the list of training ILs, then pull their
    features from il_features.csv. This gives us a 211-row matrix where
    each row is a unique IL -- the right granularity for computing variance.

    Returns (il_features_df, fp_cols, desc_cols, meta_cols).
    """
    train_df = pd.read_csv(train_csv)
    training_smiles = set(train_df["il_smiles"].unique())
    print(f"[load] Training set: {len(training_smiles)} unique ILs", flush=True)

    il_feat_df = pd.read_csv(il_features_csv)
    print(f"[load] il_features.csv: {len(il_feat_df)} ILs x {il_feat_df.shape[1]} cols",
          flush=True)

    # Filter to training ILs only for variance computation
    train_feat_df = il_feat_df[il_feat_df["il_smiles"].isin(training_smiles)].copy()
    print(f"[load] Training IL features: {len(train_feat_df)} rows (should be ~211)",
          flush=True)

    # Separate feature types
    all_cols  = il_feat_df.columns.tolist()
    fp_cols   = [c for c in all_cols if is_fp_bit(c)]
    desc_cols = [c for c in all_cols if
                 any(c.endswith(s) for s in RDKIT_DESC_SUFFIXES)]
    meta_cols = [c for c in all_cols if c not in fp_cols and c not in desc_cols]

    print(f"[load] FP bits: {len(fp_cols)} | RDKit descriptors: {len(desc_cols)} | "
          f"Meta cols: {len(meta_cols)}", flush=True)
    return il_feat_df, train_feat_df, fp_cols, desc_cols, meta_cols


def compute_fp_variance(train_feat_df: pd.DataFrame,
                        fp_cols: list) -> pd.DataFrame:
    """
    Compute variance of each Morgan FP bit across the 211 training ILs.

    Each bit is binary (0/1), so variance = p*(1-p) where p = fraction of
    ILs with that bit set. A bit set in exactly half the ILs has max variance
    0.25. A bit set in 1/211 ILs has variance ~0.005 -- below our threshold.

    Returns DataFrame with columns: feature, variance, p_set, ion, kept.
    """
    fp_matrix = train_feat_df[fp_cols].values.astype(float)  # shape (211, 4096)
    variances  = fp_matrix.var(axis=0)   # IL-level variance per bit
    p_set      = fp_matrix.mean(axis=0)  # fraction of ILs with bit set

    variance_df = pd.DataFrame({
        "feature":  fp_cols,
        "variance": variances,
        "p_set":    p_set,            # fraction of training ILs where bit=1
        "n_set":    (fp_matrix > 0).sum(axis=0),  # count of ILs with bit set
        "ion":      ["cation" if c.startswith("cat_") else "anion" for c in fp_cols],
        "kept":     variances >= VARIANCE_THRESHOLD,
    })
    return variance_df.sort_values("variance", ascending=False).reset_index(drop=True)


def print_threshold_sensitivity(variance_df: pd.DataFrame) -> None:
    """
    Print how many bits survive at different variance thresholds.
    Helps tune VARIANCE_THRESHOLD without re-running the script.
    """
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
    total      = len(variance_df)
    print("\n[sensitivity] Bits surviving at different variance thresholds:", flush=True)
    print(f"  {'Threshold':>12} {'Kept':>8} {'Dropped':>10} {'% kept':>10}", flush=True)
    for thresh in thresholds:
        kept = (variance_df["variance"] >= thresh).sum()
        print(f"  {thresh:>12.3f} {kept:>8d} {total-kept:>10d} {100*kept/total:>9.1f}%",
              flush=True)


def plot_variance_histogram(variance_df: pd.DataFrame) -> None:
    """
    Plot distribution of FP bit variances with the threshold marked.
    Saves to figures/feature_variance_histogram.png.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))

    # Separate cation and anion bits for color coding
    cat_var = variance_df[variance_df["ion"] == "cation"]["variance"]
    an_var  = variance_df[variance_df["ion"] == "anion"]["variance"]

    ax.hist(cat_var, bins=80, alpha=0.6, color="#2196F3", label="Cation FP bits")
    ax.hist(an_var,  bins=80, alpha=0.6, color="#FF5722", label="Anion FP bits")
    ax.axvline(VARIANCE_THRESHOLD, color="black", linestyle="--", linewidth=1.5,
               label=f"Threshold = {VARIANCE_THRESHOLD}")

    n_dropped = (variance_df["variance"] < VARIANCE_THRESHOLD).sum()
    n_kept    = (variance_df["variance"] >= VARIANCE_THRESHOLD).sum()
    ax.set_xlabel("IL-level variance of FP bit", fontsize=12)
    ax.set_ylabel("Number of bits", fontsize=12)
    ax.set_title(
        f"Morgan FP Bit Variance Distribution\n"
        f"Kept: {n_kept}  |  Dropped: {n_dropped}  |  Threshold: {VARIANCE_THRESHOLD}",
        fontsize=11
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(VARIANCE_HIST_PNG, dpi=150)
    plt.close()
    print(f"[plot] Saved -> {VARIANCE_HIST_PNG}", flush=True)


def filter_features(il_feat_df: pd.DataFrame, variance_df: pd.DataFrame,
                    desc_cols: list, meta_cols: list) -> pd.DataFrame:
    """
    Build the filtered feature DataFrame by keeping only:
      - Meta columns (il_smiles, il_name, cation_smiles, anion_smiles)
      - FP bits with variance >= VARIANCE_THRESHOLD
      - All RDKit descriptor columns (never dropped)

    Returns filtered DataFrame with same row count as il_feat_df.
    """
    kept_fp_cols = variance_df[variance_df["kept"]]["feature"].tolist()
    output_cols  = meta_cols + kept_fp_cols + desc_cols

    # Confirm all output cols exist in il_feat_df
    missing = [c for c in output_cols if c not in il_feat_df.columns]
    if missing:
        raise ValueError(f"Columns in filter list missing from features CSV: {missing[:5]}")

    return il_feat_df[output_cols].copy()


def main():
    """
    Full pipeline:
    1. Load IL-level feature matrix
    2. Compute per-bit variance on training ILs only (no test leakage)
    3. Print threshold sensitivity
    4. Drop low-variance bits
    5. Save filtered features + variance report + histogram
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=== FEATURE REDUCTION: Variance-threshold filtering of Morgan FP bits ===\n",
          flush=True)

    # Step 1: Load
    il_feat_df, train_feat_df, fp_cols, desc_cols, meta_cols = load_il_level_features(
        TRAIN_CSV, IL_FEATURES_CSV
    )

    # Step 2: Compute IL-level variance for every FP bit
    variance_df = compute_fp_variance(train_feat_df, fp_cols)

    n_total   = len(fp_cols)
    n_kept    = variance_df["kept"].sum()
    n_dropped = n_total - n_kept

    print(f"\n[main] Variance threshold: {VARIANCE_THRESHOLD}", flush=True)
    print(f"[main] Total FP bits:  {n_total}", flush=True)
    print(f"[main] Bits kept:      {n_kept}  ({100*n_kept/n_total:.1f}%)", flush=True)
    print(f"[main] Bits dropped:   {n_dropped}  ({100*n_dropped/n_total:.1f}%)", flush=True)

    # Per-ion breakdown
    for ion in ["cation", "anion"]:
        ion_df    = variance_df[variance_df["ion"] == ion]
        ion_kept  = ion_df["kept"].sum()
        ion_total = len(ion_df)
        print(f"  {ion.capitalize()}: {ion_kept}/{ion_total} bits kept", flush=True)

    # Step 3: Threshold sensitivity table
    print_threshold_sensitivity(variance_df)

    # Step 4: Build and save filtered feature matrix
    filtered_df = filter_features(il_feat_df, variance_df, desc_cols, meta_cols)
    total_features = len([c for c in filtered_df.columns
                          if c not in meta_cols])
    print(f"\n[main] Filtered feature matrix: {filtered_df.shape[0]} ILs x "
          f"{total_features} features (was {n_total + len(desc_cols)})", flush=True)

    filtered_df.to_csv(FILTERED_FEATURES_CSV, index=False)
    variance_df.to_csv(VARIANCE_REPORT_CSV,   index=False)
    plot_variance_histogram(variance_df)

    print(f"\n[main] Saved:", flush=True)
    print(f"  {FILTERED_FEATURES_CSV}", flush=True)
    print(f"  {VARIANCE_REPORT_CSV}", flush=True)
    print(f"  {VARIANCE_HIST_PNG}", flush=True)
    print(f"\n[main] NEXT STEPS:", flush=True)
    print(f"  1. Update build_dataset.py: change FEATURES_CSV to filtered_il_features.csv",
          flush=True)
    print(f"  2. python src/build_dataset.py", flush=True)
    print(f"  3. python src/train_stacked_model.py  (or train_model.py)", flush=True)
    print(f"  4. Compare new R² to baseline (0.714) -- expect small improvement",
          flush=True)


if __name__ == "__main__":
    main()
