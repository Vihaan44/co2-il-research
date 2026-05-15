"""
applicability_domain.py
------------------------
PURPOSE: Check whether virtual IL candidates are within the model's
         'applicability domain' -- the structural space the model was trained on.

WHY THIS MATTERS:
  Our model was trained on ~200 ILs from ILThermo. When we screen 1,226 novel
  ILs, some will be structurally similar to training ILs (the model's predictions
  are reliable), and some will be structurally exotic (the model is extrapolating).
  Without this check, we can't tell which predictions to trust.

  This is how real cheminformatics models are deployed in drug discovery and
  materials science -- predictions outside the applicability domain are explicitly
  flagged as unreliable.

METHOD -- Tanimoto Similarity to Nearest Training IL:
  For each virtual IL, compute the Tanimoto similarity between its Morgan
  fingerprint and every training IL's Morgan fingerprint. Find the maximum
  (nearest neighbor). If max similarity < SIMILARITY_THRESHOLD, flag it as
  'outside applicability domain'.

  Tanimoto similarity = |A ∩ B| / |A ∪ B|   (bits in common / bits in either)
  Range: 0 (no shared bits) to 1 (identical fingerprints).
  Threshold of 0.30 is a standard cutoff in cheminformatics (Sheridan 2004).

  Limitation: Tanimoto on Morgan FPs captures structural topology but may
  miss property-space distance. Treat the AD check as indicative, not definitive.

OUTPUTS:
  results/applicability_domain.csv  -- all candidates with AD score and flag
  figures/applicability_domain.png  -- histogram of similarity scores + threshold

INPUTS:
  data/processed/train_set.csv             -- to get training IL SMILES
  results/virtual_library_predictions.csv  -- candidates to check

Run from project root:
    python src/applicability_domain.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

# -- Constants -----------------------------------------------------------------
TRAIN_CSV       = os.path.join("data", "processed", "train_set.csv")
CANDIDATES_CSV  = os.path.join("results", "virtual_library_predictions.csv")
OUTPUT_CSV      = os.path.join("results", "applicability_domain.csv")
FIGURES_DIR     = "figures"
AD_FIGURE_PATH  = os.path.join(FIGURES_DIR, "applicability_domain.png")

MORGAN_RADIUS  = 2      # must match featurize.py
MORGAN_NBITS   = 2048   # must match featurize.py

# Tanimoto similarity threshold for in-domain classification.
# Candidates with max_tanimoto < 0.30 are flagged as outside applicability domain.
# This cutoff comes from Sheridan et al. 2004 (J. Chem. Inf. Model.).
# Note: for very novel cation families, almost everything will be below this.
SIMILARITY_THRESHOLD = 0.30


def smiles_to_morgan_fp(smiles: str) -> DataStructs.ExplicitBitVect | None:
    """
    Convert a SMILES string to an RDKit Morgan fingerprint object.
    Returns None if parsing fails (invalid SMILES).
    We use ExplicitBitVect (not numpy) because RDKit's BulkTanimotoSimilarity
    requires this format for fast batch comparison.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # GetMorganFingerprintAsBitVect returns ExplicitBitVect (required for BulkTanimoto)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=MORGAN_RADIUS, nBits=MORGAN_NBITS)
    return fp


def get_training_fingerprints(train_df: pd.DataFrame) -> list:
    """
    Compute Morgan fingerprints for all unique training ILs.
    Returns a list of (il_smiles, ExplicitBitVect) tuples for valid ILs.
    """
    unique_smiles = train_df["il_smiles"].unique()
    print(f"[get_training_fingerprints] Computing FPs for {len(unique_smiles)} training ILs...")

    training_fps = []
    failed_count = 0
    for smiles in unique_smiles:
        fp = smiles_to_morgan_fp(str(smiles))
        if fp is not None:
            training_fps.append((smiles, fp))
        else:
            failed_count += 1

    print(f"[get_training_fingerprints] {len(training_fps)} valid FPs, {failed_count} failed")
    return training_fps


def compute_max_tanimoto(candidate_fp: DataStructs.ExplicitBitVect,
                         training_fps: list) -> float:
    """
    Compute the maximum Tanimoto similarity between one candidate's fingerprint
    and all training IL fingerprints.

    BulkTanimotoSimilarity computes all similarities at once (fast C++ implementation).
    The maximum = similarity to the nearest training IL.
    """
    train_fp_list = [fp for _, fp in training_fps]  # extract just the BitVect objects
    similarities  = DataStructs.BulkTanimotoSimilarity(candidate_fp, train_fp_list)
    return max(similarities) if similarities else 0.0


def check_applicability_domain(candidates_df: pd.DataFrame,
                                training_fps: list) -> pd.DataFrame:
    """
    For each candidate IL, compute its maximum Tanimoto similarity to training ILs
    and flag it as in-domain (True) or out-of-domain (False).

    Returns candidates_df with added columns:
      max_tanimoto     -- similarity to nearest training IL (0-1)
      in_domain        -- True if max_tanimoto >= SIMILARITY_THRESHOLD
      ad_note          -- plain-English note for each candidate
    """
    print(f"[check_applicability_domain] Checking {len(candidates_df)} candidates...")
    print(f"  Threshold: Tanimoto >= {SIMILARITY_THRESHOLD} = in-domain")

    max_tanimoto_scores = []
    in_domain_flags     = []

    for idx, row in candidates_df.iterrows():
        smiles = str(row["il_smiles"])
        fp     = smiles_to_morgan_fp(smiles)

        if fp is None:
            # Invalid SMILES -- definitely outside domain
            max_tanimoto_scores.append(0.0)
            in_domain_flags.append(False)
            continue

        max_sim = compute_max_tanimoto(fp, training_fps)
        max_tanimoto_scores.append(max_sim)
        in_domain_flags.append(max_sim >= SIMILARITY_THRESHOLD)

        if (idx + 1) % 200 == 0:
            print(f"  -> {idx + 1}/{len(candidates_df)} checked")

    result_df = candidates_df.copy()
    result_df["max_tanimoto"] = max_tanimoto_scores
    result_df["in_domain"]    = in_domain_flags
    result_df["ad_note"] = result_df["in_domain"].apply(
        lambda x: "In-domain: prediction likely reliable"
        if x else "Out-of-domain: prediction may be unreliable (novel structure)"
    )

    n_in  = sum(in_domain_flags)
    n_out = len(in_domain_flags) - n_in
    print(f"\n[check_applicability_domain] Results:")
    print(f"  In-domain  (Tanimoto >= {SIMILARITY_THRESHOLD}): {n_in} "
          f"({100*n_in/len(candidates_df):.1f}%)")
    print(f"  Out-of-domain                                  : {n_out} "
          f"({100*n_out/len(candidates_df):.1f}%)")
    print(f"\n  NOTE: Out-of-domain does not mean the IL is bad -- it means the")
    print(f"  model has less precedent for its prediction. These candidates may be")
    print(f"  the most novel (interesting!) or the least reliable (risky).")
    print(f"  Prioritize in-domain candidates for DFT validation, flag out-of-domain")
    print(f"  for future experimental work.")

    return result_df


def plot_tanimoto_distribution(result_df: pd.DataFrame):
    """
    Plot histogram of max Tanimoto similarity scores for all candidates.
    Shows the threshold line and shaded regions for in-domain vs out-of-domain.
    Saves to figures/applicability_domain.png.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    scores = result_df["max_tanimoto"].values

    # Plot histogram split by in/out domain
    in_domain_scores  = scores[scores >= SIMILARITY_THRESHOLD]
    out_domain_scores = scores[scores < SIMILARITY_THRESHOLD]

    bin_edges = np.linspace(0, 1, 26)  # 25 bins across 0-1

    ax.hist(out_domain_scores, bins=bin_edges, color="#e74c3c", alpha=0.7,
            label=f"Out-of-domain (Tanimoto < {SIMILARITY_THRESHOLD})")
    ax.hist(in_domain_scores,  bins=bin_edges, color="#2ecc71", alpha=0.7,
            label=f"In-domain (Tanimoto >= {SIMILARITY_THRESHOLD})")

    # Threshold line
    ax.axvline(SIMILARITY_THRESHOLD, color="black", linestyle="--", linewidth=1.5,
               label=f"Threshold = {SIMILARITY_THRESHOLD}")

    ax.set_xlabel("Max Tanimoto Similarity to Nearest Training IL", fontsize=12)
    ax.set_ylabel("Number of Candidates", fontsize=12)
    ax.set_title("Applicability Domain Check: Virtual IL Library", fontsize=13)
    ax.legend(fontsize=10)

    # Add text annotation showing percentages
    n_in  = (scores >= SIMILARITY_THRESHOLD).sum()
    n_out = (scores  < SIMILARITY_THRESHOLD).sum()
    ax.text(0.98, 0.95,
            f"In-domain: {n_in} ({100*n_in/len(scores):.1f}%)\n"
            f"Out-of-domain: {n_out} ({100*n_out/len(scores):.1f}%)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(AD_FIGURE_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Applicability domain plot saved -> {AD_FIGURE_PATH}")


def main():
    """Main pipeline: load training FPs -> check candidates -> plot -> save."""
    os.makedirs("results", exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # -- Step 1: Load training data ------------------------------------------
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(
            f"Train set not found at {TRAIN_CSV}. Run build_dataset.py first."
        )
    train_df = pd.read_csv(TRAIN_CSV)
    print(f"[main] Train set: {train_df['il_smiles'].nunique()} unique ILs")

    training_fps = get_training_fingerprints(train_df)

    # -- Step 2: Load candidates ---------------------------------------------
    if not os.path.exists(CANDIDATES_CSV):
        raise FileNotFoundError(
            f"Candidates CSV not found at {CANDIDATES_CSV}.\n"
            f"Run src/inverse_design.py or src/build_virtual_library_expanded.py first."
        )
    candidates_df = pd.read_csv(CANDIDATES_CSV)
    print(f"[main] {len(candidates_df)} candidates to check")

    # -- Step 3: Compute applicability domain --------------------------------
    result_df = check_applicability_domain(candidates_df, training_fps)

    # -- Step 4: Sort by predicted x2 (best candidates first) ---------------
    sort_col = "pred_x2" if "pred_x2" in result_df.columns else "pred_log_x2"
    if sort_col in result_df.columns:
        result_df = result_df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    # -- Step 5: Show top in-domain candidates -------------------------------
    in_domain_df = result_df[result_df["in_domain"]]
    print(f"\n[main] Top 10 IN-DOMAIN candidates:")
    show_cols = ["il_smiles", "max_tanimoto", sort_col] if sort_col in result_df.columns \
               else ["il_smiles", "max_tanimoto"]
    print(in_domain_df.head(10)[show_cols].to_string(index=False))

    # -- Step 6: Plot and save -----------------------------------------------
    plot_tanimoto_distribution(result_df)
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[main] Saved -> {OUTPUT_CSV}")
    print("[main] Columns: max_tanimoto, in_domain, ad_note")
    print("[main] Interpretation: in_domain=True => prediction is within training distribution")


if __name__ == "__main__":
    main()
