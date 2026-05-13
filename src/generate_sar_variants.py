"""
generate_sar_variants.py
------------------------
PURPOSE: Generate structure-activity relationship (SAR) variants of the top candidate
         cation from Phase 4 inverse design and screen them with the forward model.

WHAT IS SAR ANALYSIS (for judges):
  Structure-Activity Relationship (SAR) analysis asks: "If we change one part of
  the best molecule we found, does performance go up or down?"
  This is the standard way chemists probe WHY a molecule works:
    - We identified [AETMA]+  (2-aminoethyltrimethylammonium) as the best cation.
    - Here we systematically vary it: change the functional group (NH2→OH/CN/COOH/F/CF3),
      change the chain length (ethyl→propyl→butyl), and add a second functional group.
    - We cross these cation variants with the top anions from Phase 4.
  If the NH2 group consistently outperforms OH/CN/COOH, that tells us specifically
  why the top candidate works — chemical insight, not just a ranking.

SCIENTIFIC RATIONALE:
  The NH2 (amine) group is known to react chemically with CO2 to form carbamate:
    R-NH2 + CO2 → R-NH-COO- + H+
  This is a chemical absorption mechanism (vs physical) and may explain high predicted x2.
  SAR variants test whether the NH2 is truly responsible, or whether it is the chain
  length, steric bulk, or overall polarity that drives absorption.

VARIATIONS GENERATED:
  1. Functional group substitution: swap NH2 for OH, CN, COOH, F, CF3, SH, OMe
  2. Chain length: 2-carbon (ethyl) vs 3-carbon (propyl) vs 4-carbon (butyl) linker
  3. N-alkylation: trimethyl vs triethyl vs tri-propyl ammonium core
  4. Double functionalization: NH2 + OH on same chain, NH2 + F on same chain
  5. Crossed with top 5 anions from Phase 4 predictions

INPUTS:
  models/forward_model.pkl                  (model bundle)
  results/virtual_library_predictions.csv  (Phase 4 predictions, to extract top anions)

OUTPUTS:
  data/virtual_library/sar_variants.csv          (all SAR candidates)
  results/sar_variant_predictions.csv             (screened + ranked)
  figures/sar_heatmap.png                         (functional group × anion heatmap)

Run from project root:
    python src/generate_sar_variants.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import itertools
from rdkit import Chem

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH        = os.path.join("models", "forward_model.pkl")
PHASE4_PREDS_CSV  = os.path.join("results", "virtual_library_predictions.csv")
SAR_CSV_OUT       = os.path.join("data", "virtual_library", "sar_variants.csv")
SAR_PREDS_OUT     = os.path.join("results", "sar_variant_predictions.csv")
HEATMAP_OUT       = os.path.join("figures", "sar_heatmap.png")

SCREEN_T_K     = 298.15   # K — ambient, same as all other screens
SCREEN_P_KPA   = 101.325  # kPa
TOP_N_ANIONS   = 5        # use top N anions from Phase 4 predictions
X2_MIN         = 1e-7     # physical plausibility filter
X2_MAX         = 1.0
FIG_DPI        = 300      # publication quality


# ── SAR Cation Variants ────────────────────────────────────────────────────────
#
# BASELINE: NCC[N+](C)(C)C  = 2-aminoethyl-trimethylammonium ([AETMA]+)
#   Structure: NH2-CH2-CH2-N+(CH3)3
#   The NH2 is 2 carbons from N+, trimethyl ammonium core.
#
# Each entry is (SMILES, short_name, description)
# Names will appear on the heatmap y-axis — keep them short.

SAR_CATION_VARIANTS = [
    # ── Group 1: Functional group substitution (ethyl linker, TMA core) ──
    # These isolate the effect of the terminal group on CO2 capture.
    ("NCC[N+](C)(C)C",        "NH2-Et-TMA",  "baseline: amino-ethyl (Phase 4 top candidate)"),
    ("OCC[N+](C)(C)C",        "OH-Et-TMA",   "hydroxyl: classic choline — does OH match NH2?"),
    ("N#CCC[N+](C)(C)C",      "CN-Et-TMA",   "nitrile: electron-withdrawing, no H-bond donor"),
    ("OC(=O)CC[N+](C)(C)C",   "COOH-Et-TMA", "carboxylate: acidic, can react with CO2 differently"),
    ("FCC[N+](C)(C)C",        "F-Et-TMA",    "fluorine: electronegative, no H-bonding"),
    ("FC(F)(F)CC[N+](C)(C)C", "CF3-Et-TMA",  "trifluoromethyl: strong electron withdrawer"),
    ("SCC[N+](C)(C)C",        "SH-Et-TMA",   "thiol: softer donor than NH2 or OH"),
    ("COCC[N+](C)(C)C",       "OMe-Et-TMA",  "methoxy ether: oxygen donor, no free H"),

    # ── Group 2: Chain length variation (NH2 group, TMA core) ──
    # Longer chain = more flexible, potentially better CO2 access to NH2.
    ("NC[N+](C)(C)C",         "NH2-Me-TMA",  "methyl linker: NH2 directly adjacent to N+"),
    ("NCCC[N+](C)(C)C",       "NH2-Pr-TMA",  "propyl linker: one carbon longer than baseline"),
    ("NCCCC[N+](C)(C)C",      "NH2-Bu-TMA",  "butyl linker: two carbons longer than baseline"),

    # ── Group 3: Ammonium core variation (NH2-ethyl, vary N-alkyl groups) ──
    # Tests whether the TMA (trimethyl) core is optimal vs larger N-alkyls.
    ("NCC[N+](CC)(CC)CC",     "NH2-Et-TEA",  "triethyl core: bulkier, may reduce solubility"),
    ("NCC[N+](C)(C)CC",       "NH2-Et-DMEA", "dimethylethyl: asymmetric ammonium"),
    ("NCC[N+]1(C)CCCC1",      "NH2-Et-Pyrr", "pyrrolidinium ring + NH2 ethyl arm"),

    # ── Group 4: Double functionalization (two groups on same chain) ──
    # Most novel — tests cooperative effects between two functional groups.
    ("NCC(O)[N+](C)(C)C",     "NH2OH-TMA",   "amino + hydroxyl on same carbon (beta-amino-alcohol)"),
    ("NCC(F)[N+](C)(C)C",     "NH2F-TMA",    "amino + fluorine: competing inductive effects"),
    ("N(C)CC[N+](C)(C)C",     "NHMe-Et-TMA", "secondary amine: less reactive with CO2 than NH2"),
    ("N(CC)CC[N+](C)(C)C",    "NEt-Et-TMA",  "tertiary amine: cannot form carbamate with CO2"),

    # ── Group 5: Pyridine / imidazole ring on arm ──
    # Aromatic amine behaves very differently — tests aromaticity effect.
    ("c1ccncc1CC[N+](C)(C)C", "4Py-Et-TMA",  "4-pyridyl-ethyl: aromatic ring with N lone pair"),
    ("c1cncc1CC[N+](C)(C)C",  "3Py-Et-TMA",  "3-pyridyl-ethyl: different N position on ring"),
]


def validate_sar_cations(cation_variants: list) -> list:
    """
    Validate each SAR cation SMILES with RDKit. Drop any that fail.
    Returns list of valid (smiles, name, description) tuples.
    """
    valid = []
    for smi, name, desc in cation_variants:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"  [WARN] Invalid SMILES for {name} — dropped: {smi}")
        else:
            valid.append((smi, name, desc))
    print(f"[validate_sar_cations] {len(valid)}/{len(cation_variants)} cation variants valid")
    return valid


def get_top_anions_from_phase4(preds_csv: str, n_top: int) -> list:
    """
    Extract the N most common anions among the top-20 Phase 4 predictions.
    These are the anions the model already favors — crossing SAR cations with
    the same anions isolates the cation-level SAR signal cleanly.

    If Phase 4 results aren't available, falls back to 5 literature benchmark anions.
    Returns a list of anion SMILES strings.
    """
    # Fallback anions if Phase 4 results aren't available
    FALLBACK_ANIONS = [
        "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F",  # [Tf2N]-
        "O=S(=O)([N-]S(=O)(=O)F)F",                  # [FSI]-
        "F[P-](F)(F)(F)(F)F",                          # [PF6]-
        "F[B-](F)(F)F",                                # [BF4]-
        "N(=O)[N-]C#N",                               # [DCA]-
    ]

    if not os.path.exists(preds_csv):
        print(f"[get_top_anions] {preds_csv} not found — using fallback anions")
        return FALLBACK_ANIONS

    preds_df = pd.read_csv(preds_csv)
    # Take top 20 predictions, extract anion SMILES, find most common
    top20 = preds_df.sort_values("x2_predicted", ascending=False).head(20)
    top_anions = top20["anion_smiles"].value_counts().head(n_top).index.tolist()

    print(f"[get_top_anions] Top {n_top} anions from Phase 4:")
    for ani in top_anions:
        print(f"  {ani}")
    return top_anions


def build_sar_library(valid_cation_variants: list, top_anions: list) -> pd.DataFrame:
    """
    Cross every valid SAR cation variant with every top anion.
    Returns a DataFrame with il_smiles, cation_smiles, cation_name,
    cation_description, anion_smiles, T_K, P_kPa.
    """
    rows = []
    for (cat_smi, cat_name, cat_desc), ani_smi in itertools.product(
            valid_cation_variants, top_anions):

        il_smiles = f"{cat_smi}.{ani_smi}"  # RDKit dot notation
        rows.append({
            "il_smiles":           il_smiles,
            "cation_smiles":       cat_smi,
            "cation_name":         cat_name,
            "cation_description":  cat_desc,
            "anion_smiles":        ani_smi,
            "T_K":                 SCREEN_T_K,
            "P_kPa":               SCREEN_P_KPA,
        })

    print(f"[build_sar_library] {len(rows)} SAR IL candidates "
          f"({len(valid_cation_variants)} cations × {len(top_anions)} anions)")
    return pd.DataFrame(rows)


def featurize_and_predict(sar_df: pd.DataFrame, model_path: str) -> pd.DataFrame:
    """
    Featurize all SAR ILs and predict CO2 mole fraction solubility.
    Uses the same bundle-loading and column-alignment approach as inverse_design.py.
    Returns sar_df with added log_x2_predicted and x2_predicted columns.
    """
    # Import featurize from src/ (same directory as this script)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from featurize import featurize_il_smiles

    # Load model bundle
    bundle       = joblib.load(model_path)
    model        = bundle["model"]
    feature_cols = bundle["feature_cols"]
    print(f"[featurize_and_predict] Model loaded: {bundle.get('model_name', 'unknown')}, "
          f"{len(feature_cols)} features expected")

    # Featurize each SAR IL
    feature_rows = []
    valid_idx    = []
    for idx, row in sar_df.iterrows():
        features = featurize_il_smiles(row["il_smiles"])
        if features is None:
            print(f"  [WARN] Featurization failed for {row['cation_name']} — dropped")
            continue
        features["T_K"]   = row["T_K"]
        features["P_kPa"] = row["P_kPa"]
        feature_rows.append(features)
        valid_idx.append(idx)

    feature_df       = pd.DataFrame(feature_rows)
    valid_sar_df     = sar_df.loc[valid_idx].reset_index(drop=True)

    # Align feature columns to training set (fill missing with 0, drop extra)
    missing_cols = [c for c in feature_cols if c not in feature_df.columns]
    for col in missing_cols:
        feature_df[col] = 0  # absent bit = structural feature not present
    X_aligned = feature_df[feature_cols].values  # reorder to match training order

    # Predict
    log_x2  = model.predict(X_aligned)
    x2_pred = 10 ** log_x2  # back-transform from log10 scale

    valid_sar_df["log_x2_predicted"] = log_x2
    valid_sar_df["x2_predicted"]     = x2_pred

    # Physical plausibility filter
    physical_mask = (valid_sar_df["x2_predicted"] >= X2_MIN) & \
                    (valid_sar_df["x2_predicted"] <= X2_MAX)
    n_removed = (~physical_mask).sum()
    if n_removed > 0:
        print(f"[featurize_and_predict] Removed {n_removed} physically implausible predictions")
    valid_sar_df = valid_sar_df[physical_mask].reset_index(drop=True)

    print(f"[featurize_and_predict] {len(valid_sar_df)} SAR candidates predicted")
    print(f"  x2 range: [{valid_sar_df['x2_predicted'].min():.2e}, "
          f"{valid_sar_df['x2_predicted'].max():.2e}]")
    return valid_sar_df


def plot_sar_heatmap(sar_preds_df: pd.DataFrame, output_path: str):
    """
    Plot a heatmap of predicted x2 with cation variants on Y axis and anions on X axis.

    WHY A HEATMAP:
      A heatmap makes the SAR result immediately visual:
        - Rows = cation variants (what we changed)
        - Columns = anion partners
        - Color = predicted CO2 mole fraction (brighter = better)
      Judges can instantly see which functional group and which anion pairing
      gives the best CO2 absorption — no need to read a table.

    Truncates long anion SMILES to first 20 chars for readability.
    """
    if sar_preds_df.empty:
        print("[plot_sar_heatmap] No data to plot — skipping")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Pivot: rows = cation_name, cols = anion_smiles (truncated), values = x2_predicted
    sar_preds_df["anion_short"] = sar_preds_df["anion_smiles"].str[:20]
    pivot_df = sar_preds_df.pivot_table(
        index="cation_name",
        columns="anion_short",
        values="x2_predicted",
        aggfunc="mean"     # in case any duplicates exist
    )

    # Sort rows by best mean x2 across all anions (best cation on top)
    pivot_df = pivot_df.loc[pivot_df.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(max(8, len(pivot_df.columns) * 1.5),
                                     max(6, len(pivot_df) * 0.5)))

    # Use log scale for color — x2 values span orders of magnitude
    log_pivot = np.log10(pivot_df.clip(lower=1e-10))  # clip to avoid log(0)
    im = ax.imshow(log_pivot.values, cmap="YlOrRd", aspect="auto")

    # Axis labels
    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels(pivot_df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels(pivot_df.index, fontsize=9)

    # Annotate each cell with the x2 value
    for row_i in range(len(pivot_df.index)):
        for col_j in range(len(pivot_df.columns)):
            val = pivot_df.values[row_i, col_j]
            if not np.isnan(val):
                ax.text(col_j, row_i, f"{val:.3f}",
                        ha="center", va="center", fontsize=7, color="black")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("log₁₀(x₂ predicted)", fontsize=10)

    ax.set_title(
        "SAR Analysis: Predicted CO₂ Mole Fraction by Cation Variant × Anion\n"
        "(higher = more CO₂ absorbed; baseline is NH2-Et-TMA from Phase 4)",
        fontsize=11, pad=12
    )
    ax.set_xlabel("Anion (truncated SMILES)", fontsize=10)
    ax.set_ylabel("Cation variant", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[plot_sar_heatmap] SAR heatmap saved → {output_path}")


def print_sar_summary(sar_preds_df: pd.DataFrame):
    """
    Print a human-readable summary of the SAR results:
      - Best cation variant overall
      - Whether baseline (NH2) is still top after SAR
      - Best anion across all variants
    This directly tells us whether the NH2 group is mechanistically responsible.
    """
    if sar_preds_df.empty:
        print("[sar_summary] No predictions to summarize")
        return

    print("\n=== SAR SUMMARY ===")

    # Best overall IL
    best_row = sar_preds_df.sort_values("x2_predicted", ascending=False).iloc[0]
    print(f"Best SAR candidate:")
    print(f"  Cation: {best_row['cation_name']} — {best_row['cation_description']}")
    print(f"  Anion:  {best_row['anion_smiles']}")
    print(f"  x2 predicted: {best_row['x2_predicted']:.4f}")

    # Average x2 per cation variant (across all anions)
    cation_avg = sar_preds_df.groupby(["cation_name", "cation_description"])["x2_predicted"]\
                              .mean().sort_values(ascending=False)
    print("\nAverage predicted x2 by cation variant (best first):")
    for (name, desc), avg_x2 in cation_avg.items():
        baseline_marker = " ← BASELINE" if name == "NH2-Et-TMA" else ""
        print(f"  {name:20s}  x2={avg_x2:.4f}  | {desc}{baseline_marker}")

    # Is NH2 still on top?
    best_cation = cation_avg.index[0][0]
    if best_cation == "NH2-Et-TMA":
        print("\n→ CONCLUSION: NH2 functional group remains the best performer.")
        print("  This supports chemical CO2 absorption (carbamate formation) as the mechanism.")
    else:
        print(f"\n→ CONCLUSION: {best_cation} outperforms NH2-Et-TMA baseline.")
        print("  This suggests physical absorption (not carbamate) may dominate.")
        print("  Flag as limitation: DFT needed to confirm mechanism.")


def main():
    """Full SAR pipeline: define variants -> get top anions -> screen -> plot -> summarize."""
    os.makedirs(os.path.dirname(SAR_CSV_OUT), exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # ── Step 1: Validate cation variants ────────────────────────────────────
    print("=== STEP 1: Validating SAR cation variants ===")
    valid_cations = validate_sar_cations(SAR_CATION_VARIANTS)

    # ── Step 2: Get top anions from Phase 4 ─────────────────────────────────
    print("\n=== STEP 2: Extracting top anions from Phase 4 predictions ===")
    top_anions = get_top_anions_from_phase4(PHASE4_PREDS_CSV, TOP_N_ANIONS)

    # ── Step 3: Build SAR library ────────────────────────────────────────────
    print("\n=== STEP 3: Building SAR variant library ===")
    sar_df = build_sar_library(valid_cations, top_anions)
    sar_df.to_csv(SAR_CSV_OUT, index=False)
    print(f"[main] SAR library saved → {SAR_CSV_OUT}")

    # ── Step 4: Featurize and predict ────────────────────────────────────────
    print("\n=== STEP 4: Featurizing and predicting CO2 solubility ===")
    sar_preds_df = featurize_and_predict(sar_df, MODEL_PATH)
    sar_preds_df = sar_preds_df.sort_values("x2_predicted", ascending=False)\
                                .reset_index(drop=True)
    sar_preds_df["rank"] = sar_preds_df.index + 1
    sar_preds_df.to_csv(SAR_PREDS_OUT, index=False)
    print(f"[main] SAR predictions saved → {SAR_PREDS_OUT}")

    # ── Step 5: Plot heatmap ─────────────────────────────────────────────────
    print("\n=== STEP 5: Plotting SAR heatmap ===")
    plot_sar_heatmap(sar_preds_df, HEATMAP_OUT)

    # ── Step 6: Print summary ────────────────────────────────────────────────
    print_sar_summary(sar_preds_df)

    print("\n✓ SAR analysis complete.")
    print(f"  Results: {SAR_PREDS_OUT}")
    print(f"  Heatmap: {HEATMAP_OUT}")


if __name__ == "__main__":
    main()
