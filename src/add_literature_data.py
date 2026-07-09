"""
add_literature_data.py
-----------------------
PURPOSE: Merge manually-extracted literature CO2/IL solubility data
         (from Shiflett, Jacquemin, and other groups not in ILThermo)
         into the main datapoints CSV, then rebuild train/test splits.

WHY EXPERIMENTAL LITERATURE DATA (not computational):
  COSMO-RS predicted CO2/IL solubility data has 15-30% systematic error
  vs experiment, with anion-type-dependent bias. Mixing it with experimental
  data would teach the model a blend of two different physical quantities.
  This script ONLY accepts experimental (measured) data.

WHAT YOU NEED TO PROVIDE:
  Create data/raw/literature_co2_data.csv with the following columns:
    il_smiles   -- canonical RDKit SMILES (dot-separated cation.anion)
    il_name     -- human-readable name (e.g. [BMIM][Tf2N])
    T_K         -- temperature in Kelvin
    P_kPa       -- CO2 partial pressure in kPa
    x2_CO2      -- mole fraction CO2 solubility (NOT log-transformed)
    source      -- citation (e.g. Shiflett2005_JPCB)
    data_source -- always "literature" for this file

HOW TO GET THE DATA (step by step):

  1. SHIFLETT GROUP (most valuable -- ILs not in ILThermo)
     Search: https://scholar.google.com/scholar?q=Shiflett+CO2+ionic+liquid+solubility
     Key papers:
       - Shiflett & Yokozeki (2005) J. Phys. Chem. B 109, 19597
       - Shiflett & Yokozeki (2008) Energy Fuels 22, 2585
       - Shiflett et al. (2010) ChemSusChem 3, 1086
     For each paper: find the (T, P, x2) data table. Use Tabula
     (https://tabula.technology/) to extract PDF tables automatically.

  2. JACQUEMIN GROUP
     Search: https://scholar.google.com/scholar?q=Jacquemin+CO2+ionic+liquid+mole+fraction
     Key papers 2006-2012. Same extraction process.

  3. CHECK ILTHERMO FOR RECENT ADDITIONS
     https://ilthermo.boulder.nist.gov/
     Search: Component 1 = CO2, Property = Mole fraction, Phase = Liquid
     Download all results -- ILThermo is updated regularly.

  4. CANONICALIZE SMILES for each new IL:
     python3 -c "
     from rdkit import Chem
     smi = 'your_smiles_here'
     print(Chem.MolToSmiles(Chem.MolFromSmiles(smi)))
     "
     Or use: https://cactus.nci.nih.gov/translate/

  5. CHECK FOR DUPLICATES before adding:
     Run this script with --check-only flag to see which new ILs
     are genuinely new vs already in the existing dataset.

ONCE YOU HAVE literature_co2_data.csv:
  python src/add_literature_data.py
  python src/featurize.py         # featurize new ILs
  python src/build_dataset.py     # rebuild train/test with new ILs
  nohup python src/nested_cv_model_comparison.py > logs/nested_cv_v2.log 2>&1 &

OUTPUTS:
  data/raw/all_co2_datapoints_v2.csv  -- merged datapoints
  results/literature_merge_summary.csv -- what was added, what was duplicate
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from rdkit import Chem

# -- Constants -----------------------------------------------------------------
EXISTING_CSV    = os.path.join("data", "raw", "all_co2_datapoints_merged.csv")
LITERATURE_CSV  = os.path.join("data", "raw", "literature_co2_data.csv")
OUTPUT_CSV      = os.path.join("data", "raw", "all_co2_datapoints_v2.csv")
SUMMARY_CSV     = os.path.join("results", "literature_merge_summary.csv")

REQUIRED_COLS = ["il_smiles", "il_name", "T_K", "P_kPa", "x2_CO2", "source"]

# Duplicate detection tolerance:
# Two rows are duplicates if same IL + T within 0.5K + P within 0.5kPa
TEMP_TOL_K   = 0.5
PRESS_TOL_KPA = 0.5
X2_TOL        = 0.005   # mole fraction tolerance for reporting near-duplicates


def canonicalize_smiles(smiles: str) -> str:
    """
    Convert SMILES to RDKit canonical form for consistent comparison.
    Returns empty string if RDKit cannot parse the SMILES.
    This is critical: the same IL can have many valid SMILES representations;
    canonicalization ensures we detect duplicates reliably.
    """
    if not smiles or not isinstance(smiles, str):
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"  [DATA QUALITY] Cannot parse SMILES: {smiles[:60]}", flush=True)
        return ""
    return Chem.MolToSmiles(mol)


def validate_literature_csv(lit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Check the literature CSV for required columns, valid SMILES,
    and sensible physical values. Returns cleaned DataFrame.
    """
    print("[validate] Checking literature CSV...", flush=True)

    # Check required columns
    missing_cols = [c for c in REQUIRED_COLS if c not in lit_df.columns]
    if missing_cols:
        raise ValueError(
            f"literature_co2_data.csv is missing required columns: {missing_cols}\n"
            f"See script docstring for the required column list."
        )

    original_len = len(lit_df)

    # Canonicalize SMILES
    print("  Canonicalizing SMILES...", flush=True)
    lit_df["il_smiles"] = lit_df["il_smiles"].apply(canonicalize_smiles)
    bad_smiles = lit_df["il_smiles"] == ""
    if bad_smiles.any():
        print(f"  [DATA QUALITY] {bad_smiles.sum()} rows have unparseable SMILES -- dropping",
              flush=True)
        lit_df = lit_df[~bad_smiles].copy()

    # Physical value sanity checks
    bad_temp  = (lit_df["T_K"] < 200) | (lit_df["T_K"] > 500)
    bad_press = (lit_df["P_kPa"] < 0) | (lit_df["P_kPa"] > 100000)
    bad_x2    = (lit_df["x2_CO2"] <= 0) | (lit_df["x2_CO2"] > 1)

    if bad_temp.any():
        print(f"  [DATA QUALITY] {bad_temp.sum()} rows with T_K outside 200-500K -- dropping",
              flush=True)
        lit_df = lit_df[~bad_temp].copy()
    if bad_press.any():
        print(f"  [DATA QUALITY] {bad_press.sum()} rows with P_kPa outside 0-100000 -- dropping",
              flush=True)
        lit_df = lit_df[~bad_press].copy()
    if bad_x2.any():
        print(f"  [DATA QUALITY] {bad_x2.sum()} rows with x2_CO2 outside (0,1] -- dropping",
              flush=True)
        lit_df = lit_df[~bad_x2].copy()

    lit_df["data_source"] = "literature"
    print(f"  Validated: {len(lit_df)}/{original_len} rows pass all checks", flush=True)
    return lit_df


def find_duplicates(existing_df: pd.DataFrame,
                    lit_df: pd.DataFrame) -> tuple:
    """
    Identify rows in lit_df that are already in existing_df (same IL + T + P).
    Returns (new_rows_df, duplicate_rows_df, near_duplicate_df).

    We match on:
      - Exact canonical SMILES match (same IL identity)
      - T_K within TEMP_TOL_K (0.5K)
      - P_kPa within PRESS_TOL_KPA (0.5 kPa)

    Near-duplicates (same T/P but different x2 by more than X2_TOL)
    are flagged separately -- they suggest a measurement conflict and
    should be reviewed before adding.
    """
    print("\n[duplicates] Checking for duplicates vs existing dataset...", flush=True)

    new_rows   = []
    duplicates = []
    near_dups  = []

    for _, lit_row in lit_df.iterrows():
        smi = lit_row["il_smiles"]
        # Find existing rows with the same IL
        same_il = existing_df[existing_df["il_smiles"] == smi]
        if same_il.empty:
            new_rows.append(lit_row)
            continue

        # Check T, P proximity
        same_cond = same_il[
            (abs(same_il["T_K"]    - lit_row["T_K"])    < TEMP_TOL_K) &
            (abs(same_il["P_kPa"] - lit_row["P_kPa"]) < PRESS_TOL_KPA)
        ]

        if same_cond.empty:
            # Same IL, different T/P condition -- genuine new data point
            new_rows.append(lit_row)
        else:
            # Same IL + T + P -- check if x2 agrees
            x2_diff = abs(same_cond["x2_CO2"].values[0] - lit_row["x2_CO2"])
            if x2_diff > X2_TOL:
                lit_row_copy = lit_row.copy()
                lit_row_copy["existing_x2"]   = same_cond["x2_CO2"].values[0]
                lit_row_copy["existing_source"] = same_cond.get("data_source",
                                                   pd.Series(["unknown"])).values[0]
                lit_row_copy["x2_difference"]  = x2_diff
                near_dups.append(lit_row_copy)
            else:
                duplicates.append(lit_row)

    new_df     = pd.DataFrame(new_rows)
    dup_df     = pd.DataFrame(duplicates)
    near_df    = pd.DataFrame(near_dups)

    print(f"  Genuine new rows:      {len(new_df)}", flush=True)
    print(f"  Exact duplicates:      {len(dup_df)} (will be skipped)", flush=True)
    print(f"  Near-duplicates:       {len(near_df)} (x2 differs >0.005, review)",
          flush=True)

    if not near_df.empty:
        print("\n  Near-duplicate details (check these manually):", flush=True)
        for _, row in near_df.iterrows():
            print(f"    {row['il_name']} T={row['T_K']}K P={row['P_kPa']}kPa: "
                  f"new x2={row['x2_CO2']:.4f}, existing x2={row['existing_x2']:.4f}, "
                  f"diff={row['x2_difference']:.4f}", flush=True)
        print("  Near-duplicates will be EXCLUDED pending manual review.", flush=True)

    return new_df, dup_df, near_df


def report_new_ils(existing_df: pd.DataFrame, new_df: pd.DataFrame):
    """
    Print a summary of which ILs are genuinely new vs which are additional
    T/P conditions for ILs already in the dataset.
    """
    if new_df.empty:
        print("\n[summary] No new rows to add.", flush=True)
        return

    existing_smiles = set(existing_df["il_smiles"].unique())
    new_smiles      = set(new_df["il_smiles"].unique())
    truly_new_ils   = new_smiles - existing_smiles
    extra_cond_ils  = new_smiles & existing_smiles

    print(f"\n[summary] New data breakdown:", flush=True)
    print(f"  Truly new ILs (not in existing dataset): {len(truly_new_ils)}",
          flush=True)
    for smi in sorted(truly_new_ils):
        name = new_df[new_df["il_smiles"] == smi]["il_name"].values[0]
        n_rows = (new_df["il_smiles"] == smi).sum()
        print(f"    {name}: {n_rows} new (T,P) measurements", flush=True)

    print(f"  Existing ILs with new (T,P) conditions: {len(extra_cond_ils)}",
          flush=True)
    for smi in sorted(extra_cond_ils):
        name = new_df[new_df["il_smiles"] == smi]["il_name"].values[0]
        n_rows = (new_df["il_smiles"] == smi).sum()
        print(f"    {name}: {n_rows} additional (T,P) measurements", flush=True)

    print(f"\n  Expected impact: +{len(truly_new_ils)} ILs in training pool "
          f"(currently 299, target >310 for meaningful R2 improvement)",
          flush=True)


def main():
    """Load existing data, validate and merge literature CSV, save merged output."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true",
                        help="Report what would be added without writing output files")
    args = parser.parse_args()

    print("=" * 65, flush=True)
    print("add_literature_data.py", flush=True)
    print("=" * 65, flush=True)

    if not os.path.exists(EXISTING_CSV):
        raise FileNotFoundError(f"Existing datapoints not found: {EXISTING_CSV}")

    if not os.path.exists(LITERATURE_CSV):
        print(f"\n[main] literature_co2_data.csv not found at:", flush=True)
        print(f"  {LITERATURE_CSV}", flush=True)
        print("\nSee script docstring for step-by-step extraction instructions.",
              flush=True)
        print("Create the CSV with these columns:", flush=True)
        print(f"  {REQUIRED_COLS}", flush=True)
        sys.exit(0)

    print(f"\n[load] Existing data: {EXISTING_CSV}", flush=True)
    existing_df = pd.read_csv(EXISTING_CSV)
    # Canonicalize existing SMILES for consistent comparison
    existing_df["il_smiles"] = existing_df["il_smiles"].apply(canonicalize_smiles)
    existing_df = existing_df[existing_df["il_smiles"] != ""].copy()
    print(f"  {len(existing_df)} rows, {existing_df['il_smiles'].nunique()} unique ILs",
          flush=True)

    print(f"\n[load] Literature data: {LITERATURE_CSV}", flush=True)
    lit_df = pd.read_csv(LITERATURE_CSV)
    print(f"  {len(lit_df)} rows", flush=True)

    # Validate
    lit_df = validate_literature_csv(lit_df)

    # Find duplicates
    new_df, dup_df, near_df = find_duplicates(existing_df, lit_df)

    # Report
    report_new_ils(existing_df, new_df)

    if args.check_only:
        print("\n[main] --check-only mode: no files written.", flush=True)
        sys.exit(0)

    if new_df.empty:
        print("\n[main] Nothing new to add. Exiting.", flush=True)
        sys.exit(0)

    # Merge and save
    merged_df = pd.concat([existing_df, new_df], ignore_index=True)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    merged_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[main] Saved merged data -> {OUTPUT_CSV}", flush=True)
    print(f"  {len(merged_df)} total rows, {merged_df['il_smiles'].nunique()} unique ILs",
          flush=True)

    # Save merge summary
    summary_rows = [
        {"category": "existing_rows",      "count": len(existing_df)},
        {"category": "existing_ils",       "count": existing_df["il_smiles"].nunique()},
        {"category": "literature_rows_in", "count": len(lit_df)},
        {"category": "exact_duplicates",   "count": len(dup_df)},
        {"category": "near_duplicates",    "count": len(near_df)},
        {"category": "new_rows_added",     "count": len(new_df)},
        {"category": "new_ils_added",      "count": len(set(new_df["il_smiles"]) -
                                                          set(existing_df["il_smiles"]))},
        {"category": "total_rows_out",     "count": len(merged_df)},
        {"category": "total_ils_out",      "count": merged_df["il_smiles"].nunique()},
    ]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)
    print(f"[main] Saved merge summary -> {SUMMARY_CSV}", flush=True)

    print("\n[main] NEXT STEPS:", flush=True)
    print("  1. python src/featurize.py         # featurize new ILs", flush=True)
    print("  2. python src/build_dataset.py     # rebuild train/test with new ILs", flush=True)
    print("  3. nohup python src/nested_cv_model_comparison.py > "
          "logs/nested_cv_v2.log 2>&1 &", flush=True)
    print("     (compare R2 vs previous 0.708 to measure literature data impact)",
          flush=True)


if __name__ == "__main__":
    main()
