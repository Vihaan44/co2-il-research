"""
merge_thermoml_into_pipeline.py
--------------------------------
PURPOSE: Merge the ThermoML CO2-IL dataset with the existing ILThermo dataset,
         deduplicate, and produce a unified datapoints CSV that build_dataset.py
         can use directly to expand the training set.

WHAT THIS SCRIPT DOES:
  1. Loads existing ILThermo mole fraction data (211 ILs, ~9,168 rows)
  2. Loads ThermoML converted data (126 ILs, 7,191 rows)
  3. Cross-references by SMILES to find genuinely new ILs not in ILThermo
  4. Deduplicates overlapping measurements (same IL, same T/P/x2)
  5. Saves merged CSV as data/raw/all_co2_datapoints_merged.csv
  6. Reports how many new ILs and rows were added

WHY SMILES-BASED DEDUPLICATION:
  The same IL can appear in both datasets under different names
  (e.g. "[BMIM][BF4]" vs "1-butyl-3-methylimidazolium tetrafluoroborate").
  We use canonical RDKit SMILES as the deduplication key, not names.
  Two SMILES strings referring to the same IL will canonicalize identically.

OUTPUT:
  data/raw/all_co2_datapoints_merged.csv  -- unified dataset
  data/raw/merge_summary.txt              -- statistics

NEXT STEP AFTER RUNNING THIS:
  Update build_dataset.py DATAPOINTS_CSV to point at:
    data/raw/all_co2_datapoints_merged.csv
  Then re-run:
    python src/build_dataset.py
    python src/featurize.py
    python src/train_model.py

INPUT:
  data/raw/ilthermo_mole_fraction_datapoints.csv  (existing)
  data/raw/thermoml_co2_il_with_smiles.csv        (from thermoml_inchi_to_smiles.py)

Run from project root:
    python src/merge_thermoml_into_pipeline.py
"""

import os
import sys
import pandas as pd
import numpy as np
from rdkit import Chem

sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)

# -- Constants -----------------------------------------------------------------
ILTHERMO_CSV    = os.path.join("data", "raw", "ilthermo_mole_fraction_datapoints.csv")
THERMOML_CSV    = os.path.join("data", "raw", "thermoml_co2_il_with_smiles.csv")
OUTPUT_CSV      = os.path.join("data", "raw", "all_co2_datapoints_merged.csv")
SUMMARY_FILE    = os.path.join("data", "raw", "merge_summary.txt")

# Deduplication tolerance -- measurements within these ranges are "the same"
T_TOLERANCE_K   = 0.5     # K
P_TOLERANCE_KPA = 2.0     # kPa
X2_TOLERANCE    = 0.005   # mole fraction


def canonicalize_smiles(smiles: str) -> str | None:
    """
    Convert a SMILES string to RDKit canonical form.
    Returns None if SMILES is invalid.
    Canonical SMILES is used as the deduplication key so that
    different representations of the same IL collapse to one.
    """
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def load_ilthermo(path: str) -> pd.DataFrame:
    """
    Load the existing ILThermo mole fraction dataset.
    Adds canonical SMILES column for cross-referencing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ILThermo CSV not found at {path}. Run fetch_datapoints.py first."
        )
    df = pd.read_csv(path)
    df["data_source"] = "ilthermo_mole_fraction"
    df["canonical_smiles"] = df["il_smiles"].apply(canonicalize_smiles)

    n_valid = df["canonical_smiles"].notna().sum()
    print(f"[load_ilthermo] {len(df)} rows, {df['il_smiles'].nunique()} unique ILs "
          f"({n_valid} with valid canonical SMILES)", flush=True)
    return df


def load_thermoml(path: str) -> pd.DataFrame:
    """
    Load the ThermoML dataset with SMILES already attached.
    Standardizes column names to match ILThermo format.
    Adds canonical SMILES for cross-referencing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ThermoML CSV not found at {path}. Run thermoml_inchi_to_smiles.py first."
        )
    df = pd.read_csv(path)

    # Rename columns to match ILThermo schema
    df = df.rename(columns={
        "il_smiles":  "il_smiles",
        "il_name":    "il_name",
        "x2_CO2":     "x2_CO2",
        "data_type":  "data_source",
    })

    # Map data_type → clearer source label
    df["data_source"] = df["data_source"].map({
        "mole_fraction":        "thermoml_mole_fraction",
        "henry_converted":      "thermoml_henry_converted",
    }).fillna("thermoml_other")

    df["canonical_smiles"] = df["il_smiles"].apply(canonicalize_smiles)

    n_valid = df["canonical_smiles"].notna().sum()
    print(f"[load_thermoml] {len(df)} rows, {df['il_smiles'].nunique()} unique ILs "
          f"({n_valid} with valid canonical SMILES)", flush=True)
    return df


def find_new_ils(ilthermo_df: pd.DataFrame,
                 thermoml_df: pd.DataFrame) -> tuple:
    """
    Find which ThermoML ILs are genuinely new (not in ILThermo).
    Returns (new_il_smiles_set, overlap_il_smiles_set).

    Two ILs are considered the same if their canonical SMILES match.
    This handles naming inconsistencies between the two databases.
    """
    ilthermo_smiles = set(
        ilthermo_df["canonical_smiles"].dropna().unique()
    )
    thermoml_smiles = set(
        thermoml_df["canonical_smiles"].dropna().unique()
    )

    new_ils     = thermoml_smiles - ilthermo_smiles   # in ThermoML but not ILThermo
    overlap_ils = thermoml_smiles & ilthermo_smiles   # in both

    print(f"\n[find_new_ils] ILThermo unique ILs:  {len(ilthermo_smiles)}", flush=True)
    print(f"[find_new_ils] ThermoML unique ILs:  {len(thermoml_smiles)}", flush=True)
    print(f"[find_new_ils] Overlap (same IL):     {len(overlap_ils)}", flush=True)
    print(f"[find_new_ils] NEW ILs from ThermoML: {len(new_ils)}", flush=True)

    return new_ils, overlap_ils


def deduplicate_measurements(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate measurements: rows where the same IL has the same
    T_K (within ±0.5K), P_kPa (within ±2 kPa), and x2_CO2 (within ±0.005)
    in both datasets. When duplicates exist, keep the ILThermo row
    (direct mole fraction measurement preferred over any conversion).

    Deduplication is approximate (tolerance-based) because different labs
    may report slightly different T/P for nominally the same condition.
    """
    before = len(merged_df)

    # Sort so ILThermo rows come first (they win deduplication)
    source_priority = {
        "ilthermo_mole_fraction":    0,
        "thermoml_mole_fraction":    1,
        "thermoml_henry_converted":  2,
        "thermoml_other":            3,
    }
    merged_df["_priority"] = merged_df["data_source"].map(source_priority).fillna(4)
    merged_df = merged_df.sort_values("_priority").drop(columns=["_priority"])

    # Round to dedup tolerance before dropping duplicates
    merged_df["_T_round"]  = (merged_df["T_K"]    / T_TOLERANCE_K).round()
    merged_df["_P_round"]  = (merged_df["P_kPa"]  / P_TOLERANCE_KPA).round()
    merged_df["_x2_round"] = (merged_df["x2_CO2"] / X2_TOLERANCE).round()

    merged_df = merged_df.drop_duplicates(
        subset=["canonical_smiles", "_T_round", "_P_round", "_x2_round"],
        keep="first"
    ).drop(columns=["_T_round", "_P_round", "_x2_round"])

    n_removed = before - len(merged_df)
    print(f"[deduplicate] Removed {n_removed} duplicate measurements "
          f"({before} → {len(merged_df)} rows)", flush=True)

    return merged_df.reset_index(drop=True)


def main():
    """Main: load both datasets → find new ILs → merge → deduplicate → save."""
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)

    # -- Step 1: Load both datasets ------------------------------------------
    print("=== STEP 1: Load datasets ===", flush=True)
    ilthermo_df  = load_ilthermo(ILTHERMO_CSV)
    thermoml_df  = load_thermoml(THERMOML_CSV)

    # -- Step 2: Find genuinely new ILs --------------------------------------
    print("\n=== STEP 2: Cross-reference ILs by canonical SMILES ===", flush=True)
    new_il_smiles, overlap_smiles = find_new_ils(ilthermo_df, thermoml_df)

    # Show names of new ILs
    new_il_names = thermoml_df[
        thermoml_df["canonical_smiles"].isin(new_il_smiles)
    ]["il_name"].unique()

    print(f"\n[main] New ILs from ThermoML (not in ILThermo):", flush=True)
    for name in sorted(new_il_names):
        n_rows = len(thermoml_df[thermoml_df["il_name"] == name])
        print(f"  {name}  ({n_rows} rows)", flush=True)

    # -- Step 3: Standardize columns for merge -------------------------------
    print("\n=== STEP 3: Standardize and merge ===", flush=True)

    # Ensure both DataFrames have the same core columns
    core_cols = ["il_name", "il_smiles", "canonical_smiles",
                 "T_K", "P_kPa", "x2_CO2", "data_source"]

    # ILThermo may have extra columns -- keep only core ones plus any extras
    ilthermo_core = ilthermo_df[[c for c in core_cols if c in ilthermo_df.columns]].copy()
    thermoml_core = thermoml_df[[c for c in core_cols if c in thermoml_df.columns]].copy()

    merged_df = pd.concat([ilthermo_core, thermoml_core], ignore_index=True)
    print(f"[main] Before dedup: {len(merged_df)} rows, "
          f"{merged_df['canonical_smiles'].nunique()} unique ILs", flush=True)

    # -- Step 4: Deduplicate -------------------------------------------------
    print("\n=== STEP 4: Deduplicate ===", flush=True)
    merged_df = deduplicate_measurements(merged_df)

    # -- Step 5: Final statistics --------------------------------------------
    n_total_ils      = merged_df["canonical_smiles"].nunique()
    n_ilthermo_rows  = (merged_df["data_source"] == "ilthermo_mole_fraction").sum()
    n_thermoml_rows  = merged_df["data_source"].str.startswith("thermoml").sum()

    print(f"\n=== FINAL MERGED DATASET ===", flush=True)
    print(f"  Total rows:            {len(merged_df)}", flush=True)
    print(f"  Total unique ILs:      {n_total_ils}", flush=True)
    print(f"  ILThermo rows:         {n_ilthermo_rows}", flush=True)
    print(f"  ThermoML rows added:   {n_thermoml_rows}", flush=True)
    print(f"  New ILs added:         {len(new_il_smiles)}", flush=True)
    print(f"  ILs gained: {len(ilthermo_df['canonical_smiles'].dropna().unique())} "
          f"→ {n_total_ils} unique ILs", flush=True)

    # -- Step 6: Save --------------------------------------------------------
    # Drop canonical_smiles (internal use only) before saving
    save_df = merged_df.drop(columns=["canonical_smiles"], errors="ignore")
    save_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[main] Saved → {OUTPUT_CSV}", flush=True)

    # Write summary
    summary = [
        "Merge Summary: ILThermo + ThermoML",
        f"ILThermo rows:            {n_ilthermo_rows}",
        f"ThermoML rows added:      {n_thermoml_rows}",
        f"Total rows after dedup:   {len(merged_df)}",
        f"ILThermo unique ILs:      {len(ilthermo_df['canonical_smiles'].dropna().unique())}",
        f"ThermoML unique ILs:      {thermoml_df['canonical_smiles'].nunique()}",
        f"Overlap ILs:              {len(overlap_smiles)}",
        f"New ILs from ThermoML:    {len(new_il_smiles)}",
        f"Total unique ILs (merged):{n_total_ils}",
        "",
        "New ILs added:",
    ] + [f"  {name}" for name in sorted(new_il_names)]

    with open(SUMMARY_FILE, "w") as f:
        f.write("\n".join(summary))
    print(f"[main] Summary → {SUMMARY_FILE}", flush=True)

    print(f"\n=== NEXT STEPS ===", flush=True)
    print(f"  1. Update src/build_dataset.py:", flush=True)
    print(f"     Change DATAPOINTS_CSV to 'data/raw/all_co2_datapoints_merged.csv'",
          flush=True)
    print(f"  2. Re-run the full pipeline:", flush=True)
    print(f"     python src/build_dataset.py", flush=True)
    print(f"     python src/featurize.py  (only if new ILs need featurizing)", flush=True)
    print(f"     python src/train_model.py", flush=True)


if __name__ == "__main__":
    main()
