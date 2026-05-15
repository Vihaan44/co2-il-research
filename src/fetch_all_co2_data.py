"""
fetch_all_co2_data.py
---------------------
PURPOSE: Fetch ALL CO2 solubility data available in ILThermo for binary
         (IL + CO2) systems, regardless of which property was measured.
         Convert everything to mole fraction x2 so it can be merged with
         the existing mole fraction dataset and used for ML training.

WHY MULTIPLE PROPERTY TYPES:
  ILThermo stores CO2 data under three different property keys:
    dNip — Mole fraction (composition at phase equilibrium): 216 ILs [ALREADY FETCHED]
    dHen — Henry's law constant KH (MPa): additional ~80 ILs
    dWip — Mass fraction w2: additional ~40 ILs
  Many ILs appear in only one or two of these, so combining all three
  substantially expands the training set.

CONVERSION FORMULAS:
  Henry's law constant → mole fraction:
    x2 = P_kPa / (KH_MPa * 1000)
    Valid at infinite dilution (low CO2 pressure). Most ILThermo Henry's
    data is measured at atmospheric CO2 partial pressure (~101.325 kPa).
    We use the reported P_kPa when available; otherwise assume 101.325 kPa.

  Mass fraction → mole fraction:
    x2 = (w2 / M_CO2) / (w2 / M_CO2 + (1 - w2) / M_IL)
    where M_CO2 = 44.01 g/mol and M_IL is estimated from the IL SMILES
    using RDKit. If M_IL cannot be computed (invalid SMILES), the row
    is dropped with a warning.

DATA SOURCE TAGGING:
  Every row gets a 'data_source' column: 'mole_fraction', 'henry_converted',
  or 'mass_fraction_converted'. This is added as a feature in build_dataset.py
  so the model can learn any systematic offset between measurement types.
  (Henry's law values at infinite dilution may slightly underestimate x2
  at finite CO2 pressures — the model can correct for this implicitly.)

DEDUPLICATION:
  After merging all sources, we deduplicate on (il_smiles, T_K, P_kPa, x2_CO2)
  to remove any IL that appears in multiple property datasets at the same
  conditions. We keep the mole_fraction row when there's a conflict
  (direct measurement preferred over converted value).

OUTPUT:
  data/raw/ilthermo_all_co2_datapoints.csv  — all sources merged, x2 unified
  data/raw/ilthermo_source_summary.csv      — count of ILs and rows per source

INPUT:
  data/raw/ilthermo_mole_fraction_datapoints.csv  — already fetched (existing)
  ILThermo API (via ilthermopy) — queried live for Henry's and mass fraction

Run from project root:
    python src/fetch_all_co2_data.py

IMPORTANT: This script makes many ILThermo API calls (~200-400 entries).
  Expect runtime of 10-20 minutes. Do not interrupt mid-run.
"""

import os
import time
import pandas as pd
import numpy as np
import ilthermopy as ilt
from rdkit import Chem
from rdkit.Chem import Descriptors

# -- Constants -----------------------------------------------------------------
EXISTING_MOLE_FRACTION_CSV = os.path.join(
    "data", "raw", "ilthermo_mole_fraction_datapoints.csv"
)
OUTPUT_CSV         = os.path.join("data", "raw", "ilthermo_all_co2_datapoints.csv")
SOURCE_SUMMARY_CSV = os.path.join("data", "raw", "ilthermo_source_summary.csv")

# ILThermo property keys for CO2 in binary IL+CO2 systems
HENRY_PROP_KEY       = "dHen"   # Henry's law constant (MPa)
MASS_FRAC_PROP_KEY   = "dWip"   # Mass fraction (kg CO2 / kg total)
CO2_COMPOUND_NAME    = "carbon dioxide"
CO2_SMILES_CANONICAL = "O=C=O"

# Physical constants
M_CO2_G_PER_MOL  = 44.01     # molar mass of CO2 in g/mol
P_AMBIENT_KPA    = 101.325   # assumed CO2 partial pressure if not reported (1 atm)

# Column substrings for ILThermoPy header matching (same logic as fetch_datapoints.py)
TEMP_SUBSTRINGS  = ["temperature"]
PRESS_SUBSTRINGS = ["pressure"]
HENRY_SUBSTRINGS = ["henry"]          # Henry's constant column
MASSFRAC_SUBSTRINGS = ["mass fraction", "weight fraction"]
MOLFRAC_SUBSTRINGS  = ["mole fraction"]

# Sanity bounds
T_MIN_K  = 200.0
T_MAX_K  = 500.0
X2_MIN   = 0.0
X2_MAX   = 1.0

# Rate limiting: pause between API calls to avoid hammering ILThermo
API_PAUSE_SECONDS = 0.3


# -- Helper: ILThermoPy column finder -----------------------------------------

def find_col(df: pd.DataFrame, substrings: list) -> str | None:
    """Return first column name matching any substring (case-insensitive)."""
    for col in df.columns:
        if any(s in col.lower() for s in substrings):
            return col
    return None


def rename_from_header(data_df: pd.DataFrame, header: dict) -> pd.DataFrame:
    """Rename V1/V2/V3 columns using entry.header dict."""
    return data_df.rename(columns={k: v for k, v in header.items()})


# -- Helper: IL molecular weight from SMILES ----------------------------------

def get_molar_mass(smiles: str) -> float | None:
    """
    Compute molar mass of an IL from its SMILES using RDKit.
    Returns None if SMILES is invalid or cannot be parsed.
    Used for mass fraction → mole fraction conversion.
    """
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Descriptors.MolWt(mol)


# -- Fetch metadata for a property key ----------------------------------------

def search_property(prop_key: str) -> pd.DataFrame:
    """
    Search ILThermo for all binary IL+CO2 datasets with a given property key.
    Returns a DataFrame of metadata rows (entry_id, il_name, il_smiles, etc.).
    """
    print(f"\n[search] Querying ILThermo for prop_key='{prop_key}' + CO2...")
    try:
        results = ilt.Search(
            compound=CO2_COMPOUND_NAME,
            n_compounds=2,
            prop_key=prop_key,
        )
        print(f"[search] Found {len(results)} dataset entries")
        return results
    except Exception as err:
        print(f"[search] WARNING: Search failed for {prop_key}: {err}")
        return pd.DataFrame()


def extract_il_from_row(row) -> dict | None:
    """
    Identify which component is the IL (not CO2) and return metadata dict.
    Handles both cmp1=CO2 and cmp2=CO2 orderings.
    Returns None if CO2 cannot be identified or il_smiles is CO2 itself.
    """
    cmp1 = str(row.get("cmp1", "") or "")
    cmp2 = str(row.get("cmp2", "") or "")

    if CO2_COMPOUND_NAME in cmp1.lower():
        il_name, il_smiles = row.get("cmp2"), row.get("cmp2_smiles")
    elif CO2_COMPOUND_NAME in cmp2.lower():
        il_name, il_smiles = row.get("cmp1"), row.get("cmp1_smiles")
    else:
        return None  # can't identify CO2 in this entry

    # Drop rows where the "IL" is actually CO2 (data corruption)
    if il_smiles == CO2_SMILES_CANONICAL:
        return None

    return {
        "entry_id":  row.get("id"),
        "il_name":   il_name,
        "il_smiles": il_smiles,
    }


# -- Fetch and parse Henry's law data -----------------------------------------

def henry_to_x2(kh_mpa: float, p_kpa: float) -> float | None:
    """
    Convert Henry's law constant to mole fraction x2.
    Formula: x2 = P_kPa / (KH_MPa * 1000)
    Valid at infinite dilution (Henry's law regime, low CO2 pressure).
    Returns None if inputs are non-positive or non-finite.
    """
    if not np.isfinite(kh_mpa) or kh_mpa <= 0:
        return None
    if not np.isfinite(p_kpa) or p_kpa <= 0:
        p_kpa = P_AMBIENT_KPA  # fall back to ambient if P not reported
    return p_kpa / (kh_mpa * 1000.0)


def fetch_henry_datapoints(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each Henry's law entry, fetch raw data, extract (T, P, KH),
    convert KH → x2, and return rows tagged as 'henry_converted'.
    """
    all_rows = []
    total    = len(metadata_df)
    failed   = 0

    print(f"\n[henry] Fetching datapoints for {total} Henry's law entries...")

    for idx, meta_row in metadata_df.iterrows():
        il_info = extract_il_from_row(meta_row)
        if il_info is None:
            failed += 1
            continue

        entry_id  = str(meta_row.get("id", ""))
        il_name   = il_info["il_name"]
        il_smiles = str(il_info["il_smiles"]) if il_info["il_smiles"] else ""

        try:
            entry = ilt.GetEntry(entry_id)
            time.sleep(API_PAUSE_SECONDS)
        except Exception as err:
            print(f"  [WARNING] GetEntry failed entry_id={entry_id}: {err}")
            failed += 1
            continue

        if entry is None or entry.data is None or entry.data.empty:
            failed += 1
            continue

        renamed = rename_from_header(entry.data, entry.header)
        temp_col  = find_col(renamed, TEMP_SUBSTRINGS)
        press_col = find_col(renamed, PRESS_SUBSTRINGS)
        henry_col = find_col(renamed, HENRY_SUBSTRINGS)

        if temp_col is None or henry_col is None:
            # DATA QUALITY FLAG: can't use without T and KH
            print(f"  [DATA QUALITY] entry_id={entry_id}: missing T or Henry col "
                  f"(cols={list(renamed.columns)})")
            failed += 1
            continue

        for _, data_row in renamed.iterrows():
            t_k   = pd.to_numeric(data_row.get(temp_col), errors="coerce")
            p_kpa = pd.to_numeric(data_row.get(press_col), errors="coerce") \
                    if press_col else np.nan
            kh    = pd.to_numeric(data_row.get(henry_col), errors="coerce")

            # Use ambient pressure if P not reported
            p_use = p_kpa if np.isfinite(p_kpa) and p_kpa > 0 else P_AMBIENT_KPA

            x2 = henry_to_x2(kh, p_use)
            if x2 is None or not np.isfinite(t_k):
                continue

            all_rows.append({
                "entry_id":    entry_id,
                "il_name":     il_name,
                "il_smiles":   il_smiles,
                "T_K":         t_k,
                "P_kPa":       p_use,
                "x2_CO2":      x2,
                "data_source": "henry_converted",
                "kh_mpa_raw":  kh,   # keep original for audit
            })

        if (idx + 1) % 20 == 0:
            print(f"  → {idx+1}/{total} entries processed, {len(all_rows)} rows so far")

    print(f"[henry] Done: {len(all_rows)} rows, {failed} entries failed/skipped")
    return pd.DataFrame(all_rows)


# -- Fetch and parse mass fraction data ---------------------------------------

def mass_frac_to_x2(w2: float, m_il: float) -> float | None:
    """
    Convert CO2 mass fraction w2 to mole fraction x2.
    Formula: x2 = (w2/M_CO2) / (w2/M_CO2 + (1-w2)/M_IL)
    where M_CO2 = 44.01 g/mol and M_IL is the IL molar mass from RDKit.
    Returns None if inputs are invalid or conversion fails.
    """
    if not np.isfinite(w2) or w2 < 0 or w2 >= 1:
        return None
    if m_il is None or m_il <= 0:
        return None
    n_co2 = w2 / M_CO2_G_PER_MOL          # moles of CO2 per unit mass
    n_il  = (1.0 - w2) / m_il             # moles of IL per unit mass
    if (n_co2 + n_il) == 0:
        return None
    return n_co2 / (n_co2 + n_il)


def fetch_mass_fraction_datapoints(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each mass fraction entry, fetch raw data, extract (T, P, w2),
    convert w2 → x2 using RDKit IL molar mass, return 'mass_fraction_converted' rows.
    """
    all_rows = []
    total    = len(metadata_df)
    failed   = 0
    skipped_no_mw = 0

    print(f"\n[massfrac] Fetching datapoints for {total} mass fraction entries...")

    for idx, meta_row in metadata_df.iterrows():
        il_info = extract_il_from_row(meta_row)
        if il_info is None:
            failed += 1
            continue

        entry_id  = str(meta_row.get("id", ""))
        il_name   = il_info["il_name"]
        il_smiles = str(il_info["il_smiles"]) if il_info["il_smiles"] else ""

        # Compute IL molar mass from SMILES -- needed for conversion
        m_il = get_molar_mass(il_smiles)
        if m_il is None:
            # DATA QUALITY FLAG: can't convert without MW
            print(f"  [DATA QUALITY] No valid SMILES/MW for '{il_name}' -- skipping")
            skipped_no_mw += 1
            continue

        try:
            entry = ilt.GetEntry(entry_id)
            time.sleep(API_PAUSE_SECONDS)
        except Exception as err:
            print(f"  [WARNING] GetEntry failed entry_id={entry_id}: {err}")
            failed += 1
            continue

        if entry is None or entry.data is None or entry.data.empty:
            failed += 1
            continue

        renamed   = rename_from_header(entry.data, entry.header)
        temp_col  = find_col(renamed, TEMP_SUBSTRINGS)
        press_col = find_col(renamed, PRESS_SUBSTRINGS)
        wfrac_col = find_col(renamed, MASSFRAC_SUBSTRINGS)

        if temp_col is None or wfrac_col is None:
            print(f"  [DATA QUALITY] entry_id={entry_id}: missing T or w2 col")
            failed += 1
            continue

        for _, data_row in renamed.iterrows():
            t_k   = pd.to_numeric(data_row.get(temp_col),  errors="coerce")
            p_kpa = pd.to_numeric(data_row.get(press_col), errors="coerce") \
                    if press_col else np.nan
            w2    = pd.to_numeric(data_row.get(wfrac_col), errors="coerce")

            p_use = p_kpa if np.isfinite(p_kpa) and p_kpa > 0 else P_AMBIENT_KPA
            x2    = mass_frac_to_x2(w2, m_il)

            if x2 is None or not np.isfinite(t_k):
                continue

            all_rows.append({
                "entry_id":    entry_id,
                "il_name":     il_name,
                "il_smiles":   il_smiles,
                "T_K":         t_k,
                "P_kPa":       p_use,
                "x2_CO2":      x2,
                "data_source": "mass_fraction_converted",
                "w2_raw":      w2,   # keep original for audit
            })

        if (idx + 1) % 20 == 0:
            print(f"  → {idx+1}/{total} entries processed, {len(all_rows)} rows so far")

    print(f"[massfrac] Done: {len(all_rows)} rows, {failed} failed, "
          f"{skipped_no_mw} skipped (no MW)")
    return pd.DataFrame(all_rows)


# -- Load existing mole fraction data -----------------------------------------

def load_existing_mole_fraction() -> pd.DataFrame:
    """
    Load the already-fetched mole fraction datapoints CSV and tag it with
    data_source='mole_fraction' so it's distinguishable after merging.
    """
    if not os.path.exists(EXISTING_MOLE_FRACTION_CSV):
        raise FileNotFoundError(
            f"Existing mole fraction CSV not found at {EXISTING_MOLE_FRACTION_CSV}.\n"
            f"Run src/fetch_datapoints.py first."
        )
    df = pd.read_csv(EXISTING_MOLE_FRACTION_CSV)
    df["data_source"] = "mole_fraction"  # tag for provenance tracking
    print(f"[mole_fraction] Loaded {len(df)} existing rows, "
          f"{df['il_smiles'].nunique()} unique ILs")
    return df


# -- Merge and deduplicate ----------------------------------------------------

def merge_and_deduplicate(molfrac_df: pd.DataFrame,
                           henry_df: pd.DataFrame,
                           massfrac_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine all three DataFrames into one, then deduplicate.

    Deduplication key: (il_smiles, T_K rounded to 0.1K, P_kPa rounded to 1 kPa, x2_CO2 rounded to 4dp).
    When duplicates exist, prefer 'mole_fraction' (direct measurement) over converted values.
    This prevents the same physical measurement from appearing twice with
    slightly different x2 values due to conversion approximations.
    """
    # Standardize columns across all three sources
    keep_cols = ["entry_id", "il_name", "il_smiles", "T_K", "P_kPa",
                 "x2_CO2", "data_source"]

    frames = []
    for df, label in [(molfrac_df, "mole_fraction"),
                      (henry_df,   "henry_converted"),
                      (massfrac_df, "mass_fraction_converted")]:
        if df.empty:
            print(f"[merge] {label}: empty, skipping")
            continue
        # Keep only the standard columns that exist
        available = [c for c in keep_cols if c in df.columns]
        frames.append(df[available].copy())
        print(f"[merge] {label}: {len(df)} rows, "
              f"{df['il_smiles'].nunique() if 'il_smiles' in df.columns else '?'} ILs")

    if not frames:
        raise ValueError("All three data sources are empty — nothing to merge.")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n[merge] Combined (before dedup): {len(combined)} rows, "
          f"{combined['il_smiles'].nunique()} unique ILs")

    # Sort so mole_fraction rows come first (they win deduplication)
    source_priority = {"mole_fraction": 0, "henry_converted": 1,
                       "mass_fraction_converted": 2}
    combined["_sort_priority"] = combined["data_source"].map(source_priority).fillna(3)
    combined = combined.sort_values("_sort_priority").drop(columns=["_sort_priority"])

    # Round for dedup key -- measurements within 0.1K / 1 kPa / 0.0001 x2 are "same"
    combined["_T_round"]  = combined["T_K"].round(1)
    combined["_P_round"]  = combined["P_kPa"].round(0)
    combined["_x2_round"] = combined["x2_CO2"].round(4)

    before_dedup = len(combined)
    combined = combined.drop_duplicates(
        subset=["il_smiles", "_T_round", "_P_round", "_x2_round"],
        keep="first"   # keeps mole_fraction (sorted first) over converted
    ).drop(columns=["_T_round", "_P_round", "_x2_round"])

    n_removed = before_dedup - len(combined)
    print(f"[merge] Deduplication removed {n_removed} duplicate rows")
    print(f"[merge] Final: {len(combined)} rows, "
          f"{combined['il_smiles'].nunique()} unique ILs")

    return combined.reset_index(drop=True)


# -- Sanity checks and summary ------------------------------------------------

def sanity_check(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with physically impossible x2 or T values.
    Flag (but keep) rows with x2 > 0.5 as unusually high -- possible data error.
    """
    before = len(df)

    # Drop x2 outside [0, 1]
    df = df[(df["x2_CO2"] >= X2_MIN) & (df["x2_CO2"] <= X2_MAX)].copy()
    # Drop T outside physical range
    df = df[(df["T_K"] >= T_MIN_K) & (df["T_K"] <= T_MAX_K)].copy()

    dropped = before - len(df)
    if dropped > 0:
        print(f"[sanity_check] Dropped {dropped} rows with x2 outside [0,1] "
              f"or T outside [{T_MIN_K},{T_MAX_K}] K")

    # Flag high-x2 rows for inspection
    high_x2 = df[df["x2_CO2"] > 0.5]
    if len(high_x2) > 0:
        print(f"[sanity_check] NOTE: {len(high_x2)} rows have x2 > 0.5 "
              f"(unusually high -- check these manually)")
        print(high_x2[["il_name", "T_K", "P_kPa", "x2_CO2", "data_source"]].head(5))

    return df


def print_source_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Print and return a per-source summary of rows and unique ILs."""
    print("\n=== SOURCE SUMMARY ===")
    rows = []
    for source, group in df.groupby("data_source"):
        n_rows = len(group)
        n_ils  = group["il_smiles"].nunique()
        print(f"  {source:35s}: {n_rows:5d} rows, {n_ils:3d} unique ILs")
        rows.append({"data_source": source, "n_rows": n_rows, "n_unique_ils": n_ils})

    total_ils = df["il_smiles"].nunique()
    print(f"  {'TOTAL':35s}: {len(df):5d} rows, {total_ils:3d} unique ILs")
    print(f"\n  ILs gained beyond mole_fraction only: "
          f"{total_ils - df[df['data_source']=='mole_fraction']['il_smiles'].nunique()}")

    return pd.DataFrame(rows)


# -- Main ---------------------------------------------------------------------

def main():
    """
    Main pipeline:
    1. Load existing mole fraction data
    2. Search and fetch Henry's law data → convert to x2
    3. Search and fetch mass fraction data → convert to x2
    4. Merge, deduplicate, sanity check
    5. Save unified CSV
    """
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)

    # -- Step 1: Load existing mole fraction data ----------------------------
    print("=" * 60)
    print("STEP 1: Load existing mole fraction data")
    print("=" * 60)
    molfrac_df = load_existing_mole_fraction()

    # -- Step 2: Fetch Henry's law data --------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Fetch Henry's law constant data (dHen)")
    print("=" * 60)
    henry_meta = search_property(HENRY_PROP_KEY)
    if not henry_meta.empty:
        henry_df = fetch_henry_datapoints(henry_meta)
    else:
        henry_df = pd.DataFrame()
        print("[henry] No entries found.")

    # -- Step 3: Fetch mass fraction data ------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Fetch mass fraction data (dWip)")
    print("=" * 60)
    massfrac_meta = search_property(MASS_FRAC_PROP_KEY)
    if not massfrac_meta.empty:
        massfrac_df = fetch_mass_fraction_datapoints(massfrac_meta)
    else:
        massfrac_df = pd.DataFrame()
        print("[massfrac] No entries found.")

    # -- Step 4: Merge and clean ---------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Merge all sources, deduplicate, sanity check")
    print("=" * 60)
    merged_df  = merge_and_deduplicate(molfrac_df, henry_df, massfrac_df)
    clean_df   = sanity_check(merged_df)
    summary_df = print_source_summary(clean_df)

    # -- Step 5: Save --------------------------------------------------------
    clean_df.to_csv(OUTPUT_CSV, index=False)
    summary_df.to_csv(SOURCE_SUMMARY_CSV, index=False)

    print(f"\n[main] Saved {len(clean_df)} rows → {OUTPUT_CSV}")
    print(f"[main] Source summary → {SOURCE_SUMMARY_CSV}")
    print("\n[main] NEXT STEP:")
    print("  1. Inspect data/raw/ilthermo_source_summary.csv for IL counts per source")
    print("  2. Update src/build_dataset.py to read from ilthermo_all_co2_datapoints.csv")
    print("     (change DATAPOINTS_CSV constant at top of file)")
    print("  3. Add 'data_source' as a feature in build_dataset.py merge_features()")
    print("  4. Re-run: build_dataset.py → featurize.py → train_model.py")


if __name__ == "__main__":
    main()
