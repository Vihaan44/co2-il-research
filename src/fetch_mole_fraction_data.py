"""
fetch_mole_fraction_data.py — Phase 1b data collection script (mole fraction solubility).

Queries ILThermo for CO2 mole fraction solubility (composition at phase
equilibrium) in binary IL + CO2 systems. Supplements the Henry's law
dataset from fetch_ilthermo_data.py.

Why mole fraction solubility?
  Henry's law search returned only 52 unique ILs. Mole fraction solubility
  (prop key dNip) covers 216 unique ILs across 518 datasets — much better
  structural diversity for ML training.

  These are related but not identical properties. Henry's law constant KH
  and mole fraction x2 are connected by KH = P / x2 at infinite dilution,
  but ILThermo reports them under different experiments at different T/P
  conditions. We collect them separately and decide in Phase 2 whether to
  train separate models or convert one to the other.

ILThermo property key:
  dNip — "Composition at phase equilibrium" (confirmed via ilt.ShowPropertyList())

Input:  nothing (queries ILThermo 2.0 via ILThermoPy)
Output: data/raw/ilthermo_mole_fraction_raw.csv

Run from the project root:
    python src/fetch_mole_fraction_data.py
"""

import ilthermopy as ilt
import pandas as pd

# Correct ILThermo key for mole fraction / composition at phase equilibrium
# Confirmed via ilt.ShowPropertyList() — "x2" is NOT a valid key
MOLE_FRACTION_PROP_KEY = "dNip"
CO2_COMPOUND_NAME      = "carbon dioxide"
CO2_SMILES             = "O=C=O"   # used to detect and drop corrupt rows
OUTPUT_PATH            = "data/raw/ilthermo_mole_fraction_raw.csv"


def search_mole_fraction_datasets():
    """
    Searches ILThermo for all binary (IL + CO2) datasets measuring
    CO2 mole fraction solubility. Returns a DataFrame or empty DataFrame.
    """
    print(f"Searching ILThermo for mole fraction solubility (key: {MOLE_FRACTION_PROP_KEY})...")
    try:
        results_df = ilt.Search(
            compound=CO2_COMPOUND_NAME,
            n_compounds=2,
            prop_key=MOLE_FRACTION_PROP_KEY,
        )
        print(f"  Found {len(results_df)} datasets")
        return results_df
    except Exception as error:
        print(f"  WARNING: Search failed: {error}")
        return pd.DataFrame()


def extract_row_from_search_result(row):
    """
    Converts one ILThermoPy search result row into a flat dict for our CSV.

    Uses the same CO2-detection-by-name logic as fetch_ilthermo_data.py:
    ILThermoPy component order is not guaranteed, so we always identify
    CO2 by name and assign the other component as the IL.
    Returns None if CO2 cannot be identified.
    """
    cmp1_name = row.get("cmp1", "") or ""
    cmp2_name = row.get("cmp2", "") or ""

    if CO2_COMPOUND_NAME in cmp1_name.lower():
        il_name   = row.get("cmp2")
        il_smiles = row.get("cmp2_smiles")
    elif CO2_COMPOUND_NAME in cmp2_name.lower():
        il_name   = row.get("cmp1")
        il_smiles = row.get("cmp1_smiles")
    else:
        print(f"  DATA QUALITY WARNING: Cannot identify CO2 in entry "
              f"{row.get('id')} — cmp1='{cmp1_name}', cmp2='{cmp2_name}'. Skipping.")
        return None

    if il_smiles is None:
        print(f"  DATA QUALITY WARNING: No SMILES for '{il_name}' "
              f"(entry {row.get('id')}) — will be dropped in Phase 2")

    return {
        "entry_id":        row.get("id"),
        "reference":       row.get("reference"),
        "property":        "mole_fraction_co2",
        "il_name":         il_name,
        "il_smiles":       il_smiles,
        "num_data_points": row.get("num_data_points"),
        "phases":          row.get("phases"),
    }


def drop_corrupt_co2_rows(datasets_df):
    """
    Drops rows where the 'IL' was incorrectly assigned as CO2 itself.
    Same issue as in fetch_ilthermo_data.py — see that file for explanation.
    """
    corrupt_mask = datasets_df["il_smiles"] == CO2_SMILES
    num_corrupt  = corrupt_mask.sum()
    clean_df     = datasets_df[~corrupt_mask].reset_index(drop=True)

    if num_corrupt > 0:
        print(f"  DATA QUALITY: Dropped {num_corrupt} corrupt rows where "
              f"il_smiles was CO2.")
    else:
        print(f"  DATA QUALITY: No corrupt rows found.")

    return clean_df


def fetch_mole_fraction_data():
    """
    Main function: searches ILThermo for mole fraction solubility data,
    cleans results, and saves to CSV.
    """
    search_results_df = search_mole_fraction_datasets()

    if search_results_df.empty:
        print("\nERROR: No mole fraction data found.")
        return pd.DataFrame()

    all_rows = []
    for _, row in search_results_df.iterrows():
        flat_row = extract_row_from_search_result(row)
        if flat_row is not None:
            all_rows.append(flat_row)

    datasets_df = pd.DataFrame(all_rows)
    datasets_df = drop_corrupt_co2_rows(datasets_df)

    print(f"\n--- Summary ---")
    print(f"Total clean datasets: {len(datasets_df)}")
    print(f"Unique ILs: {datasets_df['il_name'].nunique()}")

    missing_smiles = datasets_df['il_smiles'].isna().sum()
    print(f"Missing SMILES: {missing_smiles} ({100*missing_smiles/len(datasets_df):.1f}%)")

    datasets_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")
    print("\nDatasets per IL (top 20):")
    print(datasets_df['il_name'].value_counts().head(20).to_string())

    return datasets_df


if __name__ == "__main__":
    fetch_mole_fraction_data()
