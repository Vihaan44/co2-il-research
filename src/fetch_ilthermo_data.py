"""
fetch_ilthermo_data.py — Phase 1 data collection script.

Uses the ILThermoPy library to query ILThermo 2.0 for CO2-related
solubility data (Henry's law constant).

ILThermoPy is used instead of raw HTTP because:
  1. There is no official ILThermo REST API.
  2. ILThermoPy provides pre-validated SMILES for each IL, which we
     need for RDKit featurization in Phase 2.

Input:  nothing (queries ILThermo 2.0 via ILThermoPy)
Output: data/raw/ilthermo_co2_raw.csv

Run from the project root:
    python src/fetch_ilthermo_data.py
"""

import ilthermopy as ilt
import pandas as pd
import time

HENRYS_LAW_PROP_KEY   = "lIUh"
OSTWALD_PROP_KEY      = "eCTp"
CO2_COMPOUND_NAME     = "carbon dioxide"
REQUEST_DELAY_SECONDS = 1.0
OUTPUT_PATH           = "data/raw/ilthermo_co2_raw.csv"


def search_co2_datasets(prop_key, prop_label):
    """
    Searches ILThermo for all binary (IL + CO2) datasets for a given property.
    Returns a DataFrame of dataset metadata including SMILES, or empty DataFrame.
    """
    print(f"\nSearching for: {prop_label} (key: {prop_key})")
    try:
        results_df = ilt.Search(
            compound=CO2_COMPOUND_NAME,
            n_compounds=2,
            prop_key=prop_key,
        )
        print(f"  Found {len(results_df)} datasets")
        return results_df
    except Exception as error:
        print(f"  WARNING: Search failed for {prop_label}: {error}")
        return pd.DataFrame()


def extract_row_from_search_result(row, prop_label):
    """
    Converts one ILThermoPy search result row into a flat dict for our CSV.

    ILThermoPy does not guarantee cmp1=IL and cmp2=CO2 — the order can be
    flipped. We detect which component is CO2 by name and always assign the
    other one as the IL. Returns None if CO2 component can't be identified.
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
        "property":        prop_label,
        "il_name":         il_name,
        "il_smiles":       il_smiles,
        "num_data_points": row.get("num_data_points"),
        "phases":          row.get("phases"),
    }


def fetch_all_co2_data():
    """
    Searches ILThermo for Henry's law constant and Ostwald coefficient data
    involving CO2, extracts IL names and SMILES, and saves to CSV.
    """
    all_rows = []

    properties_to_fetch = [
        (HENRYS_LAW_PROP_KEY, "henrys_law_constant"),
        (OSTWALD_PROP_KEY,    "ostwald_coefficient"),
    ]

    for prop_key, prop_label in properties_to_fetch:
        search_results_df = search_co2_datasets(prop_key, prop_label)

        if search_results_df.empty:
            print(f"  No data found for {prop_label}, skipping.")
            continue

        for _, row in search_results_df.iterrows():
            flat_row = extract_row_from_search_result(row, prop_label)
            if flat_row is not None:
                all_rows.append(flat_row)

        time.sleep(REQUEST_DELAY_SECONDS)

    co2_datasets_df = pd.DataFrame(all_rows)

    if co2_datasets_df.empty:
        print("\nERROR: No data collected. Check ILThermoPy and network.")
        return co2_datasets_df

    print(f"\n--- Summary ---")
    print(f"Total datasets collected: {len(co2_datasets_df)}")
    print(f"Unique ILs: {co2_datasets_df['il_name'].nunique()}")
    print(f"Properties: {co2_datasets_df['property'].unique()}")

    missing_smiles = co2_datasets_df['il_smiles'].isna().sum()
    print(f"Missing SMILES: {missing_smiles} ({100*missing_smiles/len(co2_datasets_df):.1f}%)")

    co2_datasets_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")
    print(co2_datasets_df['il_name'].value_counts().to_string())

    return co2_datasets_df


if __name__ == "__main__":
    fetch_all_co2_data()
