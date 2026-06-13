"""
add_literature_data.py
----------------------
PURPOSE: Convert Ramdin 2015 literature data (Henry's constants + direct mole
         fraction measurements) into the same format as all_co2_datapoints_merged.csv,
         then append it to produce an expanded dataset for retraining.

WHY THIS DATA MATTERS:
  The compare_models.py run showed RF test R²=0.704 while all boosting methods
  collapsed to 0.57-0.62 on held-out ILs. This is a data coverage problem: the
  60 test ILs are structurally different from the 239 training ILs. Adding 11
  new ILs (69 rows) from Ramdin 2015 expands structural diversity and may
  partially close the test gap.

DATA SOURCES IN data.md:
  Two property types require different handling:

  1. henry (H in MPa, no P_kPa):
     Conversion: x2 = P_ref / H
     where P_ref = P_REFERENCE_KPA (101.325 kPa = 1 atm standard condition).
     Derivation: Henry's law at infinite dilution states H = P_CO2 / x2,
     so x2 = P_CO2 / H. At the standard reporting condition P_CO2 = 1 atm,
     x2 = 101.325 kPa / H(kPa). This gives the mole fraction solubility
     at 1 atm partial pressure of CO2 -- directly comparable to ILThermo
     data collected near atmospheric pressure.
     P_kPa in output is set to P_REFERENCE_KPA so the model sees a real
     pressure value, not NaN (which would cause the row to be dropped).

  2. mole_fraction (x2 already measured, P_kPa provided):
     Use x2 directly. Apply P > P_MAX_KPA cutoff to exclude supercritical
     measurements that are outside the training distribution.

EXCLUSION RULES (applied before conversion):
  - calculated_not_measured: [toa][Tf2N] rows are COSMO-RS predictions,
    not experimental data. Including calculated values as training data
    would bias the model toward computational artifacts.
  - SMILES_verify: [cprop] cation SMILES is unusual and unvalidated.
    RDKit may fail or produce wrong descriptors. Exclude until verified.
  - single_T_point: [bmim][Tf2N] henry row is a single point from a
    secondary source -- likely already in ILThermo dataset as a duplicate.
  - secondary_source without check_ILThermo cleared: [thtdp] rows sourced
    from Ramdin reference [346] (not measured by Ramdin directly). Kept
    because [thtdp] phosphonium ILs are structurally distinct and add
    genuine diversity. Flag retained in data_source column.
  - T > T_MAX_K: measurements above 340K are outside the typical operating
    range for CO2 capture and outside the ILThermo training distribution.
    Excluded to avoid T extrapolation artifacts.

OUTPUT:
  data/raw/all_co2_datapoints_expanded.csv  -- original + Ramdin rows
  results/literature_integration_report.csv -- what was added/excluded

IMPORTANT: After running this script, you MUST rerun the full pipeline:
  python src/featurize.py          (if new ILs have new SMILES)
  python src/build_dataset.py      (point DATAPOINTS_CSV at expanded file)
  python src/compare_models.py     (check whether test R² improved)

Run from project root:
  python src/add_literature_data.py
"""

import os
import pandas as pd
import numpy as np

# -- Constants -----------------------------------------------------------------
SOURCE_CSV    = os.path.join("data", "raw", "all_co2_datapoints_merged.csv")
LITERATURE_MD = os.path.join("data", "raw", "ramdin_2015_literature.md")
OUTPUT_CSV    = os.path.join("data", "raw", "all_co2_datapoints_expanded.csv")
REPORT_CSV    = os.path.join("results", "literature_integration_report.csv")

# Henry's law reference pressure for x2 conversion
# x2 = P_ref / H(kPa). Use 1 atm so output is comparable to near-atmospheric
# ILThermo measurements. Do NOT use a higher pressure -- that would overestimate
# solubility and introduce systematic bias relative to training data.
P_REFERENCE_KPA = 101.325

# Exclude measurements above this temperature -- outside ILThermo training range
T_MAX_K = 340.0

# Exclude mole fraction rows at very high pressure (supercritical regime,
# outside Henry's law linearity, outside ILThermo training distribution)
P_MAX_KPA = 2000.0

# Flag substrings that trigger row exclusion
EXCLUDE_FLAGS = [
    "calculated_not_measured",  # COSMO-RS predictions, not experimental
    "SMILES_verify",            # unvalidated unusual cation SMILES
    "single_T_point",           # secondary single-point, likely ILThermo duplicate
]

DATA_SOURCE_LABEL = "ramdin_2015_literature"


def load_existing_data(path: str) -> pd.DataFrame:
    """Load the existing merged datapoints CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Source CSV not found: {path}\n"
            "Run src/merge_thermoml_into_pipeline.py first."
        )
    df = pd.read_csv(path)
    print(f"[load] Existing dataset: {len(df)} rows, "
          f"{df['il_smiles'].nunique()} unique ILs", flush=True)
    return df


def parse_literature_md(path: str) -> pd.DataFrame:
    """
    Parse the Ramdin 2015 markdown table into a DataFrame.
    Expects pipe-delimited markdown table with header row and separator row.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Literature markdown not found: {path}\n"
            "Save data/raw/ramdin_2015_literature.md from the extracted data."
        )

    rows = []
    with open(path) as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line.startswith("|") or "---" in line:
            continue  # skip separator rows and non-table lines
        cols = [c.strip() for c in line.split("|")[1:-1]]  # strip outer pipes
        rows.append(cols)

    if len(rows) < 2:
        raise ValueError("Could not parse any rows from literature markdown.")

    header = rows[0]
    data   = rows[1:]
    df     = pd.DataFrame(data, columns=header)
    print(f"[parse_md] Parsed {len(df)} rows from markdown table", flush=True)
    return df


def apply_exclusion_rules(df: pd.DataFrame) -> tuple:
    """
    Remove rows that should not be used for training.
    Returns (kept_df, excluded_df) with an 'exclusion_reason' column on excluded_df.
    """
    excluded_rows = []
    keep_mask     = pd.Series([True] * len(df), index=df.index)

    for flag_substring in EXCLUDE_FLAGS:
        flagged = df["flags"].fillna("").str.contains(flag_substring, case=False)
        newly_excluded = df[flagged & keep_mask].copy()
        newly_excluded["exclusion_reason"] = f"flag:{flag_substring}"
        excluded_rows.append(newly_excluded)
        keep_mask = keep_mask & ~flagged
        print(f"[exclude] flag '{flag_substring}': {flagged.sum()} rows excluded",
              flush=True)

    # Exclude T > T_MAX_K
    high_t = df["T_K"].astype(float) > T_MAX_K
    newly_excluded = df[high_t & keep_mask].copy()
    newly_excluded["exclusion_reason"] = f"T>{T_MAX_K}K"
    excluded_rows.append(newly_excluded)
    keep_mask = keep_mask & ~high_t
    print(f"[exclude] T > {T_MAX_K}K: {high_t.sum()} rows excluded", flush=True)

    kept_df     = df[keep_mask].copy()
    excluded_df = pd.concat(excluded_rows, ignore_index=True) if excluded_rows else pd.DataFrame()

    print(f"[exclude] Kept {len(kept_df)} / {len(df)} rows after exclusions",
          flush=True)
    return kept_df, excluded_df


def convert_henry_to_x2(df_henry: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Henry's constant rows (H in MPa, no P) to mole fraction rows.

    Conversion: x2 = P_ref_kPa / (H_MPa * 1000)
    Derivation: H = P_CO2 / x2  =>  x2 = P_CO2 / H
    P_ref = 101.325 kPa = 0.101325 MPa (1 atm standard condition)
    H in MPa -> H_kPa = H_MPa * 1000
    x2 = 101.325 / H_kPa

    Sets P_kPa = P_REFERENCE_KPA so build_dataset.py doesn't drop the row
    for missing pressure (which is a required condition column).
    """
    df = df_henry.copy()

    h_mpa = df["value"].astype(float)
    h_kpa = h_mpa * 1000.0                         # MPa -> kPa
    x2    = P_REFERENCE_KPA / h_kpa                # Henry's law inversion

    # Sanity check: x2 must be in (0, 1)
    bad = (x2 <= 0) | (x2 >= 1)
    if bad.any():
        print(f"[henry_convert] WARNING: {bad.sum()} rows produced x2 outside (0,1) "
              f"-- will be dropped by build_dataset.py log transform check", flush=True)
        print(df[bad][["il_name", "value", "T_K"]].to_string(), flush=True)

    df["x2_CO2"]  = x2
    df["P_kPa"]   = P_REFERENCE_KPA   # reference pressure for Henry's law measurement
    df["T_K"]     = df["T_K"].astype(float)

    print(f"[henry_convert] Converted {len(df)} Henry rows to x2 "
          f"(range: {x2.min():.4f} – {x2.max():.4f})", flush=True)
    return df


def convert_mole_fraction_rows(df_mf: pd.DataFrame) -> pd.DataFrame:
    """
    Handle direct mole fraction rows (x2 already measured, P_kPa provided).
    Drop rows above P_MAX_KPA (supercritical / outside Henry's law linearity).
    """
    df = df_mf.copy()
    df["P_kPa"]  = pd.to_numeric(df["P_kPa"], errors="coerce")
    df["x2_CO2"] = pd.to_numeric(df["value"],  errors="coerce")
    df["T_K"]    = df["T_K"].astype(float)

    high_p = df["P_kPa"] > P_MAX_KPA
    if high_p.any():
        print(f"[mf_convert] Dropping {high_p.sum()} rows with P > {P_MAX_KPA} kPa",
              flush=True)
    df = df[~high_p].copy()

    print(f"[mf_convert] Kept {len(df)} direct mole fraction rows", flush=True)
    return df


def format_for_merge(df: pd.DataFrame, existing_cols: list) -> pd.DataFrame:
    """
    Build output rows matching the schema of all_co2_datapoints_merged.csv:
      il_name, il_smiles, T_K, P_kPa, x2_CO2, data_source, p_imputed
    """
    # Construct il_smiles from cation + anion SMILES (dot-separated, matching ILThermo format)
    df["il_smiles"] = df["cation_smiles"].str.strip() + "." + df["anion_smiles"].str.strip()

    output = pd.DataFrame({
        "il_name":     df["il_name"],
        "il_smiles":   df["il_smiles"],
        "T_K":         df["T_K"],
        "P_kPa":       df["P_kPa"],
        "x2_CO2":      df["x2_CO2"],
        "data_source": DATA_SOURCE_LABEL,
        "p_imputed":   0,   # pressure is real (either measured or set to P_ref for Henry)
    })

    # Add any extra columns in existing data that we don't have (fill with NaN)
    for col in existing_cols:
        if col not in output.columns:
            output[col] = np.nan

    return output[existing_cols]   # enforce column order


def check_for_duplicates(new_rows: pd.DataFrame,
                          existing_df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag any new IL SMILES that already appear in the existing dataset.
    These are likely ILThermo duplicates -- keep them but warn so they can
    be reviewed. Exact (il_smiles, T_K, P_kPa) duplicates are dropped.
    """
    existing_smiles = set(existing_df["il_smiles"].unique())
    new_smiles      = set(new_rows["il_smiles"].unique())
    overlap         = new_smiles & existing_smiles

    if overlap:
        print(f"\n[dup_check] WARNING: {len(overlap)} new IL SMILES already exist "
              f"in training data:", flush=True)
        for s in sorted(overlap):
            print(f"  {s}", flush=True)
        print("  These ILs won't add structural diversity but may add T/P coverage.",
              flush=True)

    # Drop exact (SMILES, T, P) duplicates
    merge_key = ["il_smiles", "T_K", "P_kPa"]
    before    = len(new_rows)
    combined  = pd.concat([existing_df[merge_key].assign(_existing=True),
                            new_rows[merge_key].assign(_existing=False)])
    exact_dups = combined.duplicated(subset=merge_key, keep="first")
    dup_smiles_tp = set(
        combined[exact_dups & ~combined["_existing"]]["il_smiles"]
    )
    new_rows = new_rows[~new_rows["il_smiles"].isin(dup_smiles_tp) |
                        ~new_rows.set_index(merge_key).index.isin(
                            existing_df.set_index(merge_key).index
                        )].copy()

    # Simpler approach: drop rows where (il_smiles, T_K, P_kPa) already exists
    existing_keys = set(
        zip(existing_df["il_smiles"], existing_df["T_K"], existing_df["P_kPa"])
    )
    new_rows["_key"] = list(zip(new_rows["il_smiles"], new_rows["T_K"], new_rows["P_kPa"]))
    exact_dup_mask   = new_rows["_key"].isin(existing_keys)
    if exact_dup_mask.any():
        print(f"[dup_check] Dropping {exact_dup_mask.sum()} exact (SMILES, T, P) "
              f"duplicates", flush=True)
    new_rows = new_rows[~exact_dup_mask].drop(columns=["_key"])

    print(f"[dup_check] {len(new_rows)} / {before} new rows are non-duplicate",
          flush=True)
    return new_rows


def main():
    """
    Full integration pipeline:
    1. Load existing dataset
    2. Parse Ramdin 2015 markdown table
    3. Apply exclusion rules
    4. Convert henry -> x2 and filter mole_fraction rows
    5. Duplicate check
    6. Append to existing dataset -> save expanded CSV
    """
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("=== RAMDIN 2015 LITERATURE DATA INTEGRATION ===\n", flush=True)

    # Step 1: Load existing data
    existing_df = load_existing_data(SOURCE_CSV)

    # Step 2: Parse literature markdown
    lit_df = parse_literature_md(LITERATURE_MD)

    # Step 3: Apply exclusion rules
    kept_df, excluded_df = apply_exclusion_rules(lit_df)

    # Step 4: Split by property type and convert
    henry_rows = kept_df[kept_df["property_type"] == "henry"].copy()
    mf_rows    = kept_df[kept_df["property_type"] == "mole_fraction"].copy()

    print(f"\n[split] {len(henry_rows)} henry rows, {len(mf_rows)} mole_fraction rows",
          flush=True)

    converted_rows = []
    if len(henry_rows):
        converted_rows.append(convert_henry_to_x2(henry_rows))
    if len(mf_rows):
        converted_rows.append(convert_mole_fraction_rows(mf_rows))

    if not converted_rows:
        print("[main] No usable rows after exclusions. Exiting.", flush=True)
        return

    new_data_df = pd.concat(converted_rows, ignore_index=True)

    # Step 5: Format to match existing schema
    new_formatted = format_for_merge(new_data_df, list(existing_df.columns))

    # Step 6: Duplicate check
    new_formatted = check_for_duplicates(new_formatted, existing_df)

    # Step 7: Append and save
    expanded_df = pd.concat([existing_df, new_formatted], ignore_index=True)

    new_ils = new_formatted["il_smiles"].nunique()
    print(f"\n=== INTEGRATION SUMMARY ===", flush=True)
    print(f"  Original rows:     {len(existing_df)}", flush=True)
    print(f"  New rows added:    {len(new_formatted)}", flush=True)
    print(f"  New unique ILs:    {new_ils}", flush=True)
    print(f"  Expanded rows:     {len(expanded_df)}", flush=True)
    print(f"  Expanded ILs:      {expanded_df['il_smiles'].nunique()}", flush=True)
    print(f"  Excluded rows:     {len(excluded_df)}", flush=True)

    expanded_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[main] Saved: {OUTPUT_CSV}", flush=True)

    # Save integration report for paper/audit trail
    report_rows = []
    for _, row in new_formatted.iterrows():
        report_rows.append({
            "il_smiles":    row["il_smiles"],
            "il_name":      row.get("il_name", ""),
            "T_K":          row["T_K"],
            "P_kPa":        row["P_kPa"],
            "x2_CO2":       row["x2_CO2"],
            "status":       "added",
        })
    if len(excluded_df):
        for _, row in excluded_df.iterrows():
            report_rows.append({
                "il_smiles":  row.get("cation_smiles","") + "." + row.get("anion_smiles",""),
                "il_name":    row.get("il_name", ""),
                "T_K":        row.get("T_K", ""),
                "P_kPa":      "N/A",
                "x2_CO2":     "N/A",
                "status":     row.get("exclusion_reason", "excluded"),
            })
    pd.DataFrame(report_rows).to_csv(REPORT_CSV, index=False)
    print(f"[main] Saved: {REPORT_CSV}", flush=True)

    print("\n[main] NEXT STEPS:", flush=True)
    print("  1. Save data.md as data/raw/ramdin_2015_literature.md", flush=True)
    print("  2. python src/featurize.py  (featurize any new IL SMILES)", flush=True)
    print("  3. Edit build_dataset.py: set DATAPOINTS_CSV to all_co2_datapoints_expanded.csv", flush=True)
    print("  4. python src/build_dataset.py", flush=True)
    print("  5. python src/compare_models.py  (check if test R² improved)", flush=True)
    print("[main] DONE.", flush=True)


if __name__ == "__main__":
    main()
