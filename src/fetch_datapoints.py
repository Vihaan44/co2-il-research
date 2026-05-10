"""
fetch_datapoints.py
-------------------
PURPOSE: For each entry_id in ilthermo_mole_fraction_raw.csv, fetch the actual
         experimental measurement rows (T in K, P in kPa, x2 = CO2 mole fraction)
         using ILThermoPy and save them as one flat CSV.

WHY THIS STEP:
  Phase 1 gave us metadata (one row per dataset entry). This script fetches the
  actual numbers inside each dataset entry. One entry might contain 10-50 individual
  (T, P, x2) measurements at different conditions.

HOW ILTHERMOPY WORKS:
  entry.data has generic columns V1, V2, V3, dV3.
  entry.header is a dict mapping V1 -> "Temperature, K", V2 -> "Pressure, kPa", etc.
  We rename using the header, then identify T/P/x2 by substring matching the
  descriptive names (since exact strings vary slightly by dataset).

INPUT:  data/raw/ilthermo_mole_fraction_raw.csv  (from Phase 1)
OUTPUT: data/raw/ilthermo_mole_fraction_datapoints.csv

Run from project root:
    python src/fetch_datapoints.py
"""

import ilthermopy as ilt
import pandas as pd
import os

# ── Constants ──────────────────────────────────────────────────────────────────
INPUT_CSV  = os.path.join("data", "raw", "ilthermo_mole_fraction_raw.csv")
OUTPUT_CSV = os.path.join("data", "raw", "ilthermo_mole_fraction_datapoints.csv")

# Substrings to match against header values (lowercase) to identify each column.
# We use substring matching because ILThermo header strings vary slightly:
# e.g. "Mole fraction of carbon dioxide => Liquid" vs "Mole fraction of CO2"
TEMP_SUBSTRINGS   = ["temperature"]
PRESS_SUBSTRINGS  = ["pressure"]
X2_SUBSTRINGS     = ["mole fraction"]   # mole fraction always contains this

# Physical sanity bounds — flag but don't drop automatically
T_MIN_K   = 200.0
T_MAX_K   = 450.0


def load_metadata(csv_path: str) -> pd.DataFrame:
    """
    Load the Phase 1 metadata CSV. Each row is one ILThermo dataset entry
    with an entry_id we'll use to fetch the actual measurement data.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Phase 1 CSV not found at {csv_path}. "
            f"Run src/fetch_mole_fraction_data.py first."
        )
    metadata_df = pd.read_csv(csv_path)
    print(f"[load_metadata] {len(metadata_df)} entries, {metadata_df['il_name'].nunique()} unique ILs")
    return metadata_df


def rename_columns_from_header(data_df: pd.DataFrame, header: dict) -> pd.DataFrame:
    """
    Rename ILThermoPy's generic V1/V2/V3 columns to their descriptive names
    using the entry.header dict (e.g. {'V1': 'Temperature, K', 'V2': 'Pressure, kPa'}).
    Returns the renamed DataFrame.
    """
    # header maps generic name -> descriptive string; invert for rename
    rename_map = {generic: descriptive for generic, descriptive in header.items()}
    return data_df.rename(columns=rename_map)


def find_column_by_substring(df: pd.DataFrame, substrings: list[str]) -> str | None:
    """
    Find the first column in df whose name contains any of the given substrings
    (case-insensitive). Returns the column name or None if not found.

    We use substring matching instead of exact matching because ILThermo header
    strings vary across entries (e.g. "Mole fraction of carbon dioxide => Liquid").
    """
    for col in df.columns:
        col_lower = col.lower()
        if any(sub in col_lower for sub in substrings):
            return col
    return None


def fetch_entry_datapoints(entry_id: str, il_name: str, il_smiles: str) -> list[dict]:
    """
    Fetch all (T, P, x2) measurement rows for one ILThermo entry_id.
    Returns a list of flat dicts (one per measurement), or empty list on failure.

    Key fix: rename V1/V2/V3 using entry.header before finding T/P/x2 columns.
    """
    try:
        entry = ilt.GetEntry(entry_id)
    except Exception as fetch_error:
        print(f"  [WARNING] GetEntry failed for entry_id={entry_id}: {fetch_error}")
        return []

    if entry is None or not hasattr(entry, 'data') or entry.data is None:
        print(f"  [DATA QUALITY] entry_id={entry_id} returned no data object")
        return []

    if entry.data.empty:
        print(f"  [DATA QUALITY] entry_id={entry_id} has empty data table")
        return []

    # Rename V1/V2/V3 → descriptive names using entry.header
    renamed_df = rename_columns_from_header(entry.data, entry.header)

    # Find T, P, x2 columns by substring matching header descriptions
    temp_col  = find_column_by_substring(renamed_df, TEMP_SUBSTRINGS)
    press_col = find_column_by_substring(renamed_df, PRESS_SUBSTRINGS)
    x2_col    = find_column_by_substring(renamed_df, X2_SUBSTRINGS)

    if temp_col is None or x2_col is None:
        # DATA QUALITY FLAG: entry unusable without T and x2
        print(f"  [DATA QUALITY] entry_id={entry_id}: could not find T or x2 column.")
        print(f"    Header: {entry.header}")
        print(f"    Renamed columns: {list(renamed_df.columns)}")
        return []

    # Build one dict per measurement row
    row_dicts = []
    for _, data_row in renamed_df.iterrows():
        point = {
            "entry_id":  entry_id,
            "il_name":   il_name,
            "il_smiles": il_smiles,
            "T_K":       data_row.get(temp_col),
            "P_kPa":     data_row.get(press_col) if press_col else None,
            "x2_CO2":    data_row.get(x2_col),
        }
        row_dicts.append(point)

    return row_dicts


def fetch_all_datapoints(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Loop over all entry_ids, fetch measurement rows, concatenate into one DataFrame.
    Prints progress every 25 entries.
    """
    all_rows = []
    total    = len(metadata_df)
    failed   = 0

    for idx, meta_row in metadata_df.iterrows():
        entry_id  = str(meta_row["entry_id"])
        il_name   = meta_row.get("il_name",   "unknown")
        il_smiles = meta_row.get("il_smiles", None)

        print(f"[{idx + 1}/{total}] entry_id={entry_id} ({il_name})")

        points = fetch_entry_datapoints(
            entry_id, il_name,
            str(il_smiles) if pd.notna(il_smiles) else ""
        )

        if not points:
            failed += 1
            continue

        all_rows.extend(points)

        if (idx + 1) % 25 == 0:
            print(f"  → {len(all_rows)} total datapoints so far")

    print(f"\n[fetch_all_datapoints] Done.")
    print(f"  Fetched {len(all_rows)} datapoints from {total - failed}/{total} entries")
    print(f"  Failed/empty entries: {failed}")

    return pd.DataFrame(all_rows)


def clean_datapoints(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast T/P/x2 to numeric, drop rows missing T or x2, flag physical outliers.
    Does NOT silently remove outliers — flags them for Vihaan to inspect.
    """
    print(f"\n[clean_datapoints] Starting with {len(raw_df)} rows")

    for col in ["T_K", "P_kPa", "x2_CO2"]:
        if col in raw_df.columns:
            raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

    before = len(raw_df)
    raw_df.dropna(subset=["T_K", "x2_CO2"], inplace=True)
    print(f"[clean_datapoints] Dropped {before - len(raw_df)} rows missing T or x2")

    # Flag x2 outside [0, 1] — physically impossible for a mole fraction
    bad_x2 = raw_df[(raw_df["x2_CO2"] < 0) | (raw_df["x2_CO2"] > 1)]
    if len(bad_x2) > 0:
        print(f"[DATA QUALITY] {len(bad_x2)} rows have x2_CO2 outside [0,1]:")
        print(bad_x2[["entry_id", "il_name", "T_K", "x2_CO2"]].head(10).to_string())

    # Flag unusual temperatures
    bad_T = raw_df[(raw_df["T_K"] < T_MIN_K) | (raw_df["T_K"] > T_MAX_K)]
    if len(bad_T) > 0:
        print(f"[DATA QUALITY] {len(bad_T)} rows have T_K outside [{T_MIN_K}, {T_MAX_K}] K")

    print(f"[clean_datapoints] Final: {len(raw_df)} clean rows")
    return raw_df


def print_summary(df: pd.DataFrame) -> None:
    """Print dataset summary so Vihaan can spot problems before saving."""
    print("\n── DATAPOINTS SUMMARY ───────────────────────────────────────────────")
    print(f"  Total rows       : {len(df)}")
    print(f"  Unique ILs       : {df['il_name'].nunique()}")
    print(f"  Unique entry_ids : {df['entry_id'].nunique()}")
    if "T_K" in df.columns:
        print(f"  T_K range        : {df['T_K'].min():.1f} – {df['T_K'].max():.1f} K")
    if "P_kPa" in df.columns:
        print(f"  P_kPa range      : {df['P_kPa'].min():.1f} – {df['P_kPa'].max():.1f} kPa")
    if "x2_CO2" in df.columns:
        print(f"  x2_CO2 range     : {df['x2_CO2'].min():.4f} – {df['x2_CO2'].max():.4f}")
    print(f"\n  Top 10 ILs by datapoint count:")
    print(df["il_name"].value_counts().head(10).to_string())
    print("─────────────────────────────────────────────────────────────────────\n")


def main():
    """Main: load metadata → fetch datapoints → clean → save."""
    metadata_df = load_metadata(INPUT_CSV)
    raw_df      = fetch_all_datapoints(metadata_df)

    if raw_df.empty:
        print("[ERROR] No datapoints fetched. Check ILThermoPy and entry_ids.")
        return

    clean_df = clean_datapoints(raw_df)
    print_summary(clean_df)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    clean_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[main] Saved {len(clean_df)} rows → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
