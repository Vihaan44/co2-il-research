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

# ILThermo column names returned by ILThermoPy GetEntry — these vary slightly
# by dataset but these are the standard names for binary VLE data
COL_TEMPERATURE  = "Temperature, K"
COL_PRESSURE     = "Pressure, kPa"
COL_MOLE_FRAC    = "Mole fraction of CO2"   # target variable x2

# Physical sanity bounds — flag but don't drop automatically
T_MIN_K   = 200.0
T_MAX_K   = 450.0
P_MIN_KPA = 0.0
P_MAX_KPA = 20000.0


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
    print(f"[load_metadata] Columns: {list(metadata_df.columns)}")
    return metadata_df


def fetch_entry_datapoints(entry_id: str, il_name: str, il_smiles: str) -> list[dict]:
    """
    Fetch all (T, P, x2) measurement rows for one ILThermo entry_id using
    ILThermoPy's GetEntry function.

    Returns a list of flat dicts, each representing one measurement point.
    Returns empty list if fetch fails or entry has no usable data.

    We carry il_name, il_smiles, and entry_id forward into every row so we
    can always trace a data point back to its source — essential for debugging
    and for the featurization step that needs the SMILES.
    """
    try:
        entry = ilt.GetEntry(entry_id)  # returns an ILThermoPy Entry object
    except Exception as fetch_error:
        print(f"  [WARNING] GetEntry failed for entry_id={entry_id}: {fetch_error}")
        return []

    # ILThermoPy Entry has a .data attribute which is a pandas DataFrame
    # of the actual measurement rows for this entry
    if entry is None or not hasattr(entry, 'data') or entry.data is None:
        print(f"  [DATA QUALITY] entry_id={entry_id} returned no data object")
        return []

    entry_data_df = entry.data

    if entry_data_df.empty:
        print(f"  [DATA QUALITY] entry_id={entry_id} has empty data table")
        return []

    print(f"  entry_id={entry_id}: {len(entry_data_df)} rows, columns={list(entry_data_df.columns)}")

    # ── Find T, P, x2 columns ────────────────────────────────────────────────
    # ILThermo column names can vary slightly — try multiple known variants
    temp_col  = _find_column(entry_data_df, ["Temperature, K", "T/K", "Temperature (K)", "T, K"])
    press_col = _find_column(entry_data_df, ["Pressure, kPa", "p/kPa", "Pressure (kPa)", "P, kPa"])
    x2_col    = _find_column(entry_data_df, [
        "Mole fraction of CO2", "x(CO2)", "x_2", "Mole fraction",
        "x2", "Mole fraction 2", "w(CO2)"
    ])

    if temp_col is None or x2_col is None:
        # DATA QUALITY FLAG: can't use this entry without T and x2
        print(f"  [DATA QUALITY] entry_id={entry_id}: missing T or x2 column. "
              f"Available: {list(entry_data_df.columns)}")
        return []

    # Build flat row dicts
    row_dicts = []
    for _, data_row in entry_data_df.iterrows():
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


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Return the first column name from 'candidates' that exists in df.
    Returns None if none match. Used to handle ILThermo's inconsistent
    column naming across different dataset entries.
    """
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def fetch_all_datapoints(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Loop over all entry_ids, fetch their measurement rows, and concatenate
    into one flat DataFrame. Prints progress every 25 entries.
    """
    all_rows   = []
    total      = len(metadata_df)
    failed     = 0

    for idx, meta_row in metadata_df.iterrows():
        entry_id  = str(meta_row["entry_id"])
        il_name   = meta_row.get("il_name",   "unknown")
        il_smiles = meta_row.get("il_smiles", None)

        print(f"[{idx + 1}/{total}] Fetching entry_id={entry_id} ({il_name})")

        points = fetch_entry_datapoints(entry_id, il_name, str(il_smiles) if il_smiles else "")

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
    Cast T/P/x2 to numeric, drop rows missing T or x2, and flag
    physically unreasonable values. Does NOT silently drop outliers —
    flags them so Vihaan can inspect and decide.
    """
    print(f"\n[clean_datapoints] Starting with {len(raw_df)} rows")

    # Cast to numeric — coerce unparseable strings to NaN
    for col in ["T_K", "P_kPa", "x2_CO2"]:
        if col in raw_df.columns:
            raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

    # Drop rows missing T or x2 — these are unusable for ML
    before = len(raw_df)
    raw_df.dropna(subset=["T_K", "x2_CO2"], inplace=True)
    print(f"[clean_datapoints] Dropped {before - len(raw_df)} rows missing T or x2")

    # Flag physically unreasonable x2 values (must be 0-1, it's a mole fraction)
    bad_x2 = raw_df[(raw_df["x2_CO2"] < 0) | (raw_df["x2_CO2"] > 1)]
    if len(bad_x2) > 0:
        # DATA QUALITY FLAG — don't silently drop; show Vihaan what's wrong
        print(f"[DATA QUALITY] {len(bad_x2)} rows have x2_CO2 outside [0,1]:")
        print(bad_x2[["entry_id", "il_name", "T_K", "x2_CO2"]].head(10).to_string())

    # Flag unusual temperatures
    bad_T = raw_df[(raw_df["T_K"] < T_MIN_K) | (raw_df["T_K"] > T_MAX_K)]
    if len(bad_T) > 0:
        print(f"[DATA QUALITY] {len(bad_T)} rows have T_K outside [{T_MIN_K}, {T_MAX_K}]")
        print(bad_T[["entry_id", "il_name", "T_K"]].head(5).to_string())

    print(f"[clean_datapoints] Final: {len(raw_df)} clean rows")
    return raw_df


def print_summary(df: pd.DataFrame) -> None:
    """Print dataset summary so Vihaan can spot problems before saving."""
    print("\n── DATAPOINTS SUMMARY ───────────────────────────────────────────────")
    print(f"  Total rows          : {len(df)}")
    print(f"  Unique ILs          : {df['il_name'].nunique()}")
    print(f"  Unique entry_ids    : {df['entry_id'].nunique()}")
    if "T_K" in df.columns:
        print(f"  T_K range           : {df['T_K'].min():.1f} – {df['T_K'].max():.1f} K")
    if "P_kPa" in df.columns:
        print(f"  P_kPa range         : {df['P_kPa'].min():.1f} – {df['P_kPa'].max():.1f} kPa")
    if "x2_CO2" in df.columns:
        print(f"  x2_CO2 range        : {df['x2_CO2'].min():.4f} – {df['x2_CO2'].max():.4f}")
    print(f"\n  Top 10 ILs by datapoint count:")
    print(df["il_name"].value_counts().head(10).to_string())
    print("─────────────────────────────────────────────────────────────────────\n")


def main():
    """Main: load metadata → fetch datapoints → clean → save."""
    metadata_df   = load_metadata(INPUT_CSV)
    raw_df        = fetch_all_datapoints(metadata_df)

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
