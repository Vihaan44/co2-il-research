"""
thermoml_inchi_to_smiles.py
----------------------------
PURPOSE: Convert InChI strings from the ThermoML parsed data to SMILES
         strings using the PubChem API, so the new ILs can be featurized
         by featurize.py and merged into the training pipeline.

WHY InChI → SMILES VIA PUBCHEM:
  ThermoML uses InChI as the canonical chemical identifier, not SMILES.
  Our featurize.py pipeline requires SMILES (for RDKit Morgan fingerprints).
  PubChem's REST API converts InChI → canonical SMILES reliably for most
  common ILs. For ILs not in PubChem, we fall back to RDKit's own InChI
  parser, which handles most standard structures.

RATE LIMITING:
  PubChem allows ~5 requests/second. We use a 0.25s delay between calls
  and batch requests where possible. For 126 unique ILs, total runtime
  is ~2-3 minutes.

OUTPUTS:
  data/raw/thermoml_inchi_smiles_map.csv  -- InChI → SMILES mapping
  data/raw/thermoml_co2_il_with_smiles.csv -- full dataset with SMILES added
  data/raw/thermoml_smiles_failures.txt   -- ILs where conversion failed

INPUT:
  data/raw/thermoml_co2_il_raw.csv  (from parse_thermoml.py)

Run from project root:
    python src/thermoml_inchi_to_smiles.py
"""

import os
import sys
import time
import requests
import pandas as pd
import numpy as np
from rdkit import Chem

sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)

# -- Constants -----------------------------------------------------------------
INPUT_CSV      = os.path.join("data", "raw", "thermoml_co2_il_raw.csv")
SMILES_MAP_CSV = os.path.join("data", "raw", "thermoml_inchi_smiles_map.csv")
OUTPUT_CSV     = os.path.join("data", "raw", "thermoml_co2_il_with_smiles.csv")
FAILURES_FILE  = os.path.join("data", "raw", "thermoml_smiles_failures.txt")

# PubChem REST API endpoint for InChI → SMILES conversion
PUBCHEM_INCHI_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchi/property/IsomericSMILES,CanonicalSMILES/JSON"

API_PAUSE_SECONDS = 0.25   # stay well under PubChem's rate limit
MAX_RETRIES       = 3      # retry failed API calls this many times
RETRY_WAIT        = 2.0    # seconds to wait between retries


def inchi_to_smiles_pubchem(inchi: str) -> str | None:
    """
    Query PubChem REST API to convert an InChI string to canonical SMILES.
    Returns canonical SMILES string, or None if conversion fails.

    PubChem is preferred over RDKit's InChI parser because it handles
    ionic liquid salts (cation + anion pairs) better -- RDKit often
    fails on multi-component InChI strings like [BMIM][BF4].
    """
    if not inchi or not isinstance(inchi, str) or not inchi.startswith("InChI="):
        return None

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                PUBCHEM_INCHI_URL,
                data={"inchi": inchi},
                timeout=10,
            )
            if response.status_code == 200:
                data = response.json()
                props = data.get("PropertyTable", {}).get("Properties", [])
                if props:
                    # Prefer canonical SMILES over isomeric for consistency
                    smiles = props[0].get("CanonicalSMILES") or \
                             props[0].get("IsomericSMILES")
                    return smiles
            elif response.status_code == 404:
                return None  # compound not in PubChem -- no point retrying
            else:
                time.sleep(RETRY_WAIT)
        except requests.RequestException:
            time.sleep(RETRY_WAIT)

    return None


def inchi_to_smiles_rdkit(inchi: str) -> str | None:
    """
    Fallback: use RDKit's built-in InChI parser to convert InChI → SMILES.
    Less reliable than PubChem for ionic compounds but works for many ILs.
    Returns canonical SMILES or None if parsing fails.
    """
    if not inchi or not inchi.startswith("InChI="):
        return None
    try:
        mol = Chem.inchi.MolFromInchi(inchi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def validate_smiles(smiles: str) -> bool:
    """
    Return True if a SMILES string is valid (parseable by RDKit).
    Invalid SMILES would cause featurize.py to silently skip the IL.
    """
    if not smiles or not isinstance(smiles, str):
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def convert_all_inchis(unique_inchis: list) -> dict:
    """
    Convert all unique InChI strings to SMILES.
    Tries PubChem first, falls back to RDKit.
    Returns dict mapping inchi → smiles (or None if both methods fail).
    """
    inchi_to_smiles = {}
    n_pubchem_success = 0
    n_rdkit_fallback  = 0
    n_failed          = 0

    print(f"[convert] Converting {len(unique_inchis)} unique InChIs to SMILES...",
          flush=True)

    for i, inchi in enumerate(unique_inchis):
        if not inchi or not isinstance(inchi, str):
            inchi_to_smiles[inchi] = None
            n_failed += 1
            continue

        # Try PubChem first
        smiles = inchi_to_smiles_pubchem(inchi)
        time.sleep(API_PAUSE_SECONDS)

        if smiles and validate_smiles(smiles):
            inchi_to_smiles[inchi] = smiles
            n_pubchem_success += 1
        else:
            # Fall back to RDKit
            smiles_rdkit = inchi_to_smiles_rdkit(inchi)
            if smiles_rdkit and validate_smiles(smiles_rdkit):
                inchi_to_smiles[inchi] = smiles_rdkit
                n_rdkit_fallback += 1
                print(f"  [rdkit fallback] InChI {i+1}: {inchi[:60]}...",
                      flush=True)
            else:
                inchi_to_smiles[inchi] = None
                n_failed += 1
                print(f"  [FAILED] InChI {i+1}: {inchi[:60]}...", flush=True)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(unique_inchis)}] done | "
                  f"PubChem: {n_pubchem_success} | "
                  f"RDKit: {n_rdkit_fallback} | "
                  f"Failed: {n_failed}", flush=True)

    print(f"\n[convert] Conversion complete:", flush=True)
    print(f"  PubChem success: {n_pubchem_success}", flush=True)
    print(f"  RDKit fallback:  {n_rdkit_fallback}", flush=True)
    print(f"  Failed:          {n_failed}", flush=True)

    return inchi_to_smiles


def main():
    """Main: load raw data → convert InChI → SMILES → merge → save."""
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)

    # -- Step 1: Load parsed ThermoML data -----------------------------------
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"Input not found at {INPUT_CSV}. Run parse_thermoml.py first."
        )
    df = pd.read_csv(INPUT_CSV)
    print(f"[main] Loaded {len(df)} rows, {df['il_inchi'].nunique()} unique ILs",
          flush=True)

    # -- Step 2: Convert all unique InChIs -----------------------------------
    unique_inchis = df["il_inchi"].dropna().unique().tolist()
    inchi_smiles_map = convert_all_inchis(unique_inchis)

    # -- Step 3: Save InChI → SMILES mapping ---------------------------------
    map_df = pd.DataFrame([
        {"il_inchi": inchi, "il_smiles": smiles, "smiles_valid": smiles is not None}
        for inchi, smiles in inchi_smiles_map.items()
    ])
    map_df.to_csv(SMILES_MAP_CSV, index=False)
    print(f"\n[main] SMILES map saved → {SMILES_MAP_CSV}", flush=True)

    # -- Step 4: Merge SMILES into main dataframe ----------------------------
    df["il_smiles"] = df["il_inchi"].map(inchi_smiles_map)

    before = len(df)
    df_valid = df[df["il_smiles"].notna()].copy()
    n_dropped = before - len(df_valid)
    print(f"[main] Dropped {n_dropped} rows with no SMILES conversion",
          flush=True)
    print(f"[main] Remaining: {len(df_valid)} rows, "
          f"{df_valid['il_smiles'].nunique()} unique ILs", flush=True)

    # -- Step 5: Save failures for manual review -----------------------------
    failed_ils = map_df[~map_df["smiles_valid"]][["il_inchi"]].copy()
    failed_ils = failed_ils.merge(
        df[["il_name", "il_inchi"]].drop_duplicates(),
        on="il_inchi", how="left"
    )
    with open(FAILURES_FILE, "w") as f:
        f.write(f"ILs where InChI → SMILES conversion failed ({len(failed_ils)} total):\n\n")
        for _, row in failed_ils.iterrows():
            f.write(f"  {row.get('il_name', 'Unknown')}\n")
            f.write(f"  {row['il_inchi']}\n\n")
    print(f"[main] Failures saved → {FAILURES_FILE}", flush=True)

    # -- Step 6: Final output ------------------------------------------------
    df_valid.to_csv(OUTPUT_CSV, index=False)
    print(f"[main] Output saved → {OUTPUT_CSV}", flush=True)

    # Show what we have
    print(f"\n=== THERMOML DATA READY FOR MERGE ===", flush=True)
    print(f"  Rows:        {len(df_valid)}", flush=True)
    print(f"  Unique ILs:  {df_valid['il_smiles'].nunique()}", flush=True)
    print(f"\n  Top 10 ILs:", flush=True)
    print(df_valid.groupby("il_name").size().sort_values(
        ascending=False).head(10).to_string(), flush=True)
    print(f"\n[main] NEXT STEP:", flush=True)
    print(f"  Run src/merge_thermoml_into_pipeline.py to combine with", flush=True)
    print(f"  data/raw/ilthermo_mole_fraction_datapoints.csv and rebuild", flush=True)
    print(f"  the training set.", flush=True)


if __name__ == "__main__":
    main()
