"""
featurize.py
------------
PURPOSE: Convert IL SMILES strings into numerical feature vectors for ML.

WHAT WE DO:
  ILThermo gives us a single SMILES string for each IL (e.g.
  "CCCCn1cc[n+](C)c1.F[B-](F)(F)F" for [BMIM][BF4]).
  Ionic liquid SMILES use a '.' to separate the cation from the anion.
  We split on '.' and featurize each ion separately, then concatenate.

  Each ion gets:
    - Morgan fingerprint (radius=2, 2048 bits): encodes circular substructure
      neighborhoods. Each bit = presence of a particular chemical environment.
    - 8 RDKit physicochemical descriptors: molecular weight, H-bond donors/
      acceptors, TPSA, rotatable bonds, aromatic rings, heavy atom count, logP.

  Cation features are prefixed 'cat_', anion features prefixed 'an_'.
  Total feature vector dimension: 2 x 2048 bits + 2 x 8 descriptors = 4112.

WHY SEPARATE CATION/ANION:
  CO2 absorption depends differently on cation vs anion structure.
  Treating them separately lets the model learn these independent contributions.
  This is the standard approach in IL property prediction literature.

INPUT:  data/raw/ilthermo_mole_fraction_datapoints.csv  (from fetch_datapoints.py)
OUTPUT: data/processed/il_features.csv  (one row per unique IL)

Run from project root:
    python src/featurize.py
"""

import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator  # modern API -- replaces deprecated AllChem.GetMorganFingerprintAsBitVect

# -- Constants -----------------------------------------------------------------
MORGAN_RADIUS  = 2      # each atom looks 2 bonds out -- standard for ML
MORGAN_NBITS   = 2048   # fingerprint vector length
INPUT_CSV      = os.path.join("data", "raw",       "ilthermo_mole_fraction_datapoints.csv")
OUTPUT_CSV     = os.path.join("data", "processed", "il_features.csv")

# Build the Morgan generator once at module level (requires RDKit >= 2022.03).
# This replaces AllChem.GetMorganFingerprintAsBitVect which is deprecated
# and will be removed in a future RDKit release.
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS,
    fpSize=MORGAN_NBITS,
)

# RDKit descriptors to compute per ion.
# Chosen for physical relevance to CO2 absorption and explainability to judges.
RDKIT_DESCRIPTORS = [
    ("mol_weight",          Descriptors.MolWt),
    ("num_hbd",             rdMolDescriptors.CalcNumHBD),           # H-bond donors
    ("num_hba",             rdMolDescriptors.CalcNumHBA),           # H-bond acceptors
    ("tpsa",                Descriptors.TPSA),                      # polarity proxy
    ("num_rotatable_bonds", rdMolDescriptors.CalcNumRotatableBonds),
    ("num_aromatic_rings",  rdMolDescriptors.CalcNumAromaticRings),
    ("num_heavy_atoms",     rdMolDescriptors.CalcNumHeavyAtoms),
    ("log_p",               Descriptors.MolLogP),                   # lipophilicity
]


def split_il_smiles(il_smiles: str) -> tuple[str, str]:
    """
    Split an ionic liquid SMILES into cation and anion SMILES.
    ILs in ILThermo are stored as 'cation_smiles.anion_smiles' -- the dot
    is the SMILES notation for disconnected fragments (salt).

    Returns (cation_smiles, anion_smiles).
    If splitting fails (wrong format), returns (il_smiles, "") and logs a warning.
    """
    if not isinstance(il_smiles, str) or not il_smiles.strip():
        return ("", "")

    fragments = il_smiles.split(".")

    if len(fragments) == 2:
        return (fragments[0], fragments[1])
    elif len(fragments) > 2:
        # Some ILs have more than 2 fragments (e.g. multi-component systems)
        print(f"  [NOTE] SMILES has {len(fragments)} fragments (expected 2): {il_smiles[:60]}")
        return (fragments[0], ".".join(fragments[1:]))
    else:
        # DATA QUALITY FLAG: only one fragment -- likely not a proper IL SMILES
        print(f"  [DATA QUALITY] Only 1 fragment in SMILES, cannot split: {il_smiles[:60]}")
        return (il_smiles, "")


def smiles_to_mol(smiles: str, label: str = "") -> Chem.Mol | None:
    """Parse SMILES string into an RDKit mol object. Returns None on failure."""
    if not smiles or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # DATA QUALITY FLAG: RDKit could not parse this SMILES
        print(f"  [DATA QUALITY] RDKit failed to parse SMILES for '{label}': {smiles[:60]}")
    return mol


def mol_to_morgan_fp(mol: Chem.Mol | None) -> np.ndarray:
    """
    Compute Morgan fingerprint bit vector using the modern rdFingerprintGenerator API.
    Returns all-zeros array if mol is None -- sentinel for 'no structural info'.
    GetFingerprintAsNumPy returns a numpy uint8 array directly, which is more
    efficient than the old BitVect-to-numpy conversion path.
    """
    if mol is None:
        return np.zeros(MORGAN_NBITS, dtype=np.int8)
    fp_array = MORGAN_GENERATOR.GetFingerprintAsNumPy(mol)   # direct numpy output
    return fp_array.astype(np.int8)


def mol_to_descriptors(mol: Chem.Mol | None, label: str = "") -> dict:
    """Compute RDKit physicochemical descriptors. Returns NaNs if mol is None."""
    result = {}
    for desc_name, desc_fn in RDKIT_DESCRIPTORS:
        if mol is None:
            result[desc_name] = np.nan
            continue
        try:
            result[desc_name] = desc_fn(mol)
        except Exception as err:
            print(f"  [DATA QUALITY] Descriptor '{desc_name}' failed for '{label}': {err}")
            result[desc_name] = np.nan
    return result


def featurize_one_il(il_smiles: str, il_name: str = "") -> dict:
    """
    Featurize one ionic liquid from its full SMILES string.
    Splits into cation/anion, computes fingerprints and descriptors for each,
    and returns one flat dict with all features prefixed cat_ or an_.
    """
    cation_smiles, anion_smiles = split_il_smiles(il_smiles)

    cation_mol = smiles_to_mol(cation_smiles, label=f"{il_name} cation")
    anion_mol  = smiles_to_mol(anion_smiles,  label=f"{il_name} anion")

    cation_fp   = mol_to_morgan_fp(cation_mol)
    anion_fp    = mol_to_morgan_fp(anion_mol)
    cation_desc = mol_to_descriptors(cation_mol, label=f"{il_name} cation")
    anion_desc  = mol_to_descriptors(anion_mol,  label=f"{il_name} anion")

    feature_dict = {
        "il_smiles":     il_smiles,
        "cation_smiles": cation_smiles,
        "anion_smiles":  anion_smiles,
    }

    # Fingerprint bits: cat_fp_0 ... cat_fp_2047, an_fp_0 ... an_fp_2047
    for i, bit in enumerate(cation_fp):
        feature_dict[f"cat_fp_{i}"] = int(bit)
    for i, bit in enumerate(anion_fp):
        feature_dict[f"an_fp_{i}"] = int(bit)

    # Descriptors: cat_mol_weight, cat_num_hbd, ..., an_mol_weight, ...
    for desc_name, val in cation_desc.items():
        feature_dict[f"cat_{desc_name}"] = val
    for desc_name, val in anion_desc.items():
        feature_dict[f"an_{desc_name}"] = val

    return feature_dict


def featurize_il_smiles(il_smiles: str) -> dict | None:
    """
    Public interface used by inverse_design.py.
    Wraps featurize_one_il with a None-return contract: returns None if the
    SMILES is empty or if both cation and anion fingerprints are all-zeros
    (both ions failed to parse), so the caller can skip cleanly.

    This is a named alias so that future refactoring of featurize_one_il
    doesn't silently break inverse_design.py's import.
    """
    if not il_smiles or not isinstance(il_smiles, str):
        return None

    result = featurize_one_il(il_smiles)

    # If both fingerprints are all-zeros, both ions failed to parse -- bad SMILES
    cat_sum = sum(v for k, v in result.items() if k.startswith("cat_fp_"))
    an_sum  = sum(v for k, v in result.items() if k.startswith("an_fp_"))
    if cat_sum == 0 and an_sum == 0:
        # DATA QUALITY FLAG: no structural bits extracted
        return None

    return result


def featurize_all_ils(datapoints_df: pd.DataFrame) -> pd.DataFrame:
    """
    Featurize all unique ILs in the datapoints DataFrame.
    Deduplicates by il_smiles first -- no point computing features twice
    for the same IL that appears in multiple (T, P, x2) rows.
    Returns one row per unique IL with all feature columns.
    """
    unique_ils = datapoints_df.drop_duplicates(subset=["il_smiles"]).copy()
    unique_ils = unique_ils[["il_name", "il_smiles"]].reset_index(drop=True)
    print(f"[featurize_all_ils] Featurizing {len(unique_ils)} unique ILs...")

    failed_count = 0
    feature_rows = []

    for idx, row in unique_ils.iterrows():
        il_smiles = str(row["il_smiles"]) if pd.notna(row["il_smiles"]) else ""
        il_name   = str(row["il_name"])

        if not il_smiles:
            # DATA QUALITY FLAG: no SMILES for this IL -- cannot featurize
            print(f"  [DATA QUALITY] No SMILES for '{il_name}' -- skipping")
            failed_count += 1
            continue

        features = featurize_one_il(il_smiles, il_name)
        features["il_name"] = il_name
        feature_rows.append(features)

        if (idx + 1) % 50 == 0:
            print(f"  -> {idx + 1}/{len(unique_ils)} ILs featurized")

    features_df = pd.DataFrame(feature_rows)
    print(f"[featurize_all_ils] Done: {len(features_df)} ILs featurized, {failed_count} skipped")
    print(f"  Feature columns: {len(features_df.columns)} total")
    return features_df


def main():
    """Main: load datapoints -> featurize unique ILs -> save."""
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"Input not found: {INPUT_CSV}. Run src/fetch_datapoints.py first."
        )

    datapoints_df = pd.read_csv(INPUT_CSV)
    print(f"[main] Loaded {len(datapoints_df)} rows, {datapoints_df['il_smiles'].nunique()} unique SMILES")

    features_df = featurize_all_ils(datapoints_df)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    features_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[main] Saved features -> {OUTPUT_CSV}")
    print(f"  Shape: {features_df.shape[0]} ILs x {features_df.shape[1]} columns")


if __name__ == "__main__":
    main()
