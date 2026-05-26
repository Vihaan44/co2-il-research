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
    - 18 RDKit physicochemical descriptors: molecular weight, H-bond donors/
      acceptors, TPSA, rotatable bonds, aromatic rings, heavy atom count, logP,
      plus: ring count, fraction of SP3 carbons, max/min partial charge,
      number of heteroatoms, number of radical electrons, Hall-Kier alpha,
      Ipc (information content), and number of stereocenters.

  Cation features are prefixed 'cat_', anion features prefixed 'an_'.
  Total feature vector dimension: 2 x 2048 bits + 2 x 18 descriptors = 4132.

  DESCRIPTOR ADDITIONS vs v1 (8 descriptors):
    ring_count          -- fused/multiple rings affect IL packing and CO2 cavity formation
    fraction_csp3       -- SP3 fraction correlates with chain flexibility and free volume
    max_partial_charge  -- highest partial charge on any atom; proxy for ion-CO2 electrostatics
    min_partial_charge  -- most negative site; directly relevant to Lewis acid-base CO2 binding
    num_heteroatoms     -- N/O/S count; heteroatoms are primary CO2 interaction sites
    num_stereocenters   -- structural complexity; may correlate with packing geometry
    hall_kier_alpha     -- molecular refractivity proxy; correlates with polarizability / CO2 affinity
    ipc                 -- information content of graph topology
    num_valence_electrons -- total electron count; relevant to electronic CO2 interaction
    num_radical_electrons -- almost always 0 for ILs, but flags unusual structures

WHY SEPARATE CATION/ANION:
  CO2 absorption depends differently on cation vs anion structure.
  Treating them separately lets the model learn these independent contributions.
  This is the standard approach in IL property prediction literature.

INPUT:  data/raw/all_co2_datapoints_merged.csv  (ILThermo + ThermoML merged)
OUTPUT: data/processed/il_features.csv  (one row per unique IL)

NOTE: INPUT_CSV was updated from ilthermo_mole_fraction_datapoints.csv to
      all_co2_datapoints_merged.csv so that the 88 new ThermoML ILs are
      featurized alongside the original ILThermo ILs.

IMPORTANT: After running this script, re-run build_dataset.py to regenerate
      train_set.csv and test_set.csv with the expanded feature set before
      running train_model.py or tune_hyperparameters.py.

Run from project root:
    python src/featurize.py
"""

import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator  # modern API -- replaces deprecated AllChem.GetMorganFingerprintAsBitVect
from rdkit.Chem import Crippen

# -- Constants -----------------------------------------------------------------
MORGAN_RADIUS  = 2      # each atom looks 2 bonds out -- standard for ML
MORGAN_NBITS   = 2048   # fingerprint vector length

# UPDATED: now reads from merged ILThermo + ThermoML dataset
INPUT_CSV      = os.path.join("data", "raw",       "all_co2_datapoints_merged.csv")
OUTPUT_CSV     = os.path.join("data", "processed", "il_features.csv")

# Build the Morgan generator once at module level (requires RDKit >= 2022.03).
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS,
    fpSize=MORGAN_NBITS,
)

# RDKit descriptors to compute per ion.
# v2: expanded from 8 to 18 descriptors to better capture CO2-interaction physics.
# Each descriptor is a (name, callable) pair.
#
# PHYSICAL RATIONALE FOR EACH ADDITION:
#   ring_count       -- aromatic/aliphatic rings create rigid cavities that affect
#                       CO2 free-volume solubility; imidazolium rings dominate in training set
#   fraction_csp3    -- long alkyl chains (high SP3) increase free volume and physical CO2
#                       solubility; also correlates with viscosity
#   max_partial_charge -- Gasteiger charge max; high positive charge on cation drives
#                         quadrupole interaction with CO2 oxygen
#   min_partial_charge -- most negative site; anion charge concentration governs
#                         Lewis base / chemical CO2 binding for amino-ILs
#   num_heteroatoms  -- N, O, S atoms are the primary CO2 binding sites in chemical
#                       absorption ILs (e.g., [NH2]-functionalized cations)
#   num_stereocenters -- structural asymmetry; affects packing and liquid structure
#   hall_kier_alpha  -- molecular refractivity correction; correlates with polarizability
#                       which drives van der Waals CO2-IL interaction
#   ipc              -- Bonchev-Trinajstic information content; structural complexity metric
#   num_valence_electrons -- total pi/lone-pair electrons; relevant to CO2 quadrupole interactions
#   num_radical_electrons -- flags unusual open-shell structures (should be 0 for normal ILs)
RDKIT_DESCRIPTORS = [
    # --- Original 8 descriptors ---
    ("mol_weight",             Descriptors.MolWt),
    ("num_hbd",                rdMolDescriptors.CalcNumHBD),           # H-bond donors
    ("num_hba",                rdMolDescriptors.CalcNumHBA),           # H-bond acceptors
    ("tpsa",                   Descriptors.TPSA),                      # polarity proxy
    ("num_rotatable_bonds",    rdMolDescriptors.CalcNumRotatableBonds),
    ("num_aromatic_rings",     rdMolDescriptors.CalcNumAromaticRings),
    ("num_heavy_atoms",        rdMolDescriptors.CalcNumHeavyAtoms),
    ("log_p",                  Descriptors.MolLogP),                   # lipophilicity
    # --- New in v2: 10 additional CO2-relevant descriptors ---
    ("ring_count",             rdMolDescriptors.CalcNumRings),         # total ring count (aromatic + aliphatic)
    ("fraction_csp3",         rdMolDescriptors.CalcFractionCSP3),     # fraction of SP3 carbons (chain flexibility)
    ("max_partial_charge",    Descriptors.MaxPartialCharge),           # Gasteiger max charge (electrostatics)
    ("min_partial_charge",    Descriptors.MinPartialCharge),           # Gasteiger min charge (anion binding site)
    ("num_heteroatoms",       rdMolDescriptors.CalcNumHeteroatoms),    # N/O/S count (CO2 binding sites)
    ("num_stereocenters",     rdMolDescriptors.CalcNumAtomStereoCenters), # structural asymmetry
    ("hall_kier_alpha",       Descriptors.HallKierAlpha),              # polarizability proxy
    ("ipc",                   Descriptors.Ipc),                        # graph information content
    ("num_valence_electrons",  Descriptors.NumValenceElectrons),       # total valence electron count
    ("num_radical_electrons",  Descriptors.NumRadicalElectrons),       # should be 0 for normal ILs
]


def split_il_smiles(il_smiles: str) -> tuple[str, str]:
    """
    Split an ionic liquid SMILES into cation and anion SMILES.
    ILs are stored as 'cation_smiles.anion_smiles' -- the dot
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
    """
    if mol is None:
        return np.zeros(MORGAN_NBITS, dtype=np.int8)
    fp_array = MORGAN_GENERATOR.GetFingerprintAsNumPy(mol)   # direct numpy output
    return fp_array.astype(np.int8)


def mol_to_descriptors(mol: Chem.Mol | None, label: str = "") -> dict:
    """
    Compute 18 RDKit physicochemical descriptors. Returns NaNs if mol is None.
    Gasteiger charges are computed once per mol (required for MaxPartialCharge /
    MinPartialCharge -- these descriptors need explicit Hs and computed charges).
    """
    result = {}

    if mol is not None:
        # Compute Gasteiger partial charges in-place.
        # Required before calling MaxPartialCharge / MinPartialCharge.
        # We add explicit Hs first because Gasteiger needs them for accurate charges,
        # then remove them so downstream descriptors (MW, etc.) aren't affected.
        try:
            mol_with_hs = Chem.AddHs(mol)
            from rdkit.Chem import AllChem
            AllChem.ComputeGasteigerCharges(mol_with_hs)
            # Extract max/min charge from the mol_with_hs before removing Hs
            charges = [mol_with_hs.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge')
                       for i in range(mol_with_hs.GetNumAtoms())
                       if not np.isnan(mol_with_hs.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge'))]
            gasteiger_max = float(max(charges)) if charges else np.nan
            gasteiger_min = float(min(charges)) if charges else np.nan
        except Exception as err:
            print(f"  [DATA QUALITY] Gasteiger charge failed for '{label}': {err}")
            gasteiger_max = np.nan
            gasteiger_min = np.nan
    else:
        gasteiger_max = np.nan
        gasteiger_min = np.nan

    for desc_name, desc_fn in RDKIT_DESCRIPTORS:
        if mol is None:
            result[desc_name] = np.nan
            continue
        # Gasteiger charge descriptors are pre-computed above; skip RDKit's built-in
        # version which would require re-adding Hs (slower and inconsistent).
        if desc_name == "max_partial_charge":
            result[desc_name] = gasteiger_max
            continue
        if desc_name == "min_partial_charge":
            result[desc_name] = gasteiger_min
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
    """
    if not il_smiles or not isinstance(il_smiles, str):
        return None

    result = featurize_one_il(il_smiles)

    # If both fingerprints are all-zeros, both ions failed to parse -- bad SMILES
    cat_sum = sum(v for k, v in result.items() if k.startswith("cat_fp_"))
    an_sum  = sum(v for k, v in result.items() if k.startswith("an_fp_"))
    if cat_sum == 0 and an_sum == 0:
        return None

    return result


def featurize_all_ils(datapoints_df: pd.DataFrame) -> pd.DataFrame:
    """
    Featurize all unique ILs in the datapoints DataFrame.
    Deduplicates by il_smiles first -- no point computing features twice
    for the same IL that appears in multiple (T, P, x2) rows.
    Returns one row per unique IL with all feature columns.
    """
    name_col = "il_name" if "il_name" in datapoints_df.columns else "il_smiles"

    unique_ils = datapoints_df.drop_duplicates(subset=["il_smiles"]).copy()
    unique_ils = unique_ils[[name_col, "il_smiles"]].reset_index(drop=True)
    print(f"[featurize_all_ils] Featurizing {len(unique_ils)} unique ILs...")
    print(f"[featurize_all_ils] Descriptor set: {len(RDKIT_DESCRIPTORS)} per ion "
          f"(18 total, expanded from 8 in v1)")

    failed_count = 0
    feature_rows = []

    for idx, row in unique_ils.iterrows():
        il_smiles = str(row["il_smiles"]) if pd.notna(row["il_smiles"]) else ""
        il_name   = str(row[name_col])

        if not il_smiles:
            print(f"  [DATA QUALITY] No SMILES for '{il_name}' -- skipping")
            failed_count += 1
            continue

        features = featurize_one_il(il_smiles, il_name)
        features["il_name"] = il_name
        feature_rows.append(features)

        if (idx + 1) % 50 == 0:
            print(f"  -> {idx + 1}/{len(unique_ils)} ILs featurized")

    features_df = pd.DataFrame(feature_rows)
    print(f"[featurize_all_ils] Done: {len(features_df)} ILs featurized, "
          f"{failed_count} skipped")
    print(f"  Feature columns: {len(features_df.columns)} total")
    n_desc_cols = sum(1 for c in features_df.columns if not c.startswith(("cat_fp_", "an_fp_"))
                      and c not in ("il_smiles", "cation_smiles", "anion_smiles", "il_name"))
    print(f"  Of which descriptor columns: {n_desc_cols} (should be 36 = 18 per ion)")
    return features_df


def main():
    """Main: load datapoints -> featurize unique ILs -> save."""
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"Input not found: {INPUT_CSV}. Run src/merge_thermoml_into_pipeline.py first."
        )

    datapoints_df = pd.read_csv(INPUT_CSV)
    print(f"[main] Loaded {len(datapoints_df)} rows, "
          f"{datapoints_df['il_smiles'].nunique()} unique SMILES")

    features_df = featurize_all_ils(datapoints_df)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    features_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[main] Saved features -> {OUTPUT_CSV}")
    print(f"  Shape: {features_df.shape[0]} ILs x {features_df.shape[1]} columns")
    print()
    print("NEXT STEPS:")
    print("  1. python src/build_dataset.py     # regenerates train/test with new features")
    print("  2. rm logs/optuna.db               # discard old tuning (v2 -> v3)")
    print("  3. python src/tune_hyperparameters.py  # re-tune on expanded feature set")
    print("  4. python src/train_model.py       # train final model")


if __name__ == "__main__":
    main()
