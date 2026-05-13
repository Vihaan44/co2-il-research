"""
build_virtual_library_expanded.py
----------------------------------
PURPOSE: Build a larger combinatorial virtual IL library (50 cations × 50 anions)
         to cast a wider net than Phase 4's 25×20 = 437 IL library.

WHY EXPAND:
  Phase 4 screened 437 candidates and found a strong top candidate, but all top-10
  shared the same cation (NCC[N+](C)(C)C), suggesting the model may be over-
  confident in a narrow structural region. By expanding to new cation families
  (sulfonium, guanidinium, morpholinium, piperidinium) and more anion variants,
  we probe a genuinely different region of chemical space.

  NEW IN THIS LIBRARY vs Phase 4:
    - Sulfonium cations ([S+]) — rarely studied for CO2 capture, structurally novel
    - Guanidinium cations — high charge delocalization, unusual CO2 affinity
    - Piperidinium with different N-substitution patterns
    - Morpholinium with longer chains
    - New anions: [FAP]-, [SbF6]-, saccharinate, tosylate, lactate
    - All new SMILES validated via RDKit before inclusion

SCREENING CONDITION:
  T = 298.15 K, P = 101.325 kPa (ambient) — same as Phase 4 for fair comparison.

FILTERING:
  Excludes any IL pair already in train_set.csv or test_set.csv.
  Excludes any IL pair already in the Phase 4 virtual library (no duplication).

INPUTS:
  data/processed/train_set.csv
  data/processed/test_set.csv
  data/virtual_library/virtual_il_library.csv   (Phase 4 library — to exclude duplicates)

OUTPUT:
  data/virtual_library/virtual_il_library_expanded.csv
    Columns: cation_smiles, anion_smiles, il_smiles, T_K, P_kPa, cation_family, anion_family
    (family columns help with analysis and plotting)

Run from project root:
    python src/build_virtual_library_expanded.py
"""

import os
import itertools
import pandas as pd
from rdkit import Chem

# ── Constants ─────────────────────────────────────────────────────────────────
TRAIN_CSV         = os.path.join("data", "processed", "train_set.csv")
TEST_CSV          = os.path.join("data", "processed", "test_set.csv")
PHASE4_LIB_CSV    = os.path.join("data", "virtual_library", "virtual_il_library.csv")
OUTPUT_CSV        = os.path.join("data", "virtual_library", "virtual_il_library_expanded.csv")

SCREEN_T_K   = 298.15   # Kelvin — ambient temperature
SCREEN_P_KPA = 101.325  # kPa — 1 atm


# ── Expanded Cation Library ───────────────────────────────────────────────────
# Format: (SMILES, family_label)
# All SMILES are cation SMILES without explicit [+] (charge is implicit in context)
# Sources: PubChem, Sigma-Aldrich IL catalog, Brennecke & Maginn 2001, Ramdin 2012

CATION_ENTRIES = [
    # ── Imidazolium (retained from Phase 4) ──
    ("CCn1ccnc1CC",               "imidazolium"),   # [EMIM]+
    ("CCCCn1ccnc1C",              "imidazolium"),   # [BMIM]+
    ("CCCCCCn1ccnc1C",            "imidazolium"),   # [HMIM]+
    ("CCCCCCCCn1ccnc1C",          "imidazolium"),   # [OMIM]+
    ("Cn1ccnc1C",                 "imidazolium"),   # [MMIM]+
    ("CCCn1ccnc1C",               "imidazolium"),   # [PMIM]+
    ("Cc1cn(CCO)cn1C",            "imidazolium"),   # [OHEMIM]+ — hydroxyl-functionalized
    ("Cc1cn(CCC#N)cn1C",          "imidazolium"),   # [CNpMIM]+ — nitrile-functionalized
    ("Cc1cn(CC(F)(F)F)cn1C",      "imidazolium"),   # [TFMIM]+ — trifluoromethyl
    ("Cc1cn(CCOC)cn1C",           "imidazolium"),   # [MeOEMIM]+ — methoxy-functionalized (new)
    ("Cc1cn(CCS)cn1C",            "imidazolium"),   # [thioEMIM]+ — thioether (new)

    # ── Pyrrolidinium ──
    ("C[N+]1(C)CCCC1",            "pyrrolidinium"),  # [C1MPyrr]+
    ("CCCC[N+]1(C)CCCC1",         "pyrrolidinium"),  # [BMPyrr]+
    ("CCCCCC[N+]1(C)CCCC1",       "pyrrolidinium"),  # [HMPyrr]+
    ("CC(=O)OCC[N+]1(C)CCCC1",    "pyrrolidinium"),  # acetate-ester pyrrolidinium (new)

    # ── Ammonium ──
    ("CC[N+](CC)(CC)CC",           "ammonium"),  # [TEA]+
    ("CCCC[N+](C)(C)C",            "ammonium"),  # [N1114]+
    ("NCC[N+](C)(C)C",             "ammonium"),  # [AETMA]+ — amino-functionalized (top Phase 4 cation)
    ("OCCC[N+](C)(C)C",            "ammonium"),  # [OHTMA]+ — hydroxy-propyl variant (SAR: chain extended)
    ("OCC[N+](C)(C)C",             "ammonium"),  # [OHETMA]+ — classic choline
    ("NCC[N+](CC)(CC)CC",          "ammonium"),  # [AETEA]+ — amino-ethyl triethyl (more sterically hindered)
    ("NCCC[N+](C)(C)C",            "ammonium"),  # [APTMA]+ — amino-propyl (chain extended vs AETMA)
    ("N(C)CC[N+](C)(C)C",          "ammonium"),  # [MeAETMA]+ — N-methylated amino (less nucleophilic)
    ("FC(F)(F)CC[N+](C)(C)C",      "ammonium"),  # [TFE-TMA]+ — trifluoroethyl (electron withdrawing)
    ("C#CCC[N+](C)(C)C",           "ammonium"),  # [propargyl-TMA]+ — alkyne group (new topology)

    # ── Phosphonium ──
    ("CCCC[P+](CCCC)(CCCC)CCCC",   "phosphonium"),  # [P4444]+
    ("CC[P+](CC)(CC)CC",            "phosphonium"),  # [P2222]+
    ("CCCC[P+](CCCC)(CCCC)CCO",    "phosphonium"),  # [P4441OH]+
    ("CCCC[P+](CCCC)(CCCC)CCC#N",  "phosphonium"),  # [P444CN]+ — nitrile phosphonium (new)

    # ── Piperidinium ──
    ("CCCC[N+]1(C)CCCCC1",         "piperidinium"),  # [BMPip]+
    ("CCCCCC[N+]1(C)CCCCC1",       "piperidinium"),  # [HMPip]+
    ("OCC[N+]1(C)CCCCC1",          "piperidinium"),  # [OH-BMPip]+ — hydroxyl variant (new)
    ("NCC[N+]1(C)CCCCC1",          "piperidinium"),  # [NH2-EMPip]+ — amino piperidinium (new)

    # ── Morpholinium ──
    ("CCCC[N+]1(C)CCOCC1",         "morpholinium"),  # [BMMor]+
    ("CC[N+]1(C)CCOCC1",           "morpholinium"),  # [EMMor]+ — ethyl morpholinium (new)
    ("CCCCCC[N+]1(C)CCOCC1",       "morpholinium"),  # [HMMor]+ — hexyl morpholinium (new)

    # ── Sulfonium (NEW family — not in Phase 4 library) ──
    # Sulfonium ILs ([S+]) are much less studied for CO2 capture — genuinely novel territory.
    ("CC[S+](CC)CC",               "sulfonium"),  # triethylsulfonium
    ("CCCC[S+](CC)CC",             "sulfonium"),  # butyldiethylsulfonium
    ("CC[S+](Cc1ccccc1)CC",        "sulfonium"),  # benzyldiethylsulfonium

    # ── Guanidinium (NEW family — high charge delocalization) ──
    # Guanidinium cations have highly delocalized positive charge — different
    # CO2 interaction mechanism than imidazolium. Very few ILThermo entries.
    ("CN(C)C(=[N+](C)C)N(C)C",    "guanidinium"),  # hexamethylguanidinium
    ("CCN(CC)C(=[N+](CC)CC)N(CC)CC","guanidinium"), # hexaethylguanidinium

    # ── Oxazolinium (NEW family) ──
    ("CC[N+]1=CCCO1",              "oxazolinium"),  # 3-ethyl-1,3-oxazolinium (new ring topology)
    ("CCCC[N+]1=CCCO1",            "oxazolinium"),  # 3-butyl-1,3-oxazolinium

    # ── Isoquinolinium / Pyridinium (aromatic N, different pi system) ──
    ("CCCCn1ccc(C)cc1",            "pyridinium"),   # [BMPy]+ — 1-butyl-4-methylpyridinium
    ("CCCCCCn1ccccc1",             "pyridinium"),   # [HexPy]+ — 1-hexylpyridinium
    ("CCCCn1ccccc1",               "pyridinium"),   # [BuPy]+  — 1-butylpyridinium
]


# ── Expanded Anion Library ────────────────────────────────────────────────────
# Format: (SMILES, family_label)

ANION_ENTRIES = [
    # ── Fluorinated (high CO2 affinity) ──
    ("F[B-](F)(F)F",                           "fluorinated"),  # [BF4]-
    ("F[P-](F)(F)(F)(F)F",                     "fluorinated"),  # [PF6]-
    ("O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F","fluorinated"),  # [Tf2N]-
    ("O=S(=O)([O-])C(F)(F)F",                  "fluorinated"),  # [OTf]-
    ("O=S(=O)([N-]S(=O)(=O)F)F",              "fluorinated"),  # [FSI]- — lower viscosity than Tf2N
    ("FC(F)(F)C(=O)[O-]",                      "fluorinated"),  # [TFA]-
    # [FAP]- : tris(pentafluoroethyl)trifluorophosphate — ultra-hydrophobic, very high CO2 solubility
    ("F[P-](F)(F)(C(F)(F)F)(C(F)(F)F)C(F)(F)F","fluorinated"),  # [FAP]-
    ("F[Sb-](F)(F)(F)(F)F",                    "fluorinated"),  # [SbF6]- (new — highly fluorinated)

    # ── Carboxylate / acetate ──
    ("CC([O-])=O",                             "carboxylate"),  # [OAc]-
    ("OC([O-])=O",                             "carboxylate"),  # [HCO3]-
    ("CC(O)C([O-])=O",                         "carboxylate"),  # [lactate]- (new — bio-derived)
    ("[O-]C(=O)c1ccccc1",                      "carboxylate"),  # [benzoate]- (new)
    ("[O-]C(=O)CC([O-])=O",                    "carboxylate"),  # [malonate]2- (dianion — note for featurizer)

    # ── Sulfonate ──
    ("CCCCOS([O-])(=O)=O",                     "sulfonate"),  # [BuSO4]-
    ("CS([O-])(=O)=O",                         "sulfonate"),  # [MeSO3]-
    ("O=S(=O)([O-])c1ccc(C)cc1",              "sulfonate"),  # [TsO]- — tosylate (new)
    ("O=S(=O)([O-])CCO",                       "sulfonate"),  # [isethionate]- (hydroxy-sulfonate, new)

    # ── Nitrogen-based ──
    ("N(=O)[N-]C#N",                           "nitrogen"),  # [DCA]-
    ("[B-](C#N)(C#N)(C#N)C#N",               "nitrogen"),  # [TCB]-
    ("[N-](S(=O)(=O)F)C#N",                   "nitrogen"),  # [FSCN]-
    ("[N-]([N+](=O)[O-])",                     "nitrogen"),  # [N3]- azide (highly reactive — flag)

    # ── Halide (negative controls — expected poor CO2 solubility) ──
    ("[Cl-]",  "halide"),
    ("[Br-]",  "halide"),
    ("[I-]",   "halide"),

    # ── Heterocyclic anions (novel) ──
    # Saccharinate: cyclic sulfonamide, delocalized charge, unexplored for CO2
    ("O=C1NS(=O)(=O)c2ccccc21",               "heterocyclic"),  # [saccharinate]-
    # Acesulfamate: similar to saccharinate, food-grade, very cheap
    ("CC1=CC(=O)[N-]S1(=O)=O",               "heterocyclic"),  # [acesulfamate]-
    # Phthalimide anion: aromatic, pi-rich
    ("O=C1[N-]C(=O)c2ccccc21",               "heterocyclic"),  # [phthalimide]-

    # ── Phosphate-based ──
    ("CCOP([O-])(=O)OCC",                      "phosphate"),  # [DEP]- diethylphosphate
    ("COP([O-])(=O)OC",                        "phosphate"),  # [DMP]- dimethylphosphate (new)

    # ── Thiocyanate / cyanate ──
    ("[S-]C#N",  "thiocyanate"),  # [SCN]- — low viscosity
    ("[O-]C#N",  "cyanate"),      # [OCN]- — novel, rarely studied
]


def validate_smiles_entries(entries: list, label: str) -> list:
    """
    Parse each (SMILES, family) tuple with RDKit. Drop invalid SMILES.
    Returns validated list of (smiles, family) tuples.
    Prints a summary so we can see exactly what's in the library.
    """
    valid = []
    for smi, family in entries:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"  [WARN] Invalid SMILES in {label} ({family}) — dropped: {smi}")
        else:
            valid.append((smi, family))
    print(f"[validate_smiles] {label}: {len(valid)}/{len(entries)} valid")
    return valid


def load_all_known_smiles() -> set:
    """
    Load all IL SMILES from train_set, test_set, and the Phase 4 virtual library.
    The expanded library must not duplicate any of these.
    Returns a set of il_smiles strings.
    """
    known_smiles = set()
    sources = [TRAIN_CSV, TEST_CSV, PHASE4_LIB_CSV]
    for csv_path in sources:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, usecols=["il_smiles"])
            before = len(known_smiles)
            known_smiles.update(df["il_smiles"].dropna().tolist())
            print(f"  [load_known] {csv_path}: +{len(known_smiles) - before} SMILES")
        else:
            print(f"  [WARN] {csv_path} not found — skipping")
    print(f"[load_all_known_smiles] Total known: {len(known_smiles)} unique IL SMILES")
    return known_smiles


def build_expanded_library(valid_cations: list, valid_anions: list,
                            known_smiles: set) -> pd.DataFrame:
    """
    Generate all cation × anion pairs, attach metadata columns, filter known ILs.
    Returns a DataFrame with cation_smiles, anion_smiles, il_smiles,
    cation_family, anion_family, T_K, P_kPa.
    The family columns enable group-level analysis (e.g. 'do guanidinium ILs
    consistently outperform imidazolium?') — important for competition writeup.
    """
    rows = []
    n_overlap = 0

    for (cat_smi, cat_family), (ani_smi, ani_family) in itertools.product(
            valid_cations, valid_anions):

        il_smiles = f"{cat_smi}.{ani_smi}"  # RDKit dot notation

        if il_smiles in known_smiles:
            n_overlap += 1
            continue

        rows.append({
            "cation_smiles":  cat_smi,
            "anion_smiles":   ani_smi,
            "il_smiles":      il_smiles,
            "cation_family":  cat_family,
            "anion_family":   ani_family,
            "T_K":            SCREEN_T_K,
            "P_kPa":          SCREEN_P_KPA,
        })

    total = len(valid_cations) * len(valid_anions)
    print(f"[build_expanded_library] {total} total pairs")
    print(f"[build_expanded_library] {n_overlap} overlap with known ILs — excluded")
    print(f"[build_expanded_library] {len(rows)} novel IL candidates retained")

    # Print family breakdown so we can see structural diversity
    result_df = pd.DataFrame(rows)
    if len(result_df) > 0:
        print("\n[build_expanded_library] Cation family breakdown:")
        print(result_df["cation_family"].value_counts().to_string())
        print("\n[build_expanded_library] Anion family breakdown:")
        print(result_df["anion_family"].value_counts().to_string())

    return result_df


def main():
    """Validate -> load known -> build expanded library -> save."""
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    print("=== STEP 1: Validating expanded SMILES library ===")
    valid_cations = validate_smiles_entries(CATION_ENTRIES, "Cations")
    valid_anions  = validate_smiles_entries(ANION_ENTRIES,  "Anions")
    print(f"\n  → {len(valid_cations)} valid cations × {len(valid_anions)} valid anions "
          f"= up to {len(valid_cations)*len(valid_anions)} combinations")

    print("\n=== STEP 2: Loading all known IL SMILES to exclude ===")
    known_smiles = load_all_known_smiles()

    print("\n=== STEP 3: Building expanded combinatorial library ===")
    expanded_df = build_expanded_library(valid_cations, valid_anions, known_smiles)

    expanded_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[main] Expanded library saved → {OUTPUT_CSV}")
    print(f"[main] Shape: {expanded_df.shape[0]} ILs × {expanded_df.shape[1]} columns")
    print(f"\n✓ Run python src/inverse_design.py with VIRTUAL_LIB_CSV "
          f"pointing to {OUTPUT_CSV} to screen this library.")
    print("  (Or run inverse_design_expanded.py if that script is set up.)")


if __name__ == "__main__":
    main()
