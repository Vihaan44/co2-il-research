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
# All SMILES verified against RDKit canonical parsing.
# Sources: PubChem, Sigma-Aldrich IL catalog, Brennecke & Maginn 2001, Ramdin 2012
#
# SMILES FIX NOTE (vs prior version):
#   Imidazolium cations must use CCn1cc[n+](R)c1 notation (explicit [n+] on N3).
#   The old Cc1cn(R)cn1C aromatic form causes RDKit kekulization errors because
#   the 5-membered ring with two nitrogens and a positive charge can't be
#   aromatized with that atom ordering.
#   Pyridinium must use RCCCC[n+]1ccccc1 (explicit [n+] charge on aromatic N).
#   The old CCCCn1ccc(C)cc1 form is parsed as neutral N-alkylpyridine (no charge).

CATION_ENTRIES = [
    # ── Imidazolium ──
    ("CCn1cc[n+](C)c1",           "imidazolium"),   # [EMIM]+ 1-ethyl-3-methylimidazolium
    ("CCCCn1cc[n+](C)c1",         "imidazolium"),   # [BMIM]+ 1-butyl-3-methylimidazolium
    ("CCCCCCn1cc[n+](C)c1",       "imidazolium"),   # [HMIM]+ 1-hexyl-3-methylimidazolium
    ("CCCCCCCCn1cc[n+](C)c1",     "imidazolium"),   # [OMIM]+ 1-octyl-3-methylimidazolium
    ("Cn1cc[n+](C)c1",            "imidazolium"),   # [MMIM]+ 1,3-dimethylimidazolium
    ("CCCn1cc[n+](C)c1",          "imidazolium"),   # [PMIM]+ 1-propyl-3-methylimidazolium
    # Functionalized imidazolium — FIXED from Cc1cn(R)cn1C to CCOn1cc[n+](C)c1 style
    ("CCOn1cc[n+](C)c1",          "imidazolium"),   # [OHEMIM]+ hydroxyethyl — FIXED
    ("N#CCCn1cc[n+](C)c1",        "imidazolium"),   # [CNpMIM]+ cyanopropyl — FIXED
    ("FC(F)(F)CCn1cc[n+](C)c1",   "imidazolium"),   # [TFMIM]+ trifluoroethyl — FIXED
    ("COCCn1cc[n+](C)c1",         "imidazolium"),   # [MeOEMIM]+ methoxyethyl — FIXED
    ("SCCn1cc[n+](C)c1",          "imidazolium"),   # [thioEMIM]+ thioethyl — FIXED

    # ── Pyrrolidinium ──
    ("C[N+]1(C)CCCC1",            "pyrrolidinium"),  # [C1MPyrr]+
    ("CCCC[N+]1(C)CCCC1",         "pyrrolidinium"),  # [BMPyrr]+
    ("CCCCCC[N+]1(C)CCCC1",       "pyrrolidinium"),  # [HMPyrr]+
    ("CC(=O)OCC[N+]1(C)CCCC1",    "pyrrolidinium"),  # acetate-ester pyrrolidinium

    # ── Ammonium ──
    ("CC[N+](CC)(CC)CC",           "ammonium"),  # [TEA]+
    ("CCCC[N+](C)(C)C",            "ammonium"),  # [N1114]+
    ("NCC[N+](C)(C)C",             "ammonium"),  # [AETMA]+ — top Phase 4 cation
    ("OCCC[N+](C)(C)C",            "ammonium"),  # [OHTMA]+ — hydroxypropyl-TMA
    ("OCC[N+](C)(C)C",             "ammonium"),  # [choline]+ — classic reference
    ("NCC[N+](CC)(CC)CC",          "ammonium"),  # [AETEA]+ — amino-ethyl triethylammonium
    ("NCCC[N+](C)(C)C",            "ammonium"),  # [APTMA]+ — aminopropyl-TMA (SAR chain +1)
    ("CNCC[N+](C)(C)C",            "ammonium"),  # [MeAETMA]+ — N-methyl amino
    ("FC(F)(F)CC[N+](C)(C)C",      "ammonium"),  # [TFE-TMA]+ — trifluoroethyl
    ("C#CCC[N+](C)(C)C",           "ammonium"),  # [propargyl-TMA]+ — alkyne (new topology)

    # ── Phosphonium ──
    ("CCCC[P+](CCCC)(CCCC)CCCC",   "phosphonium"),  # [P4444]+
    ("CC[P+](CC)(CC)CC",            "phosphonium"),  # [P2222]+
    ("CCCC[P+](CCCC)(CCCC)CCO",    "phosphonium"),  # [P4441OH]+
    ("CCCC[P+](CCCC)(CCCC)CCC#N",  "phosphonium"),  # [P444CN]+

    # ── Piperidinium ──
    ("CCCC[N+]1(C)CCCCC1",         "piperidinium"),  # [BMPip]+
    ("CCCCCC[N+]1(C)CCCCC1",       "piperidinium"),  # [HMPip]+
    ("OCC[N+]1(C)CCCCC1",          "piperidinium"),  # [OH-EMPip]+
    ("NCC[N+]1(C)CCCCC1",          "piperidinium"),  # [NH2-EMPip]+

    # ── Morpholinium ──
    ("CCCC[N+]1(C)CCOCC1",         "morpholinium"),  # [BMMor]+
    ("CC[N+]1(C)CCOCC1",           "morpholinium"),  # [EMMor]+
    ("CCCCCC[N+]1(C)CCOCC1",       "morpholinium"),  # [HMMor]+

    # ── Sulfonium (NEW — barely studied for CO2 capture) ──
    ("CC[S+](CC)CC",               "sulfonium"),  # triethylsulfonium
    ("CCCC[S+](CC)CC",             "sulfonium"),  # butyldiethylsulfonium
    ("CC[S+](Cc1ccccc1)CC",        "sulfonium"),  # benzyldiethylsulfonium

    # ── Guanidinium (NEW — high charge delocalization) ──
    ("CN(C)C(=[N+](C)C)N(C)C",    "guanidinium"),     # hexamethylguanidinium
    ("CCN(CC)C(=[N+](CC)CC)N(CC)CC", "guanidinium"),   # hexaethylguanidinium

    # ── Oxazolinium (NEW ring topology) ──
    ("CC[N+]1=CCCO1",              "oxazolinium"),  # 3-ethyl-1,3-oxazolinium
    ("CCCC[N+]1=CCCO1",            "oxazolinium"),  # 3-butyl-1,3-oxazolinium

    # ── Pyridinium (aromatic N cation) ──
    # FIXED: explicit [n+] charge on aromatic N — old CCCCn1ccc(C)cc1 is uncharged
    ("CCCC[n+]1ccc(C)cc1",         "pyridinium"),   # [4-MeBuPy]+ — FIXED
    ("CCCCCC[n+]1ccccc1",          "pyridinium"),   # [HexPy]+ — FIXED
    ("CCCC[n+]1ccccc1",            "pyridinium"),   # [BuPy]+ — FIXED
]


# ── Expanded Anion Library ────────────────────────────────────────────────────
# Format: (SMILES, family_label)

ANION_ENTRIES = [
    # ── Fluorinated (high CO2 affinity) ──
    ("F[B-](F)(F)F",                           "fluorinated"),  # [BF4]-
    ("F[P-](F)(F)(F)(F)F",                     "fluorinated"),  # [PF6]-
    ("O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F","fluorinated"),  # [Tf2N]-
    ("O=S(=O)([O-])C(F)(F)F",                  "fluorinated"),  # [OTf]-
    ("O=S(=O)([N-]S(=O)(=O)F)F",              "fluorinated"),  # [FSI]-
    ("FC(F)(F)C(=O)[O-]",                      "fluorinated"),  # [TFA]-
    ("F[P-](F)(F)(C(F)(F)F)(C(F)(F)F)C(F)(F)F","fluorinated"), # [FAP]-
    ("F[Sb-](F)(F)(F)(F)F",                    "fluorinated"),  # [SbF6]-

    # ── Carboxylate ──
    ("CC([O-])=O",                             "carboxylate"),  # [OAc]-
    ("OC([O-])=O",                             "carboxylate"),  # [HCO3]-
    ("CC(O)C([O-])=O",                         "carboxylate"),  # [lactate]-
    ("[O-]C(=O)c1ccccc1",                      "carboxylate"),  # [benzoate]-
    ("[O-]C(=O)CC([O-])=O",                    "carboxylate"),  # [malonate]2- (dianion — flag)

    # ── Sulfonate ──
    ("CCCCOS([O-])(=O)=O",                     "sulfonate"),  # [BuSO4]-
    ("CS([O-])(=O)=O",                         "sulfonate"),  # [MeSO3]-
    ("O=S(=O)([O-])c1ccc(C)cc1",              "sulfonate"),  # [TsO]- tosylate
    ("O=S(=O)([O-])CCO",                       "sulfonate"),  # [isethionate]-

    # ── Nitrogen-based ──
    ("N(=O)[N-]C#N",                           "nitrogen"),  # [DCA]-
    ("[B-](C#N)(C#N)(C#N)C#N",               "nitrogen"),  # [TCB]-
    ("[N-](S(=O)(=O)F)C#N",                   "nitrogen"),  # [FSCN]-
    ("[N-]([N+](=O)[O-])",                     "nitrogen"),  # [N3]- azide (reactive — flag)

    # ── Halide (negative controls) ──
    ("[Cl-]",  "halide"),
    ("[Br-]",  "halide"),
    ("[I-]",   "halide"),

    # ── Heterocyclic anions (novel) ──
    ("O=C1NS(=O)(=O)c2ccccc21",               "heterocyclic"),  # [saccharinate]-
    ("CC1=CC(=O)[N-]S1(=O)=O",               "heterocyclic"),  # [acesulfamate]-
    ("O=C1[N-]C(=O)c2ccccc21",               "heterocyclic"),  # [phthalimide]-

    # ── Phosphate-based ──
    ("CCOP([O-])(=O)OCC",                      "phosphate"),  # [DEP]-
    ("COP([O-])(=O)OC",                        "phosphate"),  # [DMP]-

    # ── Thiocyanate / cyanate ──
    ("[S-]C#N",  "thiocyanate"),  # [SCN]-
    ("[O-]C#N",  "cyanate"),      # [OCN]-
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
    print(f"\n✓ Run python src/inverse_design.py to screen this library.")


if __name__ == "__main__":
    main()
