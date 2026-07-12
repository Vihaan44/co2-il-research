"""
extract_shiflett_yokozeki_2008_2009_2007.py
--------------------------------------------
PURPOSE: Transcribe CO2 solubility (T, P, x2) data points from three
         Shiflett/Yokozeki papers into the schema required by
         add_literature_data.py, then validate and write the CSV.

SOURCES (all manually transcribed from PDF tables provided by Vihaan):
  1. Yokozeki, Shiflett, Junk, Grieco, Foo (2008) J. Phys. Chem. B 112, 16654.
     -> 9 NOVEL ionic liquids synthesized specifically for this study.
        These are essentially guaranteed not to be in ILThermo/standard
        databases since they were custom-synthesized.
  2. Shiflett & Yokozeki (2009) J. Chem. Eng. Data 54, 108.
     -> [emim][Ac] and [emim][TFA] at 323.1K and 348.1K (NEW temperatures;
        298.1K data in this paper is a reprint from source 1, so it is
        excluded here to avoid re-adding an exact duplicate).
  3. Shiflett & Yokozeki (2007) J. Phys. Chem. B 111, 2070.
     -> [hmim][Tf2N], IUPAC ultrapure sample, at 282/297/323/348K.

SCOPE NOTE: [bmim][PF6] and [bmim][BF4] (Shiflett & Yokozeki 2005) were
NOT transcribed here. Both are extremely common ILs almost certainly
already present in the ILThermo-sourced training set. Full re-transcription
of ~72 rows was judged low marginal value; can be added later if needed.

WHY MOL% -> MOLE FRACTION: papers report "100*x1" (CO2 mole percent).
Divide by 100 to get x2_CO2 in the [0,1] mole fraction scale used by
the rest of the pipeline.

WHY MPa -> kPa: papers report pressure in MPa; pipeline schema uses kPa.

DATA QUALITY: rows reported as "<0.1" mol% (below quantification limit)
are dropped rather than guessed, per project data-quality policy.
"""

import os
import pandas as pd
from rdkit import Chem

OUTPUT_CSV = os.path.join("data", "raw", "literature_co2_data.csv")

# -- SMILES building blocks (canonical form applied at the end) --------------
# Cations
BMIM = "CCCC[n+]1ccn(C)c1"          # 1-butyl-3-methylimidazolium
EMIM = "CC[n+]1ccn(C)c1"            # 1-ethyl-3-methylimidazolium
HMIM = "CCCCCC[n+]1ccn(C)c1"        # 1-hexyl-3-methylimidazolium
TBP  = "CCCC[P+](CCCC)(CCCC)CCCC"   # tetrabutylphosphonium

# Anions. TFES/PRO/IBS/TMA/FOR/LEV/SUC/IDA/IAAc are constructed directly
# from the X-COO- functional group descriptions given explicitly in the
# 2008 paper's own text (see "Discussion" section), not guessed.
ANIONS = {
    "TFES": "[O-]S(=O)(=O)C(F)(F)C(F)F",         # 1,1,2,2-tetrafluoroethanesulfonate
    "PRO":  "CCC(=O)[O-]",                        # propionate
    "IBS":  "CC(C)C(=O)[O-]",                     # isobutyrate
    "TMA":  "CC(C)(C)C(=O)[O-]",                  # trimethylacetate (pivalate)
    "FOR":  "[O-]C=O",                            # formate
    "LEV":  "CC(=O)CCC(=O)[O-]",                  # levulinate
    "SUC":  "NC(=O)CCC(=O)[O-]",                  # succinamate
    "IAAc": "OC(=O)CNCC(=O)[O-]",                 # iminoacetic acid acetate (mono-anion)
    "Ac":   "CC(=O)[O-]",                         # acetate
    "TFA":  "[O-]C(=O)C(F)(F)F",                  # trifluoroacetate
    "Tf2N": "[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F",  # bis(triflyl)imide
}
IDA_DIANION = "[O-]C(=O)CNCC(=O)[O-]"  # iminodiacetate, 2- charge, paired with 2 cations

IL_SMILES = {
    "[bmim][TFES]": f"{BMIM}.{ANIONS['TFES']}",
    "[bmim][PRO]":  f"{BMIM}.{ANIONS['PRO']}",
    "[bmim][IBS]":  f"{BMIM}.{ANIONS['IBS']}",
    "[bmim][TMA]":  f"{BMIM}.{ANIONS['TMA']}",
    "[TBP][FOR]":   f"{TBP}.{ANIONS['FOR']}",
    "[bmim][LEV]":  f"{BMIM}.{ANIONS['LEV']}",
    "[bmim][SUC]":  f"{BMIM}.{ANIONS['SUC']}",
    "[bmim]2[IDA]": f"{BMIM}.{BMIM}.{IDA_DIANION}",
    "[bmim][IAAc]": f"{BMIM}.{ANIONS['IAAc']}",
    "[emim][Ac]":   f"{EMIM}.{ANIONS['Ac']}",
    "[emim][TFA]":  f"{EMIM}.{ANIONS['TFA']}",
    "[hmim][Tf2N]": f"{HMIM}.{ANIONS['Tf2N']}",
}

# -- Transcribed (T_K, P_MPa, x2_mol_percent) data ---------------------------
# Source 1: Yokozeki et al. 2008, JPCB 112, 16654, Tables 2 and 3.
# All measured at ~298.1-298.2K (single isotherm study).
NOVEL_IL_DATA_2008 = {
    "[bmim][TFES]": [
        (298.1, 0.0501, 0.8), (298.1, 0.1000, 1.7), (298.2, 0.3998, 6.8),
        (298.2, 0.6997, 11.4), (298.2, 0.9999, 15.8), (297.9, 1.2997, 19.7),
        (298.1, 1.4995, 22.2), (298.0, 1.9998, 28.5),
    ],  # first row (<0.1 mol%, below quantification) dropped
    "[bmim][PRO]": [
        (298.1, 0.0103, 13.4), (298.1, 0.0503, 19.7), (298.2, 0.1001, 21.8),
        (298.2, 0.3994, 26.6), (298.1, 0.6999, 29.6), (298.1, 1.0001, 32.1),
        (298.2, 1.2994, 34.4), (298.2, 1.4997, 36.0), (298.2, 1.9990, 39.3),
    ],
    "[bmim][IBS]": [
        (298.2, 0.0102, 12.3), (298.1, 0.0502, 19.6), (298.2, 0.0998, 21.8),
        (298.2, 0.4002, 26.7), (298.1, 0.6998, 29.9), (298.1, 1.0011, 32.5),
        (298.2, 1.2996, 34.9), (298.2, 1.4993, 36.5), (298.2, 1.9999, 40.3),
    ],
    "[bmim][TMA]": [
        (298.1, 0.0101, 8.5), (298.1, 0.0497, 19.0), (298.1, 0.0996, 23.4),
        (298.1, 0.3999, 29.1), (298.1, 0.6997, 32.3), (298.1, 0.9995, 35.1),
        (298.1, 1.2997, 37.7), (298.1, 1.4999, 39.3), (298.1, 2.0000, 43.1),
    ],
    "[TBP][FOR]": [
        (298.0, 0.0100, 0.8), (298.2, 0.0502, 2.4), (298.1, 0.1004, 3.4),
        (298.0, 0.4004, 9.5), (298.1, 0.7004, 15.1), (298.2, 0.9999, 20.3),
        (298.1, 1.2997, 25.1), (298.0, 1.4993, 28.0), (298.1, 1.9996, 34.8),
    ],
    "[bmim][LEV]": [
        (298.1, 0.0101, 11.8), (298.1, 0.0501, 21.2), (298.1, 0.1002, 24.5),
        (298.1, 0.3996, 31.1), (298.1, 0.6998, 34.9), (298.1, 0.9997, 38.0),
        (298.1, 1.2995, 40.7), (298.1, 1.5001, 42.7), (298.1, 1.9998, 46.0),
    ],
    "[bmim][SUC]": [
        (298.1, 0.0501, 1.2), (298.1, 0.1002, 2.3), (298.1, 0.3997, 5.1),
        (298.1, 0.6998, 8.6), (298.1, 0.9997, 12.4), (298.1, 1.2998, 16.1),
        (298.1, 1.4997, 18.9), (298.1, 1.9997, 23.2),
    ],  # first row (<0.1 mol%) dropped
    "[bmim]2[IDA]": [
        (298.1, 0.0101, 2.1), (298.1, 0.0502, 5.0), (298.1, 0.1002, 7.7),
        (298.1, 0.3996, 14.0), (298.1, 0.6998, 19.8), (298.1, 0.9998, 25.6),
        (298.1, 1.2995, 30.5), (298.1, 1.4999, 34.4), (298.1, 1.9997, 39.5),
    ],
    "[bmim][IAAc]": [
        (298.1, 0.0101, 0.1), (298.1, 0.0502, 0.4), (298.1, 0.1003, 0.6),
        (298.1, 0.3997, 2.1), (298.1, 0.7002, 5.1), (298.1, 0.9996, 7.5),
        (298.1, 1.2998, 10.9), (298.1, 1.4998, 14.0), (298.1, 1.9996, 19.1),
    ],
}

# Source 2: Shiflett & Yokozeki 2009, JCED 54, 108, Table 2.
# Only 323.1K and 348.1K rows (298.1K excluded -- reprint of source 1's data).
NEW_TEMP_DATA_2009 = {
    "[emim][Ac]": [
        (323.1, 0.0100, 13.8), (323.1, 0.0499, 20.3), (323.1, 0.1000, 23.0),
        (323.1, 0.4000, 28.5), (323.1, 0.7000, 31.4), (323.1, 1.0001, 33.6),
        (323.1, 1.2997, 35.0), (323.1, 1.4998, 36.1), (323.1, 1.9996, 39.0),
        (348.1, 0.0101, 9.4),  (348.1, 0.0501, 15.7), (348.1, 0.1002, 18.6),
        (348.1, 0.3997, 24.1), (348.2, 0.6996, 26.4), (348.1, 0.9998, 28.0),
        (348.1, 1.2996, 29.4), (348.2, 1.4997, 29.8), (348.2, 1.9996, 32.1),
    ],
    "[emim][TFA]": [
        (323.1, 0.0100, 0.4),  (323.1, 0.0499, 0.9),  (323.1, 0.1000, 1.5),
        (323.1, 0.4000, 4.4),  (323.1, 0.7000, 7.2),  (323.1, 1.0001, 10.4),
        (323.1, 1.2997, 13.4), (323.1, 1.4998, 15.6), (323.1, 1.9996, 20.0),
        (348.1, 0.0101, 0.6),  (348.1, 0.0501, 1.0),  (348.1, 0.1002, 1.3),
        (348.1, 0.3997, 3.2),  (348.1, 0.6996, 5.0),  (348.1, 0.9998, 7.0),
        (348.1, 1.2996, 9.5),  (348.1, 1.4997, 10.8), (348.2, 1.9996, 14.4),
    ],
}

# Source 3: Shiflett & Yokozeki 2007, JPCB 111, 2070, Table 2, IUPAC sample.
HMIM_TF2N_DATA_2007 = {
    "[hmim][Tf2N]": [
        (282.0, 0.0089, 0.6),  (281.9, 0.0486, 2.4),  (281.9, 0.0982, 4.7),
        (281.9, 0.3944, 16.6), (282.0, 0.6921, 26.5), (281.9, 0.9882, 34.9),
        (281.9, 1.2844, 41.6), (282.1, 1.4830, 45.1), (282.0, 1.9757, 53.9),
        (297.4, 0.0091, 0.7),  (297.2, 0.0482, 2.1),  (297.4, 0.0983, 3.8),
        (297.3, 0.3944, 12.7), (297.3, 0.6922, 20.4), (297.4, 0.9889, 27.1),
        (297.2, 1.2852, 32.8), (297.3, 1.4829, 36.2), (297.2, 1.9760, 43.4),
        (322.8, 0.0091, 0.2),  (322.9, 0.0482, 1.0),  (322.9, 0.0979, 2.1),
        (322.9, 0.3951, 8.4),  (322.9, 0.6920, 13.8), (322.9, 0.9882, 18.9),
        (322.9, 1.2854, 23.1), (322.9, 1.4827, 25.6), (322.8, 1.9764, 31.6),
        (348.5, 0.0091, 0.4),  (348.5, 0.0485, 1.2),  (348.5, 0.0980, 2.1),
        (348.5, 0.3950, 6.4),  (348.5, 0.6921, 10.5), (348.5, 0.9888, 14.1),
        (348.5, 1.2851, 17.7), (348.4, 1.4828, 19.9), (348.4, 1.9755, 24.7),
    ],
}

SOURCE_TAGS = {
    "novel_2008":  "Yokozeki_Shiflett_2008_JPCB",
    "newtemp_2009": "Shiflett_Yokozeki_2009_JCED",
    "hmimtf2n_2007": "Shiflett_Yokozeki_2007_JPCB",
}


def validate_smiles(il_name: str, smiles: str) -> str:
    """Parse and canonicalize a constructed IL SMILES with RDKit.
    Raises if unparseable -- we never want to silently write bad SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"UNPARSEABLE SMILES for {il_name}: {smiles}")
    return Chem.MolToSmiles(mol)


def build_rows(data_dict: dict, source_tag: str) -> list:
    """Convert one {il_name: [(T_K, P_MPa, x2_molpercent), ...]} block
    into a list of row-dicts matching the add_literature_data.py schema."""
    rows = []
    for il_name, points in data_dict.items():
        canonical_smiles = validate_smiles(il_name, IL_SMILES[il_name])
        for T_K, P_MPa, x2_molpct in points:
            rows.append({
                "il_smiles": canonical_smiles,
                "il_name":   il_name,
                "T_K":       T_K,
                "P_kPa":     round(P_MPa * 1000.0, 2),   # MPa -> kPa
                "x2_CO2":    round(x2_molpct / 100.0, 4),  # mol% -> mole fraction
                "source":    source_tag,
            })
    return rows


def main():
    """Build all rows, validate, and write the literature CSV."""
    all_rows = []
    all_rows += build_rows(NOVEL_IL_DATA_2008, SOURCE_TAGS["novel_2008"])
    all_rows += build_rows(NEW_TEMP_DATA_2009, SOURCE_TAGS["newtemp_2009"])
    all_rows += build_rows(HMIM_TF2N_DATA_2007, SOURCE_TAGS["hmimtf2n_2007"])

    df = pd.DataFrame(all_rows)
    print(f"Total rows: {len(df)}", flush=True)
    print(f"Unique ILs: {df['il_name'].nunique()}", flush=True)
    print(df.groupby("il_name").size(), flush=True)

    # Sanity check: all x2 in (0, 1], all T in reasonable range
    assert (df["x2_CO2"] > 0).all() and (df["x2_CO2"] <= 1).all()
    assert (df["T_K"] > 250).all() and (df["T_K"] < 400).all()

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved -> {OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
