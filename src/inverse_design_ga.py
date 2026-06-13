"""
inverse_design_ga.py
--------------------
PURPOSE:
  Genetic algorithm (GA) to discover novel ionic liquids (ILs) predicted to
  have high CO2 mole fraction solubility (high log10(x2_CO2)).

  Instead of screening a fixed combinatorial library, the GA *evolves* a
  population of IL candidates toward higher predicted CO2 absorption by
  combining and mutating building blocks. This is the same strategy used in
  drug discovery pipelines to search molecular space efficiently.

HOW A GENETIC ALGORITHM WORKS (for judges):
  1. Start with a population of IL candidates (generation 0).
  2. Score each candidate using our trained forward model (fitness function).
  3. Select the best candidates as "parents" (tournament selection).
  4. Combine parent pairs (crossover) to create "offspring" IL candidates.
  5. Randomly mutate some offspring (swap chains, add groups, change ions).
  6. Replace the old population with the offspring, keeping the best few (elitism).
  7. Repeat steps 2-6 for many generations until fitness plateaus.

  Unlike brute-force screening (test every combination), a GA searches
  intelligently: it explores new chemical space guided by what worked before.

WHY BUILDING-BLOCK LEVEL (not atom-level):
  Atom-level SMILES mutation (e.g. swap a random atom) almost always produces
  chemically invalid structures. Instead, we represent each IL as a
  (cation_SMILES, anion_SMILES) pair and mutate at the functional-group or
  whole-ion level. Every generated candidate is RDKit-validated before use.

GENE REPRESENTATION:
  Each individual = (cation_smiles: str, anion_smiles: str)
  Fitness = forward_model_prediction + novelty_bonus
  The novelty bonus rewards ILs that are structurally distant from the
  training set (Tanimoto distance > 0), so the GA doesn't just rediscover
  known ILs with marginally different predictions.

MUTATION OPERATORS (applied with weighted probability):
  1. alkyl_chain_extend   -- add one -CH2- to the longest N-alkyl chain
  2. alkyl_chain_shorten  -- remove one -CH2- from the longest N-alkyl chain
  3. swap_whole_cation    -- replace cation with random one from the pool
  4. swap_whole_anion     -- replace anion with random one from the pool
  5. add_functional_group -- append a functional group to the cation
  6. swap_functional_group-- swap one functional group for another

CROSSOVER:
  With probability CROSSOVER_PROB, two parents exchange either:
  - their cations (child gets parent1.cation + parent2.anion), or
  - their anions  (child gets parent1.anion  + parent2.cation)
  Both children are RDKit-validated; invalid ones are discarded.

INPUTS:
  models/forward_model.pkl          -- trained XGBoost model + feature_cols
  data/processed/train_set.csv      -- training IL SMILES (for novelty scoring
                                       and seeding initial population)
  data/processed/test_set.csv       -- test IL SMILES (also excluded from results)

OUTPUTS:
  results/ga_top_candidates.csv     -- top 100 novel ILs ranked by fitness
  results/ga_generational_log.csv   -- best/mean fitness per generation
  figures/ga_fitness_curve.png      -- convergence diagnostic plot

Run from project root:
    python src/inverse_design_ga.py

    Or as background job:
    nohup python src/inverse_design_ga.py > logs/ga.log 2>&1 &
"""

import os
import re
import copy
import random
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Codespace / nohup runs
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

# Suppress RDKit sanitization warnings to keep output readable
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# ── Project imports ────────────────────────────────────────────────────────────
# featurize_il_smiles is the same function train_model.py uses.
# We import it to guarantee the feature vector is *identical* at inference time.
import sys
sys.path.insert(0, os.path.dirname(__file__))
from featurize import featurize_il_smiles

# ── File paths ────────────────────────────────────────────────────────────────
MODEL_PKL    = os.path.join("models", "forward_model.pkl")
TRAIN_CSV    = os.path.join("data", "processed", "train_set.csv")
TEST_CSV     = os.path.join("data", "processed", "test_set.csv")
RESULTS_DIR  = "results"
FIGURES_DIR  = "figures"
GA_RESULTS_CSV  = os.path.join(RESULTS_DIR, "ga_top_candidates.csv")
GA_GEN_LOG_CSV  = os.path.join(RESULTS_DIR, "ga_generational_log.csv")
GA_PLOT_PNG     = os.path.join(FIGURES_DIR,  "ga_fitness_curve.png")

# ── GA hyperparameters ────────────────────────────────────────────────────────
POPULATION_SIZE     = 300    # number of IL candidates per generation
N_GENERATIONS       = 100    # number of evolutionary generations
ELITE_FRACTION      = 0.10   # top 10% survive unchanged to next generation
CROSSOVER_PROB      = 0.70   # probability two selected parents mate (vs. clone)
MUTATION_PROB       = 0.35   # probability any individual undergoes mutation
TOURNAMENT_K        = 5      # number of candidates in each tournament draw
NOVELTY_WEIGHT      = 0.25   # bonus weight for Tanimoto novelty vs training set
                              # final fitness = model_score + NOVELTY_WEIGHT * novelty
NOVELTY_FP_NBITS    = 1024   # Morgan bits for novelty Tanimoto (separate from model FP)
NOVELTY_RADIUS      = 2      # Morgan radius for novelty fingerprint
NOVELTY_N_REFS      = 50     # compare each candidate against this many training ILs
                              # (random sample per evaluation for speed)
RANDOM_SEED         = 42
TOP_K_RESULTS       = 100    # how many candidates to write to results CSV
MIN_ALKYL_CHAIN_LEN = 1      # shortest allowed C chain after shortening mutation
MAX_ALKYL_CHAIN_LEN = 12     # longest allowed C chain after extension mutation

# ── Condition to use for model inference ─────────────────────────────────────
# We evaluate all candidates at ambient conditions (same as virtual library).
SCREEN_T_K   = 298.15
SCREEN_P_KPA = 101.325

# ── Mutation operator weights (must sum to 1.0) ───────────────────────────────
# Higher weight = applied more often.
MUTATION_WEIGHTS = {
    "alkyl_chain_extend":    0.20,
    "alkyl_chain_shorten":   0.10,
    "swap_whole_cation":     0.25,
    "swap_whole_anion":      0.25,
    "add_functional_group":  0.10,
    "swap_functional_group": 0.10,
}


# ── Building-block libraries ──────────────────────────────────────────────────
# These are inherited directly from build_virtual_library_expanded.py.
# The GA uses them as: (1) seeds for initial population, (2) whole-ion swap pool,
# (3) functional group additions. Having 50+ cations and 30+ anions means the GA
# can generate thousands of structurally distinct candidates.

CATION_POOL = [
    # ── Imidazolium ──
    "CCn1cc[n+](C)c1",            # [EMIM]+
    "CCCCn1cc[n+](C)c1",          # [BMIM]+
    "CCCCCCn1cc[n+](C)c1",        # [HMIM]+
    "CCCCCCCCn1cc[n+](C)c1",      # [OMIM]+
    "Cn1cc[n+](C)c1",             # [MMIM]+
    "CCCn1cc[n+](C)c1",           # [PMIM]+
    "CCOn1cc[n+](C)c1",           # [OHEMIM]+ hydroxyethyl
    "N#CCCn1cc[n+](C)c1",         # [CNpMIM]+ cyanopropyl
    "FC(F)(F)CCn1cc[n+](C)c1",    # [TFMIM]+ trifluoroethyl
    "COCCn1cc[n+](C)c1",          # [MeOEMIM]+ methoxyethyl
    "SCCn1cc[n+](C)c1",           # [thioEMIM]+ thioethyl
    # ── Pyrrolidinium ──
    "C[N+]1(C)CCCC1",             # [C1MPyrr]+
    "CCCC[N+]1(C)CCCC1",          # [BMPyrr]+
    "CCCCCC[N+]1(C)CCCC1",        # [HMPyrr]+
    "CC(=O)OCC[N+]1(C)CCCC1",     # acetate-ester pyrrolidinium
    # ── Ammonium ──
    "CC[N+](CC)(CC)CC",            # [TEA]+
    "CCCC[N+](C)(C)C",             # [N1114]+
    "NCC[N+](C)(C)C",              # [AETMA]+ — top Phase 4 cation
    "OCCC[N+](C)(C)C",             # [OHTMA]+
    "OCC[N+](C)(C)C",              # [choline]+
    "NCC[N+](CC)(CC)CC",           # [AETEA]+
    "NCCC[N+](C)(C)C",             # [APTMA]+ aminopropyl
    "CNCC[N+](C)(C)C",             # [MeAETMA]+
    "FC(F)(F)CC[N+](C)(C)C",       # [TFE-TMA]+
    "C#CCC[N+](C)(C)C",            # [propargyl-TMA]+
    # ── Phosphonium ──
    "CCCC[P+](CCCC)(CCCC)CCCC",    # [P4444]+
    "CC[P+](CC)(CC)CC",             # [P2222]+
    "CCCC[P+](CCCC)(CCCC)CCO",     # [P4441OH]+
    "CCCC[P+](CCCC)(CCCC)CCC#N",   # [P444CN]+
    # ── Piperidinium ──
    "CCCC[N+]1(C)CCCCC1",          # [BMPip]+
    "CCCCCC[N+]1(C)CCCCC1",        # [HMPip]+
    "OCC[N+]1(C)CCCCC1",           # [OH-EMPip]+
    "NCC[N+]1(C)CCCCC1",           # [NH2-EMPip]+
    # ── Morpholinium ──
    "CCCC[N+]1(C)CCOCC1",          # [BMMor]+
    "CC[N+]1(C)CCOCC1",            # [EMMor]+
    "CCCCCC[N+]1(C)CCOCC1",        # [HMMor]+
    # ── Sulfonium ──
    "CC[S+](CC)CC",                # triethylsulfonium
    "CCCC[S+](CC)CC",              # butyldiethylsulfonium
    "CC[S+](Cc1ccccc1)CC",         # benzyldiethylsulfonium
    # ── Guanidinium ──
    "CN(C)C(=[N+](C)C)N(C)C",     # hexamethylguanidinium
    "CCN(CC)C(=[N+](CC)CC)N(CC)CC", # hexaethylguanidinium
    # ── Oxazolinium ──
    "CC[N+]1=CCCO1",               # 3-ethyl-1,3-oxazolinium
    "CCCC[N+]1=CCCO1",             # 3-butyl-1,3-oxazolinium
    # ── Pyridinium ──
    "CCCC[n+]1ccc(C)cc1",          # [4-MeBuPy]+
    "CCCCCC[n+]1ccccc1",           # [HexPy]+
    "CCCC[n+]1ccccc1",             # [BuPy]+
]

ANION_POOL = [
    # ── Fluorinated ──
    "F[B-](F)(F)F",                            # [BF4]-
    "F[P-](F)(F)(F)(F)F",                      # [PF6]-
    "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F", # [Tf2N]-
    "O=S(=O)([O-])C(F)(F)F",                   # [OTf]-
    "O=S(=O)([N-]S(=O)(=O)F)F",               # [FSI]-
    "FC(F)(F)C(=O)[O-]",                       # [TFA]-
    "F[P-](F)(F)(C(F)(F)F)(C(F)(F)F)C(F)(F)F",# [FAP]-
    "F[Sb-](F)(F)(F)(F)F",                     # [SbF6]-
    # ── Carboxylate ──
    "CC([O-])=O",                              # [OAc]-
    "OC([O-])=O",                              # [HCO3]-
    "CC(O)C([O-])=O",                          # [lactate]-
    "[O-]C(=O)c1ccccc1",                       # [benzoate]-
    # ── Sulfonate ──
    "CCCCOS([O-])(=O)=O",                      # [BuSO4]-
    "CS([O-])(=O)=O",                          # [MeSO3]-
    "O=S(=O)([O-])c1ccc(C)cc1",               # [TsO]- tosylate
    "O=S(=O)([O-])CCO",                        # [isethionate]-
    # ── Nitrogen-based ──
    "N(=O)[N-]C#N",                            # [DCA]-
    "[B-](C#N)(C#N)(C#N)C#N",                # [TCB]-
    "[N-](S(=O)(=O)F)C#N",                    # [FSCN]-
    # ── Halide ──
    "[Cl-]",
    "[Br-]",
    "[I-]",
    # ── Heterocyclic ──
    "O=C1NS(=O)(=O)c2ccccc21",                # [saccharinate]-
    "CC1=CC(=O)[N-]S1(=O)=O",                # [acesulfamate]-
    "O=C1[N-]C(=O)c2ccccc21",                # [phthalimide]-
    # ── Phosphate ──
    "CCOP([O-])(=O)OCC",                       # [DEP]-
    "COP([O-])(=O)OC",                         # [DMP]-
    # ── Thiocyanate / cyanate ──
    "[S-]C#N",                                 # [SCN]-
    "[O-]C#N",                                 # [OCN]-
]

# Functional groups that can be added to or swapped on the cation.
# Expressed as SMILES fragments that are appended to existing chain ends.
# Only chain-end additions (no ring insertions) to keep SMILES valid.
FUNCTIONAL_GROUP_FRAGMENTS = [
    "O",     # -OH (hydroxyl)
    "N",     # -NH2 (amine — chemical CO2 absorption)
    "C#N",   # -CN (nitrile — polar, CO2 affinity)
    "F",     # -F (fluorine — lowers viscosity)
    "OC",    # -OMe (ether)
    "SC",    # -SMe (thioether)
    "C(F)(F)F",  # -CF3 (trifluoromethyl — low surface tension)
    "C=O",   # -C=O (carbonyl)
    "CC",    # -C2H5 extension (alkyl chain growth, same as alkyl_chain_extend)
]


# ── Novelty fingerprint generator (separate from model's Morgan generator) ────
_NOVELTY_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=NOVELTY_RADIUS,
    fpSize=NOVELTY_FP_NBITS,
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: SMILES VALIDATION UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def validate_smiles(smiles: str) -> bool:
    """Return True if RDKit can parse the SMILES without errors."""
    if not smiles or not isinstance(smiles, str):
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def canonical_smiles(smiles: str) -> str | None:
    """
    Return the RDKit canonical form of a SMILES string.
    Canonical SMILES are unique: the same molecule always maps to the same string.
    Used for deduplication across the population and against training set.
    Returns None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def make_il_smiles(cation_smiles: str, anion_smiles: str) -> str | None:
    """
    Combine cation and anion into an IL SMILES string (dot notation).
    Validates both components first. Returns None if either is invalid.
    """
    if not validate_smiles(cation_smiles) or not validate_smiles(anion_smiles):
        return None
    return f"{cation_smiles}.{anion_smiles}"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: NOVELTY SCORING
# ══════════════════════════════════════════════════════════════════════════════

def compute_morgan_fp_for_novelty(smiles: str):
    """
    Compute a Morgan fingerprint as an RDKit ExplicitBitVect object.
    Used for Tanimoto similarity, NOT the same fingerprint as the model uses.
    Returns None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _NOVELTY_FP_GEN.GetFingerprint(mol)


def tanimoto_novelty_score(il_smiles: str, reference_fps: list) -> float:
    """
    Compute the novelty of a candidate IL relative to a reference fingerprint set.

    Novelty = 1.0 - max_tanimoto_similarity (to any reference IL).
    - Novelty = 1.0: completely unlike anything in the training set.
    - Novelty = 0.0: identical to a training set IL.

    We take the maximum similarity (most similar reference) because a candidate
    only gets novelty credit if it's genuinely different from ALL known ILs.

    We compare the *full IL* fingerprint (cation + anion as combined SMILES),
    so a known cation + novel anion still gets partial novelty credit.
    """
    candidate_fp = compute_morgan_fp_for_novelty(il_smiles)
    if candidate_fp is None or not reference_fps:
        return 0.5  # neutral fallback if can't compute

    # Tanimoto similarity of candidate against all reference fingerprints
    similarities = DataStructs.BulkTanimotoSimilarity(candidate_fp, reference_fps)
    max_sim = max(similarities) if similarities else 0.0
    return 1.0 - max_sim  # novelty = 1 - max similarity


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: MODEL INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def build_inference_row(il_smiles: str, feature_cols: list) -> np.ndarray | None:
    """
    Featurize one IL and build a single-row feature matrix for model.predict().

    Uses featurize_il_smiles (same function as the training pipeline) to
    guarantee the feature vector is identical to what the model was trained on.
    Appends T_K and P_kPa as the last two features (same order as build_dataset.py).

    Returns None if featurization fails (invalid SMILES or descriptor error).
    """
    feat_dict = featurize_il_smiles(il_smiles)
    if feat_dict is None:
        return None

    # Build the feature row in the exact column order the model expects
    row = []
    for col in feature_cols:
        if col == "T_K":
            row.append(SCREEN_T_K)
        elif col == "P_kPa":
            row.append(SCREEN_P_KPA)
        else:
            # Get the molecular feature; fill NaN with 0 (safe for tree models)
            val = feat_dict.get(col, 0.0)
            row.append(0.0 if (val is None or (isinstance(val, float) and np.isnan(val))) else val)

    return np.array(row, dtype=np.float32).reshape(1, -1)


def predict_solubility(il_smiles: str, model, feature_cols: list) -> float | None:
    """
    Predict log10(x2_CO2) for one IL using the loaded forward model.
    Returns None if featurization fails.
    Higher value = better CO2 absorption (more negative log10 → worse).
    """
    X_row = build_inference_row(il_smiles, feature_cols)
    if X_row is None:
        return None
    try:
        return float(model.predict(X_row)[0])
    except Exception as err:
        print(f"  [WARN] Prediction failed for {il_smiles[:40]}: {err}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: FITNESS FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def compute_fitness(il_smiles: str, model, feature_cols: list,
                    reference_fps: list, known_smiles_set: set) -> float:
    """
    Compute the GA fitness score for one IL candidate.

    Fitness = model_score + NOVELTY_WEIGHT * novelty_score

    Where:
      model_score  = predicted log10(x2_CO2) — higher is better CO2 absorption
      novelty_score = Tanimoto novelty vs training set (0=known, 1=completely novel)

    WHY ADD NOVELTY:
      Without it, the GA converges on near-copies of the best training ILs —
      useful for confirmation but not for discovery. The novelty bonus pushes
      the GA to explore genuinely new chemical space while still optimizing
      for high predicted CO2 absorption.

    Returns -999.0 as a sentinel for invalid candidates (invalid SMILES,
    featurization failure) so they sink to the bottom of any ranking.
    """
    # Reject invalid SMILES immediately
    if not il_smiles or il_smiles not in {il_smiles}:  # always True — placeholder hook
        pass
    il_canon = canonical_smiles(il_smiles)
    if il_canon is None:
        return -999.0  # invalid SMILES

    model_score = predict_solubility(il_smiles, model, feature_cols)
    if model_score is None:
        return -999.0  # featurization failure

    novelty = tanimoto_novelty_score(il_smiles, reference_fps)
    return model_score + NOVELTY_WEIGHT * novelty


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: MUTATION OPERATORS
# ══════════════════════════════════════════════════════════════════════════════

def _extend_alkyl_chain(smiles: str) -> str | None:
    """
    Find the longest contiguous -CH2- chain in the SMILES string and add one
    more -C- unit. Works by regex pattern matching on the SMILES string.

    This mimics the most common homologation reaction in IL synthesis:
    going from e.g. [BMIM]+ (butyl) to [HxMIM]+ (hexyl) by adding two carbons.

    Returns modified SMILES, or None if no alkyl chain is found or result is invalid.
    """
    # Pattern: longest run of 'C' characters not inside brackets (aliphatic carbons)
    # We look for 'CC' minimally and extend the first match found
    match = re.search(r'(?<![A-Za-z\[])C{2,}(?![A-Za-z\]])(?=[^(])', smiles)
    if match is None:
        # No standalone alkyl chain — try a simpler single-C match
        match = re.search(r'(?<![A-Za-z\[])C(?![A-Za-z0-9\]=\)])(?=[^(])', smiles)
    if match is None:
        return None

    # Count carbons in the match
    chain_str = match.group()
    if len(chain_str) >= MAX_ALKYL_CHAIN_LEN:
        return None  # already at max length

    # Insert one more C into the chain
    new_chain = chain_str + "C"
    new_smiles = smiles[:match.start()] + new_chain + smiles[match.end():]

    # Validate result
    if validate_smiles(new_smiles):
        return new_smiles
    return None


def _shorten_alkyl_chain(smiles: str) -> str | None:
    """
    Remove one carbon from the longest aliphatic C chain in the SMILES.
    Opposite of _extend_alkyl_chain.
    Returns None if chain would become too short or result is invalid.
    """
    # Find all runs of aliphatic carbons
    matches = list(re.finditer(r'(?<![A-Za-z\[])C{2,}(?![A-Za-z\]])', smiles))
    if not matches:
        return None

    # Pick the longest match
    longest = max(matches, key=lambda m: len(m.group()))
    chain_str = longest.group()
    if len(chain_str) <= MIN_ALKYL_CHAIN_LEN + 1:
        return None  # would be too short

    new_chain = chain_str[:-1]  # remove one C from the end
    new_smiles = smiles[:longest.start()] + new_chain + smiles[longest.end():]

    if validate_smiles(new_smiles):
        return new_smiles
    return None


def _add_functional_group(cation_smiles: str, rng: random.Random) -> str | None:
    """
    Append a random functional group fragment to the end of a terminal carbon
    in the cation SMILES. This is a crude but valid approximation of N-functionalization.

    Strategy: find a terminal aliphatic C at the end of the SMILES string
    and append the fragment. Many fragments won't produce valid SMILES —
    we validate and return None if invalid.
    """
    group = rng.choice(FUNCTIONAL_GROUP_FRAGMENTS)

    # Try appending at the literal end of the SMILES string
    # (works well for chain-terminated cations like CCCC[N+](C)(C)C)
    candidate = cation_smiles + group
    if validate_smiles(candidate):
        return candidate

    # Try inserting before the ring or charge center marker
    for marker in ["[N+]", "[P+]", "[S+]", "[n+]"]:
        if marker in cation_smiles:
            idx = cation_smiles.index(marker)
            candidate = cation_smiles[:idx] + group + cation_smiles[idx:]
            if validate_smiles(candidate):
                return candidate

    return None  # none of the insertion attempts worked


def mutate(cation_smiles: str, anion_smiles: str,
           rng: random.Random) -> tuple[str, str] | None:
    """
    Apply one random mutation operator to an (cation, anion) gene pair.

    Selects operator by weighted random choice (MUTATION_WEIGHTS).
    Tries up to 5 times per operator if the first attempt produces invalid SMILES.
    Returns new (cation, anion) tuple, or None if all attempts failed.
    """
    ops = list(MUTATION_WEIGHTS.keys())
    weights = list(MUTATION_WEIGHTS.values())
    operator = rng.choices(ops, weights=weights, k=1)[0]

    # Try the selected operator (with multiple attempts for stochastic ops)
    for _attempt in range(5):
        if operator == "alkyl_chain_extend":
            new_cat = _extend_alkyl_chain(cation_smiles)
            if new_cat is not None:
                return (new_cat, anion_smiles)

        elif operator == "alkyl_chain_shorten":
            new_cat = _shorten_alkyl_chain(cation_smiles)
            if new_cat is not None:
                return (new_cat, anion_smiles)

        elif operator == "swap_whole_cation":
            new_cat = rng.choice(CATION_POOL)
            if new_cat != cation_smiles and validate_smiles(new_cat):
                return (new_cat, anion_smiles)

        elif operator == "swap_whole_anion":
            new_ani = rng.choice(ANION_POOL)
            if new_ani != anion_smiles and validate_smiles(new_ani):
                return (cation_smiles, new_ani)

        elif operator == "add_functional_group":
            new_cat = _add_functional_group(cation_smiles, rng)
            if new_cat is not None:
                return (new_cat, anion_smiles)

        elif operator == "swap_functional_group":
            # Remove a terminal group from cation, then add a new one
            shortened = _shorten_alkyl_chain(cation_smiles)
            if shortened is not None:
                new_cat = _add_functional_group(shortened, rng)
                if new_cat is not None:
                    return (new_cat, anion_smiles)

    return None  # mutation failed after all attempts


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: CROSSOVER
# ══════════════════════════════════════════════════════════════════════════════

def crossover(parent1: tuple, parent2: tuple,
              rng: random.Random) -> tuple[tuple, tuple]:
    """
    Produce two offspring by swapping either cations or anions between parents.

    Parent1 = (cat1, ani1), Parent2 = (cat2, ani2)
    Cation-swap:  Child1 = (cat2, ani1),  Child2 = (cat1, ani2)
    Anion-swap:   Child1 = (cat1, ani2),  Child2 = (cat2, ani1)

    Both choices are valid IL structures (cations and anions are interchangeable
    building blocks), so crossover always produces valid candidates here.
    We still validate with RDKit as a safety check.

    Returns two offspring tuples. Falls back to cloning parents if validation fails.
    """
    cat1, ani1 = parent1
    cat2, ani2 = parent2

    # Choose whether to swap cations or anions
    if rng.random() < 0.5:
        # swap cations
        child1 = (cat2, ani1)
        child2 = (cat1, ani2)
    else:
        # swap anions
        child1 = (cat1, ani2)
        child2 = (cat2, ani1)

    # Validate both children (building-block swap almost always valid)
    if (validate_smiles(child1[0]) and validate_smiles(child1[1]) and
            validate_smiles(child2[0]) and validate_smiles(child2[1])):
        return child1, child2

    # Fallback: return clones of parents
    return parent1, parent2


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def tournament_select(population: list, fitnesses: list,
                      rng: random.Random) -> tuple:
    """
    Tournament selection: randomly draw TOURNAMENT_K candidates and return
    the one with the highest fitness.

    Tournament selection balances exploration (random candidates enter each
    tournament) with exploitation (only the best candidate of each tournament
    advances). It's preferred over roulette selection because it doesn't
    require fitness normalization and handles negative fitness values correctly.
    """
    indices = rng.sample(range(len(population)), k=min(TOURNAMENT_K, len(population)))
    winner_idx = max(indices, key=lambda i: fitnesses[i])
    return population[winner_idx]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: POPULATION INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def seed_initial_population(train_df: pd.DataFrame, rng: random.Random) -> list:
    """
    Build the generation-0 population of (cation_smiles, anion_smiles) tuples.

    Seeding strategy (three sources):
      1. Top-50 best training ILs (highest log10(x2_CO2) in training set).
         These give the GA a head start from known high-performers.
      2. Random combinatorial pairs from CATION_POOL × ANION_POOL.
         Fills out the remaining population slots with novel candidate seeds.
      3. Any IL from training set (random sample) to maintain diversity.

    WHY MIX SOURCES:
      Seeding only from training data → GA just rediscovers known ILs.
      Seeding only from random pairs → GA starts cold and converges slowly.
      Mix → starts near good solutions and explores broad chemical space.
    """
    population = []
    seen = set()  # deduplicate on (cation_smiles, anion_smiles)

    # Source 1: Top-50 training ILs by mean log10(x2_CO2)
    if "log_x2_CO2" in train_df.columns and "cation_smiles" in train_df.columns:
        il_means = train_df.groupby(["cation_smiles", "anion_smiles"])["log_x2_CO2"].mean()
        top_train = il_means.nlargest(50).reset_index()
        print(f"[seed] Seeding with top {len(top_train)} training ILs "
              f"(mean log10(x2) range: [{top_train['log_x2_CO2'].min():.2f}, "
              f"{top_train['log_x2_CO2'].max():.2f}])")
        for _, row in top_train.iterrows():
            cat = row.get("cation_smiles", "")
            ani = row.get("anion_smiles", "")
            if validate_smiles(cat) and validate_smiles(ani):
                key = (cat, ani)
                if key not in seen:
                    population.append(key)
                    seen.add(key)

    # Source 2: Random combinatorial pairs from pool
    n_random = POPULATION_SIZE - len(population)
    print(f"[seed] Adding {n_random} random pool combinations to reach {POPULATION_SIZE}")
    attempts = 0
    while len(population) < POPULATION_SIZE and attempts < POPULATION_SIZE * 20:
        cat = rng.choice(CATION_POOL)
        ani = rng.choice(ANION_POOL)
        key = (cat, ani)
        if key not in seen and validate_smiles(cat) and validate_smiles(ani):
            population.append(key)
            seen.add(key)
        attempts += 1

    print(f"[seed] Initial population size: {len(population)} unique individuals")
    return population


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9: DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_cols() -> tuple:
    """
    Load the trained forward model and its feature column list from disk.
    The model was saved by train_model.py as a dict: {'model': ..., 'feature_cols': ...}.
    Raises FileNotFoundError if the model file is missing.
    """
    if not os.path.exists(MODEL_PKL):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PKL}. "
            "Run src/train_model.py first to train the forward model."
        )
    saved = joblib.load(MODEL_PKL)
    model = saved["model"]
    feature_cols = saved["feature_cols"]
    print(f"[load_model] Loaded forward model with {len(feature_cols)} features")
    return model, feature_cols


def load_training_data() -> tuple[pd.DataFrame, set, list]:
    """
    Load train + test CSVs.
    Returns:
      train_df       -- full training DataFrame (for population seeding)
      known_smiles   -- set of all known IL SMILES (to flag rediscoveries)
      reference_fps  -- list of RDKit fingerprint objects for novelty scoring
    """
    train_df = pd.DataFrame()
    known_smiles = set()
    reference_fps = []

    for csv_path, label in [(TRAIN_CSV, "train"), (TEST_CSV, "test")]:
        if not os.path.exists(csv_path):
            print(f"[load_data] WARNING: {csv_path} not found — skipping")
            continue
        df = pd.read_csv(csv_path)
        if label == "train":
            train_df = df
        unique_il_smiles = df["il_smiles"].dropna().unique()
        known_smiles.update(unique_il_smiles)
        for smi in unique_il_smiles:
            fp = compute_morgan_fp_for_novelty(smi)
            if fp is not None:
                reference_fps.append(fp)

    print(f"[load_data] Known ILs: {len(known_smiles)} | "
          f"Reference fingerprints: {len(reference_fps)}")

    # Subsample reference FPs for speed (random NOVELTY_N_REFS used per evaluation)
    if len(reference_fps) > NOVELTY_N_REFS:
        print(f"[load_data] Subsampling reference FPs to {NOVELTY_N_REFS} for speed")

    return train_df, known_smiles, reference_fps


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10: MAIN GA LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_ga(model, feature_cols: list, train_df: pd.DataFrame,
           known_smiles: set, reference_fps: list,
           rng: random.Random) -> tuple[list, list]:
    """
    Run the genetic algorithm for N_GENERATIONS generations.

    Each generation:
      1. Score all individuals (fitness function = model + novelty).
      2. Print generation summary (best, mean fitness).
      3. Keep elite (top ELITE_FRACTION) unchanged.
      4. Fill remaining slots via tournament selection + crossover + mutation.
      5. Validate all new individuals (drop invalid SMILES).

    Returns:
      best_individuals  -- list of (cation, anion, fitness, il_smiles) tuples
                           for all valid individuals ever evaluated, sorted by fitness
      gen_log           -- list of dicts with per-generation stats
    """
    n_elite = max(1, int(POPULATION_SIZE * ELITE_FRACTION))
    population = seed_initial_population(train_df, rng)
    gen_log = []

    # Archive: store ALL valid evaluated individuals for final ranking
    # Key: canonical_il_smiles → best fitness seen
    fitness_archive: dict[str, float] = {}

    print(f"\n=== STARTING GA: {POPULATION_SIZE} individuals, {N_GENERATIONS} generations ===")
    print(f"    Elite={n_elite}, CrossoverP={CROSSOVER_PROB}, MutationP={MUTATION_PROB}")
    print(f"    NoveltyWeight={NOVELTY_WEIGHT}\n")

    for gen_idx in range(N_GENERATIONS):
        # ── Step 1: Evaluate fitness for entire population ───────────────────
        fitnesses = []
        for cation_smi, anion_smi in population:
            il_smi = make_il_smiles(cation_smi, anion_smi)
            if il_smi is None:
                fitnesses.append(-999.0)
                continue

            # Subsample reference FPs for speed
            ref_sample = (rng.sample(reference_fps, NOVELTY_N_REFS)
                          if len(reference_fps) > NOVELTY_N_REFS else reference_fps)

            fit = compute_fitness(il_smi, model, feature_cols,
                                  ref_sample, known_smiles)
            fitnesses.append(fit)

            # Archive this individual (keep best fitness if seen multiple times)
            il_canon = canonical_smiles(il_smi)
            if il_canon and fit > -999.0:
                if il_canon not in fitness_archive or fit > fitness_archive[il_canon]:
                    fitness_archive[il_canon] = fit

        # ── Step 2: Generation statistics ───────────────────────────────────
        valid_fits = [f for f in fitnesses if f > -999.0]
        best_fit   = max(valid_fits) if valid_fits else -999.0
        mean_fit   = np.mean(valid_fits) if valid_fits else -999.0
        best_idx   = fitnesses.index(best_fit)
        best_cat, best_ani = population[best_idx]

        gen_log.append({
            "generation":     gen_idx,
            "best_fitness":   best_fit,
            "mean_fitness":   mean_fit,
            "n_valid":        len(valid_fits),
            "best_il_smiles": f"{best_cat}.{best_ani}",
        })

        # Print every 5 generations to avoid log spam
        if gen_idx % 5 == 0 or gen_idx == N_GENERATIONS - 1:
            print(f"  Gen {gen_idx:3d}/{N_GENERATIONS}: "
                  f"best_fitness={best_fit:.4f}  mean={mean_fit:.4f}  "
                  f"valid={len(valid_fits)}/{POPULATION_SIZE}  "
                  f"archive={len(fitness_archive)}")

        # ── Step 3: Build next generation ───────────────────────────────────
        # Elitism: carry best n_elite individuals unchanged
        sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i],
                                reverse=True)
        new_population = [population[i] for i in sorted_indices[:n_elite]]

        # Fill remaining slots with tournament selection + crossover + mutation
        while len(new_population) < POPULATION_SIZE:
            # Select parents via tournament
            parent1 = tournament_select(population, fitnesses, rng)
            parent2 = tournament_select(population, fitnesses, rng)

            # Crossover
            if rng.random() < CROSSOVER_PROB:
                child1, child2 = crossover(parent1, parent2, rng)
            else:
                child1, child2 = parent1, parent2

            # Mutation (applied independently to each child)
            for child in [child1, child2]:
                if len(new_population) >= POPULATION_SIZE:
                    break
                if rng.random() < MUTATION_PROB:
                    mutant = mutate(child[0], child[1], rng)
                    if mutant is not None:
                        new_population.append(mutant)
                    else:
                        new_population.append(child)  # mutation failed: keep original
                else:
                    new_population.append(child)

        population = new_population[:POPULATION_SIZE]

    print(f"\n=== GA COMPLETE: {len(fitness_archive)} unique valid ILs evaluated ===")
    return fitness_archive, gen_log


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11: RESULTS PROCESSING AND OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def build_results_dataframe(fitness_archive: dict, known_smiles: set,
                             model, feature_cols: list) -> pd.DataFrame:
    """
    Convert the fitness archive into a ranked results DataFrame.

    For each candidate:
      - Splits IL SMILES into cation + anion
      - Flags whether it's a known training IL or a novel discovery
      - Re-predicts log10(x2_CO2) without novelty bonus for clean reporting
      - Sorts by predicted solubility (the model score alone)

    WHY RE-PREDICT WITHOUT NOVELTY:
      Fitness = model_score + novelty_bonus. For the final table, we want
      to report pure model_score so judges see actual predicted CO2 absorption,
      not a fitness that includes a structural novelty bonus.
    """
    rows = []
    for il_canon, fitness in fitness_archive.items():
        if "." not in il_canon:
            continue  # not a valid IL SMILES (no cation.anion dot)

        # Split at first dot (handles multi-fragment anions)
        dot_idx = il_canon.index(".")
        cation_smi = il_canon[:dot_idx]
        anion_smi  = il_canon[dot_idx + 1:]

        # Re-predict without novelty bonus for clean reporting
        raw_prediction = predict_solubility(il_canon, model, feature_cols)
        if raw_prediction is None:
            continue

        is_known = il_canon in known_smiles

        rows.append({
            "il_smiles":              il_canon,
            "cation_smiles":          cation_smi,
            "anion_smiles":           anion_smi,
            "predicted_log10_x2":     raw_prediction,
            "predicted_x2_co2":       10 ** raw_prediction,
            "ga_fitness":             fitness,
            "is_known_training_il":   is_known,
            "T_K":                    SCREEN_T_K,
            "P_kPa":                  SCREEN_P_KPA,
        })

    results_df = pd.DataFrame(rows)
    if len(results_df) == 0:
        print("[results] WARNING: No valid candidates in archive!")
        return results_df

    # Sort by predicted log10(x2_CO2) descending (higher = better absorption)
    results_df = results_df.sort_values("predicted_log10_x2", ascending=False)
    results_df = results_df.reset_index(drop=True)

    # Separate novel vs known counts
    n_novel = (~results_df["is_known_training_il"]).sum()
    n_known = results_df["is_known_training_il"].sum()
    print(f"[results] {n_novel} novel ILs + {n_known} known ILs in archive")
    print(f"[results] Best predicted log10(x2): {results_df['predicted_log10_x2'].iloc[0]:.4f}")
    print(f"[results] Best novel IL: "
          f"{results_df[~results_df['is_known_training_il']].iloc[0]['il_smiles'][:60] if n_novel > 0 else 'N/A'}")

    return results_df


def plot_fitness_curve(gen_log: list) -> None:
    """
    Plot best and mean fitness per generation to visualize GA convergence.
    A plateau in best_fitness indicates the GA has converged — more generations
    won't improve results. A still-rising curve suggests more generations would help.
    """
    gen_df = pd.DataFrame(gen_log)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gen_df["generation"], gen_df["best_fitness"],
            color="steelblue", linewidth=2, label="Best fitness")
    ax.plot(gen_df["generation"], gen_df["mean_fitness"],
            color="coral", linewidth=1.5, linestyle="--", label="Mean fitness")

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Fitness (log₁₀(x₂) + novelty bonus)", fontsize=12)
    ax.set_title("GA Convergence: CO₂ Solubility Optimization", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(GA_PLOT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Fitness convergence curve saved → {GA_PLOT_PNG}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12: MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Full GA pipeline:
      1. Load model, training data, reference fingerprints.
      2. Run genetic algorithm for N_GENERATIONS.
      3. Build ranked results DataFrame.
      4. Save top candidates and generational log.
      5. Plot convergence curve.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    rng = random.Random(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=== STEP 1: Loading model and training data ===")
    model, feature_cols = load_model_and_cols()
    train_df, known_smiles, reference_fps = load_training_data()

    print(f"\n=== STEP 2: Validating building-block pools ===")
    valid_cations = [s for s in CATION_POOL if validate_smiles(s)]
    valid_anions  = [s for s in ANION_POOL  if validate_smiles(s)]
    print(f"  Cation pool: {len(valid_cations)}/{len(CATION_POOL)} valid")
    print(f"  Anion pool:  {len(valid_anions)}/{len(ANION_POOL)} valid")
    max_combinatorial = len(valid_cations) * len(valid_anions)
    print(f"  Max combinatorial space: {max_combinatorial} unique ILs")
    print(f"  GA will explore this + mutations & crossovers beyond this space")

    print(f"\n=== STEP 3: Running genetic algorithm ===")
    fitness_archive, gen_log = run_ga(
        model, feature_cols, train_df, known_smiles, reference_fps, rng
    )

    print(f"\n=== STEP 4: Building results DataFrame ===")
    results_df = build_results_dataframe(
        fitness_archive, known_smiles, model, feature_cols
    )

    if len(results_df) == 0:
        print("ERROR: No valid candidates found. Check model and data paths.")
        return

    # Save full archive sorted by predicted solubility
    results_df.to_csv(GA_RESULTS_CSV, index=False)
    print(f"[main] Full archive ({len(results_df)} ILs) saved → {GA_RESULTS_CSV}")

    # Print top 10 novel discoveries
    novel_df = results_df[~results_df["is_known_training_il"]].head(10)
    print(f"\n=== TOP 10 NOVEL IL DISCOVERIES ===")
    print(novel_df[["il_smiles", "predicted_log10_x2", "predicted_x2_co2",
                     "cation_smiles", "anion_smiles"]].to_string(index=False))

    # Save generational log
    gen_log_df = pd.DataFrame(gen_log)
    gen_log_df.to_csv(GA_GEN_LOG_CSV, index=False)
    print(f"\n[main] Generational log saved → {GA_GEN_LOG_CSV}")

    # Plot convergence
    print(f"\n=== STEP 5: Plotting convergence curve ===")
    plot_fitness_curve(gen_log)

    print(f"\n=== GA COMPLETE ===")
    print(f"  Total unique ILs evaluated: {len(fitness_archive)}")
    print(f"  Novel ILs discovered:       {(~results_df['is_known_training_il']).sum()}")
    print(f"  Best predicted log10(x2):   {results_df['predicted_log10_x2'].iloc[0]:.4f}")
    best_novel = results_df[~results_df["is_known_training_il"]].iloc[0]
    print(f"  Best NOVEL predicted log10(x2): {best_novel['predicted_log10_x2']:.4f}")
    print(f"  Best NOVEL IL: {best_novel['il_smiles']}")
    print(f"\n  NEXT STEPS:")
    print(f"    1. Review results/ga_top_candidates.csv — examine top novel ILs")
    print(f"    2. Check figures/ga_fitness_curve.png — did GA converge?")
    print(f"    3. If fitness still rising at gen {N_GENERATIONS}, increase N_GENERATIONS")
    print(f"    4. Run src/applicability_domain.py on top candidates to check")
    print(f"       if they fall within the model's training distribution")
    print(f"    5. Top 5 novel candidates → DFT validation with ORCA")


if __name__ == "__main__":
    main()
