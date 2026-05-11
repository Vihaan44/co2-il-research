"""
inverse_design.py
-----------------
PURPOSE: Screen the virtual IL library with the trained XGBoost forward model.
         Rank all candidates by predicted CO2 solubility and save the top candidates
         for DFT validation in Phase 5.

HOW INVERSE DESIGN WORKS (for judges):
  Traditional chemistry: synthesize an IL → measure CO2 absorption → decide if it's good.
  Our approach: generate thousands of IL structures computationally → predict CO2 absorption
  for each → select the top candidates → validate with quantum chemistry (DFT).
  This inverts the discovery process: we start from desired properties and work backward
  to find the structure, which is why it's called 'inverse design'.

SCREENING LOGIC:
  1. Featurize each virtual IL using the same Morgan fingerprint pipeline as training.
  2. Predict log10(x2_CO2) at T=298.15 K, P=101.325 kPa.
  3. Back-transform to mole fraction x2 = 10^log_x2.
  4. Filter out physically unreasonable predictions (x2 > 1 or x2 < 1e-7).
  5. Rank by predicted x2 (descending = highest CO2 solubility first).
  6. Save top 20 candidates for DFT.

WHY LOG SCALE TARGET:
  The model was trained on log10(x2), so predictions are in log10 units.
  Back-transforming (10^y) converts back to mole fraction for chemical interpretability.

LIMITATION (honest, for judges):
  The model extrapolates to novel IL combinations it has never seen. Uncertainty
  is not quantified here (no confidence intervals). DFT validation in Phase 5
  is essential before claiming these candidates are genuinely superior.

INPUTS:
  models/forward_model.pkl                       (XGBoost model from train_model.py)
  data/virtual_library/virtual_il_library.csv   (from build_virtual_library.py)
  src/featurize.py                               (featurization functions — imported directly)

OUTPUTS:
  results/virtual_library_predictions.csv        (all candidates, ranked)
  results/top_candidates.csv                     (top 20 candidates for DFT)

Run from project root:
    python src/inverse_design.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH        = os.path.join("models", "forward_model.pkl")
VIRTUAL_LIB_CSV   = os.path.join("data", "virtual_library", "virtual_il_library.csv")
ALL_PREDS_CSV     = os.path.join("results", "virtual_library_predictions.csv")
TOP_CANDIDATES_CSV = os.path.join("results", "top_candidates.csv")

TOP_N_CANDIDATES  = 20     # how many to save for DFT validation

# Physical plausibility bounds on mole fraction solubility
X2_MIN_PHYSICAL   = 1e-7   # below this is physically implausible for CO2 in IL
X2_MAX_PHYSICAL   = 1.0    # mole fraction cannot exceed 1

# Expected condition columns (must match what the model was trained on)
CONDITION_FEATURES = ["T_K", "P_kPa"]


def load_model(model_path: str):
    """
    Load the serialized forward model from disk.
    The model expects the same feature matrix as built in train_model.py:
    [Morgan FP bits (cat_* / an_*)] + [RDKit descriptors (cat_* / an_*)] + [T_K, P_kPa]
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run src/train_model.py first."
        )
    model = joblib.load(model_path)
    print(f"[load_model] Loaded: {type(model).__name__} from {model_path}")
    return model


def load_virtual_library(csv_path: str) -> pd.DataFrame:
    """
    Load the virtual IL library CSV produced by build_virtual_library.py.
    Checks that required columns exist before proceeding.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Virtual library not found at {csv_path}. Run src/build_virtual_library.py first."
        )
    virtual_lib_df = pd.read_csv(csv_path)
    print(f"[load_virtual_library] {virtual_lib_df.shape[0]} IL candidates loaded")

    required_cols = ["il_smiles", "cation_smiles", "anion_smiles", "T_K", "P_kPa"]
    missing = [c for c in required_cols if c not in virtual_lib_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in virtual library: {missing}")
    return virtual_lib_df


def featurize_virtual_library(virtual_lib_df: pd.DataFrame) -> tuple:
    """
    Convert virtual IL SMILES to the same feature matrix used during model training.

    PLAIN ENGLISH: We add the featurize.py directory to Python's module search path
    so we can import the featurization functions directly, avoiding code duplication.
    This guarantees the virtual library is processed identically to the training data.

    Returns X (feature matrix numpy array), feature_cols (list of column names),
    and valid_rows mask (bool array — False where SMILES failed to parse).
    """
    # Add src/ to path so featurize.py can be imported
    src_dir = os.path.dirname(os.path.abspath(__file__))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # Import featurization functions from Phase 2
    try:
        from featurize import featurize_il_smiles
    except ImportError:
        raise ImportError(
            "Cannot import featurize.py. Make sure src/featurize.py exists and "
            "all RDKit dependencies are installed."
        )

    print(f"[featurize_virtual_library] Featurizing {len(virtual_lib_df)} ILs...")

    feature_rows = []
    valid_mask   = []     # track which rows successfully featurized

    for idx, row in virtual_lib_df.iterrows():
        features = featurize_il_smiles(row["il_smiles"])   # returns dict or None
        if features is None:
            # DATA QUALITY FLAG: SMILES failed to featurize — skip this candidate
            valid_mask.append(False)
            feature_rows.append(None)
        else:
            # Append T_K and P_kPa to match the training feature matrix
            features["T_K"]   = row["T_K"]
            features["P_kPa"] = row["P_kPa"]
            valid_mask.append(True)
            feature_rows.append(features)

        if (idx + 1) % 100 == 0:
            print(f"  ...{idx + 1}/{len(virtual_lib_df)} processed")

    valid_mask = np.array(valid_mask)
    n_failed   = (~valid_mask).sum()
    if n_failed > 0:
        print(f"[featurize_virtual_library] ⚠  {n_failed} ILs failed featurization — dropped")

    # Build feature DataFrame from valid rows only
    valid_feature_rows = [r for r in feature_rows if r is not None]
    feature_df   = pd.DataFrame(valid_feature_rows)
    feature_cols = list(feature_df.columns)

    print(f"[featurize_virtual_library] {len(feature_df)} ILs featurized successfully")
    print(f"[featurize_virtual_library] Feature matrix shape: {feature_df.shape}")

    return feature_df.values, feature_cols, valid_mask


def align_features_to_model(X_virtual: np.ndarray, feature_cols_virtual: list,
                             model) -> np.ndarray:
    """
    Ensure the virtual library feature matrix has the exact same columns,
    in the exact same order, as the training data the model saw.

    Why this matters: XGBoost remembers feature names and their order from training.
    If our virtual library has extra or missing columns, predictions will be wrong.
    We fill any missing columns with 0 (feature absent = bit not set).
    """
    try:
        # XGBoost models trained with feature names expose this attribute
        training_feature_names = model.get_booster().feature_names
    except AttributeError:
        # Random Forest fallback — no built-in feature names, assume order matches
        print("[align_features] No feature name metadata on model — assuming column order matches")
        return X_virtual

    if training_feature_names is None:
        return X_virtual

    virtual_df = pd.DataFrame(X_virtual, columns=feature_cols_virtual)

    # Add any missing columns (fill with 0 = Morgan bit not present)
    for col in training_feature_names:
        if col not in virtual_df.columns:
            virtual_df[col] = 0   # missing bit = absent structural feature

    # Reorder to match training column order exactly
    aligned_X = virtual_df[training_feature_names].values

    n_added = sum(1 for c in training_feature_names if c not in feature_cols_virtual)
    if n_added > 0:
        print(f"[align_features] Added {n_added} missing columns (filled with 0)")

    return aligned_X


def apply_physical_filters(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove candidates with physically unreasonable predicted mole fraction solubility.

    Mole fraction must be in (0, 1]. Values outside this range indicate the model
    is extrapolating beyond its training distribution — results would be meaningless.
    """
    initial_count = len(predictions_df)

    physical_mask = (
        (predictions_df["x2_predicted"] >= X2_MIN_PHYSICAL) &
        (predictions_df["x2_predicted"] <= X2_MAX_PHYSICAL)
    )
    filtered_df = predictions_df[physical_mask].copy()

    n_removed = initial_count - len(filtered_df)
    if n_removed > 0:
        # DATA QUALITY FLAG: these ILs are outside model's reliable range
        print(f"[apply_physical_filters] Removed {n_removed} candidates with "
              f"x2 outside [{X2_MIN_PHYSICAL:.0e}, {X2_MAX_PHYSICAL}]")
    else:
        print(f"[apply_physical_filters] All {initial_count} candidates pass physical filter")

    return filtered_df


def main():
    """Full inverse design pipeline: load model → featurize → predict → rank → save."""
    os.makedirs("results", exist_ok=True)

    # ── Step 1: Load model and virtual library ──────────────────────────────
    print("=== STEP 1: Loading model and virtual library ===")
    forward_model   = load_model(MODEL_PATH)
    virtual_lib_df  = load_virtual_library(VIRTUAL_LIB_CSV)

    # ── Step 2: Featurize ───────────────────────────────────────────────────
    print("\n=== STEP 2: Featurizing virtual library ===")
    X_virtual, feature_cols_virtual, valid_mask = featurize_virtual_library(virtual_lib_df)

    # Keep only the rows that featurized successfully
    valid_meta_df = virtual_lib_df[valid_mask].reset_index(drop=True)

    # ── Step 3: Align features to model's expected input ────────────────────
    print("\n=== STEP 3: Aligning feature matrix to model ===")
    X_aligned = align_features_to_model(X_virtual, feature_cols_virtual, forward_model)
    print(f"[main] Final feature matrix: {X_aligned.shape[0]} ILs × {X_aligned.shape[1]} features")

    # ── Step 4: Predict ─────────────────────────────────────────────────────
    print("\n=== STEP 4: Predicting CO2 solubility ===")
    log_x2_predicted = forward_model.predict(X_aligned)   # log10(x2_CO2)
    x2_predicted     = 10 ** log_x2_predicted             # back-transform to mole fraction

    print(f"[main] Prediction range (log10 x2): [{log_x2_predicted.min():.3f}, {log_x2_predicted.max():.3f}]")
    print(f"[main] Mole fraction range (x2):    [{x2_predicted.min():.2e}, {x2_predicted.max():.2e}]")

    # ── Step 5: Assemble results DataFrame ──────────────────────────────────
    predictions_df = valid_meta_df[["il_smiles", "cation_smiles", "anion_smiles",
                                    "T_K", "P_kPa"]].copy()
    predictions_df["log_x2_predicted"] = log_x2_predicted
    predictions_df["x2_predicted"]     = x2_predicted

    # ── Step 6: Physical plausibility filter ────────────────────────────────
    print("\n=== STEP 5: Applying physical filters ===")
    predictions_df = apply_physical_filters(predictions_df)

    # ── Step 7: Rank by predicted CO2 solubility ────────────────────────────
    # Higher x2 = more CO2 dissolved = better capture — sort descending
    predictions_df = predictions_df.sort_values(
        "x2_predicted", ascending=False
    ).reset_index(drop=True)

    predictions_df["rank"] = predictions_df.index + 1   # 1-based rank

    print(f"\n=== STEP 6: Top 10 Candidates ===")
    print(predictions_df.head(10)[["rank", "il_smiles",
                                   "log_x2_predicted",
                                   "x2_predicted"]].to_string(index=False))

    # ── Step 8: Save all predictions and top candidates ─────────────────────
    predictions_df.to_csv(ALL_PREDS_CSV, index=False)
    print(f"\n[main] All predictions saved → {ALL_PREDS_CSV}")

    top_candidates_df = predictions_df.head(TOP_N_CANDIDATES).copy()
    top_candidates_df.to_csv(TOP_CANDIDATES_CSV, index=False)
    print(f"[main] Top {TOP_N_CANDIDATES} candidates saved → {TOP_CANDIDATES_CSV}")

    print(f"\n✓ Phase 4 inverse design complete.")
    print(f"  Best predicted IL: {top_candidates_df.iloc[0]['il_smiles']}")
    print(f"  Predicted x2_CO2: {top_candidates_df.iloc[0]['x2_predicted']:.4e}")
    print(f"  (Compare to training set max x2 for context)")
    print(f"\n  Next step: DFT validation of top candidates → src phase 5 / ORCA")


if __name__ == "__main__":
    main()
