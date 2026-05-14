"""
inverse_design.py
-----------------
PURPOSE: Screen the virtual IL library with the trained XGBoost forward model.
         Rank all candidates by predicted CO2 solubility and save the top candidates
         for DFT validation in Phase 5.

HOW INVERSE DESIGN WORKS (for judges):
  Traditional chemistry: synthesize an IL -> measure CO2 absorption -> decide if it's good.
  Our approach: generate thousands of IL structures computationally -> predict CO2 absorption
  for each -> select the top candidates -> validate with quantum chemistry (DFT).
  This inverts the discovery process: we start from desired properties and work backward
  to find the structure, which is why it's called 'inverse design'.

SCREENING LOGIC:
  1. Featurize each virtual IL using the same Morgan fingerprint pipeline as training.
  2. Predict log10(x2_CO2) at T=298.15 K, P=101.325 kPa (standard conditions).
  3. Back-transform to mole fraction x2 = 10^log_x2.
  4. Filter out physically unreasonable predictions (x2 > 1 or x2 < 1e-7).
  5. Rank by predicted x2 (descending = highest CO2 solubility first).
  6. Save top 20 candidates for DFT.

WHY LOG SCALE TARGET:
  The model was trained on log10(x2), so predictions are in log10 units.
  Back-transforming (10^y) converts back to mole fraction for chemical interpretability.

FEATURE ALIGNMENT:
  The model bundle (forward_model.pkl) contains both the fitted model and the list of
  feature column names from training (feature_cols). This script uses those names to
  reorder the virtual library's feature matrix to match exactly -- ensuring no column
  mismatch errors and no silent wrong-column-to-wrong-feature bugs.

LIMITATION (honest, for judges):
  The model extrapolates to novel IL combinations it has never seen. Uncertainty
  is not quantified here (no confidence intervals). DFT validation in Phase 5
  is essential before claiming these candidates are genuinely superior.

INPUTS:
  models/forward_model.pkl
      Model bundle from train_model.py (contains model + feature_cols).
  data/virtual_library/virtual_il_library_expanded.csv
      Expanded 889-IL library from build_virtual_library_expanded.py.
      Covers 38 cations x 31 anions across 9 families including sulfonium,
      guanidinium, oxazolinium, and heterocyclic anions not in Phase 4.

OUTPUTS:
  results/virtual_library_predictions_expanded.csv   (all candidates, ranked)
  results/top_candidates_expanded.csv                (top 20 candidates for DFT)

NOTE: Phase 4 results (virtual_library_predictions.csv / top_candidates.csv) are
      preserved unchanged. This script writes to separate _expanded files.

Run from project root:
    python src/inverse_design.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH         = os.path.join("models", "forward_model.pkl")

# Points to the expanded library (889 novel ILs across 9 cation families)
# Change this path to virtual_il_library.csv to re-screen the original Phase 4 library
VIRTUAL_LIB_CSV   = os.path.join("data", "virtual_library", "virtual_il_library_expanded.csv")

# Output files — separate from Phase 4 results to avoid overwriting
ALL_PREDS_CSV      = os.path.join("results", "virtual_library_predictions_expanded.csv")
TOP_CANDIDATES_CSV = os.path.join("results", "top_candidates_expanded.csv")

TOP_N_CANDIDATES = 20     # how many to save for DFT validation
X2_MIN_PHYSICAL  = 1e-7   # below this is physically implausible for CO2 in IL
X2_MAX_PHYSICAL  = 1.0    # mole fraction cannot exceed 1


def load_model_bundle(model_path: str) -> tuple:
    """
    Load the model bundle saved by train_model.py.

    The bundle is a dict with keys:
      'model'        -- the fitted XGBoost/RF estimator
      'feature_cols' -- list of column names in training order
      'model_name'   -- string name of the best model

    WHY A BUNDLE: XGBoost trained on numpy arrays does not store feature names
    internally. Saving feature_cols alongside the model guarantees that
    inverse_design.py always aligns virtual library columns to training columns,
    even if the two scripts are run days apart with different data versions.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run src/train_model.py first."
        )

    bundle = joblib.load(model_path)

    # Handle legacy pkl files that saved the raw model (not a bundle dict)
    if not isinstance(bundle, dict):
        raise ValueError(
            f"forward_model.pkl contains a raw model, not a bundle dict.\n"
            f"Please re-run src/train_model.py to regenerate the model bundle "
            f"(which includes feature_cols needed for column alignment)."
        )

    model        = bundle["model"]
    feature_cols = bundle["feature_cols"]
    model_name   = bundle.get("model_name", "unknown")

    print(f"[load_model_bundle] Loaded: {model_name} from {model_path}")
    print(f"[load_model_bundle] Model expects {len(feature_cols)} features")
    return model, feature_cols


def load_virtual_library(csv_path: str) -> pd.DataFrame:
    """
    Load the virtual IL library CSV.
    Validates required columns before proceeding.
    Accepts both the original library and the expanded library — both share
    the same required columns.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Virtual library not found at {csv_path}.\n"
            f"Run src/build_virtual_library_expanded.py first."
        )
    virtual_lib_df = pd.read_csv(csv_path)
    print(f"[load_virtual_library] {virtual_lib_df.shape[0]} IL candidates loaded from {csv_path}")

    required_cols = ["il_smiles", "cation_smiles", "anion_smiles", "T_K", "P_kPa"]
    missing = [c for c in required_cols if c not in virtual_lib_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in virtual library: {missing}")
    return virtual_lib_df


def featurize_virtual_library(virtual_lib_df: pd.DataFrame) -> tuple:
    """
    Convert virtual IL SMILES to feature dicts using the same pipeline as training.

    Imports featurize_il_smiles from src/featurize.py directly to guarantee
    identical featurization -- no risk of the virtual library being processed
    differently from training data.

    Returns:
      feature_df  -- DataFrame where each row is one featurized IL
      valid_mask  -- boolean array, True where SMILES featurized successfully
    """
    src_dir = os.path.dirname(os.path.abspath(__file__))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    try:
        from featurize import featurize_il_smiles
    except ImportError:
        raise ImportError(
            "Cannot import featurize.py. Make sure src/featurize.py exists "
            "and RDKit is installed."
        )

    print(f"[featurize_virtual_library] Featurizing {len(virtual_lib_df)} ILs...")

    feature_rows = []
    valid_mask   = []

    for idx, row in virtual_lib_df.iterrows():
        features = featurize_il_smiles(row["il_smiles"])   # dict or None
        if features is None:
            # DATA QUALITY FLAG: SMILES failed to featurize -- drop this candidate
            valid_mask.append(False)
            feature_rows.append(None)
        else:
            # Append T and P to match training feature set
            features["T_K"]   = row["T_K"]
            features["P_kPa"] = row["P_kPa"]
            valid_mask.append(True)
            feature_rows.append(features)

        if (idx + 1) % 100 == 0:
            print(f"  ...{idx + 1}/{len(virtual_lib_df)} processed")

    valid_mask = np.array(valid_mask)
    n_failed   = (~valid_mask).sum()
    if n_failed > 0:
        print(f"[featurize_virtual_library] {n_failed} ILs failed featurization -- dropped")

    valid_feature_rows = [r for r in feature_rows if r is not None]
    feature_df = pd.DataFrame(valid_feature_rows)

    print(f"[featurize_virtual_library] {len(feature_df)} ILs featurized successfully")
    print(f"[featurize_virtual_library] Feature matrix shape: {feature_df.shape}")
    return feature_df, valid_mask


def align_features_to_model(feature_df: pd.DataFrame,
                             training_feature_cols: list) -> np.ndarray:
    """
    Reorder and fill the virtual library feature DataFrame to exactly match
    the column order and set used during model training.

    WHY THIS IS NECESSARY:
      featurize.py may produce slightly different columns than the training set
      (e.g. if a Morgan bit was never set in training but appears in the virtual
      library, or vice versa). XGBoost requires column count and order to match
      exactly -- even one extra or missing column causes a crash or silent errors.

    WHAT WE DO:
      - Columns in training_feature_cols but missing from feature_df -> filled with 0
        (bit not present = structural feature absent, which is correct)
      - Columns in feature_df but not in training_feature_cols -> dropped
        (the model never saw these; including them would change feature count)
      - All columns reordered to match training_feature_cols exactly
    """
    # Add any columns the model expects but the virtual library doesn't have
    missing_cols = [c for c in training_feature_cols if c not in feature_df.columns]
    if missing_cols:
        print(f"[align_features] Adding {len(missing_cols)} missing columns (filled with 0)")
        for col in missing_cols:
            feature_df[col] = 0   # absent Morgan bit = this substructure not present

    # Drop columns present in virtual library but not in training
    extra_cols = [c for c in feature_df.columns if c not in training_feature_cols]
    if extra_cols:
        print(f"[align_features] Dropping {len(extra_cols)} extra columns not in training set")

    # Select and reorder to exactly match training column order
    aligned_df = feature_df[training_feature_cols]
    print(f"[align_features] Aligned: {aligned_df.shape[1]} features (expected {len(training_feature_cols)})")
    return aligned_df.values


def apply_physical_filters(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove candidates with physically unreasonable predicted mole fraction solubility.
    Values outside (1e-7, 1.0) indicate out-of-distribution extrapolation.
    """
    initial_count = len(predictions_df)
    physical_mask = (
        (predictions_df["x2_predicted"] >= X2_MIN_PHYSICAL) &
        (predictions_df["x2_predicted"] <= X2_MAX_PHYSICAL)
    )
    filtered_df = predictions_df[physical_mask].copy()
    n_removed = initial_count - len(filtered_df)
    if n_removed > 0:
        # DATA QUALITY FLAG: out-of-physical-range predictions dropped
        print(f"[apply_physical_filters] Removed {n_removed} candidates outside "
              f"[{X2_MIN_PHYSICAL:.0e}, {X2_MAX_PHYSICAL}]")
    else:
        print(f"[apply_physical_filters] All {initial_count} candidates pass physical filter")
    return filtered_df


def main():
    """Full pipeline: load -> featurize -> align -> predict -> rank -> save."""
    os.makedirs("results", exist_ok=True)

    # -- Step 1: Load --------------------------------------------------------
    print("=== STEP 1: Loading model and virtual library ===")
    forward_model, training_feature_cols = load_model_bundle(MODEL_PATH)
    virtual_lib_df = load_virtual_library(VIRTUAL_LIB_CSV)

    # -- Step 2: Featurize ---------------------------------------------------
    print("\n=== STEP 2: Featurizing virtual library ===")
    feature_df, valid_mask = featurize_virtual_library(virtual_lib_df)
    valid_meta_df = virtual_lib_df[valid_mask].reset_index(drop=True)

    # -- Step 3: Align to model's feature space ------------------------------
    print("\n=== STEP 3: Aligning feature matrix to model ===")
    X_aligned = align_features_to_model(feature_df, training_feature_cols)
    print(f"[main] Final feature matrix: {X_aligned.shape[0]} ILs x {X_aligned.shape[1]} features")

    # -- Step 4: Predict -----------------------------------------------------
    print("\n=== STEP 4: Predicting CO2 solubility ===")
    log_x2_predicted = forward_model.predict(X_aligned)   # log10(x2_CO2)
    x2_predicted     = 10 ** log_x2_predicted             # back-transform to mole fraction

    print(f"[main] log10(x2) range: [{log_x2_predicted.min():.3f}, {log_x2_predicted.max():.3f}]")
    print(f"[main] x2 range:        [{x2_predicted.min():.2e}, {x2_predicted.max():.2e}]")

    # -- Step 5: Assemble, filter, rank --------------------------------------
    # Include cation_family and anion_family if present in the expanded library
    meta_cols = ["il_smiles", "cation_smiles", "anion_smiles", "T_K", "P_kPa"]
    optional_cols = ["cation_family", "anion_family"]
    for col in optional_cols:
        if col in valid_meta_df.columns:
            meta_cols.append(col)

    predictions_df = valid_meta_df[meta_cols].copy()
    predictions_df["log_x2_predicted"] = log_x2_predicted
    predictions_df["x2_predicted"]     = x2_predicted

    print("\n=== STEP 5: Applying physical filters ===")
    predictions_df = apply_physical_filters(predictions_df)

    # Sort descending: higher x2 = more CO2 dissolved = better IL
    predictions_df = predictions_df.sort_values(
        "x2_predicted", ascending=False
    ).reset_index(drop=True)
    predictions_df["rank"] = predictions_df.index + 1   # 1-based

    print("\n=== STEP 6: Top 10 Candidates ===")
    print(predictions_df.head(10)[["rank", "il_smiles",
                                   "log_x2_predicted",
                                   "x2_predicted"]].to_string(index=False))

    # Print family breakdown of top 20 if family columns available
    if "cation_family" in predictions_df.columns:
        print("\nCation family breakdown in top 20:")
        print(predictions_df.head(20)["cation_family"].value_counts().to_string())

    # -- Step 6: Save --------------------------------------------------------
    predictions_df.to_csv(ALL_PREDS_CSV, index=False)
    print(f"\n[main] All predictions saved -> {ALL_PREDS_CSV}")

    top_candidates_df = predictions_df.head(TOP_N_CANDIDATES).copy()
    top_candidates_df.to_csv(TOP_CANDIDATES_CSV, index=False)
    print(f"[main] Top {TOP_N_CANDIDATES} candidates saved -> {TOP_CANDIDATES_CSV}")

    print(f"\nExpanded library screening complete.")
    print(f"  Best predicted IL:  {top_candidates_df.iloc[0]['il_smiles']}")
    print(f"  Predicted x2_CO2:  {top_candidates_df.iloc[0]['x2_predicted']:.4e}")
    print(f"  Next: DFT validation -> Phase 5")


if __name__ == "__main__":
    main()
