"""
ablate_feature_sets.py
-----------------------
PURPOSE: Evaluate whether each new feature block (MACCS, anion electronic,
         ion-pair cross, Mordred) improves RF generalization to unseen ILs
         beyond the current R2=0.708 baseline (Morgan+18desc, nested CV).

METHOD:
  For each of the 5 feature variants (v1-v5), run 5-fold outer GroupKFold
  (grouped by il_smiles) using the tuned RF hyperparameters. Report mean +/- std
  test R2 per variant. The same protocol as nested_cv_model_comparison.py so
  results are directly comparable to the 0.708 baseline.

DECISION RULE:
  Variant mean test R2 > 0.718 (baseline + 0.01) -> include that feature block
  If multiple variants help -> combine winning blocks -> retrain + consider
  re-tuning RF hyperparameters on the combined feature set

INPUTS:
  data/processed/train_set_v{N}.csv  (from build_dataset_v2.py)
  data/processed/test_set_v{N}.csv   (from build_dataset_v2.py)

OUTPUTS:
  results/feature_ablation.csv         -- mean/std test R2 per variant
  figures/feature_ablation_barplot.png -- visual comparison

Run from project root:
  nohup python src/ablate_feature_sets.py > logs/feature_ablation.log 2>&1 &
Then monitor:
  tail -f logs/feature_ablation.log
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GroupKFold

# -- Constants -----------------------------------------------------------------
DATA_DIR    = os.path.join("data", "processed")
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
LOGS_DIR    = "logs"

TARGET_COL         = "log_x2_CO2"
CONDITION_FEATURES = ["T_K", "P_kPa"]
N_OUTER_FOLDS       = 5
RANDOM_SEED         = 42

# Threshold above which a variant is considered meaningfully better than baseline
IMPROVEMENT_THRESHOLD_R2 = 0.010

# RF: Optuna v2 best params (same throughout project)
RF_PARAMS = dict(
    n_estimators     = 515,
    max_features     = 0.5,
    min_samples_leaf = 7,
    max_depth        = None,
    random_state     = RANDOM_SEED,
    n_jobs           = -1,
)

# Baseline nested CV R2 (from nested_cv_model_comparison.py) for comparison
BASELINE_NESTED_CV_R2  = 0.7078
BASELINE_NESTED_CV_STD = 0.1115

# Variants to evaluate: (variant_name, description, train_csv, test_csv)
# Each variant adds exactly one block on top of baseline so results are clean.
VARIANTS = [
    (
        "v1_baseline",
        "Morgan 2048-bit x2 + 18 RDKit desc x2",
        "train_set_v1_baseline.csv",
        "test_set_v1_baseline.csv",
    ),
    (
        "v2_maccs",
        "Baseline + MACCS keys (166 bits x2 ions)",
        "train_set_v2_maccs.csv",
        "test_set_v2_maccs.csv",
    ),
    (
        "v3_anion_elec",
        "Baseline + anion electronic (6 features)",
        "train_set_v3_anion_elec.csv",
        "test_set_v3_anion_elec.csv",
    ),
    (
        "v4_cross",
        "Baseline + ion-pair cross (4 features)",
        "train_set_v4_cross.csv",
        "test_set_v4_cross.csv",
    ),
    (
        "v5_mordred",
        "Baseline + Mordred 2D (<=200 desc x2 ions)",
        "train_set_v5_mordred.csv",
        "test_set_v5_mordred.csv",
    ),
]


def load_variant(train_csv, test_csv, variant_name):
    """
    Load one variant's train and test CSVs. Returns X, y, il_smiles for each
    split and the list of feature column names. Fills NaN descriptors with
    train-fold median to prevent leakage.
    """
    train_path = os.path.join(DATA_DIR, train_csv)
    test_path  = os.path.join(DATA_DIR, test_csv)

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"[{variant_name}] Train file not found: {train_path}. "
            f"Run build_dataset_v2.py first."
        )

    train_df = pd.read_csv(train_path).dropna(subset=CONDITION_FEATURES).copy()
    test_df  = pd.read_csv(test_path).dropna(subset=CONDITION_FEATURES).copy()

    # All cat_*, an_*, pair_* columns are molecular features
    molecular_cols = [
        c for c in train_df.columns
        if c.startswith("cat_") or c.startswith("an_") or c.startswith("pair_")
    ]
    feature_cols = molecular_cols + CONDITION_FEATURES

    # Fill NaN descriptor columns with train median (non-FP columns only)
    desc_only = [c for c in molecular_cols if "_fp_" not in c and "_maccs_" not in c]
    for col in desc_only:
        if col in train_df.columns and train_df[col].isna().any():
            med = train_df[col].median()
            train_df[col] = train_df[col].fillna(med)
            if col in test_df.columns:
                test_df[col] = test_df[col].fillna(med)

    # Keep only columns present in both splits
    feature_cols = [c for c in feature_cols
                    if c in train_df.columns and c in test_df.columns]

    return (
        train_df[feature_cols].values.astype(float), train_df[TARGET_COL].values,
        train_df["il_smiles"],
        test_df[feature_cols].values.astype(float),  test_df[TARGET_COL].values,
        test_df["il_smiles"],
        feature_cols,
    )


def run_nested_cv(X, y, il_smiles, variant_name):
    """
    Run N_OUTER_FOLDS-fold GroupKFold CV on the full combined pool.
    Uses fold-local median imputation to prevent leakage.
    Returns list of per-fold test R2 values.
    """
    gkf    = GroupKFold(n_splits=N_OUTER_FOLDS)
    groups = il_smiles.values
    fold_scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        # Confirm no IL leakage across the split
        assert len(set(groups[train_idx]) & set(groups[test_idx])) == 0, \
            f"IL overlap in fold {fold_idx} for {variant_name}!"

        # Fit imputer on train fold only to prevent leakage
        imputer = SimpleImputer(strategy="median")
        X_tr  = imputer.fit_transform(X[train_idx])
        X_val = imputer.transform(X[test_idx])

        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(X_tr, y[train_idx])
        fold_r2 = r2_score(y[test_idx], model.predict(X_val))
        fold_scores.append(fold_r2)

        n_ils = len(set(groups[test_idx]))
        print(f"    Fold {fold_idx+1}/{N_OUTER_FOLDS}: {n_ils} test ILs | R2={fold_r2:.4f}",
              flush=True)

    return fold_scores


def summarize_ablation(all_results):
    """
    Print mean +/- std R2 per variant with include/exclude verdict vs baseline.
    Returns summary DataFrame sorted by mean R2.
    """
    rows = []
    for variant_name, description, fold_scores in all_results:
        rows.append({
            "variant":      variant_name,
            "description":  description,
            "mean_test_r2": np.mean(fold_scores),
            "std_test_r2":  np.std(fold_scores),
            "min_test_r2":  np.min(fold_scores),
            "max_test_r2":  np.max(fold_scores),
        })
    summary_df = pd.DataFrame(rows).sort_values(
        "mean_test_r2", ascending=False
    ).reset_index(drop=True)

    print("\n" + "=" * 70, flush=True)
    print("FEATURE ABLATION SUMMARY", flush=True)
    print(f"Baseline (nested CV, Morgan+18desc): R2={BASELINE_NESTED_CV_R2:.4f} "
          f"+/- {BASELINE_NESTED_CV_STD:.4f}", flush=True)
    print(f"Improvement threshold: +{IMPROVEMENT_THRESHOLD_R2:.3f} R2", flush=True)
    print("=" * 70, flush=True)

    for _, row in summary_df.iterrows():
        delta = row["mean_test_r2"] - BASELINE_NESTED_CV_R2
        if row["variant"] == "v1_baseline":
            verdict = "REFERENCE"
        elif delta >= IMPROVEMENT_THRESHOLD_R2:
            verdict = f"INCLUDE (+{delta:.4f} R2)"
        elif delta >= 0:
            verdict = f"MARGINAL (+{delta:.4f} R2)"
        else:
            verdict = f"EXCLUDE ({delta:.4f} R2)"
        print(f"  {row['variant']:20s} | "
              f"R2={row['mean_test_r2']:.4f} +/- {row['std_test_r2']:.4f} | "
              f"{verdict}", flush=True)

    print("=" * 70, flush=True)

    winners = summary_df[
        (summary_df["variant"] != "v1_baseline") &
        (summary_df["mean_test_r2"] - BASELINE_NESTED_CV_R2 >= IMPROVEMENT_THRESHOLD_R2)
    ]["variant"].tolist()

    if winners:
        print(f"\n[decision] INCLUDE these blocks: {winners}", flush=True)
        print("  Next: combine winning blocks into featurize_v3.py and retrain.",
              flush=True)
    else:
        print("\n[decision] No variant clears +0.010 R2 threshold.", flush=True)
        print("  Morgan+18desc is the performance ceiling. Ship baseline RF.",
              flush=True)

    return summary_df


def plot_ablation(summary_df):
    """Horizontal bar chart of mean test R2 per variant with std error bars."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    labels = summary_df["variant"].tolist()
    means  = summary_df["mean_test_r2"].tolist()
    stds   = summary_df["std_test_r2"].tolist()
    colors = ["#e74c3c" if v == "v1_baseline" else "#3498db" for v in labels]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(range(len(labels)), means, xerr=stds, color=colors,
            alpha=0.75, height=0.5, capsize=4)
    ax.axvline(BASELINE_NESTED_CV_R2, color="#e74c3c", linestyle="--",
               linewidth=1.2, label=f"Baseline R2={BASELINE_NESTED_CV_R2:.4f}")
    ax.axvline(BASELINE_NESTED_CV_R2 + IMPROVEMENT_THRESHOLD_R2,
               color="green", linestyle=":", linewidth=1.2,
               label=f"Threshold (+{IMPROVEMENT_THRESHOLD_R2:.3f})")
    ax.set_yticks(list(range(len(labels))))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean test R2 (5-fold GroupKFold)", fontsize=11)
    ax.set_title("Feature Ablation: which representation additions help?",
                 fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "feature_ablation_barplot.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[figures] Saved: {out_path}", flush=True)


def main():
    """
    For each variant: load CSVs, combine into full pool, run nested CV,
    collect per-fold R2. Then summarize, save CSV, and plot.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,    exist_ok=True)

    print("=" * 70, flush=True)
    print("FEATURE ABLATION STUDY", flush=True)
    print(f"RF: n_estimators={RF_PARAMS['n_estimators']}, "
          f"max_features={RF_PARAMS['max_features']}, "
          f"min_samples_leaf={RF_PARAMS['min_samples_leaf']}", flush=True)
    print(f"CV: {N_OUTER_FOLDS}-fold GroupKFold | "
          f"Baseline: R2={BASELINE_NESTED_CV_R2:.4f}", flush=True)
    print("=" * 70 + "\n", flush=True)

    all_results = []

    for variant_name, description, train_csv, test_csv in VARIANTS:
        print(f"\n" + "-" * 60, flush=True)
        print(f"[variant] {variant_name}: {description}", flush=True)
        print("-" * 60, flush=True)

        try:
            (X_tr, y_tr, sm_tr,
             X_te, y_te, sm_te,
             feat_cols) = load_variant(train_csv, test_csv, variant_name)
        except FileNotFoundError as err:
            print(f"  SKIP: {err}", flush=True)
            continue

        # Combine train + test into full pool for nested CV
        X_full      = np.vstack([X_tr, X_te])
        y_full      = np.concatenate([y_tr, y_te])
        smiles_full = pd.concat([sm_tr, sm_te], ignore_index=True)

        print(f"  Pool: {len(y_full)} rows, {smiles_full.nunique()} ILs, "
              f"{len(feat_cols)} features", flush=True)

        fold_scores = run_nested_cv(X_full, y_full, smiles_full, variant_name)
        mean_r2 = np.mean(fold_scores)
        print(f"  --> mean R2={mean_r2:.4f} +/- {np.std(fold_scores):.4f} "
              f"(delta={mean_r2 - BASELINE_NESTED_CV_R2:+.4f})", flush=True)

        all_results.append((variant_name, description, fold_scores))

    if not all_results:
        print("[main] No variants ran. Check build_dataset_v2.py output.",
              flush=True)
        sys.exit(1)

    summary_df = summarize_ablation(all_results)
    summary_df.to_csv(
        os.path.join(RESULTS_DIR, "feature_ablation.csv"), index=False
    )
    print(f"\n[main] Saved -> results/feature_ablation.csv", flush=True)
    plot_ablation(summary_df)
    print("[main] DONE.", flush=True)


if __name__ == "__main__":
    main()
