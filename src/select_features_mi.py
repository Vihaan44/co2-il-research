"""
select_features_mi.py
----------------------
PURPOSE: Use mutual information (MI) to rank all 4134 features by their
         individual correlation with log_x2_CO2, then test whether training
         RF on the top-K features improves OOD generalization vs the full
         4134-feature baseline (nested CV R2=0.708).

WHY THIS MIGHT HELP:
  With 299 ILs and 4134 features, the sample-to-feature ratio is ~0.07.
  RF handles high-dimensional data via random column subsampling, but it
  still wastes splits on uninformative bits. Most of the 2048x2 Morgan FP
  bits are zero for the majority of ILs (rare substructures). Keeping only
  the most informative features reduces noise in RF's column sampling and
  can improve generalization on small datasets.

  MI is a non-linear measure of dependence between each feature and the
  target -- it captures threshold effects and interactions that Pearson
  correlation would miss. We use sklearn's mutual_info_regression which
  estimates MI via k-nearest-neighbors.

METHOD:
  1. Load full training pool (train + test combined, 299 ILs)
  2. Compute MI between each of the 4134 features and log_x2_CO2
  3. Test K = [100, 200, 500, 1000, 2000, 4134] via 5-fold GroupKFold RF CV
  4. Report mean +/- std test R2 per K
  5. Save the best-K feature list to results/mi_selected_features.txt
     so build_dataset_mi.py can use it to produce a filtered train/test pair

OUTPUTS:
  results/mi_feature_scores.csv    -- MI score per feature, sorted descending
  results/mi_ablation.csv          -- mean/std R2 per K value
  results/mi_selected_features.txt -- feature names for the best K
  figures/mi_ablation_plot.png     -- R2 vs K curve

Run from project root:
  nohup python src/select_features_mi.py > logs/mi_selection.log 2>&1 &
Then monitor:
  tail -f logs/mi_selection.log
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

# -- Constants -----------------------------------------------------------------
TRAIN_CSV   = os.path.join("data", "processed", "train_set.csv")
TEST_CSV    = os.path.join("data", "processed", "test_set.csv")
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
LOGS_DIR    = "logs"

TARGET_COL         = "log_x2_CO2"
CONDITION_FEATURES = ["T_K", "P_kPa"]
N_OUTER_FOLDS       = 5
RANDOM_SEED         = 42

# K values to test: from very aggressive selection to full feature set
K_VALUES = [100, 200, 500, 1000, 2000, 4134]

# Baseline from nested_cv_model_comparison.py for comparison
BASELINE_R2  = 0.7078
BASELINE_STD = 0.1115

# RF: Optuna v2 best params (same throughout project)
RF_PARAMS = dict(
    n_estimators     = 515,
    max_features     = 0.5,
    min_samples_leaf = 7,
    max_depth        = None,
    random_state     = RANDOM_SEED,
    n_jobs           = -1,
)

# MI estimation: n_neighbors controls bias-variance tradeoff in MI estimator.
# 5 is the sklearn default and works well for continuous targets.
MI_N_NEIGHBORS = 5


def load_full_pool():
    """
    Load and combine train_set.csv + test_set.csv into one pool.
    Uses the original baseline feature set (Morgan + 18 RDKit desc).
    Returns X, y, il_smiles, feature_cols.
    """
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)
    full_df  = pd.concat([train_df, test_df], ignore_index=True)
    full_df  = full_df.dropna(subset=CONDITION_FEATURES).copy()

    molecular_cols = [
        c for c in full_df.columns
        if c.startswith("cat_") or c.startswith("an_")
    ]
    feature_cols = molecular_cols + CONDITION_FEATURES

    # Fill NaN descriptors with column median
    desc_cols = [c for c in molecular_cols if "_fp_" not in c]
    for col in desc_cols:
        if full_df[col].isna().any():
            full_df[col] = full_df[col].fillna(full_df[col].median())

    X         = full_df[feature_cols].values.astype(float)
    y         = full_df[TARGET_COL].values
    il_smiles = full_df["il_smiles"]

    print(f"[load] {len(full_df)} rows, {il_smiles.nunique()} unique ILs, "
          f"{len(feature_cols)} features", flush=True)
    return X, y, il_smiles, feature_cols


def compute_mi_scores(X, y, feature_cols):
    """
    Compute mutual information between each feature and the target.
    MI is estimated via k-nearest-neighbors (sklearn implementation).
    NaN values are imputed with column median before MI computation
    (MI estimator cannot handle NaN).
    Returns a DataFrame with feature names and MI scores, sorted descending.
    """
    print(f"\n[MI] Computing mutual information for {X.shape[1]} features...",
          flush=True)
    print(f"  This uses k={MI_N_NEIGHBORS} nearest neighbors and may take 2-5 min.",
          flush=True)

    # Impute any remaining NaN before MI computation
    imputer = SimpleImputer(strategy="median")
    X_imp   = imputer.fit_transform(X)

    mi_scores = mutual_info_regression(
        X_imp, y,
        n_neighbors  = MI_N_NEIGHBORS,
        random_state = RANDOM_SEED,
    )

    mi_df = pd.DataFrame({
        "feature":  feature_cols,
        "mi_score": mi_scores,
    }).sort_values("mi_score", ascending=False).reset_index(drop=True)

    print(f"[MI] Top 10 features by MI score:", flush=True)
    for _, row in mi_df.head(10).iterrows():
        print(f"  {row['feature']:40s}: {row['mi_score']:.4f}", flush=True)

    zero_mi = (mi_df["mi_score"] == 0).sum()
    print(f"[MI] Features with zero MI (uninformative): {zero_mi}/{len(mi_df)}",
          flush=True)

    return mi_df


def run_cv_for_k(X, y, il_smiles, feature_cols, mi_df, k):
    """
    Run 5-fold GroupKFold RF CV using only the top-K features by MI score.
    Returns list of per-fold test R2 values.
    """
    # Select top-K feature indices based on MI ranking
    top_k_features = mi_df.head(k)["feature"].tolist()
    # Always include condition features (T_K, P_kPa) regardless of MI rank
    for cond in CONDITION_FEATURES:
        if cond not in top_k_features:
            top_k_features.append(cond)

    # Map feature names to column indices in X
    feature_to_idx = {name: idx for idx, name in enumerate(feature_cols)}
    col_indices = [feature_to_idx[f] for f in top_k_features if f in feature_to_idx]
    X_k = X[:, col_indices]

    gkf    = GroupKFold(n_splits=N_OUTER_FOLDS)
    groups = il_smiles.values
    fold_scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_k, y, groups)):
        assert len(set(groups[train_idx]) & set(groups[test_idx])) == 0, \
            f"IL overlap in fold {fold_idx}!"

        imputer = SimpleImputer(strategy="median")
        X_tr  = imputer.fit_transform(X_k[train_idx])
        X_val = imputer.transform(X_k[test_idx])

        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(X_tr, y[train_idx])
        fold_r2 = r2_score(y[test_idx], model.predict(X_val))
        fold_scores.append(fold_r2)

    mean_r2 = np.mean(fold_scores)
    print(f"  K={k:5d}: mean R2={mean_r2:.4f} +/- {np.std(fold_scores):.4f} "
          f"(delta={mean_r2 - BASELINE_R2:+.4f})", flush=True)
    return fold_scores


def plot_mi_curve(ablation_df):
    """Plot mean test R2 vs number of features selected by MI."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.errorbar(
        ablation_df["k"], ablation_df["mean_r2"],
        yerr=ablation_df["std_r2"],
        marker="o", linewidth=2, capsize=4, color="#3498db", label="MI-selected RF"
    )
    ax.axhline(BASELINE_R2, color="#e74c3c", linestyle="--",
               linewidth=1.5, label=f"Baseline R2={BASELINE_R2:.4f}")
    ax.axhline(BASELINE_R2 + 0.01, color="green", linestyle=":",
               linewidth=1.2, label="Improvement threshold (+0.010)")

    ax.set_xlabel("Number of features selected (K)", fontsize=11)
    ax.set_ylabel("Mean test R2 (5-fold GroupKFold)", fontsize=11)
    ax.set_title("Mutual Information Feature Selection: R2 vs K", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "mi_ablation_plot.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[figures] Saved: {out_path}", flush=True)


def main():
    """
    1. Load full pool, compute MI scores for all features
    2. Test RF at each K value via nested GroupKFold CV
    3. Save scores, ablation results, best feature list, and plot
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,    exist_ok=True)

    print("=" * 65, flush=True)
    print("MUTUAL INFORMATION FEATURE SELECTION", flush=True)
    print(f"K values to test: {K_VALUES}", flush=True)
    print(f"Baseline: R2={BASELINE_R2:.4f} +/- {BASELINE_STD:.4f}", flush=True)
    print("=" * 65 + "\n", flush=True)

    X, y, il_smiles, feature_cols = load_full_pool()

    # Step 1: compute MI scores
    mi_df = compute_mi_scores(X, y, feature_cols)
    mi_df.to_csv(os.path.join(RESULTS_DIR, "mi_feature_scores.csv"), index=False)
    print(f"\n[main] Saved -> results/mi_feature_scores.csv", flush=True)

    # Step 2: test each K
    print(f"\n[cv] Testing RF at each K value...", flush=True)
    ablation_rows = []
    best_k        = K_VALUES[-1]  # default to full feature set
    best_mean_r2  = BASELINE_R2

    for k in K_VALUES:
        actual_k = min(k, len(feature_cols))
        fold_scores = run_cv_for_k(X, y, il_smiles, feature_cols, mi_df, actual_k)
        mean_r2 = np.mean(fold_scores)
        ablation_rows.append({
            "k":        actual_k,
            "mean_r2":  mean_r2,
            "std_r2":   np.std(fold_scores),
            "min_r2":   np.min(fold_scores),
            "max_r2":   np.max(fold_scores),
        })
        if mean_r2 > best_mean_r2:
            best_mean_r2 = mean_r2
            best_k = actual_k

    ablation_df = pd.DataFrame(ablation_rows)

    # Summary
    print("\n" + "=" * 65, flush=True)
    print("MUTUAL INFORMATION ABLATION SUMMARY", flush=True)
    print(f"Baseline: R2={BASELINE_R2:.4f} +/- {BASELINE_STD:.4f}", flush=True)
    print("=" * 65, flush=True)
    for _, row in ablation_df.iterrows():
        delta = row["mean_r2"] - BASELINE_R2
        verdict = "BETTER" if delta >= 0.010 else ("marginal" if delta >= 0 else "worse")
        print(f"  K={row['k']:5.0f}: R2={row['mean_r2']:.4f} +/- {row['std_r2']:.4f} "
              f"| delta={delta:+.4f} | {verdict}", flush=True)
    print("=" * 65, flush=True)

    if best_mean_r2 > BASELINE_R2 + 0.010:
        print(f"\n[decision] USE top-{best_k} MI features "
              f"(R2={best_mean_r2:.4f} vs baseline {BASELINE_R2:.4f})", flush=True)
        # Save the selected feature names
        selected_features = mi_df.head(best_k)["feature"].tolist()
        out_txt = os.path.join(RESULTS_DIR, "mi_selected_features.txt")
        with open(out_txt, "w") as f:
            f.write(f"# Best K: {best_k}\n")
            f.write(f"# Mean R2: {best_mean_r2:.4f}\n")
            for feat in selected_features:
                f.write(feat + "\n")
        print(f"[main] Saved selected features -> {out_txt}", flush=True)
        print(f"  Next: python src/build_dataset_mi.py (uses mi_selected_features.txt)",
              flush=True)
    else:
        print(f"\n[decision] MI selection does not improve over baseline.", flush=True)
        print(f"  Full feature set (K=4134) is optimal for 299 ILs.", flush=True)
        print(f"  Proceed to literature data expansion.", flush=True)

    ablation_df.to_csv(
        os.path.join(RESULTS_DIR, "mi_ablation.csv"), index=False
    )
    print(f"[main] Saved -> results/mi_ablation.csv", flush=True)
    plot_mi_curve(ablation_df)
    print("[main] DONE.", flush=True)


if __name__ == "__main__":
    main()
