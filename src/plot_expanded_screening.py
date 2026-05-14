"""
plot_expanded_screening.py
--------------------------
PURPOSE: Visualize results from the expanded 1226-IL virtual library screening.

  Figure 1 -- Scatter: all 1226 candidates ranked by predicted log10(x2_CO2),
              top 20 highlighted. Shows the full prediction distribution.

  Figure 2 -- Top 20 bar chart with dianion candidates flagged in orange.
              Dianions ([malonate]2-, [succinate]2-) are out-of-distribution
              for a model trained on 1:1 ILs -- flagged as unreliable predictions.

  Figure 3 -- Cation family distribution in top 20.
              New vs Phase 4: shows whether novel families (sulfonium, guanidinium,
              pyridinium) appear, or whether ammonium still dominates.

  Figure 4 -- Anion family distribution in top 20.
              Compares to Phase 4: do the new anions (isethionate, saccharinate,
              tosylate) appear alongside or instead of [Tf2N]-/[FSI]-?

INPUTS:
  results/virtual_library_predictions_expanded.csv
  results/top_candidates_expanded.csv

OUTPUTS:
  figures/expanded_scatter.png
  figures/expanded_top20_bar.png
  figures/expanded_cation_families.png
  figures/expanded_anion_families.png

Run from project root:
    python src/plot_expanded_screening.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Constants ──────────────────────────────────────────────────────────────────
ALL_PREDS_CSV   = os.path.join("results", "virtual_library_predictions_expanded.csv")
TOP_CANDS_CSV   = os.path.join("results", "top_candidates_expanded.csv")
FIGURES_DIR     = "figures"
FIG_DPI         = 300
TOP_N           = 20

# Visual constants
COLOR_FIELD   = "#93C5FD"   # light blue — all candidates
COLOR_TOP     = "#DC2626"   # red — top candidates highlighted
COLOR_NORMAL  = "#2563EB"   # blue — normal top-20 bars
COLOR_DIANION = "#F97316"   # orange — dianion bars (unreliable predictions)
COLOR_PALETTE = [            # distinct colors for family charts
    "#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED",
    "#DB2777", "#0891B2", "#65A30D", "#92400E", "#065F46",
]
MAX_LABEL_LEN = 50  # truncate long SMILES strings in bar chart labels

# SMILES substrings that identify dianion species.
# These are out-of-distribution for our 1:1 IL model — predictions flagged as unreliable.
DIANION_SUBSTRINGS = [
    "[O-]C(=O)CC([O-])=O",   # malonate2-
    "[O-]C(=O)CCC([O-])=O",  # succinate2-
    "[O-]C(=O)C([O-])=O",    # oxalate2-
]


# ── Anion family classification ────────────────────────────────────────────────
# Each entry: (list of substrings ANY must match, family label)
# Ordered most-specific first to avoid false matches.
ANION_FAMILY_PATTERNS = [
    (["S(=O)(=O)C(F)(F)F"],                          "[Tf2N]-"),
    (["S(=O)(=O)F)F", "S(=O)(=O)F)S"],               "[FSI]-"),
    (["[N-](S(=O)(=O)F)C#N"],                        "[SFN]-"),
    (["S(=O)(=O)([O-])C(F)(F)F", "([O-])C(F)(F)F"],  "[OTf]-"),
    (["[B-](C#N)(C#N)(C#N)C#N"],                     "[TCB]-"),
    (["[N-](C#N)C#N"],                               "[DCA]-"),
    (["[B-](F)(F)(F)F"],                             "[BF4]-"),
    (["[P-](F)", "F[P-]"],                           "[PF6/FAP]-"),
    (["[Sb-]"],                                      "[SbF6]-"),
    # Sulfonate: check isethionate (CCO) and tosylate (c1ccc) specifically first
    (["S(=O)(=O)([O-])CCO", "([O-])CCO"],            "Isethionate"),
    (["S(=O)(=O)([O-])c", "([O-])S(=O)(=O)c"],       "Arylsulfonate"),
    (["OS([O-])(=O)=O", "S([O-])(=O)=O",
      "S(=O)(=O)[O-]"],                              "Alkylsulfate"),
    # Heterocyclic anions — check before carboxylate to avoid false match
    (["NS(=O)(=O)c2ccccc", "O=C1NS"],                "Saccharinate"),
    (["[N-]C(=O)c2ccccc", "C1[N-]C(=O)"],           "Phthalimide"),
    (["CC1=CC(=O)[N-]S"],                            "Acesulfamate"),
    # Carboxylate / TFA — comes after more specific fluorinated checks
    (["C(=O)[O-]", "[O-]C(=O)"],                    "Carboxylate"),
    (["[Cl-]"],  "[Cl]-"),
    (["[Br-]"],  "[Br]-"),
    (["[I-]"],   "[I]-"),
    (["[S-]C#N"], "[SCN]-"),
    (["[O-]C#N"], "[OCN]-"),
]


def classify_anion(anion_smiles: str) -> str:
    """
    Map an anion SMILES to a family label using substring matching.
    Returns 'Other' and prints if nothing matches — helps us catch gaps.
    """
    for substrings, family in ANION_FAMILY_PATTERNS:
        if any(s in anion_smiles for s in substrings):
            return family
    print(f"  [NOTE] Unclassified anion: {anion_smiles}")
    return "Other"


def is_dianion(il_smiles: str) -> bool:
    """
    Returns True if this IL's SMILES contains a known dianion fragment.
    Dianion ILs are 2:1 salts — outside the training distribution of 1:1 ILs.
    Predictions for these should be flagged as unreliable extrapolation.
    """
    return any(sub in il_smiles for sub in DIANION_SUBSTRINGS)


def load_results() -> tuple:
    """
    Load expanded screening results. Raises FileNotFoundError if missing.
    Returns (all_preds_df, top_cands_df).
    """
    for path in [ALL_PREDS_CSV, TOP_CANDS_CSV]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Run src/inverse_design.py first."
            )
    all_preds_df = pd.read_csv(ALL_PREDS_CSV)
    top_cands_df = pd.read_csv(TOP_CANDS_CSV)

    print(f"[load_results] {len(all_preds_df)} total predictions loaded")
    print(f"[load_results] {len(top_cands_df)} top candidates loaded")
    print(f"  x2 range: [{all_preds_df['x2_predicted'].min():.2e}, "
          f"{all_preds_df['x2_predicted'].max():.2e}]")
    return all_preds_df, top_cands_df


# ── Figure 1: Scatter ──────────────────────────────────────────────────────────
def plot_scatter(all_preds_df: pd.DataFrame, top_cands_df: pd.DataFrame) -> None:
    """
    Scatter plot of rank vs predicted log10(x2). Top 20 highlighted in red.
    Shows how the top candidates relate to the full distribution of 1226 ILs.
    """
    sorted_df = all_preds_df.sort_values("x2_predicted", ascending=False).reset_index(drop=True)
    ranks      = np.arange(1, len(sorted_df) + 1)
    log_x2     = sorted_df["log_x2_predicted"].values

    top_ranks  = top_cands_df["rank"].values
    top_log_x2 = top_cands_df["log_x2_predicted"].values

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(ranks, log_x2, color=COLOR_FIELD, alpha=0.4, s=8,
               label=f"All candidates (n={len(all_preds_df)})", zorder=2)
    ax.scatter(top_ranks, top_log_x2, color=COLOR_TOP, s=40, marker="D",
               alpha=0.9, label=f"Top {TOP_N}", zorder=3)

    ax.set_xlabel("Candidate Rank (1 = highest predicted CO₂ absorption)", fontsize=11)
    ax.set_ylabel(r"Predicted log$_{10}$(x$_2^{\mathrm{CO}_2}$)", fontsize=11)
    ax.set_title("Expanded Virtual Library Screening — 1226 Novel ILs", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.4)

    out = os.path.join(FIGURES_DIR, "expanded_scatter.png")
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_scatter] Saved -> {out}")


# ── Figure 2: Top 20 bar chart with dianion flagging ──────────────────────────
def plot_top20_bar(top_cands_df: pd.DataFrame) -> None:
    """
    Horizontal bar chart of predicted x2 for top 20 candidates.
    Dianion candidates shown in orange with a warning label — these are
    out-of-distribution for our 1:1 IL model and may be unreliable.

    WHY FLAG DIANIONS: The model was trained only on 1:1 ionic liquids
    (one cation, one anion). A dianion like malonate2- requires a 2:1 stoichiometry
    (two cations per anion), making it a different compound class.
    High predicted x2 may be an artifact of model extrapolation.
    """
    plot_df = top_cands_df.sort_values("x2_predicted", ascending=True).copy()
    plot_df["dianion_flag"] = plot_df["il_smiles"].apply(is_dianion)

    # Short labels: truncate SMILES and append dianion warning
    def make_label(row):
        smi = row["il_smiles"]
        label = smi[:MAX_LABEL_LEN] + ("..." if len(smi) > MAX_LABEL_LEN else "")
        return label + "  ⚠ dianion" if row["dianion_flag"] else label

    labels = plot_df.apply(make_label, axis=1).tolist()
    colors = [COLOR_DIANION if d else COLOR_NORMAL for d in plot_df["dianion_flag"]]

    n_dianions = plot_df["dianion_flag"].sum()
    if n_dianions > 0:
        print(f"  [FLAG] {n_dianions} dianion candidates in top 20 — predictions marked unreliable")

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(labels, plot_df["x2_predicted"],
                   color=colors, edgecolor="white", linewidth=0.4, alpha=0.9)

    # Value labels at end of each bar
    for bar_rect, val in zip(bars, plot_df["x2_predicted"]):
        ax.text(val + 0.0002, bar_rect.get_y() + bar_rect.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=7)

    # Legend entries for color coding
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=COLOR_NORMAL,  label="Valid 1:1 IL prediction"),
        Patch(color=COLOR_DIANION, label="⚠ Dianion — may be unreliable"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="lower right")

    ax.set_xlabel(r"Predicted CO$_2$ Mole Fraction Solubility (x$_2$)", fontsize=11)
    ax.set_title(f"Top {TOP_N} Expanded Library Candidates\n"
                 f"(orange = dianion anion, out-of-distribution for this model)",
                 fontsize=11)
    ax.tick_params(axis="y", labelsize=7)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)

    out = os.path.join(FIGURES_DIR, "expanded_top20_bar.png")
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_top20_bar] Saved -> {out}")


# ── Figure 3: Cation family distribution ──────────────────────────────────────
def plot_cation_families(top_cands_df: pd.DataFrame) -> None:
    """
    Bar chart of cation family counts in the top 20.
    If ammonium dominates, that's a meaningful result: the NH2-functionalized
    ammonium cation structure is uniquely predicted to favor CO2 absorption.
    If novel families (sulfonium, guanidinium) appear, that's a new finding.
    """
    if "cation_family" not in top_cands_df.columns:
        print("[plot_cation_families] No cation_family column — skipping")
        return

    counts  = top_cands_df["cation_family"].value_counts()
    colors  = COLOR_PALETTE[:len(counts)]

    print(f"  Cation family breakdown in top {TOP_N}: {dict(counts)}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.5)

    for i, cnt in enumerate(counts.values):
        ax.text(i, cnt + 0.05, str(cnt), ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xlabel("Cation Family", fontsize=11)
    ax.set_ylabel(f"Count in Top {TOP_N}", fontsize=11)
    ax.set_title(f"Cation Family Distribution — Top {TOP_N} Expanded Screening Candidates",
                 fontsize=11)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    out = os.path.join(FIGURES_DIR, "expanded_cation_families.png")
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_cation_families] Saved -> {out}")
    print(f"  Dominant family: {counts.index[0]} ({counts.iloc[0]}/{TOP_N} top candidates)")


# ── Figure 4: Anion family distribution ───────────────────────────────────────
def plot_anion_families(top_cands_df: pd.DataFrame) -> None:
    """
    Bar chart of anion family counts in the top 20 expanded screening candidates.
    Compared to Phase 4 (dominated by [Tf2N]-/[FSI]-), the expanded library
    introduces isethionate, saccharinate, and tosylate — shows whether these
    novel anions genuinely compete with fluorinated anions.
    """
    families = top_cands_df["anion_smiles"].apply(classify_anion)
    counts   = families.value_counts()
    colors   = COLOR_PALETTE[:len(counts)]

    print(f"  Anion family breakdown in top {TOP_N}: {dict(counts)}")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.5)

    for i, cnt in enumerate(counts.values):
        ax.text(i, cnt + 0.05, str(cnt), ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xlabel("Anion Family", fontsize=11)
    ax.set_ylabel(f"Count in Top {TOP_N}", fontsize=11)
    ax.set_title(f"Anion Family Distribution — Top {TOP_N} Expanded Screening Candidates\n"
                 f"(compare to Phase 4: [Tf2N]- dominated)", fontsize=11)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    out = os.path.join(FIGURES_DIR, "expanded_anion_families.png")
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_anion_families] Saved -> {out}")
    print(f"  Most common anion: {counts.index[0]} ({counts.iloc[0]}/{TOP_N} candidates)")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    """Load expanded screening results and generate all four figures."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    all_preds_df, top_cands_df = load_results()

    print("\n=== FIGURE 1: Scatter (all 1226 candidates) ===")
    plot_scatter(all_preds_df, top_cands_df)

    print("\n=== FIGURE 2: Top 20 bar chart (dianions flagged) ===")
    plot_top20_bar(top_cands_df)

    print("\n=== FIGURE 3: Cation family distribution ===")
    plot_cation_families(top_cands_df)

    print("\n=== FIGURE 4: Anion family distribution ===")
    plot_anion_families(top_cands_df)

    print("\nAll expanded screening figures saved to figures/")


if __name__ == "__main__":
    main()
