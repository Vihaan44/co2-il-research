"""
plot_inverse_design.py
----------------------
PURPOSE: Visualize the results of Phase 4 inverse design screening.

  Figure 1 -- Virtual Library Scatter
    All screened IL candidates plotted by rank vs. predicted log10(x2_CO2).
    Top 20 candidates highlighted in red. Shows the full distribution and
    where the top candidates sit relative to the entire screened field.

  Figure 2 -- Top Candidates Bar Chart
    Horizontal bar chart of predicted x2_CO2 for the top 20 candidates.
    Each bar labeled with truncated IL SMILES.
    This is the key 'output' figure for judges: these are the novel ILs
    our model predicts will have the highest CO2 absorption.

  Figure 3 -- Anion Family Distribution (Top 20)
    Bar chart showing which anion families appear most in the top candidates.
    Answers: 'Which anion types drive high predicted CO2 absorption?'
    If fluorinated anions ([Tf2N]-, [FSI]-) dominate, this aligns with
    literature: fluorinated anions favor CO2 via van der Waals interactions.

INPUTS:
  results/virtual_library_predictions.csv   (from inverse_design.py)
  results/top_candidates.csv                (from inverse_design.py)

OUTPUTS:
  figures/virtual_library_scatter.png
  figures/top_candidates_bar.png
  figures/top_candidates_anion_families.png

Run from project root:
    python src/plot_inverse_design.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# -- Constants -----------------------------------------------------------------
RESULTS_DIR     = "results"
FIGURES_DIR     = "figures"
ALL_PREDS_CSV   = os.path.join(RESULTS_DIR, "virtual_library_predictions.csv")
TOP_CANDS_CSV   = os.path.join(RESULTS_DIR, "top_candidates.csv")

FIG_DPI         = 300
TOP_N_HIGHLIGHT = 20   # how many top candidates to highlight in scatter
MAX_SMILES_LABEL_LEN = 45   # truncate SMILES in bar chart labels for readability

# Colors
COLOR_FIELD = "#93C5FD"   # light blue -- all candidates (background)
COLOR_TOP   = "#DC2626"   # red -- top candidates highlighted
COLOR_BAR   = "#2563EB"   # blue -- bar chart
COLOR_PIE   = [           # distinct colors for anion family bar chart
    "#2563EB", "#DC2626", "#16A34A", "#D97706",
    "#7C3AED", "#DB2777", "#0891B2", "#65A30D",
    "#92400E", "#065F46",
]

# Anion family classification: maps SMILES substrings to readable family names.
# Order matters -- more specific patterns first to avoid false matches.
# Patterns verified against actual virtual library anion SMILES from inverse_design.py output.
# Virtual library uses 'O=S(=O)(...)' format (explicit O= prefix on sulfonyl groups).
ANION_FAMILY_PATTERNS = [
    # Bis(trifluoromethylsulfonyl)imide [Tf2N]- -- most specific, check first
    ("S(=O)(=O)C(F)(F)F",          "[Tf2N]-"),      # matches O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F
    # Bis(fluorosulfonyl)imide [FSI]-
    ("[N-](S(=O)(=O)F)S(=O)(=O)F", "[FSI]-"),
    # Fluorosulfonyl-nitrile [SFN]- / FSIN-
    ("[N-](S(=O)(=O)F)C#N",        "[SFN]-"),
    # Trifluoromethanesulfonate [OTf]-
    ("S(=O)(=O)([O-])C(F)(F)F",    "[OTf]-"),
    # Tetracyanoborate [TCB]-
    ("[B-](C#N)(C#N)(C#N)C#N",     "[TCB]-"),
    # Dicyanamide [DCA]-
    ("[N-](C#N)C#N",               "[DCA]-"),
    # Tetrafluoroborate [BF4]-
    ("[B-](F)(F)(F)F",             "[BF4]-"),
    # Hexafluorophosphate [PF6]-
    ("[P-](F)(F)(F)(F)(F)F",       "[PF6]-"),
    # Benzenesulfonate
    ("S(=O)(=O)([O-])c1ccccc1",    "Sulfonate"),
    # Alkylsulfonate (catch-all for other sulfonates)
    ("S(=O)(=O)[O-]",              "Sulfonate"),
    # Carboxylate / trifluoroacetate
    ("C(=O)[O-]",                  "Carboxylate"),
    # Halides
    ("[Cl-]",  "[Cl]-"),
    ("[Br-]",  "[Br]-"),
    ("[I-]",   "[I]-"),
    ("[F-]",   "[F]-"),
]


def load_results() -> tuple:
    """
    Load virtual library predictions and top candidates CSVs.
    Validates that required columns exist before plotting.
    Returns: (all_preds_df, top_cands_df)
    """
    for path in [ALL_PREDS_CSV, TOP_CANDS_CSV]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Run src/inverse_design.py first."
            )

    all_preds_df = pd.read_csv(ALL_PREDS_CSV)
    top_cands_df = pd.read_csv(TOP_CANDS_CSV)

    print(f"[load_results] {len(all_preds_df)} total virtual IL predictions loaded")
    print(f"[load_results] {len(top_cands_df)} top candidates loaded")
    print(f"  Predicted x2 range: [{all_preds_df['x2_predicted'].min():.2e}, "
          f"{all_preds_df['x2_predicted'].max():.2e}]")
    return all_preds_df, top_cands_df


def classify_anion_family(anion_smiles: str) -> str:
    """
    Map an anion SMILES string to a human-readable anion family name.
    Uses substring matching against ANION_FAMILY_PATTERNS (most specific first).
    Returns 'Other' if no pattern matches -- if this appears in output,
    add the unmatched SMILES to ANION_FAMILY_PATTERNS above.
    """
    for pattern, family_name in ANION_FAMILY_PATTERNS:
        if pattern in anion_smiles:
            return family_name
    # DATA QUALITY FLAG: unclassified anion -- print for debugging
    print(f"  [NOTE] Unclassified anion SMILES: {anion_smiles}")
    return "Other"


def plot_virtual_library_scatter(all_preds_df: pd.DataFrame,
                                  top_cands_df: pd.DataFrame) -> None:
    """
    Scatter: x = candidate rank, y = predicted log10(x2_CO2).
    Top N highlighted in red to show where they sit in the full distribution.

    For judges: 'We screened N virtual ILs; our top candidates cluster at
    the high-absorption end of the distribution.'
    """
    # Sort all predictions by x2 descending so rank 1 = best (leftmost on x-axis)
    sorted_preds = all_preds_df.sort_values("x2_predicted", ascending=False).reset_index(drop=True)
    all_log_x2   = sorted_preds["log_x2_predicted"].values
    all_ranks     = np.arange(1, len(all_log_x2) + 1)

    # Top candidates by their rank column from inverse_design.py
    top_ranks  = top_cands_df["rank"].values
    top_log_x2 = top_cands_df["log_x2_predicted"].values

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(all_ranks, all_log_x2,
               color=COLOR_FIELD, alpha=0.5, s=12, zorder=2,
               label=f"All candidates (n={len(all_preds_df)})")

    ax.scatter(top_ranks, top_log_x2,
               color=COLOR_TOP, alpha=0.9, s=40, zorder=3,
               marker="D", label=f"Top {TOP_N_HIGHLIGHT} predicted ILs")

    ax.set_xlabel("Candidate Rank (1 = highest predicted CO2 absorption)", fontsize=12)
    ax.set_ylabel(r"Predicted log$_{10}$(x$_2^{\mathrm{CO}_2}$)", fontsize=12)
    ax.set_title("Virtual Library Screening -- All Candidates", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.4)

    out_path = os.path.join(FIGURES_DIR, "virtual_library_scatter.png")
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_virtual_library_scatter] Saved -> {out_path}")


def plot_top_candidates_bar(top_cands_df: pd.DataFrame) -> None:
    """
    Horizontal bar chart of predicted x2_CO2 for the top N candidates.
    Labels show truncated SMILES (full strings are too long for axis labels).

    This is the primary results figure: it directly shows which novel ILs
    our model predicts will absorb the most CO2.
    """
    # Sort ascending so highest bar appears at top of horizontal chart
    plot_df = top_cands_df.sort_values("x2_predicted", ascending=True).copy()

    # Truncate SMILES for readability
    labels = [
        s[:MAX_SMILES_LABEL_LEN] + "..." if len(s) > MAX_SMILES_LABEL_LEN else s
        for s in plot_df["il_smiles"].tolist()
    ]

    fig, ax = plt.subplots(figsize=(9, 7))

    bars = ax.barh(labels, plot_df["x2_predicted"],
                   color=COLOR_BAR, alpha=0.85, edgecolor="white", linewidth=0.4)

    ax.set_xlabel(r"Predicted CO$_2$ Mole Fraction Solubility (x$_2$)", fontsize=12)
    ax.set_title(f"Top {len(top_cands_df)} Novel ILs by Predicted CO2 Absorption", fontsize=13)
    ax.tick_params(axis="y", labelsize=7)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)

    # Annotate each bar with its numeric value
    for bar_rect, val in zip(bars, plot_df["x2_predicted"]):
        ax.text(val + 0.0001, bar_rect.get_y() + bar_rect.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=7)

    out_path = os.path.join(FIGURES_DIR, "top_candidates_bar.png")
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_top_candidates_bar] Saved -> {out_path}")
    print(f"  Best: {top_cands_df.iloc[0]['il_smiles'][:60]}")
    print(f"  x2   = {top_cands_df.iloc[0]['x2_predicted']:.4e}")


def plot_anion_family_distribution(top_cands_df: pd.DataFrame) -> None:
    """
    Bar chart of anion family frequency among the top candidates.

    Answers the judge question: 'What structural features distinguish your
    top candidates?' Fluorinated anions dominating ([Tf2N]-, [FSI]-) aligns
    with literature -- fluorine increases CO2 affinity via van der Waals.
    """
    # Classify each top candidate's anion SMILES into a named family
    anion_families = top_cands_df["anion_smiles"].apply(classify_anion_family)
    family_counts   = anion_families.value_counts()

    print(f"  Anion family breakdown: {dict(family_counts)}")

    colors_used = COLOR_PIE[:len(family_counts)]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(family_counts.index, family_counts.values,
           color=colors_used, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Anion Family", fontsize=12)
    ax.set_ylabel("Count in Top Candidates", fontsize=12)
    ax.set_title(f"Anion Family Distribution -- Top {len(top_cands_df)} Candidates", fontsize=13)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    # Count label above each bar
    for i, cnt in enumerate(family_counts.values):
        ax.text(i, cnt + 0.05, str(cnt), ha="center", va="bottom", fontsize=10)

    out_path = os.path.join(FIGURES_DIR, "top_candidates_anion_families.png")
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_anion_family_distribution] Saved -> {out_path}")
    print(f"  Most common anion in top candidates: {family_counts.index[0]} ({family_counts.iloc[0]} ILs)")


def main():
    """Load inverse design results and generate all three Phase 4 figures."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    all_preds_df, top_cands_df = load_results()

    print("\n=== FIGURE 1: Virtual Library Scatter ===")
    plot_virtual_library_scatter(all_preds_df, top_cands_df)

    print("\n=== FIGURE 2: Top Candidates Bar Chart ===")
    plot_top_candidates_bar(top_cands_df)

    print("\n=== FIGURE 3: Anion Family Distribution ===")
    plot_anion_family_distribution(top_cands_df)

    print("\nAll Phase 4 figures saved to figures/")
    print("  figures/virtual_library_scatter.png")
    print("  figures/top_candidates_bar.png")
    print("  figures/top_candidates_anion_families.png")


if __name__ == "__main__":
    main()
