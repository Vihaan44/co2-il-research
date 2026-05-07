# CO2 Capture with Ionic Liquids — ML Inverse Design

**Researcher:** Vihaan  
**Competition targets:** Regeneron STS, ISEF  
**Status:** Phase 0 — Environment Setup

---

## Project Overview

This project uses machine learning to perform **inverse design of ionic liquids (ILs) optimized for CO2 capture**. Instead of testing ILs one by one experimentally, we:

1. Collect experimental CO2 solubility data from [ILThermo](https://ilthermo.boulder.nist.gov)
2. Train an ML model to predict CO2 absorption from IL molecular structure (Morgan fingerprints via RDKit)
3. Screen thousands of novel IL combinations to find high-absorption candidates
4. Validate top candidates with DFT calculations (ORCA) to compute quantum-chemical CO2 binding energies

---

## Folder Structure

```
co2-il-research/
├── src/                   # Python source scripts (one script per pipeline stage)
├── data/
│   ├── raw/               # Raw ILThermo downloads — never edit these
│   ├── processed/         # Cleaned and featurized data for ML
│   └── virtual_library/   # Combinatorial IL pairs for inverse design
├── models/                # Saved trained models (.pkl via joblib)
├── results/               # Output CSVs: predictions, rankings, DFT comparison
├── figures/               # All plots (300 DPI, publication-ready)
├── dft/                   # ORCA input/output files
├── paper/                 # Research paper, abstract, competition materials
└── notebooks/             # Exploratory Jupyter notebooks
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/Vihaan44/co2-il-research.git
cd co2-il-research

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Build the local folder structure
python setup.py

# 4. Install dependencies
pip install -r requirements.txt
```

---

## Key Scientific Concepts

- **Ionic liquids (ILs):** Room-temperature molten salts (organic cation + anion) with tunable properties
- **Henry's law constant:** Measures CO2 solubility in the IL — lower = better absorption
- **Morgan fingerprints:** Circular molecular fingerprints encoding local chemical environment, used as ML features
- **Inverse design:** Starting from a desired property (high CO2 absorption) and searching for molecules with that property
- **DFT:** Quantum chemical method used to compute CO2 binding energy to validate ML predictions

---

## Pipeline Status

| Phase | Task | Status |
|---|---|---|
| 0 | Environment setup | ✅ Done |
| 1 | Data collection (ILThermo) | ⬜ Not started |
| 2 | Featurization (RDKit) | ⬜ Not started |
| 3 | Forward ML model | ⬜ Not started |
| 4 | Inverse design | ⬜ Not started |
| 5 | DFT validation (ORCA) | ⬜ Not started |
| 6 | Analysis & writeup | ⬜ Not started |

---

## References

- Brennecke & Maginn (2001) — foundational IL CO2 capture paper
- Ramdin et al. (2012) — comprehensive IL CO2 solubility review
- [ILThermo database](https://ilthermo.boulder.nist.gov)
- [RDKit documentation](https://www.rdkit.org/docs/)
- [ORCA manual](https://www.faccts.de/docs/orca/6.0/manual/)
