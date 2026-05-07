"""
setup.py — Project scaffolding script for co2-il-research.

Run this ONCE from your local project root after cloning the repo.
It creates the full folder structure needed for all phases of the project.

Usage:
    python setup.py

Expects: nothing (run from any empty folder)
Outputs: all project subdirectories + placeholder .gitkeep files
"""

import os

# ---------------------------------------------------------------------------
# All directories the project needs, organized by phase.
# Add new folders here if the project grows — never create ad-hoc folders.
# ---------------------------------------------------------------------------
PROJECT_DIRS = [
    "src",              # All Python source scripts
    "data/raw",         # Raw data downloaded from ILThermo (never modify these)
    "data/processed",   # Cleaned, featurized data ready for ML
    "data/virtual_library",  # Combinatorial IL pairs for inverse design screening
    "models",           # Saved trained models (joblib .pkl files)
    "results",          # Output CSVs: predictions, rankings, DFT results
    "figures",          # All plots and visualizations (saved at 300 DPI)
    "dft",              # ORCA input/output files for DFT validation
    "paper",            # Research paper drafts, abstract, competition materials
    "notebooks",        # Exploratory Jupyter notebooks (not for final pipeline)
]


def create_project_structure(directories):
    """
    Creates each directory in the list and adds a .gitkeep file so Git
    tracks the empty folder. Git doesn't track empty directories by default,
    so .gitkeep is a standard workaround.
    """
    for directory in directories:
        # Create the directory (and any missing parent dirs) if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        # Add .gitkeep so Git tracks this otherwise-empty folder
        gitkeep_path = os.path.join(directory, ".gitkeep")
        if not os.path.exists(gitkeep_path):
            open(gitkeep_path, "w").close()  # creates an empty file

        print(f"  Created: {directory}/")


def main():
    print("Setting up co2-il-research project structure...")
    print()
    create_project_structure(PROJECT_DIRS)
    print()
    print("Done! Your project is ready.")
    print("Next step: activate your virtual environment and run:")
    print("    pip install -r requirements.txt")


if __name__ == "__main__":
    main()
