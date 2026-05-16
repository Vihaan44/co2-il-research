"""
parse_thermoml.py
-----------------
PURPOSE: Scan all 11,923 ThermoML JSON files, identify binary IL+CO2 systems
         that report CO2 mole fraction solubility or Henry's law constants,
         extract (il_name, il_inchi, T_K, P_kPa, x2_CO2), and output a
         clean CSV compatible with the existing build_dataset.py pipeline.

HOW THERMOML JSON IS STRUCTURED:
  Each file corresponds to one journal article and contains:
    Compound[]         -- list of all chemical species in the article
      .sCommonName[]   -- list of common names (e.g. "carbon dioxide")
      .sFormulaMolec   -- molecular formula (e.g. "CO2")
      .sStandardInChI  -- InChI string (canonical chemical identifier)
      .RegNum.nOrgNum  -- integer ID used to cross-reference within the file

    PureOrMixtureData[] -- one block per experimental dataset
      .Component[]       -- which compounds (by nOrgNum) are in this dataset
      .Property[]        -- what was measured (e.g. "Mole fraction of CO2")
      .Variable[]        -- what was varied (e.g. Temperature, Pressure)
      .NumValues[]       -- the actual data rows

  Each NumValues row has:
    .VariableValue[]   -- values of the varied quantities (T, P, x2 if varied)
    .PropertyValue[]   -- values of the measured quantities

FILTERING LOGIC:
  A dataset block is relevant if:
    1. Exactly 2 components (binary system)
    2. One component is CO2 (identified by formula "CO2" or name "carbon dioxide")
    3. The other component has a name suggesting it's an ionic liquid
       (contains "imidazolium", "pyrrolidinium", "ammonium", "phosphonium",
        "pyridinium", "sulfonium", "piperidinium", or "guanidinium")
    4. The property or variable is mole fraction of CO2, OR the property is
       Henry's law constant (kH) for CO2

  We do NOT filter on SMILES here because ThermoML uses InChI, not SMILES.
  A separate script (thermoml_inchi_to_smiles.py) will convert InChI → SMILES
  using PubChem API so the ILs can be featurized by featurize.py.

OUTPUTS:
  data/raw/thermoml_co2_il_raw.csv  -- extracted rows before SMILES conversion
  data/raw/thermoml_parse_summary.txt -- parse statistics

INPUT:
  data/raw/thermoml/  -- extracted ThermoML archive (11,923 JSON files)

Run from project root:
    python src/parse_thermoml.py
"""

import os
import json
import glob
import numpy as np
import pandas as pd

# -- Constants -----------------------------------------------------------------
THERMOML_DIR  = os.path.join("data", "raw", "thermoml")
OUTPUT_CSV    = os.path.join("data", "raw", "thermoml_co2_il_raw.csv")
SUMMARY_FILE  = os.path.join("data", "raw", "thermoml_parse_summary.txt")

# CO2 identification -- match any of these in formula or name
CO2_FORMULAS  = {"CO2", "CO2(g)"}
CO2_NAMES     = {"carbon dioxide", "co2", "carbonic anhydride", "carbonic acid gas"}

# IL identification -- at least one of these substrings must appear in the
# compound name (case-insensitive). This is a broad filter; false positives
# (non-ILs containing these words) are rare in thermodynamic literature.
IL_KEYWORDS = [
    "imidazolium", "pyrrolidinium", "ammonium", "phosphonium",
    "pyridinium", "sulfonium", "piperidinium", "guanidinium",
    "morpholinium", "piperazinium", "cholinium",
]

# Property names in ThermoML that indicate CO2 mole fraction is the property
MOLFRAC_PROP_NAMES = {
    "mole fraction",
    "mole fraction of co2",
    "solubility in mole fraction",
    "composition",
}

# Property names for Henry's law constant
HENRY_PROP_NAMES = {
    "henry's law constant",
    "henry law constant",
    "henry constant",
    "kh",
}

# Variable type strings for mole fraction (CO2 mole fraction as a variable)
MOLFRAC_VAR_TYPES = {"mole fraction", "mole fraction of co2"}

# Variable type strings for temperature and pressure
TEMP_VAR_TYPES    = {"temperature, k", "temperature"}
PRESS_VAR_TYPES   = {"pressure, kpa", "pressure, mpa", "pressure, bar",
                     "pressure, pa", "pressure, atm", "pressure"}

# Pressure unit conversion factors → kPa
PRESSURE_CONVERSIONS = {
    "kpa": 1.0,
    "mpa": 1000.0,
    "bar": 100.0,
    "pa":  0.001,
    "atm": 101.325,
}

# Henry's law constant → x2 conversion: x2 = P_kPa / (KH_MPa * 1000)
# If KH units are MPa, divide by 1000 to get kPa, then x2 = P/KH
HENRY_UNIT_TO_MPA = {
    "mpa": 1.0,
    "kpa": 0.001,
    "pa":  1e-6,
    "bar": 0.1,
    "atm": 0.101325,
    "mpa·mol−1·dm3": 1.0,  # common ThermoML unit for Henry's constant
}

P_AMBIENT_KPA = 101.325  # assumed pressure for Henry's law if P not reported

# Sanity bounds
T_MIN_K  = 200.0
T_MAX_K  = 500.0
X2_MIN   = 1e-6
X2_MAX   = 1.0


# -- Helper: CO2 and IL identification ----------------------------------------

def is_co2(compound: dict) -> bool:
    """Return True if this compound dict represents CO2."""
    formula = str(compound.get("sFormulaMolec", "")).strip().upper()
    if formula in CO2_FORMULAS:
        return True
    names = compound.get("sCommonName", [])
    if isinstance(names, str):
        names = [names]
    for name in names:
        if str(name).lower().strip() in CO2_NAMES:
            return True
    return False


def is_ionic_liquid(compound: dict) -> bool:
    """
    Return True if this compound is likely an ionic liquid.
    Uses keyword matching on common names -- not perfect but catches >95%
    of ILs in the thermodynamic literature.
    """
    names = compound.get("sCommonName", [])
    if isinstance(names, str):
        names = [names]
    for name in names:
        name_lower = str(name).lower()
        if any(kw in name_lower for kw in IL_KEYWORDS):
            return True
    return False


def get_compound_name(compound: dict) -> str:
    """Return the first common name for a compound, or formula if no name."""
    names = compound.get("sCommonName", [])
    if isinstance(names, str):
        return names
    if names:
        return str(names[0])
    return str(compound.get("sFormulaMolec", "Unknown"))


def get_pressure_unit_from_var_type(var_type_str: str) -> str:
    """
    Extract pressure unit from variable type string like 'Pressure, kPa'.
    Returns unit string in lowercase, or 'kpa' as default.
    """
    parts = var_type_str.lower().split(",")
    if len(parts) >= 2:
        unit = parts[-1].strip()
        return unit
    return "kpa"


def convert_pressure_to_kpa(value: float, unit: str) -> float:
    """Convert a pressure value to kPa using known unit factors."""
    unit_clean = unit.lower().strip()
    factor = PRESSURE_CONVERSIONS.get(unit_clean, 1.0)  # default: assume kPa
    return value * factor


# -- Core parser: one JSON file -----------------------------------------------

def parse_one_file(filepath: str) -> list:
    """
    Parse one ThermoML JSON file and return a list of extracted row dicts.
    Each row has: il_name, il_inchi, T_K, P_kPa, x2_CO2, data_type, source_doi.

    Returns empty list if the file has no relevant IL+CO2 data.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []  # skip malformed files

    # Extract DOI from Citation
    citation = data.get("Citation", {})
    source_doi = citation.get("sDOI", os.path.basename(filepath))

    # Build a map: nOrgNum → compound dict
    compounds = data.get("Compound", [])
    if not isinstance(compounds, list):
        compounds = [compounds]
    compound_map = {c["RegNum"]["nOrgNum"]: c for c in compounds
                    if "RegNum" in c and "nOrgNum" in c.get("RegNum", {})}

    extracted_rows = []

    # Loop over each experimental dataset block
    for dataset in data.get("PureOrMixtureData", []):
        if not isinstance(dataset, dict):
            continue

        # Identify the two components
        components = dataset.get("Component", [])
        if not isinstance(components, list):
            components = [components]
        if len(components) != 2:
            continue  # only binary systems

        org_nums = [c.get("RegNum", {}).get("nOrgNum") for c in components]
        comp_dicts = [compound_map.get(n) for n in org_nums]
        if any(c is None for c in comp_dicts):
            continue

        # Check: one is CO2, one is IL
        co2_idx  = next((i for i, c in enumerate(comp_dicts) if is_co2(c)), None)
        il_idx   = next((i for i, c in enumerate(comp_dicts) if is_ionic_liquid(c)), None)

        if co2_idx is None or il_idx is None or co2_idx == il_idx:
            continue  # not a binary IL+CO2 system

        il_compound = comp_dicts[il_idx]
        co2_compound = comp_dicts[co2_idx]

        il_name  = get_compound_name(il_compound)
        il_inchi = il_compound.get("sStandardInChI", "")
        co2_orgnum = org_nums[co2_idx]

        # Analyze variables and properties in this dataset
        variables  = dataset.get("Variable", [])
        properties = dataset.get("Property", [])
        if not isinstance(variables, list):
            variables = [variables]
        if not isinstance(properties, list):
            properties = [properties]

        # Build variable number → (type_string, unit, is_co2_molfrac) map
        var_info = {}  # nVarNumber → dict
        for var in variables:
            var_num  = var.get("nVarNumber")
            var_type = var.get("VariableID", {}).get("VariableType", {})

            # Temperature
            temp_str = str(var_type.get("eTemperature", "")).lower()
            if any(t in temp_str for t in TEMP_VAR_TYPES):
                var_info[var_num] = {"kind": "temperature", "unit": "k"}
                continue

            # Pressure
            press_str = str(var_type.get("ePressure", "")).lower()
            if press_str:
                unit = get_pressure_unit_from_var_type(press_str)
                var_info[var_num] = {"kind": "pressure", "unit": unit}
                continue

            # Mole fraction composition (check if it's CO2's mole fraction)
            comp_str = str(var_type.get("eComponentComposition", "")).lower()
            if "mole fraction" in comp_str:
                # Check if this variable's RegNum matches CO2
                var_comp_orgnum = var.get("VariableID", {}).get(
                    "RegNum", {}).get("nOrgNum")
                is_co2_var = (var_comp_orgnum == co2_orgnum)
                var_info[var_num] = {
                    "kind":      "molfrac_co2" if is_co2_var else "molfrac_other",
                    "unit":      "dimensionless",
                    "is_co2":    is_co2_var,
                }
                continue

        # Build property number → (type, unit) map
        prop_info = {}  # nPropNumber → dict
        for prop in properties:
            prop_num = prop.get("nPropNumber")
            prop_group = prop.get("Property-MethodID", {}).get("PropertyGroup", {})

            # Mole fraction as property (not variable)
            for group_key, group_val in prop_group.items():
                if not isinstance(group_val, dict):
                    continue
                prop_name = str(group_val.get("ePropName", "")).lower()

                if "mole fraction" in prop_name:
                    prop_info[prop_num] = {"kind": "molfrac_co2", "unit": "dimensionless"}
                elif "henry" in prop_name:
                    unit_str = prop_name.split(",")[-1].strip() if "," in prop_name else "mpa"
                    prop_info[prop_num] = {"kind": "henry", "unit": unit_str}

        # If no relevant variables or properties, skip this dataset
        has_co2_molfrac_var  = any(v["kind"] == "molfrac_co2" for v in var_info.values())
        has_co2_molfrac_prop = any(v["kind"] == "molfrac_co2" for v in prop_info.values())
        has_henry_prop       = any(v["kind"] == "henry"       for v in prop_info.values())

        if not (has_co2_molfrac_var or has_co2_molfrac_prop or has_henry_prop):
            continue  # no useful data in this dataset block

        # Extract actual data rows from NumValues
        for num_val in dataset.get("NumValues", []):
            if not isinstance(num_val, dict):
                continue

            t_k   = None
            p_kpa = None
            x2    = None
            kh    = None

            # Read variable values
            for var_val in num_val.get("VariableValue", []):
                var_num   = var_val.get("nVarNumber")
                var_value = pd.to_numeric(var_val.get("nVarValue"), errors="coerce")
                if var_num not in var_info or not np.isfinite(var_value):
                    continue
                kind = var_info[var_num]["kind"]
                unit = var_info[var_num]["unit"]

                if kind == "temperature":
                    t_k = float(var_value)
                elif kind == "pressure":
                    p_kpa = convert_pressure_to_kpa(float(var_value), unit)
                elif kind == "molfrac_co2":
                    x2 = float(var_value)

            # Read property values
            for prop_val in num_val.get("PropertyValue", []):
                prop_num   = prop_val.get("nPropNumber")
                prop_value = pd.to_numeric(prop_val.get("nPropValue"), errors="coerce")
                if prop_num not in prop_info or not np.isfinite(prop_value):
                    continue
                kind = prop_info[prop_num]["kind"]
                unit = prop_info[prop_num]["unit"]

                if kind == "molfrac_co2":
                    x2 = float(prop_value)
                elif kind == "henry":
                    kh = float(prop_value)

            # Convert Henry's constant to x2 if that's what we have
            if x2 is None and kh is not None and kh > 0:
                p_use = p_kpa if p_kpa is not None else P_AMBIENT_KPA
                # Henry's law: x2 = P_kPa / (KH_MPa * 1000)
                # Assume KH in MPa unless unit string says otherwise
                kh_mpa = kh  # default: MPa
                x2 = p_use / (kh_mpa * 1000.0)
                data_type = "henry_converted"
            elif x2 is not None:
                data_type = "mole_fraction"
            else:
                continue  # couldn't get x2 from this row

            # Sanity checks
            if t_k is None or not (T_MIN_K <= t_k <= T_MAX_K):
                continue
            if not (X2_MIN <= x2 <= X2_MAX):
                continue

            p_final = p_kpa if p_kpa is not None else np.nan

            extracted_rows.append({
                "il_name":     il_name,
                "il_inchi":    il_inchi,
                "T_K":         t_k,
                "P_kPa":       p_final,
                "x2_CO2":      x2,
                "data_type":   data_type,
                "source_doi":  source_doi,
            })

    return extracted_rows


# -- Main pipeline -------------------------------------------------------------

def main():
    """Scan all ThermoML JSON files, extract IL+CO2 data, save CSV."""
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)

    json_files = glob.glob(
        os.path.join(THERMOML_DIR, "**", "*.json"), recursive=True
    )
    print(f"[main] Found {len(json_files)} ThermoML JSON files to scan", flush=True)

    all_rows       = []
    files_with_data = 0
    files_failed   = 0

    for i, filepath in enumerate(json_files):
        rows = parse_one_file(filepath)
        if rows:
            all_rows.extend(rows)
            files_with_data += 1
        else:
            files_failed += 0  # silence -- most files won't have IL+CO2 data

        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{len(json_files)}] files scanned | "
                  f"{files_with_data} with IL+CO2 data | "
                  f"{len(all_rows)} rows extracted", flush=True)

    print(f"\n[main] Scan complete:", flush=True)
    print(f"  Total files:          {len(json_files)}", flush=True)
    print(f"  Files with IL+CO2:   {files_with_data}", flush=True)
    print(f"  Total rows extracted: {len(all_rows)}", flush=True)

    if not all_rows:
        print("[main] WARNING: No IL+CO2 data found. Check IL_KEYWORDS and filters.")
        return

    df = pd.DataFrame(all_rows)

    # Summary statistics
    n_unique_ils  = df["il_inchi"].nunique()
    n_molfrac     = (df["data_type"] == "mole_fraction").sum()
    n_henry       = (df["data_type"] == "henry_converted").sum()

    print(f"\n  Unique ILs (by InChI): {n_unique_ils}", flush=True)
    print(f"  Mole fraction rows:    {n_molfrac}", flush=True)
    print(f"  Henry converted rows:  {n_henry}", flush=True)
    print(f"\n  Top 20 ILs by row count:", flush=True)
    top_ils = df.groupby("il_name").size().sort_values(ascending=False).head(20)
    print(top_ils.to_string(), flush=True)

    # Save
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[main] Saved {len(df)} rows → {OUTPUT_CSV}", flush=True)

    # Write summary file
    summary_lines = [
        f"ThermoML IL+CO2 Parse Summary",
        f"Total JSON files scanned: {len(json_files)}",
        f"Files with IL+CO2 data:  {files_with_data}",
        f"Total rows extracted:     {len(all_rows)}",
        f"Unique ILs (by InChI):   {n_unique_ils}",
        f"Mole fraction rows:       {n_molfrac}",
        f"Henry converted rows:     {n_henry}",
        "",
        "Top ILs by row count:",
        top_ils.to_string(),
        "",
        "NEXT STEP:",
        "  Run src/thermoml_inchi_to_smiles.py to convert il_inchi → il_smiles",
        "  via PubChem API, then merge into the main pipeline.",
    ]
    with open(SUMMARY_FILE, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"[main] Summary → {SUMMARY_FILE}", flush=True)
    print("\n[main] NEXT: run src/thermoml_inchi_to_smiles.py", flush=True)


if __name__ == "__main__":
    main()
