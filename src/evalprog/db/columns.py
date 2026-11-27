"""Which results are 'main' values and how columns are named and ordered.

Main values are the few numbers one compares between samples (e.g. the corrected
lattice parameter averaged over all film peaks); everything else (single peaks,
fit quality, positions ...) is a detail and only shown on request.
"""

from __future__ import annotations

# technique -> main result keys, in display order. Keys may end with '*' (prefix match).
MAIN = {
    "XRD": ["c", "eps_zz", "zero shift Z"],
    "RSM": ["a_ip", "c", "eps_xx", "eps_zz", "relaxation"],
    "R(T)": ["Tc_onset", "Tc_mid", "Tc_zero", "dTc", "T_MI*", "T_Rmin*", "hysteresis", "T_feature*",
             "RRR*", "rho(300 K)"],
    "R(H)": ["T", "I", "R_H", "n", "dRxy/dB", "dT_2w max"],
    "AFM": ["Sq", "Sa", "Sz", "Step height*", "Mean |step|*", "Mean terrace width*", "Miscut angle",
            "Terrace width*", "pits: density", "pits: area fraction", "islands: density",
            "islands: area fraction"],
}
# keys shown under another name
ALIAS = {("XRD", "c"): "c mean", ("RSM", "c"): "c"}

# synthesis parameters shown right after the sample columns (in this order)
KEY_SYNTHESIS = ["T_Dep", "T_Shutter", "Dep volume", "Dep sps", "Air-Flow", "Solvent volume", "Prec1 m",
                 "Prec2 m", "Prec3 m", "Substrate Nr"]

SAMPLE_COLUMNS = ["sample", "technique", "material", "substrate", "orientation", "thickness_nm"]
META_COLUMNS = ["sample_type", "label", "measured_at", "saved_at", "raw_path", "files"]
TECH_ORDER = ["XRD", "RSM", "AFM", "R(T)", "R(H)", "MPMS", "other"]


def _match(key: str, pattern: str) -> bool:
    return key.startswith(pattern[:-1]) if pattern.endswith("*") else key == pattern


def main_rank(technique: str, key: str, label: str = "") -> int | None:
    """Position of a main result in the display order, or None for a detail."""
    pats = MAIN.get(technique)
    if pats is None:
        return 0
    for i, p in enumerate(pats):
        if _match(key, p):
            return i
    return None


def display_key(technique: str, key: str) -> str:
    return ALIAS.get((technique, key), key)


def column_name(technique: str, key: str, label: str, unit: str, main: bool, show_label: bool) -> str:
    name = f"{technique} {display_key(technique, key) if main else key}"
    if label and (show_label or not main):
        name += f" [{label}]"
    if unit:
        name += f" ({unit})"
    return name


def syn_column(key: str, unit: str = "") -> str:
    return f"syn: {key}" + (f" ({unit})" if unit else "")
