"""Readers for resistance-vs-temperature data.

Formats:

* Quantum Design MultiVu ``.dat`` (PPMS ETO, Resistivity option, DynaCool, ...):
  header up to ``[Data]``, then a comma-separated column line. All columns that
  contain "Resistance"/"Resistivity" (without "Std. Dev." etc.) become channels.
* LabVIEW files from the Zurich Instruments lock-in setup (same format as the
  higher-harmonics program): optional instrument header line with ``I=``, data
  columns ``Timestamp, Temperature, Field, Angle, Frequency, ReZ/ImZ (1st..3rd
  harmonic), Re I, Im I``. Each demodulator voltage becomes a channel; the
  resistance is ``V / I`` with I from the header or from the measured current.
* Any delimited numeric table with a column-name line (generic fallback).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"

LOCKIN_COLUMNS = [
    "Timestamp VI [s]", "Temperature [K]", "Field [Oe]", "Angle [deg]", "Frequency [Hz]",
    "ReZ 1st Harm [V]", "ImZ 1st Harm [V]", "ReZ 2nd Harm [V]", "ImZ 2nd Harm [V]",
    "ReZ 3rd Harm [V]", "ImZ 3rd Harm [V]", "Re I [A]", "Im I [A]",
]

_SKIP_WORDS = ("std", "dev", "quality", "status", "phase", "excitation", "range", "drive",
               "voltage", "current")


@dataclass
class Channel:
    name: str
    values: np.ndarray
    unit: str          # 'Ohm', 'Ohm cm' or 'V'
    kind: str = "R"    # 'R' (resistance) or 'rho' (resistivity already)


@dataclass
class RTData:
    path: Path
    temperature: np.ndarray
    channels: list
    time: np.ndarray | None = None
    field_oe: np.ndarray | None = None
    current_A: float | None = None
    source: str = ""
    meta: dict = field(default_factory=dict)
    table: pd.DataFrame | None = field(default=None, repr=False)

    def channel(self, name) -> Channel:
        for c in self.channels:
            if c.name == name:
                return c
        raise KeyError(name)

    @property
    def channel_names(self):
        return [c.name for c in self.channels]


def _find_col(columns, *patterns, exclude=()):
    for pat in patterns:
        for c in columns:
            lc = c.lower()
            if re.search(pat, lc) and not any(e in lc for e in exclude):
                return c
    return None


# --------------------------------------------------------------------------
# Quantum Design
# --------------------------------------------------------------------------
def read_qd_dat(path) -> RTData:
    path = Path(path)
    lines = path.read_text(encoding="latin-1").splitlines()
    try:
        i = next(k for k, l in enumerate(lines) if l.strip().startswith("[Data]"))
    except StopIteration:
        raise ValueError("No [Data] section: not a Quantum Design file") from None
    header = "\n".join(lines[:i])
    names = [c.strip() for c in lines[i + 1].split(",")]
    rows = [l.split(",") for l in lines[i + 2:] if l.strip()]
    n = len(names)
    rows = [(r + [""] * n)[:n] for r in rows]
    df = pd.DataFrame(rows, columns=names).apply(pd.to_numeric, errors="coerce")
    tcol = _find_col(names, r"^temperature", r"temp")
    if tcol is None:
        raise ValueError("No temperature column found")
    chans = []
    for c in names:
        lc = c.lower()
        if not ("resist" in lc):
            continue
        if any(w in lc for w in ("std", "dev", "quality", "status")):
            continue
        vals = df[c].to_numpy(float)
        if np.isfinite(vals).sum() < 3:
            continue
        if "resistivity" in lc:
            unit = "Ohm cm" if "cm" in lc else "Ohm m"
            chans.append(Channel(c, vals, unit, "rho"))
        else:
            chans.append(Channel(c, vals, "Ohm", "R"))
    if not chans:
        raise ValueError("No resistance columns with data found")
    src = "QD ETO" if "eto" in header.lower() or any("ch1" in c.name.lower() or "eto" in c.name.lower()
                                                      for c in chans) else "QD"
    if any("bridge" in c.name.lower() for c in chans):
        src = "QD Resistivity option"
    fcol = _find_col(names, r"magnetic field", r"field")
    time_col = _find_col(names, r"time stamp", r"time")
    return RTData(path=path, temperature=df[tcol].to_numpy(float), channels=chans,
                  time=df[time_col].to_numpy(float) if time_col else None,
                  field_oe=df[fcol].to_numpy(float) if fcol else None,
                  source=src, meta={"header": header}, table=df)


# --------------------------------------------------------------------------
# LabVIEW lock-in files
# --------------------------------------------------------------------------
def parse_instrument_header(line: str) -> dict:
    info: dict = {"raw": line.strip()}
    for key, val in re.findall(rf"([A-Za-z_]+)=\s*_?\s*,\s*({_NUM})", line):
        info[key.strip("_")] = float(val)
    m = re.search(rf"({_NUM})\s*,\s*_Hz_", line)
    if m:
        info["f_Hz"] = float(m.group(1))
    return info


_TEXT_RE = re.compile(r"[A-DF-Za-df-z_=°]")


def _is_text(line: str) -> bool:
    return bool(_TEXT_RE.search(re.sub(r"(?i)\bnan\b", "", line)))


def _split(line, sep):
    return [p.strip() for p in line.strip().split(sep)]


def read_lockin(path, current_source: str = "auto") -> RTData:
    """LabVIEW lock-in file. ``current_source``: 'header', 'measured' or 'auto'."""
    path = Path(path)
    lines = [l for l in path.read_text(encoding="latin-1").splitlines() if l.strip()]
    data_lines = [l for l in lines if not _is_text(l)]
    text_lines = [l for l in lines if _is_text(l)]
    if len(data_lines) < 3:
        raise ValueError("No numeric data")
    sample = data_lines[0]
    sep = "\t" if "\t" in sample else (";" if ";" in sample else ",")
    hdr = {}
    names = None
    for l in text_lines:
        if "_Harm_" in l or "I=" in l:
            hdr = parse_instrument_header(l)
        elif names is None and len(_split(l, sep)) >= 3:
            names = _split(l, sep)
    rows = [_split(l, sep) for l in data_lines]
    ncol = Counter(len(r) for r in rows).most_common(1)[0][0]
    if names is None or len(names) < ncol:
        names = (names or []) + LOCKIN_COLUMNS[len(names or []):]
        names = names[:ncol] if len(names) >= ncol else names + [f"col{i}" for i in range(len(names), ncol)]
    names = names[:ncol]
    rows = [(r + [""] * ncol)[:ncol] for r in rows]
    df = pd.DataFrame(rows, columns=names).apply(pd.to_numeric, errors="coerce")

    tcol = _find_col(names, r"temp")
    if tcol is None:
        raise ValueError("No temperature column")
    fcol = _find_col(names, r"feld|field")
    time_col = _find_col(names, r"time")
    ire = _find_col(names, r"^re\s*i\b", r"re i")
    iim = _find_col(names, r"^im\s*i\b", r"im i")
    I_meas = None
    if ire is not None:
        im = df[iim].to_numpy(float) if iim is not None else 0.0
        I_meas = float(np.nanmean(np.hypot(df[ire].to_numpy(float), im)))
        if not np.isfinite(I_meas) or I_meas == 0:
            I_meas = None
    I_hdr = hdr.get("I")
    if current_source == "header":
        current = I_hdr
    elif current_source == "measured":
        current = I_meas
    else:
        current = I_hdr or I_meas

    chans = []
    for c in names:
        lc = c.lower()
        if c in (tcol, fcol, time_col, ire, iim):
            continue
        is_v = "[v]" in lc or "rez" in lc or "imz" in lc or lc.startswith("x") or "demod" in lc
        is_r = "ohm" in lc or "[Ω]" in c
        if not (is_v or is_r):
            continue
        vals = df[c].to_numpy(float)
        if np.isfinite(vals).sum() < 3:
            continue
        if is_r:
            chans.append(Channel(c, vals, "Ohm", "R"))
        elif current:
            chans.append(Channel(c.replace("[V]", "").strip() + " / I", vals / current, "Ohm", "R"))
        else:
            chans.append(Channel(c, vals, "V", "R"))
    if not chans:
        raise ValueError("No voltage or resistance columns found")
    return RTData(path=path, temperature=df[tcol].to_numpy(float), channels=chans,
                  time=df[time_col].to_numpy(float) if time_col else None,
                  field_oe=df[fcol].to_numpy(float) if fcol else None,
                  current_A=current, source="Lock-in (LabVIEW)",
                  meta={"instrument_header": hdr, "I_header": I_hdr, "I_measured": I_meas},
                  table=df)


# --------------------------------------------------------------------------
# Generic table
# --------------------------------------------------------------------------
def read_table(path, t_col=None, r_cols=None) -> RTData:
    path = Path(path)
    lines = [l for l in path.read_text(encoding="latin-1").splitlines() if l.strip()]
    data = [l for l in lines if not _is_text(l)]
    if len(data) < 3:
        raise ValueError("No numeric data")
    sep = "\t" if "\t" in data[0] else (";" if ";" in data[0] else ("," if "," in data[0] else None))
    sp = (lambda l: l.split()) if sep is None else (lambda l: _split(l, sep))
    rows = [sp(l) for l in data]
    ncol = Counter(len(r) for r in rows).most_common(1)[0][0]
    names = None
    for l in lines:
        if _is_text(l) and len(sp(l)) == ncol:
            names = sp(l)
    names = names or [f"col{i}" for i in range(ncol)]
    rows = [r for r in rows if len(r) == ncol]
    df = pd.DataFrame(rows, columns=names).apply(pd.to_numeric, errors="coerce")
    if t_col is None:
        t_col = _find_col(names, r"temp", r"^t\b", r"\bt \[") or names[0]
    if r_cols is None:
        r_cols = [c for c in names if c != t_col and re.search(r"res|ohm|\br\b|rho|ρ|Ω", c.lower())]
        if not r_cols:
            r_cols = [c for c in names if c != t_col][:1]
    chans = [Channel(c, df[c].to_numpy(float),
                     "Ohm cm" if ("rho" in c.lower() or "cm" in c.lower()) else "Ohm",
                     "rho" if ("rho" in c.lower() or "cm" in c.lower()) else "R") for c in r_cols]
    return RTData(path=path, temperature=df[t_col].to_numpy(float), channels=chans,
                  source="table", table=df)


def load_rt(path) -> RTData:
    """Detect the format and read the file."""
    path = Path(path)
    head = path.read_text(encoding="latin-1", errors="ignore")[:20000]
    if "[Data]" in head:
        return read_qd_dat(path)
    if "_Harm_" in head or "ReZ" in head or "Timestamp VI" in head or re.search(r"Feld \[Oe\]", head):
        return read_lockin(path)
    return read_table(path)
