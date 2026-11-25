"""Readers for X-ray diffraction data.

Supported:

* ``.xy`` / ``.txt`` / ``.csv`` two-column tables (2theta, intensity), any of the
  separators whitespace, comma, semicolon or tab; header lines are skipped.
* PANalytical ``.xrdml`` (1D scans and area maps / RSMs). Both
  ``startPosition``/``endPosition`` and ``listPositions`` are handled, as are
  ``intensities`` and ``counts``.
* PANalytical Data Viewer CSV export with a ``[Scan points]`` block (used for RSMs).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

CU_KA1 = 1.5405980
CU_KA2 = 1.5444260
KA2_RATIO = 0.5


@dataclass
class Scan1D:
    two_theta: np.ndarray
    intensity: np.ndarray
    wavelength: float = CU_KA1
    wavelength2: float | None = CU_KA2
    ka2_ratio: float = KA2_RATIO
    meta: dict = field(default_factory=dict)
    path: Path | None = None


@dataclass
class AreaScan:
    """2D map, one row per individual scan line (e.g. per omega offset)."""

    two_theta: np.ndarray  # shape (n_lines, n_points)
    omega: np.ndarray      # shape (n_lines, n_points)
    intensity: np.ndarray  # shape (n_lines, n_points)
    wavelength: float = CU_KA1
    meta: dict = field(default_factory=dict)
    path: Path | None = None


_NUM_RE = re.compile(r"^\s*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")


def read_two_column(path) -> Scan1D:
    path = Path(path)
    rows = []
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            if not _NUM_RE.match(line):
                continue
            parts = [p for p in re.split(r"[,;\t ]+", line.strip()) if p]
            if len(parts) < 2:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No numeric two-column data found in {path.name}")
    arr = np.array(rows, dtype=float)
    arr = arr[np.argsort(arr[:, 0])]
    return Scan1D(arr[:, 0], arr[:, 1], path=path, meta={"format": "xy"})


# --------------------------------------------------------------------------
# XRDML
# --------------------------------------------------------------------------
def _strip_ns(root: ET.Element) -> None:
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]


def _floats(text: str | None) -> np.ndarray:
    if not text:
        return np.array([], dtype=float)
    return np.array([float(t) for t in text.split()], dtype=float)


def _axis_positions(dp: ET.Element, axis: str, n: int) -> np.ndarray | None:
    for pos in dp.findall("positions"):
        if pos.get("axis", "").lower() != axis.lower():
            continue
        lst = pos.find("listPositions")
        if lst is not None and lst.text:
            vals = _floats(lst.text)
            if vals.size == n:
                return vals
            if vals.size > 0:
                return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, vals.size), vals)
        start, end = pos.find("startPosition"), pos.find("endPosition")
        if start is not None and end is not None:
            return np.linspace(float(start.text), float(end.text), n)
        common = pos.find("commonPosition")
        if common is not None:
            return np.full(n, float(common.text))
    return None


def _read_xrdml_scans(path: Path):
    root = ET.parse(path).getroot()
    _strip_ns(root)
    meta = {"format": "xrdml"}
    wl = root.find(".//usedWavelength")
    ka1, ka2, ratio = CU_KA1, CU_KA2, KA2_RATIO
    if wl is not None:
        if wl.find("kAlpha1") is not None:
            ka1 = float(wl.find("kAlpha1").text)
        if wl.find("kAlpha2") is not None:
            ka2 = float(wl.find("kAlpha2").text)
        if wl.find("ratioKAlpha2KAlpha1") is not None:
            ratio = float(wl.find("ratioKAlpha2KAlpha1").text)
    meta.update(kalpha1=ka1, kalpha2=ka2, ka2_ratio=ratio)
    sample = root.find(".//sample/id")
    if sample is not None and sample.text:
        meta["sample"] = sample.text.strip()

    scans = []
    for scan in root.iter("scan"):
        dp = scan.find("dataPoints")
        if dp is None:
            continue
        inten_el = dp.find("intensities")
        counts_el = dp.find("counts")
        el = inten_el if inten_el is not None else counts_el
        if el is None:
            continue
        y = _floats(el.text)
        n = y.size
        tt = _axis_positions(dp, "2Theta", n)
        om = _axis_positions(dp, "Omega", n)
        ct = dp.find("commonCountingTime")
        t = float(ct.text) if ct is not None else None
        scans.append({"two_theta": tt, "omega": om, "intensity": y, "counting_time": t,
                      "axis": scan.get("scanAxis", "")})
    if not scans:
        raise ValueError(f"No data points found in {path.name}")
    return scans, meta


def read_xrdml(path, counts_per_second: bool = False):
    """Return a :class:`Scan1D` for a single scan, or :class:`AreaScan` for maps."""
    path = Path(path)
    scans, meta = _read_xrdml_scans(path)

    def scale(s):
        y = s["intensity"].astype(float)
        if counts_per_second and s["counting_time"]:
            y = y / s["counting_time"]
        return y

    if len(scans) == 1:
        s = scans[0]
        if s["two_theta"] is None:
            raise ValueError("xrdml scan has no 2Theta axis")
        order = np.argsort(s["two_theta"])
        return Scan1D(s["two_theta"][order], scale(s)[order], wavelength=meta["kalpha1"],
                      wavelength2=meta["kalpha2"], ka2_ratio=meta["ka2_ratio"], meta=meta, path=path)

    n = min(s["intensity"].size for s in scans)
    tt = np.vstack([s["two_theta"][:n] for s in scans])
    om = np.vstack([s["omega"][:n] for s in scans])
    y = np.vstack([scale(s)[:n] for s in scans])
    return AreaScan(tt, om, y, wavelength=meta["kalpha1"], meta=meta, path=path)


# --------------------------------------------------------------------------
# PANalytical CSV export ([Scan points])
# --------------------------------------------------------------------------
def read_panalytical_csv(path) -> AreaScan:
    path = Path(path)
    text = path.read_bytes().decode("latin-1").splitlines()
    meta, start = {"format": "panalytical-csv"}, None
    for i, line in enumerate(text):
        if line.strip() == "[Scan points]":
            start = i + 1
            break
        if "," in line and not line.startswith("Comment -"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                meta[parts[0]] = ",".join(parts[1:]).strip()
    if start is None:
        raise ValueError("No [Scan points] block found")
    header = [h.strip() for h in text[start].split(",")]
    df = pd.read_csv(StringIO("\n".join(text[start + 1:])), header=None)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    cols = {name.lower(): i for i, name in enumerate(header)}

    def col(*keys, default=None):
        for k in keys:
            for name, i in cols.items():
                if k in name:
                    return i
        return default

    i_tt = col("2theta", default=0)
    i_om = col("omega", default=1)
    i_int = col("intensity", "counts", default=df.shape[1] - 1)
    tt = df.iloc[:, i_tt].to_numpy(float)
    om = df.iloc[:, i_om].to_numpy(float)
    y = df.iloc[:, i_int].to_numpy(float)
    wl = float(meta.get("K-Alpha1 wavelength", CU_KA1))
    return area_from_points(tt, om, y, wavelength=wl, meta=meta, path=path)


def area_from_points(tt, om, y, wavelength=CU_KA1, meta=None, path=None) -> AreaScan:
    """Arrange scattered (2theta, omega, I) points into scan lines.

    Lines are detected by the relative omega offset (omega - 2theta/2) which is
    constant along a 2theta-omega line; falls back to a pivot on omega.
    """
    tt, om, y = map(lambda a: np.asarray(a, float), (tt, om, y))
    candidates = [np.round(om - tt / 2, 4), np.round(om, 4), np.round(tt, 4)]
    sizes = [np.unique(c).size for c in candidates]
    valid = [i for i, s in enumerate(sizes) if 1 < s <= tt.size / 3]
    if not valid:
        raise ValueError("Could not arrange the points into scan lines")
    key = candidates[min(valid, key=lambda i: sizes[i])]
    lines = [np.flatnonzero(key == u) for u in np.unique(key)]
    n = min(len(i) for i in lines)
    rows = []
    for idx in lines:
        idx = idx[np.lexsort((om[idx], tt[idx]))][:n]
        rows.append(idx)
    idx = np.vstack(rows)
    return AreaScan(tt[idx], om[idx], y[idx], wavelength=wavelength, meta=meta or {}, path=path)


def load_scan(path):
    path = Path(path)
    if path.suffix.lower() == ".xrdml":
        return read_xrdml(path)
    return read_two_column(path)


def load_area(path) -> AreaScan:
    path = Path(path)
    if path.suffix.lower() == ".xrdml":
        res = read_xrdml(path)
        if not isinstance(res, AreaScan):
            raise ValueError("This xrdml file contains a single scan, not a map")
        return res
    return read_panalytical_csv(path)
