"""Hall and magnetoresistance with higher harmonics (lock-in 1f, 2f, 3f).

Port of the stand-alone higher-harmonics program. Each file is classified
automatically as a field sweep (large field range) or an R(T) curve (field fixed,
temperature varies).

Symmetrisation (Onsager, with hysteresis)::

    sym_down(B)  = [S_down(B) + S_up(-B)] / 2
    anti_down(B) = [S_down(B) - S_up(-B)] / 2
    sym_up(B)    =  sym_down(-B),   anti_up(B) = -anti_down(-B)

Heating from the 3rd harmonic::

    dT_2w(B) = 2 |V_3f| / (I |dR/dT|)

with dR/dT from a matching R(T) curve (covers T, same resistance -> same
contacts, same angle, smallest field).

Hall: the normal Hall effect is fitted to the antisymmetric part for
|B| >= B_min with ``R0*B + A*sgn(B)`` (saturated anomalous Hall) or ``R0*B`` only;
outputs dRxy/dB, R_H = t * dRxy/dB and n = 1/(e |R_H|).

Output: all CSV files go to ``<measurement folder>/output/``, figures to
``output/figures/<measurement>/`` (comparisons: ``output/figures/comparison/``).
Each CSV starts with ``# key: value`` lines followed by three column header lines
for Origin: long name, unit, comment (= legend text).

Everything here is GUI-independent (matplotlib Figure API only, no pyplot).
"""

from __future__ import annotations

import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter

try:
    from matplotlib import colormaps as _colormaps
    _VIRIDIS = _colormaps["viridis"]
except ImportError:  # older matplotlib
    from matplotlib import cm as _cm
    _VIRIDIS = _cm.get_cmap("viridis")
CMAP = ListedColormap(_VIRIDIS(np.linspace(0.0, 0.9, 256)))  # without the pale yellow

# ============================================================
# Constants
# ============================================================
TIME_COL = "Timestamp VI [s]"
TEMP_COL = "Temperature [K]"
FIELD_COL = "Field [Oe]"
ANGLE_COL = "Angle [deg]"
FREQ_COL = "Frequency [Hz]"
BRANCH = "branch"

ALL_COLUMNS = [
    TIME_COL, TEMP_COL, FIELD_COL, ANGLE_COL, FREQ_COL,
    "ReZ 1st Harm [V]", "ImZ 1st Harm [V]",
    "ReZ 2nd Harm [V]", "ImZ 2nd Harm [V]",
    "ReZ 3rd Harm [V]", "ImZ 3rd Harm [V]",
    "Re I [A]", "Im I [A]",
]
VOLTAGE_COLS = ALL_COLUMNS[5:11]
CURRENT_COLS = ALL_COLUMNS[11:13]
RE1, RE3, IM3 = "ReZ 1st Harm [V]", "ReZ 3rd Harm [V]", "ImZ 3rd Harm [V]"
HARMONIC_ORDER = {"1st": 1, "2nd": 2, "3rd": 3}

OE_PER_T = 1e4
RHO_OHM_M_TO_UOHM_CM = 1e8
E_CHARGE = 1.602176634e-19

METHODS = {"interp": "Interpolation", "bin": "Bin average (stepwise)"}
RESTART_MODES = {"last": "use last section", "each": "evaluate sections separately",
                 "all": "evaluate everything together"}
COMPONENTS = {"raw": "raw", "sym": "sym", "antisym": "antisym", "hall": "antisym − normal Hall"}
QUANTITIES = {"V": "Voltage [V]", "R": "Resistance [Ω]", "rho": "Resistivity [µΩ cm]",
              "dT": "ΔT₂ω [K] (from heating)"}
HALL_QUANT = {k: QUANTITIES[k] for k in ("V", "R", "rho")}
NORMS = {"none": "none", "In": "÷ Iⁿ", "mr": "MR: (S − S₀)/S₀ in %", "rel": "÷ S(B = 0)"}
PARAMS = {"T": ("Temperature", "K"), "I": ("Current", "µA"), "f": ("Frequency", "Hz"),
          "angle": ("Angle", "°"), "idx": ("Order", "")}
PARAM_LABELS = {k: v[0] for k, v in PARAMS.items()}
SERIES_LABELS = {k: v for k, v in PARAM_LABELS.items() if k != "idx"}
HALL_MODELS = {"sat": "R₀·B + A·sgn(B)", "lin": "R₀·B only"}
RT_QUANT = {"R": "R [Ω]", "rho": "ρ [µΩ cm]"}
BRANCHES = ("both", "down", "up")

# matplotlib is not thread-safe: background plotting and GUI drawing are serialised.
MPL_LOCK = threading.Lock()

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def fmt(v, spec: str) -> str:
    if v is None:
        return "–"
    try:
        if not np.isfinite(v):
            return "–"
    except TypeError:
        pass
    return format(v, spec)


def short_name(col: str) -> str:
    return re.sub(r"\s*\[.*?\]", "", col).strip()


def safe_name(s: str) -> str:
    s = s.replace("µ", "u").replace("°", "deg").replace("ρ", "rho").replace("Δ", "d")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")


def harmonic_of(col: str) -> int:
    return next((n for k, n in HARMONIC_ORDER.items() if k in col), 1)


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================
# Output (Origin-friendly CSV files)
# ============================================================
def out_dir(path: Path) -> Path:
    d = Path(path).parent / "output"
    d.mkdir(exist_ok=True)
    return d


def img_dir(path: Path, name: str | None = None) -> Path:
    d = Path(path).parent / "output" / "figures" / (name or Path(path).stem)
    d.mkdir(parents=True, exist_ok=True)
    return d


ORIGIN_NOTE = ("Column headers (Origin): row 1 = long name; row 2 = unit; row 3 = comment "
               "(used as legend in Origin). Comma separated; decimal point.")


def _clean(s) -> str:
    return str(s).replace(",", ";").replace("\r", " ").replace("\n", " ")


def split_unit(name: str):
    """'ReZ 1st Harm [µOhm cm]_sym' -> ('ReZ 1st Harm sym', 'µOhm cm')"""
    m = re.match(r"^(.*?)\s*\[(.*?)\](.*)$", str(name))
    if not m:
        return str(name).strip(), ""
    return (m.group(1).strip() + " " + m.group(3).replace("_", " ").strip()).strip(), m.group(2).strip()


def df_columns(df: pd.DataFrame, comment: str = "") -> list:
    cols = []
    for c in df.columns:
        n, u = split_unit(c)
        cols.append((n, u, comment if u and c != FIELD_COL else "", df[c].to_numpy()))
    return cols


def write_origin(path: Path, meta: dict, columns: list):
    """CSV with a metadata block (# key: value) and three header lines (long name,
    unit, comment). ``columns``: list of (name, unit, comment, values); columns of
    different length are padded with empty cells."""
    lines = [f"# {_clean(k)}: {_clean(v)}" for k, v in meta.items() if v not in (None, "")]
    lines.append(f"# {ORIGIN_NOTE}")
    for row in range(3):
        lines.append(",".join(_clean(c[row]) for c in columns))
    data = pd.concat([pd.Series(np.asarray(c[3])) for c in columns], axis=1, ignore_index=True)
    body = data.to_csv(header=False, index=False, na_rep="", float_format="%.10g")
    text = "\n".join(lines) + "\n" + body.replace("\r\n", "\n")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        f.write(text.replace("\n", "\r\n"))


def field_columns(B_oe: np.ndarray) -> list:
    return [("B", "T", "", np.asarray(B_oe, float) / OE_PER_T)]


def current_density(I, header: dict, s):
    """j = I / (w·t), otherwise from the instrument header (scaled to the current used)."""
    if not I:
        return None, ""
    if s.w_um > 0 and s.t_um > 0:
        return I / (s.w_um * 1e-6 * s.t_um * 1e-6), "I / (w·t)"
    j_h, I_h = header.get("j"), header.get("I")
    if j_h and I_h:
        return j_h * I / I_h, "from instrument header"
    return None, ""


def base_meta(path: Path, kind: str, header: dict, s, I, j, j_src, f_Hz, angle) -> dict:
    geo = "longitudinal (rho_xx = R·w·t / L)" if s.geometry == "xx" else "Hall (rho_xy = R·t)"
    dims = " / ".join(fmt(v if v > 0 else None, "g") for v in (s.L_um, s.w_um, s.t_um))
    return {
        "Measurement": path.name,
        "Type": kind,
        "Evaluated": now(),
        "Temperature [K]": None,
        "Magnetic field [T]": None,
        "Current I [A]": f"{I:.5e} ({'header' if s.current_source == 'header' else 'measured'})" if I else None,
        "Current density j [A/m^2]": f"{j:.4e} ({j_src})" if j else None,
        "Frequency [Hz]": fmt(f_Hz, "g"),
        "Angle [deg]": fmt(angle, ".2f"),
        "Excitation U [V]": fmt(header.get("U_V"), "g"),
        "Series resistor R [Ohm] (I = U/R)": fmt(header.get("R"), "g"),
        "Contact geometry": geo,
        "Sample L / w / t [um]": dims,
        "Instrument header": header.get("raw"),
    }


def tag_of(T_K, I, f_Hz, angle) -> str:
    """Short comment for the Origin legend, e.g. '10.00 K; 40 µA; 317 Hz; 0°'."""
    parts = []
    if T_K is not None:
        parts.append(f"{T_K:.2f} K")
    if I:
        parts.append(f"{I * 1e6:.3g} µA")
    if f_Hz is not None:
        parts.append(f"{f_Hz:g} Hz")
    if angle is not None:
        parts.append(f"{angle:.1f}°")
    return "; ".join(parts)


# ============================================================
# Settings / results
# ============================================================
@dataclass
class Settings:
    mode: str = "voltage"           # output of the single files: "voltage" or "rho"
    geometry: str = "xx"            # "xx" (longitudinal) or "xy" (Hall)
    current_source: str = "header"  # "header" or "measured"
    L_um: float = 0.0
    w_um: float = 0.0
    t_um: float = 0.0
    field_step: float = 50.0        # Oe
    method: str = "interp"          # "interp" or "bin"
    min_points: int = 20
    restart: str = "last"           # restart within one file: "last", "each" or "all"
    make_plots: bool = True
    write_files: bool = True
    dpi: int = 200


def rho_factor_from(s: Settings):
    """Factor k with rho [Ohm m] = R [Ohm] * k. None if dimensions are missing."""
    t = s.t_um * 1e-6
    if t <= 0:
        return None
    if s.geometry == "xx":
        if s.L_um > 0 and s.w_um > 0:
            return (s.w_um * 1e-6) * t / (s.L_um * 1e-6)
        return None
    return t


@dataclass
class FileResult:
    path: Path
    unit: str
    value_cols: list
    loop: pd.DataFrame        # symmetrised loop of the output quantities (column "branch")
    vloop: pd.DataFrame       # the same, always in volt (basis for all further analysis)
    down_raw: pd.DataFrame
    up_raw: pd.DataFrame
    csv_path: Path | None
    current_A: float | None
    T_K: float | None
    f_Hz: float | None
    angle: float | None
    field_step: float
    geometry: str
    rho_factor: float | None
    hall: dict = field(default_factory=dict)      # signal -> normal Hall fit
    heating: pd.DataFrame | None = None
    meta: dict = field(default_factory=dict)
    j_Am2: float | None = None

    @property
    def tag(self) -> str:
        return tag_of(self.T_K, self.current_A, self.f_Hz, self.angle)

    @property
    def label(self) -> str:
        T = f"{self.T_K:.2f} K" if self.T_K is not None else "T ?"
        I = f"{self.current_A * 1e6:.3g} µA" if self.current_A else "I ?"
        return f"{self.path.stem}   ({T}, {I})"


@dataclass
class RTCurve:
    path: Path
    T: np.ndarray
    R: np.ndarray             # Ohm, sorted by T
    source: str
    B_oe: float | None = None
    angle: float | None = None
    current_A: float | None = None
    rho_factor: float | None = None
    f_Hz: float | None = None
    table: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return str(self.path) + ("|table" if self.table else "")

    @property
    def label(self) -> str:
        return self.path.stem + (" [table]" if self.table else "")


# ============================================================
# Reading
# ============================================================
def parse_header_line(line: str) -> dict:
    """Instrument header line, e.g.
    1.000,_Harm_,317.000,_Hz_,0.30000000,_V_,j= ,4102844.639,_Orders=_,,0.000,°,A=,8.0,I=,0.00003282,R=,9140.000
    """
    info: dict = {"raw": line.strip()}
    for key, val in re.findall(rf"([A-Za-z_]+)=\s*_?\s*,\s*({_NUM})", line):
        info[key.strip("_")] = float(val)
    m = re.search(rf"({_NUM})\s*,\s*_Hz_", line)
    if m:
        info["f_Hz"] = float(m.group(1))
    m = re.search(rf"({_NUM})\s*,\s*_V_", line)
    if m:
        info["U_V"] = float(m.group(1))
    return info


def read_instrument_header(path: Path) -> dict:
    with open(path, "r", encoding="latin1") as f:
        return parse_header_line(f.readline())


_TEXT_RE = re.compile(r"[A-DF-Za-df-z_=°]")


def is_text_line(line: str) -> bool:
    """Header or column-name line (contains letters other than e/E of numbers and 'nan')."""
    return bool(_TEXT_RE.search(re.sub(r"(?i)\bnan\b", "", line)))


def parse_data_lines(lines: list) -> pd.DataFrame:
    sample = lines[0]
    sep = "\t" if "\t" in sample else (";" if ";" in sample else ",")
    rows = [ln.strip().split(sep) for ln in lines]
    ncol = Counter(len(r) for r in rows).most_common(1)[0][0]
    if ncol < 3:
        raise ValueError(f"Only {ncol} column(s) found – wrong separator '{sep}'?")
    n = min(ncol, len(ALL_COLUMNS))
    rows = [(r + [""] * n)[:n] for r in rows]
    df = pd.DataFrame(rows, columns=ALL_COLUMNS[:n]).apply(pd.to_numeric, errors="coerce")
    return df.dropna(subset=[TIME_COL, FIELD_COL]).reset_index(drop=True)


def read_segments(path: Path) -> list:
    """Read a measurement file and split it into sections.

    A new section starts when the measurement was restarted in the same file:
    a header/column-name line appears again inside the data, or the time stamp
    jumps back. Large time gaps are only reported. Row order is kept (no sorting
    by time, otherwise two measurements would mix).
    Returns a list of dicts with header, df, start (line number), reason, gap_note.
    """
    with open(path, "r", encoding="latin1") as f:
        lines = f.readlines()

    blocks = []
    cur = {"hline": None, "lines": [], "start": None, "reason": "start of file"}
    for no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        if is_text_line(line):
            if cur["lines"]:
                blocks.append(cur)
                cur = {"hline": None, "lines": [], "start": None, "reason": "header line repeated"}
            if "_Harm_" in line or "I=" in line:
                cur["hline"] = line
            continue
        if cur["start"] is None:
            cur["start"] = no
        cur["lines"].append(line)
    if cur["lines"]:
        blocks.append(cur)
    if not blocks:
        raise ValueError("No measurement data found.")

    segs, hline = [], next((b["hline"] for b in blocks if b["hline"]), "")
    for b in blocks:
        hline = b["hline"] or hline
        df = parse_data_lines(b["lines"])
        if df.empty:
            continue
        t = df[TIME_COL].to_numpy(float)
        dt = np.diff(t)
        cuts = list(np.flatnonzero(dt < 0) + 1)
        pos = dt[dt > 0]
        gap = max(600.0, 100.0 * float(np.median(pos))) if pos.size else np.inf
        bounds = [0] + cuts + [len(df)]
        for k in range(len(bounds) - 1):
            part = df.iloc[bounds[k]:bounds[k + 1]].reset_index(drop=True)
            if part.empty:
                continue
            pdt = np.diff(part[TIME_COL].to_numpy(float))
            big = pdt[pdt > gap]
            segs.append({
                "header": parse_header_line(hline), "df": part,
                "start": b["start"] + bounds[k] if k else b["start"],
                "reason": b["reason"] if k == 0 else "time stamp jumps back",
                "gap_note": f"{big.size} time gap(s) up to {big.max():.0f} s" if big.size else None,
            })
    return segs


def read_numeric_table(path: Path) -> np.ndarray:
    """Read any numeric table (separator , ; tab or space). Lines with text are skipped."""
    rows = []
    with open(path, "r", encoding="latin1") as f:
        for line in f:
            parts = [p for p in re.split(r"[,;\t ]+", line.strip()) if p]
            if len(parts) < 2:
                continue
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue
    if not rows:
        raise ValueError("No numeric data found in the file.")
    n = Counter(len(r) for r in rows).most_common(1)[0][0]
    return np.array([r for r in rows if len(r) == n], dtype=float)


def column_mean(df: pd.DataFrame, col: str):
    if col in df.columns:
        v = float(df[col].mean())
        if np.isfinite(v):
            return v
    return None


def classify(df: pd.DataFrame, field_step: float) -> str:
    """'sweep' (field varies) or 'rt' (field fixed, temperature varies)."""
    B = df[FIELD_COL].to_numpy(float)
    b_span = float(np.nanpercentile(B, 99) - np.nanpercentile(B, 1))
    t_span = 0.0
    if TEMP_COL in df.columns and df[TEMP_COL].notna().sum() > 10:
        T = df[TEMP_COL].to_numpy(float)
        t_span = float(np.nanpercentile(T, 99) - np.nanpercentile(T, 1))
    b_lim = max(4 * field_step, 200.0)
    if b_span > b_lim:
        return "sweep"
    if t_span >= 0.5:
        return "rt"
    raise ValueError(f"Neither field sweep nor R(T) curve (field range {b_span:.0f} Oe, "
                     f"temperature range {t_span:.2f} K).")


# ============================================================
# Current and resistivity
# ============================================================
def determine_current(df: pd.DataFrame, header: dict, source: str):
    I_hdr = header.get("I")
    I_meas = None
    if all(c in df.columns for c in CURRENT_COLS):
        I_meas = float(np.nanmean(np.hypot(df[CURRENT_COLS[0]], df[CURRENT_COLS[1]])))
        if not np.isfinite(I_meas) or I_meas == 0:
            I_meas = None

    if source == "header":
        if not I_hdr:
            raise ValueError("No current 'I=' in the header.")
        return I_hdr, I_hdr, I_meas
    if I_meas is None:
        raise ValueError("No valid current columns (Re I / Im I).")
    return I_meas, I_hdr, I_meas


def to_resistivity(df: pd.DataFrame, current_A: float, factor: float):
    df = df.copy()
    cols = []
    for c in VOLTAGE_COLS:
        if c not in df.columns:
            continue
        base = c.replace("[V]", "").strip()
        name = f"{base} [µOhm cm]"
        df[name] = df[c] / current_A * factor * RHO_OHM_M_TO_UOHM_CM
        cols.append(name)
    return df, cols


# ============================================================
# Sweep separation
# ============================================================
def _plateau_end(mask: np.ndarray, i: int) -> int:
    """Last index of the contiguous True region that starts at i."""
    rest = np.flatnonzero(~mask[i:])
    return i + int(rest[0]) - 1 if rest.size else len(mask) - 1


def split_loop(df: pd.DataFrame, field_step: float, min_points: int):
    """Split a field loop into down and up sweep.

    Works for +Bmax -> -Bmax -> +Bmax and -Bmax -> +Bmax -> -Bmax, also with a
    virgin curve at the start (0 -> Bmax) or extra data at the end. Plateaus at the
    turning points (stepwise) belong to both sweeps.
    """
    B = df[FIELD_COL].to_numpy(float)
    n = len(B)
    tol = max(0.5 * abs(field_step), 1e-9)
    if B.max() - B.min() < 4 * tol:
        raise ValueError("Field range too small for a loop (or field step too large).")

    hi = np.flatnonzero(B >= B.max() - tol)
    lo = np.flatnonzero(B <= B.min() + tol)
    starts_high = hi[0] <= lo[0]
    x = B if starts_high else -B          # x always runs: top -> bottom -> top

    top = x.max() - tol
    bottom = x.min() + tol
    m_top, m_bottom = x >= top, x <= bottom

    i_start = int(np.flatnonzero(m_top)[0])
    lo_after = np.flatnonzero(m_bottom[i_start:])
    if lo_after.size == 0:
        raise ValueError("No turning point found.")
    i_lo0 = i_start + int(lo_after[0])
    i_lo1 = _plateau_end(m_bottom, i_lo0)

    hi_after = np.flatnonzero(m_top[i_lo1:])
    i_end = _plateau_end(m_top, i_lo1 + int(hi_after[0])) if hi_after.size else n - 1

    first = df.iloc[i_start:i_lo1 + 1].reset_index(drop=True)
    second = df.iloc[i_lo0:i_end + 1].reset_index(drop=True)

    notes = []
    if i_start > min_points:
        notes.append(f"{i_start} points before the loop (virgin curve) ignored.")
    if n - 1 - i_end > min_points:
        notes.append(f"{n - 1 - i_end} points after the loop ignored (only the first loop is evaluated).")
    if hi_after.size == 0:
        notes.append("Second sweep does not fully reach the start field.")

    if len(first) < min_points or len(second) < min_points:
        raise ValueError(f"Sweeps too short ({len(first)} / {len(second)} points, minimum {min_points}).")

    down, up = (first, second) if starts_high else (second, first)
    return down, up, notes


# ============================================================
# Field grid, resampling, symmetrisation
# ============================================================
def symmetric_grid(down: pd.DataFrame, up: pd.DataFrame, step: float) -> np.ndarray:
    lo = max(down[FIELD_COL].min(), up[FIELD_COL].min())
    hi = min(down[FIELD_COL].max(), up[FIELD_COL].max())
    bm = min(-lo, hi)
    if bm <= 0:
        raise ValueError("The common field range does not contain B = 0 – cannot symmetrise.")
    # +0.1 step tolerance so that e.g. 89999.7 Oe still counts as 90000 Oe
    n = int(np.floor(bm / step + 0.1))
    if n < 1:
        raise ValueError("Field grid too small – reduce the field step.")
    return np.arange(-n, n + 1) * step   # exactly symmetric by construction


def resample(sweep: pd.DataFrame, grid: np.ndarray, step: float, cols: list, method: str) -> pd.DataFrame:
    data = sweep[[FIELD_COL] + cols]
    if method == "bin":
        keys = np.rint(data[FIELD_COL].to_numpy() / step).astype(np.int64)
    else:
        keys = data[FIELD_COL].round(6).to_numpy()
    g = data.groupby(keys).mean().sort_values(FIELD_COL)

    x = g[FIELD_COL].to_numpy()
    out = {FIELD_COL: grid}
    for c in cols:
        y = g[c].to_numpy()
        ok = np.isfinite(y) & np.isfinite(x)
        out[c] = np.interp(grid, x[ok], y[ok]) if ok.sum() >= 2 else np.full(grid.shape, np.nan)
    return pd.DataFrame(out)


def symmetrize(down_rs: pd.DataFrame, up_rs: pd.DataFrame, cols: list) -> pd.DataFrame:
    B = down_rs[FIELD_COL].to_numpy()
    if not np.array_equal(B, up_rs[FIELD_COL].to_numpy()):
        raise ValueError("down and up sweep are not on the same grid.")
    if not np.allclose(B, -B[::-1], atol=1e-9 * max(1.0, np.abs(B).max())):
        raise ValueError("Field grid is not symmetric around 0.")

    down_part = {FIELD_COL: B[::-1], BRANCH: "down"}   # +Bmax -> -Bmax
    up_part = {FIELD_COL: B, BRANCH: "up"}             # -Bmax -> +Bmax
    for c in cols:
        d = down_rs[c].to_numpy()          # S_down(B)
        u = up_rs[c].to_numpy()            # S_up(B)
        u_mirror = u[::-1]                 # S_up(-B)
        sym = 0.5 * (d + u_mirror)
        anti = 0.5 * (d - u_mirror)

        down_part[f"{c}_raw"] = d[::-1]
        down_part[f"{c}_sym"] = sym[::-1]
        down_part[f"{c}_antisym"] = anti[::-1]
        up_part[f"{c}_raw"] = u
        up_part[f"{c}_sym"] = sym[::-1]        #  sym_up(B) = sym_down(-B)
        up_part[f"{c}_antisym"] = -anti[::-1]  # anti_up(B) = -anti_down(-B)

    return pd.concat([pd.DataFrame(down_part), pd.DataFrame(up_part)], ignore_index=True)


def make_loop(down, up, grid, s: Settings, cols: list) -> pd.DataFrame:
    return symmetrize(resample(down, grid, s.field_step, cols, s.method),
                      resample(up, grid, s.field_step, cols, s.method), cols)


# ============================================================
# R(T) curves
# ============================================================
def make_rt(path, T, R, source, **kw) -> RTCurve:
    T = np.asarray(T, float)
    R = np.asarray(R, float)
    ok = np.isfinite(T) & np.isfinite(R)
    T, R = T[ok], R[ok]
    order = np.argsort(T, kind="stable")
    T, R = T[order], R[order]
    if len(T) < 5 or np.ptp(T) <= 0:
        raise ValueError("Too few valid points in the R(T) curve.")
    return RTCurve(path=Path(path), T=T, R=R, source=source, **kw)


def load_rt_table(path: Path, col_T: int, col_R: int, rho_factor=None) -> RTCurve:
    path = Path(path)
    tab = read_numeric_table(path)
    if tab.shape[1] < max(col_T, col_R):
        raise ValueError(f"R(T) table has only {tab.shape[1]} columns.")
    src = f"table, T = column {col_T}, R = column {col_R}"
    rt = make_rt(path, tab[:, col_T - 1], tab[:, col_R - 1], src, rho_factor=rho_factor, table=True)
    rt.meta = {"Measurement": path.name, "Type": "R(T) table (imported)", "Source": src,
               "Evaluated": now(), "Temperature [K]": f"{rt.T.min():.3f} … {rt.T.max():.3f}"}
    return rt


def smooth_derivative(T: np.ndarray, y: np.ndarray, window: float, n: int = 400):
    """Smoothed curve and derivative from local polynomial fits (±window) on n grid points."""
    grid = np.linspace(T[0], T[-1], n)
    val = np.full(n, np.nan)
    der = np.full(n, np.nan)
    lo = np.searchsorted(T, grid - window, "left")
    hi = np.searchsorted(T, grid + window, "right")
    for k in range(n):
        x = T[lo[k]:hi[k]] - grid[k]
        if x.size >= 5 and np.ptp(x) > 0:
            deg = 2 if x.size >= 7 else 1
            p = np.polyfit(x, y[lo[k]:hi[k]], deg)
            val[k], der[k] = p[-1], p[-2]
    return grid, val, der


def rt_table(rt: RTCurve, quantity: str, window: float) -> dict:
    if quantity == "rho":
        if not rt.rho_factor:
            raise ValueError("no sample dimensions – rho cannot be calculated")
        y = rt.R * rt.rho_factor * RHO_OHM_M_TO_UOHM_CM
        unit = "µΩ cm"
    else:
        y = rt.R
        unit = "Ω"
    grid, val, der = smooth_derivative(rt.T, y, window)
    return {"T": rt.T, "y": y, "grid": grid, "val": val, "der": der, "unit": unit}


def rt_local_fit(rt: RTCurve, T0: float, window: float):
    """Local polynomial fit around T0 -> R(T0), dR/dT(T0) and the points used."""
    T, R = rt.T, rt.R
    if T0 < T.min() - window or T0 > T.max() + window:
        raise ValueError(f"T = {T0:.2f} K is outside the R(T) curve ({T.min():.2f} … {T.max():.2f} K).")
    mask = np.abs(T - T0) <= window
    if mask.sum() < 7:   # too few points -> take the nearest 9
        mask = np.zeros(len(T), bool)
        mask[np.argsort(np.abs(T - T0))[:min(9, len(T))]] = True
    x = T[mask] - T0
    if np.ptp(x) <= 0:
        raise ValueError("R(T) curve has no temperature variation in the fit window.")
    deg = 2 if mask.sum() >= 7 else 1
    p = np.polyfit(x, R[mask], deg)
    return float(p[-1]), float(p[-2]), mask


def export_rt(items: list, window: float, marks: list, dpi: int = 200) -> Path:
    """Write smoothed R(T)/dR/dT CSVs for each curve and one figure. Returns the figure path."""
    for rt, d in items:
        sym = "rho" if d["unit"] == "µΩ cm" else "R"
        u = "µOhm cm" if sym == "rho" else "Ohm"
        meta = {**rt.meta, "Content": f"{sym}(T) smoothed and d{sym}/dT (local polynomial fit ±{window:g} K)"}
        write_origin(out_dir(rt.path) / f"{rt.path.stem}_RT_{sym}_smoothed.csv", meta,
                     [("T", "K", "", d["grid"]), (sym, u, rt.label, d["val"]),
                      (f"d{sym}/dT", f"{u}/K", rt.label, d["der"])])
    first = items[0][0].path
    unit = items[0][1]["unit"]
    if len(items) == 1:
        path = img_dir(first) / f"{first.stem}_RT_{'rho' if unit == 'µΩ cm' else 'R'}.png"
    else:
        path = img_dir(first, "comparison") / f"RT_comparison_{stamp()}.png"
    save_figure(path, dpi, plot_rt, items, marks, figsize=(9, 7))
    return path


# ============================================================
# Heating from the 3rd harmonic
# ============================================================
def r1f_at_zero(res: FileResult):
    v = res.vloop
    key = f"{RE1}_sym"
    if key not in v.columns or not res.current_A:
        return None
    B = v[FIELD_COL].to_numpy()
    return float(v[key].to_numpy()[np.argmin(np.abs(B))] / res.current_A)


def pick_rt(res: FileResult, rts: list) -> RTCurve:
    """Matching R(T) curve: covers T, same resistance (= same contacts), same angle,
    smallest field."""
    if res.T_K is None:
        raise ValueError("This measurement has no temperature column.")
    cands = [rt for rt in rts if rt.T.min() - 0.5 <= res.T_K <= rt.T.max() + 0.5]
    if not cands:
        raise ValueError(f"No loaded R(T) curve covers T = {res.T_K:.2f} K.")
    R1 = r1f_at_zero(res)

    def score(rt):
        dev = 0.0
        if R1:
            dev = abs(np.interp(res.T_K, rt.T, rt.R) - R1) / abs(R1)
        da = abs(rt.angle - res.angle) if rt.angle is not None and res.angle is not None else 0.0
        db = abs(rt.B_oe) if rt.B_oe is not None else 0.0
        return (dev > 0.2, round(da), db, dev)

    return min(cands, key=score)


def heating_analysis(res: FileResult, rt: RTCurve, window: float):
    if res.current_A is None:
        raise ValueError("No current known for this measurement (header / current columns).")
    if res.T_K is None:
        raise ValueError("This measurement has no temperature column.")
    v = res.vloop
    for c in (RE3, IM3):
        if f"{c}_sym" not in v.columns:
            raise ValueError("The measurement contains no 3rd harmonic.")

    R0, dRdT, mask = rt_local_fit(rt, res.T_K, window)
    if not np.isfinite(dRdT) or dRdT == 0:
        raise ValueError("dR/dT ≈ 0 – heating cannot be determined from the 3f signal.")

    I = res.current_A
    re_s = v[f"{RE3}_sym"].to_numpy()
    im_s = v[f"{IM3}_sym"].to_numpy()
    V3 = np.hypot(re_s, im_s)
    out = pd.DataFrame({
        FIELD_COL: v[FIELD_COL].to_numpy(),
        BRANCH: v[BRANCH].to_numpy(),
        "|V3f| sym [V]": V3,
        "dT_2w [K]": 2.0 * V3 / (I * abs(dRdT)),
        "Phase 3f [deg]": np.degrees(np.arctan2(im_s, re_s)),
    })
    fit = {"T0": res.T_K, "R0": R0, "dRdT": dRdT, "I": I, "mask": mask, "R_1f": r1f_at_zero(res)}
    return out, fit


def run_heating(targets: list, rts: list, window: float, fixed_rt: RTCurve | None = None,
                dpi: int = 200, log=print, write_summary=False):
    """Heating analysis for several sweeps, with CSV/figure output. Returns (summary rows,
    last (res, rt, out, fit) or None)."""
    summary, last = [], None
    for res in targets:
        try:
            rt = fixed_rt or pick_rt(res, rts)
            out, fit = heating_analysis(res, rt, window)
        except ValueError as e:
            log(f"Heating {res.path.name}: ✗ {e}")
            continue
        res.heating = out
        dT = out["dT_2w [K]"].to_numpy()
        dT0 = float(dT[np.argmin(np.abs(out[FIELD_COL].to_numpy()))])
        meta = {**res.meta,
                "Content": "Heating from the 3rd harmonic: dT_2w = 2 |V3f| / (I |dR/dT|)",
                "R(T) curve": f"{rt.label} ({rt.source})",
                "R(T0) [Ohm]": f"{fit['R0']:.6g}",
                "dR/dT [Ohm/K]": f"{fit['dRdT']:.6g} (fit window ±{window:g} K)",
                "dT_2w(B=0) [K]": f"{dT0:.4g}"}
        write_origin(out_dir(res.path) / f"{res.path.stem}_heating_3f.csv", meta,
                     field_columns(out[FIELD_COL]) + df_columns(out, res.tag))
        save_figure(img_dir(res.path) / f"{res.path.stem}_heating_3f.png", dpi, plot_heating,
                    res, rt, out, fit, window, figsize=(11, 6))
        summary.append({"File": res.path.name, "T [K]": res.T_K, "I [A]": res.current_A,
                        "j [A/m^2]": res.j_Am2, "f [Hz]": res.f_Hz, "Angle [deg]": res.angle,
                        "R(T) curve": rt.label, "dR/dT [Ohm/K]": fit["dRdT"],
                        "dT_2w(B=0) [K]": dT0, "dT_2w max [K]": float(np.nanmax(dT))})
        last = (res, rt, out, fit)
        log(f"Heating {res.path.name}: R(T) = {rt.label}, dR/dT = {fit['dRdT']:.4g} Ω/K, "
            f"ΔT₂ω(0) = {dT0 * 1e3:.3g} mK, max = {np.nanmax(dT) * 1e3:.3g} mK "
            f"({np.nanmax(dT) / fit['T0']:.2%} of T)")
        if fit["R_1f"]:
            dev = abs(fit["R_1f"] - fit["R0"]) / abs(fit["R0"])
            if dev > 0.2:
                log(f"  WARNING: R from 1f ({fit['R_1f']:.4g} Ω) deviates by {dev:.0%} from R(T0) = "
                    f"{fit['R0']:.4g} Ω – does the R(T) curve belong to these contacts?")
    if write_summary and summary:
        name = f"Heating_summary_{stamp()}.csv"
        write_origin(out_dir(targets[0].path) / name,
                     {"Content": "Summary heating from the 3rd harmonic: dT_2w = 2 |V3f| / (I |dR/dT|)",
                      "Evaluated": now(), "Fit window dR/dT [K]": f"±{window:g}"},
                     df_columns(pd.DataFrame(summary)))
        log(f"Heating: {len(summary)}/{len(targets)} measurements, summary: output/{name}")
    return summary, last


# ============================================================
# Hall: subtract the normal Hall effect
# ============================================================
def fit_normal_hall(res: FileResult, col: str, bmin_T: float, model: str) -> dict:
    v = res.vloop
    key = f"{col}_antisym"
    if key not in v.columns:
        raise ValueError("Signal not present.")
    B = v[FIELD_COL].to_numpy(float)
    y = v[key].to_numpy(float)
    m = (np.abs(B) >= bmin_T * OE_PER_T) & np.isfinite(y)
    if model == "sat":
        if (m & (B > 0)).sum() < 2 or (m & (B < 0)).sum() < 2:
            raise ValueError(f"Too few points with |B| ≥ {bmin_T:g} T for the fit.")
        X = np.column_stack([B[m], np.sign(B[m])])
    else:
        if m.sum() < 3:
            raise ValueError(f"Too few points with |B| ≥ {bmin_T:g} T for the fit.")
        X = B[m][:, None]
    coef, *_ = np.linalg.lstsq(X, y[m], rcond=None)
    resid = y[m] - X @ coef
    return {"slope_V_per_Oe": float(coef[0]), "A_V": float(coef[1]) if model == "sat" else 0.0,
            "bmin_T": bmin_T, "model": model, "n_fit": int(m.sum()), "rms_V": float(np.sqrt(np.mean(resid ** 2)))}


def hall_numbers(res: FileResult, fit: dict, t_m) -> dict:
    out = {"dVxy/dB [V/T]": fit["slope_V_per_Oe"] * OE_PER_T}
    if res.current_A:
        Rs = out["dVxy/dB [V/T]"] / res.current_A
        out["dRxy/dB [Ohm/T]"] = Rs
        if t_m:
            RH = Rs * t_m                      # Ohm m / T = m^3 / C
            out["R_H [m^3/C]"] = RH
            out["R_H [cm^3/C]"] = RH * 1e6
            if RH != 0:
                out["n [cm^-3]"] = 1.0 / (E_CHARGE * abs(RH)) * 1e-6
                out["Type"] = "holes (R_H > 0)" if RH > 0 else "electrons (R_H < 0)"
    return out


def hall_thickness(res: FileResult, t_um: float | None):
    """Thickness (m) for R_H: from the Hall geometry if set, else the given t."""
    if res.geometry == "xy" and res.rho_factor:
        return res.rho_factor
    return t_um * 1e-6 if t_um else None


def run_hall(sel: list, col: str, bmin: float, model: str, quantity: str, t_um=None,
             dpi: int = 200, log=print):
    """Fit and subtract the normal Hall effect for several sweeps; writes CSVs and a
    figure. Returns (items for plot_hall, unit)."""
    items, unit = [], None
    for res in sel:
        if res.geometry != "xy":
            log(f"  Note: {res.path.name} was evaluated with longitudinal geometry "
                f"(evaluate as Hall for rho_xy).")
        try:
            fit = fit_normal_hall(res, col, bmin, model)
        except ValueError as e:
            log(f"Hall {res.path.name}: ✗ {e}")
            continue
        res.hall[col] = fit
        for q in (quantity, "R", "V"):
            try:
                sc, u = unit_scale(res, q)
                break
            except ValueError:
                continue
        if q != quantity:
            log(f"  WARNING: {res.path.name}: {QUANTITIES[quantity]} not possible, showing {QUANTITIES[q]}")
        v = res.vloop
        B = v[FIELD_COL].to_numpy(float)
        anti = v[f"{col}_antisym"].to_numpy(float)
        c = {"B": B, BRANCH: v[BRANCH].to_numpy(), "anti": anti * sc,
             "fit": (fit["slope_V_per_Oe"] * B + fit["A_V"] * np.sign(B)) * sc,
             "corr": (anti - fit["slope_V_per_Oe"] * B) * sc}

        nums = hall_numbers(res, fit, hall_thickness(res, t_um))
        name = short_name(col)
        meta = {**res.meta,
                "Content": f"Hall: {name} antisym; normal Hall effect R0·B subtracted",
                "Fit model": f"{HALL_MODELS[model]} for |B| ≥ {bmin:g} T ({fit['n_fit']} points)",
                f"R0 slope [{u}/T]": f"{fit['slope_V_per_Oe'] * OE_PER_T * sc:.6g}",
                f"A saturation anomalous [{u}]": f"{fit['A_V'] * sc:.6g}" if model == "sat" else None,
                "R_H [cm^3/C]": fmt(nums.get("R_H [cm^3/C]"), ".6g"),
                "n [cm^-3]": fmt(nums.get("n [cm^-3]"), ".4e"),
                "Carriers": nums.get("Type")}
        unit_o = u.replace("Ω", "Ohm")
        write_origin(out_dir(res.path) / f"{res.path.stem}_hall_{safe_name(name)}.csv", meta,
                     field_columns(B) + [("Field", "Oe", "", B), ("Branch", "", "", c[BRANCH]),
                                         (f"{name} antisym", unit_o, res.tag, c["anti"]),
                                         ("Fit R0·B + A·sgn(B)", unit_o, res.tag, c["fit"]),
                                         (f"{name} antisym − R0·B", unit_o, res.tag, c["corr"])])
        line = (f"Hall {res.path.name} ({name}, {HALL_MODELS[model]}, |B| ≥ {bmin:g} T, {fit['n_fit']} points): "
                f"R₀ slope = {fit['slope_V_per_Oe'] * OE_PER_T * sc:.4g} {u}/T")
        if model == "sat":
            line += f", A (saturation, anomalous) = {fit['A_V'] * sc:.4g} {u}"
        log(line)
        if "R_H [cm^3/C]" in nums:
            log(f"  R_H = {nums['R_H [cm^3/C]']:.4g} cm³/C, n = {nums.get('n [cm^-3]', float('nan')):.3e} cm⁻³ "
                f"→ {nums.get('Type', '')} (sign depends on the contact orientation)")
        elif "dRxy/dB [Ohm/T]" in nums:
            log("  enter the thickness t for R_H and n")
        if unit is None:
            unit = u
        if u != unit:
            log(f"  {res.path.name} not in the plot (different unit)")
            continue
        items.append((res, c))
    if items:
        first = items[0][0].path
        if len(items) == 1:
            path = img_dir(first) / f"{first.stem}_hall_{safe_name(short_name(col))}.png"
        else:
            path = img_dir(first, "comparison") / f"Hall_comparison_{stamp()}.png"
        save_figure(path, dpi, plot_hall, items, unit, bmin, col)
    return items, unit


# ============================================================
# Quantities, normalisation, comparison
# ============================================================
def unit_scale(res: FileResult, quantity: str):
    """Factor from volt to the requested quantity."""
    if quantity == "V":
        return 1.0, "V"
    if not res.current_A:
        raise ValueError("no current known")
    if quantity == "R":
        return 1.0 / res.current_A, "Ω"
    if not res.rho_factor:
        raise ValueError("no sample dimensions – rho cannot be calculated")
    return res.rho_factor * RHO_OHM_M_TO_UOHM_CM / res.current_A, "µΩ cm"


def get_curve(res: FileResult, col: str, comp: str, quantity: str):
    v = res.vloop
    B = v[FIELD_COL].to_numpy(float)
    br = v[BRANCH].to_numpy()
    if quantity == "dT":
        if res.heating is None:
            raise ValueError("no heating analysis yet (Heating tab)")
        return B, br, res.heating["dT_2w [K]"].to_numpy(float), "K"
    if comp == "hall":
        h = res.hall.get(col)
        if not h:
            raise ValueError("no normal Hall fit for this signal (Hall tab)")
        y = v[f"{col}_antisym"].to_numpy(float) - h["slope_V_per_Oe"] * B
    else:
        key = f"{col}_{comp}"
        if key not in v.columns:
            raise ValueError("signal not present")
        y = v[key].to_numpy(float)
    sc, unit = unit_scale(res, quantity)
    return B, br, y * sc, unit


def apply_norm(B, y, unit, norm: str, n: float, I):
    if norm == "none":
        return y, unit
    if norm == "In":
        if not I:
            raise ValueError("no current known")
        return y / I ** n, f"{unit}/A^{n:g}"
    y0 = float(np.nanmean(y[np.abs(B) == np.abs(B).min()]))
    if not np.isfinite(y0) or y0 == 0:
        raise ValueError("S(B = 0) is 0 – cannot normalise")
    if norm == "mr":
        return 100.0 * (y - y0) / y0, "%"
    return y / y0, "rel."


def param_value(res: FileResult, key: str):
    if key == "T":
        return res.T_K
    if key == "I":
        return res.current_A * 1e6 if res.current_A else None
    if key == "f":
        return res.f_Hz
    if key == "angle":
        return res.angle
    return None


def build_comparison(results: list, col: str, comp: str, quantity: str, norm: str, n: float, param: str):
    curves, skipped, units = [], [], []
    for k, res in enumerate(results):
        try:
            B, br, y, unit = get_curve(res, col, comp, quantity)
            y, unit = apply_norm(B, y, unit, norm, n, res.current_A)
        except ValueError as e:
            skipped.append(f"{res.path.name}: {e}")
            continue
        p = k + 1 if param == "idx" else param_value(res, param)
        curves.append({"res": res, "B": B, BRANCH: br, "y": y, "p": p})
        units.append(unit)
    unit = " / ".join(dict.fromkeys(units))
    return curves, unit, skipped


def dependence(curves: list, b0_oe) -> pd.DataFrame:
    rows = []
    for c in curves:
        if c["p"] is None:
            continue
        res, B, y = c["res"], c["B"], c["y"]
        if b0_oe is None:
            if not np.isfinite(y).any():
                continue
            i = int(np.nanargmax(np.abs(y)))
            val, Bv = float(y[i]), float(B[i])
        else:
            vals = []
            for br in ("down", "up"):
                m = c[BRANCH] == br
                if not m.any():
                    continue
                j = int(np.argmin(np.abs(B[m] - b0_oe)))
                if abs(B[m][j] - b0_oe) <= 0.51 * res.field_step:
                    vals.append(y[m][j])
            if not vals:
                continue
            val, Bv = float(np.nanmean(vals)), float(b0_oe)
        rows.append({"File": res.path.name, "x": float(c["p"]), "y": val, "B [Oe]": Bv})
    return pd.DataFrame(rows, columns=["File", "x", "y", "B [Oe]"])


def power_fit(x: np.ndarray, y: np.ndarray):
    good = (x > 0) & np.isfinite(y) & (y != 0)
    if good.sum() < 2 or np.unique(x[good]).size < 2:
        return None
    lx, ly = np.log(x[good]), np.log(np.abs(y[good]))
    err = np.nan
    if good.sum() >= 4:
        p, cov = np.polyfit(lx, ly, 1, cov=True)
        err = float(np.sqrt(cov[0, 0]))
    else:
        p = np.polyfit(lx, ly, 1)
    return {"n": float(p[0]), "ln_a": float(p[1]), "err": err}


def same_series(a: FileResult, b: FileResult, vary: str) -> bool:
    """True if a and b differ at most in the parameter 'vary'."""
    if a.geometry != b.geometry:
        return False
    checks = {
        "T": lambda x, y: abs(x - y) <= max(0.15, 0.01 * abs(y)),
        "I": lambda x, y: abs(x - y) <= 0.02 * abs(y),
        "f": lambda x, y: abs(x - y) <= max(0.5, 0.01 * abs(y)),
        "angle": lambda x, y: abs(x - y) <= 1.0,
    }
    for k, close in checks.items():
        if k == vary:
            continue
        x, y = param_value(a, k), param_value(b, k)
        if (x is None) != (y is None):
            return False
        if x is not None and not close(x, y):
            return False
    return True


def comparison_ylabel(col, comp, quantity):
    if quantity == "dT":
        return "ΔT₂ω"
    return f"{short_name(col)} {COMPONENTS[comp]}"


def export_comparison(lc: dict, dpi: int = 200) -> str:
    """Save curves, values vs. parameter and figure of a comparison. ``lc`` holds 'args'
    (as passed to plot_comparison), 'col', 'comp', 'quantity', 'norm', 'n', 'param', 'first'."""
    curves, unit, ylabel, param, _, branches, dep, fit, _, b0 = lc["args"]
    first = lc["first"].path
    what = "dT" if lc["quantity"] == "dT" else f"{short_name(lc['col'])}_{lc['comp']}"
    base = f"Comparison_{safe_name(what)}_{param}_{stamp()}"
    pname, punit = PARAMS[param]
    unit_o = unit.replace("Ω", "Ohm")
    where = f"B = {b0 / OE_PER_T:g} T" if b0 is not None else "maximum |S| in the loop"
    meta = {
        "Content": f"Comparison: {ylabel}",
        "Evaluated": now(),
        "Signal": lc["col"] if lc["quantity"] != "dT" else "dT_2w (heating)",
        "Component": COMPONENTS[lc["comp"]] if lc["quantity"] != "dT" else None,
        "Quantity": QUANTITIES[lc["quantity"]],
        "Normalisation": NORMS[lc["norm"]] + (f" (n = {lc['n']:g})" if lc["norm"] == "In" else ""),
        "Parameter": f"{pname} [{punit}]" if punit else pname,
        "Branches": branches,
        "Measurements": "; ".join(f"{c['res'].path.name} ({c['res'].tag})" for c in curves),
    }
    cols = []
    for c in curves:
        r = c["res"]
        m = np.ones(len(c["B"]), bool) if branches == "both" else (c[BRANCH] == branches)
        legend = f"{c['p']:g} {punit}".strip() if c["p"] is not None and param != "idx" else r.tag
        cols.append((f"B ({r.path.stem})", "T", legend, c["B"][m] / OE_PER_T))
        cols.append((ylabel, unit_o, legend, c["y"][m]))
    write_origin(out_dir(first) / f"{base}_curves.csv", meta, cols)
    if len(dep):
        dmeta = {**meta, "Content": f"{ylabel} at {where} vs {pname}"}
        if fit:
            dmeta["Power law |S| ∝ x^n"] = f"n = {fit['n']:.4f}" + (
                "" if np.isnan(fit["err"]) else f" ± {fit['err']:.4f}")
        write_origin(out_dir(first) / f"{base}_values.csv", dmeta,
                     [(pname, punit, "", dep["x"].to_numpy()), (ylabel, unit_o, where, dep["y"].to_numpy()),
                      ("B", "T", "", dep["B [Oe]"].to_numpy() / OE_PER_T), ("File", "", "", dep["File"].to_numpy())])
    img = img_dir(first, "comparison") / f"{base}.png"
    save_figure(img, dpi, plot_comparison, *lc["args"], figsize=(13, 6))
    return base


# ============================================================
# Plots (Figure API only, no pyplot)
# ============================================================
def plot_sym_antisym(fig: Figure, loop: pd.DataFrame, col: str, unit: str, title: str):
    fig.clear()
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    for branch, ls, color in (("down", "-", "C0"), ("up", "--", "C1")):
        part = loop[loop[BRANCH] == branch]
        B = part[FIELD_COL].to_numpy() / OE_PER_T
        ax1.plot(B, part[f"{col}_sym"], ls, color=color, lw=1.6, label=f"sym ({branch})")
        ax2.plot(B, part[f"{col}_antisym"], ls, color=color, lw=1.6, label=f"antisym ({branch})")
    name = short_name(col)
    ax1.set_ylabel(f"{name} sym [{unit}]")
    ax2.set_ylabel(f"{name} antisym [{unit}]")
    ax2.set_xlabel("Magnetic field [T]")
    ax2.axhline(0, color="0.7", lw=0.8, zorder=0)
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=10)


def plot_raw(fig: Figure, res: FileResult, col: str):
    fig.clear()
    ax = fig.add_subplot()
    for raw, branch, color in ((res.down_raw, "down", "C0"), (res.up_raw, "up", "C1")):
        ax.plot(raw[FIELD_COL] / OE_PER_T, raw[col], ".", ms=2.5, alpha=0.35, color=color,
                label=f"{branch}: measured")
        part = res.loop[res.loop[BRANCH] == branch]
        ax.plot(part[FIELD_COL] / OE_PER_T, part[f"{col}_raw"], "-", lw=1.4, color=color,
                label=f"{branch}: on grid")
    ax.set_xlabel("Magnetic field [T]")
    ax.set_ylabel(f"{short_name(col)} [{res.unit}]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"{res.path.stem} – raw data", fontsize=10)


def plot_rt(fig: Figure, items: list, marks: list):
    fig.clear()
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    unit = items[0][1]["unit"] if items else "Ω"
    sym = "ρ" if unit == "µΩ cm" else "R"
    for k, (rt, d) in enumerate(items):
        color = f"C{k % 10}"
        ax1.plot(d["T"], d["y"], ".", ms=1.5, alpha=0.3, color=color)
        ax1.plot(d["grid"], d["val"], "-", lw=1.3, color=color, label=rt.label)
        ax2.plot(d["grid"], d["der"], "-", lw=1.3, color=color, label=rt.label)
    for T in marks:
        for ax in (ax1, ax2):
            ax.axvline(T, color="C3", ls="--", lw=0.8, alpha=0.5, zorder=0)
    ax1.set_ylabel(f"{sym} [{unit}]")
    ax2.set_ylabel(f"d{sym}/dT [{unit}/K]")
    ax2.set_xlabel("T [K]")
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
    if items and len(items) <= 10:
        ax1.legend(fontsize=8)
    if marks:
        ax1.set_title("red dashed: temperatures of the loaded field sweeps", fontsize=9)


def plot_heating(fig: Figure, res: FileResult, rt: RTCurve, out: pd.DataFrame, fit: dict, window: float):
    fig.clear()
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.5])
    ax_rt = fig.add_subplot(gs[:, 0])
    ax_dt = fig.add_subplot(gs[0, 1])
    ax_ph = fig.add_subplot(gs[1, 1], sharex=ax_dt)

    T0, R0, dRdT, mask = fit["T0"], fit["R0"], fit["dRdT"], fit["mask"]
    ax_rt.plot(rt.T, rt.R, ".", ms=2, color="0.6", label=f"R(T): {rt.label}")
    ax_rt.plot(rt.T[mask], rt.R[mask], ".", ms=3, color="C3", label="fit range")
    span = max(3 * window, np.ptp(rt.T[mask]))
    tt = np.linspace(T0 - span, T0 + span, 50)
    ax_rt.plot(tt, R0 + dRdT * (tt - T0), "-", color="C3", lw=1.2, label=f"dR/dT = {dRdT:.3g} Ω/K")
    ax_rt.axvline(T0, color="C3", ls=":", lw=1)
    ax_rt.set_xlabel("T [K]")
    ax_rt.set_ylabel("R [Ω]")
    ax_rt.grid(True, alpha=0.3)
    ax_rt.legend(fontsize=8)

    for branch, ls, color in (("down", "-", "C0"), ("up", "--", "C1")):
        part = out[out[BRANCH] == branch]
        B = part[FIELD_COL].to_numpy() / OE_PER_T
        ax_dt.plot(B, part["dT_2w [K]"], ls, color=color, lw=1.5, label=branch)
        ax_ph.plot(B, part["Phase 3f [deg]"], ls, color=color, lw=1.2, label=branch)
    ax_dt.set_ylabel("ΔT₂ω [K]")
    ax_ph.set_ylabel("Phase 3f [°]")
    ax_ph.set_xlabel("Magnetic field [T]")
    for ax in (ax_dt, ax_ph):
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.ticklabel_format(axis="y", useOffset=False)
    ph = out["Phase 3f [deg]"].to_numpy()
    if np.isfinite(ph).any() and np.nanmax(ph) - np.nanmin(ph) < 2:   # almost constant phase
        mid = float(np.nanmedian(ph))
        ax_ph.set_ylim(mid - 5, mid + 5)
    fig.suptitle(f"{res.path.stem}:  T = {T0:.2f} K,  I = {fit['I'] * 1e6:.3g} µA", fontsize=10)


def plot_hall(fig: Figure, items: list, unit: str, bmin_T: float, col: str):
    fig.clear()
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    bmax = 0.0
    for k, (res, c) in enumerate(items):
        color = f"C{k % 10}"
        for br, ls in (("down", "-"), ("up", "--")):
            m = c[BRANCH] == br
            B = c["B"][m] / OE_PER_T
            lab = res.path.stem if br == "down" else None
            ax1.plot(B, c["anti"][m], ls, color=color, lw=1.3, label=lab)
            ax2.plot(B, c["corr"][m], ls, color=color, lw=1.3, label=lab)
        m = c[BRANCH] == "up"
        ax1.plot(c["B"][m] / OE_PER_T, c["fit"][m], ":", color="k", lw=1.1,
                 label="fit normal Hall" if k == 0 else None)
        bmax = max(bmax, float(np.nanmax(np.abs(c["B"]))) / OE_PER_T)
    if bmax > bmin_T:
        for ax in (ax1, ax2):
            ax.axvspan(bmin_T, bmax, color="C3", alpha=0.07, lw=0)
            ax.axvspan(-bmax, -bmin_T, color="C3", alpha=0.07, lw=0)
    name = short_name(col)
    ax1.set_ylabel(f"{name} antisym [{unit}]")
    ax2.set_ylabel(f"antisym − R₀·B [{unit}]")
    ax2.set_xlabel("Magnetic field [T]")
    ax2.axhline(0, color="0.7", lw=0.8, zorder=0)
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
        if len(items) <= 10:
            ax.legend(fontsize=8)
    ax1.set_title(f"fit range |B| ≥ {bmin_T:g} T (shaded red)", fontsize=9)


def plot_comparison(fig: Figure, curves: list, unit: str, ylabel: str, param: str, offset: float,
                    branches: str, dep: pd.DataFrame, fit, loglog: bool, b0_oe):
    fig.clear()
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    pname, punit = PARAMS[param]

    ps = np.array([c["p"] if c["p"] is not None else np.nan for c in curves], float)
    use_cbar = np.isfinite(ps).sum() >= 2 and np.nanmax(ps) > np.nanmin(ps)
    norm = Normalize(np.nanmin(ps), np.nanmax(ps)) if use_cbar else None
    for k, c in enumerate(curves):
        if use_cbar and np.isfinite(ps[k]):
            color = CMAP(norm(ps[k]))
        else:
            color = CMAP(k / max(1, len(curves) - 1))
        if c["p"] is not None and param != "idx":
            lab = f"{c['p']:g} {punit}".strip()
        else:
            lab = c["res"].path.stem
        first = True
        for br, ls in (("down", "-"), ("up", "--")):
            if branches not in ("both", br):
                continue
            m = c[BRANCH] == br
            ax1.plot(c["B"][m] / OE_PER_T, c["y"][m] + k * offset, ls, color=color, lw=1.3,
                     label=lab if first else None)
            first = False
    ax1.set_xlabel("Magnetic field [T]")
    ax1.set_ylabel(f"{ylabel} [{unit}]" + (f"  (+ offset {offset:g})" if offset else ""))
    ax1.grid(True, alpha=0.3)
    if use_cbar:
        sm = ScalarMappable(norm=norm, cmap=CMAP)
        sm.set_array([])
        fig.colorbar(sm, ax=ax1, label=f"{pname} [{punit}]" if punit else pname, pad=0.01)
    if len(curves) <= 12:
        ax1.legend(fontsize=7, ncol=1 if len(curves) <= 6 else 2)
    if branches == "both":
        ax1.set_title("solid: down,  dashed: up", fontsize=9)

    if dep is not None and len(dep):
        x = dep["x"].to_numpy(float)
        y = dep["y"].to_numpy(float)
        xlabel = f"{pname} [{punit}]" if punit else pname
        if loglog:
            ok = x > 0
            pos, neg = ok & (y > 0), ok & (y < 0)
            if pos.any():
                ax2.loglog(x[pos], y[pos], "o", color="C0", label="value > 0")
            if neg.any():
                ax2.loglog(x[neg], -y[neg], "o", mfc="none", color="C0", label="value < 0")
            if fit:
                xx = np.geomspace(x[ok].min(), x[ok].max(), 50)
                err = "" if np.isnan(fit["err"]) else f" ± {fit['err']:.2f}"
                ax2.loglog(xx, np.exp(fit["ln_a"]) * xx ** fit["n"], "-", color="C3",
                           label=f"fit: n = {fit['n']:.2f}{err}")
            ax2.set_xscale("log")
            ax2.set_yscale("log")
            ax2.xaxis.set_major_formatter(ScalarFormatter())
            ax2.xaxis.set_minor_formatter(ScalarFormatter())
            ax2.set_ylabel(f"|S| [{unit}]")
            ax2.legend(fontsize=8)
        else:
            order = np.argsort(x)
            ax2.plot(x[order], y[order], "o-", color="C0", lw=1.2)
            ax2.axhline(0, color="0.7", lw=0.8, zorder=0)
            ax2.set_ylabel(f"S [{unit}]")
        ax2.set_xlabel(xlabel)
        ax2.grid(True, which="both", alpha=0.3)
        where = f"value at B = {b0_oe / OE_PER_T:g} T" if b0_oe is not None else "maximum |S| in the loop"
        ax2.set_title(where, fontsize=9)
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "no parameter values\nfor this plot", ha="center", va="center", color="0.55")


def save_figure(path: Path, dpi: int, draw, *args, figsize=(8, 7)):
    fig = Figure(figsize=figsize, layout="constrained")
    FigureCanvasAgg(fig)
    with MPL_LOCK:
        draw(fig, *args)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")


# ============================================================
# Evaluation of one file
# ============================================================
def frequency_of(header: dict, df: pd.DataFrame):
    return header.get("f_Hz") or column_mean(df, FREQ_COL)


def process_path(path: Path, s: Settings, log=print) -> list:
    """Evaluate one file. If the measurement was restarted in the same file, s.restart
    decides: 'last' = last evaluable section, 'each' = every section separately (as
    <name>_part<k>), 'all' = everything together. Returns a list of results
    (FileResult for sweeps, RTCurve for R(T) curves)."""
    path = Path(path)
    segs = read_segments(path)
    short = [g for g in segs if len(g["df"]) < 10]
    segs = [g for g in segs if len(g["df"]) >= 10]
    if short:
        log(f"  WARNING: {len(short)} section(s) with < 10 points ignored.")
    if not segs:
        raise ValueError("Too few valid data points.")
    n = len(segs)
    for g in segs:
        if g["gap_note"]:
            log(f"  WARNING: {g['gap_note']} from line {g['start']} – measurement interrupted?")

    if n == 1:
        return [process_df(path, segs[0]["df"], segs[0]["header"], s, log)]

    log(f"  WARNING: file contains {n} sections – measurement was restarted in the same file:")
    for k, g in enumerate(segs, 1):
        log(f"    section {k}: from line {g['start']}, {len(g['df'])} points ({g['reason']})")

    if s.restart == "all":
        log("  → all sections are evaluated together")
        df = pd.concat([g["df"] for g in segs], ignore_index=True)
        return [process_df(path, df, segs[0]["header"], s, log,
                           note=f"all {n} sections together (measurement restarted in the same file)")]

    if s.restart == "each":
        results = []
        for k, g in enumerate(segs, 1):
            vpath = path.with_name(f"{path.stem}_part{k}{path.suffix}")
            log(f"  — section {k} → {vpath.stem}")
            try:
                results.append(process_df(vpath, g["df"], g["header"], s, log,
                                          note=f"section {k} of {n} from {path.name} (from line {g['start']})"))
            except ValueError as e:
                log(f"  WARNING: section {k} cannot be evaluated: {e}")
        if not results:
            raise ValueError("No section can be evaluated.")
        return results

    for k in range(n, 0, -1):            # 'last': from the last section backwards
        g = segs[k - 1]
        buf: list = []
        try:
            res = process_df(path, g["df"], g["header"], s, buf.append,
                             note=f"section {k} of {n} (from line {g['start']}); "
                                  f"measurement restarted in the same file, other sections discarded")
        except ValueError as e:
            log(f"  section {k} cannot be evaluated ({e}) – trying the previous one")
            continue
        log(f"  → using section {k} of {n}")
        for m in buf:
            log(m)
        return [res]
    raise ValueError("No section can be evaluated.")


def process_df(path: Path, df: pd.DataFrame, header: dict, s: Settings, log=print, note=None):
    parts = []
    if "f_Hz" in header:
        parts.append(f"f = {header['f_Hz']:g} Hz")
    if "U_V" in header:
        parts.append(f"U = {header['U_V']:g} V")
    if "I" in header:
        parts.append(f"I = {header['I']:.4e} A")
    if "R" in header:
        parts.append(f"R = {header['R']:g} Ω")
    if parts:
        log("  Header: " + ", ".join(parts))
    if classify(df, s.field_step) == "rt":
        return process_rt(path, df, header, s, log, note)
    return process_sweep(path, df, header, s, log, note)


def process_rt(path: Path, df: pd.DataFrame, header: dict, s: Settings, log=print, note=None) -> RTCurve:
    if RE1 not in df.columns:
        raise ValueError("R(T) file contains no ReZ 1f column.")
    I, _, _ = determine_current(df, header, s.current_source)
    f_Hz, angle = frequency_of(header, df), column_mean(df, ANGLE_COL)
    rt = make_rt(path, df[TEMP_COL], df[RE1] / I, f"measurement file, R = ReZ1f / I (I = {I:.4e} A)",
                 B_oe=column_mean(df, FIELD_COL), angle=angle, current_A=I,
                 rho_factor=rho_factor_from(s), f_Hz=f_Hz)
    j, j_src = current_density(I, header, s)
    meta = base_meta(path, "R(T) curve", header, s, I, j, j_src, f_Hz, angle)
    meta["Temperature [K]"] = f"{rt.T.min():.3f} … {rt.T.max():.3f}"
    meta["Magnetic field [T]"] = f"{(rt.B_oe or 0) / OE_PER_T:.4g} (mean)"
    meta["Resistance"] = "R = ReZ 1st Harm / I"
    meta["Section"] = note
    rt.meta = meta

    if s.write_files:
        cols = [("T", "K", "", rt.T), ("R", "Ohm", path.stem, rt.R)]
        if rt.rho_factor:
            cols.append(("rho", "µOhm cm", path.stem, rt.R * rt.rho_factor * RHO_OHM_M_TO_UOHM_CM))
        write_origin(out_dir(path) / f"{path.stem}_RT.csv", {**meta, "Content": "R(T) raw data, sorted by T"}, cols)
        log(f"  saved: output/{path.stem}_RT.csv")
    log(f"  → R(T) curve: {rt.T.min():.2f} … {rt.T.max():.2f} K, B = {fmt((rt.B_oe or 0) / OE_PER_T, '.3g')} T, "
        f"R = {rt.R[0]:.4g} Ω ({rt.T[0]:.1f} K) … {rt.R[-1]:.4g} Ω ({rt.T[-1]:.1f} K)")
    return rt


def process_sweep(path: Path, df: pd.DataFrame, header: dict, s: Settings, log=print, note=None) -> FileResult:
    log(f"  → field sweep: {len(df)} data points, field {df[FIELD_COL].min():.0f} … {df[FIELD_COL].max():.0f} Oe")
    T_K = column_mean(df, TEMP_COL)
    if T_K is not None:
        log(f"  Temperature: {T_K:.3f} K (± {df[TEMP_COL].std():.3f} K)")
    angle = column_mean(df, ANGLE_COL)
    f_Hz = frequency_of(header, df)

    volt_cols = [c for c in VOLTAGE_COLS if c in df.columns]
    if not volt_cols:
        raise ValueError("No harmonic columns found in the file.")

    try:
        current, I_hdr, I_meas = determine_current(df, header, s.current_source)
        log(f"  Current: {current:.4e} A ({'header' if s.current_source == 'header' else 'measured'})")
        if I_hdr and I_meas:
            dev = abs(I_meas - I_hdr) / I_hdr
            if dev > 0.05:
                log(f"  WARNING: measured current ({I_meas:.4e} A) deviates by {dev:.0%} from the header "
                    f"(RMS vs. amplitude?)")
    except ValueError as e:
        if s.mode == "rho":
            raise
        current = None
        log(f"  WARNING: {e} R, rho, heating and current scaling are not possible for this file.")

    factor = rho_factor_from(s)
    if s.mode == "rho":
        if factor is None:
            raise ValueError("Sample dimensions missing for rho.")
        df, cols = to_resistivity(df, current, factor)
        unit = "µΩ cm"
        suffix = f"rho_{s.geometry}_uOhmcm"
    else:
        cols = volt_cols
        unit = "V"
        suffix = "V"

    down, up, notes = split_loop(df, s.field_step, s.min_points)
    for n in notes:
        log(f"  WARNING: {n}")
    log(f"  down sweep: {len(down)} points, up sweep: {len(up)} points")

    grid = symmetric_grid(down, up, s.field_step)
    log(f"  Field grid: ±{grid[-1]:.0f} Oe in {s.field_step:g} Oe steps ({len(grid)} points)")
    loop = make_loop(down, up, grid, s, cols)
    vloop = loop if s.mode == "voltage" else make_loop(down, up, grid, s, volt_cols)

    j, j_src = current_density(current, header, s)
    meta = base_meta(path, "Field sweep (loop)", header, s, current, j, j_src, f_Hz, angle)
    meta["Section"] = note
    if T_K is not None:
        meta["Temperature [K]"] = f"{T_K:.3f} ± {df[TEMP_COL].std():.3f}"
    meta["Magnetic field [T]"] = f"{grid[0] / OE_PER_T:g} … {grid[-1] / OE_PER_T:g}"
    meta["Field grid"] = f"{s.field_step:g} Oe; {METHODS[s.method]}; {len(down)} / {len(up)} points down / up"
    meta["Symmetrisation"] = "sym(B) = [S_down(B) + S_up(-B)] / 2;  antisym(B) = [S_down(B) - S_up(-B)] / 2"

    tag = tag_of(T_K, current, f_Hz, angle)
    csv_path = None
    if s.write_files:
        csv_path = out_dir(path) / f"{path.stem}_sym_antisym_{suffix}.csv"
        write_origin(csv_path, {**meta, "Content": f"symmetrised loop in {unit} (branch down: +B → -B; up: -B → +B)"},
                     field_columns(loop[FIELD_COL]) + df_columns(loop, tag))
        if s.make_plots:
            plotdir = img_dir(path)
            for c in cols:
                save_figure(plotdir / f"{path.stem}_{safe_name(c)}_sym_antisym.png", s.dpi,
                            plot_sym_antisym, loop, c, unit, f"{path.stem}   ({tag})")
            log(f"  {len(cols)} plots saved in output/figures/{plotdir.name}/")

    keep = [FIELD_COL] + cols
    return FileResult(
        path=path, unit=unit, value_cols=cols, loop=loop, vloop=vloop,
        down_raw=down[keep].copy(), up_raw=up[keep].copy(), csv_path=csv_path,
        current_A=current, T_K=T_K, f_Hz=f_Hz, angle=angle, field_step=s.field_step,
        geometry=s.geometry, rho_factor=factor, meta=meta, j_Am2=j,
    )
