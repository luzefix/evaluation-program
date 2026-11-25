"""AFM topography: file reading, image corrections and analysis.

Reading
-------
* Bruker / Veeco NanoScope files (``.spm``, ``.000`` …): ASCII header with one
  ``\\*Ciao image list`` block per channel, followed by binary integer data.
  Physical value = raw · (Z scale / 2^(8·bytes per pixel)) · sensitivity, where
  *Z scale* is the soft-scale value of ``\\@2:Z scale`` (e.g. 9.99 V) and the
  sensitivity comes from the referenced ``\\@Sens. …`` entry (e.g. Zsens in nm/V).
  NanoScope ≥ 9 stores 32-bit words even when the header says 2 bytes/pixel; the
  word size is taken from the data length, the scale from the header.
* Plain-text matrices (Gwyddion ASCII export: ``# Width: 5 µm`` header lines).

Corrections (all functions return new arrays; masks are True where a pixel is
*excluded* from a fit, e.g. pits or particles)
--------------------------------------------------------------------------
* ``remove_scars``       – single-line streaks along the fast scan axis
* ``level_polynomial``   – plane / polynomial background
* ``level_terraces``     – facet levelling (median local slope → 0), keeps the
                           step structure of vicinal surfaces intact
* ``align_rows``         – line-by-line offsets (median, median of differences,
                           mean, linear / quadratic per line)
* ``remove_outliers``    – isolated spikes (median filter)
* ``set_zero``           – minimum / mean / median to zero

Analysis
--------
roughness (Sq, Sa, Sz, Ssk, Sku), height histogram and terrace levels (step
heights), line profiles, 1D power spectral density, pit / island statistics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage, signal

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_LENGTH_NM = {"nm": 1.0, "~m": 1e3, "um": 1e3, "µm": 1e3, "μm": 1e3, "pm": 1e-3,
              "mm": 1e6, "m": 1e9, "Å": 0.1, "A": 0.1}


# --------------------------------------------------------------------------- data


@dataclass
class AFMChannel:
    name: str
    data: np.ndarray            # rows = slow axis (top row first), columns = fast axis
    unit: str                   # e.g. "nm", "mV", "deg"
    width_nm: float             # physical size along x (columns)
    height_nm: float            # physical size along y (rows)
    direction: str = ""         # Trace / Retrace
    meta: dict = field(default_factory=dict)

    @property
    def dx(self) -> float:
        return self.width_nm / self.data.shape[1]

    @property
    def dy(self) -> float:
        return self.height_nm / self.data.shape[0]

    @property
    def label(self) -> str:
        d = f" ({self.direction})" if self.direction else ""
        return f"{self.name}{d}"

    @property
    def is_height(self) -> bool:
        n = self.name.lower()
        return self.unit == "nm" and ("height" in n or n.startswith("z"))


@dataclass
class AFMFile:
    path: Path
    channels: list
    header: dict = field(default_factory=dict)

    def height_channel(self) -> AFMChannel:
        for c in self.channels:
            if c.is_height:
                return c
        return self.channels[0]


# --------------------------------------------------------------------------- reading


def _split_header(text: str):
    sections = []
    for line in text.replace("\r", "").split("\n"):
        line = line.strip()
        if not line.startswith("\\"):
            continue
        if line.startswith("\\*"):
            sections.append((line[2:].strip(), {}))
            continue
        if not sections:
            sections.append(("", {}))
        key, sep, val = line[1:].partition(": ")
        if not sep:
            key, _, val = line[1:].partition(":")
        sections[-1][1].setdefault(key.strip(), val.strip())
    return sections


def _scale_entry(value: str):
    """Parse 'V [Sens. Zsens] (0.0067 V/LSB) 9.9976 V' → (sens_name, soft value, unit)."""
    m = re.match(rf"\s*\w*\s*(?:\[(.*?)\])?\s*(?:\(\s*({_NUM})\s*([^)]*)\))?\s*({_NUM})?\s*(\S*)", value)
    if not m:
        return None, None, ""
    sens, _hard, hard_unit, soft, unit = m.groups()
    if soft is None:  # only a hard value: per-LSB scale
        return sens, None, (hard_unit or "").split("/")[0]
    return sens, float(soft), unit


def _sensitivity(all_keys: dict, name: str):
    v = all_keys.get(f"@{name}")
    if v is None:
        return 1.0, None
    m = re.match(rf"\s*\w*\s*({_NUM})\s*(\S*)", v)
    if not m:
        return 1.0, None
    unit = m.group(2) or None
    if unit and "/" in unit:
        unit = unit.split("/")[0]
    return float(m.group(1)), unit


def _length_nm(text: str):
    """'5 5 ~m' or '5000 nm' → (x_nm, y_nm)."""
    parts = text.split()
    nums = [float(p) for p in parts if re.fullmatch(_NUM, p)]
    unit = next((p for p in parts if not re.fullmatch(_NUM, p)), "nm")
    f = _LENGTH_NM.get(unit, 1.0)
    if not nums:
        return np.nan, np.nan
    x = nums[0] * f
    y = (nums[1] if len(nums) > 1 else nums[0]) * f
    return x, y


def _unit_clean(u: str) -> str:
    u = (u or "").strip()
    if u in ("º", "°", "\xb0", "�", "deg") or u.endswith("\xb0"):
        return "deg"
    if u == "~m":
        return "µm"
    return u


def read_nanoscope(path) -> AFMFile:
    path = Path(path)
    raw = path.read_bytes()
    end = raw.find(b"\\*File list end")
    if end < 0:
        raise ValueError(f"{path.name}: not a NanoScope file (no '\\*File list end').")
    text = raw[:end].decode("latin-1")
    sections = _split_header(text)
    all_keys = {}
    for _, d in sections:
        for k, v in d.items():
            all_keys.setdefault(k, v)
    scan = next((d for n, d in sections if n.lower().startswith("ciao scan list")), {})
    channels = []
    for name, d in sections:
        if not name.lower().startswith("ciao image list"):
            continue
        offset = int(d["Data offset"])
        length = int(d["Data length"])
        nx = int(d.get("Samps/line", scan.get("Samps/line", 0)))
        ny = int(d.get("Number of lines", scan.get("Lines", 0)))
        bpp_header = int(d.get("Bytes/pixel", 2))
        word = length // (nx * ny) if nx and ny else bpp_header
        dtype = {1: "<i1", 2: "<i2", 4: "<i4"}.get(word)
        if dtype is None:
            raise ValueError(f"{path.name}: unsupported data word size {word}.")
        arr = np.frombuffer(raw, dtype=dtype, count=nx * ny, offset=offset).astype(float)
        arr = arr.reshape(ny, nx)
        img_key = next((k for k in d if k.endswith("Image Data")), None)
        cname = "Channel"
        if img_key:
            m = re.search(r'"(.*?)"', d[img_key])
            cname = m.group(1) if m else d[img_key].split()[-1]
        zkey = next((k for k in d if k.endswith("Z scale")), None)
        sens_name, soft, unit = _scale_entry(d[zkey]) if zkey else (None, None, "")
        scale = 1.0
        if soft is not None:
            scale = soft / 2 ** (8 * bpp_header)
        if sens_name:
            sens, sunit = _sensitivity(all_keys, sens_name)
            scale *= sens
            if sunit:
                unit = sunit
        width, height = _length_nm(d.get("Scan Size", scan.get("Scan Size", "0 nm")))
        if height == width and nx != ny:  # aspect ratio follows the pixel count
            height = width * ny / nx
        data = np.flipud(arr * scale)  # stored bottom line first
        channels.append(AFMChannel(
            name=cname, data=data, unit=_unit_clean(unit), width_nm=width, height_nm=height,
            direction=d.get("Line Direction", ""),
            meta={"plane_fit": d.get("Plane fit", ""), "frame": d.get("Frame direction", ""),
                  "scale": scale}))
    if not channels:
        raise ValueError(f"{path.name}: no image data found.")
    header = {k: all_keys[k] for k in ("Date", "Scan Rate", "Microscope", "Operating mode",
                                       "Scan Size", "Rotate Ang.") if k in all_keys}
    return AFMFile(path=path, channels=channels, header=header)


def read_ascii_matrix(path) -> AFMFile:
    """Gwyddion-style ASCII matrix. Header lines '# Width: 5.000 µm', '# Height: …',
    '# Value units: m|nm'. Without a header the size has to be set afterwards."""
    path = Path(path)
    meta = {}
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            k, _, v = s[1:].partition(":")
            meta[k.strip().lower()] = v.strip()
            continue
        rows.append([float(t) for t in re.split(r"[\s,;]+", s) if t])
    data = np.array(rows, float)
    w = _length_nm(meta.get("width", "nan nm"))[0]
    h = _length_nm(meta.get("height", "nan nm"))[0]
    if not np.isfinite(w):
        w = float(data.shape[1])
    if not np.isfinite(h):
        h = w * data.shape[0] / data.shape[1]
    unit = meta.get("value units", "nm")
    if unit == "m":
        data = data * 1e9
        unit = "nm"
    ch = AFMChannel(name=meta.get("channel", "Height"), data=data, unit=unit,
                    width_nm=w, height_nm=h)
    return AFMFile(path=path, channels=[ch], header=meta)


def load_afm(path) -> AFMFile:
    path = Path(path)
    head = path.read_bytes()[:64]
    if head.lstrip().startswith(b"\\*File list") or head.lstrip().startswith(b"\\*Force file list"):
        return read_nanoscope(path)
    return read_ascii_matrix(path)


def save_ascii_matrix(path, z, width_nm, height_nm, channel="Height", unit="nm"):
    """Gwyddion-compatible ASCII matrix (can be opened in Gwyddion)."""
    path = Path(path)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"# Channel: {channel}\n# Width: {width_nm / 1e3:.6g} µm\n"
                 f"# Height: {height_nm / 1e3:.6g} µm\n# Value units: {unit}\n")
        np.savetxt(fh, z, fmt="%.6g", delimiter="\t")
    return path


# --------------------------------------------------------------------------- helpers


def robust_sigma(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def _coords(shape):
    ny, nx = shape
    y = np.linspace(-1, 1, ny)
    x = np.linspace(-1, 1, nx)
    return np.meshgrid(x, y)


def _poly_terms(X, Y, order):
    return [X ** i * Y ** j for n in range(order + 1) for j in range(n + 1) for i in [n - j]]


# --------------------------------------------------------------------------- corrections


def polynomial_background(z, order=1, mask=None):
    X, Y = _coords(z.shape)
    terms = _poly_terms(X, Y, order)
    ok = np.isfinite(z) if mask is None else (~mask & np.isfinite(z))
    if ok.sum() < len(terms):
        ok = np.isfinite(z)
    # subsample very large images for speed
    idx = np.flatnonzero(ok.ravel())
    if idx.size > 400_000:
        idx = idx[:: idx.size // 400_000 + 1]
    A = np.stack([t.ravel()[idx] for t in terms], axis=1)
    c, *_ = np.linalg.lstsq(A, z.ravel()[idx], rcond=None)
    return sum(ci * t for ci, t in zip(c, terms))


def level_polynomial(z, order=1, mask=None):
    return z - polynomial_background(z, order, mask)


def plane_slopes(z, dx, dy, mask=None):
    """Mean slopes (dz/dx, dz/dy) of a least-squares plane, in units of z per nm."""
    ny, nx = z.shape
    y, x = np.mgrid[:ny, :nx]
    ok = np.isfinite(z) if mask is None else (~mask & np.isfinite(z))
    A = np.c_[np.ones(ok.sum()), x[ok] * dx, y[ok] * dy]
    c, *_ = np.linalg.lstsq(A, z[ok], rcond=None)
    return float(c[1]), float(c[2])


def _inlier_mean(g, nsigma=3.0):
    """Mean of the gradients after rejecting step edges and pits (> nσ from the
    median). Unlike the plain median it is not biased by the one-sided step
    pixels of a staircase."""
    g = g[np.isfinite(g)]
    if g.size == 0:
        return np.nan
    med = np.median(g)
    sig = robust_sigma(g) or 1e-12
    ok = np.abs(g - med) < nsigma * sig
    return float(g[ok].mean()) if ok.any() else float(med)


def level_terraces(z, mask=None, iterations=4):
    """Facet levelling: remove the plane whose slope equals the typical local slope
    (mean gradient after rejecting step edges). On vicinal (stepped) surfaces the
    terraces dominate the area, so this is the terrace inclination and the terraces end up horizontal, while
    a least-squares plane would tilt them by the miscut."""
    z = z.copy()
    ny, nx = z.shape
    y, x = np.mgrid[:ny, :nx]
    for _ in range(iterations):
        gx = np.diff(z, axis=1)
        gy = np.diff(z, axis=0)
        if mask is not None:
            gx = np.where(mask[:, 1:] | mask[:, :-1], np.nan, gx)
            gy = np.where(mask[1:, :] | mask[:-1, :], np.nan, gy)
        sx = _inlier_mean(gx)
        sy = _inlier_mean(gy)
        if not np.isfinite(sx) or not np.isfinite(sy):
            break
        z -= sx * x + sy * y
        if abs(sx) < 1e-9 and abs(sy) < 1e-9:
            break
    return z - np.nanmedian(z)


ROW_METHODS = ("median", "median of differences", "mean", "linear", "quadratic")


def align_rows(z, method="median", mask=None):
    """Remove line-to-line offsets along the slow scan axis.

    * ``median`` – subtract the median of each line (robust against particles),
    * ``median of differences`` – shift each line so that the median height
      difference to the previous line is zero. Preferred for terraces and large
      features that span many lines, because it does not assume that every line
      has the same mean height,
    * ``mean``, ``linear``, ``quadratic`` – subtract a polynomial per line.
    """
    z = np.array(z, float)
    m = np.zeros(z.shape, bool) if mask is None else mask
    zm = np.where(m, np.nan, z)
    if method == "median":
        off = np.nanmedian(zm, axis=1)
        return z - np.nan_to_num(off)[:, None]
    if method == "mean":
        off = np.nanmean(zm, axis=1)
        return z - np.nan_to_num(off)[:, None]
    if method == "median of differences":
        d = np.nanmedian(zm[1:] - zm[:-1], axis=1)
        d = np.nan_to_num(d)
        off = np.concatenate([[0.0], np.cumsum(d)])
        return z - off[:, None]
    order = {"linear": 1, "quadratic": 2}.get(method)
    if order is None:
        raise ValueError(f"unknown row method {method!r}")
    nx = z.shape[1]
    t = np.linspace(-1, 1, nx)
    V = np.vander(t, order + 1)                       # (nx, p)
    w = (~m).astype(float) * np.isfinite(z)
    zz = np.nan_to_num(z)
    A = np.einsum("rn,np,nq->rpq", w, V, V)           # (rows, p, p)
    b = np.einsum("rn,rn,np->rp", w, zz, V)
    bad = np.linalg.matrix_rank(A) < order + 1
    A[bad] = np.einsum("np,nq->pq", V, V)
    b[bad] = zz[bad] @ V
    c = np.linalg.solve(A, b[..., None])[..., 0]
    return z - c @ V.T


def remove_scars(z, threshold=3.0, min_length=12, max_width=3):
    """Detect streaks (1 … ``max_width`` lines wide) that are offset from the lines
    directly above *and* below by more than ``threshold`` × the robust line-to-line
    noise, with a length of at least ``min_length`` pixels along the fast axis;
    replace them by interpolation along the slow axis. Returns (corrected, mask)."""
    z = np.array(z, float)
    ny = z.shape[0]
    sig = robust_sigma(z[1:] - z[:-1]) or 1e-12
    cand = np.zeros(z.shape, bool)
    for w in range(1, int(max_width) + 1):
        if ny < w + 2:
            break
        above = z[: ny - w - 1]                 # row i-1
        below = z[w + 1:]                       # row i+w
        block = [z[1 + k: ny - w + k] for k in range(w)]
        hit = np.ones(above.shape, bool)
        for b in block:
            up, dn = b - above, b - below
            hit &= (np.sign(up) == np.sign(dn)) & (np.minimum(np.abs(up), np.abs(dn)) > threshold * sig)
        if w > 1:  # all lines of the block offset in the same direction
            sgn = np.sign(block[0] - above)
            for b in block[1:]:
                hit &= np.sign(b - above) == sgn
        for k in range(w):
            cand[1 + k: ny - w + k] |= hit
    if min_length > 1:
        cand = ndimage.binary_opening(cand, structure=np.ones((1, int(min_length)), bool))
    if not cand.any():
        return z, cand
    rows = np.arange(ny)
    for j in np.flatnonzero(cand.any(axis=0)):
        bad = cand[:, j]
        good = ~bad
        if good.sum() >= 2:
            z[bad, j] = np.interp(rows[bad], rows[good], z[good, j])
    return z, cand


def remove_outliers(z, nsigma=5.0, size=3):
    """Replace isolated spikes (|z − median filter| > nσ) by the median-filtered value."""
    med = ndimage.median_filter(z, size=size, mode="nearest")
    r = z - med
    bad = np.abs(r) > nsigma * (robust_sigma(r) or 1e-12)
    out = np.where(bad, med, z)
    return out, bad


def feature_mask(z, nsigma=2.5, dilate=2):
    """Pixels deviating by more than nσ (robust) from the median: pits, particles."""
    r = z - np.nanmedian(z)
    m = np.abs(r) > nsigma * (robust_sigma(r) or 1e-12)
    if dilate:
        m = ndimage.binary_dilation(m, iterations=int(dilate))
    return m


def set_zero(z, mode="minimum"):
    if mode == "minimum":
        return z - np.nanmin(z)
    if mode == "mean":
        return z - np.nanmean(z)
    if mode == "median":
        return z - np.nanmedian(z)
    if mode in ("none", None, ""):
        return z
    raise ValueError(mode)


BACKGROUNDS = ("none", "plane", "polynomial 2", "polynomial 3", "terraces (facet level)")


@dataclass
class Corrections:
    scars: bool = True
    scar_threshold: float = 3.0
    scar_length: int = 12
    scar_width: int = 3
    background: str = "plane"
    rows: str = "median"            # 'none' or one of ROW_METHODS
    mask_features: bool = True
    mask_sigma: float = 2.5
    outliers: bool = False
    outlier_sigma: float = 6.0
    zero: str = "median"


@dataclass
class Corrected:
    z: np.ndarray
    mask: np.ndarray                # features excluded from the fits
    scars: np.ndarray
    outliers: np.ndarray
    log: list


def correct(z, c: Corrections = Corrections()) -> Corrected:
    """Standard order: background + line alignment → scars (detected on the levelled
    image, where line offsets no longer hide them) → background + line alignment
    again with a mask of pits/particles → spikes → zero."""
    z = np.array(z, float)
    log = []

    def level(z, mask):
        if c.background == "plane":
            z = level_polynomial(z, 1, mask)
        elif c.background.startswith("polynomial"):
            z = level_polynomial(z, int(c.background.split()[-1]), mask)
        elif c.background.startswith("terraces"):
            z = level_terraces(z, mask)
        if c.rows and c.rows != "none":
            z = align_rows(z, c.rows, mask)
        return z

    z = level(z, None)
    scars = np.zeros(z.shape, bool)
    if c.scars:
        z, scars = remove_scars(z, c.scar_threshold, c.scar_length, c.scar_width)
        lab, nseg = ndimage.label(scars)
        log.append(f"scars: {nseg} streaks, {int(scars.sum())} px replaced")
        if scars.any():
            z = level(z, None)
    base = z
    mask = np.zeros(z.shape, bool)
    if c.mask_features:
        for _ in range(2):
            mask = feature_mask(z, c.mask_sigma)
            if mask.mean() > 0.5:  # mask would be meaningless
                mask[:] = False
                break
            z = level(base, mask)
        log.append(f"feature mask: {100 * mask.mean():.1f} % of the image excluded from fits")
    if c.background != "none":
        log.append(f"background: {c.background}")
    if c.rows and c.rows != "none":
        log.append(f"line alignment: {c.rows}")
    outl = np.zeros(z.shape, bool)
    if c.outliers:
        z, outl = remove_outliers(z, c.outlier_sigma)
        log.append(f"outliers: {int(outl.sum())} px")
    z = set_zero(z, c.zero)
    return Corrected(z=z, mask=mask, scars=scars, outliers=outl, log=log)


# --------------------------------------------------------------------------- analysis


def roughness(z, mask=None) -> dict:
    """ISO 25178 amplitude parameters (on the levelled image)."""
    v = np.asarray(z, float)
    if mask is not None:
        v = v[~mask]
    v = v[np.isfinite(v)].ravel()
    m = v.mean()
    r = v - m
    sq = float(np.sqrt(np.mean(r ** 2)))
    return {
        "Sq": sq,
        "Sa": float(np.mean(np.abs(r))),
        "Sz": float(v.max() - v.min()),
        "Ssk": float(np.mean(r ** 3) / sq ** 3) if sq > 0 else np.nan,
        "Sku": float(np.mean(r ** 4) / sq ** 4) if sq > 0 else np.nan,
        "mean": float(m),
    }


@dataclass
class Histogram:
    centers: np.ndarray
    counts: np.ndarray
    smooth: np.ndarray
    levels: np.ndarray        # heights of the terrace levels (histogram peaks)
    level_weights: np.ndarray  # area fraction of the peaks (relative heights)
    steps: np.ndarray         # differences between neighbouring levels


def height_histogram(z, bins=400, smooth_bins=2.0, prominence=0.05, clip=(0.2, 99.8)) -> Histogram:
    v = np.asarray(z, float)
    v = v[np.isfinite(v)]
    lo, hi = np.percentile(v, clip)
    counts, edges = np.histogram(v, bins=bins, range=(lo, hi))
    centers = 0.5 * (edges[1:] + edges[:-1])
    sm = ndimage.gaussian_filter1d(counts.astype(float), smooth_bins) if smooth_bins else counts.astype(float)
    pk, prop = signal.find_peaks(sm, prominence=prominence * sm.max(), distance=max(2, int(2 * smooth_bins)))
    levels = []
    for p in pk:
        if 0 < p < len(sm) - 1:
            a, b, c = sm[p - 1], sm[p], sm[p + 1]
            den = a - 2 * b + c
            shift = 0.5 * (a - c) / den if den != 0 else 0.0
            levels.append(centers[p] + shift * (centers[1] - centers[0]))
        else:
            levels.append(centers[p])
    levels = np.array(levels)
    weights = sm[pk] / sm[pk].max() if len(pk) else np.array([])
    return Histogram(centers, counts, sm, levels, weights, np.diff(levels))


def line_profile(z, dx, dy, p0, p1, width=1, n=None):
    """Height along the line p0 → p1 (points as (x_nm, y_nm), y measured from the
    top row), averaged over ``width`` pixels perpendicular to the line.
    Returns (distance_nm, height)."""
    (x0, y0), (x1, y1) = p0, p1
    L = np.hypot(x1 - x0, y1 - y0)
    step = min(dx, dy)
    n = n or max(2, int(L / step) + 1)
    t = np.linspace(0, 1, n)
    xs = (x0 + t * (x1 - x0)) / dx - 0.5
    ys = (y0 + t * (y1 - y0)) / dy - 0.5
    if width > 1 and L > 0:
        nx_, ny_ = -(y1 - y0) / L, (x1 - x0) / L
        offs = np.arange(width) - (width - 1) / 2
        prof = np.mean([ndimage.map_coordinates(z, [ys + o * ny_ * step / dy, xs + o * nx_ * step / dx],
                                                order=1, mode="nearest") for o in offs], axis=0)
    else:
        prof = ndimage.map_coordinates(z, [ys, xs], order=1, mode="nearest")
    return t * L, prof


def psd_1d(z, dx, axis=1):
    """Averaged one-dimensional PSD along the fast (axis=1) or slow axis.
    Returns spatial frequency f (1/nm) and PSD (z²·nm), normalised so that
    ∫PSD df = mean square of the (line-wise mean-free) data."""
    a = np.asarray(z, float)
    if axis == 0:
        a = a.T
    a = a - a.mean(axis=1, keepdims=True)
    n = a.shape[1]
    w = np.hanning(n)
    F = np.fft.rfft(a * w, axis=1)
    p = (np.abs(F) ** 2).mean(axis=0) * dx / np.sum(w ** 2)
    p[1:] *= 2  # one-sided
    if n % 2 == 0:
        p[-1] /= 2  # Nyquist bin appears only once
    f = np.fft.rfftfreq(n, dx)
    return f[1:], p[1:]


def correlation_length(z, dx, axis=1):
    """Lateral correlation length: distance where the averaged autocorrelation
    function along the fast axis drops to 1/e."""
    a = np.asarray(z, float)
    if axis == 0:
        a = a.T
    a = a - a.mean(axis=1, keepdims=True)
    n = a.shape[1]
    F = np.fft.rfft(a, 2 * n, axis=1)
    acf = np.fft.irfft(np.abs(F) ** 2, axis=1)[:, :n].mean(axis=0)
    acf /= acf[0] if acf[0] else 1
    below = np.flatnonzero(acf < np.exp(-1))
    if not below.size:
        return np.nan
    i = below[0]
    x0, x1 = acf[i - 1], acf[i]
    return float(dx * (i - 1 + (x0 - np.exp(-1)) / (x0 - x1)))


@dataclass
class Features:
    kind: str                  # 'pits' or 'islands'
    reference: np.ndarray      # reference level per pixel
    threshold: float
    labels: np.ndarray
    table: list                # dicts per feature
    summary: dict


def local_reference(z, dx, dy, window_nm, nsigma=2.0):
    """Local reference height: moving average over ``window_nm`` that ignores
    pits/particles (pixels > nσ from the local level are excluded and the
    average is recomputed). Follows terraces and long-wavelength waviness."""
    size = (max(3, int(round(window_nm / dy)) | 1), max(3, int(round(window_nm / dx)) | 1))
    w = np.ones_like(z)
    ref = ndimage.uniform_filter(z, size, mode="reflect")
    for _ in range(3):
        r = z - ref
        w = (np.abs(r) < nsigma * (robust_sigma(r) or 1e-12)).astype(float)
        num = ndimage.uniform_filter(z * w, size, mode="reflect")
        den = ndimage.uniform_filter(w, size, mode="reflect")
        ref = np.where(den > 1e-3, num / np.maximum(den, 1e-3), ref)
    return ref


def find_features(z, dx, dy, threshold, kind="pits", reference=None, min_pixels=4,
                  window_nm=0.0) -> Features:
    """Pits (below reference − threshold) or islands/particles (above reference +
    threshold). Reference: ``reference`` if given, else a local reference level
    (``window_nm`` > 0, see ``local_reference``; use it for terraces or wavy
    surfaces), else the median height."""
    if reference is not None:
        ref = np.full(z.shape, float(reference))
    elif window_nm and window_nm > 0:
        ref = local_reference(z, dx, dy, window_nm)
    else:
        ref = np.full(z.shape, float(np.nanmedian(z)))
    rel = z - ref
    sel = (rel < -threshold) if kind == "pits" else (rel > threshold)
    lab, n = ndimage.label(sel)
    if n:
        sizes = ndimage.sum(np.ones_like(z), lab, index=np.arange(1, n + 1))
        keep = np.flatnonzero(sizes >= min_pixels) + 1
        remap = np.zeros(n + 1, int)
        remap[keep] = np.arange(1, len(keep) + 1)
        lab = remap[lab]
        n = len(keep)
    pix = dx * dy
    rows = []
    if n:
        idx = np.arange(1, n + 1)
        npx = ndimage.sum(np.ones_like(z), lab, idx)
        ext = ndimage.minimum(rel, lab, idx) if kind == "pits" else ndimage.maximum(rel, lab, idx)
        vol = ndimage.sum(rel, lab, idx) * pix
        com = ndimage.center_of_mass(np.ones_like(z), lab, idx)
        for k in range(n):
            area = npx[k] * pix
            rows.append({"#": k + 1, "x_nm": com[k][1] * dx, "y_nm": com[k][0] * dy,
                         "area_nm2": area, "d_eq_nm": 2 * np.sqrt(area / np.pi),
                         "extreme_nm": float(ext[k]), "volume_nm3": float(vol[k])})
    total = z.size * pix
    summ = {"count": n, "density_um2": n / (total / 1e6) if total else np.nan,
            "area_fraction": float((lab > 0).mean())}
    if rows:
        summ.update({"mean_d_eq_nm": float(np.mean([r["d_eq_nm"] for r in rows])),
                     "mean_extreme_nm": float(np.mean([r["extreme_nm"] for r in rows])),
                     "max_extreme_nm": float((min if kind == "pits" else max)(r["extreme_nm"] for r in rows))})
    return Features(kind, ref, threshold, lab, rows, summ)
