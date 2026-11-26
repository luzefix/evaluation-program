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
import warnings
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


def _terrace_slopes(z, ok, min_px=50):
    """Common slope of all terraces: least squares z = a·x + b·y + c_k with an own
    offset c_k for every connected terrace region (``ok`` = pixels away from step
    edges). Unlike a gradient average it is not biased by left-over edge pixels."""
    lab, n = ndimage.label(ok)
    if n == 0:
        return None, None
    idx = np.arange(1, n + 1)
    cnt = ndimage.sum(np.ones_like(z), lab, idx)
    good = np.zeros(n + 1, bool)
    good[1:] = cnt >= min_px
    sel = good[lab]
    if sel.sum() < 100:
        return None, None
    ny, nx = z.shape
    y, x = np.mgrid[:ny, :nx].astype(float)
    def demean(v):
        m = np.zeros(n + 1)
        m[1:] = ndimage.mean(v, lab, idx)
        return (v - m[lab])[sel]
    A = np.c_[demean(x), demean(y)]
    c, *_ = np.linalg.lstsq(A, demean(z), rcond=None)
    return float(c[0]), float(c[1])


def _terrace_pixels(z, mask=None, smooth_px=2.0, edge_sigma=2.0):
    """Pixels away from step edges (and outside ``mask``). Edges are pixels whose
    gradient magnitude (of a smoothed copy, relative to the median gradient) deviates
    by > edge_sigma·σ from the typical value, dilated by 3× the smoothing width so
    that their tails are also excluded. Returns (ok, gx, gy) with gradients per pixel."""
    zs = ndimage.gaussian_filter(z, smooth_px, mode="nearest") if smooth_px else z
    gy, gx = np.gradient(zs)
    gx0, gy0 = np.median(gx), np.median(gy)
    mag = np.hypot(gx - gx0, gy - gy0)
    edge = mag > np.median(mag) + edge_sigma * (robust_sigma(mag) or 1e-12)
    edge = ndimage.binary_dilation(edge, iterations=max(1, int(round(3 * smooth_px))))
    ok = ~edge
    if mask is not None:
        ok &= ~mask
    if ok.sum() < 0.05 * z.size:
        ok = np.ones_like(ok) if mask is None else ~mask
    return ok, gx, gy


def level_terraces(z, mask=None, iterations=4, smooth_px=2.0, edge_sigma=2.0, fit_rows=False,
                   bow_order=0, info=None):
    """Facet levelling: remove the plane whose slope equals the slope of the
    terraces. On vicinal (stepped) surfaces the terraces end up horizontal, while a
    least-squares plane would tilt them by the miscut angle.

    The terrace slope is the mean gradient of a smoothed copy outside the step
    edges (see ``_terrace_pixels``; otherwise noisy images give a slope biased
    towards the miscut).

    ``fit_rows``: also determine the line offsets from the terraces;
    ``bow_order`` (2 or 3): also remove a 2D polynomial background (piezo bow,
    sample bending) whose curvature is measured *inside* the terraces, with an own
    offset per terrace (see ``_terrace_model_fit``). A plain polynomial fit would
    flatten the staircase itself; this one keeps the steps. If the step edges are
    not resolved (one terrace region covers most of the image) the bow is skipped;
    ``info`` (dict) then gets 'bow_skipped' = True."""
    z = np.array(z, float)
    ny, nx = z.shape
    y, x = np.mgrid[:ny, :nx]
    for _ in range(iterations):
        ok, gx, gy = _terrace_pixels(z, mask, smooth_px, edge_sigma)
        sx, sy = _terrace_slopes(z, ok)
        if sx is None:
            sx, sy = float(np.mean(gx[ok])), float(np.mean(gy[ok]))
        z -= sx * x + sy * y
        if abs(sx) < 1e-9 and abs(sy) < 1e-9:
            break
    bow = int(bow_order) if bow_order and int(bow_order) >= 2 else 0
    if fit_rows or bow:
        for k in range(2 if bow else 1):
            if k:  # edges are found more reliably once the bow is gone
                ok, _, _ = _terrace_pixels(z, mask, smooth_px, edge_sigma)
            z = _terrace_model_fit(z, ok, rows=fit_rows, order=bow or 1, info=info)
    return z - np.nanmedian(z)


def _terrace_model_fit(z, ok, rows=True, order=1, min_px=50, max_points=300_000, info=None,
                       max_region=0.5):
    """Joint least-squares fit z = c_k(terrace) [+ r_i(line)] + P(x, y) on the pixels
    away from step edges (sparse solver) and subtraction of r_i and P.

    * c_k: one free offset per connected terrace region, so the staircase itself is
      not part of the background;
    * r_i (``rows``): line offsets referenced to whole terraces – they do not
      accumulate noise the way the line-to-line 'median of differences' does (a
      random walk over the image);
    * P: 2D polynomial of ``order`` without constant and linear terms (except x with
      line offsets); with line offsets the pure-y terms are left out (r_i already
      contains any function of y).
    Curvature terms (order ≥ 2) are only fitted when the largest terrace region holds
    less than ``max_region`` of the terrace pixels – otherwise the edges are not
    resolved, the regions span several steps and the polynomial would flatten the
    staircase."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import lsqr

    lab, n = ndimage.label(ok)
    if n == 0:
        return z
    cnt = np.bincount(lab.ravel(), minlength=n + 1)
    keep = cnt >= min_px
    keep[0] = False
    sel = keep[lab]
    if sel.mean() < 0.2:
        return z
    if order >= 2 and cnt[keep].max() > max_region * cnt[keep].sum():
        order = 1
        if info is not None:
            info["bow_skipped"] = True
    if not rows and order < 2:
        return z
    ny, nx = z.shape
    ii, jj = np.nonzero(sel)
    if ii.size > max_points:
        pick = np.random.default_rng(0).choice(ii.size, max_points, replace=False)
        ii, jj = ii[pick], jj[pick]
    remap = -np.ones(n + 1, int)
    remap[keep] = np.arange(keep.sum())
    k = remap[lab[ii, jj]]
    nk = int(keep.sum())
    nr = ny if rows else 0
    m = ii.size
    X = np.linspace(-1, 1, nx)
    Y = np.linspace(-1, 1, ny)
    # linear terms: only x with line offsets (as before); the plane itself is set by the
    # facet levelling – a free plane here could absorb the miscut ramp of the
    # staircase where step edges are not detected and terraces merge
    powers = [(p - q, q) for p in range(1, int(order) + 1) for q in range(p + 1)
              if not (rows and p - q == 0) and (p >= 2 or (rows and q == 0))]
    r_idx, c_idx, vals = [np.arange(m)], [k], [np.ones(m)]
    if rows:
        r_idx.append(np.arange(m))
        c_idx.append(nk + ii)
        vals.append(np.ones(m))
    for t, (px, py) in enumerate(powers):
        r_idx.append(np.arange(m))
        c_idx.append(np.full(m, nk + nr + t))
        vals.append(X[jj] ** px * Y[ii] ** py)
    A = coo_matrix((np.concatenate(vals), (np.concatenate(r_idx), np.concatenate(c_idx))),
                   shape=(m, nk + nr + len(powers))).tocsr()
    sol = lsqr(A, z[ii, jj], damp=1e-6, atol=1e-10, btol=1e-10, iter_lim=5000)[0]
    out = z.copy()
    if rows:
        r = sol[nk:nk + ny]
        seen = np.zeros(ny, bool)
        seen[np.unique(ii)] = True
        if not seen.all():  # lines without terrace pixels: interpolate
            r[~seen] = np.interp(np.flatnonzero(~seen), np.flatnonzero(seen), r[seen])
        out -= r[:, None]
    if powers:
        XX, YY = np.meshgrid(X, Y)
        out -= sum(c * XX ** px * YY ** py for c, (px, py) in zip(sol[nk + nr:], powers))
    return out


def _terrace_row_offsets(z, ok, min_px=50, max_points=300_000):
    """Line offsets + linear x term referenced to the terraces (kept for
    compatibility, see ``_terrace_model_fit``)."""
    return _terrace_model_fit(z, ok, rows=True, order=1, min_px=min_px, max_points=max_points)


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
    if method in ("median", "mean"):
        with warnings.catch_warnings():  # fully masked lines → offset 0
            warnings.simplefilter("ignore", RuntimeWarning)
            off = (np.nanmedian if method == "median" else np.nanmean)(zm, axis=1)
        return z - np.nan_to_num(off)[:, None]
    if method == "median of differences":
        # ignore step edges (from the fast-axis gradient, which line offsets do not
        # affect): an edge that shifts slightly from line to line would otherwise
        # bias every difference a little and the offsets would drift
        g = np.gradient(ndimage.gaussian_filter1d(np.nan_to_num(z), 2.0, axis=1), axis=1)
        g = g - np.median(g)
        edge = np.abs(g) > 3 * (robust_sigma(g) or 1e-12)
        edge = ndimage.binary_dilation(edge, structure=np.ones((3, 9), bool))
        if (~edge).mean() > 0.3:
            zm = np.where(edge, np.nan, zm)
        with warnings.catch_warnings():  # all-NaN lines (fully masked) → offset 0
            warnings.simplefilter("ignore", RuntimeWarning)
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


SMOOTHING = ("none", "gaussian", "median")


def smooth(z, method="gaussian", size_nm=5.0, dx=1.0, dy=None):
    """Noise reduction. ``gaussian``: σ = size_nm (keeps terrace levels, broadens
    step edges by ~σ); ``median``: window of size_nm (keeps edges sharper, removes
    spikes). Smoothing lowers Sq – use it for step heights, not for roughness."""
    dy = dy or dx
    if method == "gaussian":
        return ndimage.gaussian_filter(z, (size_nm / dy, size_nm / dx), mode="nearest")
    if method == "median":
        size = (max(1, int(round(size_nm / dy)) | 1), max(1, int(round(size_nm / dx)) | 1))
        return ndimage.median_filter(z, size=size, mode="nearest")
    if method in ("none", None, ""):
        return z
    raise ValueError(method)


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
    smooth: str = "none"            # 'none', 'gaussian', 'median'
    smooth_nm: float = 0.0          # Gaussian σ or median window (nm)
    zero: str = "median"
    terrace_bow: int = 2            # polynomial bow order for 'terraces' (0 = off, 2, 3)


@dataclass
class Corrected:
    z: np.ndarray
    mask: np.ndarray                # features excluded from the fits
    scars: np.ndarray
    outliers: np.ndarray
    log: list


def correct(z, c: Corrections = Corrections(), dx=None, dy=None) -> Corrected:
    """Standard order: background + line alignment → scars (detected on the levelled
    image, where line offsets no longer hide them) → background + line alignment
    again with a mask of pits/particles → spikes → smoothing → zero.
    ``dx``, ``dy`` (nm/pixel) are needed for smoothing only."""
    z = np.array(z, float)
    log = []
    info = {}

    def level(z, mask):
        if c.background == "plane":
            z = level_polynomial(z, 1, mask)
        elif c.background.startswith("polynomial"):
            z = level_polynomial(z, int(c.background.split()[-1]), mask)
        elif c.background.startswith("terraces"):
            bow = int(c.terrace_bow or 0)
            if c.rows and c.rows != "none":
                # line offsets would otherwise look like step edges in dz/dy
                z = align_rows(z, c.rows, mask)
            if c.rows == "median of differences":
                # final line offsets referenced to the terraces (no random walk)
                return level_terraces(z, mask, fit_rows=True, bow_order=bow, info=info)
            z = level_terraces(z, mask, bow_order=bow, info=info)
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
        bow = int(c.terrace_bow or 0) if c.background.startswith("terraces") else 0
        log.append(f"background: {c.background}" + (f", terrace bow order {bow}" if bow >= 2 else ""))
        if bow >= 2 and info.get("bow_skipped"):
            log.append("terrace bow skipped: step edges not resolved (terraces merge)")
    if c.rows and c.rows != "none":
        log.append(f"line alignment: {c.rows}")
    outl = np.zeros(z.shape, bool)
    if c.outliers:
        z, outl = remove_outliers(z, c.outlier_sigma)
        log.append(f"outliers: {int(outl.sum())} px")
    if c.smooth != "none" and c.smooth_nm > 0 and dx:
        z = smooth(z, c.smooth, c.smooth_nm, dx, dy or dx)
        log.append(f"smoothing: {c.smooth}, {c.smooth_nm:g} nm")
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


# --------------------------------------------------------------------------- steps


def crop(z, dx, dy, roi):
    """Sub-image for roi = (x0, x1, y0, y1) in nm (y from the top row)."""
    x0, x1 = sorted(roi[:2])
    y0, y1 = sorted(roi[2:])
    j0, j1 = int(np.floor(x0 / dx)), int(np.ceil(x1 / dx))
    i0, i1 = int(np.floor(y0 / dy)), int(np.ceil(y1 / dy))
    ny, nx = z.shape
    return z[max(0, i0):min(ny, i1), max(0, j0):min(nx, j1)]


def _gauss_sum(x, *p):
    y = np.full_like(x, p[-1])
    for k in range(0, len(p) - 1, 3):
        a, c, w = p[k:k + 3]
        y = y + a * np.exp(-0.5 * ((x - c) / w) ** 2)
    return y


@dataclass
class LevelFit:
    centers: np.ndarray
    errors: np.ndarray        # 1σ fit uncertainty of the centres
    widths: np.ndarray        # Gaussian σ of each level (terrace roughness + noise)
    amplitudes: np.ndarray
    steps: np.ndarray
    step_errors: np.ndarray
    model: np.ndarray         # fitted histogram
    ladder: dict              # equidistant fit: step, error, n per level


def fit_levels(hist: Histogram, reference_step=None) -> LevelFit | None:
    """Fit a sum of Gaussians (one per detected level) to the height histogram.
    The centres are the terrace heights; neighbouring differences are the step
    heights. If ``reference_step`` is given, every level is assigned an integer
    number of steps n (rounded) and a weighted straight line L = L0 + n·d gives one
    mean step height d from all levels (robust against a missing level)."""
    from scipy.optimize import curve_fit

    x, y = hist.centers, hist.counts.astype(float)
    if len(hist.levels) == 0:
        return None
    bw = x[1] - x[0]
    sep = np.min(np.diff(hist.levels)) if len(hist.levels) > 1 else (x[-1] - x[0]) / 4
    p0, lo, hi = [], [], []
    for lv in hist.levels:
        a0 = float(np.interp(lv, x, hist.smooth))
        p0 += [a0, lv, max(sep / 4, 2 * bw)]
        lo += [0, lv - sep / 2, bw / 2]
        hi += [np.inf, lv + sep / 2, sep * 2]
    p0.append(0.0)
    lo.append(0.0)
    hi.append(max(y.max(), 1.0))
    sig = np.sqrt(np.maximum(y, 1.0))
    p0 = np.clip(p0, np.array(lo) + 1e-12, np.where(np.isfinite(hi), np.array(hi) - 1e-12, np.inf))
    try:
        p, cov = curve_fit(_gauss_sum, x, y, p0=p0, bounds=(lo, hi), sigma=sig, maxfev=50000,
                           x_scale="jac")
        err = np.sqrt(np.clip(np.diag(cov), 0, None))
    except Exception:
        # fall back to independent fits of each peak within ± half the level spacing
        p, err = [], []
        for k, lv in enumerate(hist.levels):
            sel = np.abs(x - lv) < sep / 2
            try:
                pk, ck = curve_fit(lambda t, a, c, w, b: a * np.exp(-0.5 * ((t - c) / w) ** 2) + b,
                                   x[sel], y[sel], p0=[p0[3 * k], lv, p0[3 * k + 2], 0.0],
                                   sigma=sig[sel], maxfev=20000)
                ek = np.sqrt(np.clip(np.diag(ck), 0, None))
            except Exception:
                return None
            p += list(pk[:3])
            err += list(ek[:3])
        p = np.array(p + [0.0])
        err = np.array(err + [0.0])
    centers = p[1:-1:3]
    order = np.argsort(centers)
    centers = centers[order]
    cerr = err[1:-1:3][order]
    widths = np.abs(p[2:-1:3][order])
    amps = p[0:-1:3][order]
    steps = np.diff(centers)
    serr = np.hypot(cerr[1:], cerr[:-1])
    ladder = _ladder_fit(centers, cerr, reference_step, floor=bw / 10)
    return LevelFit(centers, cerr, widths, amps, steps, serr, _gauss_sum(x, *p), ladder)


def _ladder_fit(centers, cerr, reference_step, floor=1e-6) -> dict:
    """Weighted straight line L = L0 + n·d over all levels (n = integer number of
    reference steps). Centre errors are floored (a fit at a bound can return 0 or
    inf/NaN), the normal equations are solved with a pseudo-inverse; on any failure
    an empty dict is returned instead of raising."""
    if not reference_step or len(centers) < 2:
        return {}
    try:
        n = np.round((centers - centers[0]) / reference_step).astype(int)
        if len(np.unique(n)) < 2:
            return {}
        e = np.asarray(cerr, float).copy()
        e[~np.isfinite(e)] = np.nan
        typical = np.nanmedian(e) if np.isfinite(e).any() else floor
        e = np.where(np.isfinite(e), e, max(typical, floor) * 10)
        e = np.maximum(e, max(floor, 1e-6))
        w = 1 / e ** 2
        A = np.c_[np.ones_like(n, float), n]
        W = np.diag(w)
        cov_l = np.linalg.pinv(A.T @ W @ A)
        coef = cov_l @ A.T @ W @ centers
        res = centers - A @ coef
        dof = len(centers) - 2
        scale = max(1.0, float(res @ W @ res) / dof) if dof > 0 else 1.0
        step, err = float(coef[1]), float(np.sqrt(max(cov_l[1, 1], 0.0) * scale))
        if not (np.isfinite(step) and np.isfinite(err)):
            return {}
        return {"step": step, "error": err, "n": n.tolist()}
    except Exception:
        return {}


@dataclass
class ProfileSteps:
    edges: np.ndarray         # positions of the step edges along the profile (nm)
    heights: np.ndarray       # signed step heights (rise along the profile)
    errors: np.ndarray        # 1σ (fit covariance scaled by the residual)
    slope: float              # residual slope along the profile (height unit / nm)
    terraces: np.ndarray      # terrace widths between neighbouring edges (nm)
    model: np.ndarray         # fitted staircase on the profile grid
    used: np.ndarray          # points used in the fit (edge zones excluded)
    residual_rms: float


def _staircase_fit(s, h, edges, edge_exclusion_nm, curvature=False, exclude=()):
    used = np.ones_like(s, bool)
    for a, b in exclude:  # pits / islands crossed by the profile
        used &= (s < a - edge_exclusion_nm / 2) | (s > b + edge_exclusion_nm / 2)
    for e in edges:
        used &= np.abs(s - e) > edge_exclusion_nm / 2
    u = s - s.mean()
    cols = [np.ones_like(s), u] + ([u ** 2 / max(abs(u).max(), 1e-12)] if curvature else [])         + [(s > e).astype(float) for e in edges]
    A = np.stack(cols, axis=1)
    if used.sum() <= A.shape[1]:
        used[:] = True
    coef, *_ = np.linalg.lstsq(A[used], h[used], rcond=None)
    res = h[used] - A[used] @ coef
    dof = max(1, used.sum() - A.shape[1])
    s2 = float(res @ res) / dof
    cov = s2 * np.linalg.pinv(A[used].T @ A[used])
    err = np.sqrt(np.clip(np.diag(cov), 0, None))
    return coef, err, A, used, res


def profile_steps(s, h, min_step=None, edge_exclusion_nm=20.0, smooth_nm=0.0,
                  reference_step=0.3905, curvature=False, drop_pairs=True) -> ProfileSteps:
    """Staircase fit of a (wide, averaged) profile perpendicular to the steps.

    Edges are located where the height changes by more than ``min_step``/2 within
    ``edge_exclusion_nm`` (default ``min_step`` = half the reference step). Then
    h(s) = a + b·s + Σ_k H_k·Θ(s − e_k) is fitted by least squares to all points
    farther than ``edge_exclusion_nm``/2 from an edge, so a residual tilt along the
    profile (b) does not bias the step heights H_k; ``curvature`` adds a quadratic
    term for a residual bow along the profile. Candidate edges whose fitted height
    is below ``min_step``/2 (noise, roughness) are dropped and the fit repeated.
    ``drop_pairs``: two neighbouring edges of opposite sign that return to the same
    level (|H1 + H2| < min_step/2, closer than 3 × edge_exclusion_nm) are a pit or
    an island crossed by the profile, not two steps – they are removed as well."""
    if min_step is None:
        min_step = 0.5 * reference_step
    s = np.asarray(s, float)
    h = np.asarray(h, float)
    ds = s[1] - s[0] if len(s) > 1 else 1.0
    hs = ndimage.gaussian_filter1d(h, smooth_nm / ds) if smooth_nm > 0 else h
    half = max(1, int(round(edge_exclusion_nm / 2 / ds)))
    D = np.full_like(hs, 0.0)
    if len(hs) > 2 * half:
        D[half:-half] = hs[2 * half:] - hs[:-2 * half]
    pk, _ = signal.find_peaks(np.abs(D), height=0.5 * min_step, distance=max(1, 2 * half))
    # an edge needs a usable plateau on both sides (not just the profile end); the
    # profile beyond an edge too close to an end is cut off – left in, the unmodelled
    # step would tilt the fitted slope b and bias all step heights
    exclude = []
    near_lo = pk[s[pk] - s[0] <= edge_exclusion_nm]
    near_hi = pk[s[-1] - s[pk] <= edge_exclusion_nm]
    if len(near_lo):
        exclude.append((s[0] - edge_exclusion_nm, s[near_lo].max()))
    if len(near_hi):
        exclude.append((s[near_hi].min(), s[-1] + edge_exclusion_nm))
    pk = pk[(s[pk] - s[0] > edge_exclusion_nm) & (s[-1] - s[pk] > edge_exclusion_nm)]
    edges = s[pk]
    k0 = 3 if curvature else 2
    while True:
        coef, err, A, used, res = _staircase_fit(s, h, edges, edge_exclusion_nm, curvature, exclude)
        H = coef[k0:]
        drop = np.abs(H) < 0.5 * min_step
        if drop_pairs and not drop.any():
            for i in range(len(H) - 1):
                if (H[i] * H[i + 1] < 0 and abs(H[i] + H[i + 1]) < 0.5 * min_step
                        and edges[i + 1] - edges[i] < 3 * edge_exclusion_nm):
                    drop[i] = drop[i + 1] = True
                    exclude.append((edges[i], edges[i + 1]))
                    break
        if not drop.any():
            break
        edges = edges[~drop]
    return ProfileSteps(edges=edges, heights=coef[k0:], errors=err[k0:], slope=float(coef[1]),
                        terraces=np.diff(edges), model=A @ coef, used=used,
                        residual_rms=float(np.sqrt(np.mean(res ** 2))))


# --------------------------------------------------------------------------- step direction


@dataclass
class StepDirection:
    direction: tuple          # unit vector (ux, uy) ⟂ step edges, pointing uphill (x right, y down)
    angle_deg: float          # atan2(uy, ux)
    coherence: float          # (λ1 − λ2)/(λ1 + λ2) of the edge structure tensor (1 = straight steps)
    terrace_nm: float         # mean terrace width = reference step / mean slope (NaN if flat)
    miscut_deg: float         # atan(mean slope along the direction)


def _chord(cx, cy, ux, uy, W, H):
    """Parameter range t of the line (cx, cy) + t·(ux, uy) inside [0, W] × [0, H]."""
    lo, hi = -np.inf, np.inf
    for c, u, L in ((cx, ux, W), (cy, uy, H)):
        if abs(u) < 1e-12:
            if not 0 <= c <= L:
                return 0.0, 0.0
            continue
        a, b = (0 - c) / u, (L - c) / u
        lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
    return (lo, hi) if hi > lo else (0.0, 0.0)


def step_direction(z, dx, dy, smooth_nm=None, edge_sigma=3.0, reference_step=0.3905,
                   line_filter=7) -> StepDirection:
    """Direction perpendicular to the step edges and the mean terrace width of a
    facet-levelled image (terraces horizontal).

    * Direction: structure tensor J = Σ ∇z ∇zᵀ over the step-edge pixels of a
      smoothed copy. The gradient is high-pass filtered first (a slowly varying
      slope – bow, waviness – would otherwise dominate) and the image is median
      filtered over ``line_filter`` lines along the slow axis, which removes
      residual line offsets (they would show up as edges along x) but keeps step
      edges, which extend over many lines. The main eigenvector is the step normal,
      oriented uphill (sign of the mean slope).
    * Terrace width: with horizontal terraces the mean slope along the normal is
      step height / terrace width, so T = reference_step / mean slope. This needs no
      edge detection and is not affected by pits (down + up cancel). Note: the
      *median* gradient would be the (zero) terrace slope, not the miscut."""
    z = np.asarray(z, float)
    ny, nx = z.shape
    dy = dy or dx
    sm = smooth_nm or 2.0 * max(dx, dy)
    z0 = np.nan_to_num(z)
    if line_filter and line_filter > 1 and ny > 3 * line_filter:
        z0 = ndimage.median_filter(z0, size=(int(line_filter), 1), mode="nearest")
    zs = ndimage.gaussian_filter(z0, (sm / dy, sm / dx), mode="nearest")
    gy0, gx0 = np.gradient(zs, dy, dx)
    big = max(ny, nx) / 12
    gx = gx0 - ndimage.gaussian_filter(gx0, big, mode="nearest")
    gy = gy0 - ndimage.gaussian_filter(gy0, big, mode="nearest")
    mag = np.hypot(gx, gy)
    edge = mag > np.median(mag) + edge_sigma * (robust_sigma(mag) or 1e-12)
    if edge.sum() < 20:
        edge = mag > np.percentile(mag, 90)
    jxx, jyy, jxy = (float(np.sum(v[edge])) for v in (gx * gx, gy * gy, gx * gy))
    lam, vec = np.linalg.eigh(np.array([[jxx, jxy], [jxy, jyy]]))
    ux, uy = vec[:, 1]
    mean_slope = float(np.mean(gx0) * ux + np.mean(gy0) * uy)
    if mean_slope < 0:
        ux, uy, mean_slope = -ux, -uy, -mean_slope
    coh = float((lam[1] - lam[0]) / (lam[1] + lam[0])) if lam.sum() > 0 else 0.0
    terrace = reference_step / mean_slope if mean_slope > 0 and reference_step else np.nan
    return StepDirection((float(ux), float(uy)), float(np.degrees(np.arctan2(uy, ux))), coh,
                         float(terrace), float(np.degrees(np.arctan(mean_slope))))


@dataclass
class AutoProfile:
    p0: tuple                 # (x, y) nm
    p1: tuple
    width_px: int             # averaging width along the steps
    steps: StepDirection


def auto_profile(z, dx, dy, fraction=0.8, width_factor=0.4, min_width=3, max_width=None,
                 sd: StepDirection | None = None) -> AutoProfile:
    """Profile line through the image centre perpendicular to the steps, covering
    ``fraction`` of the chord inside the image, and the averaging width
    clamp(width_factor · terrace width / pixel, min_width, max_width) – wide enough
    to average out noise, narrow enough that meandering edges are not smeared."""
    z = np.asarray(z, float)
    ny, nx = z.shape
    dy = dy or dx
    sd = sd or step_direction(z, dx, dy)
    ux, uy = sd.direction
    W, H = nx * dx, ny * dy
    cx, cy = W / 2, H / 2
    t0, t1 = _chord(cx, cy, ux, uy, W, H)
    half = 0.5 * fraction * (t1 - t0)
    p0 = (cx - half * ux, cy - half * uy)
    p1 = (cx + half * ux, cy + half * uy)
    max_width = max_width or max(min_width, int(0.1 * min(nx, ny)))
    if np.isfinite(sd.terrace_nm):
        w = int(round(width_factor * sd.terrace_nm / min(dx, dy)))
    else:
        w = int(round(0.02 * min(nx, ny)))
    return AutoProfile(p0, p1, int(np.clip(w, min_width, max_width)), sd)
