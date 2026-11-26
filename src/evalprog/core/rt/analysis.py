"""R(T) analysis: smoothing, derivatives, superconducting and magnetic transitions,
metal-insulator transitions, RRR, activation energies and geometry conversion."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq
from scipy.signal import find_peaks, savgol_filter

K_B_EV = 8.617333262e-5


# --------------------------------------------------------------------------
# Preparation
# --------------------------------------------------------------------------
def split_branches(T, y, time=None, min_points=10):
    """Split a measurement into cooling and warming parts (in acquisition order).

    Returns a dict {'cooling': (T, y), 'warming': (T, y)} with only the branches that
    contain at least ``min_points`` points.
    """
    T = np.asarray(T, float)
    y = np.asarray(y, float)
    if time is not None:
        o = np.argsort(np.asarray(time, float), kind="stable")
        T, y = T[o], y[o]
    ok = np.isfinite(T) & np.isfinite(y)
    T, y = T[ok], y[ok]
    if T.size < 3:
        return {}
    # smoothed direction of temperature change
    n = max(5, T.size // 200)
    Ts = np.convolve(T, np.ones(n) / n, mode="same")
    dT = np.gradient(Ts)
    out = {}
    for name, m in (("cooling", dT < 0), ("warming", dT > 0)):
        if m.sum() >= min_points:
            out[name] = (T[m], y[m])
    if not out:
        out["all"] = (T, y)
    return out


def resample(T, y, step=None):
    """Average onto an equidistant temperature grid (bins of width ``step``)."""
    T = np.asarray(T, float)
    y = np.asarray(y, float)
    ok = np.isfinite(T) & np.isfinite(y)
    T, y = T[ok], y[ok]
    o = np.argsort(T)
    T, y = T[o], y[o]
    if T.size < 4:
        raise ValueError("too few points")
    if step is None:
        d = np.diff(T)
        d = d[d > 0]
        step = max(float(np.median(d)) if d.size else 0.01, (T[-1] - T[0]) / 5000, 1e-4)
    edges = np.arange(T[0], T[-1] + step, step)
    idx = np.digitize(T, edges) - 1
    cnt = np.bincount(idx, minlength=edges.size)
    sy = np.bincount(idx, weights=y, minlength=edges.size)
    st = np.bincount(idx, weights=T, minlength=edges.size)
    m = cnt > 0
    Tb, yb = st[m] / cnt[m], sy[m] / cnt[m]
    grid = np.arange(Tb[0], Tb[-1] + step / 2, step)
    return grid, np.interp(grid, Tb, yb), step


def derivatives(T, y, window_K=1.0, step=None):
    """Savitzky-Golay smoothed y, dy/dT and d2y/dT2 on an equidistant grid."""
    g, yg, step = resample(T, y, step)
    n = max(int(round(window_K / step)) | 1, 5)
    if n >= g.size:
        n = g.size - 1 if (g.size - 1) % 2 == 1 else g.size - 2
    if n < 3:
        raise ValueError("too few points for the derivative window")
    poly = 3 if n > 5 else 2
    ys = savgol_filter(yg, n, poly)
    d1 = savgol_filter(yg, n, poly, deriv=1, delta=step)
    d2 = savgol_filter(yg, n, poly, deriv=2, delta=step)
    return g, ys, d1, d2


# --------------------------------------------------------------------------
# Superconductivity
# --------------------------------------------------------------------------
@dataclass
class SCResult:
    t_onset: float            # R = onset_frac * R_n
    t_mid: float              # R = 0.5 R_n
    t_low: float              # R = low_frac * R_n (used for delta_tc)
    t_zero: float             # highest T below which R < zero_frac * R_n
    t_onset_tangent: float    # intersection of normal-state line and tangent at T_mid
    delta_tc: float           # t_onset - t_low
    t_peak_deriv: float       # maximum of dR/dT
    r_normal: float           # R_n at t_onset
    normal_fit: tuple         # (slope, intercept) of the normal-state line
    normal_range: tuple
    fractions: dict = field(default_factory=dict)


def _crossing_from_above(T, y, level):
    """Highest temperature where y crosses ``level`` (y(T) - level changes sign),
    searched from high to low T; linear interpolation."""
    f = y - level
    for i in range(len(T) - 1, 0, -1):
        if f[i] >= 0 and f[i - 1] < 0:
            return T[i - 1] + (T[i] - T[i - 1]) * (0 - f[i - 1]) / (f[i] - f[i - 1])
    return float("nan")


def superconducting_transition(T, R, *, window_K=0.5, onset_frac=0.9, low_frac=0.1,
                               zero_frac=0.01, normal_range=None, t_range=None) -> SCResult:
    """Characteristic temperatures of a superconducting transition.

    The normal-state resistance R_n(T) is a straight line fitted above the transition
    (``normal_range`` = (T1, T2) or automatic: from T_peak + 3*width over a span of
    max(3*width, 5 K)). All fractions refer to R_n extrapolated to the temperature.
    """
    T = np.asarray(T, float)
    R = np.asarray(R, float)
    if t_range is not None:
        m = (T >= t_range[0]) & (T <= t_range[1])
        T, R = T[m], R[m]
    g, Rs, d1, _ = derivatives(T, R, window_K)
    i_pk = int(np.argmax(d1))
    t_pk = g[i_pk]
    # width of the transition from the dR/dT peak (half maximum)
    half = d1[i_pk] / 2
    l, r = i_pk, i_pk
    while l > 0 and d1[l] > half:
        l -= 1
    while r < d1.size - 1 and d1[r] > half:
        r += 1
    width = max(g[r] - g[l], 2 * (g[1] - g[0]))
    if normal_range is None:
        t1 = t_pk + 3 * width
        t2 = t1 + max(3 * width, 5.0)
        if t1 >= g[-1] - 2 * (g[1] - g[0]):
            # not enough normal state measured: use the top 10 % of the data
            t1 = g[-1] - 0.1 * (g[-1] - g[0])
            t2 = g[-1]
        normal_range = (t1, min(t2, g[-1]))
    mn = (g >= normal_range[0]) & (g <= normal_range[1])
    if mn.sum() >= 3:
        slope, icpt = np.polyfit(g[mn], Rs[mn], 1)
    else:
        slope, icpt = 0.0, float(np.max(Rs))
    Rn = slope * g + icpt
    ratio = Rs / Rn
    lo_mask = g <= normal_range[0]
    gg, rr = g[lo_mask], ratio[lo_mask]
    t_on = _crossing_from_above(gg, rr, onset_frac)
    t_mid = _crossing_from_above(gg, rr, 0.5)
    t_low = _crossing_from_above(gg, rr, low_frac)
    t_zero = _crossing_from_above(gg, rr, zero_frac)
    # tangent at T_mid
    if np.isfinite(t_mid):
        s_mid = np.interp(t_mid, g, d1)
        r_mid = np.interp(t_mid, g, Rs)
        # line: r_mid + s_mid (T - t_mid) = slope T + icpt
        den = s_mid - slope
        t_tan = (icpt - r_mid + s_mid * t_mid) / den if abs(den) > 1e-30 else float("nan")
    else:
        t_tan = float("nan")
    rn_on = slope * t_on + icpt if np.isfinite(t_on) else float("nan")
    return SCResult(t_onset=t_on, t_mid=t_mid, t_low=t_low, t_zero=t_zero, t_onset_tangent=t_tan,
                    delta_tc=t_on - t_low, t_peak_deriv=t_pk, r_normal=rn_on,
                    normal_fit=(float(slope), float(icpt)), normal_range=tuple(normal_range),
                    fractions={"onset": onset_frac, "low": low_frac, "zero": zero_frac})


# --------------------------------------------------------------------------
# Magnetic transitions and MIT
# --------------------------------------------------------------------------
@dataclass
class Feature:
    temperature: float
    kind: str          # e.g. 'dR/dT max', 'd2R/dT2 min', 'R min'
    value: float
    prominence: float


def transition_features(T, R, *, window_K=2.0, t_range=None, n_max=4) -> list:
    """Candidate transition temperatures from the derivatives.

    A ferromagnetic transition in a metal (e.g. SrRuO3) shows up as a kink in R(T):
    a step in dR/dT and an extremum in d2R/dT2. Antiferromagnetic or charge-ordering
    transitions often give a peak in dR/dT. The most prominent extrema of dR/dT and
    d2R/dT2 and the resistance minima are returned, strongest first.
    """
    T = np.asarray(T, float)
    R = np.asarray(R, float)
    if t_range is not None:
        m = (T >= t_range[0]) & (T <= t_range[1])
        T, R = T[m], R[m]
    g, Rs, d1, d2 = derivatives(T, R, window_K)
    feats = []
    edge = max(int(round(window_K / (g[1] - g[0]))), 2)
    inner = slice(edge, g.size - edge)

    def add(sig, kind):
        s = sig[inner]
        if s.size < 3:
            return
        scale = np.ptp(s) or 1.0
        pk, props = find_peaks(s, prominence=0.05 * scale)
        for i, p in zip(pk, props["prominences"]):
            j = i + edge
            feats.append(Feature(float(g[j]), kind, float(sig[j]), float(p / scale)))

    add(d1, "dR/dT max")
    add(-d1, "dR/dT min")
    add(d2, "d2R/dT2 max")
    add(-d2, "d2R/dT2 min")
    # sign change of dR/dT: resistance minimum / maximum (metal-insulator crossover)
    s = np.sign(d1[inner])
    for i in np.flatnonzero(np.diff(s) != 0):
        j = i + edge
        t0 = g[j] - d1[j] * (g[j + 1] - g[j]) / (d1[j + 1] - d1[j])
        kind = "R min" if d1[j] < 0 < d1[j + 1] else "R max"
        feats.append(Feature(float(t0), kind, float(np.interp(t0, g, Rs)), 1.0))
    feats.sort(key=lambda f: -f.prominence)
    return feats[: max(n_max * 2, 6)]


@dataclass
class MITResult:
    t_mit: float          # extremum of d ln R / dT
    t_r_min: float        # resistance minimum (nan if none)
    branch: str = ""


def mit_analysis(T, R, *, window_K=2.0, branch="") -> MITResult:
    """Metal-insulator transition from the extremum of d(ln R)/dT (most negative
    slope on the insulating side) and the minimum of R."""
    T = np.asarray(T, float)
    R = np.asarray(R, float)
    m = R > 0
    g, ls, d1, _ = derivatives(T[m], np.log(R[m]), window_K)
    edge = max(int(round(window_K / (g[1] - g[0]))), 2)
    s = slice(edge, g.size - edge)
    t_mit = float(g[s][np.argmin(d1[s])]) if g[s].size else float("nan")
    sign = np.sign(d1[s])
    cross = np.flatnonzero((sign[:-1] < 0) & (sign[1:] > 0))
    t_min = float(g[s][cross[-1]]) if cross.size else float("nan")
    return MITResult(t_mit=t_mit, t_r_min=t_min, branch=branch)


# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------
def rrr(T, R, t_high=300.0, t_low=None, window=5.0):
    """R(t_high) / R(t_low). ``t_low`` defaults to the lowest measured temperature;
    values are averaged over +/- window/2."""
    T = np.asarray(T, float)
    R = np.asarray(R, float)
    ok = np.isfinite(T) & np.isfinite(R)
    T, R = T[ok], R[ok]
    if T.size == 0:
        return float("nan")
    if t_low is None:
        t_low = T.min() + window / 2
    hi = np.abs(T - t_high) <= window / 2
    lo = np.abs(T - t_low) <= window / 2
    if not hi.any() or not lo.any():
        return float("nan")
    r_lo = R[lo].mean()
    return float(R[hi].mean() / r_lo) if r_lo != 0 else float("nan")


def activation_energy(T, R, t_range):
    """Arrhenius fit ln R = ln R0 + Ea / (kB T) in ``t_range``. Returns Ea (meV), R0."""
    T = np.asarray(T, float)
    R = np.asarray(R, float)
    m = (T >= t_range[0]) & (T <= t_range[1]) & (R > 0)
    if m.sum() < 3:
        raise ValueError("too few points in range")
    slope, icpt = np.polyfit(1 / T[m], np.log(R[m]), 1)
    return float(slope * K_B_EV * 1e3), float(np.exp(icpt))


# --------------------------------------------------------------------------
# Geometry: resistance -> resistivity
# --------------------------------------------------------------------------
RHO_UNITS = {"Ohm cm": 1e2, "mOhm cm": 1e5, "uOhm cm": 1e8, "Ohm m": 1.0}


@dataclass
class Geometry:
    kind: str = "bar"           # 'bar' (Hall bar / 4-point bar), 'vdp' (van der Pauw), 'sheet'
    thickness_nm: float = 0.0
    length_um: float = 0.0      # distance between voltage contacts
    width_um: float = 0.0
    vdp_f: float = 1.0          # van der Pauw correction factor

    def factor_ohm_m(self) -> float:
        """k with rho [Ohm m] = R [Ohm] * k (or R_sheet [Ohm] for kind 'sheet' -> k=1)."""
        t = self.thickness_nm * 1e-9
        if self.kind == "bar":
            if t <= 0 or self.length_um <= 0 or self.width_um <= 0:
                raise ValueError("thickness, length and width must be > 0")
            return self.width_um * 1e-6 * t / (self.length_um * 1e-6)
        if self.kind == "vdp":
            if t <= 0:
                raise ValueError("thickness must be > 0")
            return np.pi * t / np.log(2) * self.vdp_f
        raise ValueError("sheet geometry has no resistivity")

    def sheet_factor(self) -> float:
        """k with R_sheet [Ohm/sq] = R * k."""
        if self.kind == "bar":
            return self.width_um / self.length_um
        if self.kind == "vdp":
            return np.pi / np.log(2) * self.vdp_f
        return 1.0


def resistivity(R, geom: Geometry, unit="uOhm cm"):
    return np.asarray(R, float) * geom.factor_ohm_m() * RHO_UNITS[unit]


def vdp_sheet_resistance(r_a, r_b) -> float:
    """Solve exp(-pi R_A / R_s) + exp(-pi R_B / R_s) = 1 for the sheet resistance."""
    r_a, r_b = float(r_a), float(r_b)
    if r_a <= 0 or r_b <= 0:
        raise ValueError("resistances must be positive")
    f = lambda rs: np.exp(-np.pi * r_a / rs) + np.exp(-np.pi * r_b / rs) - 1
    guess = np.pi / np.log(2) * (r_a + r_b) / 2
    return brentq(f, guess * 0.2, guess * 5)


def vdp_factor(r_a, r_b) -> float:
    """Correction factor f with R_s = pi/ln2 * (R_A + R_B)/2 * f."""
    return vdp_sheet_resistance(r_a, r_b) / (np.pi / np.log(2) * (r_a + r_b) / 2)
