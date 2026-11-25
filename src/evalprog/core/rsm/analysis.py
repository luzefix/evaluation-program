"""Reciprocal space maps: Q conversion, 2D peak fits, offset correction, strain.

Coordinates (coplanar geometry, film normal along z)::

    qx = k (cos w - cos(2t - w)) = -2k sin(t) sin(w - t)
    qz = k (sin w + sin(2t - w)) =  2k sin(t) cos(w - t)

with k = 2 pi / lambda, t = 2theta/2 and w = omega. Q is in 1/Angstrom
(including the 2 pi), i.e. d = 2 pi / |Q|.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from evalprog.core.xrd.io import CU_KA1, AreaScan
from evalprog.materials import Film, Substrate

LN2 = np.log(2.0)


def angles_to_q(two_theta, omega, wavelength=CU_KA1):
    t = np.radians(np.asarray(two_theta, float)) / 2
    w = np.radians(np.asarray(omega, float))
    k = 2 * np.pi / wavelength
    return k * (np.cos(w) - np.cos(2 * t - w)), k * (np.sin(w) + np.sin(2 * t - w))


def q_to_angles(qx, qz, wavelength=CU_KA1):
    k = 2 * np.pi / wavelength
    q = np.hypot(qx, qz)
    t = np.arcsin(np.clip(q / (2 * k), -1, 1))
    psi = np.arctan2(-np.asarray(qx, float), np.asarray(qz, float))
    return np.degrees(2 * t), np.degrees(t + psi)


@dataclass
class RSMap:
    two_theta: np.ndarray
    omega: np.ndarray
    intensity: np.ndarray
    wavelength: float = CU_KA1
    d_two_theta: float = 0.0   # applied corrections (deg)
    d_omega: float = 0.0
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_area(cls, area: AreaScan) -> "RSMap":
        return cls(area.two_theta.astype(float), area.omega.astype(float),
                   area.intensity.astype(float), wavelength=area.wavelength, meta=dict(area.meta))

    @property
    def qx(self):
        return angles_to_q(self.two_theta - self.d_two_theta, self.omega - self.d_omega,
                           self.wavelength)[0]

    @property
    def qz(self):
        return angles_to_q(self.two_theta - self.d_two_theta, self.omega - self.d_omega,
                           self.wavelength)[1]

    def q(self):
        return angles_to_q(self.two_theta - self.d_two_theta, self.omega - self.d_omega,
                           self.wavelength)


# --------------------------------------------------------------------------
# 2D peak model
# --------------------------------------------------------------------------
PARAM_NAMES = ["amp", "x0", "z0", "hx", "hz", "theta", "eta", "b0", "bx", "bz"]


def pv2d(x, z, amp, x0, z0, hx, hz, theta, eta, b0=0.0, bx=0.0, bz=0.0):
    """Rotated 2D pseudo-Voigt (hx, hz: half widths at half maximum) + plane background."""
    ct, st = np.cos(theta), np.sin(theta)
    dx, dz = x - x0, z - z0
    xr = ct * dx + st * dz
    zr = -st * dx + ct * dz
    r2 = (xr / hx) ** 2 + (zr / hz) ** 2
    peak = eta / (1 + r2) + (1 - eta) * np.exp(-LN2 * r2)
    return amp * peak + b0 + bx * dx + bz * dz


@dataclass
class Peak2D:
    qx: float
    qz: float
    qx_err: float
    qz_err: float
    amp: float
    hwhm_x: float
    hwhm_z: float
    theta_deg: float
    eta: float
    background: float
    r2: float
    rmse: float
    rel_rmse: float
    chi2_red: float
    weighting: str
    params: np.ndarray = field(repr=False, default=None)

    def model(self, x, z):
        return pv2d(x, z, *self.params)


def _moments(x, z, y, bg):
    w = np.clip(y - bg, 0, None)
    if w.sum() <= 0:
        return x[np.argmax(y)], z[np.argmax(y)], np.ptp(x) / 6, np.ptp(z) / 6, 0.0
    # use only the upper part of the peak for robust moments
    w = np.where(w > 0.2 * w.max(), w, 0)
    s = w.sum()
    mx, mz = (w * x).sum() / s, (w * z).sum() / s
    cxx = (w * (x - mx) ** 2).sum() / s
    czz = (w * (z - mz) ** 2).sum() / s
    cxz = (w * (x - mx) * (z - mz)).sum() / s
    ev, evec = np.linalg.eigh(np.array([[cxx, cxz], [cxz, czz]]))
    ev = np.clip(ev, 1e-12, None)
    # principal axis with larger variance -> first axis
    theta = float(np.arctan2(evec[1, 1], evec[0, 1]))
    theta = (theta + np.pi / 2) % np.pi - np.pi / 2
    return mx, mz, np.sqrt(ev[1]) * 1.18, np.sqrt(ev[0]) * 1.18, theta


def fit_peak_2d(x, z, y, *, weighting="poisson", fit_background_slope=True, robust=False) -> Peak2D:
    """Fit a rotated 2D pseudo-Voigt to scattered points.

    ``weighting``: 'poisson' (1/sqrt(I), default), 'log' (fit log10 I; weights the tails
    strongly), or 'none' (ordinary least squares as in the old Gaussian fit).
    """
    x, z, y = (np.asarray(a, float).ravel() for a in (x, z, y))
    ok = np.isfinite(x) & np.isfinite(z) & np.isfinite(y)
    x, z, y = x[ok], z[ok], y[ok]
    if x.size < 12:
        raise ValueError("ROI contains too few points")
    bg0 = float(np.percentile(y, 15))
    i_max = int(np.argmax(y))
    amp0 = max(y[i_max] - bg0, 1.0)
    mx, mz, hx0, hz0, th0 = _moments(x, z, y, bg0)
    span_x, span_z = np.ptp(x), np.ptp(z)
    hx0 = float(np.clip(hx0, span_x / 200, span_x))
    hz0 = float(np.clip(hz0, span_z / 200, span_z))
    p0 = np.array([amp0, x[i_max], z[i_max], hx0, hz0, th0, 0.5, bg0, 0.0, 0.0])
    big = max(abs(span_x), abs(span_z), 1e-6)
    lo = np.array([0, x.min(), z.min(), big * 1e-4, big * 1e-4, -np.pi / 2, 0, -np.inf, -np.inf, -np.inf])
    hi = np.array([np.inf, x.max(), z.max(), 2 * big, 2 * big, np.pi / 2, 1, np.inf, np.inf, np.inf])
    free = np.ones(10, bool)
    if not fit_background_slope:
        free[8:] = False

    def full(pf):
        p = p0.copy()
        p[free] = pf
        return p

    if weighting == "log":
        floor = max(np.percentile(y[y > 0], 2) if np.any(y > 0) else 1.0, 1e-6) * 0.5

        def resid(pf):
            m = pv2d(x, z, *full(pf))
            return np.log10(np.maximum(m, floor)) - np.log10(np.maximum(y, floor))
        sig = None
    else:
        sig = np.sqrt(np.maximum(y, 1.0)) if weighting == "poisson" else np.ones_like(y)

        def resid(pf):
            return (pv2d(x, z, *full(pf)) - y) / sig

    p_start = np.clip(p0[free], lo[free] + 1e-12, hi[free] - 1e-12)
    res = least_squares(resid, p_start, bounds=(lo[free], hi[free]), x_scale="jac",
                        loss="soft_l1" if robust else "linear", max_nfev=20000)
    p = full(res.x)
    y_fit = pv2d(x, z, *p)
    rr = y - y_fit
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - float(np.sum(rr ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(rr ** 2)))
    dof = max(x.size - free.sum(), 1)
    chi2_red = float(np.sum(res.fun ** 2)) / dof
    try:
        cov = np.linalg.pinv(res.jac.T @ res.jac) * chi2_red
        perr_free = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        perr_free = np.full(free.sum(), np.nan)
    perr = np.full(10, np.nan)
    perr[free] = perr_free
    return Peak2D(qx=float(p[1]), qz=float(p[2]), qx_err=float(perr[1]), qz_err=float(perr[2]),
                  amp=float(p[0]), hwhm_x=float(abs(p[3])), hwhm_z=float(abs(p[4])),
                  theta_deg=float(np.degrees(p[5])), eta=float(p[6]), background=float(p[7]),
                  r2=r2, rmse=rmse, rel_rmse=rmse / float(p[0]) if p[0] > 0 else float("nan"),
                  chi2_red=chi2_red, weighting=weighting, params=p)


def roi_points(rsm: RSMap, roi, subtract=()):
    """Points inside the ROI. ``subtract``: fitted peaks (e.g. the substrate) whose
    peak part (without background) is removed from the intensity first."""
    qx, qz = rsm.q()
    x1, x2, z1, z2 = roi
    m = (qx >= min(x1, x2)) & (qx <= max(x1, x2)) & (qz >= min(z1, z2)) & (qz <= max(z1, z2))
    y = rsm.intensity[m].astype(float)
    for pk in subtract:
        p = np.array(pk.params, float)
        p[7:] = 0.0
        y = y - pv2d(qx[m], qz[m], *p)
    return qx[m], qz[m], y


def fit_roi(rsm: RSMap, roi, subtract=(), **kw) -> Peak2D:
    return fit_peak_2d(*roi_points(rsm, roi, subtract), **kw)


# --------------------------------------------------------------------------
# Substrate reference and offset correction
# --------------------------------------------------------------------------
def expected_q(hkl, a_inplane, c_out, sign=1.0):
    """Expected (qx, qz) of a reflection (h k l) with in-plane spacing a and
    out-of-plane repeat c (both Angstrom)."""
    h, k, l = hkl
    qx = sign * 2 * np.pi * np.hypot(h, k) / a_inplane
    qz = 2 * np.pi * l / c_out
    return qx, qz


def substrate_expected_q(sub: Substrate, hkl, direction=None, sign=1.0):
    return expected_q(hkl, sub.inplane_spacing(direction), sub.d_oop, sign)


@dataclass
class OffsetResult:
    peak: Peak2D  # substrate peak, fitted in corrected coordinates
    qx_expected: float
    qz_expected: float
    d_two_theta: float
    d_omega: float


def find_substrate_peak(rsm: RSMap, q_expected, radius=0.08):
    """ROI around the strongest point near the expected substrate position."""
    qx, qz = rsm.q()
    y = rsm.intensity
    dist = np.hypot(qx - q_expected[0], qz - q_expected[1])
    m = dist < radius
    if not np.any(m):
        m = np.ones_like(y, bool)
    i = np.flatnonzero(m.ravel())[np.argmax(y.ravel()[m.ravel()])]
    x0, z0 = qx.ravel()[i], qz.ravel()[i]
    # ROI size from where the intensity drops to ~2 % around the maximum
    peak = y.ravel()[i]
    near = (np.hypot(qx - x0, qz - z0) < radius) & (y > 0.02 * peak)
    dx = max(np.ptp(qx[near]) / 2 if near.any() else 0, 0.004)
    dz = max(np.ptp(qz[near]) / 2 if near.any() else 0, 0.004)
    dx, dz = min(dx * 1.3, radius), min(dz * 1.3, radius)
    return (x0 - dx, x0 + dx, z0 - dz, z0 + dz)


def correct_offset(rsm: RSMap, sub: Substrate, hkl, direction=None, roi=None, **fit_kw) -> OffsetResult:
    """Fit the substrate peak and shift 2theta and omega so that it sits at the
    calculated position. Modifies ``rsm`` in place (d_two_theta, d_omega)."""
    rsm.d_two_theta = rsm.d_omega = 0.0
    qx, _ = rsm.q()
    sign = 1.0 if np.nanmedian(qx) >= 0 else -1.0
    q_exp = substrate_expected_q(sub, hkl, direction, sign)
    if roi is None:
        roi = find_substrate_peak(rsm, q_exp)
    pk = fit_roi(rsm, roi, **fit_kw)
    tt_m, om_m = q_to_angles(pk.qx, pk.qz, rsm.wavelength)
    tt_e, om_e = q_to_angles(*q_exp, rsm.wavelength)
    rsm.d_two_theta = float(tt_m - tt_e)
    rsm.d_omega = float(om_m - om_e)
    # refit in corrected coordinates so the model can be subtracted from film ROIs
    dq = (q_exp[0] - pk.qx, q_exp[1] - pk.qz)
    roi_c = (roi[0] + dq[0], roi[1] + dq[0], roi[2] + dq[1], roi[3] + dq[1])
    try:
        pk = fit_roi(rsm, roi_c, **fit_kw)
    except ValueError:
        pass
    return OffsetResult(peak=pk, qx_expected=q_exp[0], qz_expected=q_exp[1],
                        d_two_theta=rsm.d_two_theta, d_omega=rsm.d_omega)


# --------------------------------------------------------------------------
# Lattice parameters, strain, relaxation
# --------------------------------------------------------------------------
@dataclass
class StrainResult:
    a_inplane: float
    c_out: float
    a_inplane_err: float
    c_out_err: float
    eps_xx: float = float("nan")
    eps_zz: float = float("nan")
    relaxation: float = float("nan")         # vs bulk: 0 = fully strained, 1 = relaxed
    a_relaxed: float = float("nan")          # unstrained in-plane parameter from Poisson
    relaxation_poisson: float = float("nan")  # relaxation vs a_relaxed
    poisson_eff: float = float("nan")
    a_substrate: float = float("nan")


def lattice_from_q(qx, qz, hkl, qx_err=np.nan, qz_err=np.nan):
    h, k, l = hkl
    hip = np.hypot(h, k)
    a = 2 * np.pi * hip / abs(qx) if hip > 0 and abs(qx) > 1e-12 else np.nan
    c = 2 * np.pi * abs(l) / abs(qz) if l != 0 and abs(qz) > 1e-12 else np.nan
    a_err = a * abs(qx_err / qx) if np.isfinite(a) else np.nan
    c_err = c * abs(qz_err / qz) if np.isfinite(c) else np.nan
    return a, c, a_err, c_err


def strain_analysis(peak: Peak2D, hkl, film: Film | None = None, a_substrate=np.nan) -> StrainResult:
    a, c, ae, ce = lattice_from_q(peak.qx, peak.qz, hkl, peak.qx_err, peak.qz_err)
    res = StrainResult(a_inplane=a, c_out=c, a_inplane_err=ae, c_out_err=ce, a_substrate=a_substrate)
    if film is None:
        return res
    res.eps_xx = (a - film.a) / film.a
    res.eps_zz = (c - film.c) / film.c
    if np.isfinite(a_substrate) and abs(film.a - a_substrate) > 1e-6:
        res.relaxation = (a - a_substrate) / (film.a - a_substrate)
    denom = res.eps_zz - 2 * res.eps_xx
    if abs(denom) > 1e-9:
        res.poisson_eff = res.eps_zz / denom
    nu = film.poisson
    kfac = 2 * nu / (1 - nu)
    ratio = film.c / film.a
    res.a_relaxed = (c / ratio + kfac * a) / (1 + kfac)
    if np.isfinite(a_substrate) and abs(res.a_relaxed - a_substrate) > 1e-6:
        res.relaxation_poisson = (a - a_substrate) / (res.a_relaxed - a_substrate)
    return res


def line_cuts(rsm: RSMap, peak: Peak2D, width=None):
    """Intensity along qx (at qz ~ peak) and along qz (at qx ~ peak), data and model."""
    qx, qz = rsm.q()
    y = rsm.intensity
    wx = width or max(peak.hwhm_z, 1e-4)
    wz = width or max(peak.hwhm_x, 1e-4)
    mx = np.abs(qz - peak.qz) < wx * 0.5
    mz = np.abs(qx - peak.qx) < wz * 0.5
    if mx.sum() < 5:
        mx = np.abs(qz - peak.qz) < wx * 2
    if mz.sum() < 5:
        mz = np.abs(qx - peak.qx) < wz * 2
    ox = np.argsort(qx[mx])
    oz = np.argsort(qz[mz])
    xs, zs = qx[mx][ox], qz[mz][oz]
    # model evaluated exactly through the peak centre
    cut_x = (xs, y[mx][ox], peak.model(xs, np.full_like(xs, peak.qz)))
    cut_z = (zs, y[mz][oz], peak.model(np.full_like(zs, peak.qx), zs))
    return cut_x, cut_z
