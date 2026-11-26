"""Automatic substrate peak search, fit and 2theta correction.

Correction model (Bragg-Brentano geometry, goniometer radius R)::

    2theta_meas - 2theta_calc = Z + D * cos(theta)

``Z`` is the zero shift of the 2theta axis. ``D`` is the sample-displacement
term; a sample surface lying a height ``s`` above the goniometer axis gives
``D = -2 s / R`` (radians). With one substrate peak only ``Z`` is determined;
with two or more peaks both are fitted (least squares, weighted by the fit
uncertainties of the peak positions).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from evalprog.materials import Substrate

from .io import CU_KA1, CU_KA2, KA2_RATIO
from .peaks import FitResult, bragg_two_theta, fit_with_extra_peak, ka2_position

EMPYREAN_RADIUS_MM = 240.0


@dataclass
class SubstratePeak:
    label: str
    order: int
    d_calc: float
    tt_calc: float
    tt_meas: float
    tt_err: float
    fit: FitResult

    @property
    def delta(self) -> float:
        return self.tt_meas - self.tt_calc


@dataclass
class Correction:
    zero_shift: float = 0.0      # Z (deg)
    displacement: float = 0.0    # D (deg)
    radius_mm: float = EMPYREAN_RADIUS_MM
    zero_err: float = float("nan")
    displacement_err: float = float("nan")
    model: str = "none"          # 'none', 'zero', 'zero+height'
    residual_rms: float = float("nan")

    @property
    def sample_height_um(self) -> float:
        """Sample surface height above the goniometer axis (micrometre)."""
        return -np.radians(self.displacement) * self.radius_mm / 2.0 * 1e3

    def shift(self, two_theta):
        tt = np.asarray(two_theta, float)
        return self.zero_shift + self.displacement * np.cos(np.radians(tt) / 2.0)

    def apply(self, two_theta):
        """Corrected 2theta axis."""
        tt = np.asarray(two_theta, float)
        return tt - self.shift(tt)

    def describe(self) -> str:
        if self.model == "none":
            return "no correction"
        txt = f"Z = {self.zero_shift:+.4f} deg"
        if self.model == "zero+height":
            txt += (f", D = {self.displacement:+.4f} deg "
                    f"(sample height {self.sample_height_um:+.0f} um for R = {self.radius_mm:.0f} mm)")
        return txt


@dataclass
class SubstrateAnalysis:
    substrate: Substrate
    peaks: list = field(default_factory=list)
    correction: Correction = field(default_factory=Correction)
    skipped: list = field(default_factory=list)  # (label, reason)


def expected_peaks(substrate: Substrate, tt_min: float, tt_max: float, wavelength=CU_KA1):
    out = []
    for n, d, label in substrate.reflections():
        tt = float(bragg_two_theta(d, wavelength))
        if np.isfinite(tt) and tt_min <= tt <= tt_max:
            out.append((n, d, tt, label))
    return out


def _smooth(y, n=3):
    if n <= 1 or y.size < n:
        return y
    k = np.ones(n) / n
    return np.convolve(y, k, mode="same")


def find_and_fit(two_theta, intensity, substrate: Substrate, *, wavelength=CU_KA1,
                 wavelength2=CU_KA2, ka2_ratio=KA2_RATIO, search=1.0, split=True,
                 max_peaks=4, extra_peaks=1, min_prominence=5.0) -> SubstrateAnalysis:
    """Locate the substrate reflections inside the scan range and fit each one.

    ``search``: +/- window (deg) around the calculated position in which the maximum is
    searched. ``extra_peaks``: allow up to n additional peaks (film/shoulders) in the fit
    window. ``min_prominence``: peak maximum must exceed this factor times the local
    background, otherwise the reflection is skipped.
    """
    x = np.asarray(two_theta, float)
    y = np.asarray(intensity, float)
    res = SubstrateAnalysis(substrate=substrate)
    exp = expected_peaks(substrate, x.min() + 0.2, x.max() - 0.2, wavelength)
    ys = _smooth(y, 3)
    candidates = []
    for n, d, tt, label in exp:
        m = np.abs(x - tt) <= search
        if m.sum() < 5:
            res.skipped.append((label, "outside data"))
            continue
        idx = np.flatnonzero(m)
        i_max = idx[np.argmax(ys[idx])]
        wide = np.abs(x - tt) <= search + 2.0
        bg = float(np.percentile(y[wide], 10))
        if y[i_max] < min_prominence * max(bg, 1.0):
            res.skipped.append((label, "no clear peak"))
            continue
        candidates.append((y[i_max], n, d, tt, label, x[i_max]))
    # strongest max_peaks reflections, then sorted by angle
    candidates = sorted(candidates, key=lambda c: -c[0])[:max_peaks]
    candidates.sort(key=lambda c: c[3])

    step = float(np.median(np.diff(x)))
    for _, n, d, tt, label, x_peak in candidates:
        i = int(np.argmin(np.abs(x - x_peak)))
        # half-maximum width estimate
        half = y[i] / 2
        l, r = i, i
        while l > 0 and y[l] > half:
            l -= 1
        while r < y.size - 1 and y[r] > half:
            r += 1
        fw = max(x[r] - x[l], 3 * step)
        sep = ka2_position(x_peak, wavelength, wavelength2) - x_peak
        lo = x_peak - max(0.25, 8 * fw)
        hi = x_peak + sep + max(0.25, 8 * fw)
        m = (x >= lo) & (x <= hi)
        try:
            fr = fit_with_extra_peak(x[m], y[m], x_peak, max_extra=extra_peaks, split=split,
                                     wavelength=wavelength, wavelength2=wavelength2,
                                     ka2_ratio=ka2_ratio,
                                     center_bounds=None)
        except Exception as exc:  # pragma: no cover - defensive
            res.skipped.append((label, f"fit failed: {exc}"))
            continue
        pk = fr.main
        if abs(pk.center - x_peak) > 3 * max(fw, 0.02):
            # the fit has wandered to the extra peak: refit with single peak
            fr = fit_with_extra_peak(x[m], y[m], x_peak, max_extra=0, split=split,
                                     wavelength=wavelength, wavelength2=wavelength2,
                                     ka2_ratio=ka2_ratio)
            pk = fr.main
        err = pk.center_err if np.isfinite(pk.center_err) and pk.center_err > 0 else step / 10
        res.peaks.append(SubstratePeak(label=label, order=n, d_calc=d, tt_calc=tt,
                                       tt_meas=pk.center, tt_err=err, fit=fr))
    res.correction = fit_correction(res.peaks)
    return res


def fit_correction(peaks, radius_mm=EMPYREAN_RADIUS_MM, model="auto") -> Correction:
    """Fit Z (and D) from the measured minus calculated substrate positions."""
    if not peaks:
        return Correction(radius_mm=radius_mm)
    delta = np.array([p.delta for p in peaks])
    err = np.array([max(p.tt_err, 1e-5) for p in peaks])
    cos_t = np.cos(np.radians([p.tt_calc for p in peaks]) / 2.0)
    if model == "auto":
        model = "zero+height" if len(peaks) >= 2 else "zero"
    if model == "zero" or len(peaks) == 1:
        w = 1 / err ** 2
        z = float(np.sum(w * delta) / np.sum(w))
        z_err = float(np.sqrt(1 / np.sum(w))) if len(peaks) > 1 else float(err[0])
        rms = float(np.sqrt(np.mean((delta - z) ** 2)))
        return Correction(zero_shift=z, zero_err=z_err, model="zero", radius_mm=radius_mm,
                          residual_rms=rms)
    A = np.column_stack([np.ones_like(cos_t), cos_t])
    W = np.diag(1 / err ** 2)
    cov = np.linalg.pinv(A.T @ W @ A)
    coef = cov @ A.T @ W @ delta
    resid = delta - A @ coef
    dof = len(peaks) - 2
    if dof > 0:
        cov = cov * max(float(resid @ W @ resid) / dof, 1.0)
    return Correction(zero_shift=float(coef[0]), displacement=float(coef[1]),
                      zero_err=float(np.sqrt(cov[0, 0])), displacement_err=float(np.sqrt(cov[1, 1])),
                      model="zero+height", radius_mm=radius_mm,
                      residual_rms=float(np.sqrt(np.mean(resid ** 2))))
