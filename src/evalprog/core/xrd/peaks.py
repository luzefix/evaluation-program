"""1D peak profiles and fitting for symmetric 2theta-omega scans.

Each peak is a pseudo-Voigt (optionally split, i.e. different left/right widths)
with a Cu K-alpha2 satellite at the position given by Bragg's law and a fixed
intensity ratio. The background is linear. Fits use Poisson weights
(1/sqrt(I)) so that the tails and the peak top both count, which is what makes
substrate fits converge to the true centre instead of to the noise floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .io import CU_KA1, CU_KA2, KA2_RATIO

LN2 = np.log(2.0)


def bragg_two_theta(d, wavelength=CU_KA1):
    s = wavelength / (2.0 * np.asarray(d, float))
    with np.errstate(invalid="ignore"):
        return np.degrees(2.0 * np.arcsin(s))


def d_from_two_theta(tt, wavelength=CU_KA1):
    return wavelength / (2.0 * np.sin(np.radians(np.asarray(tt, float)) / 2.0))


def ka2_position(tt1, wl1=CU_KA1, wl2=CU_KA2):
    s = np.sin(np.radians(tt1) / 2.0) * wl2 / wl1
    return np.degrees(2.0 * np.arcsin(np.clip(s, -1, 1)))


def pseudo_voigt(x, center, height, fwhm, eta):
    dx = (x - center) / fwhm
    lor = 1.0 / (1.0 + 4.0 * dx * dx)
    gau = np.exp(-4.0 * LN2 * dx * dx)
    return height * (eta * lor + (1.0 - eta) * gau)


def split_pseudo_voigt(x, center, height, fwhm_l, fwhm_r, eta):
    w = np.where(x < center, fwhm_l, fwhm_r)
    return pseudo_voigt(x, center, height, w, eta)


@dataclass
class PeakResult:
    center: float            # K-alpha1 position (deg 2theta)
    center_err: float
    height: float
    fwhm: float              # mean FWHM (deg)
    eta: float
    fwhm_left: float | None = None
    fwhm_right: float | None = None
    area: float = float("nan")


@dataclass
class FitResult:
    peaks: list
    background: tuple        # (c0, c1) around x_ref
    x_ref: float
    x: np.ndarray
    y: np.ndarray
    y_fit: np.ndarray
    r_wp: float
    chi2_red: float
    r2: float
    success: bool
    message: str = ""
    options: dict = field(default_factory=dict)

    @property
    def main(self) -> PeakResult:
        """The first peak (the one the fit was started for)."""
        return self.peaks[0]

    def evaluate(self, x):
        return _model(np.asarray(x, float), self._params, self.options)

    _params: np.ndarray = field(default=None, repr=False)


# --------------------------------------------------------------------------
# Model with parameter vector
#   per peak: center, log(height), log(fwhm_l), log(fwhm_r) (split only), eta
#   then background c0, c1
# --------------------------------------------------------------------------
def _n_per_peak(split: bool) -> int:
    return 5 if split else 4


def _unpack(p, opts):
    npp = _n_per_peak(opts["split"])
    n = opts["n_peaks"]
    peaks = []
    for i in range(n):
        q = p[i * npp:(i + 1) * npp]
        if opts["split"]:
            c, lh, lwl, lwr, eta = q
            peaks.append((c, np.exp(lh), np.exp(lwl), np.exp(lwr), eta))
        else:
            c, lh, lw, eta = q
            peaks.append((c, np.exp(lh), np.exp(lw), np.exp(lw), eta))
    c0, c1 = p[n * npp:n * npp + 2]
    return peaks, (c0, c1)


def _model(x, p, opts):
    peaks, (c0, c1) = _unpack(p, opts)
    y = c0 + c1 * (x - opts["x_ref"])
    wl1, wl2, ratio = opts["wl1"], opts["wl2"], opts["ratio"]
    for c, h, wl, wr, eta in peaks:
        y = y + split_pseudo_voigt(x, c, h, wl, wr, eta)
        if wl2 and ratio > 0:
            c2 = ka2_position(c, wl1, wl2)
            y = y + split_pseudo_voigt(x, c2, h * ratio, wl, wr, eta)
    return y


def estimate_fwhm(x, y, i_max, bg):
    half = bg + (y[i_max] - bg) / 2.0
    left = i_max
    while left > 0 and y[left] > half:
        left -= 1
    right = i_max
    while right < len(y) - 1 and y[right] > half:
        right += 1
    w = x[right] - x[left]
    step = np.median(np.diff(x)) if len(x) > 1 else 0.01
    return max(w, 2 * step)


def fit_peaks(x, y, centers=None, *, split=False, wavelength=CU_KA1, wavelength2=CU_KA2,
              ka2_ratio=KA2_RATIO, fit_ka2=True, weights="poisson", center_bounds=None) -> FitResult:
    """Fit one or more K-alpha doublets plus a linear background.

    ``centers``: initial K-alpha1 positions; if None, the maximum is used.
    ``center_bounds``: optional (lo, hi) per peak.
    ``weights``: 'poisson' (1/sqrt(I)), 'log' (fit log I) or 'none'.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 6:
        raise ValueError("Too few points in fit range")
    order = np.argsort(x)
    x, y = x[order], y[order]
    bg0 = float(np.percentile(y, 10))
    if centers is None or len(centers) == 0:
        centers = [x[np.argmax(y)]]
    opts = dict(split=split, n_peaks=len(centers), x_ref=float(np.mean(x)), wl1=wavelength,
                wl2=wavelength2 if fit_ka2 else None, ratio=ka2_ratio if fit_ka2 else 0.0)

    p0, lo, hi = [], [], []
    span = x[-1] - x[0]
    step = float(np.median(np.diff(x)))
    for k, c in enumerate(centers):
        i = int(np.argmin(np.abs(x - c)))
        h = max(y[i] - bg0, 1e-6 * max(np.max(y), 1.0), 1e-12)
        w = estimate_fwhm(x, y, i, bg0)
        cb = center_bounds[k] if center_bounds else (x[0], x[-1])
        lw = np.log(min(w, span))
        p0 += [c, np.log(h)] + ([lw, lw] if split else [lw]) + [0.5]
        lo += [cb[0], np.log(h) - 12] + ([np.log(step / 4)] * (2 if split else 1)) + [0.0]
        hi += [cb[1], np.log(h) + 6] + ([np.log(span)] * (2 if split else 1)) + [1.0]
    p0 += [bg0, 0.0]
    lo += [-np.inf, -np.inf]
    hi += [np.inf, np.inf]
    p0 = np.clip(p0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)

    if weights == "poisson":
        sig = np.sqrt(np.maximum(y, 1.0))
    else:
        sig = np.ones_like(y)

    if weights == "log":
        floor = max(np.percentile(y[y > 0], 1) if np.any(y > 0) else 1.0, 1e-9) * 0.5

        def resid(p):
            return np.log(np.maximum(_model(x, p, opts), floor)) - np.log(np.maximum(y, floor))
    else:
        def resid(p):
            return (_model(x, p, opts) - y) / sig

    res = least_squares(resid, p0, bounds=(lo, hi), x_scale="jac", max_nfev=20000)
    p = res.x
    y_fit = _model(x, p, opts)

    # uncertainties from the Jacobian
    dof = max(x.size - p.size, 1)
    chi2 = float(np.sum(((y_fit - y) / sig) ** 2)) if weights != "log" else float(np.sum(res.fun ** 2))
    chi2_red = chi2 / dof
    try:
        jtj = res.jac.T @ res.jac
        cov = np.linalg.pinv(jtj) * chi2_red
        perr = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        perr = np.full_like(p, np.nan)

    peaks, bg = _unpack(p, opts)
    npp = _n_per_peak(split)
    results = []
    for k, (c, h, wl, wr, eta) in enumerate(peaks):
        fw = 0.5 * (wl + wr)
        # area of a pseudo-Voigt with height h
        area = h * fw * (eta * np.pi / 2 + (1 - eta) * np.sqrt(np.pi / (4 * LN2)))
        results.append(PeakResult(center=float(c), center_err=float(perr[k * npp]), height=float(h),
                                  fwhm=float(fw), eta=float(eta),
                                  fwhm_left=float(wl) if split else None,
                                  fwhm_right=float(wr) if split else None, area=float(area)))
    w2 = 1.0 / sig ** 2
    r_wp = float(np.sqrt(np.sum(w2 * (y - y_fit) ** 2) / np.sum(w2 * y ** 2)))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - y_fit) ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    fr = FitResult(peaks=results, background=bg, x_ref=opts["x_ref"], x=x, y=y, y_fit=y_fit,
                   r_wp=r_wp, chi2_red=chi2_red, r2=r2, success=bool(res.success),
                   message=res.message, options=opts)
    fr._params = p
    return fr


def bic(fr: FitResult) -> float:
    n = fr.x.size
    k = fr._params.size
    return n * np.log(max(fr.chi2_red * max(n - k, 1) / n, 1e-300)) + k * np.log(n)


def fit_with_extra_peak(x, y, main_center, *, max_extra=1, **kw) -> FitResult:
    """Fit the main doublet; add up to ``max_extra`` further peaks (e.g. a film peak or
    a shoulder inside the window) when this lowers the Bayesian information criterion."""
    best = fit_peaks(x, y, [main_center], **kw)
    for _ in range(max_extra):
        resid = best.y - best.y_fit
        # strongest residual away from the main peak
        far = np.abs(best.x - best.main.center) > 1.5 * best.main.fwhm
        if not np.any(far):
            break
        r = np.where(far, resid, -np.inf)
        c_new = best.x[int(np.argmax(r))]
        centers = [p.center for p in best.peaks] + [c_new]
        try:
            trial = fit_peaks(x, y, centers, **kw)
        except Exception:
            break
        if bic(trial) < bic(best) - 10:
            best = trial
        else:
            break
    return best
