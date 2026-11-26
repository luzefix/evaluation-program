"""Film reflections in symmetric scans: lattice parameter, strain, Laue fringes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import CU_KA1
from .peaks import d_from_two_theta


@dataclass
class FilmPeak:
    phase: str
    hkl: tuple
    two_theta: float        # corrected K-alpha1 position
    two_theta_err: float
    fwhm: float
    d: float
    lattice: float          # out-of-plane lattice parameter from this reflection
    lattice_err: float

    @property
    def hkl_label(self) -> str:
        return "({} {} {})".format(*self.hkl)


def lattice_from_peak(two_theta, hkl, wavelength=CU_KA1, model="00l", in_plane=None):
    """Out-of-plane lattice parameter from one reflection.

    ``model``: '00l' -> c = l * d;  'cubic' -> a = d * sqrt(h^2+k^2+l^2);
    'tetragonal' -> c from d and a known in-plane parameter ``in_plane``.
    """
    d = float(d_from_two_theta(two_theta, wavelength))
    h, k, l = hkl
    if model == "cubic":
        return d * np.sqrt(h * h + k * k + l * l), d
    if model == "tetragonal":
        if not in_plane or l == 0:
            raise ValueError("tetragonal model needs l != 0 and the in-plane parameter")
        inv = 1 / d ** 2 - (h * h + k * k) / in_plane ** 2
        if inv <= 0:
            raise ValueError("inconsistent in-plane parameter")
        return abs(l) / np.sqrt(inv), d
    if h or k or not l:
        raise ValueError("model '00l' needs a (0 0 l) reflection")
    return abs(l) * d, d


def make_film_peak(phase, hkl, two_theta, two_theta_err, fwhm, wavelength=CU_KA1, model="00l",
                   in_plane=None) -> FilmPeak:
    c, d = lattice_from_peak(two_theta, hkl, wavelength, model, in_plane)
    # dc/c = -cot(theta) dtheta
    th = np.radians(two_theta / 2)
    c_err = c / np.tan(th) * np.radians(two_theta_err / 2) if np.isfinite(two_theta_err) else np.nan
    return FilmPeak(phase=phase, hkl=tuple(hkl), two_theta=two_theta, two_theta_err=two_theta_err,
                    fwhm=fwhm, d=d, lattice=c, lattice_err=abs(c_err))


def combine_lattice(peaks) -> tuple[float, float]:
    """Weighted mean of the lattice parameters of several reflections of one phase."""
    vals = np.array([p.lattice for p in peaks], float)
    errs = np.array([p.lattice_err for p in peaks], float)
    if vals.size == 0:
        return float("nan"), float("nan")
    if vals.size == 1:
        return float(vals[0]), float(errs[0])
    good = np.isfinite(errs) & (errs > 0)
    if good.all():
        w = 1 / errs ** 2
        mean = float(np.sum(w * vals) / np.sum(w))
        err = max(float(np.sqrt(1 / np.sum(w))), float(np.std(vals, ddof=1) / np.sqrt(vals.size)))
        return mean, err
    return float(np.mean(vals)), float(np.std(vals, ddof=1) / np.sqrt(vals.size))


def out_of_plane_strain(c_measured, c_bulk) -> float:
    return (c_measured - c_bulk) / c_bulk


def thickness_from_fringes(two_theta_positions, wavelength=CU_KA1) -> tuple[float, float]:
    """Film thickness (Angstrom) from the positions of consecutive Laue fringes
    (maxima or minima). Uses 2 sin(theta)/lambda = n / t  ->  linear fit vs n."""
    tt = np.sort(np.asarray(two_theta_positions, float))
    if tt.size < 2:
        raise ValueError("need at least two fringe positions")
    q = 2 * np.sin(np.radians(tt / 2)) / wavelength
    n = np.arange(tt.size)
    A = np.column_stack([n, np.ones_like(n)])
    coef, res, *_ = np.linalg.lstsq(A, q, rcond=None)
    slope = coef[0]
    t = 1 / slope
    if tt.size > 2:
        resid = q - A @ coef
        s_err = np.sqrt(np.sum(resid ** 2) / (tt.size - 2) / np.sum((n - n.mean()) ** 2))
        t_err = s_err / slope ** 2
    else:
        t_err = float("nan")
    return float(t), float(t_err)
