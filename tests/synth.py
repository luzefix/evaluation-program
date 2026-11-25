"""Synthetic data files for tests and demos."""

import numpy as np

from evalprog.core.rsm.analysis import angles_to_q, expected_q, pv2d, q_to_angles
from evalprog.core.xrd.peaks import bragg_two_theta, ka2_position, pseudo_voigt
from evalprog.materials import get_film, get_substrate


def xrd_scan(substrate, z=0.05, height_um=50.0, film_c=3.95, seed=1):
    rng = np.random.default_rng(seed)
    x = np.arange(15, 110, 0.005)
    y = np.full_like(x, 20.0)
    D = np.degrees(-2 * height_um * 1e-3 / 240.0)
    shift = lambda tt: z + D * np.cos(np.radians(tt) / 2)
    for n, d, _ in substrate.reflections():
        tt = bragg_two_theta(d)
        if np.isfinite(tt) and tt < 108:
            c1 = tt + shift(tt)
            y += pseudo_voigt(x, c1, 2e5 / n, 0.02, 0.6) + pseudo_voigt(x, ka2_position(c1), 1e5 / n, 0.02, 0.6)
    if film_c:
        for l in (1, 2, 3, 4):
            tt = bragg_two_theta(film_c / l)
            if tt < 108:
                c1 = tt + shift(tt)
                y += pseudo_voigt(x, c1, 3000 / l, 0.15, 0.3) + pseudo_voigt(x, ka2_position(c1), 1500 / l, 0.15, 0.3)
    return x, rng.poisson(y).astype(float)


def write_xy(path, x, y):
    np.savetxt(path, np.column_stack([x, y]), fmt="%.4f")


def rsm_grid(d_tt=0.03, d_om=0.12, seed=2):
    sub = get_substrate("SrTiO3", "(001)")
    film = get_film("SrRuO3")
    hkl = (1, 0, 3)
    a_s = sub.inplane_spacing()
    nu = film.poisson
    c_f = film.c - 2 * nu / (1 - nu) * (a_s - film.a)
    qs = expected_q(hkl, a_s, sub.d_oop, sign=-1)
    qf = expected_q(hkl, a_s, c_f, sign=-1)
    tt0, om0 = q_to_angles(*qs)
    tt = np.arange(tt0 - 2.5, tt0 + 1.0, 0.01)
    offs = np.arange(-0.8, 0.8, 0.01)
    TT, OF = np.meshgrid(tt, offs)
    OM = TT / 2 + (om0 - tt0 / 2) + OF
    qx, qz = angles_to_q(TT, OM)
    I = pv2d(qx, qz, 1e5, *qs, 0.0015, 0.0015, 0.0, 0.5) + pv2d(qx, qz, 3e3, *qf, 0.004, 0.012, 0.0, 0.4) + 5
    I = np.random.default_rng(seed).poisson(I).astype(float)
    return TT + d_tt, OM + d_om, I, qf


def write_xrdml_map(path, TT, OM, I):
    scans = []
    for i in range(TT.shape[0]):
        scans.append(
            '<scan scanAxis="2Theta-Omega"><dataPoints>'
            f'<positions axis="2Theta" unit="deg"><startPosition>{TT[i, 0]:.5f}</startPosition>'
            f'<endPosition>{TT[i, -1]:.5f}</endPosition></positions>'
            f'<positions axis="Omega" unit="deg"><startPosition>{OM[i, 0]:.5f}</startPosition>'
            f'<endPosition>{OM[i, -1]:.5f}</endPosition></positions>'
            '<commonCountingTime unit="seconds">1</commonCountingTime>'
            f'<intensities unit="counts">{" ".join(str(int(v)) for v in I[i])}</intensities>'
            '</dataPoints></scan>')
    path.write_text('<?xml version="1.0"?><xrdMeasurements xmlns="http://www.xrdml.com/XRDMeasurement/1.5">'
                    '<xrdMeasurement><usedWavelength><kAlpha1 unit="Angstrom">1.5405980</kAlpha1>'
                    '<kAlpha2 unit="Angstrom">1.5444260</kAlpha2><ratioKAlpha2KAlpha1>0.5</ratioKAlpha2KAlpha1>'
                    '</usedWavelength>' + "".join(scans) + '</xrdMeasurement></xrdMeasurements>')


def write_qd_sc(path, tc=30.0):
    T = np.r_[np.linspace(60, 2, 600), np.linspace(2, 60, 600) + 0.2]
    t = np.arange(T.size) * 2.0
    R1 = (100 + 0.5 * T) / (1 + np.exp(-(T - tc) / 0.25))
    R2 = 2 * R1
    lines = ["[Header]", "; ETO", "[Data]",
             "Comment,Time Stamp (sec),Temperature (K),Magnetic Field (Oe),Resistance Ch1 (Ohms),Resistance Ch2 (Ohms)"]
    lines += [f",{a},{b},0,{c},{d}" for a, b, c, d in zip(t, T, R1, R2)]
    path.write_text("\n".join(lines))
