import numpy as np
import pytest

from evalprog.core.rsm.analysis import (RSMap, angles_to_q, correct_offset, expected_q,
                                        find_substrate_peak, fit_roi, pv2d, q_to_angles,
                                        strain_analysis)
from evalprog.core.xrd.io import area_from_points
from evalprog.materials import get_film, get_substrate


def test_q_angle_roundtrip():
    tt, om = 77.3, 30.1
    qx, qz = angles_to_q(tt, om)
    tt2, om2 = q_to_angles(qx, qz)
    assert tt2 == pytest.approx(tt) and om2 == pytest.approx(om)


def make_map(d_tt=0.03, d_om=0.12, seed=2):
    sub = get_substrate("SrTiO3", "(001)")
    film = get_film("SrRuO3")
    hkl = (1, 0, 3)
    a_s = sub.inplane_spacing()
    nu = film.poisson
    a_f = a_s  # fully strained
    c_f = film.c - 2 * nu / (1 - nu) * (a_f - film.a)
    qs = expected_q(hkl, a_s, sub.d_oop, sign=-1)
    qf = expected_q(hkl, a_f, c_f, sign=-1)
    tt0, om0 = q_to_angles(*qs)
    tt = np.arange(tt0 - 2.5, tt0 + 1.0, 0.01)
    offs = np.arange(-0.8, 0.8, 0.01)
    TT, OF = np.meshgrid(tt, offs)
    OM = TT / 2 + (om0 - tt0 / 2) + OF
    qx, qz = angles_to_q(TT, OM)
    I = (pv2d(qx, qz, 1e5, *qs, 0.0015, 0.0015, 0.0, 0.5) +
         pv2d(qx, qz, 3e3, *qf, 0.004, 0.012, 0.0, 0.4) + 5)
    I = np.random.default_rng(seed).poisson(I).astype(float)
    # measured angles carry an offset
    rsm = RSMap(TT + d_tt, OM + d_om, I)
    return rsm, sub, film, hkl, qf, a_f, c_f


def test_offset_correction_and_strain():
    rsm, sub, film, hkl, qf, a_f, c_f = make_map()
    off = correct_offset(rsm, sub, hkl)
    assert off.d_two_theta == pytest.approx(0.03, abs=3e-3)
    assert off.d_omega == pytest.approx(0.12, abs=3e-3)
    assert off.peak.qx == pytest.approx(off.qx_expected, abs=1e-4)
    # deliberately large ROI that contains the substrate tail: subtract the substrate
    roi = (qf[0] - 0.03, qf[0] + 0.03, qf[1] - 0.06, qf[1] + 0.05)
    pk = fit_roi(rsm, roi, subtract=[off.peak])
    s = strain_analysis(pk, hkl, film, a_substrate=sub.inplane_spacing())
    assert s.a_inplane == pytest.approx(a_f, abs=2e-3)
    assert s.c_out == pytest.approx(c_f, abs=2e-3)
    assert abs(s.relaxation) < 0.1
    assert s.poisson_eff == pytest.approx(film.poisson, abs=0.05)
    assert s.a_relaxed == pytest.approx(film.a, abs=3e-3)


def test_poisson_fit_better_than_unweighted():
    rsm, sub, film, hkl, qf, *_ = make_map()
    roi = find_substrate_peak(rsm, expected_q(hkl, 3.905, 3.905, -1))
    p1 = fit_roi(rsm, roi, weighting="poisson")
    p2 = fit_roi(rsm, roi, weighting="log")
    for p in (p1, p2):
        assert p.r2 > 0.98


def test_area_from_points():
    tt = np.repeat(np.linspace(40, 42, 5)[None, :], 3, 0)
    om = tt / 2 + np.array([[-0.1], [0.0], [0.1]])
    a = area_from_points(tt.ravel(), om.ravel(), np.ones(15))
    assert a.intensity.shape in ((3, 5), (5, 3))
