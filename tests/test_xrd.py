import numpy as np
import pytest

from evalprog.core.xrd import io as xio
from evalprog.core.xrd.film import lattice_from_peak, thickness_from_fringes
from evalprog.core.xrd.peaks import bragg_two_theta, fit_peaks, ka2_position, pseudo_voigt
from evalprog.core.xrd.substrate import find_and_fit
from evalprog.materials import get_film, get_substrate, orientations, substrate_names


def synth_scan(substrate, z=0.05, height_um=50.0, film_c=None, seed=1):
    rng = np.random.default_rng(seed)
    x = np.arange(15, 110, 0.005)
    y = np.full_like(x, 20.0)
    D = -2 * height_um * 1e-3 / 240.0  # rad
    for n, d, label in substrate.reflections():
        tt = bragg_two_theta(d)
        if not np.isfinite(tt) or tt > 108:
            continue
        shift = z + np.degrees(D) * np.cos(np.radians(tt) / 2)
        c1 = tt + shift
        c2 = ka2_position(c1)
        h = 2e5 / n
        y += pseudo_voigt(x, c1, h, 0.02, 0.6) + pseudo_voigt(x, c2, 0.5 * h, 0.02, 0.6)
    if film_c:
        for l in (1, 2, 3, 4):
            tt = bragg_two_theta(film_c / l)
            if tt > 108:
                continue
            shift = z + np.degrees(D) * np.cos(np.radians(tt) / 2)
            y += pseudo_voigt(x, tt + shift, 3000 / l, 0.15, 0.3)
            y += pseudo_voigt(x, ka2_position(tt + shift), 1500 / l, 0.15, 0.3)
    return x, rng.poisson(y).astype(float)


def test_database_complete():
    names = substrate_names()
    for n in ["SrTiO3", "NdGaO3", "LaAlO3", "SrLaAlO4", "DyScO3", "GdScO3", "MgO", "LSAT"]:
        assert n in names
        for o in orientations(n):
            s = get_substrate(n, o)
            assert s.d_oop > 1
            assert all(np.isfinite(d.spacing) for d in s.inplane)
    ngo = get_substrate("NdGaO3", "(110)o")
    assert ngo.d_oop == pytest.approx(3.8625, abs=2e-3)
    assert get_film("SrRuO3").a == pytest.approx(3.93)


def test_single_peak_fit_recovers_center():
    x = np.arange(45, 48, 0.004)
    c = 46.472
    y = 10 + pseudo_voigt(x, c, 1e5, 0.03, 0.5) + pseudo_voigt(x, ka2_position(c), 5e4, 0.03, 0.5)
    y = np.random.default_rng(0).poisson(y).astype(float)
    fr = fit_peaks(x, y, [c + 0.02])
    assert fr.main.center == pytest.approx(c, abs=2e-3)
    assert fr.r_wp < 0.05


@pytest.mark.parametrize("name,orient", [("SrTiO3", "(001)"), ("NdGaO3", "(110)o"),
                                         ("SrLaAlO4", "(001)")])
def test_substrate_correction(name, orient):
    sub = get_substrate(name, orient)
    x, y = synth_scan(sub, z=0.05, height_um=50.0, film_c=3.95)
    res = find_and_fit(x, y, sub, max_peaks=4)
    assert len(res.peaks) >= 2
    corr = res.correction
    assert corr.model == "zero+height"
    assert corr.zero_shift == pytest.approx(0.05, abs=0.01)
    assert corr.sample_height_um == pytest.approx(50.0, abs=15.0)
    xc = corr.apply(np.array([p.tt_meas for p in res.peaks]))
    assert np.allclose(xc, [p.tt_calc for p in res.peaks], atol=3e-3)


def test_film_lattice_and_fringes():
    tt = bragg_two_theta(3.95 / 2)
    c, d = lattice_from_peak(tt, (0, 0, 2))
    assert c == pytest.approx(3.95, abs=1e-6)
    # fringes of a 20 nm film: q spacing 1/t
    t = 200.0
    q = 2 * np.sin(np.radians(44.0 / 2)) / xio.CU_KA1 + np.arange(5) / t
    pos = np.degrees(2 * np.arcsin(q * xio.CU_KA1 / 2))
    t_fit, _ = thickness_from_fringes(pos)
    assert t_fit == pytest.approx(t, rel=1e-6)


def test_two_column_and_xrdml(tmp_path):
    p = tmp_path / "scan.xy"
    p.write_text("# header\n10 5\n10.01 6\n10.02 7\n10.03 6\n10.04 5\n10.05 4\n")
    s = xio.read_two_column(p)
    assert s.two_theta.size == 6
    xml = """<?xml version="1.0"?>
<xrdMeasurements xmlns="http://www.xrdml.com/XRDMeasurement/1.5">
 <xrdMeasurement>
  <usedWavelength intended="K-Alpha 1"><kAlpha1 unit="Angstrom">1.5405980</kAlpha1>
   <kAlpha2 unit="Angstrom">1.5444260</kAlpha2><ratioKAlpha2KAlpha1>0.5</ratioKAlpha2KAlpha1></usedWavelength>
  {scans}
 </xrdMeasurement>
</xrdMeasurements>"""
    scan = """<scan scanAxis="2Theta-Omega"><dataPoints>
   <positions axis="2Theta" unit="deg"><startPosition>{a}</startPosition><endPosition>{b}</endPosition></positions>
   <positions axis="Omega" unit="deg"><startPosition>{c}</startPosition><endPosition>{d}</endPosition></positions>
   <commonCountingTime unit="seconds">1.0</commonCountingTime>
   <intensities unit="counts">1 2 3 4 5</intensities></dataPoints></scan>"""
    p1 = tmp_path / "one.xrdml"
    p1.write_text(xml.format(scans=scan.format(a=20, b=21, c=10, d=10.5)))
    s1 = xio.load_scan(p1)
    assert isinstance(s1, xio.Scan1D) and s1.two_theta[-1] == pytest.approx(21)
    p2 = tmp_path / "map.xrdml"
    p2.write_text(xml.format(scans="".join(scan.format(a=20, b=21, c=10 + o, d=10.5 + o)
                                           for o in (0, 0.1, 0.2))))
    a = xio.load_area(p2)
    assert a.intensity.shape == (3, 5)
