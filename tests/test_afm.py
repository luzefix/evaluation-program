from pathlib import Path

import numpy as np
import pytest

from evalprog.core import afm

from synth import afm_terraces, write_nanoscope

REAL = Path(__file__).parent / "data" / "afm_example.spm"


def test_nanoscope_reader_roundtrip(tmp_path):
    z, _ = afm_terraces(n=128, scars=())
    p = tmp_path / "img.spm"
    lsb = write_nanoscope(p, [("Height", z), ("Amplitude", z * 0)])
    f = afm.load_afm(p)
    assert len(f.channels) == 2
    ch = f.height_channel()
    assert ch.name == "Height" and ch.unit == "nm"
    assert ch.width_nm == pytest.approx(2000.0)
    assert np.allclose(ch.data, z, atol=lsb)


def test_terrace_levelling_recovers_steps():
    z, true = afm_terraces()
    c = afm.Corrections(background="terraces (facet level)", rows="median of differences",
                        mask_features=True, zero="median")
    r = afm.correct(z, c)
    assert r.scars[60].sum() > 60 and r.scars[150].sum() > 60
    h = afm.height_histogram(r.z, bins=400)
    steps = h.steps[h.steps > 0.2]
    assert len(steps) >= 4
    assert np.median(steps) == pytest.approx(0.3905, abs=0.03)


def test_plane_level_and_roughness():
    rng = np.random.default_rng(1)
    n = 200
    y, x = np.mgrid[:n, :n]
    noise = rng.normal(0, 0.5, (n, n))
    z = noise + 0.01 * x - 0.02 * y + 5
    r = afm.correct(z, afm.Corrections(scars=False, rows="none", mask_features=False))
    rough = afm.roughness(r.z)
    assert rough["Sq"] == pytest.approx(0.5, rel=0.03)
    assert rough["Sa"] == pytest.approx(0.5 * np.sqrt(2 / np.pi), rel=0.03)
    f, p = afm.psd_1d(noise, 1.0)
    assert np.sum(p) * (f[1] - f[0]) == pytest.approx(noise.var(), rel=0.05)


def test_align_rows_methods():
    rng = np.random.default_rng(2)
    base = rng.normal(0, 0.05, (100, 120))
    off = rng.normal(0, 1, 100)[:, None]
    for m in afm.ROW_METHODS:
        out = afm.align_rows(base + off + (0.01 * np.arange(120) if m in ("linear", "quadratic") else 0), m)
        assert np.std(out - out.mean()) < 0.08, m


def test_pits():
    z = np.zeros((200, 200))
    yy, xx = np.mgrid[:200, :200]
    for cx, cy in ((50, 50), (150, 60), (100, 150)):
        z[(xx - cx) ** 2 + (yy - cy) ** 2 < 25] = -3.0
    fe = afm.find_features(z, 10.0, 10.0, 1.0, "pits")
    assert fe.summary["count"] == 3
    assert fe.summary["mean_extreme_nm"] == pytest.approx(-3.0)
    assert fe.summary["density_um2"] == pytest.approx(3 / 4.0)


def test_line_profile():
    z = np.tile(np.arange(100, dtype=float), (50, 1))
    s, h = afm.line_profile(z, 1.0, 1.0, (10.5, 20.5), (60.5, 20.5), width=3)
    assert s[-1] == pytest.approx(50.0)
    assert h[0] == pytest.approx(10.0) and h[-1] == pytest.approx(60.0)


@pytest.mark.skipif(not REAL.exists(), reason="real AFM example not in tests/data")
def test_real_bruker_file():
    f = afm.load_afm(REAL)
    ch = f.height_channel()
    assert ch.unit == "nm" and ch.data.shape == (1024, 1024)
    r = afm.correct(ch.data)
    assert 0.5 < afm.roughness(r.z)["Sq"] < 5


def _noisy_terraces():
    rng = np.random.default_rng(5)
    z, true = afm_terraces(n=512, size_nm=2000, scars=(), pits=0, row_sigma=0.05)
    return z + rng.normal(0, 0.15, z.shape), true, 2000 / 512


def test_noisy_terraces_histogram_with_smoothing():
    z, true, dx = _noisy_terraces()
    c = afm.Corrections(background="terraces (facet level)", rows="median of differences",
                        smooth="gaussian", smooth_nm=8)
    r = afm.correct(z, c, dx, dx)
    lf = afm.fit_levels(afm.height_histogram(r.z, bins=400), 0.3905)
    assert len(lf.centers) >= 6
    assert lf.ladder["step"] == pytest.approx(0.3905, abs=0.005)
    assert np.allclose(lf.steps, 0.3905, atol=0.02)


def test_profile_staircase_fit():
    z, true, dx = _noisy_terraces()
    r = afm.correct(z, afm.Corrections(background="terraces (facet level)",
                                       rows="median of differences"), dx, dx)
    s, h = afm.line_profile(r.z, dx, dx, (50, 1000), (1950, 1000), width=40)
    ps = afm.profile_steps(s, h, min_step=0.2, edge_exclusion_nm=60)
    assert len(ps.heights) == 7
    assert np.allclose(ps.heights, -0.3905, atol=0.02)
    assert np.mean(ps.terraces) == pytest.approx(250, rel=0.05)


def test_smoothing_keeps_levels():
    z = np.zeros((100, 100))
    z[:, 50:] = 1.0
    zs = afm.smooth(z + np.random.default_rng(0).normal(0, 0.2, z.shape), "gaussian", 3.0, 1.0)
    assert zs[:, :40].mean() == pytest.approx(0.0, abs=0.02)
    assert zs[:, 60:].mean() == pytest.approx(1.0, abs=0.02)
    assert zs[:, :40].std() < 0.05


def test_terrace_bow_removed():
    z, true, dx = _noisy_terraces()
    n = z.shape[0]
    X, Y = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    z = z + 1.0 * (X ** 2 + Y ** 2)            # 2 nm quadratic bow (centre to corner)
    base = dict(background="terraces (facet level)", rows="median of differences",
                smooth="gaussian", smooth_nm=8)
    r = afm.correct(z, afm.Corrections(**base, terrace_bow=2), dx, dx)
    lf = afm.fit_levels(afm.height_histogram(r.z, bins=400), 0.3905)
    assert len(lf.centers) >= 6
    assert lf.ladder["step"] == pytest.approx(0.3905, abs=0.01)
    # without the bow correction the terrace levels are smeared out
    r0 = afm.correct(z, afm.Corrections(**base, terrace_bow=0), dx, dx)
    assert afm.roughness(r0.z - r.z)["Sq"] > 0.1


def test_ladder_fit_never_raises():
    c = np.array([0.0, 0.39, 0.78, 1.17])
    for e in (np.zeros(4), np.full(4, np.inf), np.array([0.0, np.nan, 1e-3, np.inf])):
        lad = afm._ladder_fit(c, e, 0.3905)
        assert lad == {} or lad["step"] == pytest.approx(0.39, abs=1e-3)
    assert afm._ladder_fit(np.array([0.0, 0.01]), np.ones(2), 0.3905) == {}


def test_step_direction_and_auto_profile():
    z, true, dx = _noisy_terraces()
    r = afm.correct(z, afm.Corrections(background="terraces (facet level)",
                                       rows="median of differences", smooth="gaussian",
                                       smooth_nm=8), dx, dx)
    sd = afm.step_direction(r.z, dx, dx)
    # synthetic steps go down along +x → uphill normal is −x
    assert sd.direction[0] == pytest.approx(-1.0, abs=0.02)
    assert sd.terrace_nm == pytest.approx(250, rel=0.1)
    ap = afm.auto_profile(r.z, dx, dx, sd=sd)
    assert ap.width_px == int(np.clip(round(0.4 * sd.terrace_nm / dx), 3, 51))
    s, h = afm.line_profile(r.z, dx, dx, ap.p0, ap.p1, width=ap.width_px)
    assert s[-1] == pytest.approx(0.8 * 2000, rel=0.02)
    ps = afm.profile_steps(s, h, edge_exclusion_nm=60)
    assert len(ps.heights) >= 5
    assert np.mean(ps.heights) == pytest.approx(0.3905, abs=0.03)


def test_profile_steps_ignores_pits():
    s = np.arange(0, 2000, 4.0)
    h = 0.3905 * np.floor(s / 400)
    h[(s > 560) & (s < 640)] -= 0.4               # a pit crossed by the profile
    h += np.random.default_rng(3).normal(0, 0.02, s.size)
    ps = afm.profile_steps(s, h, edge_exclusion_nm=30)
    assert len(ps.heights) == 4
    assert np.allclose(ps.heights, 0.3905, atol=0.03)
