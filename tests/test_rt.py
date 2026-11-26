import numpy as np
import pytest

from evalprog.core.rt import analysis as rta
from evalprog.core.rt.io import load_rt


def sc_curve(tc=30.0, width=1.0, n=3000):
    T = np.linspace(2, 60, n)
    Rn = 100 + 0.5 * T
    R = Rn / (1 + np.exp(-(T - tc) / (width / 4.4)))
    return T, R + np.random.default_rng(0).normal(0, 0.02, n)


def test_superconducting_transition():
    T, R = sc_curve()
    r = rta.superconducting_transition(T, R)
    assert r.t_mid == pytest.approx(30.0, abs=0.1)
    assert r.t_onset > r.t_mid > r.t_low > r.t_zero
    assert r.delta_tc == pytest.approx(1.0, abs=0.3)
    assert r.t_peak_deriv == pytest.approx(30.0, abs=0.2)


def test_ferromagnetic_kink():
    T = np.linspace(5, 300, 3000)
    tc = 150.0
    # slope 0.5 Ohm/K below Tc, 0.3 Ohm/K above: kink at Tc
    R = 50 + 0.3 * T + np.where(T < tc, 0.2 * (T - tc), 0.0)
    f = rta.transition_features(T, R, window_K=3.0)
    temps = [x.temperature for x in f if x.kind.startswith("d2R")]
    assert any(abs(t - tc) < 3 for t in temps)


def test_rrr_mit_and_geometry():
    T = np.linspace(2, 300, 1000)
    R = 10 + 0.1 * T
    assert rta.rrr(T, R) == pytest.approx(40 / (10 + 0.1 * 4.5), rel=0.02)
    g = rta.Geometry("bar", thickness_nm=20, length_um=500, width_um=100)
    rho = rta.resistivity(np.array([100.0]), g, "uOhm cm")
    assert rho[0] == pytest.approx(100 * 100e-6 * 20e-9 / 500e-6 * 1e8)
    assert rta.vdp_factor(100, 100) == pytest.approx(1.0, rel=1e-6)
    assert rta.vdp_factor(100, 300) < 1.0
    # insulating upturn below 100 K
    R2 = 10 + 0.05 * T + 2000 * np.exp(-T / 15)
    mit = rta.mit_analysis(T, R2)
    assert 20 < mit.t_r_min < 120


def test_branches():
    T = np.r_[np.linspace(300, 10, 500), np.linspace(10, 300, 500)]
    R = T * 1.0
    b = rta.split_branches(T, R)
    assert set(b) == {"cooling", "warming"}


QD = """[Header]
; ETO
TITLE,test
[Data]
Comment,Time Stamp (sec),Temperature (K),Magnetic Field (Oe),Resistance Ch1 (Ohms),Resistance Ch1 Std. Dev. (Ohms),Resistance Ch2 (Ohms)
{rows}
"""

LOCKIN = """1.000,_Harm_,317.000,_Hz_,0.30000000,_V_,j= ,4102844.639,_Orders=_,,0.000,°,A=,8.0,I=,0.00001,R=,9140.000
{rows}
"""


def test_readers(tmp_path):
    rows = "\n".join(f",{i},{300 - i},0,{100 + i},0.1,{200 + i}" for i in range(50))
    p = tmp_path / "eto.dat"
    p.write_text(QD.format(rows=rows))
    d = load_rt(p)
    assert d.channel_names == ["Resistance Ch1 (Ohms)", "Resistance Ch2 (Ohms)"]
    assert d.temperature[0] == 300

    rows = "\n".join("\t".join(str(v) for v in [i, 300 - i, 0, 0, 317, 1e-3, 0, 0, 0, 0, 0, 1e-5, 0])
                     for i in range(50))
    p = tmp_path / "lockin.txt"
    p.write_text(LOCKIN.format(rows=rows))
    d = load_rt(p)
    assert d.current_A == pytest.approx(1e-5)
    c = d.channels[0]
    assert c.unit == "Ohm" and c.values[0] == pytest.approx(100.0)

    p = tmp_path / "table.txt"
    p.write_text("T (K)\tR (Ohm)\n" + "\n".join(f"{t}\t{2 * t}" for t in range(10, 40)))
    d = load_rt(p)
    assert d.channels[0].values[0] == 20
