import numpy as np
import pytest

from evalprog.core import harmonics as hh

from synth import write_lockin_rt, write_lockin_sweep


@pytest.fixture
def files(tmp_path):
    paths = []
    for k, T in enumerate((5.0, 10.0, 15.0)):
        p = tmp_path / f"sweep_{k}.txt"
        write_lockin_sweep(p, T=T, v2=3e-11 * (k + 1), seed=k)
        paths.append(p)
    rt = tmp_path / "rt.txt"
    write_lockin_rt(rt)
    return paths, rt


def test_sweep_symmetrisation_and_hall(files):
    paths, rt_path = files
    s = hh.Settings(geometry="xy", t_um=0.02, make_plots=False)
    res = hh.process_path(paths[1], s, log=lambda m: None)[0]
    assert isinstance(res, hh.FileResult)
    assert res.current_A == pytest.approx(1e-5)
    assert res.T_K == pytest.approx(10.0, abs=0.01)
    v = res.vloop
    B = v[hh.FIELD_COL].to_numpy()
    anti = v[f"{hh.RE1}_antisym"].to_numpy()
    sym = v[f"{hh.RE1}_sym"].to_numpy()
    # sym part contains R0 + MR, antisym part Hall + hysteresis
    i0 = np.argmin(np.abs(B))
    assert sym[i0] == pytest.approx(1e-5 * 100, rel=1e-3)
    fit = hh.fit_normal_hall(res, hh.RE1, 5.0, "sat")
    assert fit["slope_V_per_Oe"] == pytest.approx(1e-5 * 2e-6, rel=1e-3)
    nums = hh.hall_numbers(res, fit, 0.02e-6)
    assert nums["dRxy/dB [Ohm/T]"] == pytest.approx(2e-6 * 1e4, rel=1e-3)
    assert (res.path.parent / "output").is_dir()


def test_rt_heating_and_comparison(files):
    paths, rt_path = files
    s = hh.Settings(make_plots=False)
    rt = hh.process_path(rt_path, s, log=lambda m: None)[0]
    assert isinstance(rt, hh.RTCurve)
    R0, dRdT, _ = hh.rt_local_fit(rt, 10.0, 2.0)
    assert dRdT == pytest.approx(5.0, rel=1e-3)
    sweeps = [hh.process_path(p, s, log=lambda m: None)[0] for p in paths]
    summary, last = hh.run_heating(sweeps, [rt], 2.0, log=lambda m: None, write_summary=True)
    assert len(summary) == 3
    res, rt_used, out, fit = last
    i0 = np.argmin(np.abs(out[hh.FIELD_COL].to_numpy()))
    v3 = np.hypot(1e-12, 0.2e-12)
    assert out["dT_2w [K]"].to_numpy()[i0] == pytest.approx(2 * v3 / (1e-5 * 5.0), rel=1e-3)

    curves, unit, skipped = hh.build_comparison(sweeps, "ImZ 2nd Harm [V]", "antisym", "V", "none", 0, "T")
    assert not skipped and unit == "V"
    dep = hh.dependence(curves, 1e4)
    assert np.allclose(dep["y"], [3e-11 * 1e4 * (k + 1) for k in range(3)], rtol=1e-2)
    fit = hh.power_fit(dep["x"].to_numpy(), dep["y"].to_numpy())
    assert fit["n"] == pytest.approx(np.log(3) / np.log(3), rel=0.05)  # y ∝ T here
    args = (curves, unit, "ImZ2 antisym", "T", 0.0, "both", dep, fit, True, 1e4)
    base = hh.export_comparison({"args": args, "col": "ImZ 2nd Harm [V]", "comp": "antisym", "quantity": "V",
                                 "norm": "none", "param": "T", "first": sweeps[0], "n": 0})
    assert (paths[0].parent / "output" / f"{base}_curves.csv").is_file()
    assert hh.same_series(sweeps[0], sweeps[1], "T")
    assert not hh.same_series(sweeps[0], sweeps[1], "I")


def test_restart_in_same_file(tmp_path):
    p = tmp_path / "a.txt"
    write_lockin_sweep(p)
    q = tmp_path / "b.txt"
    write_lockin_sweep(q, seed=3)
    text = p.read_text() + "\n" + q.read_text()  # second header line -> second section
    r = tmp_path / "restart.txt"
    r.write_text(text)
    segs = hh.read_segments(r)
    assert len(segs) == 2
    s = hh.Settings(restart="each", make_plots=False)
    assert len(hh.process_path(r, s, log=lambda m: None)) == 2
