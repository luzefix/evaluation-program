import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from evalprog.materials import get_substrate  # noqa: E402

from synth import (afm_terraces, rsm_grid, write_nanoscope, write_lockin_rt, write_lockin_sweep, write_qd_sc,  # noqa: E402
                   write_xrdml_map, write_xy, xrd_scan)


@pytest.fixture(scope="module")
def window():
    from evalprog.gui import app as app_mod, rsm_tab, rt_tab, xrd_tab
    MainWindow = app_mod.MainWindow

    def raiser(parent, title, exc):
        raise AssertionError(f"{title}: {exc}")

    from evalprog.gui import afm_tab, harmonics_tab
    for mod in (rsm_tab, rt_tab, xrd_tab, harmonics_tab, afm_tab):
        mod.show_error = raiser
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    w.show()
    yield w
    w.close()


def test_xrd_tab(window, tmp_path):
    tab = window.xrd
    x, y = xrd_scan(get_substrate("SrTiO3", "(001)"))
    p = tmp_path / "s.xy"
    write_xy(p, x, y)
    tab.load(str(p))
    tab.on_fit_substrate()
    assert len(tab.sub_result.peaks) == 4
    assert tab.correction.zero_shift == pytest.approx(0.05, abs=0.01)
    tab._on_span(45.0, 47.0)  # SRO (002) at ~46.2 deg, STO (002) at 46.47
    tab.sp_l.setValue(2)
    tab.chk_fextra.setChecked(True)
    tab.on_fit_film()
    assert tab.film_peaks
    assert tab.results_frame().shape[0] == 5


def test_rsm_tab(window, tmp_path):
    tab = window.rsm
    TT, OM, I, qf = rsm_grid()
    p = tmp_path / "m.xrdml"
    write_xrdml_map(p, TT, OM, I)
    tab.load(str(p))
    tab.on_offset()
    assert tab.offset.d_omega == pytest.approx(0.12, abs=5e-3)
    tab.roi = (qf[0] - 0.03, qf[0] + 0.03, qf[1] - 0.04, qf[1] + 0.04)
    tab.on_fit()
    st = tab.peaks[0]["strain"]
    assert abs(st.relaxation) < 0.1
    tab.table.selectRow(0)
    window.rsm.plot.figure.savefig(tmp_path / "rsm.png")


def test_rt_tab(window, tmp_path):
    tab = window.rt
    p = tmp_path / "eto.dat"
    write_qd_sc(p)
    tab.add_file(str(p))
    assert tab.list.count() == 2
    tab.cb_mode.setCurrentIndex(0)
    tab.sp_win.setValue(0.3)
    tab.on_analyse()
    assert tab.results[-1][5] == pytest.approx(30.0, abs=0.3)  # T_mid
    tab.cb_qty.setCurrentIndex(1)
    tab.cb_branch.setCurrentIndex(2)
    tab.on_analyse()
    tab.cb_mode.setCurrentIndex(2)
    tab.cb_branch.setCurrentIndex(0)
    tab.on_analyse()
    assert len(tab.results) == 3


def test_harmonics_tab(window, tmp_path):
    tab = window.harmonics
    paths = []
    for k, T in enumerate((5.0, 10.0)):
        p = tmp_path / f"sw{k}.txt"
        write_lockin_sweep(p, T=T, seed=k)
        paths.append(p)
    rt = tmp_path / "rt.txt"
    write_lockin_rt(rt)
    tab.add_files(paths + [rt])
    tab.chk_plots.setChecked(False)
    tab.on_run()
    assert len(tab.sweeps) == 2 and len(tab.rts) == 1
    tab.on_rt_plot(save=True)
    tab.on_heating(all_=True)
    assert all(r.heating is not None for r in tab.sweeps.values())
    tab.list_hall.selectAll()
    tab.on_hall()
    assert all(r.hall for r in tab.sweeps.values())
    tab.list_cmp.selectAll()
    tab.on_compare()
    tab.on_compare_save()
    tab.rb_raw.setChecked(True)
    assert (tmp_path / "output").is_dir()


def test_afm_tab(window, tmp_path):
    tab = window.afm
    z, _ = afm_terraces()
    p = tmp_path / "img.spm"
    write_nanoscope(p, [("Height", z), ("Phase", z * 0)])
    tab.load(str(p))
    assert tab.cb_channel.count() == 2
    tab.cb_bg.setCurrentText("terraces (facet level)")
    tab.cb_rows.setCurrentText("median of differences")
    steps = tab.hist.steps[tab.hist.steps > 0.2]
    assert np.median(steps) == pytest.approx(0.3905, abs=0.03)
    tab.set_profile((100.0, 1000.0), (1900.0, 1000.0))
    assert tab.profile is not None
    tab.cb_kind.setCurrentText("pits")
    tab.sp_thr.setValue(1.0)
    tab.on_features()
    assert tab.features.summary["count"] >= 5
    tab.chk_showmask.setChecked(True)
    tab.plot.figure.savefig(tmp_path / "afm.png")
    # step analysis: wide profile across the steps, then a region
    tab.cb_smooth.setCurrentText("gaussian")
    tab.sp_smooth.setValue(8.0)
    tab.sp_width.setValue(10)
    tab.sp_edge.setValue(60.0)
    tab.set_profile((20.0, 300.0), (1980.0, 300.0))
    ps = tab.prof_steps
    assert len(ps.heights) >= 5
    assert np.median(np.abs(ps.heights)) == pytest.approx(0.3905, abs=0.03)
    tab.set_roi((300.0, 1000.0, 1100.0, 1900.0))
    assert tab.levelfit is not None and len(tab.levelfit.centers) >= 2
    assert tab.levelfit.ladder["step"] == pytest.approx(0.3905, abs=0.04)
    tab.plot_hist.figure.savefig(tmp_path / "afm_hist.png")
    tab.clear_roi()
    tab.cb_bow.setCurrentText("3")
    tab.on_auto_profile()
    assert tab.profile is not None and tab.auto is not None
    assert tab.sp_width.value() == tab.auto.width_px
    assert abs(tab.auto.steps.direction[0]) > 0.9   # synthetic steps run along y
