import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from evalprog.materials import get_substrate  # noqa: E402

from synth import rsm_grid, write_qd_sc, write_xrdml_map, write_xy, xrd_scan  # noqa: E402


@pytest.fixture(scope="module")
def window():
    from evalprog.gui import app as app_mod, rsm_tab, rt_tab, xrd_tab
    MainWindow = app_mod.MainWindow

    def raiser(parent, title, exc):
        raise AssertionError(f"{title}: {exc}")

    for mod in (rsm_tab, rt_tab, xrd_tab):
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
