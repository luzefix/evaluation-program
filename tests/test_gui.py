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
def window(tmp_path_factory):
    os.environ["EVALPROG_DB"] = str(tmp_path_factory.mktemp("db") / "test.sqlite")
    os.environ["EVALPROG_NO_SYNC"] = "1"
    from evalprog.gui import app as app_mod, rsm_tab, rt_tab, xrd_tab
    MainWindow = app_mod.MainWindow

    def raiser(parent, title, exc):
        raise AssertionError(f"{title}: {exc}")

    from evalprog.gui import afm_tab, harmonics_tab
    from evalprog.gui import db_widgets
    for mod in (rsm_tab, rt_tab, xrd_tab, harmonics_tab, afm_tab, db_widgets):
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
    from evalprog.gui.db_widgets import current_db
    tab.db.ed_sample.setText("CIO9")
    tab.db.on_save()
    t = current_db().table()
    rh = t[t["technique"] == "R(H)"]
    assert len(rh) == 2 and rh["R(H) T (K)"].notna().all()
    assert current_db().curves(int(rh["id"].iloc[0]))


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


def test_database(window, tmp_path):
    from evalprog.gui.db_widgets import current_db
    x, y = xrd_scan(get_substrate("SrTiO3", "(001)"))
    p = tmp_path / "CIO9_xrd.xy"
    write_xy(p, x, y)
    tab = window.xrd
    tab.load(str(p))
    assert tab.db.sample() == "CIO9"
    tab.on_fit_substrate()
    tab._on_span(45.0, 47.0)
    tab.sp_l.setValue(2)
    tab.on_fit_film()
    tab.db.on_save()
    tab.db.on_save()                       # overwrite, not duplicate
    rt = window.rt
    rt.on_clear()
    rt.ed_name.setText("")
    q = tmp_path / "CIO9_eto.dat"
    write_qd_sc(q)
    rt.add_file(str(q))
    assert rt.ed_name.text() == "CIO9"
    rt.cb_mode.setCurrentIndex(0)
    rt.on_analyse()
    rt.db.on_save()
    db = current_db()
    t = db.table()
    n0 = len(t)
    assert (t["technique"] == "XRD").sum() == 1
    assert t["R(T) Tc_mid (K)"].dropna().item() == pytest.approx(30.0, abs=0.5)
    dbt = window.database
    dbt.reload()
    assert dbt.model.rowCount() == n0
    dbt.cb_tech.setCurrentText("XRD")
    assert dbt.model.rowCount() == 1
    dbt.cb_tech.setCurrentText("all")
    dbt.add_filter()
    f = dbt.filters[-1]
    f.cb_col.setCurrentText("R(T) Tc_mid (K)")
    f.cb_op.setCurrentText(">")
    f.ed_val.setText("20")
    dbt.refilter()
    assert dbt.model.rowCount() == 1
    dbt.remove_filter(f)
    dbt.view.selectAll()
    dbt._update_curve_names()
    dbt.cb_curve.setCurrentIndex(0)
    dbt.on_overlay()
    db.set_params("CIO9", {"T_Dep": 720})
    dbt.reload()
    dbt.cb_x.setCurrentText("syn: T_Dep")
    dbt.cb_y.setCurrentText("R(T) Tc_mid (K)")
    dbt.on_scatter()
    dbt.plot.figure.savefig(tmp_path / "db.png")
    # one row per sample
    dbt.cb_view.setCurrentText("per sample")
    df = dbt.model.df
    row = df[df["sample"] == "CIO9"].iloc[0]
    assert "XRD" in row["techniques"] and "R(T)" in row["techniques"]
    assert row["R(T) Tc_mid (K)"] == pytest.approx(30.0, abs=0.5)
    assert "XRD c mean (Å)" in df.columns
    dbt.cb_x.setCurrentText("syn: T_Dep")
    dbt.cb_y.setCurrentText("R(T) Tc_mid (K)")
    dbt.on_scatter()
    dbt.view.selectRow(int(df.index[df["sample"] == "CIO9"][0]))
    assert len(dbt.selected()) == row["measurements"]
    dbt.cb_view.setCurrentText("per measurement")
    # exported files are referenced, automatic saving is armed after an evaluation
    csv = tmp_path / "CIO9_xrd_results.csv"
    csv.write_text("x")
    tab.db.attach(csv, "results csv")
    t = db.table()
    mid = int(t.loc[t["technique"] == "XRD", "id"].item())
    assert ("results csv", str(csv)) in db.files(mid)
    tab.db.on_save()                       # re-saving keeps the file reference
    t = db.table()
    mid = int(t.loc[t["technique"] == "XRD", "id"].item())
    assert len(db.files(mid)) == 1
    tab.on_fit_film()
    assert tab.db._timer.isActive()
    tab.db._timer.stop()
    # elabFTW dialog and the results that would be appended
    from evalprog.gui.elab_dialog import ElabTargetDialog
    rows = tab.db.elab_rows([tab.db_record()])
    assert rows and rows[0][0] == "c mean"
    ents = [{"id": 3, "title": "CIO8_STO(001)", "category_title": "Thin Film Growth", "status_title": "Running",
             "date": "2026-11-01", "fullname": "A", "modified_at": "2026-11-02 10:00:00"},
            {"id": 12, "title": "CIO9_STO(001)", "category_title": "Thin Film Growth", "status_title": "Success",
             "date": "2026-11-20", "fullname": "B", "modified_at": "2026-11-21 10:00:00"},
            {"id": 40, "title": "Termination STO", "category_title": "Sample Preparation", "status_title": "",
             "date": "2026-11-22", "fullname": "A", "modified_at": "2026-11-22 10:00:00"}]
    dlg = ElabTargetDialog(None, "XRD: x.xy", "CIO9", None, entries=ents)
    assert dlg.target() == ("experiments", 12)          # preselected from the sample name
    dlg.ed_search.setText("termination")
    assert sum(not dlg.table.isRowHidden(i) for i in range(3)) == 1
    dlg.ed_search.setText("")
    dlg.cb_cat_f.setCurrentText("Thin Film Growth")
    assert sum(not dlg.table.isRowHidden(i) for i in range(3)) == 2
    dlg.table.sortItems(0)
    assert dlg.table.item(0, 0).text() == "3"
    dlg.rb_new.setChecked(True)
    assert dlg.new_entry()["title"] == "CIO9"
    dlg.close()
    # one click: raw file back into its tab
    row = t[t["technique"] == "XRD"].to_dict("records")
    window.xrd.scan = None
    window.open_measurements(row)
    assert window.xrd.scan is not None and window.tabs.currentWidget() is window.xrd
