"""R(H) / higher-harmonics tab (Hall, MR, 1f/2f/3f lock-in data)."""

from __future__ import annotations

import os
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QHBoxLayout,
                               QLabel, QListWidget, QListWidgetItem, QPlainTextEdit, QProgressBar,
                               QPushButton, QRadioButton, QSplitter, QTabWidget, QVBoxLayout, QWidget)

from evalprog.core import harmonics as hh

from evalprog.db.store import Curve, Measurement, Result

from .db_widgets import SaveToDB, file_date
from .widgets import (PlotWidget, combo, group, hbox, info_label, ispin, open_files, scroll_panel,
                      settings, show_error, spin)


def _pick(mapping: dict, label: str, default: str) -> str:
    return next((k for k, v in mapping.items() if v == label), default)


class HarmonicsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.files: list[Path] = []
        self.sweeps: dict[str, hh.FileResult] = {}
        self.rts: dict[str, hh.RTCurve] = {}
        self.settings_used: hh.Settings | None = None
        self._last_cmp = None
        self._build()

    # ================================================================ UI
    def _build(self):
        panel, lay = scroll_panel(360)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.file_list.setMinimumHeight(130)
        b_add, b_rem, b_clr = QPushButton("Add …"), QPushButton("Remove"), QPushButton("Clear")
        b_add.clicked.connect(self.on_add)
        b_rem.clicked.connect(self.on_remove)
        b_clr.clicked.connect(self.on_clear)
        g = group("1  Measurement files", [self.file_list, hbox(b_add, b_rem, b_clr),
                                           info_label("Field sweeps and R(T) curves are detected automatically.")])
        lay.addWidget(g)

        self.rb_xx = QRadioButton("longitudinal   ρxx = R·w·t / L")
        self.rb_xy = QRadioButton("Hall   ρxy = R·t")
        self.rb_xx.setChecked(True)
        self.sp_L = spin(0, 0, 1e6, 2, 10, " µm")
        self.sp_w = spin(0, 0, 1e6, 2, 10, " µm")
        self.sp_t = spin(0, 0, 1e6, 4, 0.001, " µm")
        for s_ in (self.sp_L, self.sp_w, self.sp_t):
            s_.setSpecialValueText("–")
        self.rb_hdr = QRadioButton("from header (I=)")
        self.rb_meas = QRadioButton("measured |Re I + i·Im I|")
        self.rb_hdr.setChecked(True)
        geo, cur = QButtonGroup(self), QButtonGroup(self)
        geo.addButton(self.rb_xx), geo.addButton(self.rb_xy)
        cur.addButton(self.rb_hdr), cur.addButton(self.rb_meas)
        lay.addWidget(group("2  Sample && contacts", [
            self.rb_xx, self.rb_xy, ("L (V contacts)", self.sp_L), ("w", self.sp_w), ("t", self.sp_t),
            info_label("optional – for ρ, j, R_H and n"), QLabel("Current:"), self.rb_hdr, self.rb_meas]))

        self.cb_out = combo(["V", "ρ [µΩ cm]"])
        self.sp_step = spin(50, 0.001, 1e5, 3, 10, " Oe")
        self.cb_method = combo(hh.METHODS.values())
        self.sp_minpts = ispin(20, 3, 100000)
        self.cb_restart = combo(hh.RESTART_MODES.values())
        self.chk_plots = QCheckBox("save plots as PNG")
        self.chk_plots.setChecked(True)
        self.sp_dpi = ispin(200, 50, 1200)
        self.btn_run = QPushButton("▶  Run evaluation")
        self.btn_run.clicked.connect(self.on_run)
        self.progress = QProgressBar()
        b_out = QPushButton("Open output folder")
        b_out.clicked.connect(self.on_open_output)
        lay.addWidget(group("3  Evaluation", [
            ("Output", self.cb_out), ("Field step", self.sp_step), ("Method", self.cb_method),
            ("Min. points / sweep", self.sp_minpts), ("Restart in same file", self.cb_restart),
            hbox(self.chk_plots, "DPI", self.sp_dpi)]))
        lay.addWidget(self.btn_run)
        lay.addWidget(self.progress)
        lay.addWidget(b_out)
        self.db = SaveToDB("R(H)", self.db_record, self, figure=self._current_figure)
        self.db.btn.setText("Save all evaluated curves to database")
        lay.addWidget(self.db.group)

        right = QSplitter(Qt.Vertical)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_preview(), "Preview")
        self.tabs.addTab(self._build_rt(), "R(T)")
        self.tabs.addTab(self._build_heating(), "Heating (3f)")
        self.tabs.addTab(self._build_hall(), "Hall")
        self.tabs.addTab(self._build_compare(), "Comparison")
        right.addWidget(self.tabs)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        right.addWidget(self.log_box)
        right.setSizes([720, 160])

        main = QHBoxLayout(self)
        main.addWidget(panel)
        main.addWidget(right, 1)
        self._load_config()

    def _tab(self, controls, plot):
        w = QWidget()
        v = QVBoxLayout(w)
        for c in controls:
            v.addWidget(c)
        v.addWidget(plot, 1)
        return w

    def _build_preview(self):
        self.cb_prev_file = combo([])
        self.cb_prev_col = combo([])
        self.rb_sym = QRadioButton("Sym / antisym")
        self.rb_raw = QRadioButton("Raw data")
        self.rb_sym.setChecked(True)
        self.cb_prev_file.currentIndexChanged.connect(self._on_prev_file)
        self.cb_prev_col.currentIndexChanged.connect(self._update_preview)
        self.rb_sym.toggled.connect(self._update_preview)
        self.plot_prev = PlotWidget()
        return self._tab([hbox("Measurement", self.cb_prev_file, "Signal", self.cb_prev_col,
                               self.rb_sym, self.rb_raw)], self.plot_prev)

    def _build_rt(self):
        self.list_rt = QListWidget()
        self.list_rt.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_rt.setMaximumHeight(110)
        b_all, b_rem, b_imp = QPushButton("All"), QPushButton("Remove"), QPushButton("Import table …")
        b_all.clicked.connect(self.list_rt.selectAll)
        b_rem.clicked.connect(self.on_rt_remove)
        b_imp.clicked.connect(self.on_rt_import)
        self.sp_colT, self.sp_colR = ispin(1, 1, 100), ispin(2, 1, 100)
        self.cb_rt_q = combo(hh.RT_QUANT.values())
        self.sp_rt_win = spin(0.5, 0.001, 100, 3, 0.1, " K")
        self.chk_rt_marks = QCheckBox("mark temperatures of the field sweeps")
        self.chk_rt_marks.setChecked(True)
        b_plot, b_save = QPushButton("Plot"), QPushButton("Save")
        b_plot.clicked.connect(lambda: self.on_rt_plot(False))
        b_save.clicked.connect(lambda: self.on_rt_plot(True))
        self.plot_rt = PlotWidget()
        top = QWidget()
        h = QHBoxLayout(top)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.list_rt, 1)
        v = QVBoxLayout()
        for b in (b_all, b_rem, b_imp):
            v.addWidget(b)
        v.addWidget(hbox("col. T", self.sp_colT, "R [Ω]", self.sp_colR))
        h.addLayout(v)
        return self._tab([top, hbox("Quantity", self.cb_rt_q, "Smoothing dR/dT ±", self.sp_rt_win,
                                    self.chk_rt_marks, b_plot, b_save)], self.plot_rt)

    def _build_heating(self):
        self.cb_heat_file = combo([])
        self.cb_heat_rt = combo(["automatic"])
        self.sp_heat_win = spin(2.0, 0.01, 100, 2, 0.5, " K")
        b1, b2 = QPushButton("Calculate && save"), QPushButton("For all field sweeps")
        b1.clicked.connect(lambda: self.on_heating(False))
        b2.clicked.connect(lambda: self.on_heating(True))
        self.plot_heat = PlotWidget()
        return self._tab([hbox("Sweep", self.cb_heat_file, "R(T) curve", self.cb_heat_rt, "Fit window ±",
                               self.sp_heat_win, b1, b2),
                          info_label("ΔT₂ω = 2·|V₃f| / (I·|dR/dT|). 'automatic' picks an R(T) curve that covers T "
                                     "and has the same resistance (= same contacts), the same angle and the "
                                     "smallest field.")], self.plot_heat)

    def _build_hall(self):
        self.list_hall = QListWidget()
        self.list_hall.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_hall.setMaximumHeight(110)
        self.cb_hall_sig = combo(hh.VOLTAGE_COLS, hh.RE1)
        self.cb_hall_model = combo(hh.HALL_MODELS.values())
        self.sp_hall_bmin = spin(5.0, 0, 100, 2, 0.5, " T")
        self.cb_hall_q = combo(hh.HALL_QUANT.values(), hh.HALL_QUANT["rho"])
        b1, b2 = QPushButton("Fit && subtract"), QPushButton("Remove fit")
        b1.clicked.connect(self.on_hall)
        b2.clicked.connect(self.on_hall_clear)
        self.plot_hall = PlotWidget()
        return self._tab([self.list_hall,
                          hbox("Signal", self.cb_hall_sig, "Model", self.cb_hall_model, "Fit for |B| ≥",
                               self.sp_hall_bmin, "Quantity", self.cb_hall_q, b1, b2),
                          info_label("Fit to the antisymmetric part at high field; only R₀·B is subtracted "
                                     "(the anomalous saturation A stays in the signal).")], self.plot_hall)

    def _build_compare(self):
        self.list_cmp = QListWidget()
        self.list_cmp.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_cmp.setMaximumHeight(120)
        b_all, b_none = QPushButton("All"), QPushButton("None")
        b_all.clicked.connect(self.list_cmp.selectAll)
        b_none.clicked.connect(self.list_cmp.clearSelection)
        self.cb_vary = combo(hh.SERIES_LABELS.values())
        b_series = QPushButton("Select series")
        b_series.clicked.connect(self.on_select_series)
        top = QWidget()
        h = QHBoxLayout(top)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.list_cmp, 1)
        v = QVBoxLayout()
        for w in (b_all, b_none, QLabel("Series of the marked\nmeasurement, varied:"), self.cb_vary, b_series):
            v.addWidget(w)
        h.addLayout(v)

        self.cb_cmp_sig = combo(hh.VOLTAGE_COLS, "ImZ 2nd Harm [V]")
        self.cb_cmp_sig.currentIndexChanged.connect(self._on_cmp_signal)
        self.cb_cmp_comp = combo(hh.COMPONENTS.values(), hh.COMPONENTS["antisym"])
        self.cb_cmp_q = combo(hh.QUANTITIES.values())
        self.cb_cmp_norm = combo(hh.NORMS.values())
        self.sp_cmp_n = spin(2, -10, 10, 2, 1)
        self.cb_cmp_param = combo(hh.PARAM_LABELS.values())
        self.cb_cmp_branch = combo(hh.BRANCHES)
        self.sp_cmp_off = spin(0, -1e12, 1e12, 6, 0)
        self.rb_b0 = QRadioButton("value at B₀ =")
        self.rb_max = QRadioButton("maximum |S|")
        self.rb_b0.setChecked(True)
        self.sp_cmp_b0 = spin(1.0, -100, 100, 3, 0.5, " T")
        self.chk_loglog = QCheckBox("log-log + power-law fit")
        b_plot, b_save = QPushButton("Plot"), QPushButton("Save")
        b_plot.clicked.connect(self.on_compare)
        b_save.clicked.connect(self.on_compare_save)
        self.plot_cmp = PlotWidget()
        return self._tab([top,
                          hbox("Signal", self.cb_cmp_sig, "Component", self.cb_cmp_comp, "Quantity", self.cb_cmp_q),
                          hbox("Normalisation", self.cb_cmp_norm, "n =", self.sp_cmp_n, "Colour/x", self.cb_cmp_param,
                               "Branches", self.cb_cmp_branch, "Offset", self.sp_cmp_off),
                          hbox("Right:", self.rb_b0, self.sp_cmp_b0, self.rb_max, self.chk_loglog, b_plot, b_save)],
                         self.plot_cmp)

    # ================================================================ helpers
    def log(self, text):
        self.log_box.appendPlainText(text)
        QApplication.processEvents()

    def collect_settings(self) -> hh.Settings:
        s = hh.Settings(
            mode="voltage" if self.cb_out.currentIndex() == 0 else "rho",
            geometry="xx" if self.rb_xx.isChecked() else "xy",
            current_source="header" if self.rb_hdr.isChecked() else "measured",
            L_um=self.sp_L.value(), w_um=self.sp_w.value(), t_um=self.sp_t.value(),
            field_step=self.sp_step.value(), method=_pick(hh.METHODS, self.cb_method.currentText(), "interp"),
            min_points=self.sp_minpts.value(),
            restart=_pick(hh.RESTART_MODES, self.cb_restart.currentText(), "last"),
            make_plots=self.chk_plots.isChecked(), dpi=self.sp_dpi.value())
        if s.mode == "rho" and hh.rho_factor_from(s) is None:
            raise ValueError("For output in ρ enter t (and L and w for the longitudinal geometry).")
        return s

    def _sorted_sweeps(self):
        return sorted(self.sweeps.values(), key=lambda r: (r.T_K is None, r.T_K or 0, r.path.name))

    def _fill_list(self, lw: QListWidget, items):
        keep = {it.data(Qt.UserRole) for it in lw.selectedItems()}
        lw.clear()
        for key, text in items:
            it = QListWidgetItem(text)
            it.setData(Qt.UserRole, key)
            lw.addItem(it)
            it.setSelected(key in keep)

    def _selected(self, lw: QListWidget):
        return [self.sweeps[it.data(Qt.UserRole)] for it in lw.selectedItems() if it.data(Qt.UserRole) in self.sweeps]

    def _row(self, r: hh.FileResult):
        extra = (" · Hall" if r.hall else "") + (" · ΔT" if r.heating is not None else "")
        return (f"{r.path.stem}   T={hh.fmt(r.T_K, '.2f')} K, I={hh.fmt(r.current_A * 1e6 if r.current_A else None, '.3g')} µA, "
                f"f={hh.fmt(r.f_Hz, '.1f')} Hz, {hh.fmt(r.angle, '.1f')}°, {r.geometry}{extra}")

    def refresh_lists(self):
        sw = [(str(r.path), self._row(r)) for r in self._sorted_sweeps()]
        for lw in (self.list_hall, self.list_cmp):
            self._fill_list(lw, sw)
        self._fill_list(self.list_rt, [(k, f"{rt.label}   B={hh.fmt((rt.B_oe or 0) / hh.OE_PER_T, '.3g')} T, "
                                            f"{rt.T.min():.1f}…{rt.T.max():.1f} K, {len(rt.T)} pts")
                                       for k, rt in self.rts.items()])
        cur = self.cb_prev_file.currentText()
        labels = [r.label for r in self._sorted_sweeps()]
        for cb in (self.cb_prev_file, self.cb_heat_file):
            t = cb.currentText()
            cb.blockSignals(True)
            cb.clear()
            cb.addItems(labels)
            if t in labels:
                cb.setCurrentText(t)
            cb.blockSignals(False)
        t = self.cb_heat_rt.currentText()
        self.cb_heat_rt.clear()
        self.cb_heat_rt.addItems(["automatic"] + [rt.label for rt in self.rts.values()])
        self.cb_heat_rt.setCurrentText(t if t else "automatic")
        if self.cb_prev_file.currentText() != cur:
            self._on_prev_file()

    def _sweep_by_label(self, label):
        return next((r for r in self.sweeps.values() if r.label == label), None)

    # ================================================================ files / run
    def on_add(self):
        paths = open_files(self, "Add measurement files", "Data (*.txt *.dat *.csv *.lvm);;All files (*)",
                           multi=True, key="hh_dir")
        for p in paths:
            p = Path(p)
            if p not in self.files:
                self.files.append(p)
                self.file_list.addItem(p.name)
        if paths and not self.db.sample():
            self.db.set_source(paths[0])

    def add_files(self, paths):
        paths = list(map(Path, paths))
        for p in paths:
            if p not in self.files:
                self.files.append(p)
                self.file_list.addItem(p.name)
        if paths and not self.db.sample():
            self.db.set_source(paths[0])

    # ================================================================ database
    def _current_figure(self):
        """Figure of the visible sub-tab (Preview, R(T), Heating, Hall, Comparison)."""
        w = self.tabs.currentWidget()
        plots = [w] if isinstance(w, PlotWidget) else w.findChildren(PlotWidget)
        if not plots:
            raise ValueError("No plot visible.")
        return plots[0].figure

    def db_record(self) -> list:
        """One database entry per evaluated field sweep (sym/antisym of every signal,
        Hall numbers if fitted) and per R(T) curve."""
        if not self.sweeps and not self.rts:
            raise ValueError("Run the evaluation first.")
        out = []
        t_um = self.sp_t.value() or None
        for r in self._sorted_sweeps():
            res = [Result("T", r.T_K, None, "K"), Result("I", r.current_A, None, "A"),
                   Result("f", r.f_Hz, None, "Hz"), Result("angle", r.angle, None, "deg"),
                   Result("B_max", float(np.abs(r.loop[hh.FIELD_COL]).max()) / hh.OE_PER_T, None, "T"),
                   Result("geometry", r.geometry)]
            for col, fit in r.hall.items():
                for k, v in hh.hall_numbers(r, fit, hh.hall_thickness(r, t_um)).items():
                    key, unit = (k.split(" [", 1) + [""])[:2]
                    res.append(Result(key, v, None, unit.rstrip("]"), hh.short_name(col)))
            up = r.loop[r.loop[hh.BRANCH] == "up"]
            B = up[hh.FIELD_COL].to_numpy(float) / hh.OE_PER_T
            curves = []
            for col in r.value_cols:
                name = hh.short_name(col)
                for part in ("sym", "antisym"):
                    curves.append(Curve(f"{name} {part}", B, up[f"{col}_{part}"].to_numpy(float), "B (T)",
                                        f"{name} {part} ({r.unit})", single=True))
            settings = {"unit": r.unit, "field_step_Oe": r.field_step}
            settings.update({k: str(v) for k, v in (r.meta or {}).items()})
            files = [("evaluated csv", r.csv_path)] if r.csv_path else []
            if (r.path.parent / "output").is_dir():
                files.append(("output folder", r.path.parent / "output"))
            if r.heating is not None and "dT_2w [K]" in r.heating.columns:
                hu = r.heating[r.heating[hh.BRANCH] == "up"]
                dT = hu["dT_2w [K]"].to_numpy(float)
                res.append(Result("dT_2w max", float(np.nanmax(np.abs(dT))), None, "K"))
                curves.append(Curve("heating dT_2w", hu[hh.FIELD_COL].to_numpy(float) / hh.OE_PER_T, dT,
                                    "B (T)", "ΔT_2ω (K)", single=True))
            out.append(Measurement(sample="", technique="R(H)", raw_path=str(r.path), label=r.tag,
                                   measured_at=file_date(r.path), settings=settings, curves=curves,
                                   results=[x for x in res if x.value is not None], files=files))
        for rt in self.rts.values():
            res = [Result("B", (rt.B_oe or 0) / hh.OE_PER_T, None, "T")]
            if rt.current_A:
                res.append(Result("I", rt.current_A, None, "A"))
            out.append(Measurement(sample="", technique="R(T)", raw_path=str(rt.path), label=rt.label,
                                   measured_at=file_date(rt.path), results=res,
                                   curves=[Curve("R (Ohm)", rt.T, rt.R, "T (K)", "R (Ω)")]))
        return out

    def on_remove(self):
        rows = sorted({self.file_list.row(it) for it in self.file_list.selectedItems()}, reverse=True)
        for r in rows:
            self.file_list.takeItem(r)
            del self.files[r]

    def on_clear(self):
        self.files.clear()
        self.file_list.clear()
        self.sweeps.clear()
        self.rts.clear()
        self.refresh_lists()

    def on_run(self):
        if not self.files:
            show_error(self, "No files", ValueError("Add measurement files first."))
            return
        try:
            s = self.collect_settings()
        except ValueError as e:
            show_error(self, "Invalid input", e)
            return
        self.settings_used = s
        self._save_config()
        self.btn_run.setEnabled(False)
        self.progress.setRange(0, len(self.files))
        self.log(f"=== Start: {len(self.files)} file(s), geometry {s.geometry}, output "
                 f"{'V' if s.mode == 'voltage' else 'ρ'}, {hh.METHODS[s.method]} ===")
        ok = 0
        try:
            for i, path in enumerate(self.files, 1):
                self.progress.setValue(i - 1)
                self.log(f"[{i}/{len(self.files)}] {path.name}")
                try:
                    for res in hh.process_path(path, s, self.log):
                        if isinstance(res, hh.FileResult):
                            self.sweeps[str(res.path)] = res
                            if res.csv_path:
                                self.log(f"  ✓ saved: output/{res.csv_path.name}")
                        else:
                            self.rts[res.key] = res
                            self.log(f"  ✓ R(T) curve {res.label} loaded (R(T) tab)")
                    ok += 1
                except ValueError as e:
                    self.log(f"  ✗ {e}")
                except Exception as e:  # keep going with the other files
                    self.log(f"  ✗ unexpected error {e!r}\n{traceback.format_exc()}")
        finally:
            self.progress.setValue(len(self.files))
            self.btn_run.setEnabled(True)
        self.log(f"=== Done: {ok}/{len(self.files)} files evaluated ===")
        self.refresh_lists()
        self.db.auto()
        if self.rts and not self.sweeps:
            self.tabs.setCurrentIndex(1)
            self.on_rt_plot(False)

    def on_open_output(self):
        r = self._sweep_by_label(self.cb_prev_file.currentText())
        folder = r.path.parent / "output" if r else (self.files[0].parent / "output" if self.files else None)
        if not folder or not folder.exists():
            show_error(self, "Output folder", ValueError("There is no output folder yet."))
            return
        if sys.platform.startswith("win"):
            os.startfile(folder)  # noqa
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])

    # ================================================================ preview
    def _on_prev_file(self):
        r = self._sweep_by_label(self.cb_prev_file.currentText())
        self.cb_prev_col.blockSignals(True)
        self.cb_prev_col.clear()
        if r:
            self.cb_prev_col.addItems(r.value_cols)
        self.cb_prev_col.blockSignals(False)
        self._update_preview()

    def _update_preview(self):
        r = self._sweep_by_label(self.cb_prev_file.currentText())
        col = self.cb_prev_col.currentText()
        fig = self.plot_prev.figure
        if not r or not col:
            fig.clear()
            self.plot_prev.draw()
            return
        with hh.MPL_LOCK:
            if self.rb_sym.isChecked():
                hh.plot_sym_antisym(fig, r.loop, col, r.unit, f"{r.path.stem}   ({r.tag})")
            else:
                hh.plot_raw(fig, r, col)
        self.plot_prev.draw()

    # ================================================================ R(T)
    def on_rt_import(self):
        paths = open_files(self, "Import R(T) table", "Data (*.txt *.dat *.csv);;All files (*)", key="hh_dir")
        if not paths:
            return
        s = self.settings_used or self.collect_settings()
        try:
            rt = hh.load_rt_table(Path(paths[0]), self.sp_colT.value(), self.sp_colR.value(), hh.rho_factor_from(s))
        except ValueError as e:
            show_error(self, "R(T) import", e)
            return
        self.rts[rt.key] = rt
        self.log(f"R(T) table imported: {rt.label} ({rt.T.min():.2f} … {rt.T.max():.2f} K, {len(rt.T)} points)")
        self.refresh_lists()

    def on_rt_remove(self):
        for it in self.list_rt.selectedItems():
            self.rts.pop(it.data(Qt.UserRole), None)
        self.refresh_lists()

    def on_rt_plot(self, save=False):
        keys = [it.data(Qt.UserRole) for it in self.list_rt.selectedItems()] or list(self.rts)
        if not keys:
            show_error(self, "R(T)", ValueError("No R(T) curves loaded yet."))
            return
        window = self.sp_rt_win.value()
        quantity = _pick(hh.RT_QUANT, self.cb_rt_q.currentText(), "R")
        items, unit = [], None
        for key in keys:
            rt = self.rts[key]
            try:
                d = hh.rt_table(rt, quantity, window)
            except ValueError as e:
                self.log(f"  WARNING: {rt.label}: {e} – showing R instead of ρ")
                d = hh.rt_table(rt, "R", window)
            unit = unit or d["unit"]
            if d["unit"] != unit:
                self.log(f"  WARNING: {rt.label} skipped (different unit)")
                continue
            items.append((rt, d))
        marks = sorted({round(r.T_K, 2) for r in self.sweeps.values() if r.T_K is not None}) \
            if self.chk_rt_marks.isChecked() else []
        with hh.MPL_LOCK:
            hh.plot_rt(self.plot_rt.figure, items, marks)
        self.plot_rt.draw()
        if save:
            path = hh.export_rt(items, window, marks, self.sp_dpi.value())
            self.log(f"R(T): {len(items)} curve(s) saved – CSV in output/, figure {path}")

    # ================================================================ heating
    def on_heating(self, all_=False):
        targets = self._sorted_sweeps() if all_ else [r for r in [self._sweep_by_label(self.cb_heat_file.currentText())] if r]
        if not targets:
            show_error(self, "Heating", ValueError("Evaluate field sweeps first."))
            return
        if not self.rts:
            show_error(self, "Heating", ValueError("Load an R(T) curve first (evaluate a measurement file or "
                                                   "import a table in the R(T) tab)."))
            return
        fixed = next((rt for rt in self.rts.values() if rt.label == self.cb_heat_rt.currentText()), None)
        window = self.sp_heat_win.value()
        _, last = hh.run_heating(targets, list(self.rts.values()), window, fixed, self.sp_dpi.value(),
                                 self.log, write_summary=all_)
        if last:
            with hh.MPL_LOCK:
                hh.plot_heating(self.plot_heat.figure, *last, window)
            self.plot_heat.draw()
        self.refresh_lists()
        self.db.auto()

    # ================================================================ Hall
    def on_hall(self):
        sel = self._selected(self.list_hall)
        if not sel:
            show_error(self, "Hall", ValueError("Mark at least one measurement in the list."))
            return
        col = self.cb_hall_sig.currentText()
        model = _pick(hh.HALL_MODELS, self.cb_hall_model.currentText(), "sat")
        quantity = _pick(hh.HALL_QUANT, self.cb_hall_q.currentText(), "rho")
        bmin = self.sp_hall_bmin.value()
        items, unit = hh.run_hall(sel, col, bmin, model, quantity, self.sp_t.value() or None,
                                  self.sp_dpi.value(), self.log)
        if items:
            with hh.MPL_LOCK:
                hh.plot_hall(self.plot_hall.figure, items, unit, bmin, col)
            self.plot_hall.draw()
        self.refresh_lists()
        self.db.auto()

    def on_hall_clear(self):
        col = self.cb_hall_sig.currentText()
        for r in self._selected(self.list_hall):
            r.hall.pop(col, None)
        self.refresh_lists()

    # ================================================================ comparison
    def _on_cmp_signal(self):
        n = hh.harmonic_of(self.cb_cmp_sig.currentText())
        self.sp_cmp_n.setValue(n)
        self.cb_cmp_comp.setCurrentText(hh.COMPONENTS["antisym" if n == 2 else "sym"])

    def on_select_series(self):
        sel = self._selected(self.list_cmp)
        if not sel:
            show_error(self, "Series", ValueError("Mark one measurement as reference first."))
            return
        vary = _pick(hh.SERIES_LABELS, self.cb_vary.currentText(), "T")
        ref = sel[0]
        members = {str(r.path) for r in self.sweeps.values() if hh.same_series(r, ref, vary)}
        for i in range(self.list_cmp.count()):
            it = self.list_cmp.item(i)
            it.setSelected(it.data(Qt.UserRole) in members)
        self.cb_cmp_param.setCurrentText(hh.PARAM_LABELS[vary])
        self.chk_loglog.setChecked(vary == "I")
        self.log(f"Series: {len(members)} measurement(s), varied: {hh.SERIES_LABELS[vary]}")

    def on_compare(self):
        sel = self._selected(self.list_cmp)
        if not sel:
            show_error(self, "Comparison", ValueError("Mark measurements in the list."))
            return
        col = self.cb_cmp_sig.currentText()
        comp = _pick(hh.COMPONENTS, self.cb_cmp_comp.currentText(), "sym")
        quantity = _pick(hh.QUANTITIES, self.cb_cmp_q.currentText(), "V")
        norm = _pick(hh.NORMS, self.cb_cmp_norm.currentText(), "none")
        param = _pick(hh.PARAM_LABELS, self.cb_cmp_param.currentText(), "T")
        n = self.sp_cmp_n.value() if norm == "In" else 0.0
        offset = self.sp_cmp_off.value()
        b0 = self.sp_cmp_b0.value() * hh.OE_PER_T if self.rb_b0.isChecked() else None
        if param != "idx":
            sel.sort(key=lambda r: (hh.param_value(r, param) is None, hh.param_value(r, param) or 0))
        curves, unit, skipped = hh.build_comparison(sel, col, comp, quantity, norm, n, param)
        for sk in skipped:
            self.log(f"  WARNING: skipped – {sk}")
        if not curves:
            show_error(self, "Comparison", ValueError("None of the marked measurements can be evaluated (see log)."))
            return
        if " / " in unit:
            self.log(f"  WARNING: different units ({unit})")
        dep = hh.dependence(curves, b0)
        loglog = self.chk_loglog.isChecked()
        fit = hh.power_fit(dep["x"].to_numpy(float), dep["y"].to_numpy(float)) if loglog and len(dep) else None
        ylabel = hh.comparison_ylabel(col, comp, quantity)
        args = (curves, unit, ylabel, param, offset, self.cb_cmp_branch.currentText(), dep, fit, loglog, b0)
        with hh.MPL_LOCK:
            hh.plot_comparison(self.plot_cmp.figure, *args)
        self.plot_cmp.draw()
        self._last_cmp = {"args": args, "col": col, "comp": comp, "quantity": quantity, "norm": norm,
                          "param": param, "first": sel[0], "n": n}
        pname, punit = hh.PARAMS[param]
        self.log(f"Comparison: {len(curves)} curves, {ylabel} [{unit}], colour/x = {pname}")
        if len(dep):
            where = f"at B = {b0 / hh.OE_PER_T:g} T" if b0 is not None else "maximum"
            for _, r in dep.sort_values("x").iterrows():
                self.log(f"  {pname} = {r['x']:8.4g} {punit:3s}  S({where}) = {r['y']:+.4e} {unit}")
        if fit:
            err = "" if np.isnan(fit["err"]) else f" ± {fit['err']:.3f}"
            self.log(f"  power law |S| ∝ {pname}^n:  n = {fit['n']:.3f}{err}")
        Ts = [r.T_K for r in sel if r.T_K is not None]
        if param != "T" and Ts and max(Ts) - min(Ts) > 1.0:
            self.log(f"  Note: temperatures differ by {max(Ts) - min(Ts):.2f} K.")

    def on_compare_save(self):
        if not self._last_cmp:
            show_error(self, "Comparison", ValueError("Plot first."))
            return
        base = hh.export_comparison(self._last_cmp, self.sp_dpi.value())
        self.log(f"Comparison saved: output/{base}_curves.csv (+ _values.csv), figure output/figures/comparison/{base}.png")

    # ================================================================ config
    def _save_config(self):
        s = settings()
        s.beginGroup("harmonics")
        for k, w in {"L": self.sp_L, "w": self.sp_w, "t": self.sp_t, "step": self.sp_step}.items():
            s.setValue(k, w.value())
        s.setValue("geometry", "xy" if self.rb_xy.isChecked() else "xx")
        s.setValue("current", "measured" if self.rb_meas.isChecked() else "header")
        s.setValue("output", self.cb_out.currentIndex())
        s.endGroup()

    def _load_config(self):
        s = settings()
        s.beginGroup("harmonics")
        try:
            for k, w in {"L": self.sp_L, "w": self.sp_w, "t": self.sp_t, "step": self.sp_step}.items():
                v = s.value(k)
                if v is not None:
                    w.setValue(float(v))
            if s.value("geometry") == "xy":
                self.rb_xy.setChecked(True)
            if s.value("current") == "measured":
                self.rb_meas.setChecked(True)
            if s.value("output") is not None:
                self.cb_out.setCurrentIndex(int(s.value("output")))
        except (TypeError, ValueError):
            pass
        s.endGroup()
