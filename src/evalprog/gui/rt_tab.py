"""R(T) tab: transport vs temperature (QD ETO / Resistivity option / lock-in)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QHBoxLayout, QLineEdit, QListWidget, QListWidgetItem,
                               QPushButton, QSplitter, QWidget)

from evalprog.core.rt import analysis as rta
from evalprog.core.rt.io import load_rt

from evalprog.db.store import Curve, Measurement, Result

from .db_widgets import SaveToDB, file_date
from .widgets import (PlotWidget, ResultsTable, combo, group, hbox, info_label, open_files,
                      save_file, scroll_panel, show_error, spin)

RESULT_NAMES = {0: ["Tc_onset", "Tc_mid", "Tc_zero", "Tc_onset_tangent", "dTc"],
                1: ["T_feature1", "T_feature2", "T_feature3", "T_feature4", None]}

MODES = ["Superconducting", "Magnetic / kinks", "Metal–insulator"]
QUANTITIES = ["R (Ω)", "ρ", "R_sheet (Ω/□)", "R / R(300 K)"]


class RTTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.datasets = []     # list of RTData
        self.results = []      # summary rows
        self.marker = None
        self._build()

    def _build(self):
        panel, lay = scroll_panel()
        self.btn_add = QPushButton("Add files …")
        self.btn_add.clicked.connect(self.on_add)
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.on_clear)
        lay.addWidget(hbox(self.btn_add, self.btn_clear))
        self.list = QListWidget()
        self.list.setMinimumHeight(140)
        self.list.itemChanged.connect(lambda *_: self.redraw())
        self.list.currentItemChanged.connect(lambda *_: self.redraw())
        lay.addWidget(info_label("Tick curves to plot; the selected (highlighted) one is analysed."))
        lay.addWidget(self.list)

        self.ed_name = QLineEdit()
        self.ed_name.setPlaceholderText("sample name")
        self.cb_branch = combo(["all", "cooling", "warming"])
        self.cb_qty = combo(QUANTITIES)
        self.chk_d2 = QCheckBox("show d²/dT² instead of d/dT")
        for w in (self.cb_branch, self.cb_qty):
            w.currentIndexChanged.connect(self.redraw)
        self.chk_d2.toggled.connect(self.redraw)
        lay.addWidget(group("Display", [("Sample", self.ed_name), ("Branch", self.cb_branch),
                                        ("Quantity", self.cb_qty), self.chk_d2]))

        self.cb_geom = combo(["bar (L, w)", "van der Pauw", "none"])
        self.sp_t = spin(20.0, 0, 1e6, 2, 1, " nm")
        self.sp_L = spin(1000.0, 0, 1e6, 1, 10, " µm")
        self.sp_w = spin(100.0, 0, 1e6, 1, 10, " µm")
        self.sp_f = spin(1.0, 0.01, 1.0, 4, 0.01)
        self.cb_unit = combo(list(rta.RHO_UNITS), "uOhm cm")
        for w in (self.sp_t, self.sp_L, self.sp_w, self.sp_f):
            w.valueChanged.connect(self.redraw)
        self.cb_geom.currentIndexChanged.connect(self.redraw)
        self.cb_unit.currentIndexChanged.connect(self.redraw)
        lay.addWidget(group("Geometry", [("Type", self.cb_geom), ("Thickness t", self.sp_t),
                                         ("Length L (V contacts)", self.sp_L), ("Width w", self.sp_w),
                                         ("vdP factor f", self.sp_f), ("ρ unit", self.cb_unit)]))

        self.cb_mode = combo(MODES)
        self.sp_win = spin(1.0, 0.01, 100, 2, 0.1, " K")
        self.sp_tmin = spin(0.0, 0, 1000, 2, 1, " K")
        self.sp_tmax = spin(0.0, 0, 1000, 2, 1, " K")
        self.sp_tmax.setSpecialValueText("max")
        self.sp_on = spin(90, 1, 99.9, 1, 1, " %")
        self.sp_low = spin(10, 0.1, 99, 1, 1, " %")
        self.sp_zero = spin(1, 0.001, 50, 3, 0.5, " %")
        self.sp_rrr_hi = spin(300, 1, 1000, 1, 1, " K")
        self.btn_run = QPushButton("Analyse selected curve")
        self.btn_run.clicked.connect(self.on_analyse)
        self.lbl_res = info_label("")
        lay.addWidget(group("Analysis", [
            ("Type", self.cb_mode), ("Derivative window", self.sp_win),
            ("T range", hbox(self.sp_tmin, self.sp_tmax)),
            ("SC: onset / low / zero", hbox(self.sp_on, self.sp_low, self.sp_zero)),
            ("RRR: T_high", self.sp_rrr_hi), self.btn_run, self.lbl_res]))
        lay.addWidget(info_label("Click in the plot to set a manual marker temperature."))

        self.btn_save_data = QPushButton("Save curve (.csv)")
        self.btn_save_data.clicked.connect(self.on_save_data)
        self.btn_save_res = QPushButton("Save results table (.csv)")
        self.btn_save_res.clicked.connect(self.on_save_results)
        self.btn_save_fig = QPushButton("Save figure (.png)")
        self.btn_save_fig.clicked.connect(self.on_save_fig)
        lay.addWidget(group("Export", [self.btn_save_data, self.btn_save_res, self.btn_save_fig]))
        self.db = SaveToDB("R(T)", self.db_record, self, name_edit=self.ed_name)
        self.db.group.setTitle("Database (sample name: see Display)")
        lay.addWidget(self.db.group)
        self.analysed = {}     # (dataset, channel) -> {"values": {...}, "mode": str, "branch": str}

        right = QSplitter(Qt.Vertical)
        self.plot = PlotWidget(figsize=(9, 7))
        self.ax1 = self.plot.figure.add_subplot(211)
        self.ax2 = self.plot.figure.add_subplot(212, sharex=self.ax1)
        self.plot.canvas.mpl_connect("button_press_event", self._on_click)
        right.addWidget(self.plot)
        self.table = ResultsTable(["Sample", "Curve", "Branch", "Type", "T1 (K)", "T2 (K)", "T3 (K)",
                                   "T4 (K)", "ΔT (K)", "RRR", "ρ(300 K)", "Details"])
        right.addWidget(self.table)
        right.setSizes([650, 200])
        main = QHBoxLayout(self)
        main.addWidget(panel)
        main.addWidget(right, 1)
        self.last = None

    # --------------------------------------------------------------- data
    def on_add(self):
        paths = open_files(self, "Open R(T) files", "Data (*.dat *.txt *.csv *.lvm);;All files (*)",
                           multi=True, key="rt_dir")
        for p in paths:
            self.add_file(p)

    def add_file(self, path):
        try:
            d = load_rt(path)
        except Exception as exc:
            show_error(self, f"Could not read {Path(path).name}", exc)
            return
        self.datasets.append(d)
        di = len(self.datasets) - 1
        self.list.blockSignals(True)
        for ci, ch in enumerate(d.channels):
            it = QListWidgetItem(f"{Path(path).name} | {ch.name}")
            it.setData(Qt.UserRole, (di, ci))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if ci == 0 else Qt.Unchecked)
            self.list.addItem(it)
        self.list.blockSignals(False)
        if self.list.currentRow() < 0:
            self.list.setCurrentRow(0)
        if not self.ed_name.text():
            self.db.set_source(path)
        if not self.ed_name.text():
            self.ed_name.setText(Path(path).stem)
        self.redraw()

    def on_clear(self):
        self.datasets.clear()
        self.list.clear()
        self.last = None
        self.marker = None
        self.redraw()

    def geometry(self):
        kind = {0: "bar", 1: "vdp", 2: "none"}[self.cb_geom.currentIndex()]
        return rta.Geometry(kind=kind, thickness_nm=self.sp_t.value(), length_um=self.sp_L.value(),
                            width_um=self.sp_w.value(), vdp_f=self.sp_f.value())

    def convert(self, ch, T, y):
        """Apply the selected quantity (R, rho, R_sheet, normalised). Returns (y, unit)."""
        q = self.cb_qty.currentIndex()
        unit = ch.unit
        if ch.kind == "R" and ch.unit == "Ohm":
            g = self.geometry()
            if q == 1 and g.kind != "none":
                y = rta.resistivity(y, g, self.cb_unit.currentText())
                unit = self.cb_unit.currentText()
            elif q == 2 and g.kind != "none":
                y = y * g.sheet_factor()
                unit = "Ohm/sq"
        if q == 3:
            hi = np.abs(T - self.sp_rrr_hi.value()) < 2.5
            if hi.any():
                y = y / np.nanmean(y[hi])
                unit = "norm."
        return y, unit

    def curve(self, item, branch=None):
        """(T, y, label, unit) for a list item, branch- and quantity-converted."""
        di, ci = item.data(Qt.UserRole)
        d = self.datasets[di]
        ch = d.channels[ci]
        T = d.temperature
        y, unit = self.convert(ch, T, ch.values)
        b = branch or self.cb_branch.currentText()
        if b != "all":
            br = rta.split_branches(T, y, d.time)
            if b not in br:
                return None
            T, y = br[b]
        else:
            ok = np.isfinite(T) & np.isfinite(y)
            T, y = T[ok], y[ok]
        return T, y, item.text(), unit

    # --------------------------------------------------------------- plot
    def redraw(self):
        self.ax1.clear()
        self.ax2.clear()
        unit = ""
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.checkState() != Qt.Checked and it is not self.list.currentItem():
                continue
            c = self.curve(it)
            if c is None or c[0].size < 6:
                continue
            T, y, label, unit = c
            o = np.argsort(T)
            lw = 1.6 if it is self.list.currentItem() else 0.9
            (line,) = self.ax1.plot(T[o], y[o], lw=lw, label=label)
            try:
                g, ys, d1, d2 = rta.derivatives(T, y, self.sp_win.value())
                self.ax2.plot(g, d2 if self.chk_d2.isChecked() else d1, lw=lw, color=line.get_color())
            except ValueError:
                pass
        self.ax1.set_ylabel(f"{self.cb_qty.currentText().split(' (')[0]} ({unit})" if unit else "")
        self.ax2.set_ylabel("d²/dT²" if self.chk_d2.isChecked() else "d/dT")
        self.ax2.set_xlabel("T (K)")
        for ax in (self.ax1, self.ax2):
            ax.grid(alpha=0.25)
        self._draw_markers()
        self.plot.draw()

    def _draw_markers(self):
        if self.last:
            for t, lab, col in self.last.get("markers", []):
                if np.isfinite(t):
                    self.ax1.axvline(t, color=col, ls="--", lw=0.9, label=f"{lab}: {t:.2f} K")
                    self.ax2.axvline(t, color=col, ls="--", lw=0.9)
            nl = self.last.get("normal_line")
            if nl is not None:
                self.ax1.plot(*nl, "k:", lw=1, label="normal state")
        if self.marker is not None:
            self.ax1.axvline(self.marker, color="magenta", lw=1, label=f"manual: {self.marker:.2f} K")
            self.ax2.axvline(self.marker, color="magenta", lw=1)
        if self.ax1.lines:
            self.ax1.legend(fontsize=7, loc="best")

    def _on_click(self, ev):
        if ev.inaxes in (self.ax1, self.ax2) and ev.xdata is not None and \
                self.plot.toolbar.mode == "":
            self.marker = float(ev.xdata)
            self.redraw()

    # --------------------------------------------------------------- analysis
    def on_analyse(self):
        it = self.list.currentItem()
        if it is None:
            return
        c = self.curve(it)
        if c is None:
            show_error(self, "No data", ValueError("This branch has no data."))
            return
        T, y, label, unit = c
        tmin = self.sp_tmin.value()
        tmax = self.sp_tmax.value() or float(np.max(T))
        rng = (tmin, tmax)
        mode = self.cb_mode.currentIndex()
        win = self.sp_win.value()
        # RRR and rho(300 K) always from resistance
        di, ci = it.data(Qt.UserRole)
        d = self.datasets[di]
        ch = d.channels[ci]
        rrr_v = rta.rrr(d.temperature, ch.values, self.sp_rrr_hi.value())
        rho300 = float("nan")
        g = self.geometry()
        if ch.kind == "R" and ch.unit == "Ohm" and g.kind != "none":
            try:
                m300 = np.abs(d.temperature - 300) < 2.5
                if m300.any():
                    rho300 = float(rta.resistivity(np.nanmean(ch.values[m300]), g, self.cb_unit.currentText()))
            except ValueError:
                pass
        row = [self.ed_name.text(), label, self.cb_branch.currentText(), MODES[mode]]
        try:
            if mode == 0:
                r = rta.superconducting_transition(T, y, window_K=win, onset_frac=self.sp_on.value() / 100,
                                                   low_frac=self.sp_low.value() / 100,
                                                   zero_frac=self.sp_zero.value() / 100, t_range=rng)
                s, i0 = r.normal_fit
                tt = np.linspace(r.t_low if np.isfinite(r.t_low) else T.min(), r.normal_range[1], 50)
                self.last = {"markers": [(r.t_onset, "onset", "tab:red"), (r.t_mid, "mid", "tab:orange"),
                                         (r.t_zero, "zero", "tab:green"),
                                         (r.t_onset_tangent, "tangent", "tab:purple")],
                             "normal_line": (tt, s * tt + i0)}
                txt = (f"T_c,onset ({self.sp_on.value():.0f} %) = {r.t_onset:.3f} K<br>"
                       f"T_c,mid (50 %) = {r.t_mid:.3f} K<br>"
                       f"T_c,zero ({self.sp_zero.value():g} %) = {r.t_zero:.3f} K<br>"
                       f"T_c,onset (tangent) = {r.t_onset_tangent:.3f} K<br>"
                       f"ΔT_c ({self.sp_on.value():.0f}–{self.sp_low.value():.0f} %) = {r.delta_tc:.3f} K<br>"
                       f"max dR/dT at {r.t_peak_deriv:.3f} K, R_n = {r.r_normal:.4g} {unit}")
                row += [r.t_onset, r.t_mid, r.t_zero, r.t_onset_tangent, r.delta_tc]
                details = f"normal state {r.normal_range[0]:.1f}-{r.normal_range[1]:.1f} K"
            elif mode == 1:
                feats = rta.transition_features(T, y, window_K=win, t_range=rng)
                cols = ["tab:red", "tab:blue", "tab:green", "tab:purple"]
                self.last = {"markers": [(f.temperature, f.kind, cols[i % 4]) for i, f in enumerate(feats[:4])]}
                txt = "<br>".join(f"{f.kind}: {f.temperature:.2f} K (prominence {f.prominence:.2f})"
                                  for f in feats[:6]) or "no features found"
                ts = [f.temperature for f in feats[:4]] + [float("nan")] * 4
                row += ts[:4] + [float("nan")]
                details = "; ".join(f"{f.kind} {f.temperature:.2f}" for f in feats[:4])
            else:
                rows_txt, markers, vals = [], [], []
                if self.cb_branch.currentText() == "all":
                    br = {}
                    for name in ("cooling", "warming"):
                        cb = self.curve(it, name)
                        if cb is not None and cb[0].size > 10:
                            br[name] = (cb[0], cb[1])
                    br = br or {"all": (T, y)}
                else:
                    br = {self.cb_branch.currentText(): (T, y)}
                for name, (tb, yb) in br.items():
                    m = (tb >= rng[0]) & (tb <= rng[1])
                    r = rta.mit_analysis(tb[m], yb[m], window_K=win, branch=name)
                    rows_txt.append(f"{name}: T_MI (max |dlnR/dT|) = {r.t_mit:.2f} K, R_min at {r.t_r_min:.2f} K")
                    markers.append((r.t_mit, f"T_MI {name}", "tab:red" if name == "cooling" else "tab:blue"))
                    vals += [r.t_mit, r.t_r_min]
                hyst = float("nan")
                tm = {n: v for (v, n, _) in markers}
                if "T_MI cooling" in tm and "T_MI warming" in tm:
                    hyst = tm["T_MI warming"] - tm["T_MI cooling"]
                    rows_txt.append(f"hysteresis width = {hyst:.2f} K")
                self.last = {"markers": markers}
                names = [f"{q} ({n})" for n in br for q in ("T_MI", "T_Rmin")][:4]
                names = (names + [None] * 4)[:4] + ["hysteresis"]
                txt = "<br>".join(rows_txt)
                vals = (vals + [float("nan")] * 4)[:4]
                row += vals + [hyst]
                details = ", ".join(br.keys())
        except Exception as exc:
            show_error(self, "Analysis failed", exc)
            return
        if np.isfinite(rrr_v):
            txt += f"<br>RRR (R({self.sp_rrr_hi.value():.0f} K)/R(T_min)) = {rrr_v:.3f}"
        if np.isfinite(rho300):
            txt += f"<br>ρ(300 K) = {rho300:.4g} {self.cb_unit.currentText()}"
        if self.marker is not None:
            txt += f"<br>manual marker: {self.marker:.2f} K"
            details += f"; manual {self.marker:.2f} K"
        row += [rrr_v, rho300, details]
        names = RESULT_NAMES.get(mode) or names
        vals = {n: v for n, v in zip(names, row[4:9]) if n and np.isfinite(v)}
        vals[f"RRR ({self.sp_rrr_hi.value():.0f} K)"] = rrr_v
        vals["rho(300 K)"] = rho300
        if self.marker is not None:
            vals["T_manual"] = self.marker
        self.analysed[(di, ci)] = {"values": vals, "mode": MODES[mode], "branch": self.cb_branch.currentText(),
                                   "rho_unit": self.cb_unit.currentText()}
        self.lbl_res.setText(txt)
        self.results.append(row)
        self.table.set_rows(self.results)
        self.redraw()
        self.db.auto()

    # --------------------------------------------------------------- export
    def _base(self):
        it = self.list.currentItem()
        if it is None:
            return Path.home() / "rt"
        di, _ = it.data(Qt.UserRole)
        p = self.datasets[di].path
        name = self.ed_name.text().strip() or p.stem
        return p.parent / f"RT_{name}"

    def on_save_data(self):
        it = self.list.currentItem()
        if it is None:
            return
        c = self.curve(it)
        if c is None:
            return
        T, y, label, unit = c
        p = save_file(self, "Save curve", f"{self._base()}.csv", "CSV (*.csv)")
        if not p:
            return
        g, ys, d1, d2 = rta.derivatives(T, y, self.sp_win.value())
        o = np.argsort(T)
        df = pd.DataFrame({"T": T[o], "y": y[o]})
        dd = pd.DataFrame({"T_grid": g, "y_smooth": ys, "dy_dT": d1, "d2y_dT2": d2})
        out = pd.concat([df, dd], axis=1)
        geom = self.geometry()
        with open(p, "w", encoding="utf-8") as f:
            f.write(f"# sample: {self.ed_name.text()}\n# curve: {label}\n# quantity: {self.cb_qty.currentText()} [{unit}]\n")
            f.write(f"# geometry: {geom}\n# branch: {self.cb_branch.currentText()}\n")
            f.write(f"# derivative window: {self.sp_win.value()} K\n")
            # Origin: long name, unit row
            f.write(",".join(["Temperature", self.cb_qty.currentText(), "Temperature", "smoothed", "d/dT", "d2/dT2"]) + "\n")
            f.write(",".join(["K", unit, "K", unit, f"{unit}/K", f"{unit}/K^2"]) + "\n")
            out.to_csv(f, index=False, header=False)
        self.db.attach(p, "curve csv")

    def db_record(self) -> Measurement:
        it = self.list.currentItem()
        if it is None:
            raise ValueError("Add a file and select a curve first.")
        di, ci = it.data(Qt.UserRole)
        d = self.datasets[di]
        ch = d.channels[ci]
        a = self.analysed.get((di, ci))
        res = []
        if a is not None:
            for k, v in a["values"].items():
                unit = a["rho_unit"] if k.startswith("rho") else ("" if k.startswith("RRR") else "K")
                res.append(Result(k, v, None, unit))
            res.append(Result("analysis", a["mode"]))
        g = self.geometry()
        settings = {"geometry": str(g), "thickness_nm": self.sp_t.value(), "channel": ch.name,
                    "source": d.source, "branch": a["branch"] if a else None,
                    "derivative_window_K": self.sp_win.value()}
        curves = [Curve(f"{ch.kind} ({ch.unit})", d.temperature, ch.values, "T (K)", f"{ch.kind} ({ch.unit})")]
        if ch.kind == "R" and ch.unit == "Ohm" and g.kind != "none":
            try:
                curves.append(Curve(f"rho ({self.cb_unit.currentText()})", d.temperature,
                                    rta.resistivity(ch.values, g, self.cb_unit.currentText()), "T (K)",
                                    f"ρ ({self.cb_unit.currentText()})"))
            except ValueError:
                pass
        return Measurement(sample="", technique="R(T)", raw_path=str(d.path), label=ch.name,
                           measured_at=file_date(d.path), settings=settings, results=res, curves=curves)

    def on_save_results(self):
        if not self.results:
            return
        p = save_file(self, "Save results", f"{self._base()}_results.csv", "CSV (*.csv)")
        if p:
            pd.DataFrame(self.results, columns=self.table.columns).to_csv(p, index=False)
            self.db.attach(p, "results csv")

    def on_save_fig(self):
        p = save_file(self, "Save figure", f"{self._base()}.png", "PNG (*.png);;PDF (*.pdf)")
        if p:
            self.plot.figure.savefig(p, dpi=300)
            if self.list.currentItem() is not None:
                self.db.attach(p, "figure")
