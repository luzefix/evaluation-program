"""XRD tab: symmetric 2theta-omega scans with substrate-based correction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import SpanSelector
from PySide6.QtWidgets import (QCheckBox, QHBoxLayout, QLabel, QPushButton, QSplitter,
                               QVBoxLayout, QWidget)
from PySide6.QtCore import Qt

from evalprog import materials
from evalprog.core.xrd import io as xio
from evalprog.core.xrd.film import combine_lattice, make_film_peak, out_of_plane_strain
from evalprog.core.xrd.peaks import bragg_two_theta, fit_peaks, fit_with_extra_peak, ka2_position
from evalprog.core.xrd.substrate import EMPYREAN_RADIUS_MM, Correction, find_and_fit, fit_correction

from .widgets import (PlotWidget, ResultsTable, combo, group, hbox, info_label, ispin,
                      open_files, save_file, scroll_panel, show_error, spin)

COLORS = ["tab:red", "tab:blue", "tab:green", "tab:purple", "tab:orange", "tab:brown"]


class XRDTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scan: xio.Scan1D | None = None
        self.sub_result = None
        self.correction = Correction()
        self.film_peaks = []
        self.span = None
        self.last_fit = None
        self._build()

    # ------------------------------------------------------------------ UI
    def _build(self):
        panel, lay = scroll_panel()

        self.btn_load = QPushButton("Load scan (.xy / .xrdml) …")
        self.btn_load.clicked.connect(self.on_load)
        self.lbl_file = info_label("No file loaded.")
        lay.addWidget(self.btn_load)
        lay.addWidget(self.lbl_file)

        self.cb_sub = combo(materials.substrate_names(), "SrTiO3")
        self.cb_orient = combo([])
        self.cb_sub.currentTextChanged.connect(self._update_orientations)
        self._update_orientations()
        self.sp_npeaks = ispin(4, 1, 8)
        self.sp_search = spin(1.0, 0.05, 5, 2, 0.1, " °")
        self.chk_split = QCheckBox("asymmetric (split) pseudo-Voigt")
        self.chk_split.setChecked(True)
        self.chk_extra = QCheckBox("allow one extra peak in the window (film / shoulder)")
        self.chk_extra.setChecked(True)
        self.cb_model = combo(["auto", "zero shift only", "zero shift + sample height", "none"])
        self.sp_radius = spin(EMPYREAN_RADIUS_MM, 50, 1000, 1, 1, " mm")
        self.btn_sub = QPushButton("Find && fit substrate peaks")
        self.btn_sub.clicked.connect(self.on_fit_substrate)
        self.cb_model.currentIndexChanged.connect(self._recompute_correction)
        self.sp_radius.valueChanged.connect(self._recompute_correction)
        self.lbl_corr = info_label("")
        lay.addWidget(group("Substrate", [
            ("Substrate", self.cb_sub), ("Orientation", self.cb_orient),
            ("Max. peaks", self.sp_npeaks), ("Search ±", self.sp_search),
            self.chk_split, self.chk_extra,
            ("Correction", self.cb_model), ("Goniometer R", self.sp_radius),
            self.btn_sub, self.lbl_corr]))

        self.cb_phase = combo(materials.film_names(), "SrRuO3", editable=True)
        self.sp_h, self.sp_k, self.sp_l = ispin(0, -20, 20), ispin(0, -20, 20), ispin(1, -20, 20)
        self.cb_fmodel = combo(["00l", "cubic"])
        self.chk_fextra = QCheckBox("allow one extra peak")
        self.btn_film = QPushButton("Fit selected range")
        self.btn_film.clicked.connect(self.on_fit_film)
        self.btn_del = QPushButton("Remove selected")
        self.btn_del.clicked.connect(self.on_remove_film)
        lay.addWidget(group("Film peak (drag a range in the upper plot)", [
            ("Phase", self.cb_phase), ("h k l", hbox(self.sp_h, self.sp_k, self.sp_l)),
            ("Model", self.cb_fmodel), self.chk_fextra, hbox(self.btn_film, self.btn_del)]))

        self.lbl_film = info_label("")
        lay.addWidget(self.lbl_film)

        self.btn_save_tab = QPushButton("Save results (.csv)")
        self.btn_save_tab.clicked.connect(self.on_save_results)
        self.btn_save_xy = QPushButton("Save corrected scan (.xy)")
        self.btn_save_xy.clicked.connect(self.on_save_xy)
        self.btn_save_fig = QPushButton("Save figure (.png)")
        self.btn_save_fig.clicked.connect(self.on_save_fig)
        lay.addWidget(group("Export", [self.btn_save_tab, self.btn_save_xy, self.btn_save_fig]))

        right = QSplitter(Qt.Vertical)
        self.plot = PlotWidget(figsize=(9, 7))
        gs = self.plot.figure.add_gridspec(3, 1, height_ratios=[2.2, 1.4, 0.5])
        self.ax_main = self.plot.figure.add_subplot(gs[0])
        self.ax_zoom = self.plot.figure.add_subplot(gs[1])
        self.ax_res = self.plot.figure.add_subplot(gs[2], sharex=self.ax_zoom)
        right.addWidget(self.plot)

        tables = QWidget()
        tl = QHBoxLayout(tables)
        tl.setContentsMargins(0, 0, 0, 0)
        self.tab_sub = ResultsTable(["Substrate", "2θ calc", "2θ meas", "Δ2θ", "FWHM", "R_wp"],
                                    formats=[None, ".4f", ".4f", "+.4f", ".4f", ".4f"])
        self.tab_film = ResultsTable(["Phase", "hkl", "2θ corr", "± ", "FWHM", "d (Å)", "c (Å)", "± (Å)"],
                                     formats=[None, None, ".4f", ".4f", ".4f", ".5f", ".4f", ".4f"])
        v1, v2 = QVBoxLayout(), QVBoxLayout()
        v1.addWidget(QLabel("Substrate peaks"))
        v1.addWidget(self.tab_sub)
        v2.addWidget(QLabel("Film peaks"))
        v2.addWidget(self.tab_film)
        tl.addLayout(v1)
        tl.addLayout(v2)
        right.addWidget(tables)
        right.setSizes([650, 220])

        main = QHBoxLayout(self)
        main.addWidget(panel)
        main.addWidget(right, 1)
        self._empty_plot()

    def _update_orientations(self):
        self.cb_orient.clear()
        name = self.cb_sub.currentText()
        if name:
            self.cb_orient.addItems(materials.orientations(name))

    def _empty_plot(self):
        for ax in (self.ax_main, self.ax_zoom, self.ax_res):
            ax.clear()
        self.ax_main.set_title("Load a scan")
        self.plot.draw()

    # ------------------------------------------------------------- actions
    def on_load(self):
        paths = open_files(self, "Open XRD scan", "XRD scans (*.xy *.txt *.csv *.dat *.xrdml);;All files (*)",
                           key="xrd_dir")
        if paths:
            self.load(paths[0])

    def load(self, path):
        try:
            scan = xio.load_scan(path)
            if isinstance(scan, xio.AreaScan):
                raise ValueError("This file is a 2D map: open it in the RSM tab")
        except Exception as exc:
            show_error(self, "Could not read file", exc)
            return
        self.scan = scan
        self.sub_result = None
        self.correction = Correction()
        self.film_peaks = []
        self.last_fit = None
        self.lbl_file.setText(f"{Path(path).name}  ({scan.two_theta.size} points, "
                              f"{scan.two_theta.min():.2f}–{scan.two_theta.max():.2f}°, λ = {scan.wavelength:.5f} Å)")
        self.lbl_corr.setText("")
        self._refresh()

    def corrected_x(self):
        return self.correction.apply(self.scan.two_theta)

    def on_fit_substrate(self):
        if self.scan is None:
            return
        try:
            sub = materials.get_substrate(self.cb_sub.currentText(), self.cb_orient.currentText())
            self.sub_result = find_and_fit(
                self.scan.two_theta, self.scan.intensity, sub, wavelength=self.scan.wavelength,
                wavelength2=self.scan.wavelength2 or xio.CU_KA2, ka2_ratio=self.scan.ka2_ratio,
                search=self.sp_search.value(), split=self.chk_split.isChecked(),
                max_peaks=self.sp_npeaks.value(), extra_peaks=1 if self.chk_extra.isChecked() else 0)
        except Exception as exc:
            show_error(self, "Substrate fit failed", exc)
            return
        if not self.sub_result.peaks:
            self.lbl_corr.setText("No substrate peak found. Check substrate / orientation or the search window.")
        self._recompute_correction()

    def _recompute_correction(self):
        if self.sub_result is None:
            return
        model = {0: "auto", 1: "zero", 2: "zero+height", 3: "none"}[self.cb_model.currentIndex()]
        if model == "none":
            self.correction = Correction(radius_mm=self.sp_radius.value())
        else:
            self.correction = fit_correction(self.sub_result.peaks, radius_mm=self.sp_radius.value(),
                                             model=model)
        txt = self.correction.describe()
        if self.correction.model != "none" and np.isfinite(self.correction.residual_rms):
            txt += f"\nresidual rms {self.correction.residual_rms * 1e3:.1f} m°"
        if self.sub_result.skipped:
            txt += "\nskipped: " + ", ".join(f"{l} ({r})" for l, r in self.sub_result.skipped)
        self.lbl_corr.setText(txt)
        # re-evaluate film peaks with the new correction
        self.film_peaks = [self._refit_film(fp) for fp in self.film_peaks]
        self._refresh()

    def _refit_film(self, fp):
        raw = fp["raw_center"]
        tt = float(self.correction.apply(raw))
        peak = make_film_peak(fp["peak"].phase, fp["peak"].hkl, tt, fp["peak"].two_theta_err,
                              fp["peak"].fwhm, self.scan.wavelength, fp["model"])
        return dict(fp, peak=peak)

    def _on_span(self, lo, hi):
        if hi - lo <= 0:
            return
        self.span_range = (lo, hi)
        self._draw_zoom(lo, hi)
        self.plot.draw()

    def on_fit_film(self):
        if self.scan is None or not getattr(self, "span_range", None):
            show_error(self, "No range", ValueError("Drag a range around the film peak in the upper plot."))
            return
        lo, hi = self.span_range
        xc = self.corrected_x()
        m = (xc >= lo) & (xc <= hi)
        try:
            hkl = (self.sp_h.value(), self.sp_k.value(), self.sp_l.value())
            xr, yr = self.scan.two_theta[m], self.scan.intensity[m]
            # substrate peaks inside the range are fitted together with the film peak
            sub_c = []
            if self.sub_result is not None:
                for sp in self.sub_result.peaks:
                    if xr.min() < sp.tt_meas < xr.max():
                        sub_c.append((sp.tt_meas, sp.fit.main.fwhm))
            free = np.ones(xr.size, bool)
            for c, w in sub_c:
                c2 = ka2_position(c, self.scan.wavelength, self.scan.wavelength2 or xio.CU_KA2)
                free &= (np.abs(xr - c) > 4 * max(w, 0.01)) & (np.abs(xr - c2) > 4 * max(w, 0.01))
            if not free.any():
                raise ValueError("The range only contains the substrate peak")
            film_guess = xr[free][np.argmax(yr[free])]
            kw = dict(split=False, wavelength=self.scan.wavelength,
                      wavelength2=self.scan.wavelength2 or xio.CU_KA2, ka2_ratio=self.scan.ka2_ratio)
            if sub_c:
                lo_b = [(xr.min(), xr.max())] + [(c - 0.05, c + 0.05) for c, _ in sub_c]
                fr = fit_peaks(xr, yr, [film_guess] + [c for c, _ in sub_c], center_bounds=lo_b, **kw)
            else:
                fr = fit_with_extra_peak(xr, yr, film_guess,
                                         max_extra=1 if self.chk_fextra.isChecked() else 0, **kw)
            raw = fr.main.center
            tt = float(self.correction.apply(raw))
            peak = make_film_peak(self.cb_phase.currentText().strip() or "film", hkl, tt, fr.main.center_err,
                                  fr.main.fwhm, self.scan.wavelength, self.cb_fmodel.currentText())
        except Exception as exc:
            show_error(self, "Film fit failed", exc)
            return
        self.film_peaks.append({"peak": peak, "raw_center": raw, "model": self.cb_fmodel.currentText(),
                                "fit": fr})
        self.last_fit = fr
        self._refresh(zoom=(lo, hi))

    def on_remove_film(self):
        i = self.tab_film.selected_row()
        if i is None and self.film_peaks:
            i = len(self.film_peaks) - 1
        if i is not None and 0 <= i < len(self.film_peaks):
            del self.film_peaks[i]
            self._refresh()

    # --------------------------------------------------------------- plots
    def _refresh(self, zoom=None):
        ax = self.ax_main
        ax.clear()
        if self.scan is None:
            self._empty_plot()
            return
        x = self.corrected_x()
        y = self.scan.intensity
        ax.semilogy(x, np.clip(y, 0.5, None), "k-", lw=0.8)
        ax.set_xlabel("2θ (°)" + (" corrected" if self.correction.model != "none" else ""))
        ax.set_ylabel("Intensity")
        ax.set_title(Path(self.scan.path).name if self.scan.path else "")
        ax.grid(alpha=0.25)
        if self.sub_result is not None:
            for n, d, label in self.sub_result.substrate.reflections():
                tt = bragg_two_theta(d, self.scan.wavelength)
                if np.isfinite(tt) and x.min() < tt < x.max():
                    ax.axvline(tt, color="tab:cyan", lw=0.8, ls="--")
                    ax.text(tt, ax.get_ylim()[1], f" {label}", color="tab:cyan", fontsize=8,
                            rotation=90, va="top", ha="right")
        for i, fp in enumerate(self.film_peaks):
            p = fp["peak"]
            ax.axvline(p.two_theta, color=COLORS[i % len(COLORS)], lw=0.9)
            ax.text(p.two_theta, ax.get_ylim()[1], f" {p.phase} {p.hkl_label}", rotation=90, va="top",
                    ha="left", fontsize=8, color=COLORS[i % len(COLORS)])
        self.span = SpanSelector(ax, self._on_span, "horizontal", useblit=True, interactive=True,
                                 props=dict(alpha=0.2, facecolor="tab:orange"))
        self._update_tables()
        if zoom is not None:
            self._draw_zoom(*zoom)
        elif self.sub_result is not None and self.sub_result.peaks:
            self._draw_substrate_fits()
        else:
            self.ax_zoom.clear()
            self.ax_res.clear()
        self.plot.draw()

    def _draw_fit(self, fr, color="tab:red", label=None):
        xc = self.correction.apply(fr.x)
        self.ax_zoom.semilogy(xc, np.clip(fr.y, 0.5, None), ".", ms=2, color="0.4")
        self.ax_zoom.semilogy(xc, np.clip(fr.y_fit, 0.5, None), "-", color=color, lw=1.4, label=label)
        sig = np.sqrt(np.maximum(fr.y, 1))
        self.ax_res.plot(xc, (fr.y - fr.y_fit) / sig, "-", color=color, lw=0.7)

    def _draw_substrate_fits(self):
        self.ax_zoom.clear()
        self.ax_res.clear()
        # show the strongest substrate peak fit, others are in the table
        for i, sp in enumerate(self.sub_result.peaks):
            self._draw_fit(sp.fit, COLORS[i % len(COLORS)],
                           f"{sp.label}: Δ = {sp.delta * 1e3:+.1f} m°, Rwp = {sp.fit.r_wp:.3f}")
        self.ax_zoom.set_xlim(min(sp.fit.x.min() for sp in self.sub_result.peaks[:1]),
                              max(sp.fit.x.max() for sp in self.sub_result.peaks[:1]))
        self.ax_zoom.legend(fontsize=8)
        self.ax_zoom.set_title("Substrate fit (zoom with the toolbar to the other peaks)")
        self.ax_res.set_ylabel("res./σ")
        self.ax_res.axhline(0, color="k", lw=0.5)
        self.ax_res.yaxis.set_major_locator(MaxNLocator(3))

    def _draw_zoom(self, lo, hi):
        self.ax_zoom.clear()
        self.ax_res.clear()
        x = self.corrected_x()
        m = (x >= lo) & (x <= hi)
        self.ax_zoom.semilogy(x[m], np.clip(self.scan.intensity[m], 0.5, None), "k.", ms=2)
        fr = self.last_fit
        if fr is not None and self.correction.apply(fr.x).min() >= lo - 1e-9:
            self._draw_fit(fr, "tab:red", "fit")
            fp = self.film_peaks[-1]["peak"] if self.film_peaks else None
            if fp:
                self.ax_zoom.axvline(fp.two_theta, color="tab:red", ls=":")
                self.ax_zoom.set_title(f"{fp.phase} {fp.hkl_label}: 2θ = {fp.two_theta:.4f}°, "
                                       f"FWHM = {fp.fwhm:.3f}°, c = {fp.lattice:.4f} Å, "
                                       f"R_wp = {fr.r_wp:.3f}", fontsize=9)
        self.ax_zoom.set_xlim(lo, hi)
        self.ax_res.axhline(0, color="k", lw=0.5)
        self.ax_res.set_ylabel("res./σ")
        self.ax_res.yaxis.set_major_locator(MaxNLocator(3))
        self.ax_res.set_xlabel("2θ (°)")

    def _update_tables(self):
        rows = []
        if self.sub_result is not None:
            for sp in self.sub_result.peaks:
                rows.append([sp.label, sp.tt_calc, sp.tt_meas, sp.delta, sp.fit.main.fwhm, sp.fit.r_wp])
        self.tab_sub.set_rows(rows)
        self.tab_film.set_rows([[fp["peak"].phase, fp["peak"].hkl_label, fp["peak"].two_theta,
                                 fp["peak"].two_theta_err, fp["peak"].fwhm, fp["peak"].d,
                                 fp["peak"].lattice, fp["peak"].lattice_err] for fp in self.film_peaks])
        lines = []
        for phase in dict.fromkeys(fp["peak"].phase for fp in self.film_peaks):
            pk = [fp["peak"] for fp in self.film_peaks if fp["peak"].phase == phase]
            c, ce = combine_lattice(pk)
            txt = f"<b>{phase}</b>: c = {c:.4f} ± {ce:.4f} Å from {len(pk)} peak(s)"
            if phase in materials.film_names():
                f = materials.get_film(phase)
                txt += f", ε_zz = {out_of_plane_strain(c, f.c) * 100:+.2f} % (bulk {f.c:.4f} Å)"
            lines.append(txt)
        self.lbl_film.setText("<br>".join(lines))

    # --------------------------------------------------------------- export
    def _base(self):
        p = Path(self.scan.path) if self.scan and self.scan.path else Path.home() / "scan"
        return p.with_suffix("")

    def results_frame(self) -> pd.DataFrame:
        rows = []
        if self.sub_result is not None:
            for sp in self.sub_result.peaks:
                rows.append({"type": "substrate", "phase": self.sub_result.substrate.label, "hkl": sp.label,
                             "two_theta_calc": sp.tt_calc, "two_theta_meas": sp.tt_meas,
                             "two_theta_corr": float(self.correction.apply(sp.tt_meas)),
                             "fwhm": sp.fit.main.fwhm, "r_wp": sp.fit.r_wp})
        for fp in self.film_peaks:
            p = fp["peak"]
            rows.append({"type": "film", "phase": p.phase, "hkl": p.hkl_label,
                         "two_theta_meas": fp["raw_center"], "two_theta_corr": p.two_theta,
                         "two_theta_err": p.two_theta_err, "fwhm": p.fwhm, "d": p.d,
                         "lattice": p.lattice, "lattice_err": p.lattice_err, "r_wp": fp["fit"].r_wp})
        df = pd.DataFrame(rows)
        df.attrs["correction"] = self.correction.describe()
        return df

    def on_save_results(self):
        if self.scan is None:
            return
        p = save_file(self, "Save results", f"{self._base()}_xrd_results.csv", "CSV (*.csv)")
        if not p:
            return
        df = self.results_frame()
        with open(p, "w", encoding="utf-8") as f:
            f.write(f"# file: {self.scan.path}\n# correction: {self.correction.describe()}\n")
            df.to_csv(f, index=False)

    def on_save_xy(self):
        if self.scan is None:
            return
        p = save_file(self, "Save corrected scan", f"{self._base()}_corrected.xy", "xy (*.xy)")
        if p:
            np.savetxt(p, np.column_stack([self.corrected_x(), self.scan.intensity]), fmt="%.5f",
                       header=f"2theta_corrected intensity  ({self.correction.describe()})")

    def on_save_fig(self):
        if self.scan is None:
            return
        p = save_file(self, "Save figure", f"{self._base()}_xrd.png", "PNG (*.png);;PDF (*.pdf)")
        if p:
            self.plot.figure.savefig(p, dpi=300)
