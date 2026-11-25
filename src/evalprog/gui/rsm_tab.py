"""RSM tab: reciprocal space maps, substrate offset correction, strain."""

from __future__ import annotations

from pathlib import Path

import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.widgets import RectangleSelector
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QPushButton, QSplitter, QWidget
from scipy.ndimage import gaussian_filter

from evalprog import materials
from evalprog.core.rsm import analysis as ra
from evalprog.core.xrd import io as xio

from .widgets import (PlotWidget, ResultsTable, combo, group, hbox, info_label, ispin,
                      open_files, save_file, scroll_panel, show_error, spin)


class RSMTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rsm: ra.RSMap | None = None
        self.path = None
        self.offset = None
        self.peaks = []   # dicts: phase, hkl, peak, strain
        self.roi = None
        self.selector = None
        self._cbar = None
        self._build()

    def _build(self):
        panel, lay = scroll_panel()
        self.btn_load = QPushButton("Load RSM (.xrdml / .csv) …")
        self.btn_load.clicked.connect(self.on_load)
        self.lbl_file = info_label("No file loaded.")
        lay.addWidget(self.btn_load)
        lay.addWidget(self.lbl_file)

        self.cb_scale = combo(["log10", "linear"])
        self.cb_smooth = combo(["raw", "smooth 0.6", "smooth 1.0"])
        self.cb_cmap = combo(["magma", "inferno", "viridis", "plasma", "cividis", "turbo", "gray"])
        self.sp_vmin = spin(0, -10, 1e9, 2)
        self.sp_vmax = spin(0, -10, 1e9, 2)
        self.sp_vmin.setSpecialValueText("auto")
        self.sp_vmax.setSpecialValueText("auto")
        self.sp_vmin.setMinimum(0)
        self.sp_vmax.setMinimum(0)
        for w in (self.cb_scale, self.cb_smooth, self.cb_cmap):
            w.currentIndexChanged.connect(self.redraw)
        for w in (self.sp_vmin, self.sp_vmax):
            w.valueChanged.connect(self.redraw)
        lay.addWidget(group("Display", [("Scale", self.cb_scale), ("Smoothing", self.cb_smooth),
                                         ("Colormap", self.cb_cmap),
                                         ("min / max", hbox(self.sp_vmin, self.sp_vmax))]))

        self.cb_sub = combo(materials.substrate_names(), "SrTiO3")
        self.cb_orient = combo([])
        self.cb_dir = combo([])
        self.cb_sub.currentTextChanged.connect(self._update_orient)
        self.cb_orient.currentTextChanged.connect(self._update_dirs)
        self._update_orient()
        self.sh, self.sk, self.sl = ispin(1, -20, 20), ispin(0, -20, 20), ispin(3, -40, 40)
        self.btn_offset = QPushButton("Find substrate peak && correct offset")
        self.btn_offset.clicked.connect(self.on_offset)
        self.btn_offset_roi = QPushButton("Use current ROI as substrate peak")
        self.btn_offset_roi.clicked.connect(lambda: self.on_offset(use_roi=True))
        self.btn_reset = QPushButton("Remove correction")
        self.btn_reset.clicked.connect(self.on_reset_offset)
        self.lbl_offset = info_label("")
        lay.addWidget(group("Substrate reference", [
            ("Substrate", self.cb_sub), ("Orientation", self.cb_orient), ("In-plane dir.", self.cb_dir),
            ("h k l", hbox(self.sh, self.sk, self.sl)), self.btn_offset, self.btn_offset_roi,
            self.btn_reset, self.lbl_offset]))

        self.cb_phase = combo(materials.film_names(), "SrRuO3", editable=True)
        self.fh, self.fk, self.fl = ispin(1, -20, 20), ispin(0, -20, 20), ispin(3, -40, 40)
        self.cb_weight = combo(["poisson", "log", "none"])
        self.chk_sub = QCheckBox("subtract fitted substrate peak")
        self.chk_sub.setChecked(True)
        self.chk_robust = QCheckBox("robust loss (outliers, streaks)")
        self.btn_fit = QPushButton("Fit ROI (drag a rectangle)")
        self.btn_fit.clicked.connect(self.on_fit)
        self.btn_del = QPushButton("Remove selected")
        self.btn_del.clicked.connect(self.on_delete)
        self.lbl_fit = info_label("")
        lay.addWidget(group("Film peak", [
            ("Phase", self.cb_phase), ("h k l (film cell)", hbox(self.fh, self.fk, self.fl)),
            ("Weighting", self.cb_weight), self.chk_sub, self.chk_robust,
            hbox(self.btn_fit, self.btn_del), self.lbl_fit]))

        self.btn_save = QPushButton("Save results (.csv) and figure")
        self.btn_save.clicked.connect(self.on_save)
        lay.addWidget(self.btn_save)

        right = QSplitter(Qt.Vertical)
        self.plot = PlotWidget(figsize=(9, 8))
        gs = self.plot.figure.add_gridspec(2, 2, height_ratios=[2.4, 1])
        self.ax = self.plot.figure.add_subplot(gs[0, :])
        self.ax_cx = self.plot.figure.add_subplot(gs[1, 0])
        self.ax_cz = self.plot.figure.add_subplot(gs[1, 1])
        right.addWidget(self.plot)
        self.table = ResultsTable(["Phase", "hkl", "Qx", "Qz", "a_ip (Å)", "c (Å)", "ε_xx %", "ε_zz %",
                                   "R %", "a_relaxed", "R(ν) %", "ν_eff", "R²", "rel.RMSE", "χ²_red"],
                                  formats=[None, None, ".4f", ".4f", ".4f", ".4f", "+.3f", "+.3f", ".1f",
                                           ".4f", ".1f", ".3f", ".4f", ".4f", ".2f"])
        self.table.itemSelectionChanged.connect(self._show_cuts)
        right.addWidget(self.table)
        right.setSizes([700, 180])
        main = QHBoxLayout(self)
        main.addWidget(panel)
        main.addWidget(right, 1)

    def _update_orient(self):
        self.cb_orient.clear()
        self.cb_orient.addItems(materials.orientations(self.cb_sub.currentText()))
        self._update_dirs()

    def _update_dirs(self):
        self.cb_dir.clear()
        name, o = self.cb_sub.currentText(), self.cb_orient.currentText()
        if name and o:
            s = materials.get_substrate(name, o)
            self.cb_dir.addItems([f"{d.label}  ({d.spacing:.4f} Å)" for d in s.inplane])

    def substrate(self):
        return materials.get_substrate(self.cb_sub.currentText(), self.cb_orient.currentText())

    def direction(self):
        t = self.cb_dir.currentText()
        return t.split("  (")[0] if t else None

    # ------------------------------------------------------------- loading
    def on_load(self):
        paths = open_files(self, "Open RSM", "RSM (*.xrdml *.csv);;All files (*)", key="rsm_dir")
        if paths:
            self.load(paths[0])

    def load(self, path):
        try:
            area = xio.load_area(path)
        except Exception as exc:
            show_error(self, "Could not read RSM", exc)
            return
        self.rsm = ra.RSMap.from_area(area)
        self.path = Path(path)
        self.offset = None
        self.peaks = []
        self.roi = None
        self.lbl_file.setText(f"{self.path.name}: {area.intensity.shape[0]} × {area.intensity.shape[1]} points, "
                              f"λ = {area.wavelength:.5f} Å")
        self.lbl_offset.setText("")
        self.redraw(keep_view=False)

    # ------------------------------------------------------------- drawing
    def _display(self):
        z = np.array(self.rsm.intensity, float)
        if self.cb_scale.currentText() == "log10":
            z = np.log10(np.clip(z, 0, None) + 1)
        sm = self.cb_smooth.currentText()
        if sm != "raw":
            z = gaussian_filter(z, float(sm.split()[1]))
        return z

    def redraw(self, keep_view=True):
        ax = self.ax
        view = (ax.get_xlim(), ax.get_ylim()) if (keep_view and getattr(self, "_has_view", False)) else None
        ax.clear()
        if self.rsm is None:
            self.plot.draw()
            return
        qx, qz = self.rsm.q()
        z = self._display()
        fin = z[np.isfinite(z)]
        vmin = self.sp_vmin.value() or float(np.percentile(fin, 1))
        vmax = self.sp_vmax.value() or float(np.percentile(fin, 99.7))
        cmap = self.cb_cmap.currentText()
        try:
            mesh = ax.pcolormesh(qx, qz, z, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                                 rasterized=True)
        except Exception:
            tri = mtri.Triangulation(qx.ravel(), qz.ravel())
            mesh = ax.tripcolor(tri, z.ravel(), cmap=cmap, vmin=vmin, vmax=vmax)
        if self._cbar is None:
            self._cbar = self.plot.figure.colorbar(mesh, ax=ax)
        else:
            self._cbar.update_normal(mesh)
        self._cbar.set_label("log10(I + 1)" if self.cb_scale.currentText() == "log10" else "I")
        ax.set_xlabel("Qx (1/Å)")
        ax.set_ylabel("Qz (1/Å)")
        title = self.path.name if self.path else ""
        if self.offset is not None:
            title += f"   (corrected: Δ2θ = {self.offset.d_two_theta:+.4f}°, Δω = {self.offset.d_omega:+.4f}°)"
        ax.set_title(title, fontsize=9)
        if self.offset is not None:
            ax.plot(self.offset.qx_expected, self.offset.qz_expected, "+", color="cyan", ms=14, mew=1.5)
            self._draw_peak(self.offset.peak, "cyan", "substrate")
        for p in self.peaks:
            self._draw_peak(p["peak"], "red", f"{p['phase']} {p['hkl_label']}")
        if self.roi is not None:
            x1, x2, z1, z2 = self.roi
            ax.add_patch(Rectangle((x1, z1), x2 - x1, z2 - z1, fill=False, ec="white", ls="--", lw=1))
        if view is None:
            pad_x, pad_z = 0.02 * np.ptp(qx), 0.02 * np.ptp(qz)
            view = ((np.nanmin(qx) - pad_x, np.nanmax(qx) + pad_x), (np.nanmin(qz) - pad_z, np.nanmax(qz) + pad_z))
        ax.set_xlim(*view[0])
        ax.set_ylim(*view[1])
        self._has_view = True
        self.selector = RectangleSelector(ax, self._on_select, useblit=True, button=[1],
                                          minspanx=1e-4, minspany=1e-4, spancoords="data",
                                          interactive=True)
        ax.set_autoscale_on(False)
        self.plot.draw()

    def _draw_peak(self, pk, color, label):
        ax = self.ax
        ax.plot(pk.qx, pk.qz, "o", ms=5, mec="black", mfc=color)
        ax.add_patch(Ellipse((pk.qx, pk.qz), 4 * pk.hwhm_x, 4 * pk.hwhm_z, angle=pk.theta_deg,
                             fill=False, ec=color, lw=1.2))
        ax.annotate(label, (pk.qx, pk.qz), xytext=(8, 6), textcoords="offset points", fontsize=8,
                    color="black", bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))

    def _on_select(self, e1, e2):
        if e1.xdata is None or e2.xdata is None:
            return
        self.roi = (min(e1.xdata, e2.xdata), max(e1.xdata, e2.xdata),
                    min(e1.ydata, e2.ydata), max(e1.ydata, e2.ydata))

    # ------------------------------------------------------------- actions
    def _hkl_sub(self):
        return (self.sh.value(), self.sk.value(), self.sl.value())

    def _fit_kw(self):
        return dict(weighting=self.cb_weight.currentText(), robust=self.chk_robust.isChecked())

    def on_offset(self, use_roi=False):
        if self.rsm is None:
            return
        if use_roi and self.roi is None:
            show_error(self, "No ROI", ValueError("Drag a rectangle around the substrate peak first."))
            return
        try:
            roi = None
            if use_roi:
                # ROI was drawn in the currently displayed (maybe corrected) coordinates
                qx, qz = self.rsm.q()
                x1, x2, z1, z2 = self.roi
                m = (qx >= x1) & (qx <= x2) & (qz >= z1) & (qz <= z2)
                self.rsm.d_two_theta = self.rsm.d_omega = 0.0
                qx0, qz0 = self.rsm.q()
                roi = (qx0[m].min(), qx0[m].max(), qz0[m].min(), qz0[m].max())
            self.offset = ra.correct_offset(self.rsm, self.substrate(), self._hkl_sub(), self.direction(),
                                            roi=roi, **self._fit_kw())
        except Exception as exc:
            show_error(self, "Offset correction failed", exc)
            return
        pk = self.offset.peak
        self.lbl_offset.setText(
            f"Δ2θ = {self.offset.d_two_theta:+.4f}°, Δω = {self.offset.d_omega:+.4f}°<br>"
            f"expected Q = ({self.offset.qx_expected:.4f}, {self.offset.qz_expected:.4f}) 1/Å<br>"
            f"substrate fit: R² = {pk.r2:.4f}, rel. RMSE = {pk.rel_rmse:.4f}")
        self.roi = None
        self._refit_all()
        self.redraw(keep_view=False)

    def on_reset_offset(self):
        if self.rsm is None:
            return
        self.rsm.d_two_theta = self.rsm.d_omega = 0.0
        self.offset = None
        self.lbl_offset.setText("")
        self._refit_all()
        self.redraw(keep_view=False)

    def _refit_all(self):
        """Peaks were fitted in the old coordinates: refit them in the new ones."""
        new = []
        for p in self.peaks:
            try:
                new.append(self._fit(p["roi_angles"], p["phase"], p["hkl"]))
            except Exception:
                pass
        self.peaks = new
        self._update_table()

    def _roi_to_angles(self, roi):
        """Store a ROI as the set of point indices (independent of the correction)."""
        qx, qz = self.rsm.q()
        x1, x2, z1, z2 = roi
        return (qx >= x1) & (qx <= x2) & (qz >= z1) & (qz <= z2)

    def _fit(self, mask, phase, hkl):
        qx, qz = self.rsm.q()
        y = self.rsm.intensity[mask].astype(float)
        if self.chk_sub.isChecked() and self.offset is not None:
            p = np.array(self.offset.peak.params, float)
            p[7:] = 0
            y = y - ra.pv2d(qx[mask], qz[mask], *p)
        pk = ra.fit_peak_2d(qx[mask], qz[mask], y, **self._fit_kw())
        film = materials.get_film(phase) if phase in materials.film_names() else None
        a_sub = self.substrate().inplane_spacing(self.direction())
        st = ra.strain_analysis(pk, hkl, film, a_substrate=a_sub)
        return {"phase": phase, "hkl": hkl, "hkl_label": "({} {} {})".format(*hkl), "peak": pk,
                "strain": st, "roi_angles": mask}

    def on_fit(self):
        if self.rsm is None or self.roi is None:
            show_error(self, "No ROI", ValueError("Drag a rectangle around the film peak first."))
            return
        try:
            mask = self._roi_to_angles(self.roi)
            res = self._fit(mask, self.cb_phase.currentText().strip() or "film",
                            (self.fh.value(), self.fk.value(), self.fl.value()))
        except Exception as exc:
            show_error(self, "Fit failed", exc)
            return
        self.peaks.append(res)
        pk, st = res["peak"], res["strain"]
        txt = (f"a_ip = {st.a_inplane:.4f} ± {st.a_inplane_err:.4f} Å, c = {st.c_out:.4f} ± {st.c_out_err:.4f} Å"
               f"<br>R² = {pk.r2:.4f}, rel. RMSE = {pk.rel_rmse:.4f}, χ²_red = {pk.chi2_red:.2f}")
        if np.isfinite(st.relaxation):
            txt += (f"<br>ε_xx = {st.eps_xx * 100:+.2f} %, ε_zz = {st.eps_zz * 100:+.2f} %, "
                    f"relaxation R = {st.relaxation * 100:.1f} %")
        self.lbl_fit.setText(txt)
        self.roi = None
        self._update_table()
        self.table.selectRow(len(self.peaks) - 1)
        self.redraw()

    def on_delete(self):
        i = self.table.selected_row()
        if i is not None and 0 <= i < len(self.peaks):
            del self.peaks[i]
            self._update_table()
            self.redraw()

    def _update_table(self):
        rows = []
        for p in self.peaks:
            pk, st = p["peak"], p["strain"]
            rows.append([p["phase"], p["hkl_label"], pk.qx, pk.qz, st.a_inplane, st.c_out,
                         st.eps_xx * 100, st.eps_zz * 100, st.relaxation * 100, st.a_relaxed,
                         st.relaxation_poisson * 100, st.poisson_eff, pk.r2, pk.rel_rmse, pk.chi2_red])
        self.table.set_rows(rows)

    def _show_cuts(self):
        i = self.table.selected_row()
        for ax in (self.ax_cx, self.ax_cz):
            ax.clear()
        if i is None or i >= len(self.peaks) or self.rsm is None:
            self.plot.draw()
            return
        pk = self.peaks[i]["peak"]
        (x, yx, mx), (z, yz, mz) = ra.line_cuts(self.rsm, pk)
        for ax, q, y, m, lab in ((self.ax_cx, x, yx, mx, "Qx"), (self.ax_cz, z, yz, mz, "Qz")):
            ax.semilogy(q, np.clip(y, 0.5, None), ".", ms=3, color="0.3", label="data")
            ax.semilogy(q, np.clip(m, 0.5, None), "-", color="tab:red", lw=1.2, label="fit")
            ax.axvline(pk.qx if lab == "Qx" else pk.qz, color="k", ls="--", lw=0.7)
            ax.set_xlabel(f"{lab} (1/Å)")
            ax.legend(fontsize=7)
        self.ax_cx.set_ylabel("Intensity")
        self.plot.draw()

    # ------------------------------------------------------------- export
    def results_frame(self):
        rows = []
        for p in self.peaks:
            pk, st = p["peak"], p["strain"]
            rows.append({"phase": p["phase"], "hkl": p["hkl_label"], "qx": pk.qx, "qz": pk.qz,
                         "qx_err": pk.qx_err, "qz_err": pk.qz_err, "a_inplane": st.a_inplane,
                         "a_inplane_err": st.a_inplane_err, "c_out": st.c_out, "c_out_err": st.c_out_err,
                         "eps_xx": st.eps_xx, "eps_zz": st.eps_zz, "relaxation": st.relaxation,
                         "a_relaxed_poisson": st.a_relaxed, "relaxation_poisson": st.relaxation_poisson,
                         "poisson_eff": st.poisson_eff, "a_substrate": st.a_substrate,
                         "hwhm_x": pk.hwhm_x, "hwhm_z": pk.hwhm_z, "tilt_deg": pk.theta_deg, "eta": pk.eta,
                         "r2": pk.r2, "rel_rmse": pk.rel_rmse, "chi2_red": pk.chi2_red,
                         "weighting": pk.weighting})
        return pd.DataFrame(rows)

    def on_save(self):
        if self.rsm is None:
            return
        base = self.path.with_suffix("") if self.path else Path.home() / "rsm"
        p = save_file(self, "Save RSM results", f"{base}_rsm_results.csv", "CSV (*.csv)")
        if not p:
            return
        with open(p, "w", encoding="utf-8") as f:
            f.write(f"# file: {self.path}\n")
            if self.offset is not None:
                f.write(f"# substrate: {self.substrate().label}, hkl {self._hkl_sub()}, "
                        f"d2theta = {self.offset.d_two_theta:.5f} deg, domega = {self.offset.d_omega:.5f} deg\n")
            self.results_frame().to_csv(f, index=False)
        self.plot.figure.savefig(Path(p).with_suffix(".png"), dpi=250)
