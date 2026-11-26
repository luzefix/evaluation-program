"""AFM tab: topography images (Bruker NanoScope .spm, ASCII matrices)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QHBoxLayout, QPushButton, QSplitter, QTabWidget,
                               QWidget)

from evalprog.core import afm

from evalprog.db.store import Curve, Measurement, Result

from .db_widgets import SaveToDB, file_date
from .widgets import (PlotWidget, ResultsTable, combo, group, hbox, info_label, ispin,
                      open_files, save_file, scroll_panel, show_error, spin)

CMAPS = ["afmhot", "viridis", "gray", "cividis", "inferno", "RdBu_r"]
ROWS = ["none"] + list(afm.ROW_METHODS)
ZERO = ["median", "minimum", "mean", "none"]


class AFMTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.file = None
        self.channel = None
        self.result = None          # afm.Corrected
        self.features = None
        self.hist = None
        self.profile = None         # (s, h, p0, p1)
        self.prof_steps = None      # afm.ProfileSteps
        self.levelfit = None        # afm.LevelFit
        self.auto = None            # afm.AutoProfile
        self.roi = None             # (x0, x1, y0, y1) in nm
        self.results = []
        self._drag = None
        self._line = None
        self._build()

    # --------------------------------------------------------------- layout
    def _build(self):
        panel, lay = scroll_panel()
        self.btn_load = QPushButton("Load image …")
        self.btn_load.clicked.connect(self.on_load)
        self.cb_channel = combo([])
        self.cb_channel.currentIndexChanged.connect(self.on_channel)
        self.lbl_file = info_label("")
        lay.addWidget(group("File", [self.btn_load, ("Channel", self.cb_channel), self.lbl_file]))

        self.chk_scars = QCheckBox("Remove scars (streaks along fast axis)")
        self.chk_scars.setChecked(True)
        self.sp_scar = spin(3.0, 0.5, 50, 1, 0.5, " σ")
        self.sp_scar_len = ispin(12, 1, 10000)
        self.cb_bg = combo(afm.BACKGROUNDS, "plane")
        self.cb_bow = combo(["0", "2", "3"], "2")
        self.cb_bow.setToolTip("Only for 'terraces (facet level)': 2D polynomial (piezo bow) fitted "
                               "inside the terraces with an own offset per terrace; 0 = off")
        self.cb_rows = combo(ROWS, "median")
        self.chk_mask = QCheckBox("Exclude pits/particles from fits")
        self.chk_mask.setChecked(True)
        self.sp_mask = spin(2.5, 0.5, 20, 1, 0.5, " σ")
        self.chk_outl = QCheckBox("Remove spikes")
        self.sp_outl = spin(6.0, 1, 50, 1, 0.5, " σ")
        self.cb_smooth = combo(afm.SMOOTHING)
        self.sp_smooth = spin(5.0, 0.1, 1e4, 1, 1, " nm")
        self.cb_zero = combo(ZERO, "median")
        for w in (self.chk_scars, self.chk_mask, self.chk_outl):
            w.toggled.connect(self.apply)
        for w in (self.sp_scar, self.sp_scar_len, self.sp_mask, self.sp_outl, self.sp_smooth):
            w.valueChanged.connect(self.apply)
        for w in (self.cb_bg, self.cb_bow, self.cb_rows, self.cb_zero, self.cb_smooth):
            w.currentIndexChanged.connect(self.apply)
        lay.addWidget(group("Corrections (applied in this order)", [
            self.chk_scars, ("Scar threshold / min. length", hbox(self.sp_scar, self.sp_scar_len)),
            ("Background", self.cb_bg), ("Terrace bow order", self.cb_bow),
            ("Line alignment", self.cb_rows),
            self.chk_mask, ("Mask threshold", self.sp_mask),
            self.chk_outl, ("Spike threshold", self.sp_outl),
            ("Smoothing (σ / window)", hbox(self.cb_smooth, self.sp_smooth)),
            ("Zero level", self.cb_zero)]))
        lay.addWidget(info_label(
            "Terraces / step edges: use 'terraces (facet level)' and 'median of differences'. "
            "Rough films and particles: 'plane' or 'polynomial 2' and 'median'. "
            "Smoothing reduces noise for step heights but lowers Sq."))

        self.cb_cmap = combo(CMAPS)
        self.sp_lo = spin(1.0, 0, 49, 1, 0.5, " %")
        self.sp_hi = spin(99.5, 51, 100, 1, 0.5, " %")
        self.chk_showmask = QCheckBox("Show excluded area")
        for w in (self.sp_lo, self.sp_hi):
            w.valueChanged.connect(self.redraw)
        self.cb_cmap.currentIndexChanged.connect(self.redraw)
        self.chk_showmask.toggled.connect(self.redraw)
        lay.addWidget(group("Display", [("Colour map", self.cb_cmap),
                                        ("Contrast (percentiles)", hbox(self.sp_lo, self.sp_hi)),
                                        self.chk_showmask]))

        self.cb_mouse = combo(["profile line", "region (rectangle)"])
        self.btn_clear_roi = QPushButton("Clear region")
        self.btn_clear_roi.clicked.connect(self.clear_roi)
        self.sp_width = ispin(3, 1, 5000)
        self.sp_width.valueChanged.connect(self._width_changed)
        self.sp_ref = spin(0.3905, 0.01, 100, 4, 0.01, " nm")
        self.sp_bins = ispin(300, 20, 5000)
        self.sp_prom = spin(5.0, 0.5, 90, 1, 1, " %")
        self.chk_relevel = QCheckBox("Re-level terraces inside region")
        self.sp_minstep = spin(0.195, 0.001, 100, 3, 0.05, " nm")
        self.btn_auto = QPushButton("Profile ⟂ steps (auto)")
        self.btn_auto.setToolTip("Line through the centre perpendicular to the step edges "
                                 "(structure tensor), width = 0.4 × terrace width")
        self.btn_auto.clicked.connect(self.on_auto_profile)
        self.sp_edge = spin(30.0, 0, 1e4, 1, 5, " nm")
        for w in (self.sp_ref, self.sp_bins, self.sp_prom):
            w.valueChanged.connect(self.analyse)
        self.chk_relevel.toggled.connect(self.analyse)
        for w in (self.sp_minstep, self.sp_edge):
            w.valueChanged.connect(self._width_changed)
        lay.addWidget(group("Steps and terraces", [
            ("Mouse in image", hbox(self.cb_mouse, self.btn_clear_roi)),
            self.btn_auto,
            info_label("Profile line: drag across the steps (perpendicular). The profile is "
                       "averaged over 'width' pixels along the steps and fitted with a "
                       "staircase. Region: the histogram and the level fit use only the "
                       "rectangle (e.g. 2–4 clean terraces). Toolbar zoom/pan must be off."),
            ("Profile width (px, along steps)", self.sp_width),
            ("Min. step / edge exclusion", hbox(self.sp_minstep, self.sp_edge)),
            ("Reference step (unit cell)", self.sp_ref),
            ("Histogram bins", self.sp_bins), ("Level detection: prominence", self.sp_prom),
            self.chk_relevel]))

        self.cb_kind = combo(["pits", "islands"])
        self.sp_thr = spin(1.0, 0.001, 1e4, 3, 0.1, " nm")
        self.sp_minpx = ispin(4, 1, 100000)
        self.sp_win = spin(250.0, 0, 1e6, 0, 50, " nm")
        self.sp_win.setSpecialValueText("global median")
        self.btn_feat = QPushButton("Find pits / islands")
        self.btn_feat.clicked.connect(self.on_features)
        lay.addWidget(group("Pits / islands (relative to local level)", [
            ("Type", self.cb_kind), ("Threshold", self.sp_thr), ("Min. size (px)", self.sp_minpx),
            ("Local reference window", self.sp_win), self.btn_feat]))

        self.btn_save_z = QPushButton("Save corrected image (.txt, Gwyddion)")
        self.btn_save_z.clicked.connect(self.on_save_image)
        self.btn_save_res = QPushButton("Save results (.csv)")
        self.btn_save_res.clicked.connect(self.on_save_results)
        self.btn_save_prof = QPushButton("Save profile (.csv)")
        self.btn_save_prof.clicked.connect(self.on_save_profile)
        self.btn_save_fig = QPushButton("Save figures (.png)")
        self.btn_save_fig.clicked.connect(self.on_save_figs)
        lay.addWidget(group("Export", [self.btn_save_z, self.btn_save_prof, self.btn_save_res,
                                       self.btn_save_fig]))
        self.db = SaveToDB("AFM", self.db_record, self)
        lay.addWidget(self.db.group)
        self.lbl_log = info_label("")
        lay.addWidget(self.lbl_log)

        self.tabs = QTabWidget()
        self.plot = PlotWidget(figsize=(8, 9))
        gs = self.plot.figure.add_gridspec(2, 1, height_ratios=[3, 1])
        self.ax_img = self.plot.figure.add_subplot(gs[0])
        self.ax_prof = self.plot.figure.add_subplot(gs[1])
        self.cbar = None
        self.plot.canvas.mpl_connect("button_press_event", self._press)
        self.plot.canvas.mpl_connect("motion_notify_event", self._motion)
        self.plot.canvas.mpl_connect("button_release_event", self._release)
        self.tabs.addTab(self.plot, "Image && profile")
        self.plot_hist = PlotWidget()
        self.ax_hist = self.plot_hist.figure.add_subplot(111)
        self.tabs.addTab(self.plot_hist, "Histogram")
        self.plot_psd = PlotWidget()
        self.ax_psd = self.plot_psd.figure.add_subplot(111)
        self.tabs.addTab(self.plot_psd, "PSD")
        self.feat_table = ResultsTable(["#", "x (nm)", "y (nm)", "Area (nm²)", "d_eq (nm)",
                                        "Depth/height (nm)", "Volume (nm³)"],
                                       formats=[None, ".0f", ".0f", ".0f", ".1f", ".3f", ".0f"])
        self.tabs.addTab(self.feat_table, "Pits / islands")

        right = QSplitter(Qt.Vertical)
        right.addWidget(self.tabs)
        self.table = ResultsTable(["Quantity", "Value", "±", "Unit", "Note"])
        right.addWidget(self.table)
        right.setSizes([700, 220])
        main = QHBoxLayout(self)
        main.addWidget(panel)
        main.addWidget(right, 1)

    # --------------------------------------------------------------- data
    def on_load(self):
        paths = open_files(self, "Open AFM image",
                           "AFM (*.spm *.0* *.txt *.asc *.dat);;All files (*)", key="afm_dir")
        if paths:
            self.load(paths[0])

    def load(self, path):
        try:
            self.file = afm.load_afm(path)
        except Exception as exc:
            show_error(self, "Load AFM image", exc)
            return
        self.cb_channel.blockSignals(True)
        self.cb_channel.clear()
        self.cb_channel.addItems([f"{i + 1}: {c.label} [{c.unit}]" for i, c in enumerate(self.file.channels)])
        h = self.file.height_channel()
        self.cb_channel.setCurrentIndex(self.file.channels.index(h))
        self.cb_channel.blockSignals(False)
        hdr = self.file.header
        info = ", ".join(f"{k}: {v}" for k, v in hdr.items() if k in ("Date", "Microscope", "Scan Rate"))
        self.lbl_file.setText(f"{Path(path).name}\n{info}")
        self.profile = None
        self.prof_steps = None
        self.roi = None
        self.db.set_source(path)
        self.on_channel()

    def on_channel(self, *_):
        if self.file is None or self.cb_channel.currentIndex() < 0:
            return
        self.channel = self.file.channels[self.cb_channel.currentIndex()]
        self.features = None
        self.apply()

    def corrections(self) -> afm.Corrections:
        return afm.Corrections(
            scars=self.chk_scars.isChecked(), scar_threshold=self.sp_scar.value(),
            scar_length=self.sp_scar_len.value(), background=self.cb_bg.currentText(),
            rows=self.cb_rows.currentText(), mask_features=self.chk_mask.isChecked(),
            mask_sigma=self.sp_mask.value(), outliers=self.chk_outl.isChecked(),
            outlier_sigma=self.sp_outl.value(), smooth=self.cb_smooth.currentText(),
            smooth_nm=self.sp_smooth.value(), zero=self.cb_zero.currentText(),
            terrace_bow=int(self.cb_bow.currentText()))

    def apply(self, *_):
        if self.channel is None:
            return
        try:
            self.result = afm.correct(self.channel.data, self.corrections(), self.channel.dx,
                                      self.channel.dy)
        except Exception as exc:
            show_error(self, "AFM corrections", exc)
            return
        self.lbl_log.setText("\n".join(self.result.log))
        self.features = None
        if self.profile is not None:
            self._compute_profile(self.profile[2], self.profile[3])
        self.analyse()

    # --------------------------------------------------------------- analysis
    def analysis_image(self):
        """Corrected image, or the selected region of it."""
        z = self.result.z
        if self.roi is None:
            return z
        sub = afm.crop(z, self.channel.dx, self.channel.dy, self.roi)
        if sub.size < 16:
            return z
        if self.chk_relevel.isChecked():
            sub = afm.level_terraces(sub)
        return sub

    def analyse(self, *_):
        if self.result is None:
            return
        ch, z = self.channel, self.result.z
        u = ch.unit
        rows = []
        r = afm.roughness(z)
        rm = afm.roughness(z, self.result.mask) if self.result.mask.any() else None
        for k, note in (("Sq", "RMS roughness"), ("Sa", "mean abs. deviation"), ("Sz", "peak to valley"),
                        ("Ssk", "skewness (<0: pits dominate)"), ("Sku", "kurtosis (3 = Gaussian)")):
            rows.append((k, r[k], "", "" if k in ("Ssk", "Sku") else u, note))
        if rm is not None:
            rows.append(("Sq (without excluded features)", rm["Sq"], "", u, "terrace / matrix roughness"))
        try:
            rows.append(("Correlation length (fast axis)", afm.correlation_length(z, ch.dx), "", "nm",
                         "ACF = 1/e"))
        except Exception:
            pass
        ref = self.sp_ref.value()
        za = self.analysis_image()
        where = "region" if za is not z else "whole image"
        self.hist = afm.height_histogram(za, bins=self.sp_bins.value(), prominence=self.sp_prom.value() / 100)
        self.levelfit = afm.fit_levels(self.hist, ref) if len(self.hist.levels) else None
        lf = self.levelfit
        if lf is not None:
            for i, (c, e, w) in enumerate(zip(lf.centers, lf.errors, lf.widths)):
                rows.append((f"Level {i + 1}", c, e, u, f"{where}, Gaussian fit, σ = {w:.3f} {u}"))
            for i, (st, e) in enumerate(zip(lf.steps, lf.step_errors)):
                rows.append((f"Step {i + 1}→{i + 2} (histogram)", st, e, u, f"= {st / ref:.2f} × reference"))
            if lf.ladder:
                d, e = lf.ladder["step"], lf.ladder["error"]
                rows.append(("Step height, all levels (histogram)", d, e, u,
                             f"= {d / ref:.3f} × reference; linear fit L = L0 + n·d"))
        else:
            for i, lv in enumerate(self.hist.levels):
                rows.append((f"Level {i + 1}", lv, "", u, where))
        if self.profile is not None:
            s_, h_ = self.profile[0], self.profile[1]
            rows.append(("Profile length", s_[-1], "", "nm", f"width {self.sp_width.value()} px"))
            if self.auto is not None and self.auto.p0 == self.profile[2]:
                sd = self.auto.steps
                rows.append(("Step normal (auto)", sd.angle_deg, "", "deg",
                             f"coherence {sd.coherence:.2f}; 0° = +x, 90° = down"))
                rows.append(("Terrace width (mean slope)", sd.terrace_nm, "", "nm",
                             f"reference step / mean slope; miscut {sd.miscut_deg:.3f}°"))
            ps = self.prof_steps
            if ps is not None and len(ps.heights):
                for k, (e_, hgt, err) in enumerate(zip(ps.edges, ps.heights, ps.errors)):
                    rows.append((f"Profile step {k + 1} at {e_:.0f} nm", hgt, err, u,
                                 f"= {hgt / ref:.2f} × reference"))
                hs = np.abs(ps.heights)
                n = len(hs)
                sem = hs.std(ddof=1) / np.sqrt(n) if n > 1 else float(ps.errors[0])
                rows.append(("Mean |step| (profile)", hs.mean(), sem, u,
                             f"{n} steps, ± = std/√N; = {hs.mean() / ref:.3f} × reference"))
                if len(ps.terraces):
                    tw = ps.terraces.mean()
                    tw_e = ps.terraces.std(ddof=1) / np.sqrt(len(ps.terraces)) if len(ps.terraces) > 1 else np.nan
                    rows.append(("Mean terrace width (profile)", tw, tw_e, "nm", "profile ⟂ steps"))
                    if u == "nm":
                        ang = np.degrees(np.arctan(hs.mean() / tw))
                        rows.append(("Miscut angle", ang, "", "deg", "atan(step / terrace width)"))
                rows.append(("Profile fit: residual RMS", ps.residual_rms, "", u,
                             f"residual slope {ps.slope * 1e3:.3g} {u}/µm"))
            else:
                rows.append(("Profile peak-to-valley", float(np.ptp(h_)), "", u, "no steps found"))
        if self.features is not None:
            s = self.features.summary
            k = self.features.kind
            rows.append((f"{k}: count", s["count"], "", "", f"threshold {self.features.threshold:g} {u}"))
            rows.append((f"{k}: density", s["density_um2"], "", "µm⁻²", ""))
            rows.append((f"{k}: area fraction", 100 * s["area_fraction"], "", "%", ""))
            if "mean_d_eq_nm" in s:
                rows.append((f"{k}: mean diameter", s["mean_d_eq_nm"], "", "nm", "equivalent disc"))
                rows.append((f"{k}: mean depth/height", s["mean_extreme_nm"], "", u, "vs. reference level"))
                rows.append((f"{k}: max depth/height", s["max_extreme_nm"], "", u, ""))
        self.results = rows
        self.table.set_rows(rows)
        self.redraw()
        self.db.auto()

    def on_features(self):
        if self.result is None:
            return
        try:
            self.features = afm.find_features(self.result.z, self.channel.dx, self.channel.dy,
                                              self.sp_thr.value(), self.cb_kind.currentText(),
                                              min_pixels=self.sp_minpx.value(),
                                              window_nm=self.sp_win.value())
        except Exception as exc:
            show_error(self, "Pits / islands", exc)
            return
        self.feat_table.set_rows([(r["#"], r["x_nm"], r["y_nm"], r["area_nm2"], r["d_eq_nm"],
                                   r["extreme_nm"], r["volume_nm3"]) for r in self.features.table])
        self.analyse()

    # --------------------------------------------------------------- mouse: profile / region
    def _toolbar_idle(self):
        return not str(self.plot.toolbar.mode)

    def _region_mode(self):
        return self.cb_mouse.currentIndex() == 1

    def _press(self, ev):
        if ev.inaxes is self.ax_img and ev.button == 1 and self._toolbar_idle() and self.result is not None:
            self._drag = (ev.xdata, ev.ydata)

    def _motion(self, ev):
        if self._drag is None or ev.inaxes is not self.ax_img:
            return
        x0, y0 = self._drag
        x1, y1 = ev.xdata, ev.ydata
        xs, ys = ([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0]) if self._region_mode() else ([x0, x1], [y0, y1])
        if self._line is None:
            (self._line,) = self.ax_img.plot(xs, ys, "c-", lw=1.5)
        else:
            self._line.set_data(xs, ys)
        self.plot.draw()

    def _release(self, ev):
        if self._drag is None:
            return
        p0 = self._drag
        self._drag = None
        if ev.inaxes is not self.ax_img or ev.xdata is None:
            return
        p1 = (ev.xdata, ev.ydata)
        if np.hypot(p1[0] - p0[0], p1[1] - p0[1]) < 1e-3 * self.channel.width_nm / 1e3:
            return
        # axes are in µm
        if self._region_mode():
            self.set_roi((p0[0] * 1e3, p1[0] * 1e3, p0[1] * 1e3, p1[1] * 1e3))
        else:
            self.set_profile((p0[0] * 1e3, p0[1] * 1e3), (p1[0] * 1e3, p1[1] * 1e3))

    def set_roi(self, roi_nm):
        x0, x1 = sorted(roi_nm[:2])
        y0, y1 = sorted(roi_nm[2:])
        self.roi = (x0, x1, y0, y1)
        self.analyse()

    def clear_roi(self):
        self.roi = None
        self.analyse()

    def on_auto_profile(self):
        if self.result is None:
            return
        try:
            ap = afm.auto_profile(self.result.z, self.channel.dx, self.channel.dy)
        except Exception as exc:
            show_error(self, "Automatic profile", exc)
            return
        self.auto = ap
        self.sp_width.blockSignals(True)
        self.sp_width.setValue(ap.width_px)
        self.sp_width.blockSignals(False)
        self.set_profile(ap.p0, ap.p1)

    def set_profile(self, p0_nm, p1_nm):
        self._compute_profile(p0_nm, p1_nm)
        self.analyse()

    def _width_changed(self, *_):
        if self.profile is not None and self.result is not None:
            self._compute_profile(self.profile[2], self.profile[3])
            self.analyse()

    def _compute_profile(self, p0, p1):
        s, h = afm.line_profile(self.result.z, self.channel.dx, self.channel.dy, p0, p1,
                                width=self.sp_width.value())
        self.profile = (s, h, p0, p1)
        try:
            self.prof_steps = afm.profile_steps(s, h, self.sp_minstep.value(), self.sp_edge.value(),
                                                reference_step=self.sp_ref.value())
        except Exception:
            self.prof_steps = None

    # --------------------------------------------------------------- drawing
    def redraw(self, *_):
        if self.result is None:
            return
        ch, z = self.channel, self.result.z
        fig = self.plot.figure
        self.ax_img.clear()
        self._line = None
        lo, hi = np.nanpercentile(z, [self.sp_lo.value(), self.sp_hi.value()])
        ext = (0, ch.width_nm / 1e3, ch.height_nm / 1e3, 0)
        im = self.ax_img.imshow(z, cmap=self.cb_cmap.currentText(), vmin=lo, vmax=hi, extent=ext,
                                interpolation="nearest")
        if self.cbar is None:
            self.cbar = fig.colorbar(im, ax=self.ax_img, shrink=0.9)
        else:
            self.cbar.update_normal(im)
        self.cbar.set_label(f"{ch.name} ({ch.unit})")
        if self.chk_showmask.isChecked() and self.result.mask.any():
            ov = np.ma.masked_where(~self.result.mask, np.ones_like(z))
            self.ax_img.imshow(ov, cmap="cool", alpha=0.35, extent=ext, interpolation="nearest")
        if self.features is not None and self.features.table:
            self.ax_img.contour(self.features.labels > 0, levels=[0.5], colors="c", linewidths=0.6,
                                extent=ext, origin="upper")
        self.ax_img.set_xlabel("x (µm)")
        self.ax_img.set_ylabel("y (µm)")
        self.ax_img.set_title(f"{self.file.path.stem}\n{ch.label}", fontsize=8)
        self.ax_prof.clear()
        if self.roi is not None:
            x0, x1, y0, y1 = (v / 1e3 for v in self.roi)
            self.ax_img.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "-", color="lime", lw=1.2)
        if self.profile is not None:
            s, h, p0, p1 = self.profile
            self.ax_img.plot([p0[0] / 1e3, p1[0] / 1e3], [p0[1] / 1e3, p1[1] / 1e3], "c-", lw=1.2)
            self.ax_img.plot([p0[0] / 1e3], [p0[1] / 1e3], "co", ms=4)
            wpx = self.sp_width.value()
            L = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
            if wpx > 1 and L > 0:
                half = 0.5 * wpx * min(ch.dx, ch.dy)
                nx_, ny_ = -(p1[1] - p0[1]) / L * half, (p1[0] - p0[0]) / L * half
                bx = [p0[0] + nx_, p1[0] + nx_, p1[0] - nx_, p0[0] - nx_, p0[0] + nx_]
                by = [p0[1] + ny_, p1[1] + ny_, p1[1] - ny_, p0[1] - ny_, p0[1] + ny_]
                self.ax_img.plot(np.array(bx) / 1e3, np.array(by) / 1e3, "c--", lw=0.8)
            self.ax_prof.plot(s, h, "k-", lw=1, label="profile")
            ps = self.prof_steps
            if ps is not None and len(ps.heights):
                self.ax_prof.plot(s, ps.model, "-", color="tab:red", lw=1, label="staircase fit")
                for e_, hgt in zip(ps.edges, ps.heights):
                    self.ax_prof.axvline(e_, color="tab:red", lw=0.5, ls=":")
                    self.ax_prof.annotate(f"{hgt:+.3f}", (e_, np.interp(e_, s, ps.model)), fontsize=7,
                                          color="tab:red", xytext=(3, 3), textcoords="offset points")
                self.ax_prof.legend(loc="best", fontsize=7)
            self.ax_prof.set_xlabel("distance (nm)")
            self.ax_prof.set_ylabel(f"height ({ch.unit})")
        else:
            self.ax_prof.text(0.5, 0.5, "drag a line in the image", ha="center", va="center",
                              transform=self.ax_prof.transAxes, color="0.5")
            self.ax_prof.set_axis_off()
        self.plot.draw()

        a = self.ax_hist
        a.clear()
        if self.hist is not None:
            hst = self.hist
            w = hst.centers[1] - hst.centers[0]
            a.bar(hst.centers, hst.counts, width=w, color="0.7", label="histogram")
            lf = self.levelfit
            levels = lf.centers if lf is not None else hst.levels
            if lf is not None:
                a.plot(hst.centers, lf.model, "k-", lw=1, label="Gaussian fit")
            else:
                a.plot(hst.centers, hst.smooth, "k-", lw=1, label="smoothed")
            for lv in levels:
                a.axvline(lv, color="tab:red", lw=0.8)
                a.text(lv, a.get_ylim()[1] * 0.95, f" {lv:.3f}", color="tab:red", va="top", fontsize=8)
            a.set_xlabel(f"height ({ch.unit})")
            a.set_ylabel("counts")
            ref = self.sp_ref.value()
            where = "region" if self.roi is not None else "whole image"
            if lf is not None and lf.ladder:
                a.set_title(f"{where}: step = {lf.ladder['step']:.4f} ± {lf.ladder['error']:.4f} {ch.unit} "
                            f"({lf.ladder['step'] / ref:.3f} u.c.)")
            elif len(hst.steps):
                a.set_title(f"{where}: steps " + ", ".join(f"{s:.3f}" for s in hst.steps) + f" {ch.unit}")
            a.legend(loc="best")
        self.plot_hist.draw()

        a = self.ax_psd
        a.clear()
        f, p = afm.psd_1d(z, ch.dx, axis=1)
        a.loglog(f * 1e3, p, "k-", lw=1, label="fast axis")
        f2, p2 = afm.psd_1d(z, ch.dy, axis=0)
        a.loglog(f2 * 1e3, p2, "-", color="tab:blue", lw=1, alpha=0.7, label="slow axis")
        a.set_xlabel("spatial frequency (µm⁻¹)")
        a.set_ylabel(f"PSD ({ch.unit}² nm)")
        a.legend()
        self.plot_psd.draw()

    # --------------------------------------------------------------- export
    def _out_dir(self):
        d = self.file.path.parent / "output"
        d.mkdir(exist_ok=True)
        return d

    def _stem(self):
        return f"{self.file.path.stem}_{self.channel.name.replace(' ', '')}"

    def on_save_image(self):
        if self.result is None:
            return
        p = save_file(self, "Save corrected image", self._out_dir() / f"{self._stem()}_corrected.txt",
                      "Text (*.txt)")
        if p:
            afm.save_ascii_matrix(p, self.result.z, self.channel.width_nm, self.channel.height_nm,
                                  self.channel.name, self.channel.unit)
            self.db.attach(p, "corrected image")

    def on_save_profile(self):
        if self.profile is None:
            return
        p = save_file(self, "Save profile", self._out_dir() / f"{self._stem()}_profile.csv", "CSV (*.csv)")
        if p:
            s, h, p0, p1 = self.profile
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(f"# profile from ({p0[0]:.1f}, {p0[1]:.1f}) nm to ({p1[0]:.1f}, {p1[1]:.1f}) nm, "
                         f"width {self.sp_width.value()} px\n")
                if self.prof_steps is not None and len(self.prof_steps.heights):
                    fh.write(f"Distance,Height,Staircase fit\nnm,{self.channel.unit},{self.channel.unit}\n")
                    np.savetxt(fh, np.c_[s, h, self.prof_steps.model], delimiter=",", fmt="%.6g")
                else:
                    fh.write(f"Distance,Height\nnm,{self.channel.unit}\n")
                    np.savetxt(fh, np.c_[s, h], delimiter=",", fmt="%.6g")
            self.db.attach(p, "profile csv")

    def db_record(self) -> Measurement:
        if self.result is None:
            raise ValueError("Load an image first.")
        res = []
        for q, v, e, u, note in self.results:
            if q.startswith("Profile step ") or q.startswith("Level "):
                continue      # position-dependent rows: kept in the CSV export only
            res.append(Result(q, v, e if e != "" else None, u))
        curves = []
        if self.profile is not None:
            curves.append(Curve("profile", self.profile[0], self.profile[1], "distance (nm)",
                                f"height ({self.channel.unit})"))
        settings = {"channel": self.channel.label, "corrections": str(self.corrections()),
                    "width_nm": self.channel.width_nm, "height_nm": self.channel.height_nm}
        return Measurement(sample="", technique="AFM", raw_path=str(self.file.path), label=self.channel.label,
                           measured_at=file_date(self.file.path), settings=settings, results=res, curves=curves)

    def on_save_results(self):
        if self.result is None:
            return
        p = save_file(self, "Save results", self._out_dir() / f"{self._stem()}_results.csv", "CSV (*.csv)")
        if p:
            df = pd.DataFrame(self.results, columns=["Quantity", "Value", "Error", "Unit", "Note"])
            c = self.corrections()
            with open(p, "w", encoding="utf-8", newline="") as fh:
                fh.write(f"# file: {self.file.path.name}, channel: {self.channel.label}\n")
                fh.write(f"# corrections: {c}\n")
                if self.roi is not None:
                    fh.write("# histogram region (nm): x %.0f-%.0f, y %.0f-%.0f\n" % self.roi)
                if self.profile is not None:
                    p0, p1 = self.profile[2], self.profile[3]
                    fh.write(f"# profile ({p0[0]:.0f}, {p0[1]:.0f}) -> ({p1[0]:.0f}, {p1[1]:.0f}) nm, "
                             f"width {self.sp_width.value()} px\n")
                df.to_csv(fh, index=False)
            self.db.attach(p, "results csv")
            if self.features is not None and self.features.table:
                fp = Path(p).with_name(Path(p).stem + f"_{self.features.kind}.csv")
                pd.DataFrame(self.features.table).to_csv(fp, index=False)
                self.db.attach(fp, f"{self.features.kind} csv")

    def on_save_figs(self):
        if self.result is None:
            return
        p = save_file(self, "Save figures", self._out_dir() / f"{self._stem()}.png", "PNG (*.png)")
        if p:
            p = Path(p)
            self.plot.figure.savefig(p, dpi=200)
            self.plot_hist.figure.savefig(p.with_name(p.stem + "_histogram.png"), dpi=200)
            self.plot_psd.figure.savefig(p.with_name(p.stem + "_psd.png"), dpi=200)
            self.db.attach(p, "figure")
