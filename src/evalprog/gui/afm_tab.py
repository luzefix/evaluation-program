"""AFM tab: topography images (Bruker NanoScope .spm, ASCII matrices)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QHBoxLayout, QPushButton, QSplitter, QTabWidget,
                               QWidget)

from evalprog.core import afm

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
        self.cb_rows = combo(ROWS, "median")
        self.chk_mask = QCheckBox("Exclude pits/particles from fits")
        self.chk_mask.setChecked(True)
        self.sp_mask = spin(2.5, 0.5, 20, 1, 0.5, " σ")
        self.chk_outl = QCheckBox("Remove spikes")
        self.sp_outl = spin(6.0, 1, 50, 1, 0.5, " σ")
        self.cb_zero = combo(ZERO, "median")
        for w in (self.chk_scars, self.chk_mask, self.chk_outl):
            w.toggled.connect(self.apply)
        for w in (self.sp_scar, self.sp_scar_len, self.sp_mask, self.sp_outl):
            w.valueChanged.connect(self.apply)
        for w in (self.cb_bg, self.cb_rows, self.cb_zero):
            w.currentIndexChanged.connect(self.apply)
        lay.addWidget(group("Corrections (applied in this order)", [
            self.chk_scars, ("Scar threshold / min. length", hbox(self.sp_scar, self.sp_scar_len)),
            ("Background", self.cb_bg), ("Line alignment", self.cb_rows),
            self.chk_mask, ("Mask threshold", self.sp_mask),
            self.chk_outl, ("Spike threshold", self.sp_outl), ("Zero level", self.cb_zero)]))
        lay.addWidget(info_label(
            "Terraces / step edges: use 'terraces (facet level)' and 'median of differences'. "
            "Rough films and particles: 'plane' or 'polynomial 2' and 'median'."))

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

        self.sp_width = ispin(3, 1, 200)
        self.sp_ref = spin(0.3905, 0.01, 100, 4, 0.01, " nm")
        self.sp_bins = ispin(300, 20, 5000)
        self.sp_prom = spin(5.0, 0.5, 90, 1, 1, " %")
        for w in (self.sp_ref, self.sp_bins, self.sp_prom):
            w.valueChanged.connect(self.analyse)
        lay.addWidget(group("Profiles and steps", [
            info_label("Drag a line in the image for a profile (toolbar zoom/pan off)."),
            ("Profile width (px)", self.sp_width),
            ("Reference step (unit cell)", self.sp_ref),
            ("Histogram bins", self.sp_bins), ("Level detection: prominence", self.sp_prom)]))

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
        self.table = ResultsTable(["Quantity", "Value", "Unit", "Note"])
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
            outlier_sigma=self.sp_outl.value(), zero=self.cb_zero.currentText())

    def apply(self, *_):
        if self.channel is None:
            return
        try:
            self.result = afm.correct(self.channel.data, self.corrections())
        except Exception as exc:
            show_error(self, "AFM corrections", exc)
            return
        self.lbl_log.setText("\n".join(self.result.log))
        self.features = None
        if self.profile is not None:
            self._compute_profile(self.profile[2], self.profile[3])
        self.analyse()

    # --------------------------------------------------------------- analysis
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
            rows.append((k, r[k], "" if k in ("Ssk", "Sku") else u, note))
        if rm is not None:
            rows.append(("Sq (without excluded features)", rm["Sq"], u, "terrace / matrix roughness"))
        try:
            rows.append(("Correlation length (fast axis)", afm.correlation_length(z, ch.dx), "nm", "ACF = 1/e"))
        except Exception:
            pass
        self.hist = afm.height_histogram(z, bins=self.sp_bins.value(), prominence=self.sp_prom.value() / 100)
        ref = self.sp_ref.value()
        for i, lv in enumerate(self.hist.levels):
            rows.append((f"Level {i + 1}", lv, u, f"rel. weight {self.hist.level_weights[i]:.2f}"))
        for i, st in enumerate(self.hist.steps):
            rows.append((f"Step {i + 1}→{i + 2}", st, u, f"= {st / ref:.2f} × reference"))
        if self.features is not None:
            s = self.features.summary
            rows.append((f"{self.features.kind}: count", s["count"], "", f"threshold {self.features.threshold:g} {u}"))
            rows.append((f"{self.features.kind}: density", s["density_um2"], "µm⁻²", ""))
            rows.append((f"{self.features.kind}: area fraction", 100 * s["area_fraction"], "%", ""))
            if "mean_d_eq_nm" in s:
                rows.append((f"{self.features.kind}: mean diameter", s["mean_d_eq_nm"], "nm", "equivalent disc"))
                rows.append((f"{self.features.kind}: mean depth/height", s["mean_extreme_nm"], u, "vs. reference level"))
                rows.append((f"{self.features.kind}: max depth/height", s["max_extreme_nm"], u, ""))
        if self.profile is not None:
            s_, h_ = self.profile[0], self.profile[1]
            rows.append(("Profile length", s_[-1], "nm", ""))
            rows.append(("Profile peak-to-valley", float(np.ptp(h_)), u, ""))
        self.results = rows
        self.table.set_rows(rows)
        self.redraw()

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

    # --------------------------------------------------------------- profile
    def _toolbar_idle(self):
        return not str(self.plot.toolbar.mode)

    def _press(self, ev):
        if ev.inaxes is self.ax_img and ev.button == 1 and self._toolbar_idle() and self.result is not None:
            self._drag = (ev.xdata, ev.ydata)

    def _motion(self, ev):
        if self._drag is None or ev.inaxes is not self.ax_img:
            return
        x0, y0 = self._drag
        if self._line is None:
            (self._line,) = self.ax_img.plot([x0, ev.xdata], [y0, ev.ydata], "c-", lw=1.5)
        else:
            self._line.set_data([x0, ev.xdata], [y0, ev.ydata])
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
        self.set_profile((p0[0] * 1e3, p0[1] * 1e3), (p1[0] * 1e3, p1[1] * 1e3))

    def set_profile(self, p0_nm, p1_nm):
        self._compute_profile(p0_nm, p1_nm)
        self.analyse()

    def _compute_profile(self, p0, p1):
        s, h = afm.line_profile(self.result.z, self.channel.dx, self.channel.dy, p0, p1,
                                width=self.sp_width.value())
        self.profile = (s, h, p0, p1)

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
        if self.profile is not None:
            s, h, p0, p1 = self.profile
            self.ax_img.plot([p0[0] / 1e3, p1[0] / 1e3], [p0[1] / 1e3, p1[1] / 1e3], "c-", lw=1.2)
            self.ax_img.plot([p0[0] / 1e3], [p0[1] / 1e3], "co", ms=4)
            self.ax_prof.plot(s, h, "k-", lw=1)
            for lv in (self.hist.levels if self.hist is not None else []):
                self.ax_prof.axhline(lv, color="tab:red", lw=0.5, ls=":")
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
            a.plot(hst.centers, hst.smooth, "k-", lw=1, label="smoothed")
            for i, lv in enumerate(hst.levels):
                a.axvline(lv, color="tab:red", lw=0.8)
                a.text(lv, a.get_ylim()[1] * 0.95, f" {lv:.3f}", color="tab:red", va="top", fontsize=8)
            a.set_xlabel(f"height ({ch.unit})")
            a.set_ylabel("counts")
            ref = self.sp_ref.value()
            if len(hst.steps):
                a.set_title("steps: " + ", ".join(f"{s:.3f} {ch.unit} ({s / ref:.2f} u.c.)" for s in hst.steps))
            a.legend(loc="upper left")
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

    def on_save_profile(self):
        if self.profile is None:
            return
        p = save_file(self, "Save profile", self._out_dir() / f"{self._stem()}_profile.csv", "CSV (*.csv)")
        if p:
            s, h, p0, p1 = self.profile
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(f"# profile from ({p0[0]:.1f}, {p0[1]:.1f}) nm to ({p1[0]:.1f}, {p1[1]:.1f}) nm, "
                         f"width {self.sp_width.value()} px\n")
                fh.write(f"Distance,Height\nnm,{self.channel.unit}\n")
                np.savetxt(fh, np.c_[s, h], delimiter=",", fmt="%.6g")

    def on_save_results(self):
        if self.result is None:
            return
        p = save_file(self, "Save results", self._out_dir() / f"{self._stem()}_results.csv", "CSV (*.csv)")
        if p:
            df = pd.DataFrame(self.results, columns=["Quantity", "Value", "Unit", "Note"])
            c = self.corrections()
            with open(p, "w", encoding="utf-8", newline="") as fh:
                fh.write(f"# file: {self.file.path.name}, channel: {self.channel.label}\n")
                fh.write(f"# corrections: {c}\n")
                df.to_csv(fh, index=False)
            if self.features is not None and self.features.table:
                pd.DataFrame(self.features.table).to_csv(
                    Path(p).with_name(Path(p).stem + f"_{self.features.kind}.csv"), index=False)

    def on_save_figs(self):
        if self.result is None:
            return
        p = save_file(self, "Save figures", self._out_dir() / f"{self._stem()}.png", "PNG (*.png)")
        if p:
            p = Path(p)
            self.plot.figure.savefig(p, dpi=200)
            self.plot_hist.figure.savefig(p.with_name(p.stem + "_histogram.png"), dpi=200)
            self.plot_psd.figure.savefig(p.with_name(p.stem + "_psd.png"), dpi=200)
