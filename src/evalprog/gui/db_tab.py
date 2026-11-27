"""Database tab: browse, filter and sort all saved measurements, compare them."""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QDialog, QDialogButtonBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton,
                               QSplitter, QTableView, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from evalprog.db import elab
from evalprog.db.store import BASE_COLUMNS, SAMPLE_TYPES, TECHNIQUES, Database

from .db_widgets import BUS, current_db, db_path
from .widgets import PlotWidget, combo, fmt, group, hbox, info_label, scroll_panel, settings, show_error, spin

OPS = ["=", "≠", "<", "≤", ">", "≥", "contains", "is set", "is empty"]
NORMALISE = ["none", "max = 1", "min-max", "at x (value below)"]
COLORS = ["tab:blue", "tab:red", "tab:green", "tab:purple", "tab:orange", "tab:brown", "tab:pink",
          "tab:gray", "tab:olive", "tab:cyan"]


# ------------------------------------------------------------------ model
class FrameModel(QAbstractTableModel):
    """Read-only pandas DataFrame model; sorting is numeric where possible."""

    def __init__(self, df=None):
        super().__init__()
        self.df = df if df is not None else pd.DataFrame()

    def set_frame(self, df):
        self.beginResetModel()
        self.df = df.reset_index(drop=True)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.df)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self.df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        v = self.df.iat[index.row(), index.column()]
        if role == Qt.DisplayRole:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return ""
            if isinstance(v, (float, np.floating)):
                return fmt(float(v))
            return str(v)
        if role == Qt.ToolTipRole and isinstance(v, str) and len(v) > 30:
            return v
        if role == Qt.TextAlignmentRole and isinstance(v, (int, float, np.number)):
            return int(Qt.AlignRight | Qt.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self.df.columns[section])
            return str(section + 1)
        return None

    def sort(self, column, order=Qt.AscendingOrder):
        if self.df.empty:
            return
        col = self.df.columns[column]
        self.layoutAboutToBeChanged.emit()
        key = pd.to_numeric(self.df[col], errors="coerce")
        if key.notna().sum() < self.df[col].notna().sum():
            key = self.df[col].astype(str).str.lower()
        idx = key.sort_values(ascending=order == Qt.AscendingOrder, na_position="last", kind="stable").index
        self.df = self.df.loc[idx].reset_index(drop=True)
        self.layoutChanged.emit()


# ----------------------------------------------------------------- filters
def apply_filter(df: pd.DataFrame, col: str, op: str, value: str) -> pd.DataFrame:
    if col not in df.columns:
        return df
    s = df[col]
    if op == "is set":
        return df[s.notna() & (s.astype(str) != "")]
    if op == "is empty":
        return df[s.isna() | (s.astype(str) == "")]
    if op == "contains":
        return df[s.astype(str).str.contains(value, case=False, regex=False, na=False)]
    num = pd.to_numeric(s, errors="coerce")
    try:
        v = float(value.replace(",", "."))
    except ValueError:
        v = None
    if v is not None and num.notna().any():
        m = {"=": num == v, "≠": num != v, "<": num < v, "≤": num <= v, ">": num > v, "≥": num >= v}[op]
        return df[m.fillna(False)]
    st = s.astype(str).str.lower()
    val = value.lower()
    m = {"=": st == val, "≠": st != val, "<": st < val, "≤": st <= val, ">": st > val, "≥": st >= val}[op]
    return df[m]


SAMPLE_BASE = ["sample", "sample_type", "material", "substrate", "orientation", "thickness_nm",
               "measurements", "techniques"]


def per_sample(df: pd.DataFrame) -> pd.DataFrame:
    """One row per sample. Result columns (already '<technique> <key> (unit)') take the
    latest saved value, synthesis parameters are taken once. Column 'ids' lists the
    measurements that went into the row."""
    if df.empty:
        return pd.DataFrame(columns=SAMPLE_BASE + ["ids"])
    res_cols = [c for c in df.columns[len(BASE_COLUMNS):] if not str(c).startswith("syn: ")]
    syn_cols = [c for c in df.columns if str(c).startswith("syn: ")]
    df = df.sort_values("saved_at", kind="stable")
    rows = []
    for sample, g in df.groupby("sample", sort=False):
        first = g.iloc[0]
        row = {"sample": sample, "sample_type": first["sample_type"], "material": first["material"],
               "substrate": first["substrate"], "orientation": first["orientation"],
               "thickness_nm": first["thickness_nm"], "measurements": len(g),
               "techniques": ", ".join(sorted(g["technique"].unique())), "ids": list(g["id"])}
        for c in res_cols + syn_cols:
            s = g[c].dropna()
            if len(s):
                row[c] = s.iloc[-1]
        rows.append(row)
    out = pd.DataFrame(rows)
    out = out[SAMPLE_BASE + ["ids"] + [c for c in res_cols + syn_cols if c in out.columns]]
    for c in res_cols:
        if c in out.columns:
            conv = pd.to_numeric(out[c], errors="coerce")
            if conv.notna().sum() == out[c].notna().sum():
                out[c] = conv
    return out


def default_columns(df: pd.DataFrame, attrs: dict, sample_view: bool, details: bool, all_syn: bool) -> list:
    """Visible columns in display order: sample info, key synthesis parameters, main
    results, then (on request) details, remaining synthesis parameters and file info."""
    head = (["sample", "material", "substrate", "orientation", "thickness_nm", "techniques", "measurements"]
            if sample_view else ["sample", "technique", "material", "substrate", "orientation", "thickness_nm"])
    cols_ = head + attrs.get("syn_key", []) + attrs.get("main", [])
    if details:
        cols_ += attrs.get("detail", [])
    if all_syn or details:
        cols_ += attrs.get("syn", [])
    if details:
        cols_ += ["sample_type"] if sample_view else ["sample_type", "label", "measured_at", "saved_at",
                                                      "raw_path", "files"]
    return [c for c in dict.fromkeys(cols_) if c in df.columns]


class ColumnsDialog(QDialog):
    """Choose and order the visible columns (drag to reorder)."""

    def __init__(self, all_cols, visible, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Columns")
        self.resize(420, 620)
        self.lw = QListWidget()
        self.lw.setDragDropMode(QAbstractItemView.InternalMove)
        for c in list(visible) + [c for c in all_cols if c not in visible]:
            it = QListWidgetItem(c)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if c in visible else Qt.Unchecked)
            self.lw.addItem(it)
        self.reset = False
        b_reset = QPushButton("Reset to default")
        b_reset.clicked.connect(self._reset)
        b_none = QPushButton("Untick all")
        b_none.clicked.connect(lambda: [self.lw.item(i).setCheckState(Qt.Unchecked) for i in range(self.lw.count())])
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v = QVBoxLayout(self)
        v.addWidget(QLabel("Tick the columns to show; drag to change the order."))
        v.addWidget(self.lw)
        v.addWidget(hbox(b_none, b_reset))
        v.addWidget(bb)

    def _reset(self):
        self.reset = True
        self.accept()

    def columns(self):
        return [self.lw.item(i).text() for i in range(self.lw.count())
                if self.lw.item(i).checkState() == Qt.Checked]


class FilterRow(QWidget):
    changed = Signal()
    removed = Signal(object)

    def __init__(self, columns):
        super().__init__()
        self.cb_col = combo(columns, editable=True)
        self.cb_col.setMinimumWidth(130)
        self.cb_op = combo(OPS)
        self.ed_val = QLineEdit()
        self.ed_val.setPlaceholderText("value")
        b = QPushButton("✕")
        b.setFixedWidth(26)
        b.clicked.connect(lambda: self.removed.emit(self))
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        for w in (self.cb_col, self.cb_op, self.ed_val, b):
            lay.addWidget(w)
        self.cb_col.currentTextChanged.connect(self.changed)
        self.cb_op.currentIndexChanged.connect(self.changed)
        self.ed_val.editingFinished.connect(self.changed)

    def set_columns(self, columns):
        cur = self.cb_col.currentText()
        self.cb_col.blockSignals(True)
        self.cb_col.clear()
        self.cb_col.addItems(columns)
        self.cb_col.setCurrentText(cur)
        self.cb_col.blockSignals(False)

    def apply(self, df):
        return apply_filter(df, self.cb_col.currentText(), self.cb_op.currentText(), self.ed_val.text().strip())


# ------------------------------------------------------------ sample dialog
class SampleDialog(QDialog):
    """Edit a sample: type, material, film fields, parameters."""

    def __init__(self, db: Database, name: str, parent=None):
        super().__init__(parent)
        self.db = db
        self.name = name
        s = db.get_sample(name) or {}
        self.setWindowTitle(f"Sample {name}")
        self.resize(560, 640)
        form = QFormLayout()
        self.ed_name = QLineEdit(name)
        self.cb_type = combo(SAMPLE_TYPES, s.get("sample_type") or "thin film", editable=True)
        self.ed_mat = QLineEdit(s.get("material") or "")
        self.ed_sub = QLineEdit(s.get("substrate") or "")
        self.ed_or = QLineEdit(s.get("orientation") or "")
        self.ed_t = QLineEdit("" if s.get("thickness_nm") is None else fmt(s["thickness_nm"]))
        self.ed_parent = QLineEdit(s.get("parent") or "")
        self.ed_parent.setPlaceholderText("e.g. parent crystal of a lamella")
        self.ed_notes = QPlainTextEdit(s.get("notes") or "")
        self.ed_notes.setMaximumHeight(70)
        for lab, w in (("Name", self.ed_name), ("Type", self.cb_type), ("Material", self.ed_mat),
                       ("Substrate (films)", self.ed_sub), ("Orientation", self.ed_or),
                       ("Thickness (nm)", self.ed_t), ("Parent sample", self.ed_parent), ("Notes", self.ed_notes)):
            form.addRow(lab, w)
        if s.get("elab_id"):
            form.addRow("elabFTW", QLabel(f"{s.get('elab_kind')} #{s['elab_id']}, modified {s.get('elab_modified')}"))
        self.params = QTableWidget(0, 4)
        self.params.setHorizontalHeaderLabels(["Parameter", "Value", "Unit", "Source"])
        self.params.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for _, r in db.get_params(name).iterrows():
            self._add_param(r["key"], r["value"], r["unit"], r["source"])
        b_add = QPushButton("Add parameter")
        b_add.clicked.connect(lambda: self._add_param("", "", "", "manual"))
        b_del = QPushButton("Remove selected")
        b_del.clicked.connect(lambda: [self.params.removeRow(i.row()) for i in
                                       sorted(self.params.selectionModel().selectedRows(), key=lambda i: -i.row())])
        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(QLabel("Parameters (synthesis etc.; elabFTW values are replaced by the next sync)"))
        lay.addWidget(self.params)
        lay.addWidget(hbox(b_add, b_del))
        lay.addWidget(bb)

    def _add_param(self, k, v, u, src):
        i = self.params.rowCount()
        self.params.insertRow(i)
        for j, x in enumerate((k, v, u, src)):
            it = QTableWidgetItem("" if x is None else str(x))
            if j == 3:
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            self.params.setItem(i, j, it)

    def accept(self):
        try:
            t = self.ed_t.text().strip().replace(",", ".")
            self.db.update_sample(self.name, sample_type=self.cb_type.currentText(),
                                  material=self.ed_mat.text().strip() or None,
                                  substrate=self.ed_sub.text().strip() or None,
                                  orientation=self.ed_or.text().strip() or None,
                                  thickness_nm=float(t) if t else None,
                                  parent=self.ed_parent.text().strip() or None,
                                  notes=self.ed_notes.toPlainText().strip() or None)
            old = self.db.get_params(self.name)
            keep = set()
            by_src: dict[str, dict] = {}
            units = {}
            for i in range(self.params.rowCount()):
                k = (self.params.item(i, 0).text() if self.params.item(i, 0) else "").strip()
                if not k:
                    continue
                v = self.params.item(i, 1).text() if self.params.item(i, 1) else ""
                u = self.params.item(i, 2).text() if self.params.item(i, 2) else ""
                src = (self.params.item(i, 3).text() if self.params.item(i, 3) else "") or "manual"
                by_src.setdefault(src, {})[k] = v
                units[k] = u
                keep.add(k)
            removed = {k: None for k in old["key"] if k not in keep}
            if removed:
                self.db.set_params(self.name, removed)
            for src, p in by_src.items():
                self.db.set_params(self.name, p, source=src, units=units)
            new = self.ed_name.text().strip()
            if new and new != self.name:
                if self.db.get_sample(new) and QMessageBox.question(
                        self, "Merge samples", f"{new} exists already. Merge {self.name} into {new}?") \
                        != QMessageBox.Yes:
                    return
                self.db.rename_sample(self.name, new)
        except Exception as exc:
            show_error(self, "Save sample", exc)
            return
        super().accept()


class ElabDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("elabFTW connection")
        cfg = elab.load_config()
        self.ed_url = QLineEdit(cfg["url"])
        self.ed_url.setPlaceholderText("https://elab.example.org")
        self.ed_key = QLineEdit(cfg["api_key"])
        self.ed_key.setEchoMode(QLineEdit.Password)
        self.ed_cats = QLineEdit(", ".join(cfg.get("categories") or []))
        self.ed_cats.setPlaceholderText("empty = all categories")
        self.chk_items = QCheckBox("also read items (resources)")
        self.chk_items.setChecked("items" in (cfg.get("entities") or []))
        self.chk_ssl = QCheckBox("verify SSL certificate")
        self.chk_ssl.setChecked(bool(cfg.get("verify_ssl", True)))
        self.chk_auto = QCheckBox("check automatically once a day")
        self.chk_auto.setChecked(bool(cfg.get("auto_sync", True)))
        form = QFormLayout(self)
        form.addRow("Instance URL", self.ed_url)
        form.addRow("API key", self.ed_key)
        form.addRow("Categories", self.ed_cats)
        form.addRow(self.chk_items)
        form.addRow(self.chk_ssl)
        form.addRow(self.chk_auto)
        form.addRow(info_label(f"Stored in {elab.CONFIG_PATH} (outside the repository)."))
        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)
        self.cfg = cfg

    def accept(self):
        self.cfg.update(url=self.ed_url.text().strip(), api_key=self.ed_key.text().strip(),
                        categories=[c.strip() for c in self.ed_cats.text().split(",") if c.strip()],
                        entities=["experiments"] + (["items"] if self.chk_items.isChecked() else []),
                        verify_ssl=self.chk_ssl.isChecked(), auto_sync=self.chk_auto.isChecked())
        elab.save_config(self.cfg)
        super().accept()


# -------------------------------------------------------------- sync thread
class SyncWorker(QObject):
    finished = Signal(str, bool)     # message, ok

    def __init__(self, path, force=False):
        super().__init__()
        self.path, self.force = path, force

    def run(self):
        db = None
        try:
            db = Database(self.path)      # own connection: sqlite objects are per thread
            rep = elab.sync(db, force=self.force, log=None)
            self.finished.emit(f"elabFTW sync: {rep}", True)
        except Exception as exc:
            self.finished.emit(f"elabFTW sync failed: {exc}", False)
        finally:
            if db is not None:
                db.close()


def start_sync(owner, force=False, on_done=None):
    """Run a sync in a background thread; ``on_done(message, ok)``."""
    if getattr(owner, "_sync_thread", None) is not None:
        return False
    th = QThread(owner)
    w = SyncWorker(current_db().path, force)
    w.moveToThread(th)
    th.started.connect(w.run)

    def done(msg, ok):
        th.quit()
        th.wait()
        owner._sync_thread = None
        owner._sync_worker = None
        BUS.changed.emit()
        if on_done:
            on_done(msg, ok)

    w.finished.connect(done)
    owner._sync_thread, owner._sync_worker = th, w
    th.start()
    return True


# --------------------------------------------------------------------- tab
class DatabaseTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.full = pd.DataFrame()
        self.filters: list[FilterRow] = []
        self._sync_thread = None
        self._build()
        BUS.changed.connect(self.reload)
        self.reload()

    # ------------------------------------------------------------------ UI
    def _build(self):
        panel, lay = scroll_panel(400)
        self.lbl_db = info_label("")
        b_db = QPushButton("Change database file …")
        b_db.clicked.connect(self.on_change_db)
        b_reload = QPushButton("Reload")
        b_reload.clicked.connect(self.reload)
        lay.addWidget(group("Database", [self.lbl_db, hbox(b_db, b_reload)]))

        self.cb_view = combo(["per measurement", "per sample"])
        self.cb_view.setToolTip("per sample: one row per sample, results as '<technique>: <quantity>' "
                                "(latest saved value), synthesis parameters once")
        self.cb_view.currentIndexChanged.connect(self._on_view)
        self.cb_tech = combo(["all"] + TECHNIQUES)
        self.cb_stype = combo(["all"] + SAMPLE_TYPES)
        self.ed_search = QLineEdit()
        self.ed_search.setPlaceholderText("sample / material / file, e.g. CIO9, CIO12")
        self.chk_empty = QCheckBox("hide empty columns")
        self.chk_empty.setChecked(True)
        self.chk_syn = QCheckBox("all synthesis parameters")
        self.chk_syn.setToolTip("The key ones (T_Dep, T_Shutter, Dep volume, ...) are always shown")
        self.chk_details = QCheckBox("details (single peaks, fit quality, files, dates)")
        self.b_cols = QPushButton("Columns …")
        self.b_cols.setToolTip("Choose and order the visible columns (saved)")
        self.b_cols.clicked.connect(self.on_columns)
        for w in (self.cb_tech, self.cb_stype):
            w.currentIndexChanged.connect(self.refilter)
        self.ed_search.textChanged.connect(self.refilter)
        self.chk_empty.toggled.connect(self.refilter)
        self.chk_syn.toggled.connect(self.refilter)
        self.chk_details.toggled.connect(self.refilter)
        self.filter_box = QVBoxLayout()
        self.filter_box.setSpacing(2)
        fw = QWidget()
        fw.setLayout(self.filter_box)
        b_add = QPushButton("+ parameter filter")
        b_add.clicked.connect(self.add_filter)
        lay.addWidget(group("Filter", [("View", self.cb_view), ("Technique", self.cb_tech), ("Sample type", self.cb_stype),
                                       ("Search", self.ed_search), fw, b_add]))
        lay.addWidget(group("Show", [self.chk_details, self.chk_syn, self.chk_empty, self.b_cols]))

        self.cb_curve = combo([])
        self.cb_norm = combo(NORMALISE)
        self.sp_normx = spin(300.0, -1e9, 1e9, 3, 1)
        self.sp_offset = spin(0.0, -1e12, 1e12, 3, 0.1)
        self.chk_logy = QCheckBox("log y")
        b_ov = QPushButton("Overlay curves of selected rows")
        b_ov.clicked.connect(self.on_overlay)
        lay.addWidget(group("Compare curves", [("Curve", self.cb_curve), ("Normalise", self.cb_norm),
                                               ("x for normalising", self.sp_normx),
                                               ("Waterfall offset", self.sp_offset), self.chk_logy, b_ov]))

        self.cb_x = combo([], editable=True)
        self.cb_y = combo([], editable=True)
        self.cb_group = combo(["sample", "technique", "material", "substrate", "sample_type", "none"])
        self.chk_sel = QCheckBox("only selected rows")
        b_sc = QPushButton("Plot y vs x")
        b_sc.clicked.connect(self.on_scatter)
        lay.addWidget(group("Compare parameters", [("x", self.cb_x), ("y", self.cb_y),
                                                   ("Colour by", self.cb_group), self.chk_sel, b_sc]))

        b_tab = QPushButton("Open selected in analysis tab")
        b_tab.setToolTip("Loads the raw file(s) of the selected rows into XRD / RSM / R(T) / R(H) / AFM")
        b_tab.clicked.connect(self.on_open_in_tab)
        b_files = QPushButton("Evaluated files …")
        b_files.clicked.connect(self.on_files)
        b_edit = QPushButton("Edit sample …")
        b_edit.clicked.connect(self.on_edit_sample)
        b_move = QPushButton("Assign to sample …")
        b_move.clicked.connect(self.on_assign)
        b_open = QPushButton("Raw file folder")
        b_open.clicked.connect(self.on_open_raw)
        b_del = QPushButton("Delete selected")
        b_del.clicked.connect(self.on_delete)
        b_csv = QPushButton("Export table (.csv)")
        b_csv.clicked.connect(self.on_export)
        lay.addWidget(group("Entries", [b_tab, hbox(b_files, b_open), hbox(b_edit, b_move), hbox(b_del, b_csv)]))
        self.open_in_tab = None     # set by the main window: callable(list of row dicts)

        b_sync = QPushButton("Sync now")
        b_sync.clicked.connect(lambda: self.on_sync(False))
        b_cfg = QPushButton("Connection …")
        b_cfg.clicked.connect(self.on_elab_settings)
        b_imp = QPushButton("Import elabFTW JSON export …")
        b_imp.clicked.connect(self.on_import_json)
        self.lbl_sync = info_label("")
        lay.addWidget(group("elabFTW (synthesis parameters)", [hbox(b_sync, b_cfg), b_imp, self.lbl_sync]))

        self.model = FrameModel()
        self.view = QTableView()
        self.view.setModel(self.model)
        self.view.setSortingEnabled(True)
        self.view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.view.setAlternatingRowColors(True)
        self.view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.view.horizontalHeader().setSectionsMovable(True)
        self.view.doubleClicked.connect(lambda *_: self.on_edit_sample())
        self.view.selectionModel().selectionChanged.connect(self._update_curve_names)
        self.lbl_count = QLabel("")
        self.plot = PlotWidget(figsize=(8, 5))
        self.ax = self.plot.figure.add_subplot(111)
        top = QWidget()
        tl = QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(self.lbl_count)
        tl.addWidget(self.view)
        right = QSplitter(Qt.Vertical)
        right.addWidget(top)
        right.addWidget(self.plot)
        right.setSizes([450, 450])
        main = QHBoxLayout(self)
        main.addWidget(panel)
        main.addWidget(right, 1)

    # ---------------------------------------------------------------- data
    def reload(self):
        try:
            db = current_db()
            self.full = db.table()
            self.lbl_db.setText(f"{db.path}<br>{len(db.sample_names())} samples, {len(self.full)} measurements")
            last = db.get_meta("elab_last_sync")
            self.lbl_sync.setText(("not configured" if not elab.is_configured() else
                                   f"last sync: {last or 'never'}"))
        except Exception as exc:
            self.full = pd.DataFrame(columns=BASE_COLUMNS)
            self.lbl_db.setText(f"Cannot open database: {exc}")
        cols = list(self.full.columns)
        for f in self.filters:
            f.set_columns(cols)
        self.refilter()

    def per_sample_mode(self) -> bool:
        return self.cb_view.currentIndex() == 1

    def _on_view(self, *_):
        self.cb_group.setCurrentText("material" if self.per_sample_mode() else "sample")
        self.refilter()

    def _update_axes(self, df):
        num = [c for c in df.columns if c not in ("id", "files", "measurements", "ids")
               and pd.api.types.is_numeric_dtype(df[c])]
        dates = [] if self.per_sample_mode() else ["measured_at", "saved_at"]
        for cb, default in ((self.cb_x, "syn: T_Dep (°C)"), (self.cb_y, "XRD c mean (Å)")):
            cur = cb.currentText() or default
            cb.blockSignals(True)
            cb.clear()
            cb.addItems(dates + num)
            cb.setCurrentText(cur)
            cb.blockSignals(False)

    def filtered(self) -> pd.DataFrame:
        df = self.full
        if df.empty:
            return df
        if self.cb_tech.currentText() != "all":
            df = df[df["technique"] == self.cb_tech.currentText()]
        if self.cb_stype.currentText() != "all":
            df = df[df["sample_type"] == self.cb_stype.currentText()]
        terms = [t.strip().lower() for t in re.split(r"[,;]", self.ed_search.text()) if t.strip()]
        if terms:
            hay = (df["sample"].fillna("") + " " + df["material"].fillna("") + " " + df["substrate"].fillna("")
                   + " " + df["label"].fillna("") + " " + df["raw_path"].fillna("")).str.lower()
            m = pd.Series(False, index=df.index)
            for t in terms:
                m |= hay.str.contains(t, regex=False)
            df = df[m]
        for f in self.filters:
            df = f.apply(df)
        return df

    def _cols_key(self):
        return "db_columns_sample" if self.per_sample_mode() else "db_columns_measurement"

    def visible_columns(self, df) -> list:
        saved = settings().value(self._cols_key(), None)
        if saved:
            saved = [saved] if isinstance(saved, str) else list(saved)
            return [c for c in saved if c in df.columns]
        return default_columns(df, self.full.attrs, self.per_sample_mode(), self.chk_details.isChecked(),
                               self.chk_syn.isChecked())

    def on_columns(self):
        df = self._frame()
        all_cols = [c for c in df.columns if c not in ("id", "ids")]
        dlg = ColumnsDialog(all_cols, self.visible_columns(df), self)
        if dlg.exec():
            if dlg.reset:
                settings().remove(self._cols_key())
            else:
                settings().setValue(self._cols_key(), dlg.columns())
            self.refilter()

    def _frame(self):
        df = self.filtered()
        return per_sample(df) if self.per_sample_mode() else df

    def refilter(self, *_):
        df = self.filtered()
        nmeas = len(df)
        key = "ids" if self.per_sample_mode() else "id"
        if self.per_sample_mode():
            df = per_sample(df)
        show = self.visible_columns(df)
        custom = bool(settings().value(self._cols_key(), None))
        self.b_cols.setText("Columns … (custom)" if custom else "Columns …")
        if self.chk_empty.isChecked() and len(df):
            fixed = {"sample", "technique", "material", "substrate", "orientation"}
            show = [c for c in show if c in fixed or df[c].notna().any()]
        df = df[show + [key]] if key in df.columns else df[show]
        self.model.set_frame(df)
        self.view.resizeColumnsToContents()
        for i in range(self.model.columnCount()):
            if self.view.columnWidth(i) > 220:
                self.view.setColumnWidth(i, 220)
            self.view.setColumnHidden(i, str(df.columns[i]) in ("id", "ids"))
        if self.per_sample_mode():
            self.lbl_count.setText(f"{len(df)} samples ({nmeas} of {len(self.full)} measurements) — "
                                   f"result columns show the latest saved value")
        else:
            self.lbl_count.setText(f"{len(df)} of {len(self.full)} measurements — click a header to sort, "
                                   f"double-click a row to edit the sample")
        self._update_axes(df)
        self._update_curve_names()

    def add_filter(self):
        f = FilterRow(list(self.full.columns))
        f.changed.connect(self.refilter)
        f.removed.connect(self.remove_filter)
        self.filters.append(f)
        self.filter_box.addWidget(f)

    def remove_filter(self, f):
        self.filters.remove(f)
        f.setParent(None)
        f.deleteLater()
        self.refilter()

    def selected_rows(self) -> pd.DataFrame:
        """Selected rows as shown (measurements or samples)."""
        rows = sorted({i.row() for i in self.view.selectionModel().selectedRows()})
        return self.model.df.iloc[rows] if rows else self.model.df.iloc[[]]

    def selected(self) -> pd.DataFrame:
        """Selected measurements (in sample view: all filtered measurements of the selected samples)."""
        sel = self.selected_rows()
        if self.per_sample_mode():
            ids = [i for lst in sel.get("ids", []) for i in lst]
        else:
            ids = list(sel["id"]) if "id" in sel else []
        full = self.full.set_index("id", drop=False)
        return full.loc[[i for i in ids if i in full.index]].reset_index(drop=True)

    def _update_curve_names(self, *_):
        sel = self.selected()
        db = current_db()
        names = []
        for mid in sel["id"] if "id" in sel else []:
            for n in db.curve_names(int(mid)):
                if n not in names:
                    names.append(n)
        cur = self.cb_curve.currentText()
        self.cb_curve.clear()
        self.cb_curve.addItems(names)
        if cur in names:
            self.cb_curve.setCurrentText(cur)

    # -------------------------------------------------------------- plots
    def on_overlay(self):
        sel = self.selected()
        if sel.empty:
            QMessageBox.information(self, "Overlay", "Select rows in the table first (Ctrl/Shift-click).")
            return
        name = self.cb_curve.currentText()
        db = current_db()
        ax = self.ax
        self.plot.figure.clear()
        ax = self.ax = self.plot.figure.add_subplot(111)
        n = 0
        xl = yl = ""
        for _, r in sel.iterrows():
            curves = {c.name: c for c in db.curves(int(r["id"]))}
            c = curves.get(name) or (next(iter(curves.values())) if len(curves) == 1 else None)
            if c is None:
                continue
            x, y = np.asarray(c.x, float), np.asarray(c.y, float)
            norm = self.cb_norm.currentText()
            if norm == "max = 1":
                y = y / np.nanmax(np.abs(y))
            elif norm == "min-max":
                y = (y - np.nanmin(y)) / (np.nanmax(y) - np.nanmin(y))
            elif norm.startswith("at x"):
                i = np.nanargmin(np.abs(x - self.sp_normx.value()))
                y = y / y[i]
            y = y + n * self.sp_offset.value()
            lab = f"{r['sample']} {r['technique']}" + (f" {r['label']}" if r.get("label") else "")
            ax.plot(x, y, lw=1.1, color=COLORS[n % len(COLORS)], label=lab)
            xl, yl = c.x_label, c.y_label
            n += 1
        if not n:
            QMessageBox.information(self, "Overlay", f"None of the selected entries has a curve '{name}'.")
            return
        if self.chk_logy.isChecked():
            ax.set_yscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl + ("" if self.cb_norm.currentText() == "none" else " (normalised)"))
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
        self.plot.draw()

    def on_scatter(self):
        xc, yc = self.cb_x.currentText(), self.cb_y.currentText()
        if self.per_sample_mode():
            d = self.selected_rows() if self.chk_sel.isChecked() else self.model.df
            if xc not in d.columns or yc not in d.columns:
                QMessageBox.information(self, "Plot", "Choose existing columns for x and y.")
                return
        else:
            df = self.selected() if self.chk_sel.isChecked() else self.model.df
            if xc not in self.full.columns or yc not in self.full.columns:
                QMessageBox.information(self, "Plot", "Choose existing columns for x and y.")
                return
            # take values from the full table (the column may be hidden in the view)
            d = self.full.set_index("id").loc[df["id"]] if len(df) else self.full.iloc[[]]
        x = pd.to_datetime(d[xc], errors="coerce") if xc in ("measured_at", "saved_at") else \
            pd.to_numeric(d[xc], errors="coerce")
        y = pd.to_numeric(d[yc], errors="coerce")
        ok = x.notna() & y.notna()
        self.plot.figure.clear()
        ax = self.ax = self.plot.figure.add_subplot(111)
        g = self.cb_group.currentText()
        groups = d[g].fillna("–").astype(str) if g in d.columns else pd.Series("all", index=d.index)
        for k, key in enumerate(dict.fromkeys(groups[ok])):
            m = ok & (groups == key)
            ax.plot(x[m], y[m], "o", ms=6, color=COLORS[k % len(COLORS)], label=key)
            for xi, yi, s in zip(x[m], y[m], d.loc[m, "sample"]):
                ax.annotate(s, (xi, yi), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(xc)
        ax.set_ylabel(yc)
        ax.grid(alpha=0.25)
        if ok.any():
            ax.legend(fontsize=7)
        else:
            ax.set_title("no entries with both values")
        self.plot.draw()

    # ------------------------------------------------------------ entries
    def _one(self, what):
        sel = self.selected()
        if sel.empty:
            QMessageBox.information(self, what, "Select a row first.")
            return None
        return sel.iloc[0]

    def on_edit_sample(self):
        r = self._one("Edit sample")
        if r is not None and SampleDialog(current_db(), r["sample"], self).exec():
            BUS.changed.emit()

    def on_assign(self):
        sel = self.selected()
        if sel.empty:
            return
        db = current_db()
        name, ok = QInputDialog.getItem(self, "Assign to sample", "Sample (new name creates a sample):",
                                        db.sample_names(), editable=True)
        if ok and name.strip():
            for mid in sel["id"]:
                db.set_measurement_sample(int(mid), name.strip())
            BUS.changed.emit()

    def on_open_raw(self):
        r = self._one("Open raw file")
        if r is None:
            return
        p = r.get("raw_path")
        if not p or not Path(p).exists():
            QMessageBox.information(self, "Open raw file", f"File not found:\n{p}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(p).parent)))

    def on_open_in_tab(self):
        sel = self.selected()
        if sel.empty:
            QMessageBox.information(self, "Open in tab", "Select rows in the table first.")
            return
        rows = sel.to_dict("records")
        missing = [r["raw_path"] for r in rows if not r.get("raw_path") or not Path(r["raw_path"]).exists()]
        if missing:
            QMessageBox.information(self, "Open in tab", "Raw file(s) not found:\n" + "\n".join(map(str, missing)))
            rows = [r for r in rows if r.get("raw_path") and Path(r["raw_path"]).exists()]
        if rows and self.open_in_tab:
            self.open_in_tab(rows)

    def on_files(self):
        r = self._one("Evaluated files")
        if r is None:
            return
        files = current_db().files(int(r["id"]))
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Files of {r['sample']} {r['technique']} {r.get('label') or ''}")
        dlg.resize(760, 320)
        lw = QListWidget()
        raw = r.get("raw_path")
        for role, path in ([("raw data", raw)] if raw else []) + files:
            ok = path.startswith(("http://", "https://")) or Path(path).exists()
            it = QListWidgetItem(f"{role}:   {path}" + ("" if ok else "   (not found)"))
            it.setData(Qt.UserRole, path)
            lw.addItem(it)

        def open_(folder=False):
            it = lw.currentItem()
            if it is None:
                return
            if str(it.data(Qt.UserRole)).startswith(("http://", "https://")):
                QDesktopServices.openUrl(QUrl(it.data(Qt.UserRole)))
                return
            p = Path(it.data(Qt.UserRole))
            target = p if (p.is_dir() or not folder) else p.parent
            if target.exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

        lw.itemDoubleClicked.connect(lambda *_: open_())
        b1, b2 = QPushButton("Open"), QPushButton("Show in folder")
        b1.clicked.connect(lambda: open_())
        b2.clicked.connect(lambda: open_(True))
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel("Raw file and every file saved from the analysis tabs (double-click opens):"))
        v.addWidget(lw)
        v.addWidget(hbox(b1, b2))
        dlg.exec()

    def on_delete(self):
        sel = self.selected()
        if sel.empty:
            return
        if QMessageBox.question(self, "Delete", f"Delete {len(sel)} measurement(s) from the database?\n"
                                "(The raw files are not touched.)") != QMessageBox.Yes:
            return
        db = current_db()
        for mid in sel["id"]:
            db.delete_measurement(int(mid))
        BUS.changed.emit()

    def on_export(self):
        p, _ = QFileDialog.getSaveFileName(self, "Export table", str(Path.home() / "database.csv"), "CSV (*.csv)")
        if p:
            self.model.df.drop(columns=["ids"], errors="ignore").to_csv(p, index=False)

    def on_change_db(self):
        p, _ = QFileDialog.getSaveFileName(self, "Database file (existing or new)", str(current_db().path),
                                           "SQLite (*.sqlite *.db)", options=QFileDialog.DontConfirmOverwrite)
        if p:
            settings().setValue("db_path", p)
            BUS.changed.emit()

    # -------------------------------------------------------------- elab
    def on_elab_settings(self):
        if ElabDialog(self).exec():
            self.reload()

    def on_sync(self, force=False):
        if not elab.is_configured():
            self.on_elab_settings()
            if not elab.is_configured():
                return
        self.lbl_sync.setText("syncing …")
        start_sync(self, force, lambda msg, ok: self.lbl_sync.setText(msg))

    def on_import_json(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "elabFTW JSON export", str(Path.home()), "JSON (*.json)")
        names = []
        try:
            for p in paths:
                names += elab.import_json(p, current_db())
        except Exception as exc:
            show_error(self, "Import elabFTW JSON", exc)
        if paths:
            self.lbl_sync.setText(f"imported: {', '.join(names) or 'no sample name found in the titles'}")
            BUS.changed.emit()
