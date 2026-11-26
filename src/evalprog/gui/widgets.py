"""Shared Qt widgets."""

from __future__ import annotations

import traceback
from pathlib import Path

import matplotlib

matplotlib.use("QtAgg")
matplotlib.rcParams.update({"font.size": 9, "axes.titlesize": 9, "legend.fontsize": 8})

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PySide6.QtCore import QSettings, Qt  # noqa: E402
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDoubleSpinBox, QFileDialog,  # noqa: E402
                               QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                               QMessageBox, QScrollArea, QSizePolicy, QSpinBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

SETTINGS = ("EvaluationProgram", "EvaluationProgram")


def settings() -> QSettings:
    return QSettings(*SETTINGS)


class PlotWidget(QWidget):
    """Matplotlib figure with the standard navigation toolbar (zoom, pan, save)."""

    def __init__(self, parent=None, figsize=(8, 6)):
        super().__init__(parent)
        self.figure = Figure(figsize=figsize, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)

    def draw(self):
        self.canvas.draw_idle()


class ResultsTable(QTableWidget):
    """Read-only table. ``formats``: optional list of format specs per column
    (e.g. '.4f'); None means automatic."""

    def __init__(self, columns, parent=None, formats=None):
        super().__init__(0, len(columns), parent)
        self.columns = list(columns)
        self.formats = formats or [None] * len(self.columns)
        self.setHorizontalHeaderLabels(self.columns)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setAlternatingRowColors(True)

    def set_rows(self, rows):
        self.setRowCount(0)
        for r in rows:
            i = self.rowCount()
            self.insertRow(i)
            for j, v in enumerate(r):
                f = self.formats[j] if j < len(self.formats) else None
                txt = format(v, f) if (f and isinstance(v, (float, int)) and v == v) else fmt(v)
                it = QTableWidgetItem(txt)
                it.setTextAlignment(Qt.AlignCenter)
                self.setItem(i, j, it)

    def selected_row(self):
        rows = self.selectionModel().selectedRows()
        return rows[0].row() if rows else None


def fmt(v, digits=5):
    if isinstance(v, float):
        if v != v:  # nan
            return "–"
        if v == 0:
            return "0"
        if abs(v) >= 1e5 or abs(v) < 1e-3:
            return f"{v:.4g}"
        return f"{v:.{digits}g}"
    return str(v)


def spin(value=0.0, lo=-1e9, hi=1e9, decimals=3, step=None, suffix=""):
    s = QDoubleSpinBox()
    s.setDecimals(decimals)
    s.setRange(lo, hi)
    s.setValue(value)
    if step:
        s.setSingleStep(step)
    if suffix:
        s.setSuffix(suffix)
    s.setKeyboardTracking(False)
    return s


def ispin(value=0, lo=-1000, hi=1000):
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(value)
    return s


def combo(items, current=None, editable=False):
    c = QComboBox()
    c.addItems(list(items))
    c.setEditable(editable)
    if current is not None:
        i = c.findText(current)
        if i >= 0:
            c.setCurrentIndex(i)
        elif editable:
            c.setEditText(current)
    return c


def group(title, rows=None):
    """QGroupBox with a form layout; ``rows`` = list of (label, widget) or widget."""
    g = QGroupBox(title)
    form = QFormLayout(g)
    form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    for r in rows or []:
        if isinstance(r, tuple):
            form.addRow(r[0], r[1])
        else:
            form.addRow(r)
    return g


def hbox(*widgets):
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    for x in widgets:
        if isinstance(x, str):
            lay.addWidget(QLabel(x))
        else:
            lay.addWidget(x)
    return w


def scroll_panel(width=380):
    """Left control panel inside a scroll area. Returns (scroll_area, vbox_layout)."""
    inner = QWidget()
    lay = QVBoxLayout(inner)
    lay.setAlignment(Qt.AlignTop)
    area = QScrollArea()
    area.setWidget(inner)
    area.setWidgetResizable(True)
    area.setMinimumWidth(width)
    area.setMaximumWidth(width + 140)
    return area, lay


def info_label(text=""):
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return lab


def open_files(parent, title, filters, multi=False, key="last_dir"):
    s = settings()
    start = s.value(key, str(Path.home()))
    if multi:
        paths, _ = QFileDialog.getOpenFileNames(parent, title, start, filters)
    else:
        p, _ = QFileDialog.getOpenFileName(parent, title, start, filters)
        paths = [p] if p else []
    if paths:
        s.setValue(key, str(Path(paths[0]).parent))
    return paths


def save_file(parent, title, default_path, filters):
    p, _ = QFileDialog.getSaveFileName(parent, title, str(default_path), filters)
    return p


def show_error(parent, title, exc):
    msg = f"{exc}"
    box = QMessageBox(QMessageBox.Warning, title, msg, QMessageBox.Ok, parent)
    box.setDetailedText(traceback.format_exc())
    box.exec()
