"""Main window."""

from __future__ import annotations

import os
import sys

from PySide6.QtCore import QUrl
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QTabWidget

from evalprog import __version__, materials

from .afm_tab import AFMTab
from .placeholder import PLANNED, PlaceholderTab
from .harmonics_tab import HarmonicsTab
from .rsm_tab import RSMTab
from .rt_tab import RTTab
from .widgets import settings
from .xrd_tab import XRDTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Evaluation Program {__version__}")
        self.tabs = QTabWidget()
        self.xrd = XRDTab()
        self.rsm = RSMTab()
        self.rt = RTTab()
        self.tabs.addTab(self.xrd, "XRD")
        self.tabs.addTab(self.rsm, "RSM")
        self.tabs.addTab(self.rt, "R(T)")
        self.harmonics = HarmonicsTab()
        self.tabs.addTab(self.harmonics, "R(H) / Harmonics")
        self.afm = AFMTab()
        self.tabs.addTab(self.afm, "AFM")
        for title, text in PLANNED:
            self.tabs.addTab(PlaceholderTab(title, text), title)
        self.setCentralWidget(self.tabs)
        self._menus()
        s = settings()
        geo = s.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        else:
            self.resize(1500, 950)
        self.tabs.setCurrentIndex(int(s.value("tab", 0)))

    def _menus(self):
        m = self.menuBar().addMenu("&File")
        quit_ = QAction("Quit", self)
        quit_.setShortcut("Ctrl+Q")
        quit_.triggered.connect(self.close)
        m.addAction(quit_)
        m = self.menuBar().addMenu("&Materials")
        a = QAction("Open user materials folder", self)
        a.triggered.connect(self.open_materials)
        m.addAction(a)
        a = QAction("Reload materials", self)
        a.triggered.connect(materials.reload)
        m.addAction(a)
        m = self.menuBar().addMenu("&Help")
        a = QAction("About", self)
        a.triggered.connect(lambda: QMessageBox.about(
            self, "Evaluation Program",
            f"Evaluation Program {__version__}<br>Thin-film XRD, RSM, transport and AFM analysis."))
        m.addAction(a)

    def open_materials(self):
        materials.USER_DIR.mkdir(parents=True, exist_ok=True)
        for name in ("substrates.json", "films.json"):
            f = materials.USER_DIR / name
            if not f.exists():
                f.write_text('{\n  "_comment": "Entries here override or extend the built-in table '
                             f'({name}). Copy an entry from the built-in file as a template."\n}}\n',
                             encoding="utf-8")
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(materials.USER_DIR)))

    def closeEvent(self, ev):
        s = settings()
        s.setValue("geometry", self.saveGeometry())
        s.setValue("tab", self.tabs.currentIndex())
        super().closeEvent(ev)


def run(argv=None) -> int:
    app = QApplication.instance() or QApplication(argv or sys.argv)
    app.setApplicationName("Evaluation Program")
    app.setOrganizationName("EvaluationProgram")
    w = MainWindow()
    w.show()
    if os.environ.get("EVALPROG_SMOKE"):
        return 0
    return app.exec()
