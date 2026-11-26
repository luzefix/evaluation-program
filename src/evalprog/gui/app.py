"""Main window."""

from __future__ import annotations

import os
import sys

from PySide6.QtCore import QTimer, QUrl
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QTabWidget

from evalprog import __version__, materials

from evalprog.db import elab

from .afm_tab import AFMTab
from .db_tab import DatabaseTab, start_sync
from .db_widgets import current_db
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
        self.database = DatabaseTab()
        self.tabs.addTab(self.database, "Database")
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
        # elabFTW: check on start and then every hour whether the daily sync is due
        self._sync_thread = None
        self.sync_timer = QTimer(self)
        self.sync_timer.setInterval(3600 * 1000)
        self.sync_timer.timeout.connect(self.auto_sync)
        if not os.environ.get("EVALPROG_SMOKE") and not os.environ.get("EVALPROG_NO_SYNC"):
            QTimer.singleShot(5000, self.auto_sync)
            self.sync_timer.start()

    def auto_sync(self):
        try:
            cfg = elab.load_config()
            if not (elab.is_configured(cfg) and cfg.get("auto_sync", True) and elab.due(current_db())):
                return
        except Exception:
            return
        self.statusBar().showMessage("elabFTW: checking for new entries …")
        start_sync(self, on_done=lambda msg, ok: self.statusBar().showMessage(msg, 0 if not ok else 30000))

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
