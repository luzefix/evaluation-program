"""Widgets shared by the analysis tabs for saving results into the database."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import QCheckBox, QCompleter, QLineEdit, QMessageBox, QPushButton

from evalprog.db.naming import guess_sample
from evalprog.db.store import SAMPLE_TYPES, Database, Measurement, get_db

from .widgets import combo, group, info_label, settings, show_error


class _Bus(QObject):
    changed = Signal()          # database content changed (saved, synced, edited)


BUS = _Bus()
_EMIT = None


def changed_soon():
    """Emit BUS.changed once after a burst of saves (e.g. automatic saving)."""
    global _EMIT
    if _EMIT is None:
        _EMIT = QTimer()
        _EMIT.setSingleShot(True)
        _EMIT.setInterval(300)
        _EMIT.timeout.connect(BUS.changed.emit)
    _EMIT.start()


def db_path() -> str | None:
    """Database file: $EVALPROG_DB, else the one chosen in the Database tab, else the default."""
    import os
    p = os.environ.get("EVALPROG_DB") or settings().value("db_path", "")
    return str(p) if p else None


def current_db() -> Database:
    return get_db(db_path())


class SaveToDB:
    """'Database' group for an analysis tab.

    ``build`` is called on *Save* and returns a :class:`Measurement` without the
    sample name (it is taken from the editable box), or raises ``ValueError``.
    Call :meth:`set_source` after loading a file to pre-fill the sample name.
    """

    def __init__(self, technique: str, build: Callable[[], Measurement], parent=None,
                 name_edit: QLineEdit | None = None):
        self.technique = technique
        self.build = build
        self.parent = parent
        self.own_edit = name_edit is None
        self.ed_sample = name_edit or QLineEdit()
        self.ed_sample.setPlaceholderText("e.g. CIO9 (guessed from file name)")
        self.ed_sample.setToolTip("Sample name. Guessed from the file name; correct it if needed.")
        self.ed_sample.editingFinished.connect(self._on_name)
        self.cb_type = combo(SAMPLE_TYPES)
        self.ed_note = QLineEdit()
        self.ed_note.setPlaceholderText("optional note")
        self.chk_replace = QCheckBox("overwrite earlier entry of this file")
        self.chk_replace.setChecked(True)
        self.chk_auto = QCheckBox("save automatically after each evaluation / export")
        self.chk_auto.setChecked(str(settings().value(f"db_auto_{technique}", "true")).lower() == "true")
        self.chk_auto.toggled.connect(lambda v: settings().setValue(f"db_auto_{technique}", "true" if v else "false"))
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(lambda: self._save(silent=True))
        self.btn = QPushButton("Save to database")
        self.btn.clicked.connect(self.on_save)
        self.lbl = info_label("")
        rows = [("Sample", self.ed_sample)] if self.own_edit else []
        self.group = group("Database", rows + [("Sample type", self.cb_type), ("Note", self.ed_note),
                                               self.chk_auto, self.chk_replace, self.btn, self.lbl])
        BUS.changed.connect(self._refresh_completer)
        self._refresh_completer()

    def _refresh_completer(self):
        try:
            names = current_db().sample_names()
        except Exception:
            names = []
        c = QCompleter(names, self.ed_sample)
        c.setCaseSensitivity(Qt.CaseInsensitive)
        self.ed_sample.setCompleter(c)

    def _on_name(self):
        try:
            s = current_db().get_sample(self.sample())
        except Exception:
            s = None
        if s:
            self.cb_type.setCurrentText(s["sample_type"])
            self.cb_type.setEnabled(False)
            self.cb_type.setToolTip("Existing sample: edit its type in the Database tab")
        else:
            self.cb_type.setEnabled(True)
            self.cb_type.setToolTip("")

    def sample(self) -> str:
        return self.ed_sample.text().strip()

    def set_source(self, path):
        if not path:
            return
        try:
            known = current_db().sample_names()
        except Exception:
            known = []
        name = guess_sample(path, known)
        if name:
            self.ed_sample.setText(name)
        self._on_name()
        self.lbl.setText("")

    def on_save(self):
        self._timer.stop()
        self._save(silent=False)

    def auto(self):
        """Call after every evaluation: saves (debounced) if automatic saving is on."""
        if self.chk_auto.isChecked() and self.sample():
            self._timer.start()

    def attach(self, path, role="export"):
        """Call after writing an evaluated file: stores the entry and the file's path."""
        if not path:
            return
        if not self.sample():
            self.lbl.setText("Enter a sample name to keep track of exported files in the database.")
            return
        self._timer.stop()
        self._save(silent=True, files=[(role, str(path))])

    def _save(self, silent=False, files=()):
        name = self.sample()

        def fail(msg, exc=None):
            if silent:
                self.lbl.setText(f"Not saved to database: {msg}")
            elif exc is not None:
                show_error(self.parent, "Save to database", exc)
            else:
                QMessageBox.information(self.parent, "Save to database", msg)

        if not name:
            return fail("enter the sample name first.")
        try:
            ms = self.build()
        except ValueError as exc:
            return fail(str(exc))
        except Exception as exc:
            return fail(str(exc), exc)
        ms = ms if isinstance(ms, list) else [ms]
        try:
            db = current_db()
            new_sample = db.get_sample(name) is None
            for m in ms:
                m.sample = name
                m.technique = m.technique or self.technique
                m.sample_type = self.cb_type.currentText()
                if self.ed_note.text().strip():
                    m.notes = self.ed_note.text().strip()
                m.files = list(m.files) + list(files)
                db.save_measurement(m, replace=self.chk_replace.isChecked())
        except Exception as exc:
            return fail(str(exc), exc)
        nres, ncur = sum(len(m.results) for m in ms), sum(len(m.curves) for m in ms)
        what = (f"{len(ms)} measurements, " if len(ms) > 1 else "") + f"{nres} result(s)" + \
            (f", {ncur} curve(s)" if ncur else "")
        if files:
            what += f", file {Path(files[-1][1]).name}"
        from datetime import datetime
        self.lbl.setText(f"{datetime.now():%H:%M:%S} saved {what} for <b>{name}</b>"
                         + (" (new sample)" if new_sample else "") + ".")
        self._on_name()
        changed_soon()


def file_date(path) -> str | None:
    """Modification time of the raw file (ISO), as a fallback measurement date."""
    try:
        from datetime import datetime
        return datetime.fromtimestamp(Path(path).stat().st_mtime).isoformat(timespec="seconds")
    except (OSError, TypeError, ValueError):
        return None
