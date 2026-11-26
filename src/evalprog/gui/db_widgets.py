"""Widgets shared by the analysis tabs for saving results into the database."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QCheckBox, QCompleter, QLineEdit, QMessageBox, QPushButton

from evalprog.db.naming import guess_sample
from evalprog.db.store import SAMPLE_TYPES, Database, Measurement, get_db

from .widgets import combo, group, info_label, settings, show_error


class _Bus(QObject):
    changed = Signal()          # database content changed (saved, synced, edited)


BUS = _Bus()


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
        self.btn = QPushButton("Save to database")
        self.btn.clicked.connect(self.on_save)
        self.lbl = info_label("")
        rows = [("Sample", self.ed_sample)] if self.own_edit else []
        self.group = group("Database", rows + [("Sample type", self.cb_type), ("Note", self.ed_note),
                                               self.chk_replace, self.btn, self.lbl])
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
        name = self.sample()
        if not name:
            QMessageBox.information(self.parent, "Save to database", "Enter the sample name first.")
            return
        try:
            m = self.build()
        except ValueError as exc:
            QMessageBox.information(self.parent, "Save to database", str(exc))
            return
        except Exception as exc:
            show_error(self.parent, "Save to database", exc)
            return
        m.sample = name
        m.technique = m.technique or self.technique
        m.sample_type = self.cb_type.currentText()
        if self.ed_note.text().strip():
            m.notes = self.ed_note.text().strip()
        try:
            db = current_db()
            new_sample = db.get_sample(name) is None
            db.save_measurement(m, replace=self.chk_replace.isChecked())
        except Exception as exc:
            show_error(self.parent, "Save to database", exc)
            return
        what = f"{len(m.results)} result(s)" + (f", {len(m.curves)} curve(s)" if m.curves else "")
        self.lbl.setText(f"Saved {what} for <b>{name}</b>" + (" (new sample)" if new_sample else "") + ".")
        self._on_name()
        BUS.changed.emit()


def file_date(path) -> str | None:
    """Modification time of the raw file (ISO), as a fallback measurement date."""
    try:
        from datetime import datetime
        return datetime.fromtimestamp(Path(path).stat().st_mtime).isoformat(timespec="seconds")
    except (OSError, TypeError, ValueError):
        return None
