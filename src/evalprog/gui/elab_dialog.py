"""Dialog: choose an existing elabFTW entry (searchable, sortable) or create a new one,
and what to append (heading, text, figure, results)."""

from __future__ import annotations

from datetime import date

from PySide6.QtCore import QDate, Qt
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox,
                               QDateEdit, QDialog, QDialogButtonBox, QFormLayout, QGroupBox, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton,
                               QRadioButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

from evalprog.db import elab

COLS = ["ID", "Title", "Category", "Status", "Date", "Owner", "Last change"]


class _NumItem(QTableWidgetItem):
    def __lt__(self, other):
        try:
            return int(self.text()) < int(other.text())
        except ValueError:
            return super().__lt__(other)


class ElabTargetDialog(QDialog):
    """``client``: an :class:`elab.Client`. ``sample``: preselects the sample's entry;
    ``elab_id``: its known entry id (from the database)."""

    def __init__(self, client: elab.Client, title: str, sample: str = "", elab_id=None, parent=None,
                 entries: list | None = None):
        super().__init__(parent)
        self.client = client
        self.sample = sample
        self.elab_id = elab_id
        self.setWindowTitle("Append to elabFTW")
        self.resize(980, 720)
        self._entries = {}
        self._me = None

        self.rb_old = QRadioButton("existing entry")
        self.rb_new = QRadioButton("new entry")
        self.rb_old.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.rb_old)
        grp.addButton(self.rb_new)
        self.rb_old.toggled.connect(self._mode)

        # --- existing
        self.cb_kind = QComboBox()
        self.cb_kind.addItems(["experiments", "items"])
        self.ed_search = QLineEdit()
        self.ed_search.setPlaceholderText("search title, category, status, owner, ID …")
        self.chk_mine = QCheckBox("only mine")
        self.cb_cat_f = QComboBox()
        b_reload = QPushButton("Reload")
        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.doubleClicked.connect(self.accept)
        self.ed_search.textChanged.connect(self._filter)
        self.chk_mine.toggled.connect(self._filter)
        self.cb_cat_f.currentIndexChanged.connect(self._filter)
        self.cb_kind.currentIndexChanged.connect(lambda *_: self.load(force=True))
        b_reload.clicked.connect(lambda: self.load(force=True))
        top = QHBoxLayout()
        for w in (QLabel("Type"), self.cb_kind, QLabel("Category"), self.cb_cat_f, self.ed_search,
                  self.chk_mine, b_reload):
            top.addWidget(w)
        self.box_old = QGroupBox("Choose an entry (click a header to sort, double-click to choose)")
        v = QVBoxLayout(self.box_old)
        v.addLayout(top)
        v.addWidget(self.table)

        # --- new
        self.ed_new_title = QLineEdit(sample)
        self.cb_template = QComboBox()
        self.cb_category = QComboBox()
        self.cb_status = QComboBox()
        self.de_date = QDateEdit(QDate.currentDate())
        self.de_date.setCalendarPopup(True)
        self.de_date.setDisplayFormat("yyyy-MM-dd")
        self.ed_tags = QLineEdit()
        self.ed_tags.setPlaceholderText("comma separated, e.g. Thin Film, Oxides")
        self.cb_template.currentIndexChanged.connect(self._template_changed)
        self.box_new = QGroupBox("New experiment")
        f = QFormLayout(self.box_new)
        f.addRow("Title", self.ed_new_title)
        f.addRow("Template", self.cb_template)
        f.addRow("Category", self.cb_category)
        f.addRow("Status", self.cb_status)
        f.addRow("Date", self.de_date)
        f.addRow("Tags", self.ed_tags)

        # --- content
        self.ed_title = QLineEdit(title)
        self.ed_comment = QPlainTextEdit()
        self.ed_comment.setPlaceholderText("optional text above the figure")
        self.ed_comment.setMaximumHeight(70)
        self.chk_fig = QCheckBox("current figure (PNG, uploaded as attachment and shown in the text)")
        self.chk_fig.setChecked(True)
        self.chk_res = QCheckBox("table of the main results")
        self.chk_res.setChecked(True)
        self.chk_attach = QCheckBox("note the elabFTW link in the database entry")
        self.chk_attach.setChecked(bool(sample))
        box_c = QGroupBox("What to append (at the end of the main text; existing text is not changed)")
        fc = QFormLayout(box_c)
        fc.addRow("Heading", self.ed_title)
        fc.addRow("Text", self.ed_comment)
        fc.addRow(self.chk_fig)
        fc.addRow(self.chk_res)
        fc.addRow(self.chk_attach)

        bb = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.b_ok = bb.addButton("Append to elabFTW", QDialogButtonBox.AcceptRole)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(_row(self.rb_old, self.rb_new))
        lay.addWidget(self.box_old, 3)
        lay.addWidget(self.box_new)
        lay.addWidget(box_c)
        lay.addWidget(bb)
        self._mode()
        if entries is not None:          # for tests / cached lists
            self._entries["experiments"] = entries
            self._fill()
        else:
            self.load()

    # ------------------------------------------------------------ data
    def kind(self):
        return self.cb_kind.currentText()

    def load(self, force=False):
        kind = self.kind()
        if kind in self._entries and not force:
            self._fill()
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._entries[kind] = list(self.client.entities(kind))
            if self._me is None:
                try:
                    self._me = self.client._get("users/me")
                except Exception:
                    self._me = {}
            if self.cb_template.count() == 0:
                self._load_new_lists()
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "elabFTW", f"Could not read the entries: {exc}")
            return
        QApplication.restoreOverrideCursor()
        self._fill()

    def _load_new_lists(self):
        self.templates = self.client._get("experiments_templates")
        self.categories = self.client._get("teams/current/experiments_categories")
        self.statuses = self.client._get("teams/current/experiments_status")
        self.cb_template.addItem("none", None)
        for t in self.templates:
            self.cb_template.addItem(t.get("title", ""), t)
        self.cb_category.addItem("–", None)
        for c in self.categories:
            self.cb_category.addItem(c.get("title", ""), c.get("id"))
        self.cb_status.addItem("–", None)
        for s in self.statuses:
            self.cb_status.addItem(s.get("title", ""), s.get("id"))
        # sensible defaults: the template / category of the sample's own entry, status 'Running'
        i = self.cb_status.findText("Running")
        if i >= 0:
            self.cb_status.setCurrentIndex(i)

    def _fill(self):
        ents = self._entries.get(self.kind(), [])
        cats = sorted({e.get("category_title") or "" for e in ents} - {""})
        cur = self.cb_cat_f.currentText()
        self.cb_cat_f.blockSignals(True)
        self.cb_cat_f.clear()
        self.cb_cat_f.addItems(["all"] + cats)
        self.cb_cat_f.setCurrentText(cur if cur in cats else "all")
        self.cb_cat_f.blockSignals(False)
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        pick = None
        for e in ents:
            i = self.table.rowCount()
            self.table.insertRow(i)
            vals = [str(e.get("id", "")), e.get("title") or "", e.get("category_title") or "",
                    e.get("status_title") or "", str(e.get("date") or "")[:10], e.get("fullname") or "",
                    str(e.get("modified_at") or "")[:16]]
            for j, v in enumerate(vals):
                it = _NumItem(v) if j == 0 else QTableWidgetItem(v)
                if j == 0:
                    it.setData(Qt.UserRole, e)
                self.table.setItem(i, j, it)
            if self.elab_id and e.get("id") == self.elab_id and self.kind() == "experiments":
                pick = e.get("id")
            elif pick is None and self.sample and (e.get("title") or "").lower().startswith(self.sample.lower()):
                pick = e.get("id")
        self.table.setSortingEnabled(True)
        self.table.sortItems(6, Qt.DescendingOrder)
        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(1, min(self.table.columnWidth(1), 360))
        self._filter()
        if pick is not None:
            self.select_id(pick)
        elif self.sample:
            self.ed_search.setText(self.sample)

    def select_id(self, eid):
        for i in range(self.table.rowCount()):
            if self.table.item(i, 0).text() == str(eid):
                self.table.selectRow(i)
                self.table.scrollToItem(self.table.item(i, 0))
                return True
        return False

    def _filter(self, *_):
        words = self.ed_search.text().lower().split()
        cat = self.cb_cat_f.currentText()
        me = (self._me or {}).get("fullname") or (
            f"{(self._me or {}).get('firstname', '')} {(self._me or {}).get('lastname', '')}".strip())
        for i in range(self.table.rowCount()):
            texts = [self.table.item(i, j).text() for j in range(len(COLS))]
            hay = " ".join(texts).lower()
            ok = all(w in hay for w in words)
            ok &= cat in ("all", "") or texts[2] == cat
            if self.chk_mine.isChecked() and me:
                ok &= texts[5] == me
            self.table.setRowHidden(i, not ok)

    def _template_changed(self, *_):
        t = self.cb_template.currentData()
        if t and t.get("category"):
            i = self.cb_category.findData(t["category"])
            if i >= 0:
                self.cb_category.setCurrentIndex(i)

    def _mode(self, *_):
        self.box_old.setVisible(self.rb_old.isChecked())
        self.box_new.setVisible(self.rb_new.isChecked())

    # ---------------------------------------------------------- result
    def new_entry(self) -> dict | None:
        if not self.rb_new.isChecked():
            return None
        t = self.cb_template.currentData()
        return {"title": self.ed_new_title.text().strip(), "template": t.get("id") if t else None,
                "category": self.cb_category.currentData(), "status": self.cb_status.currentData(),
                "date": self.de_date.date().toString("yyyy-MM-dd"),
                "tags": [x.strip() for x in self.ed_tags.text().split(",") if x.strip()]}

    def target(self):
        """(kind, id) of the chosen existing entry."""
        rows = self.table.selectionModel().selectedRows()
        if not rows or self.table.isRowHidden(rows[0].row()):
            raise ValueError("Choose an entry in the list (or 'new entry').")
        return self.kind(), int(self.table.item(rows[0].row(), 0).text())

    def accept(self):
        try:
            if self.rb_new.isChecked():
                if not self.ed_new_title.text().strip():
                    raise ValueError("Enter a title for the new entry.")
            else:
                self.target()
        except ValueError as exc:
            QMessageBox.information(self, "elabFTW", str(exc))
            return
        super().accept()


def _row(*widgets):
    h = QHBoxLayout()
    for w in widgets:
        h.addWidget(w)
    h.addStretch(1)
    return h


def today() -> str:
    return date.today().isoformat()
