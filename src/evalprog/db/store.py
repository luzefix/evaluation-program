"""Local SQLite database of samples, synthesis parameters and measurements.

Layout (all tables are created on first use)::

    samples        one row per sample (thin film, bulk crystal, lamella, ...)
    sample_params  free key/value parameters of a sample (synthesis, from elabFTW or typed in)
    measurements   one row per saved analysis (technique, raw file, settings)
    results        scalar results of a measurement (key, label, value, error, unit)
    curves         x/y data of a measurement for overlays (stored as compressed npz)

Results and parameters are stored as key/value rows, so new quantities need no
schema change. :meth:`Database.table` pivots everything into one flat
:class:`pandas.DataFrame` (one row per measurement) for filtering and sorting.
"""

from __future__ import annotations

import io
import json
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from evalprog.materials import USER_DIR

DEFAULT_PATH = USER_DIR / "evalprog.sqlite"

SAMPLE_TYPES = ["thin film", "bulk crystal", "lamella", "powder", "other"]
TECHNIQUES = ["XRD", "RSM", "R(T)", "R(H)", "AFM", "MPMS", "other"]

# sample columns that the GUI exposes; film-specific ones are optional (NULL)
SAMPLE_FIELDS = ["name", "sample_type", "material", "substrate", "orientation", "thickness_nm",
                 "parent", "notes"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    id           INTEGER PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE COLLATE NOCASE,
    sample_type  TEXT NOT NULL DEFAULT 'thin film',
    material     TEXT,
    substrate    TEXT,
    orientation  TEXT,
    thickness_nm REAL,
    parent       TEXT,
    notes        TEXT,
    elab_kind    TEXT,
    elab_id      INTEGER,
    elab_modified TEXT,
    created      TEXT NOT NULL,
    updated      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sample_params (
    sample_id INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
    key       TEXT NOT NULL,
    value     TEXT,
    num       REAL,
    unit      TEXT,
    source    TEXT NOT NULL DEFAULT 'manual',
    PRIMARY KEY (sample_id, key)
);
CREATE TABLE IF NOT EXISTS measurements (
    id          INTEGER PRIMARY KEY,
    sample_id   INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
    technique   TEXT NOT NULL,
    raw_path    TEXT,
    label       TEXT,
    measured_at TEXT,
    saved_at    TEXT NOT NULL,
    settings    TEXT,
    notes       TEXT
);
CREATE TABLE IF NOT EXISTS results (
    measurement_id INTEGER NOT NULL REFERENCES measurements(id) ON DELETE CASCADE,
    key   TEXT NOT NULL,
    label TEXT NOT NULL DEFAULT '',
    value REAL,
    text  TEXT,
    error REAL,
    unit  TEXT,
    PRIMARY KEY (measurement_id, key, label)
);
CREATE TABLE IF NOT EXISTS curves (
    measurement_id INTEGER NOT NULL REFERENCES measurements(id) ON DELETE CASCADE,
    name    TEXT NOT NULL,
    x_label TEXT,
    y_label TEXT,
    data    BLOB NOT NULL,
    PRIMARY KEY (measurement_id, name)
);
CREATE TABLE IF NOT EXISTS files (
    measurement_id INTEGER NOT NULL REFERENCES measurements(id) ON DELETE CASCADE,
    path  TEXT NOT NULL,
    role  TEXT,
    added TEXT,
    PRIMARY KEY (measurement_id, path)
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX IF NOT EXISTS ix_meas_sample ON measurements(sample_id);
CREATE INDEX IF NOT EXISTS ix_res_key ON results(key);
"""

# fixed columns of Database.table(); everything else is a result or parameter
BASE_COLUMNS = ["id", "sample", "sample_type", "technique", "label", "measured_at", "saved_at",
                "material", "substrate", "orientation", "thickness_nm", "raw_path", "files"]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _num(v):
    """float(v) if v is a finite number (or numeric string), else None."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def result_column(key: str, label: str = "") -> str:
    return f"{key} [{label}]" if label else key


@dataclass
class Result:
    key: str
    value: float | str | None
    error: float | None = None
    unit: str = ""
    label: str = ""          # e.g. phase / reflection; keeps several peaks apart


@dataclass
class Curve:
    name: str
    x: np.ndarray
    y: np.ndarray
    x_label: str = "x"
    y_label: str = "y"
    single: bool = False     # store as float32 (7 digits; halves the size of noisy data)


@dataclass
class Measurement:
    """Everything an analysis tab hands to :meth:`Database.save_measurement`."""

    sample: str
    technique: str
    raw_path: str | None = None
    label: str = ""
    measured_at: str | None = None
    settings: dict = field(default_factory=dict)
    results: list[Result] = field(default_factory=list)
    curves: list[Curve] = field(default_factory=list)
    notes: str = ""
    sample_type: str | None = None   # used only when the sample is created
    files: list = field(default_factory=list)   # (role, path) of evaluated / exported files


def _pack(x, y, single=False) -> bytes:
    buf = io.BytesIO()
    dt = np.float32 if single else float
    np.savez_compressed(buf, x=np.asarray(x, dt), y=np.asarray(y, dt))
    return buf.getvalue()


def _unpack(blob: bytes):
    with np.load(io.BytesIO(blob)) as z:
        return z["x"].astype(float), z["y"].astype(float)


class Database:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(self.path))
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA foreign_keys = ON")
        self.con.executescript(SCHEMA)
        self.con.commit()

    def close(self):
        self.con.close()

    # ----------------------------------------------------------------- meta
    def get_meta(self, key, default=None):
        r = self.con.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return r[0] if r else default

    def set_meta(self, key, value):
        with self.con:
            self.con.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, str(value)))

    # -------------------------------------------------------------- samples
    def sample_names(self) -> list[str]:
        return [r[0] for r in self.con.execute("SELECT name FROM samples ORDER BY name")]

    def get_sample(self, name: str) -> dict | None:
        r = self.con.execute("SELECT * FROM samples WHERE name = ?", (name,)).fetchone()
        return dict(r) if r else None

    def ensure_sample(self, name: str, sample_type: str | None = None, **fields) -> int:
        """Return the id of sample ``name``, creating it if necessary."""
        name = name.strip()
        if not name:
            raise ValueError("Sample name is empty")
        s = self.get_sample(name)
        if s is not None:
            return s["id"]
        now = _now()
        cols = {"name": name, "sample_type": sample_type or "thin film", "created": now, "updated": now}
        cols.update({k: v for k, v in fields.items() if k in SAMPLE_FIELDS + ["elab_kind", "elab_id",
                                                                             "elab_modified"]})
        with self.con:
            cur = self.con.execute(
                f"INSERT INTO samples ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                list(cols.values()))
        return cur.lastrowid

    def update_sample(self, name: str, **fields):
        allowed = SAMPLE_FIELDS + ["elab_kind", "elab_id", "elab_modified"]
        fields = {k: v for k, v in fields.items() if k in allowed}
        if not fields:
            return
        fields["updated"] = _now()
        sets = ", ".join(f"{k} = ?" for k in fields)
        with self.con:
            self.con.execute(f"UPDATE samples SET {sets} WHERE name = ?", [*fields.values(), name])

    def rename_sample(self, old: str, new: str):
        """Rename, or merge into ``new`` if that sample already exists."""
        new = new.strip()
        src, dst = self.get_sample(old), self.get_sample(new)
        if src is None or not new or old == new:
            return
        with self.con:
            if dst is None or dst["id"] == src["id"]:
                self.con.execute("UPDATE samples SET name = ?, updated = ? WHERE id = ?",
                                 (new, _now(), src["id"]))
            else:
                self.con.execute("UPDATE measurements SET sample_id = ? WHERE sample_id = ?",
                                 (dst["id"], src["id"]))
                self.con.execute("INSERT OR IGNORE INTO sample_params SELECT ?, key, value, num, unit, source "
                                 "FROM sample_params WHERE sample_id = ?", (dst["id"], src["id"]))
                self.con.execute("DELETE FROM samples WHERE id = ?", (src["id"],))

    def delete_sample(self, name: str):
        with self.con:
            self.con.execute("DELETE FROM samples WHERE name = ?", (name,))

    def samples_frame(self) -> pd.DataFrame:
        """One row per sample with its parameters as extra columns."""
        df = pd.read_sql_query("SELECT * FROM samples ORDER BY name", self.con)
        par = pd.read_sql_query("SELECT sample_id, key, value, num FROM sample_params", self.con)
        if len(par):
            par["v"] = par["num"].where(par["num"].notna(), par["value"])
            wide = par.pivot_table(index="sample_id", columns="key", values="v", aggfunc="first")
            df = df.merge(wide, left_on="id", right_index=True, how="left")
        return df

    # ----------------------------------------------------------- parameters
    def set_params(self, sample: str, params: dict, source: str = "manual", units: dict | None = None,
                   replace_source: bool = False):
        """Store key/value parameters of a sample. ``replace_source`` first removes
        all parameters of this source (used by the elabFTW sync)."""
        sid = self.ensure_sample(sample)
        units = units or {}
        with self.con:
            if replace_source:
                self.con.execute("DELETE FROM sample_params WHERE sample_id = ? AND source = ?",
                                 (sid, source))
            for k, v in params.items():
                if v is None or str(v).strip() == "":
                    self.con.execute("DELETE FROM sample_params WHERE sample_id = ? AND key = ?", (sid, k))
                    continue
                self.con.execute("INSERT OR REPLACE INTO sample_params VALUES (?, ?, ?, ?, ?, ?)",
                                 (sid, k, str(v), _num(v), units.get(k, ""), source))

    def get_params(self, sample: str) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT p.key, p.value, p.unit, p.source FROM sample_params p JOIN samples s ON s.id = p.sample_id "
            "WHERE s.name = ? ORDER BY p.key", self.con, params=(sample,))

    # --------------------------------------------------------- measurements
    def save_measurement(self, m: Measurement, replace: bool = False) -> int:
        """Insert a measurement. With ``replace`` an earlier entry with the same sample,
        technique, raw file and label is overwritten (re-analysis of the same file)."""
        sid = self.ensure_sample(m.sample, m.sample_type)
        keep_files = []
        with self.con:
            if replace:
                keep_files = self.con.execute(
                    "SELECT f.role, f.path FROM files f JOIN measurements m ON m.id = f.measurement_id "
                    "WHERE m.sample_id = ? AND m.technique = ? AND IFNULL(m.raw_path, '') = ? AND "
                    "IFNULL(m.label, '') = ?", (sid, m.technique, m.raw_path or "", m.label or "")).fetchall()
                self.con.execute(
                    "DELETE FROM measurements WHERE sample_id = ? AND technique = ? AND "
                    "IFNULL(raw_path, '') = ? AND IFNULL(label, '') = ?",
                    (sid, m.technique, m.raw_path or "", m.label or ""))
            cur = self.con.execute(
                "INSERT INTO measurements (sample_id, technique, raw_path, label, measured_at, saved_at, "
                "settings, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (sid, m.technique, m.raw_path, m.label, m.measured_at, _now(),
                 json.dumps(m.settings, default=str), m.notes))
            mid = cur.lastrowid
            for r in m.results:
                num = _num(r.value)
                txt = None if num is not None or r.value is None else str(r.value)
                self.con.execute("INSERT OR REPLACE INTO results VALUES (?, ?, ?, ?, ?, ?, ?)",
                                 (mid, r.key, r.label or "", num, txt, _num(r.error), r.unit))
            for c in m.curves:
                self.con.execute("INSERT OR REPLACE INTO curves VALUES (?, ?, ?, ?, ?)",
                                 (mid, c.name, c.x_label, c.y_label, _pack(c.x, c.y, c.single)))
        self.add_files(mid, [tuple(f) for f in keep_files] + list(m.files))
        return mid

    def add_files(self, mid: int, files):
        """Attach evaluated / exported files: iterable of (role, path)."""
        with self.con:
            for role, path in files:
                if path:
                    self.con.execute("INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?)",
                                     (int(mid), str(Path(path)), role, _now()))

    def files(self, mid: int) -> list[tuple[str, str]]:
        return [(r[0], r[1]) for r in self.con.execute(
            "SELECT role, path FROM files WHERE measurement_id = ? ORDER BY added", (int(mid),))]

    def find_measurement(self, sample: str, technique: str, raw_path: str | None, label: str = ""):
        r = self.con.execute(
            "SELECT m.id FROM measurements m JOIN samples s ON s.id = m.sample_id WHERE s.name = ? AND "
            "m.technique = ? AND IFNULL(m.raw_path, '') = ? AND IFNULL(m.label, '') = ?",
            (sample, technique, raw_path or "", label or "")).fetchone()
        return r[0] if r else None

    def delete_measurement(self, mid: int):
        with self.con:
            self.con.execute("DELETE FROM measurements WHERE id = ?", (int(mid),))

    def set_measurement_sample(self, mid: int, sample: str):
        sid = self.ensure_sample(sample)
        with self.con:
            self.con.execute("UPDATE measurements SET sample_id = ? WHERE id = ?", (sid, int(mid)))

    def measurement(self, mid: int) -> dict:
        r = self.con.execute("SELECT m.*, s.name AS sample FROM measurements m JOIN samples s "
                             "ON s.id = m.sample_id WHERE m.id = ?", (int(mid),)).fetchone()
        if r is None:
            raise KeyError(mid)
        d = dict(r)
        d["settings"] = json.loads(d["settings"] or "{}")
        d["results"] = pd.read_sql_query("SELECT key, label, value, text, error, unit FROM results "
                                         "WHERE measurement_id = ?", self.con, params=(int(mid),))
        return d

    def curve_names(self, mid: int) -> list[str]:
        return [r[0] for r in self.con.execute("SELECT name FROM curves WHERE measurement_id = ?", (int(mid),))]

    def curves(self, mid: int) -> list[Curve]:
        out = []
        for r in self.con.execute("SELECT name, x_label, y_label, data FROM curves WHERE measurement_id = ?",
                                  (int(mid),)):
            x, y = _unpack(r["data"])
            out.append(Curve(r["name"], x, y, r["x_label"], r["y_label"]))
        return out

    # ------------------------------------------------------------ flat view
    def table(self, with_params: bool = True) -> pd.DataFrame:
        """One row per measurement: base columns, then every result key (as
        ``key`` or ``key [label]``), then sample parameters (prefixed ``syn:``)."""
        base = pd.read_sql_query(
            "SELECT m.id, s.name AS sample, s.sample_type, m.technique, m.label, m.measured_at, m.saved_at, "
            "s.material, s.substrate, s.orientation, s.thickness_nm, m.raw_path, m.sample_id "
            "FROM measurements m JOIN samples s ON s.id = m.sample_id ORDER BY m.saved_at", self.con)
        res = pd.read_sql_query("SELECT measurement_id, key, label, value, text FROM results", self.con)
        if len(res):
            res["col"] = [result_column(k, lab) for k, lab in zip(res["key"], res["label"])]
            res["v"] = res["value"].astype(object).where(res["value"].notna(), res["text"])
            wide = res.pivot_table(index="measurement_id", columns="col", values="v", aggfunc="first")
            wide = wide[sorted(wide.columns, key=str.lower)]
            base = base.merge(wide, left_on="id", right_index=True, how="left")
        if with_params:
            par = pd.read_sql_query("SELECT sample_id, key, value, num FROM sample_params", self.con)
            if len(par):
                par["v"] = par["num"].astype(object).where(par["num"].notna(), par["value"])
                par["key"] = "syn: " + par["key"]
                wide = par.pivot_table(index="sample_id", columns="key", values="v", aggfunc="first")
                wide = wide[sorted(wide.columns, key=str.lower)]
                base = base.merge(wide, left_on="sample_id", right_index=True, how="left")
        nf = pd.read_sql_query("SELECT measurement_id, COUNT(*) AS n FROM files GROUP BY measurement_id", self.con)
        base.insert(len(BASE_COLUMNS) - 1, "files", base["id"].map(dict(zip(nf["measurement_id"], nf["n"])))
                    .fillna(0).astype(int))
        base = base.drop(columns="sample_id")
        for c in base.columns[len(BASE_COLUMNS):]:
            conv = pd.to_numeric(base[c], errors="coerce")
            if conv.notna().sum() == base[c].notna().sum():
                base[c] = conv
        return base


_DB: Database | None = None


def get_db(path: str | Path | None = None) -> Database:
    """Shared database instance (re-opened when ``path`` changes)."""
    global _DB
    if path is not None and _DB is not None and Path(path) != _DB.path:
        _DB.close()
        _DB = None
    if _DB is None:
        _DB = Database(path)
    return _DB
