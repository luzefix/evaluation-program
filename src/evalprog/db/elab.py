"""elabFTW synchronisation: synthesis parameters -> local sample database.

Configuration (never stored in the repository)::

    ~/.evaluation_program/elabftw.json
    {
      "url": "https://elab.example.org",          # instance, without /api/v2
      "api_key": "3-abc...",                       # read-only key is enough
      "verify_ssl": true,
      "entities": ["experiments"],                 # and/or "items"
      "categories": ["Thin Film Growth"]           # empty list = all categories
    }

The API key can also be given in the environment variable ``ELABFTW_API_KEY``.

The official client ``elabapi-python`` is used when installed
(``pip install elabapi-python``); otherwise plain HTTPS requests to the same
REST API v2 endpoints are made.

Run ``python -m evalprog.db.elab`` for a sync from the command line (e.g. from
the Windows Task Scheduler); the GUI checks once a day on its own.
"""

from __future__ import annotations

import html
import json
import os
import re
import ssl
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path

from evalprog.materials import USER_DIR

from .naming import guess_sample
from .store import Database, get_db

CONFIG_PATH = USER_DIR / "elabftw.json"
SOURCE = "elabFTW"
SYNC_INTERVAL = timedelta(days=1)


# ----------------------------------------------------------------- config
ALT_CONFIG_PATH = Path(os.environ.get("APPDATA", Path.home())) / "evalprog" / "elabftw.json"


def config_path() -> Path:
    """~/.evaluation_program/elabftw.json, or %APPDATA%/evalprog/elabftw.json if only that exists."""
    if not CONFIG_PATH.is_file() and ALT_CONFIG_PATH.is_file():
        return ALT_CONFIG_PATH
    return CONFIG_PATH


def load_config(path: Path | None = None) -> dict:
    p = Path(path) if path else config_path()
    cfg = {"url": "", "api_key": "", "verify_ssl": True, "entities": ["experiments"],
           "categories": ["Thin Film Growth"], "auto_sync": True}
    if p.is_file():
        cfg.update(json.loads(p.read_text(encoding="utf-8")))
    if os.environ.get("ELABFTW_API_KEY"):
        cfg["api_key"] = os.environ["ELABFTW_API_KEY"]
    cfg["url"] = cfg["url"].strip().rstrip("/")
    if cfg["url"].startswith("http://") and not cfg.get("allow_http"):
        # a redirect http -> https drops the Authorization header (login page instead of JSON)
        cfg["url"] = "https://" + cfg["url"][7:]
    if cfg["url"].endswith("/api/v2"):
        cfg["url"] = cfg["url"][:-7]
    return cfg


def save_config(cfg: dict, path: Path | None = None):
    p = Path(path) if path else config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass


def is_configured(cfg: dict | None = None) -> bool:
    cfg = cfg or load_config()
    return bool(cfg.get("url") and cfg.get("api_key"))


# ----------------------------------------------------------------- client
class Client:
    """Minimal read-only access to /api/v2/{experiments,items}."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.base = cfg["url"] + "/api/v2"
        self._api = None
        try:
            import elabapi_python as ea   # official client
            conf = ea.Configuration()
            conf.host = self.base
            conf.api_key["api_key"] = cfg["api_key"]
            conf.api_key_prefix["api_key"] = ""
            conf.verify_ssl = bool(cfg.get("verify_ssl", True))
            client = ea.ApiClient(conf)
            client.set_default_header("Authorization", cfg["api_key"])
            self._api = {"experiments": ea.ExperimentsApi(client), "items": ea.ItemsApi(client)}
        except ImportError:
            pass

    def _get(self, path: str, **params):
        url = f"{self.base}/{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"Authorization": self.cfg["api_key"],
                                                   "Accept": "application/json"})
        ctx = None
        if url.startswith("https") and not self.cfg.get("verify_ssl", True):
            ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
            data = r.read().decode("utf-8")
            if "json" not in (r.headers.get("Content-Type") or ""):
                raise RuntimeError(f"elabFTW answered with {r.headers.get('Content-Type')} from {r.geturl()} "
                                   "instead of JSON: check the URL (https) and the API key")
            return json.loads(data)

    def entity(self, kind: str, eid: int) -> dict:
        """One full entity (the list endpoint leaves out the body)."""
        if self._api is not None:
            fn = (self._api[kind].get_experiment if kind == "experiments" else self._api[kind].get_item)
            e = fn(int(eid))
            return e.to_dict() if hasattr(e, "to_dict") else dict(e)
        return self._get(f"{kind}/{int(eid)}")

    def entities(self, kind: str, page: int = 100):
        """All entities of ``kind`` ('experiments' or 'items') as dicts."""
        offset = 0
        while True:
            if self._api is not None:
                fn = (self._api[kind].read_experiments if kind == "experiments" else self._api[kind].read_items)
                batch = [e.to_dict() if hasattr(e, "to_dict") else dict(e) for e in fn(limit=page, offset=offset)]
            else:
                batch = self._get(kind, limit=page, offset=offset)
            yield from batch
            if len(batch) < page:
                return
            offset += page


# ------------------------------------------------------------ HTML tables
class _Tables(HTMLParser):
    """Collect tables as lists of rows of cell texts. Link texts lose their
    category prefix ('Chemicals - DMF' -> 'DMF'); <sub>x</sub> becomes '_x'."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables, self._row, self._cell, self._in_a, self._a = [], None, None, False, ""

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.tables.append([])
        elif tag == "tr" and self.tables:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
        elif tag == "a" and self._cell is not None:
            self._in_a, self._a = True, ""
        elif tag == "sub" and self._cell is not None:
            self._cell.append("_")
        elif tag == "sup" and self._cell is not None:
            self._cell.append("^")
        elif tag in ("p", "br") and self._cell:
            self._cell.append(" ")

    def handle_endtag(self, tag):
        if tag == "a" and self._in_a:
            self._in_a = False
            t = self._a.strip()
            self._cell.append(t.split(" - ", 1)[1] if " - " in t else t)
        elif tag in ("td", "th") and self._cell is not None:
            self._row.append(_clean("".join(self._cell)))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.tables[-1].append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._in_a:
            self._a += data
        elif self._cell is not None:
            self._cell.append(data)


def _clean(t: str) -> str:
    t = html.unescape(t).replace("\xa0", " ").replace("​", "")
    t = re.sub(r"\$_?\{?([^$}]*)\}?\$", r"\1", t)       # LaTeX CaIrO$_3$ -> CaIrO3
    t = re.sub(r"_\s+", "_", t)
    return re.sub(r"\s+", " ", t).strip()


def _num(s: str):
    s = (s or "").strip().replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


RE_PREC = re.compile(r"^(Prec\d+)\s*:\s*(.*?)\s*\bm\s*:\s*(.*?)\s*mg$", re.I)
RE_DEP = re.compile(r"^Dep\s*:\s*([\d.,]*)\s*[µμu]?l\s*in\s*([\d.,]*)\s*sps", re.I)
RE_SOL = re.compile(r"^([\d.,]*)\s*ml\s+(.+)$", re.I)
RE_EQ = re.compile(r"^(.+?)\s*=\s*([-+]?[\d.,]*)\s*(.*)$")
RE_KV = re.compile(r"^([^:]{1,40}?)\s*:\s*(.*)$")


def parse_body(body: str) -> tuple[dict, dict]:
    """Parameters (and units) from the body of a 'Thin Film Growth' entry.

    Layer rows (those with a *Material* cell) are numbered: the first layer gives
    plain keys (``Material``, ``Prec1``, ``Prec1 m``, ``Dep volume``, ...), further
    layers are prefixed ``L2``, ``L3``; the capping layer ``cap``. Empty cells
    are skipped. A 'Parameter | ... / Results | ...' table gives ``result: <name>``.
    """
    p = _Tables()
    p.feed(body or "")
    params, units = {}, {}

    def put(k, v, unit=""):
        if v is None or v == "":
            return
        params[k] = v
        if unit:
            units[k] = unit.strip()

    layer = 0
    for table in p.tables:
        if table and table[0] and table[0][0].lower() == "parameter" and len(table) > 1:
            head = table[0][1:]
            for row in table[1:]:
                for h, v in zip(head, row[1:]):
                    m = re.match(r"^(.*?)\s+in\s+(.*)$", h)
                    name, unit = (m.group(1), m.group(2)) if m else (h, "")
                    n = _num(v)
                    put(f"result: {name}", n if n is not None else v, unit)
            continue
        for row in table:
            if not row:
                continue
            first = row[0]
            has_mat = any(c.lower().startswith("material") for c in row)
            if has_mat:
                layer += 1
            if first.lower().startswith("capping"):
                pre = "cap "
            elif has_mat and layer > 1:
                pre = f"L{layer} "
            else:
                pre = ""
            plain = []
            for c in row[1:]:
                if not c:
                    continue
                if m := RE_PREC.match(c):
                    put(pre + m.group(1), m.group(2))
                    nums = re.findall(r"\d+(?:[.,]\d+)?", m.group(3))
                    put(pre + m.group(1) + " m", _num(nums[-1]) if nums else None, "mg")   # last = weighed
                    if "ideal" in m.group(3).lower() and len(nums) > 1:
                        put(pre + m.group(1) + " m ideal", _num(nums[0]), "mg")
                elif m := RE_DEP.match(c):
                    put(pre + "Dep volume", _num(m.group(1)), "µl")
                    put(pre + "Dep sps", _num(m.group(2)), "sps")
                elif m := RE_SOL.match(c):
                    if _num(m.group(1)) is not None:   # template default without volume = unused layer
                        put(pre + "Solvent", m.group(2))
                        put(pre + "Solvent volume", _num(m.group(1)), "ml")
                elif m := RE_EQ.match(c):
                    k = m.group(1).replace(" ", "")
                    put(pre + k, _num(m.group(2)), m.group(3))
                elif m := RE_KV.match(c):
                    k, v = m.group(1).strip(), m.group(2).strip()
                    if k.lower() in ("overview", "data in"):
                        continue                   # equipment table of the template
                    if k.lower() == "material" and v:
                        v = v.split()[0]           # 'CaIrO3 CIO' -> 'CaIrO3'
                    n = _num(v)
                    put(pre + k, n if n is not None else v)
                else:
                    plain.append(c)
            if plain and first.endswith(":"):
                put(pre + first.rstrip(":").strip(), "; ".join(plain))
    return params, units


def parse_extra_fields(metadata) -> tuple[dict, dict]:
    """elabFTW 'extra fields' (templates with metadata) -> params, units."""
    if not metadata:
        return {}, {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            return {}, {}
    params, units = {}, {}
    for k, f in (metadata.get("extra_fields") or {}).items():
        v = f.get("value") if isinstance(f, dict) else f
        if v in (None, ""):
            continue
        n = _num(str(v))
        params[k] = n if n is not None else v
        if isinstance(f, dict) and f.get("unit"):
            units[k] = f["unit"]
    return params, units


# ----------------------------------------------------------------- mapping
@dataclass
class Entry:
    sample: str
    fields: dict = field(default_factory=dict)      # columns of the samples table
    params: dict = field(default_factory=dict)
    units: dict = field(default_factory=dict)


def entry_from_entity(e: dict, kind: str = "experiments") -> Entry | None:
    title = _clean(e.get("title") or "")
    sample = guess_sample(title.replace("(", " ("))
    if not sample:
        return None
    params, units = parse_body(e.get("body") or e.get("body_html") or "")
    xp, xu = parse_extra_fields(e.get("metadata"))
    params.update(xp)
    units.update(xu)
    rest = title.split(sample, 1)[1] if sample in title else ""
    rest = rest[1:].strip() if rest.startswith("_") else ""      # 'CIO38_STO(001)': substrate after '_' 
    fields = {"elab_kind": kind, "elab_id": e.get("id"), "elab_modified": e.get("modified_at")}
    m = re.match(r"^([A-Za-z0-9]+)\s*(\(\s*\d[\d\s\-]*\))?", rest)
    if m:
        fields["substrate"] = m.group(1)
        if m.group(2):
            fields["orientation"] = m.group(2).replace(" ", "")
    if isinstance(params.get("Material"), str):
        fields["material"] = params["Material"]
    category = e.get("category_title") or ""
    fields["sample_type"] = "thin film" if "film" in category.lower() else None
    meta = {"elab: id": e.get("id"), "elab: title": title, "elab: date": e.get("date"),
            "elab: status": e.get("status_title"), "elab: category": category,
            "elab: link": e.get("sharelink"), "elab: tags": (e.get("tags") or "").replace("|", ", ")}
    params.update({k: v for k, v in meta.items() if v not in (None, "")})
    return Entry(sample, fields, params, units)


# -------------------------------------------------------------------- sync
@dataclass
class SyncReport:
    seen: int = 0
    updated: list = field(default_factory=list)
    created: list = field(default_factory=list)
    skipped: int = 0

    def __str__(self):
        return (f"{self.seen} entries checked, {len(self.created)} new sample(s), "
                f"{len(self.updated)} updated, {self.skipped} unchanged/ignored")


def apply_entry(db: Database, entry: Entry, force: bool = False) -> str:
    """Write one entry into the database. Returns 'created', 'updated' or 'unchanged'."""
    s = db.get_sample(entry.sample)
    if s and not force and s.get("elab_id") == entry.fields.get("elab_id") and \
            s.get("elab_modified") == entry.fields.get("elab_modified"):
        return "unchanged"
    f = {k: v for k, v in entry.fields.items() if v not in (None, "")}
    if s is None:
        db.ensure_sample(entry.sample, f.pop("sample_type", None) or "thin film", **f)
        status = "created"
    else:
        f.pop("sample_type", None)
        # do not overwrite values the user typed in by hand
        for k in ("substrate", "orientation", "material"):
            if s.get(k):
                f.pop(k, None)
        db.update_sample(entry.sample, **f)
        status = "updated"
    db.set_params(entry.sample, entry.params, source=SOURCE, units=entry.units, replace_source=True)
    return status


def sync(db: Database | None = None, cfg: dict | None = None, force: bool = False, log=print) -> SyncReport:
    cfg = cfg or load_config()
    if not is_configured(cfg):
        raise RuntimeError(f"elabFTW is not configured: create {CONFIG_PATH} with url and api_key")
    db = db or get_db()
    client = Client(cfg)
    cats = [c.lower() for c in cfg.get("categories") or []]
    rep = SyncReport()
    for kind in cfg.get("entities") or ["experiments"]:
        for e in client.entities(kind):
            rep.seen += 1
            if cats and (e.get("category_title") or "").lower() not in cats:
                rep.skipped += 1
                continue
            sample = guess_sample(_clean(e.get("title") or "").replace("(", " ("))
            if not sample:
                rep.skipped += 1
                continue
            s = db.get_sample(sample)
            if s and not force and s.get("elab_id") == e.get("id") and s.get("elab_modified") == e.get("modified_at"):
                rep.skipped += 1
                continue
            if not (e.get("body") or e.get("body_html")):
                e = client.entity(kind, e["id"])
            entry = entry_from_entity(e, kind)
            if entry is None:
                rep.skipped += 1
                continue
            st = apply_entry(db, entry, force)
            if st == "created":
                rep.created.append(entry.sample)
            elif st == "updated":
                rep.updated.append(entry.sample)
            else:
                rep.skipped += 1
    db.set_meta("elab_last_sync", datetime.now().isoformat(timespec="seconds"))
    if log:
        log(str(rep))
    return rep


def due(db: Database | None = None) -> bool:
    """True if the last successful sync is older than one day."""
    db = db or get_db()
    last = db.get_meta("elab_last_sync")
    if not last:
        return True
    try:
        return datetime.now() - datetime.fromisoformat(last) >= SYNC_INTERVAL
    except ValueError:
        return True


def import_json(path, db: Database | None = None) -> list[str]:
    """Import an elabFTW JSON export (one entity or a list) without the API."""
    db = db or get_db()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for e in data if isinstance(data, list) else [data]:
        entry = entry_from_entity(e, e.get("type") or "experiments")
        if entry is not None:
            apply_entry(db, entry, force=True)
            out.append(entry.sample)
    return out


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Sync synthesis parameters from elabFTW")
    ap.add_argument("--db", help="database file (default: ~/.evaluation_program/evalprog.sqlite)")
    ap.add_argument("--force", action="store_true", help="re-import unchanged entries")
    ap.add_argument("--if-due", action="store_true", help="only sync if the last sync is > 1 day old")
    ap.add_argument("--import-json", nargs="*", help="import exported JSON files instead of the API")
    a = ap.parse_args(argv)
    db = get_db(a.db)
    if a.import_json:
        for p in a.import_json:
            print(p, "->", ", ".join(import_json(p, db)) or "no sample name found")
        return 0
    if a.if_due and not due(db):
        print("last sync less than a day ago, nothing to do")
        return 0
    sync(db, force=a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
