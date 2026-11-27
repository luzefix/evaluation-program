import numpy as np
import pytest

from evalprog.db import elab
from evalprog.db.naming import guess_sample
from evalprog.db.store import Curve, Database, Measurement, Result

BODY = """
<table>
<tr><td>Stack</td><td>Substrate: <a href="x">Substrate - STO (001) BOE terminated</a></td>
<td>Batch: <a href="y">Sample Preparation - STO - BOE - 11/26</a></td><td>Substrate Nr: 3</td></tr>
<tr><td>&nbsp;</td><td><strong>Material</strong>: CaIrO$_3$ CIO</td>
<td><strong>Prec1</strong>: <a href="z">Chemicals - Calcium acetylacetonate</a> &nbsp; <strong>m</strong>:&nbsp; 13,6&nbsp; mg</td>
<td><strong>Prec3</strong>:&nbsp;&nbsp; <strong>m</strong>:&nbsp; mg</td>
<td><strong>Dep</strong>:&nbsp;1000&nbsp;µl&nbsp;in 200 sps</td>
<td><strong>3 </strong>ml <a href="d">Chemicals - Dimethylformamide</a></td></tr>
<tr><td>&nbsp;</td><td><strong>Material</strong>:</td><td><strong>Dep</strong>: &nbsp; µl in sps</td>
<td>ml <a href="d">Chemicals - Dimethylformamide</a></td></tr>
<tr><td>Repetitions:</td><td>&nbsp;</td><td>5min USB @ 30°C</td></tr>
<tr><td>&nbsp;</td><td>T<sub>Dep&nbsp;</sub>= 720 °C</td><td>Air-Flow = 1,55m<sup>3</sup>/h</td>
<td>V<sub>Heater</sub>= V</td></tr>
</table>
<table><tr><td>Parameter</td><td>$d$ in nm</td><td>$RRR$</td></tr>
<tr><td>Results</td><td>21,5</td><td>&nbsp;</td></tr></table>
"""


@pytest.mark.parametrize("path, name", [
    ("CIO9_RSM_103.xrdml", "CIO9"), ("2026-01-03_CIO12_XRD.xy", "CIO12"), ("XRD_SRO4b.xy", "SRO4b"),
    ("RT1_CIO3.dat", "CIO3"), ("sample.dat", ""), ("C:/data/CIO7/afm_001.spm", "CIO7")])
def test_guess_sample(path, name):
    assert guess_sample(path) == name


def test_guess_known():
    assert guess_sample("lam-a3 RT.dat", known=["Lam-A3"]) == "Lam-A3"


def test_store_roundtrip(tmp_path):
    db = Database(tmp_path / "t.sqlite")
    x = np.linspace(20, 80, 50)
    m = Measurement("CIO9", "XRD", raw_path="a.xy", results=[Result("c", 3.95, 0.01, "Å", "CaIrO3"),
                                                            Result("substrate", "STO (001)")],
                    curves=[Curve("scan", x, x ** 2, "2θ", "I")])
    db.save_measurement(m)
    db.save_measurement(m, replace=True)
    db.save_measurement(Measurement("Lam1", "R(T)", sample_type="lamella", results=[Result("Tc_mid", 30.1)]))
    t = db.table()
    assert len(t) == 2
    assert t.loc[t["sample"] == "CIO9", "XRD c mean (Å)"].item() == pytest.approx(3.95)
    assert "XRD c mean (Å)" in t.attrs["main"]
    assert t.loc[t["sample"] == "Lam1", "sample_type"].item() == "lamella"
    mid = int(t.loc[t["sample"] == "CIO9", "id"].item())
    c = db.curves(mid)[0]
    assert np.allclose(c.y, x ** 2)
    db.set_params("CIO9", {"T_Dep": "720", "note": "ok"}, units={"T_Dep": "°C"})
    t = db.table()
    assert t.loc[t["sample"] == "CIO9", "syn: T_Dep (°C)"].item() == 720
    db.rename_sample("Lam1", "CIO9")      # merge
    assert db.sample_names() == ["CIO9"]
    assert len(db.table()) == 2


def test_parse_body():
    p, u = elab.parse_body(BODY)
    assert p["Material"] == "CaIrO3"
    assert p["Prec1"] == "Calcium acetylacetonate" and p["Prec1 m"] == pytest.approx(13.6)
    assert "Prec3" not in p and "L2 Solvent" not in p
    assert p["Dep volume"] == 1000 and p["Dep sps"] == 200
    assert p["Solvent"] == "Dimethylformamide" and p["Solvent volume"] == 3
    assert p["T_Dep"] == 720 and u["T_Dep"] == "°C"
    assert p["Air-Flow"] == pytest.approx(1.55)
    assert p["Substrate Nr"] == 3
    assert p["Repetitions"] == "5min USB @ 30°C"
    assert p["result: d"] == pytest.approx(21.5) and u["result: d"] == "nm"
    assert "V_Heater" not in p


def test_entry_and_apply(tmp_path):
    e = {"id": 1165, "title": "CIO38_STO(001)", "body": BODY, "metadata": None,
         "category_title": "Thin Film Growth", "modified_at": "2026-11-23 15:01:06", "date": "2026-11-23"}
    entry = elab.entry_from_entity(e)
    assert entry.sample == "CIO38"
    assert entry.fields["substrate"] == "STO" and entry.fields["orientation"] == "(001)"
    db = Database(tmp_path / "t.sqlite")
    assert elab.apply_entry(db, entry) == "created"
    assert elab.apply_entry(db, entry) == "unchanged"
    s = db.get_sample("CIO38")
    assert s["material"] == "CaIrO3" and s["sample_type"] == "thin film"
    assert (db.get_params("CIO38")["key"] == "T_Dep").any()


def test_sync_with_fake_client(tmp_path, monkeypatch):
    ents = [{"id": 1, "title": "CIO40_STO(001)", "body": BODY, "category_title": "Thin Film Growth",
             "modified_at": "x"},
            {"id": 2, "title": "Termination STO", "body": "", "category_title": "Sample Preparation"}]
    monkeypatch.setattr(elab.Client, "entities", lambda self, kind: iter(ents))
    db = Database(tmp_path / "t.sqlite")
    rep = elab.sync(db, cfg={"url": "https://x", "api_key": "k", "categories": ["Thin Film Growth"]}, log=None)
    assert rep.created == ["CIO40"] and rep.skipped == 1
    assert not elab.due(db)


def test_main_and_detail_columns(tmp_path):
    db = Database(tmp_path / "t.sqlite")
    res = [Result("a_ip", 3.90, 0.01, "Å", "LSCO (1 0 9)"), Result("a_ip", 3.92, 0.01, "Å", "LSCO (1 0 3)"),
           Result("a_ip", 3.91, 0.007, "Å", "LSCO"), Result("Qx", 1.1, None, "1/Å", "LSCO (1 0 9)")]
    db.save_measurement(Measurement("LSCO1", "RSM", results=res))
    db.save_measurement(Measurement("LSCO2", "RSM", results=[Result("a_ip", 3.89, 0.01, "Å", "LSCO (1 0 9)")]))
    t = db.table()
    assert t.attrs["main"] == ["RSM a_ip (Å)"]                    # mean, or the single reflection
    assert "RSM a_ip [LSCO (1 0 9)] (Å)" in t.attrs["detail"]
    assert list(t["RSM a_ip (Å)"]) == [3.91, 3.89]


def test_append_result(tmp_path, monkeypatch):
    calls = []
    cfg = {"url": "https://elab.test", "api_key": "k"}
    c = elab.Client(cfg)
    c._api = None

    def fake_request(method, path, data=None, content_type=None):
        calls.append((method, path, data, content_type))
        return 201, {"Location": "https://elab.test/api/v2/experiments/7/uploads/42"}, ""

    def fake_get(path, **kw):
        if path == "experiments/7/uploads/42":
            return {"id": 42, "real_name": "fig.png", "long_name": "ab/abc.png", "storage": 1}
        if path == "experiments/7":
            return {"body": "<p>old</p>"}
        raise AssertionError(path)

    monkeypatch.setattr(c, "_request", fake_request)
    monkeypatch.setattr(c, "_get", fake_get)
    fig = tmp_path / "fig.png"
    fig.write_bytes(b"\x89PNG")
    link = elab.append_result(cfg, "experiments", 7, "XRD: CIO9.xy", [("c mean", 3.95, 0.002, "Å")], fig,
                              "after anneal", client=c)
    assert link == "https://elab.test/experiments.php?mode=view&id=7"
    up, patch = calls
    assert up[0] == "POST" and up[1] == "experiments/7/uploads" and b'filename="fig.png"' in up[2]
    body = __import__("json").loads(patch[2])["body"]
    assert patch[0] == "PATCH" and body.startswith("<p>old</p>")
    assert "app/download.php?" in body and "f=ab%2Fabc.png" in body and "3.95 ± 0.002" in body
    assert "after anneal" in body


def test_create_experiment(monkeypatch):
    calls = []
    c = elab.Client({"url": "https://elab.test", "api_key": "k"})

    def fake_request(method, path, data=None, content_type=None):
        calls.append((method, path, __import__("json").loads(data)))
        return 201, {"Location": "https://elab.test/api/v2/experiments/99"}, ""

    monkeypatch.setattr(c, "_request", fake_request)
    eid = c.create_experiment("CIO40_STO(001)", template=10, category=3, status=26, date="2026-11-27",
                              tags=["Thin Film"])
    assert eid == 99
    assert calls[0] == ("POST", "experiments", {"template": 10})
    assert calls[1] == ("PATCH", "experiments/99", {"title": "CIO40_STO(001)", "category": 3, "status": 26,
                                                   "date": "2026-11-27"})
    assert calls[2] == ("POST", "experiments/99/tags", {"tag": "Thin Film"})
