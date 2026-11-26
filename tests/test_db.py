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
    assert t.loc[t["sample"] == "CIO9", "c [CaIrO3]"].item() == pytest.approx(3.95)
    assert t.loc[t["sample"] == "Lam1", "sample_type"].item() == "lamella"
    mid = int(t.loc[t["sample"] == "CIO9", "id"].item())
    c = db.curves(mid)[0]
    assert np.allclose(c.y, x ** 2)
    db.set_params("CIO9", {"T_Dep": "720", "note": "ok"}, units={"T_Dep": "°C"})
    t = db.table()
    assert t.loc[t["sample"] == "CIO9", "syn: T_Dep"].item() == 720
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
