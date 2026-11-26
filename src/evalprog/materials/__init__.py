"""Substrate and film databases.

The built-in tables live in ``materials/data/*.json``. A user can override or add
entries by placing files with the same name in ``~/.evaluation_program/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
USER_DIR = Path.home() / ".evaluation_program"


def _load(name: str) -> dict:
    data = json.loads((DATA_DIR / name).read_text(encoding="utf-8"))
    user_file = USER_DIR / name
    if user_file.is_file():
        try:
            data.update(json.loads(user_file.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    return {k: v for k, v in data.items() if not k.startswith("_")}


# --------------------------------------------------------------------------
# Lattice helpers (orthogonal cells: cubic, tetragonal, orthorhombic)
# --------------------------------------------------------------------------
def cell_lengths(lattice: dict) -> tuple[float, float, float]:
    a = float(lattice["a"])
    b = float(lattice.get("b", a))
    c = float(lattice.get("c", a))
    return a, b, c


def d_spacing(lattice: dict, hkl) -> float:
    """d-spacing of (hkl) for an orthogonal cell."""
    a, b, c = cell_lengths(lattice)
    h, k, l = hkl
    inv = (h / a) ** 2 + (k / b) ** 2 + (l / c) ** 2
    if inv <= 0:
        raise ValueError("(000) is not a reflection")
    return 1.0 / np.sqrt(inv)


def direction_length(lattice: dict, uvw) -> float:
    a, b, c = cell_lengths(lattice)
    u, v, w = uvw
    return float(np.sqrt((u * a) ** 2 + (v * b) ** 2 + (w * c) ** 2))


@dataclass
class InPlaneDirection:
    label: str
    spacing: float  # pseudocubic in-plane spacing (Angstrom)


@dataclass
class Substrate:
    name: str
    orientation: str
    system: str
    lattice: dict
    normal: tuple
    orders: list
    d_oop: float  # out-of-plane repeat for order 1 (Angstrom)
    inplane: list = field(default_factory=list)
    note: str = ""

    @property
    def label(self) -> str:
        return f"{self.name} {self.orientation}"

    def reflections(self) -> list[tuple[int, float, str]]:
        """List of (order n, d-spacing, label) for the symmetric scan."""
        out = []
        for n in self.orders:
            hkl = tuple(int(n * x) for x in self.normal)
            out.append((n, self.d_oop / n, "({} {} {})".format(*hkl)))
        return out

    def inplane_spacing(self, direction_label: str | None = None) -> float:
        if not self.inplane:
            return float("nan")
        if direction_label is None:
            return self.inplane[0].spacing
        for d in self.inplane:
            if d.label == direction_label:
                return d.spacing
        raise KeyError(direction_label)


@dataclass
class Film:
    name: str
    a: float
    c: float
    poisson: float
    cell: str = "pseudocubic"
    note: str = ""


@lru_cache(maxsize=1)
def substrate_table() -> dict:
    return _load("substrates.json")


@lru_cache(maxsize=1)
def film_table() -> dict:
    return _load("films.json")


def reload() -> None:
    substrate_table.cache_clear()
    film_table.cache_clear()


def substrate_names() -> list[str]:
    return list(substrate_table().keys())


def orientations(name: str) -> list[str]:
    return list(substrate_table()[name]["orientations"].keys())


def get_substrate(name: str, orientation: str | None = None) -> Substrate:
    entry = substrate_table()[name]
    ors = entry["orientations"]
    if orientation is None:
        orientation = next(iter(ors))
    o = ors[orientation]
    lattice = entry["lattice"]
    normal = tuple(o["normal"])
    d_oop = float(o["d_oop"]) if "d_oop" in o else d_spacing(lattice, normal)
    inplane = [
        InPlaneDirection(d["label"], float(d["spacing"]) if "spacing" in d
                         else direction_length(lattice, d["uvw"]) / float(d.get("divisor", 1)))
        for d in o.get("inplane", [])
    ]
    return Substrate(name=name, orientation=orientation, system=entry.get("system", ""),
                     lattice=lattice, normal=normal, orders=list(o["orders"]),
                     d_oop=d_oop, inplane=inplane, note=entry.get("note", ""))


def film_names() -> list[str]:
    return list(film_table().keys())


def get_film(name: str) -> Film:
    e = film_table()[name]
    return Film(name=name, a=float(e["a"]), c=float(e.get("c", e["a"])),
                poisson=float(e.get("poisson", 0.3)), cell=e.get("cell", "pseudocubic"),
                note=e.get("note", ""))
