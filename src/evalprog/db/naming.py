"""Guess the sample name from a measurement file name.

Convention for thin films: element initials in capitals followed by the sample
number, e.g. ``CIO9`` (CaIrO3 no. 9), optionally with a letter suffix for pieces
(``CIO9b``). Known sample names from the database are preferred, so names that
do not follow the convention (bulk crystals, lamellae) are found once they exist.
"""

from __future__ import annotations

import re
from pathlib import Path

# tokens that look like sample names but are measurement types / instruments
STOPWORDS = {"XRD", "RSM", "AFM", "RT", "RH", "MPMS", "PPMS", "ETO", "VSM", "TEM", "SEM", "EDX", "FIB",
             "XRR", "HR", "OMEGA", "PHI", "SCAN", "MAP", "SPM", "LVM", "RUN", "T", "H", "K"}

PATTERN = re.compile(r"(?<![A-Za-z0-9])([A-Z]{2,10})(\d{1,4})([a-z]?)(?![0-9A-Za-z])")


def _tokens(stem: str):
    for m in PATTERN.finditer(stem):
        letters, num, suf = m.groups()
        if letters in STOPWORDS:
            continue
        yield m.start(), letters + num + suf


def guess_sample(path: str | Path, known: list[str] | None = None) -> str:
    """Best guess of the sample name for ``path`` ('' if nothing fits)."""
    p = Path(str(path))
    candidates = [p.stem] + [part for part in reversed(p.parent.parts[-3:])]
    if known:
        for text in candidates:
            low = re.sub(r"[\s_\-.]+", " ", text.lower())
            best = ""
            for name in known:
                n = re.sub(r"[\s_\-.]+", " ", name.lower())
                if re.search(r"(?<![a-z0-9])" + re.escape(n) + r"(?![0-9])", low) and len(n) > len(best):
                    best = name
            if best:
                return best
    for text in candidates:
        toks = list(_tokens(text))
        if toks:
            return toks[0][1]
    return ""
