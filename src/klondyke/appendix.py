"""Parse the drill-hole attribute table printed in van Loon (2016), Appendix 1.

The paper (Yukon Exploration and Geology 2015, p. 261-271) prints one row per
hole:  HOLE GROUND SURFACE_ELEV BEDROCK_ELEV OVERBURDEN_M DREDGE_SECTION_M CENTS_YD3
with '<Null>' for missing values. The table has no coordinates; it is used
(1) as an independent check on whatever the YCGC geodatabase contains and
(2) for attribute-level QC before coordinates are available.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

NUM = r"(<Null>|-?\d+(?:\.\d+)?)"
ROW = re.compile(
    rf"^\s*(?P<hole>\d+[A-Za-z]?-[0-9A-Za-z]+)\s+(?P<ground>Frozen|Partial|Thawed)\s+"
    rf"{NUM}\s+{NUM}\s+{NUM}\s+{NUM}\s+{NUM}\s*$"
)


def _num(v):
    return np.nan if v == "<Null>" else float(v)


def parse_text(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        m = ROW.match(line)
        if m:
            g = m.groups()
            rows.append({
                "hole_id": g[0], "ground": g[1],
                "surface_elev_raw": _num(g[2]), "bedrock_elev_raw": _num(g[3]),
                "overburden_m": _num(g[4]), "dredge_section_m": _num(g[5]),
                "grade_cents_yd3": _num(g[6]),
            })
    df = pd.DataFrame(rows)
    df["source"] = "vanLoon2016_App1"
    df["row_order"] = np.arange(len(df))
    return df


def parse_pdf(pdf_path: str | Path, pages: range | None = None) -> pd.DataFrame:
    import pdfplumber  # local import: optional dependency

    texts = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        idx = pages if pages is not None else range(len(pdf.pages))
        for i in idx:
            texts.append(pdf.pages[i].extract_text() or "")
    return parse_text("\n".join(texts))
