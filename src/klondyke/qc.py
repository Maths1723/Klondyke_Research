"""Quality control for digitised YCGC drill-hole tables.

Principle: raw values are never overwritten. Every correction is a new column
and every suspicion is a boolean ``flag_*`` column, so any decision can be
audited and reversed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import units

# Canonical column names used everywhere downstream.
CANONICAL = [
    "hole_id",          # as printed on the map, e.g. "99-9", "87B-12", "141-a"
    "ground",           # Frozen / Partial / Thawed
    "surface_elev_raw", # as recorded (unit and datum to be inferred)
    "bedrock_elev_raw",
    "overburden_m",     # muck above the gravel
    "dredge_section_m", # pay gravel + bedrock dug; grade applies to this
    "grade_cents_yd3",  # cents per cubic yard at $20.67/oz
]

HOLE_RE = re.compile(r"^\s*(?P<claim>\d+)(?P<suffix>[A-Za-z]?)\s*-\s*(?P<hole>[0-9A-Za-z]+)\s*$")


def parse_hole_id(hole_id: str) -> dict:
    """Split '87B-12' into drill line '87B', claim 87, hole '12'."""
    m = HOLE_RE.match(str(hole_id))
    if not m:
        return {"line_id": None, "claim_no": np.nan, "hole_no": None, "hole_num": np.nan}
    claim = int(m["claim"])
    suffix = m["suffix"].upper()
    hole = m["hole"]
    return {
        "line_id": f"{claim}{suffix}",
        "claim_no": claim,
        "hole_no": hole,
        "hole_num": float(hole) if hole.isdigit() else np.nan,
    }


def add_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parsed = pd.DataFrame([parse_hole_id(h) for h in out["hole_id"]], index=out.index)
    for c in parsed:
        out[c] = parsed[c]
    out["flag_bad_id"] = out["line_id"].isna()
    return out


@dataclass
class DatumReport:
    n_used: int
    median_ratio: float      # (overburden+section) in m / (surface - bedrock) in raw units
    iqr_ratio: tuple
    inferred_unit: str       # "ft" or "m" or "unclear"
    note: str


def infer_elevation_unit(df: pd.DataFrame) -> DatumReport:
    """Infer whether elevations are in feet or metres from internal consistency.

    surface - bedrock should equal overburden + dredge section (slightly less
    if the dredge section includes some bedrock). If the ratio of the metric
    thickness to the raw elevation difference is ~0.30-0.33 the elevations are
    in feet; if ~1 they are in metres.
    """
    d = df["surface_elev_raw"] - df["bedrock_elev_raw"]
    t = df["overburden_m"] + df["dredge_section_m"]
    ok = (d > 0) & t.notna() & d.notna()
    r = (t[ok] / d[ok]).astype(float)
    if len(r) == 0:
        return DatumReport(0, np.nan, (np.nan, np.nan), "unclear", "no usable rows")
    med = float(np.median(r))
    q1, q3 = np.percentile(r, [25, 75])
    if 0.27 < med < 0.37:
        unit, note = "ft", "thickness/elevation-difference ~ 0.3048 m/ft: elevations are in feet"
    elif 0.85 < med < 1.2:
        unit, note = "m", "ratio ~ 1: elevations are in metres"
    else:
        unit, note = "unclear", "ratio fits neither feet nor metres; check digitisation"
    return DatumReport(int(ok.sum()), med, (float(q1), float(q3)), unit, note)


def robust_z(x: pd.Series) -> pd.Series:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    return (x - med) / (mad if mad > 0 else np.nan)


def run_qc(df: pd.DataFrame, elev_unit: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Add derived columns and flags. Returns (table, summary dict)."""
    out = add_ids(df)

    # Duplicates: same hole id printed twice. Keep first, flag the rest.
    out["flag_duplicate"] = out.duplicated(subset=["hole_id"], keep="first")
    out["flag_duplicate_identical"] = out.duplicated(subset=CANONICAL, keep="first")

    # Missing values
    for c in ["surface_elev_raw", "bedrock_elev_raw", "overburden_m", "dredge_section_m"]:
        out[f"flag_missing_{c}"] = out[c].isna()

    # Trace and zero assays
    g = out["grade_cents_yd3"].astype(float)
    out["flag_trace"] = np.isclose(g, units.TRACE_CENTS)
    out["flag_zero"] = g == 0

    # Elevation unit and consistency
    rep = infer_elevation_unit(out)
    unit = elev_unit or rep.inferred_unit
    factor = units.FT_M if unit == "ft" else 1.0
    out["elev_unit_assumed"] = unit
    out["section_from_elev_m"] = (out["surface_elev_raw"] - out["bedrock_elev_raw"]) * factor
    out["section_recorded_m"] = out["overburden_m"] + out["dredge_section_m"]
    mismatch = (out["section_from_elev_m"] - out["section_recorded_m"]).abs()
    # 1 ft rounding on each elevation plus 1 ft on each thickness -> allow ~1.5 m
    out["flag_elev_inconsistent"] = mismatch > 1.5
    out["flag_bedrock_above_surface"] = out["bedrock_elev_raw"] > out["surface_elev_raw"]

    # Derived quantities (never overwrite raw)
    out["grade_mg_m3"] = units.cents_per_yd3_to_mg_per_m3(g)
    out["accum_mg_m2"] = units.accumulation_mg_per_m2(out["grade_mg_m3"], out["dredge_section_m"])
    out["total_depth_m"] = out["section_recorded_m"]

    # Outliers on log grade (nugget effect or transcription error)
    lg = np.log10(g.where(g > 0))
    z = robust_z(lg)
    out["flag_outlier_high"] = z > 3.5
    sorted_g = np.sort(g.dropna().values)[::-1]
    if len(sorted_g) > 1:
        out["flag_extreme_vs_next"] = (g == sorted_g[0]) & (sorted_g[0] > 3 * sorted_g[1])
    else:
        out["flag_extreme_vs_next"] = False

    # Line-id plausibility: claim numbers far from their neighbours in table order
    # catch typos such as '878-4' for '87B-4'.
    claims = out["claim_no"]
    med_claim = claims.rolling(9, center=True, min_periods=3).median()
    out["flag_suspect_id"] = (claims - med_claim).abs() > 50

    flag_cols = [c for c in out.columns if c.startswith("flag_")]
    summary = {
        "n_rows": int(len(out)),
        "n_unique_holes": int(out["hole_id"].nunique()),
        "n_lines": int(out["line_id"].nunique()),
        "datum": rep.__dict__,
        "elev_unit_used": unit,
        "flag_counts": {c: int(out[c].sum()) for c in flag_cols},
        "grade_cents_quantiles": {
            q: float(np.nanpercentile(g, q * 100)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
        },
        "share_zero": float((g == 0).mean()),
    }
    return out, summary


def clean_for_model(qc: pd.DataFrame, drop_extreme: bool = False) -> pd.DataFrame:
    """Rows usable as model targets: unique, with grade and section thickness.

    Extreme values are kept by default; the decision to cap or drop them is
    made in the modelling step and recorded in the decision log.
    """
    m = (~qc["flag_duplicate"]) & qc["grade_cents_yd3"].notna() & qc["dredge_section_m"].notna()
    m &= ~qc["flag_bad_id"]
    if drop_extreme:
        m &= ~qc["flag_extreme_vs_next"]
    return qc[m].copy()
