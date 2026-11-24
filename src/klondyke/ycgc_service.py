"""Ingest the YGS 'Historic placer data' map service export (Esri JSON).

Source: https://deptweb.gov.yk.ca/prod/rest/services/YGS_HistoricPlacer/Historic_placer_data/MapServer
  layer 0  Historic drillholes   (13,358 points)
  layer 1  Dredge limits and cuts (5,352 polygons; type, year, estimated vs actual value)
  layer 2  Other historic workings
  layer 3  Historic map index    (933 source maps; title, year, placement confidence)
Coordinates: EPSG:3578 (NAD83 / Yukon Albers), metres.

Facts established by inspecting the service (24 Nov 2026), encoded below:
  * OVERBURDEN, DREDGE_SECTION, SURFACE_/BEDROCK_ELEVATION are in FEET. On every
    map with all four, (OB + DS) / (surface - bedrock) = 1.04-1.10, i.e. the
    dredge section includes ~1-2 ft of bedrock.
  * GOLD_PRICE is filled for ~14% of holes: $20.67 (1,418), $35 (288), $35.30 (83),
    $38.50 (77). Grade must be normalised to a common price before modelling.
  * The digitisers left notes such as "No legend, 3rd num might not be cents per
    cu yd - depth to BR?" (~600 holes). Those grades are not trusted.
  * The same physical hole is often drawn on several maps: 675 cross-map pairs
    lie within 15 m with identical non-zero grade (median 4.8 m apart).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import units

AMBIGUOUS_NOTE = re.compile(
    r"might not be cents|no legend|last # unknown|grade recorded in cents per cubic yard\?|"
    r"overburden might be overall depth|data not legible|conflicting", re.I)
TRACE_NOTE = re.compile(r"^\s*t\s*$|trace|^\s*t;", re.I)


def read_esri_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def points_frame(js: dict) -> pd.DataFrame:
    rows = []
    for f in js["features"]:
        a = dict(f["attributes"])
        g = f.get("geometry") or {}
        a["x"], a["y"] = g.get("x"), g.get("y")
        rows.append(a)
    return pd.DataFrame(rows)


def polygons_frame(js: dict) -> pd.DataFrame:
    rows = []
    for f in js["features"]:
        a = dict(f["attributes"])
        a["rings"] = [np.asarray(r, float) for r in (f.get("geometry") or {}).get("rings", [])]
        rows.append(a)
    return pd.DataFrame(rows)


def build_holes(drill: pd.DataFrame, maps: pd.DataFrame) -> pd.DataFrame:
    """Canonical hole table: raw columns kept, derived columns added, flags explicit."""
    # the map index repeats some MAP_IDs (several footprints per map): one row per map
    m = (maps.sort_values("OBJECTID") if "OBJECTID" in maps else maps).drop_duplicates("MAP_ID")
    m = m[["MAP_ID", "TITLE", "YEAR", "CONFIDENCE_OF_PLACEMENT", "LINK_TO_MAP"]].rename(
        columns={"TITLE": "map_title", "YEAR": "map_year", "CONFIDENCE_OF_PLACEMENT": "map_confidence",
                 "LINK_TO_MAP": "map_link"})
    d = drill.merge(m, on="MAP_ID", how="left", validate="many_to_one")
    assert len(d) == len(drill)
    out = pd.DataFrame({
        "uid": d.MAP_ID.astype(str) + ":" + d.OBJECTID.astype(str),
        "map_id": d.MAP_ID, "hole_id": d.DRILL_HOLE_ID.astype(str).str.strip(),
        "point_type": d.POINT_TYPE.astype(str).str.strip().str.lower(),
        "ground": d.GROUND_CONDITION.replace({"Partly Thawed": "Partial", "Not Specified": None}),
        "surface_elev_ft": d.SURFACE_ELEVATION, "bedrock_elev_ft": d.BEDROCK_ELEVATION,
        "overburden_ft": d.OVERBURDEN, "dredge_section_ft": d.DREDGE_SECTION,
        "grade_cents_raw": d.CENTS_PER_CUBIC_YARD, "gold_price_recorded": d.GOLD_PRICE,
        "notes": d.NOTES, "x": d.x, "y": d.y,
        "map_title": d.map_title, "map_year": d.map_year, "map_confidence": d.map_confidence,
        "map_link": d.map_link,
    })
    # price basis: a map's recorded price applies to its unrecorded holes; else assume $20.67.
    # Recorded values that are not historical gold prices ($1.00, $7.60, $15.60) are not used:
    # those holes are flagged and fall back to $20.67 like any unrecorded basis.
    valid = [20.67, 35.0, 35.3, 38.5]
    rec = out.gold_price_recorded.where(out.gold_price_recorded.isin(valid))
    out["flag_price_odd"] = out.gold_price_recorded.notna() & ~out.gold_price_recorded.isin(valid)
    per_map = rec.groupby(out.map_id).agg(lambda v: v.dropna().mode().iloc[0] if v.notna().any() else np.nan)
    out["gold_price"] = rec.fillna(out.map_id.map(per_map))
    out["flag_price_assumed"] = out.gold_price.isna()
    out["gold_price"] = out.gold_price.fillna(units.HISTORIC_PRICE_PER_OZ)
    out["grade_cents_2067"] = out.grade_cents_raw * units.HISTORIC_PRICE_PER_OZ / out.gold_price

    notes = out.notes.fillna("")
    out["flag_grade_ambiguous"] = notes.str.contains(AMBIGUOUS_NOTE)
    out["flag_trace_note"] = notes.str.contains(TRACE_NOTE)
    out["flag_trace"] = out.flag_trace_note | np.isclose(out.grade_cents_raw.fillna(-1), units.TRACE_CENTS)
    out["flag_not_drillhole"] = ~out.point_type.isin(["drill hole", "bedrock test hole"])

    # unit consistency per hole (feet throughout -> ratio ~1.06)
    diff = out.surface_elev_ft - out.bedrock_elev_ft
    out["thickness_ratio"] = (out.overburden_ft + out.dredge_section_ft) / diff.where(diff > 0)
    out["flag_thickness_inconsistent"] = (out.thickness_ratio - 1.06).abs() > 0.35

    # metric and mass units
    out["overburden_m"] = out.overburden_ft * units.FT_M
    out["dredge_section_m"] = out.dredge_section_ft * units.FT_M
    out["grade_mg_m3"] = units.cents_per_yd3_to_mg_per_m3(out.grade_cents_2067)
    out["accum_mg_m2"] = out.grade_mg_m3 * out.dredge_section_m
    return out


def mark_cross_map_duplicates(h: pd.DataFrame, radius: float = 15.0) -> pd.DataFrame:
    """Same physical hole on several maps: within `radius` m, different map, same non-zero grade.

    Keeps one record per cluster, preferring (1) a record with bedrock elevation,
    (2) a record with a recorded gold price, (3) the earliest map. Returns the
    table with `dup_group` and `flag_duplicate`.
    """
    from scipy.spatial import cKDTree
    h = h.copy()
    ok = h.grade_cents_raw.notna() & (h.grade_cents_raw > 0.1) & h.x.notna()
    idx = np.where(ok)[0]
    tree = cKDTree(h.loc[ok, ["x", "y"]].to_numpy())
    parent = np.arange(len(h))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    g = h.grade_cents_raw.to_numpy(); mp = h.map_id.to_numpy()
    for i, j in tree.query_pairs(radius):
        a, b = idx[i], idx[j]
        if mp[a] != mp[b] and abs(g[a] - g[b]) < 0.05:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
    roots = np.array([find(i) for i in range(len(h))])
    h["dup_group"] = roots
    pref = (h.bedrock_elev_ft.notna().astype(int) * 4 + (~h.flag_price_assumed).astype(int) * 2
            - h.map_year.fillna(9999) / 1e4)
    order = h.assign(_p=pref).sort_values(["dup_group", "_p"], ascending=[True, False])
    keep_idx = order.groupby("dup_group").head(1).index
    h["flag_duplicate"] = ~h.index.isin(keep_idx)
    h["dup_size"] = h.groupby("dup_group").uid.transform("size")
    return h


PRICE_RATIOS = {35.0: 35.0 / 20.67, 35.3: 35.3 / 20.67, 38.5: 38.5 / 20.67}


def infer_price_basis(h: pd.DataFrame, radius: float = 25.0, min_pairs: int = 3) -> pd.DataFrame:
    """Detect maps whose grades were revalued at a later gold price without recording it.

    The same hole redrawn on a later map at $35 (or $35.30, $38.50) shows a grade
    exactly 1.690x (1.708x, 1.863x) the $20.67 original. For each map pair with
    co-located holes, count pairs whose ratio matches one of these to 0.6%
    (or to one rounding unit). A map whose holes are consistently the richer member
    of a matching pair, against a map with a known $20.67 basis, is assigned that price.

    Returns one row per (rich map, reference map, price) with the pair count.
    """
    from scipy.spatial import cKDTree
    ok = h.grade_cents_raw.notna() & (h.grade_cents_raw > 0.5) & h.x.notna()
    g = h.loc[ok, "grade_cents_raw"].to_numpy(); mp = h.loc[ok, "map_id"].to_numpy()
    tree = cKDTree(h.loc[ok, ["x", "y"]].to_numpy())
    rows = []
    for i, j in tree.query_pairs(radius):
        if mp[i] == mp[j]:
            continue
        hi, lo = (i, j) if g[i] > g[j] else (j, i)
        for price, r in PRICE_RATIOS.items():
            if abs(g[hi] / g[lo] - r) / r < 0.006 or abs(g[hi] - g[lo] * r) <= 0.11:
                rows.append((mp[hi], mp[lo], price))
                break
    if not rows:
        return pd.DataFrame(columns=["map_id", "reference_map", "inferred_price", "n_pairs"])
    t = pd.DataFrame(rows, columns=["map_id", "reference_map", "inferred_price"])
    t = t.groupby(["map_id", "reference_map", "inferred_price"]).size().rename("n_pairs").reset_index()
    return t[t.n_pairs >= min_pairs].sort_values("n_pairs", ascending=False)


def apply_inferred_prices(h: pd.DataFrame, inferred: pd.DataFrame) -> pd.DataFrame:
    """Set the price basis of maps with no recorded price from the co-location evidence."""
    h = h.copy()
    best = inferred.sort_values("n_pairs", ascending=False).drop_duplicates("map_id")
    for _, r in best.iterrows():
        m = (h.map_id == r.map_id) & h.flag_price_assumed
        h.loc[m, "gold_price"] = r.inferred_price
        h.loc[m, "flag_price_assumed"] = False
        h.loc[m, "price_source"] = f"inferred from {r.n_pairs} co-located holes on {r.reference_map}"
    h["grade_cents_2067"] = h.grade_cents_raw * units.HISTORIC_PRICE_PER_OZ / h.gold_price
    h["grade_mg_m3"] = units.cents_per_yd3_to_mg_per_m3(h.grade_cents_2067)
    h["accum_mg_m2"] = h.grade_mg_m3 * h.dredge_section_m
    return h


def points_in_polygons(x, y, polys: pd.DataFrame) -> np.ndarray:
    """Index of the first polygon containing each point (-1 if none). Holes in rings ignored."""
    from matplotlib.path import Path as MPath
    P = np.column_stack([x, y])
    out = np.full(len(P), -1)
    for k, rings in enumerate(polys.rings):
        for r in rings:
            if len(r) < 3:
                continue
            xmin, ymin = r.min(0); xmax, ymax = r.max(0)
            cand = (out < 0) & (P[:, 0] >= xmin) & (P[:, 0] <= xmax) & (P[:, 1] >= ymin) & (P[:, 1] <= ymax)
            if cand.any():
                inside = MPath(r).contains_points(P[cand])
                ci = np.where(cand)[0][inside]
                out[ci] = k
    return out
