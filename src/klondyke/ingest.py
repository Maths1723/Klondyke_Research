"""Read the YCGC Historic Placer geodatabase / shapefiles.

The layer and field names of the YGS compilation are not documented in the
papers, so ingestion is two-step:

1. ``inspect`` writes a schema report (layers, geometry, CRS, fields,
   example values) to reports/ycgc_schema.md.
2. ``load_drillholes`` maps fields to the canonical schema using
   config/field_map.json. If that file is missing, a best-guess map is
   proposed from field names and written for a human to confirm.

Requires geopandas + pyogrio (installed by setup_venv).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# keywords used to propose a mapping from source fields to canonical columns
GUESS = {
    "hole_id": [r"hole", r"drill.?id", r"dh.?id", r"^id$", r"label", r"name"],
    "ground": [r"ground", r"frozen", r"thaw", r"condition", r"frost"],
    "surface_elev_raw": [r"surf", r"collar", r"ground.?elev", r"top.?elev"],
    "bedrock_elev_raw": [r"bedrock.?elev", r"br.?elev", r"bed.?elev"],
    "overburden_m": [r"overburden", r"muck", r"ob"],
    "dredge_section_m": [r"dredge.?sec", r"gravel", r"section", r"pay.?thick"],
    "grade_cents_yd3": [r"cent", r"grade", r"value", r"yd3", r"cuyd", r"per.?yard"],
    "creek": [r"creek", r"stream", r"drainage"],
    "map_no": [r"map", r"sheet", r"ycgc.?no", r"source"],
    "year": [r"year", r"date"],
}


def _gpd():
    try:
        import geopandas as gpd
        return gpd
    except ImportError as e:  # pragma: no cover
        raise SystemExit("geopandas is not installed - run setup_venv.ps1 first") from e


def find_sources(raw_dir: Path) -> list[Path]:
    """All .gdb folders and .shp files under raw/ (zips must be extracted first)."""
    out = sorted(p for p in raw_dir.rglob("*.gdb") if p.is_dir())
    out += sorted(raw_dir.rglob("*.shp"))
    out += sorted(raw_dir.rglob("*.gpkg"))
    return out


def list_layers(src: Path) -> list[str]:
    if src.suffix.lower() == ".shp":
        return [src.stem]
    import pyogrio
    return [str(l[0]) for l in pyogrio.list_layers(str(src))]


def read_layer(src: Path, layer: str):
    gpd = _gpd()
    if src.suffix.lower() == ".shp":
        return gpd.read_file(src)
    return gpd.read_file(src, layer=layer)


def inspect(raw_dir: Path, report_path: Path) -> dict:
    lines = ["# YCGC data: schema report", "", f"Scanned: `{raw_dir}`", ""]
    summary = {}
    for src in find_sources(raw_dir):
        for layer in list_layers(src):
            g = read_layer(src, layer)
            key = f"{src.name}:{layer}"
            geom = g.geom_type.value_counts().to_dict() if g.geometry is not None else {}
            summary[key] = {"n": len(g), "crs": str(g.crs), "geom": geom, "fields": list(g.columns)}
            lines += [f"## `{key}`", f"- features: {len(g)}", f"- CRS: `{g.crs}`",
                      f"- geometry: {geom}", "", "| field | dtype | non-null | examples |",
                      "|---|---|---|---|"]
            for c in g.columns:
                if c == "geometry":
                    continue
                ex = ", ".join(map(str, g[c].dropna().astype(str).unique()[:4]))
                lines.append(f"| {c} | {g[c].dtype} | {g[c].notna().sum()} | {ex[:80]} |")
            lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return summary


def propose_field_map(fields: list[str]) -> dict:
    out = {}
    for canon, pats in GUESS.items():
        for f in fields:
            if any(re.search(p, f, re.I) for p in pats) and f not in out.values():
                out[canon] = f
                break
    return out


def pick_drillhole_layer(summary: dict) -> str | None:
    """Heuristic: the point layer with the most features and a grade-like field."""
    best, best_n = None, -1
    for key, info in summary.items():
        if "Point" not in "".join(info["geom"].keys()):
            continue
        has_grade = any(re.search(r"cent|grade|value", f, re.I) for f in info["fields"])
        if has_grade and info["n"] > best_n:
            best, best_n = key, info["n"]
    return best


def load_drillholes(raw_dir: Path, field_map_path: Path, target_crs: str = "EPSG:3578"):
    """Return a GeoDataFrame in the canonical schema, reprojected to Yukon Albers.

    EPSG:3578 (NAD83 / Yukon Albers) is equal-area and metric; the YGS papers
    show UTM 7N (EPSG:26907/3154) coordinates, which would also work locally.
    """
    gpd = _gpd()
    fm = json.loads(field_map_path.read_text())
    src_key = fm["layer"]  # "<source name>:<layer>"
    src_name, layer = src_key.split(":", 1)
    src = next(p for p in find_sources(raw_dir) if p.name == src_name)
    g = read_layer(src, layer)
    cols = {v: k for k, v in fm["fields"].items() if v}
    out = g.rename(columns=cols)
    for c in fm.get("scale", {}):  # e.g. feet -> metres if the GDB stores feet
        out[c] = out[c].astype(float) * fm["scale"][c]
    keep = [c for c in fm["fields"] if c in out.columns] + ["geometry"]
    out = out[keep].to_crs(target_crs)
    out["x"], out["y"] = out.geometry.x, out.geometry.y
    out["source_layer"] = src_key
    return out


def load_polygons(raw_dir: Path, layer_key: str, target_crs: str = "EPSG:3578"):
    src_name, layer = layer_key.split(":", 1)
    src = next(p for p in find_sources(raw_dir) if p.name == src_name)
    return read_layer(src, layer).to_crs(target_crs)
