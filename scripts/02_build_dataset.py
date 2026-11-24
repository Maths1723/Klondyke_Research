"""Step 2: canonical drill-hole table for one creek -> QC -> valley coordinates.

Usage:
  python scripts/02_build_dataset.py                    # uses config/project.json + field_map.json
  python scripts/02_build_dataset.py --csv holes.csv    # any CSV in the canonical schema + x, y

Outputs:
  data/processed/<name>_holes.csv    one row per hole, raw + derived + flags + s, n
  data/processed/<name>_axis.csv     valley centreline (map coordinates)
  reports/qc_<name>.md               QC summary, datum check, cross-check vs van Loon (2016)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from klondyke import experiment, qc, valley  # noqa: E402

DEFAULT_PROJECT = {
    "name": "dominion",
    "creek_filter": None,          # {"field": "creek", "contains": "Dominion"}
    "bbox": None,                  # [xmin, ymin, xmax, ymax] in the target CRS
    "target_crs": "EPSG:3578",
    "line_order": "claim_no",      # claims numbered below discovery increase downstream
    "downstream": "increasing",
    "dredge_layer": None,          # "<source>:<layer>" polygons of dredged ground
    "elev_unit": None,             # None = infer from data ("ft" / "m")
}


def load_project():
    p = ROOT / "config" / "project.json"
    cfg = dict(DEFAULT_PROJECT)
    if p.exists():
        cfg.update(json.loads(p.read_text()))
    return cfg


def load_from_gdb(cfg):
    from klondyke import ingest
    g = ingest.load_drillholes(ROOT / "raw", ROOT / "config" / "field_map.json", cfg["target_crs"])
    if cfg.get("creek_filter"):
        f = cfg["creek_filter"]
        g = g[g[f["field"]].astype(str).str.contains(f["contains"], case=False, na=False)]
    if cfg.get("bbox"):
        x0, y0, x1, y1 = cfg["bbox"]
        g = g.cx[x0:x1, y0:y1]
    dredge = None
    if cfg.get("dredge_layer"):
        dredge = ingest.load_polygons(ROOT / "raw", cfg["dredge_layer"], cfg["target_crs"])
        g["in_dredge_limit"] = g.within(dredge.union_all()).values
    return pd.DataFrame(g.drop(columns="geometry")), dredge


def cross_check_vanloon(df: pd.DataFrame) -> str:
    p = ROOT / "data" / "interim" / "vanloon2016_app1.csv"
    if not p.exists():
        return "_van Loon (2016) appendix not parsed (PDF not in raw/); cross-check skipped._"
    vl = pd.read_csv(p).drop_duplicates("hole_id")
    m = df.merge(vl, on="hole_id", suffixes=("", "_vl"))
    if len(m) == 0:
        return "No hole ids in common with van Loon (2016) Appendix 1."
    agree = np.isclose(m.grade_cents_yd3, m.grade_cents_yd3_vl, atol=0.05)
    bad = m.loc[~agree, ["hole_id", "grade_cents_yd3", "grade_cents_yd3_vl"]]
    txt = (f"Matched {len(m)} holes with van Loon (2016) Appendix 1; grade agrees on "
           f"{agree.sum()} ({agree.mean():.0%}).")
    if len(bad):
        txt += "\n\nDisagreements (first 20):\n\n```\n" + bad.head(20).to_string(index=False) + "\n```"
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="canonical-schema CSV with x, y columns (bypasses the GDB)")
    ap.add_argument("--name", help="dataset name (default from config/project.json)")
    a = ap.parse_args()
    cfg = load_project()
    name = a.name or cfg["name"]

    if a.csv:
        raw = pd.read_csv(a.csv)
        dredge = None
    else:
        raw, dredge = load_from_gdb(cfg)
    print(f"{len(raw)} holes loaded")

    tab, summ = qc.run_qc(raw, elev_unit=cfg.get("elev_unit"))
    if "in_dredge_limit" in raw:
        tab["in_dredge_limit"] = raw["in_dredge_limit"].values
    df = qc.clean_for_model(tab)
    df = df[df.x.notna() & df.y.notna()].copy()

    # valley frame from the drill lines themselves
    key = df[cfg["line_order"]].to_numpy(float)
    if cfg["downstream"] == "decreasing":
        key = -key
    poly = valley.centreline_from_drill_lines(df.x.values, df.y.values, key)
    s, n = valley.project_to_centreline(df.x.values, df.y.values, poly)
    offset = float(s.min())
    df["s"], df["n"] = s - offset, n
    df["halfwidth"] = valley.valley_halfwidth(df.s.values, n, df.line_id.values)
    df[experiment.TARGET] = experiment.make_target(df.accum_mg_m2)
    # datum-free bedrock feature: bedrock relative to the lowest bedrock on the same line
    if df.bedrock_elev_raw.notna().any():
        f = 0.3048 if summ["elev_unit_used"] == "ft" else 1.0
        br = df.bedrock_elev_raw * f
        df["bedrock_rel_m"] = br - br.groupby(df.line_id).transform("min")
    df["thawed"] = df.ground.astype(str).str.lower().map({"thawed": 1.0, "partial": 0.5, "frozen": 0.0})

    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}_holes.csv", index=False)
    pd.DataFrame(poly, columns=["x", "y"]).assign(s_offset=offset).to_csv(out / f"{name}_axis.csv", index=False)

    lines = [f"# QC report: {name}", "",
             f"- rows loaded: {summ['n_rows']}; unique holes: {summ['n_unique_holes']}; drill lines: {summ['n_lines']}",
             f"- rows kept for modelling: {len(df)}",
             f"- share of zero assays: {summ['share_zero']:.1%}",
             f"- elevation unit check: {summ['datum']['note']} "
             f"(median ratio {summ['datum']['median_ratio']:.3f}, n={summ['datum']['n_used']})",
             f"- valley length covered: {df.s.max():.0f} m; median half-width of drilled floor: {np.nanmedian(df.halfwidth):.0f} m",
             "", "## Flags", "", "| flag | count |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summ["flag_counts"].items()]
    lines += ["", "## Grade distribution (cents/yd3 at $20.67)", "",
              ", ".join(f"p{int(q*100)}={v:g}" for q, v in summ["grade_cents_quantiles"].items()),
              "", "## Cross-check", "", cross_check_vanloon(tab)]
    (ROOT / "reports" / f"qc_{name}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:9]))


if __name__ == "__main__":
    main()
