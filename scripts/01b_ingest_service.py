"""Step 1b: ingest the YGS map-service export and audit it.

Input (raw/): ycgc_layer0_drillholes.json, ycgc_layer1_dredge_limits_cuts.json,
              ycgc_layer3_map_index.json   (Esri JSON, EPSG:3578)
Outputs:
  data/interim/ycgc_holes.csv          every record, canonical columns + flags
  data/interim/ycgc_dredge.pkl         dredge/thaw/cut polygons (attributes + rings)
  reports/ycgc_audit.md                what is in the data and what is wrong with it
  reports/ycgc_map_audit.csv           one row per source map

Usage:  python scripts/01b_ingest_service.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from klondyke import ycgc_service as Y  # noqa: E402

RAW = ROOT / "raw"


def main():
    f0 = next(RAW.glob("ycgc_layer0_*.json"))
    f1 = next(RAW.glob("ycgc_layer1_*.json"))
    f3 = next(RAW.glob("ycgc_layer3_*.json"))
    drill = Y.points_frame(Y.read_esri_json(f0))
    dredge = Y.polygons_frame(Y.read_esri_json(f1))
    maps = pd.DataFrame([f["attributes"] for f in Y.read_esri_json(f3)["features"]])

    h = Y.build_holes(drill, maps)
    h["price_source"] = np.where(h.gold_price_recorded.notna(), "recorded", "map default / assumed")
    inferred = Y.infer_price_basis(h)
    rec = h.groupby("map_id").gold_price_recorded.agg(lambda v: v.dropna().mode().iloc[0] if v.notna().any() else None)
    inferred["recorded_price"] = inferred.map_id.map(rec)
    inferred["status"] = np.where(inferred.recorded_price.isna(), "NEW (unrecorded)",
                                  np.where(np.isclose(inferred.recorded_price.astype(float), inferred.inferred_price),
                                           "confirms recorded", "CONFLICTS with recorded"))
    inferred.to_csv(ROOT / "reports" / "ycgc_inferred_price_basis.csv", index=False)
    h = Y.apply_inferred_prices(h, inferred)
    h = Y.mark_cross_map_duplicates(h)

    # mined-out ground: which polygon (if any) contains each hole, and of what type / year
    k = Y.points_in_polygons(h.x.values, h.y.values, dredge)
    h["dredge_poly"] = k
    h["dredge_type"] = np.where(k >= 0, dredge.TYPE.values[np.maximum(k, 0)], None)
    h["dredge_year"] = np.where(k >= 0, dredge.YEAR.values[np.maximum(k, 0)], np.nan)

    h["usable"] = (h.grade_cents_raw.notna() & ~h.flag_grade_ambiguous & ~h.flag_duplicate
                   & ~h.flag_not_drillhole & h.dredge_section_ft.notna() & (h.dredge_section_ft > 0))
    (ROOT / "data" / "interim").mkdir(parents=True, exist_ok=True)
    h.to_csv(ROOT / "data" / "interim" / "ycgc_holes.csv", index=False)
    dredge.to_pickle(ROOT / "data" / "interim" / "ycgc_dredge.pkl")

    # ---------------------------------------------------------------- audit
    usable = h.usable
    per_map = h.groupby("map_id").agg(
        title=("map_title", "first"), year=("map_year", "first"), confidence=("map_confidence", "first"),
        n=("uid", "size"), n_grade=("grade_cents_raw", lambda v: v.notna().sum()),
        n_usable=("usable", "sum"), n_bedrock=("bedrock_elev_ft", lambda v: v.notna().sum()),
        price=("gold_price", "first"), price_assumed=("flag_price_assumed", "mean"),
        ambiguous=("flag_grade_ambiguous", "mean"), duplicates=("flag_duplicate", "sum"),
        median_thickness_ratio=("thickness_ratio", "median"),
        median_surface_ft=("surface_elev_ft", "median"),
        median_grade_2067=("grade_cents_2067", "median"),
        share_zero=("grade_cents_raw", lambda v: float((v == 0).mean()) if v.notna().any() else np.nan),
        x=("x", "median"), y=("y", "median"),
    ).sort_values("n", ascending=False)
    per_map.to_csv(ROOT / "reports" / "ycgc_map_audit.csv")

    dv = dredge[dredge.VALUE.notna() & dredge.ESTIMATED_VALUE.notna()]
    lines = [
        "# YCGC historic placer data: audit", "",
        f"Source: YGS map service export, {f0.name} etc.", "",
        "## Contents", "",
        f"- records: {len(h):,} on {h.map_id.nunique()} source maps "
        f"({(h.point_type=='drill hole').sum():,} drill holes, {(h.point_type=='shaft').sum():,} shafts, "
        f"{(h.point_type=='bedrock test hole').sum():,} bedrock test holes)",
        f"- with a grade: {h.grade_cents_raw.notna().sum():,}; with bedrock elevation: {h.bedrock_elev_ft.notna().sum():,}",
        f"- usable for grade modelling (grade + section, unambiguous, de-duplicated, drill holes): **{usable.sum():,}**",
        f"- dredge/thaw/cut polygons: {len(dredge):,}; with estimated AND actual value: {len(dv)}",
        "", "## Problems found", "",
        f"- **Units:** thickness and elevation fields are in feet; median (OB+DS)/(surface-bedrock) "
        f"= {h.thickness_ratio.median():.3f} (dredge section includes some bedrock). "
        f"{int(h.flag_thickness_inconsistent.sum())} holes are inconsistent by >35%.",
        f"- **Gold price basis:** recorded for {h.gold_price_recorded.notna().mean():.0%} of holes; "
        f"values {sorted(h.gold_price_recorded.dropna().unique().tolist())}. "
        f"Assumed $20.67 for {h.flag_price_assumed.mean():.0%} (maps with no price anywhere).",
        f"- **Unrecorded price bases detected** from co-located redrawn holes (grade ratio 1.690 / 1.708 / 1.863): "
        f"{'; '.join(f'{r.map_id} = ${r.inferred_price} ({r.n_pairs} pairs, {r.status})' for _, r in inferred.drop_duplicates('map_id').iterrows()) or 'none'}.",
        f"- **Ambiguous grades** (digitiser notes doubt the number is cents/yd3): {int(h.flag_grade_ambiguous.sum())}.",
        f"- **Duplicates across maps** (same grade, <15 m apart): {int(h.flag_duplicate.sum())} records removed; "
        f"{int((h.dup_size>1).groupby(h.dup_group).any().sum())} clusters.",
        f"- **Trace assays:** {int(h.flag_trace.sum())} (notes or 0.1 c/yd3).",
        f"- **Already mined:** {int((h.dredge_poly>=0).sum())} holes fall inside a dredge cut, thaw block or "
        f"other recorded working.",
        "", "## Largest source maps", "", "```",
        per_map.head(30)[["title", "year", "confidence", "n", "n_usable", "n_bedrock", "price",
                          "median_thickness_ratio", "median_grade_2067"]].to_string(), "```",
    ]
    (ROOT / "reports" / "ycgc_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:18]))


if __name__ == "__main__":
    main()
