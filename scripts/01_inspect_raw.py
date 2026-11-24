"""Step 1: unzip downloads in raw/, report the schema, propose a field map.

Usage:  python scripts/01_inspect_raw.py
Outputs:
  reports/ycgc_schema.md             every layer, CRS, field and example values
  config/field_map.proposed.json     best-guess mapping to the canonical schema
  data/interim/vanloon2016_app1.csv  the paper's attribute table, if the PDF is in raw/

Then: check the proposal, save it as config/field_map.json, run step 2.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RAW = ROOT / "raw"


def unzip_all():
    for z in RAW.glob("*.zip"):
        dest = RAW / "extracted" / z.stem
        if dest.exists():
            continue
        with zipfile.ZipFile(z) as zf:
            for m in zf.namelist():  # refuse path traversal
                if m.startswith("/") or ".." in Path(m).parts:
                    raise SystemExit(f"unsafe path in {z.name}: {m}")
            zf.extractall(dest)
        print("extracted", z.name, "->", dest.relative_to(ROOT))


def parse_papers():
    from klondyke import appendix
    for pdf in RAW.glob("*.pdf"):
        if "16_vanloon" in pdf.name.lower():
            df = appendix.parse_pdf(pdf)
            out = ROOT / "data" / "interim" / "vanloon2016_app1.csv"
            df.to_csv(out, index=False)
            print(f"parsed {len(df)} rows from {pdf.name} -> {out.relative_to(ROOT)}")


def main():
    unzip_all()
    parse_papers()
    from klondyke import ingest
    srcs = ingest.find_sources(RAW)
    if not srcs:
        print("No .gdb / .shp / .gpkg found under raw/. Download the YCGC files first (see DATA_SOURCES.md).")
        return
    summary = ingest.inspect(RAW, ROOT / "reports" / "ycgc_schema.md")
    for k, v in summary.items():
        print(f"{k:60s} n={v['n']:6d}  {v['geom']}  {v['crs'][:40]}")
    layer = ingest.pick_drillhole_layer(summary)
    proposal = {"layer": layer,
                "fields": ingest.propose_field_map(summary[layer]["fields"]) if layer else {},
                "scale": {},
                "_note": "Check every mapping against reports/ycgc_schema.md. 'scale' multiplies a "
                         "column (e.g. 0.3048 if a thickness is stored in feet)."}
    cfg = ROOT / "config"; cfg.mkdir(exist_ok=True)
    (cfg / "field_map.proposed.json").write_text(json.dumps(proposal, indent=2))
    print("\nproposed drill-hole layer:", layer)
    print(json.dumps(proposal["fields"], indent=2))
    print("\nReview, then copy config/field_map.proposed.json -> config/field_map.json")


if __name__ == "__main__":
    main()
