"""Write a synthetic creek in the canonical drill-hole schema (for testing steps 2-4).

Elevations are written in FEET on an arbitrary local datum and thicknesses in
metres, mimicking van Loon (2016), so the QC step must detect the unit.

Usage:  python scripts/make_synthetic_csv.py  -> data/interim/synthetic_holes.csv
Then:   python scripts/02_build_dataset.py --csv data/interim/synthetic_holes.csv --name synthetic
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from klondyke import synthetic, units  # noqa: E402

df, _ = synthetic.make_dataset(synthetic.SynthParams(seed=7))
rng = np.random.default_rng(7)
n = len(df)
df["hole_id"] = [f"{c}-{h.split('-')[1]}" for c, h in zip(df.claim_no, df.hole_id)]
df["ground"] = rng.choice(["Frozen", "Partial", "Thawed"], n, p=[0.75, 0.15, 0.10])
df["overburden_m"] = np.round(np.clip(rng.gamma(4, 1.3, n) + 0.002 * np.abs(df.n_true) ** 1.3, 0, 25), 2)
df["dredge_section_m"] = np.round(np.clip(rng.normal(2.5, 0.9, n), 0.3, 9), 2)
bed_m = 280 - 0.004 * df.s_true + 0.0004 * df.n_true ** 2 + rng.normal(0, 0.5, n)
df["bedrock_elev_raw"] = np.round(units.m_to_ft(bed_m)).astype(int)
df["surface_elev_raw"] = np.round(units.m_to_ft(bed_m + df.overburden_m + df.dredge_section_m)).astype(int)
grade_mg = df.accum_mg_m2 / df.dredge_section_m
cents = grade_mg / units.cents_per_yd3_to_mg_per_m3(1.0)
df["grade_cents_yd3"] = np.where((cents > 0) & (cents < 0.1), 0.1, np.round(cents, 1))
cols = ["hole_id", "ground", "surface_elev_raw", "bedrock_elev_raw", "overburden_m",
        "dredge_section_m", "grade_cents_yd3", "x", "y"]
out = ROOT / "data" / "interim" / "synthetic_holes.csv"
df[cols].to_csv(out, index=False)
print(f"wrote {n} holes -> {out.relative_to(ROOT)}")
