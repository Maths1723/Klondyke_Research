"""Effective gold grain mass across the Klondike, one estimate per drilling campaign.

For every source map with >= MIN_HOLES usable drill grades (excluding thaw /
strip / progress layouts, whose numbers are not grades), fit the
grain-counting latent field (Laplace, base model) and record the grain
parameters. Each map is one campaign with one set of conventions, so the fit
does not mix drilling practices.

Coordinates: each map's holes are rotated to their principal axes (u along
the reach, v across); separate length scales per axis capture the valley
anisotropy for reaches that are roughly straight over a map (1-3 km).

Covariates for interpretation:
  * straight-line distance of the map centroid from King Solomon Dome
    (63.86806 N, 138.95278 W; Canadian Geographical Names), the inferred
    lode source of most Klondike placer gold;
  * creek, from the map title;
  * Knight et al. (1994) transport-distance estimates from gold flatness,
    where available for that creek.

Usage: python scripts/10_regional_grain.py [--min-holes 60]
Outputs: reports/regional/grain_by_map.csv, grain_vs_distance.png
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from klondyke import physical, sampling  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "reports" / "regional"
KSD = (63.8680556, -138.9527778)
NOT_GRADE = re.compile(r"thaw|strip|progress|workings|tailing|dredge no\.? ?\d+ (area|workings)", re.I)
CREEKS = ["Eldorado", "Bonanza", "Hunker", "Last Chance", "Gold Bottom", "Dominion", "Sulphur", "Gold Run",
          "Quartz", "Bear", "Allgold", "Indian", "Eureka", "Henderson", "Klondike", "Montana", "Arlington",
          "Granville", "Lovett", "Paradise", "Ophir", "Australia", "Bonanza"]


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def creek_of(title):
    t = str(title)
    for c in CREEKS:
        if re.search(c, t, re.I):
            return c
    return "other/unknown"


def lonlat_to_xy_model(raw_json):
    """Quadratic polynomial lon/lat -> Yukon Albers, fitted on the holes themselves."""
    feats = json.loads(Path(raw_json).read_text())["features"]
    A = np.array([[f["attributes"]["LONGITUDE_DD"], f["attributes"]["LATITUDE_DD"], f["geometry"]["x"], f["geometry"]["y"]]
                  for f in feats if f["attributes"].get("LONGITUDE_DD") is not None])
    lon, lat, x, y = A.T
    D = np.column_stack([np.ones_like(lon), lon, lat, lon ** 2, lat ** 2, lon * lat])
    cx, *_ = np.linalg.lstsq(D, x, rcond=None)
    cy, *_ = np.linalg.lstsq(D, y, rcond=None)
    res = np.hypot(D @ cx - x, D @ cy - y)
    f = lambda lo, la: (np.array([1, lo, la, lo ** 2, la ** 2, lo * la]) @ cx,
                        np.array([1, lo, la, lo ** 2, la ** 2, lo * la]) @ cy)
    return f, float(np.percentile(res, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-holes", type=int, default=60)
    ap.add_argument("--sd-hole", type=float, default=0.0,
                    help="fix a per-hole log-normal error (e.g. 0.7 from twin holes); 0 = grains only")
    a = ap.parse_args()
    suffix = f"_sdhole{a.sd_hole:g}" if a.sd_hole > 0 else ""
    OUT.mkdir(parents=True, exist_ok=True)
    to_xy, err95 = lonlat_to_xy_model(ROOT / "raw" / "ycgc_layer0_drillholes.json")
    kx, ky = to_xy(KSD[1], KSD[0])
    log(f"King Solomon Dome -> ({kx:.0f}, {ky:.0f}) in EPSG:3578 (fit 95th pct residual {err95:.1f} m)")

    h = pd.read_csv(ROOT / "data" / "interim" / "ycgc_holes.csv", low_memory=False)
    h = h[h.usable & ~h.map_title.fillna("").str.contains(NOT_GRADE)]
    counts = h.map_id.value_counts()
    maps = counts[counts >= a.min_holes].index.tolist()
    log(f"{len(maps)} campaigns with >= {a.min_holes} usable grades")
    rows = []
    for m in maps:
        g = h[h.map_id == m]
        P = g[["x", "y"]].to_numpy()
        c0 = P.mean(0)
        U = np.linalg.svd(P - c0, full_matrices=False)[2]
        X = (P - c0) @ U.T
        kind, sm, st, V = sampling.observations_from_grade(g.grade_cents_raw.values, g.dredge_section_ft.values,
                                                           price=g.gold_price.values)
        t0 = time.time()
        try:
            kw = dict(drill_noise=True, fixed={"sd_drill": a.sd_hole}) if a.sd_hole > 0 else {}
            f = physical.CPGField(maxiter=60, init={"ls_s": 150.0, "ls_n": 40.0}, **kw).fit(X, kind, sm, st, V)
            sm_ = f.summary()
        except Exception as e:  # keep going; record failure
            log(f"{m}: fit failed ({e})"); continue
        row = {"map_id": m, "title": g.map_title.iloc[0], "year": g.map_year.iloc[0], "creek": creek_of(g.map_title.iloc[0]),
               "n": len(g), "zero_share": float((g.grade_cents_raw == 0).mean()),
               "median_grade_c": float(g.grade_cents_2067.median()), "median_ds_ft": float(g.dredge_section_ft.median()),
               "price": float(g.gold_price.iloc[0]), "price_assumed": bool(g.flag_price_assumed.any()),
               "x": float(c0[0]), "y": float(c0[1]), "dist_ksd_km": float(np.hypot(c0[0] - kx, c0[1] - ky) / 1000),
               "extent_km": float(np.ptp(X[:, 0]) / 1000), "secs": round(time.time() - t0)}
        row.update({k: sm_[k] for k in ["m_eff", "mbar", "alpha", "sigma", "ls_s", "ls_n", "logZ", "converged"]})
        rows.append(row)
        log(f"{m:12s} {row['creek']:12s} n={row['n']:4d} d={row['dist_ksd_km']:5.1f} km  m_eff={row['m_eff']:7.2f} mg "
            f"mbar={row['mbar']:.2f} alpha={row['alpha']:.2f} ({row['secs']}s)")
        pd.DataFrame(rows).to_csv(OUT / f"grain_by_map{suffix}.csv", index=False)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"grain_by_map{suffix}.csv", index=False)
    plot(res, suffix)


def plot(res, suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    fig, ax = plt.subplots(figsize=(8, 5))
    cr = sorted(res.creek.unique())
    cmap = plt.get_cmap("tab20")
    for i, c in enumerate(cr):
        g = res[res.creek == c]
        ax.scatter(g.dist_ksd_km, g.m_eff, s=np.clip(g.n / 3, 15, 150), color=cmap(i % 20), label=c, alpha=0.85, edgecolor="k", lw=0.4)
    ax.set_yscale("log"); ax.set_xlabel("distance of campaign from King Solomon Dome (km)")
    ax.set_ylabel("effective grain mass m_eff (mg), from assays")
    rho = spearmanr(res.dist_ksd_km, res.m_eff)
    ax.set_title(f"Gold grain mass inferred from 1910s-1960s drill assays (Spearman ρ = {rho.statistic:.2f}, p = {rho.pvalue:.3f})")
    ax.legend(fontsize=7, ncol=2, loc="lower left"); fig.tight_layout()
    fig.savefig(OUT / f"grain_vs_distance{suffix}.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
