"""Step 4: predicted accumulation, uncertainty and ranked targets.

Fits the chosen model on ALL holes, predicts on a grid in valley coordinates
covering the drilled floor plus a margin, converts to the probability that
accumulation exceeds a threshold, and groups contiguous high-probability
cells into ranked targets. Cells inside dredge limits are reported
separately (re-mining targets) from those outside (virgin-ground targets).

Usage:
  python scripts/04_target_map.py --name dominion --model kriging_sn_auto \
         --threshold-cents 50 --thickness-m 2.0

The threshold is expressed as a grade (cents/yd3 at $20.67) over a thickness,
converted to accumulation. What threshold is economic today is an open
question for operators (see OPEN_QUESTIONS.md); 50 c/yd3 over 2 m is a
placeholder, not a recommendation.

Outputs: reports/targets_<name>.csv, reports/figures/map_<name>_*.png,
         data/processed/<name>_grid.csv (+ .gpkg if geopandas is installed)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from klondyke import experiment, units, valley  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="dominion")
    ap.add_argument("--model", default="kriging_sn_auto")
    ap.add_argument("--threshold-cents", type=float, default=50.0)
    ap.add_argument("--thickness-m", type=float, default=2.0)
    ap.add_argument("--cell", type=float, default=10.0, help="grid cell (m) in s and n")
    ap.add_argument("--margin", type=float, default=100.0, help="metres beyond outermost holes")
    ap.add_argument("--pmin", type=float, default=0.5)
    ap.add_argument("--point", action="store_true",
                    help="use single-sample uncertainty (with nugget) instead of block-level")
    a = ap.parse_args()

    df = pd.read_csv(ROOT / "data" / "processed" / f"{a.name}_holes.csv")
    axis = pd.read_csv(ROOT / "data" / "processed" / f"{a.name}_axis.csv")
    poly, offset = axis[["x", "y"]].to_numpy(), float(axis.s_offset.iloc[0])

    table = {m: (f, fac) for m, f, fac in experiment.method_table()}
    feat, fac = table[a.model]
    if feat != "sn":
        raise SystemExit("mapping needs a model on (s, n) only: kriging_sn_auto, gp_sn or idw_sn_aniso5")
    model = fac().fit(experiment.features(df, "sn"), df[experiment.TARGET].values)

    # grid over the drilled corridor: per s-bin, n within the observed extent + margin
    sg = np.arange(0, df.s.max() + a.cell, a.cell)
    ng = np.arange(df.n.min() - a.margin, df.n.max() + a.margin + a.cell, a.cell)
    S, N = np.meshgrid(sg, ng)
    lo = np.interp(sg, *_rolling_extent(df, "min")) - a.margin
    hi = np.interp(sg, *_rolling_extent(df, "max")) + a.margin
    inside = (N >= lo[None, :]) & (N <= hi[None, :])
    mu = np.full(S.shape, np.nan); sd = np.full(S.shape, np.nan)
    Q = np.column_stack([S[inside], N[inside]])
    if not a.point and hasattr(model, "predict_latent"):
        m, s = model.predict_latent(Q)   # block-level: nugget averages out over a mining block
    else:
        m, s = model.predict(Q)
    mu[inside] = m
    sd[inside] = s if s is not None else np.nan

    thr_accum = float(units.accumulation_mg_per_m2(units.cents_per_yd3_to_mg_per_m3(a.threshold_cents), a.thickness_m))
    thr = np.log1p(thr_accum)
    prob = 1 - norm.cdf((thr - mu) / sd)
    X, Y = valley.sn_to_xy(S.ravel(), N.ravel(), poly, s_offset=offset)

    # distance to the nearest hole: predictions far from data are flagged
    from scipy.spatial import cKDTree
    d_near, _ = cKDTree(df[["s", "n"]].to_numpy()).query(np.column_stack([S.ravel(), N.ravel()]))

    grid = pd.DataFrame({"s": S.ravel(), "n": N.ravel(), "x": X, "y": Y,
                         "mu_log": mu.ravel(), "sd_log": sd.ravel(), "p_exceed": prob.ravel(),
                         "dist_nearest_hole_m": d_near})
    grid["accum_median_mg_m2"] = np.expm1(grid.mu_log)
    grid = grid[inside.ravel()]

    in_dredge = None
    if "in_dredge_limit" in df and _has_geopandas():
        in_dredge = _flag_dredge(grid, a.name)
        grid["in_dredge_limit"] = in_dredge

    # ranked targets: connected high-probability cells
    P = np.where(inside, prob, 0.0)
    lab, nlab = ndimage.label(P >= a.pmin)
    rows = []
    cell_area = a.cell ** 2
    for k in range(1, nlab + 1):
        mk = lab == k
        if mk.sum() < 3:
            continue
        idx = np.where(mk.ravel())[0]
        rows.append({
            "target": k, "n_cells": int(mk.sum()), "area_m2": float(mk.sum() * cell_area),
            "s_from": float(S[mk].min()), "s_to": float(S[mk].max()),
            "n_mean": float(N[mk].mean()), "limit": "left" if N[mk].mean() > 0 else "right",
            "p_mean": float(P[mk].mean()),
            # median-based gold estimate in grams (conservative: exp of log-mean)
            "gold_g_median": float(np.expm1(mu[mk]).sum() * cell_area / 1000.0),
            "max_dist_to_hole_m": float(d_near[idx].max()),
            "x_centre": float(X[idx].mean()), "y_centre": float(Y[idx].mean()),
        })
    tg = pd.DataFrame(rows)
    if len(tg):
        tg = tg.sort_values("gold_g_median", ascending=False).reset_index(drop=True)
        tg.index += 1
    rep = ROOT / "reports"
    tg.to_csv(rep / f"targets_{a.name}.csv")
    grid.to_csv(ROOT / "data" / "processed" / f"{a.name}_grid.csv", index=False)
    _export_gpkg(grid, a.name)
    _plot(df, S, N, mu, sd, prob, inside, tg, a, thr_accum)
    print(f"model={a.model}  threshold={a.threshold_cents} c/yd3 over {a.thickness_m} m = {thr_accum:.0f} mg/m2")
    print(tg.head(15).round(1).to_string() if len(tg) else "no targets above p_min")


def _rolling_extent(df, how):
    g = df.groupby("line_id").agg(s=("s", "median"), v=("n", how)).sort_values("s")
    return g.s.values, g.v.values


def _has_geopandas():
    try:
        import geopandas  # noqa: F401
        return True
    except ImportError:
        return False


def _flag_dredge(grid, name):
    import json
    import geopandas as gpd
    from klondyke import ingest
    cfg = json.loads((ROOT / "config" / "project.json").read_text())
    dredge = ingest.load_polygons(ROOT / "raw", cfg["dredge_layer"], cfg["target_crs"])
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(grid.x, grid.y), crs=cfg["target_crs"])
    return pts.within(dredge.union_all()).values


def _export_gpkg(grid, name):
    if not _has_geopandas():
        return
    import json
    import geopandas as gpd
    crs = "EPSG:3578"
    p = ROOT / "config" / "project.json"
    if p.exists():
        crs = json.loads(p.read_text()).get("target_crs", crs)
    g = gpd.GeoDataFrame(grid, geometry=gpd.points_from_xy(grid.x, grid.y), crs=crs)
    g.to_file(ROOT / "data" / "processed" / f"{name}_grid.gpkg", driver="GPKG")


def _plot(df, S, N, mu, sd, prob, inside, tg, a, thr_accum):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    panels = [(np.where(inside, mu, np.nan), "YlOrBr", "predicted log1p(accumulation, mg/m²)"),
              (np.where(inside, sd, np.nan), "viridis", "predictive sd (log scale)"),
              (np.where(inside, prob, np.nan), "magma", f"P(accumulation > {thr_accum:.0f} mg/m²)")]
    for ax, (Z, cm, t) in zip(axs, panels):
        im = ax.pcolormesh(S, N, Z, shading="auto", cmap=cm)
        ax.scatter(df.s, df.n, s=3, c="k", alpha=0.5)
        ax.set_title(t); ax.set_ylabel("n (m, + = left limit)")
        fig.colorbar(im, ax=ax, shrink=0.8)
    for _, r in tg.head(10).iterrows():
        axs[2].annotate(str(r.name), ((r.s_from + r.s_to) / 2, r.n_mean), color="cyan",
                        fontsize=9, ha="center", weight="bold")
    axs[-1].set_xlabel("s, distance down valley (m)")
    fig.suptitle(f"{a.name}: {a.model}")
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "figures" / f"map_{a.name}_{a.model}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
