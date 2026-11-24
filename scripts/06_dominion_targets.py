"""Mid-Dominion: block-level gold map, mined-out mask, ranked targets.

Uses the model selected in 05_dominion_real.py (grain-counting latent field).
Grade predictions are the in-situ concentration c over the dredge section,
i.e. what a mining block sees, with uncertainty from the posterior of the
field only (grain-counting noise of a single drill sample is excluded).

Mined ground comes from the YGS 'Dredge limits and cuts' layer: dredge cuts
(D, 1947-1966), thaw blocks (T, thawed ahead of dredging) and the 1915 open
cut / ground sluice (O). Anything mined after the YCGC era is NOT in this
layer: modern cuts must be checked on imagery before any target is acted on.

Usage: python scripts/06_dominion_targets.py [--threshold-cents 20] [--pmin 0.5]
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from klondyke import baselines as B  # noqa: E402
from klondyke import units, valley  # noqa: E402

OUT = ROOT / "reports" / "dominion"
X0, Y0 = 197000.0, 1040000.0


def load_polys():
    rows = []
    for line in (ROOT / "data" / "interim" / "dominion_mid_dredge_polys.txt").read_text().splitlines():
        head, geom = line.split(":", 1)
        t, yr = head[0], head[1:]
        rings = [np.array([[float(v) for v in p.split()] for p in r.split(";")]) + [X0, Y0] for r in geom.split("|")]
        rows.append({"type": t, "year": int(yr) if yr else None, "rings": rings})
    return pd.DataFrame(rows)


def mined_mask(x, y, polys):
    from matplotlib.path import Path as MPath
    P = np.column_stack([x, y])
    kind = np.full(len(P), "", dtype=object)
    year = np.full(len(P), np.nan)
    for _, r in polys.iterrows():
        for ring in r.rings:
            inside = MPath(ring).contains_points(P)
            new = inside & (kind == "")
            kind[new] = r.type
            year[new] = r.year if r.year else np.nan
    return kind, year


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold-cents", type=float, default=20.0,
                    help="grade threshold, cents/yd3 at $20.67 (placeholder; see OPEN_QUESTIONS #10)")
    ap.add_argument("--pmin", type=float, default=0.5)
    ap.add_argument("--cell", type=float, default=10.0)
    ap.add_argument("--strip", action="store_true")
    ap.add_argument("--sd-hole", type=float, default=0.0,
                    help="refit with a fixed per-hole error (e.g. 0.7) instead of the saved best model")
    a = ap.parse_args()
    sfx = f"_sdhole{a.sd_hole:g}" if a.sd_hole > 0 else ""

    d = pd.read_csv(OUT / "holes_curated.csv", dtype={"mapid": str, "hid": str})
    bm = pickle.load(open(OUT / "best_model.pkl", "rb"))
    model, poly, off = bm["model"], bm["poly"], bm["off"]
    if a.sd_hole > 0:
        from klondyke import physical, sampling
        kind_, sm_, st_, V_ = sampling.observations_from_grade(d.c.values, d.ds.values)
        model = physical.CPGField(maxiter=80, drill_noise=True, fixed={"sd_drill": a.sd_hole}).fit(
            d[["s", "n"]].to_numpy(), kind_, sm_, st_, V_)
    polys = load_polys()

    # grid over the drilled floor: per down-valley bin, n within the line extents +- 60 m
    g = d.groupby("line").agg(s=("s", "median"), nmin=("n", "min"), nmax=("n", "max")).sort_values("s")
    sg = np.arange(0, d.s.max() + a.cell, a.cell)
    ng = np.arange(d.n.min() - 80, d.n.max() + 80 + a.cell, a.cell)
    S, N = np.meshgrid(sg, ng)
    lo = np.interp(sg, g.s, g.nmin) - 60
    hi = np.interp(sg, g.s, g.nmax) + 60
    inside = (N >= lo[None, :]) & (N <= hi[None, :])
    q = np.column_stack([S[inside], N[inside]])

    mu = np.full(S.shape, np.nan); sd = np.full(S.shape, np.nan)
    m_, s_ = model.predict_latent(q)
    mu[inside], sd[inside] = m_, s_
    thr = np.log(units.cents_per_yd3_to_mg_per_m3(a.threshold_cents))
    P = 1 - norm.cdf((thr - mu) / sd)

    X, Y = valley.sn_to_xy(S.ravel(), N.ravel(), poly, s_offset=off)
    X = X.reshape(S.shape); Y = Y.reshape(S.shape)
    kind = np.full(S.shape, "", dtype=object); yr = np.full(S.shape, np.nan)
    k_, y_ = mined_mask(X[inside], Y[inside], polys)
    kind[inside], yr[inside] = k_, y_
    dist, _ = cKDTree(d[["s", "n"]].to_numpy()).query(np.column_stack([S.ravel(), N.ravel()]))
    dist = dist.reshape(S.shape)

    # overburden (muck) thickness, ft: ordinary kriging in valley coordinates (feasibility context)
    okb = B.OrdinaryKriging(anisotropy="auto").fit(d[["s", "n"]].to_numpy(), d.ob.values.astype(float))
    ob = np.full(S.shape, np.nan); ob[inside] = okb.predict(q)[0]

    # holes that fall inside mined ground (sanity: the rich holes should mostly be mined)
    hk, _ = mined_mask(d.x.values, d.y.values, polys)
    d["mined"] = hk
    c_mg = units.cents_per_yd3_to_mg_per_m3(d.c.values)
    gold_share_mined = float((c_mg * d.ds.values)[d.mined != ""].sum() / (c_mg * d.ds.values).sum())

    unmined = inside & (kind == "")
    lab, nl = ndimage.label(unmined & (P >= a.pmin))
    rows = []
    for k in range(1, nl + 1):
        mk = lab == k
        if mk.sum() < 5:                                 # < 500 m2
            continue
        rows.append({
            "area_m2": float(mk.sum() * a.cell ** 2),
            "s_from_m": float(S[mk].min()), "s_to_m": float(S[mk].max()),
            "limit": "left" if N[mk].mean() > 0 else "right", "n_mean_m": float(N[mk].mean()),
            "p_mean": float(P[mk].mean()),
            "median_grade_cents_yd3": float(np.median(np.exp(mu[mk])) / units.cents_per_yd3_to_mg_per_m3(1.0)),
            "median_grade_g_m3": float(np.median(np.exp(mu[mk])) / 1000.0),
            "overburden_ft": float(np.nanmedian(ob[mk])),
            "max_dist_to_hole_m": float(dist[mk].max()),
            "x_centre": float(X[mk].mean()), "y_centre": float(Y[mk].mean()),
        })
    tg = pd.DataFrame(rows)
    if len(tg):
        tg = tg.sort_values(["p_mean", "area_m2"], ascending=False).reset_index(drop=True)
        tg.index += 1
        tg.index.name = "target"
    tg.to_csv(OUT / f"targets{sfx}.csv")

    frac_mined = float((kind[inside] != "").mean())
    summary = {
        "grid_cells": int(inside.sum()), "cell_m": a.cell,
        "share_of_drilled_floor_mined": frac_mined,
        "share_of_drill_gold_inside_mined_ground": gold_share_mined,
        "threshold_cents_yd3": a.threshold_cents, "p_min": a.pmin,
        "share_of_floor_above_threshold_p>pmin": float((P[inside] >= a.pmin).mean()),
        "share_of_unmined_floor_above_threshold": float((P[unmined] >= a.pmin).mean()) if unmined.any() else None,
        "n_targets": int(len(tg)),
    }
    pd.Series(summary).to_json(OUT / f"targets_summary{sfx}.json", indent=2)
    np.savez_compressed(OUT / f"target_grid{sfx}.npz", P=P, mu=mu, sd=sd, inside=inside, unmined=unmined, S=S, N=N)
    print(pd.Series(summary).to_string())
    print(tg.round(2).head(15).to_string() if len(tg) else "no unmined targets above p_min")
    if not sfx:
        plot(d, polys, X, Y, mu, sd, P, inside, kind, tg, a)


def plot(d, polys, X, Y, mu, sd, P, inside, kind, tg, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MPoly
    fig, axs = plt.subplots(1, 3, figsize=(18, 9), sharex=True, sharey=True)
    cents = np.exp(mu) / units.cents_per_yd3_to_mg_per_m3(1.0)
    panels = [(np.where(inside, np.log10(cents), np.nan), "YlOrBr", "Predicted in-situ grade\nlog10(cents/yd³ at $20.67)"),
              (np.where(inside, sd, np.nan), "viridis", "Uncertainty (sd of ln c)"),
              (np.where(inside, P, np.nan), "magma", f"P(grade > {a.threshold_cents:g} c/yd³)\nwith YCGC-era mined ground")]
    for ax, (Z, cm, t) in zip(axs, panels):
        im = ax.pcolormesh(X, Y, Z, shading="auto", cmap=cm)
        ax.scatter(d.x, d.y, s=3, c="k", alpha=0.6)
        ax.set_title(t); ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, shrink=0.5)
    colors = {"D": "#1f77b4", "T": "#2ca02c", "O": "#7f7f7f"}
    for _, r in polys.iterrows():
        for ring in r.rings:
            axs[2].add_patch(MPoly(ring, closed=True, fill=False, ec=colors.get(r.type, "k"), lw=0.5, alpha=0.8))
    for i, r in tg.head(12).iterrows():
        axs[2].annotate(str(i), (r.x_centre, r.y_centre), color="cyan", fontsize=10, weight="bold", ha="center")
    axs[2].plot([], [], color=colors["D"], label="dredge cut 1947-66"); axs[2].plot([], [], color=colors["T"], label="thaw block")
    axs[2].plot([], [], color=colors["O"], label="1915 open cut"); axs[2].legend(loc="lower left", fontsize=8)
    for ax in axs:
        ax.set_xlabel("Easting, Yukon Albers (m)")
    axs[0].set_ylabel("Northing (m)")
    fig.suptitle("Mid-Dominion Creek: 1936 YCGC drilling (612 holes), grain-counting latent field", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "map_targets.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__" and "--strip" not in sys.argv:
    main()


def plot_strip():
    """Unrolled valley view: s (down-valley) x n (across), mined ground outlined."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MPoly
    d = pd.read_csv(OUT / "holes_curated.csv", dtype={"mapid": str, "hid": str})
    bm = pickle.load(open(OUT / "best_model.pkl", "rb"))
    model, poly, off = bm["model"], bm["poly"], bm["off"]
    tg = pd.read_csv(OUT / "targets.csv", index_col=0)
    polys = load_polys()
    sg = np.arange(0, d.s.max() + 10, 10.0); ng = np.arange(-420, 430, 10.0)
    S, N = np.meshgrid(sg, ng)
    g = d.groupby("line").agg(s=("s", "median"), nmin=("n", "min"), nmax=("n", "max")).sort_values("s")
    inside = (N >= np.interp(sg, g.s, g.nmin)[None, :] - 60) & (N <= np.interp(sg, g.s, g.nmax)[None, :] + 60)
    mu, sd = model.predict_latent(np.column_stack([S.ravel(), N.ravel()]))
    cents = (np.exp(mu) / units.cents_per_yd3_to_mg_per_m3(1.0)).reshape(S.shape)
    P = (1 - norm.cdf((np.log(units.cents_per_yd3_to_mg_per_m3(20.0)) - mu) / sd)).reshape(S.shape)
    halves = [(0, 6300), (6200, d.s.max() + 10)]
    fig, axs = plt.subplots(4, 1, figsize=(16, 13))
    for row, (s0, s1) in enumerate(halves):
        for col, (Z, cm, title, vmin, vmax) in enumerate([
                (np.log10(np.maximum(cents, 0.1)), "YlOrBr", "predicted in-situ grade, log10 c/yd³ ($20.67)", -0.5, 2.3),
                (P, "magma", "P(grade > 20 c/yd³); YCGC mined ground outlined; numbers = ranked unmined targets", 0, 1)]):
            ax = axs[row * 2 + col]
            sel = (sg >= s0) & (sg <= s1)
            im = ax.pcolormesh(S[:, sel], N[:, sel], np.where(inside, Z, np.nan)[:, sel], shading="auto",
                               cmap=cm, vmin=vmin, vmax=vmax)
            dd = d[(d.s >= s0) & (d.s <= s1)]
            ax.scatter(dd.s, dd.n, s=np.clip(np.log1p(dd.c) * 6, 2, 40), c="k", alpha=0.55, lw=0)
            for _, r in polys.iterrows():
                for ring in r.rings:
                    rs, rn = valley.project_to_centreline(ring[:, 0], ring[:, 1], poly)
                    rs = rs - off
                    if rs.max() < s0 or rs.min() > s1 or np.abs(rn).max() > 600:
                        continue
                    ax.add_patch(MPoly(np.column_stack([rs, rn]), closed=True, fill=False,
                                       ec={"D": "#5ab4ff", "T": "#7CFC00", "O": "#bbbbbb"}[r.type], lw=0.6))
            if col == 1:
                for i, r in tg.head(15).iterrows():
                    sm = (r.s_from_m + r.s_to_m) / 2
                    if s0 <= sm <= s1:
                        ax.annotate(str(i), (sm, r.n_mean_m), color="cyan", fontsize=10, weight="bold", ha="center")
            ax.set_xlim(s0, s1); ax.set_ylim(-420, 430); ax.set_ylabel("n (m)\n+ left limit")
            ax.set_title(title, fontsize=10, loc="left")
            fig.colorbar(im, ax=ax, pad=0.01, fraction=0.02)
    axs[-1].set_xlabel("s: distance down Dominion Creek from claim ~77 below Lower Discovery (m)")
    fig.suptitle("Mid-Dominion Creek, unrolled along the valley. Dots: 1936 YCGC drill holes (size ~ grade)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "strip_targets.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__" and "--strip" in sys.argv:
    plot_strip()
