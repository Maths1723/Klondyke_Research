"""Figures for the paper (paper/figures/*.pdf), built from the outputs in reports/.

Run after the analysis scripts (run_pipeline.bat):  python paper/make_figures.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from klondyke import sampling, units  # noqa: E402

OUT = ROOT / "paper" / "figures"
MM = 1 / 25.4
W1, W2 = 84 * MM, 174 * MM                      # Springer single / double column
KSD_XY = (183735.0, 1058358.0)                  # King Solomon Dome, EPSG:3578 (scripts/10)
C = {"k": "#222222", "grey": "#9a9a9a", "blue": "#1f5fa8", "orange": "#d9661f", "red": "#b2182b",
     "green": "#2c8c4a", "purple": "#6a3d9a", "light": "#d9d9d9"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.1,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})


def panel(ax, letter):
    ax.text(-0.02, 1.02, f"({letter})", transform=ax.transAxes, fontsize=8, fontweight="bold", ha="right", va="bottom")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def holes():
    return pd.read_csv(ROOT / "data" / "interim" / "ycgc_holes.csv", low_memory=False)


# ---------------------------------------------------------------- Fig. 1
def fig_overview(h):
    u = h[h.usable]
    dom = pd.read_csv(ROOT / "reports" / "dominion" / "holes_curated.csv")
    sul = pd.read_csv(ROOT / "reports" / "sulphur" / "holes_used.csv", low_memory=False)
    pr = pd.read_csv(ROOT / "reports" / "twins" / "pairs.csv")
    xy = h.set_index("uid")[["x", "y"]]
    tw = xy.loc[pr.uid1.values].to_numpy()
    camp = pd.read_csv(ROOT / "reports" / "regional" / "grain_by_map.csv")
    fig, ax = plt.subplots(figsize=(W2, W2 * 0.55))
    k = 1000.0
    ax.scatter(h.x / k, h.y / k, s=0.6, color=C["light"], lw=0, rasterized=True, label="all archive holes (13,358)")
    ax.scatter(u.x / k, u.y / k, s=0.8, color=C["grey"], lw=0, rasterized=True, label="usable grades (6,179)")
    ax.scatter(dom.x / k, dom.y / k, s=1.5, color=C["blue"], lw=0, rasterized=True, label="mid-Dominion set (612)")
    ax.scatter(sul.x / k, sul.y / k, s=1.5, color=C["orange"], lw=0, rasterized=True, label="upper Sulphur set (376)")
    ax.scatter(camp.x / k, camp.y / k, s=np.clip(camp.n / 8, 5, 50), facecolor="none", edgecolor=C["k"], lw=0.5,
               label="campaign centroids, ≥ 60 grades (33)")
    ax.scatter(tw[:, 0] / k, tw[:, 1] / k, s=9, marker="x", color=C["red"], lw=0.8, label="twin pairs (78)")
    ax.scatter([KSD_XY[0] / k], [KSD_XY[1] / k], s=55, marker="*", color=C["k"], zorder=5, label="King Solomon Dome")
    ax.set_xlim(152, 210); ax.set_ylim(1000, 1086)
    ax.set_aspect("equal")
    ax.set_xlabel("easting (km, Yukon Albers, EPSG:3578)")
    ax.set_ylabel("northing (km)")
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    for hd in leg.legend_handles:
        try:
            hd.set_sizes([18])
        except AttributeError:
            pass
    cen = camp.groupby("creek")[["x", "y"]].mean() / k
    offs = {"Sulphur": (-30, 6), "Dominion": (8, 2), "Gold Run": (8, -4), "Hunker": (6, -2), "Bear": (-10, 8),
            "Indian": (8, -6), "Eureka": (8, -4), "Klondike": (-36, 6), "Quartz": (-34, -2), "Henderson": (6, 2)}
    for cr, o in offs.items():
        if cr in cen.index:
            ax.annotate(cr + (" Cr." if cr not in ("Klondike", "Indian") else " R."), cen.loc[cr], xytext=o,
                        textcoords="offset points", fontsize=6.2, style="italic", color="#333333")
    fig.savefig(OUT / "fig1_overview.pdf", dpi=400)
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 2
def fig_price(h):
    ok = h.grade_cents_raw.gt(0.5) & h.x.notna() & ~h.flag_not_drillhole.fillna(False)
    d = h[ok].reset_index(drop=True)
    P = np.array(sorted(cKDTree(d[["x", "y"]].values).query_pairs(25.0)))
    a, b = d.iloc[P[:, 0]].reset_index(drop=True), d.iloc[P[:, 1]].reset_index(drop=True)
    diff = a.map_id.values != b.map_id.values
    g1, g2 = a.grade_cents_raw.values[diff], b.grade_cents_raw.values[diff]
    r = np.maximum(g1, g2) / np.minimum(g1, g2)
    fig, ax = plt.subplots(figsize=(W1, W1 * 0.62))
    bins = np.arange(0.995, 2.4, 0.01)
    ax.hist(r[r < 2.4], bins=bins, color=C["grey"], edgecolor="none")
    for v, lab in [(1.0, "1"), (35 / 20.67, "35/20.67"), (38.5 / 20.67, "38.5/20.67")]:
        ax.axvline(v, color=C["red"], lw=0.7, ls="--")
        ax.text(v + 0.015, ax.get_ylim()[1] * 0.92, lab, fontsize=6, color=C["red"], rotation=90, va="top")
    ax.set_yscale("log")
    ax.set_xlabel("ratio of grades, holes < 25 m apart on different maps")
    ax.set_ylabel("pairs")
    fig.savefig(OUT / "fig2_price_ratios.pdf")
    plt.close(fig)
    return {"pairs": int(diff.sum()), "exact_1": int(np.sum(np.abs(r - 1) < 0.005)),
            "near_1.69": int(np.sum(np.abs(r - 35 / 20.67) < 0.01)), "near_1.86": int(np.sum(np.abs(r - 38.5 / 20.67) < 0.01))}


# ---------------------------------------------------------------- Fig. 3
def fig_synthetic():
    d = pd.concat([pd.read_csv(ROOT / "reports" / f"physical_benchmark_seed{s}.csv").assign(seed=s) for s in (11, 12, 13)])
    d = d.dropna(subset=["rmse"])
    names = {"kriging_loggrade": "kriging\nlog-grade", "gp_loggrade": "GP\nlog-grade", "cpg_field": "grain-counting\nfield"}
    cols = [C["grey"], C["purple"], C["blue"]]
    fig, axs = plt.subplots(1, 3, figsize=(W2, W2 * 0.27))
    for j, (metric, lab) in enumerate([("bias", "bias of log c"), ("rmse", "RMSE of log c"), ("crps", "CRPS of log c")]):
        ax = axs[j]
        for i, sch in enumerate(["LOLO", "block1000m"]):
            g = d[d.scheme == sch].groupby("method")[metric].agg(["mean", "std"]).reindex(list(names))
            x = np.arange(3) + (i - 0.5) * 0.36
            ax.bar(x, g["mean"], width=0.34, yerr=g["std"], color=cols, alpha=1.0 if i == 0 else 0.45,
                   edgecolor="k", lw=0.3, error_kw=dict(lw=0.6, capsize=1.5))
        ax.set_xticks(np.arange(3)); ax.set_xticklabels(list(names.values()), fontsize=6)
        ax.set_title(lab); ax.axhline(0, color="k", lw=0.5)
        panel(ax, "abc"[j])
    axs[0].text(0.98, 0.05, "solid: held-out lines\npale: 1 km held-out blocks", transform=axs[0].transAxes,
                ha="right", va="bottom", fontsize=6)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT / "fig3_synthetic.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 4
def fig_dominion():
    z = np.load(ROOT / "reports" / "dominion" / "target_grid.npz")
    d = pd.read_csv(ROOT / "reports" / "dominion" / "holes_curated.csv")
    S, N, mu, P, ins, um = z["S"], z["N"], z["mu"], z["P"], z["inside"], z["unmined"]
    smax = 6800.0
    cols = S[0] <= smax
    S, N, mu, P, ins, um = S[:, cols], N[:, cols], mu[:, cols], P[:, cols], ins[:, cols], um[:, cols]
    ext = [S.min(), S.max(), N.min(), N.max()]
    c_per = units.cents_per_yd3_to_mg_per_m3(1.0)
    G = np.where(ins, np.log10(np.exp(mu) / c_per), np.nan)
    mined = ins & ~um
    dd = d[d.s <= smax]
    fig, axs = plt.subplots(2, 1, figsize=(W2, W2 * 0.42), sharex=True)
    im = axs[0].imshow(G, origin="lower", extent=ext, aspect="auto", cmap="YlOrBr", vmin=-0.3, vmax=2.1, interpolation="nearest")
    axs[0].scatter(dd.s, dd.n, s=np.clip(dd.c, 0, 150) / 12 + 0.8, color="k", lw=0)
    cb = fig.colorbar(im, ax=axs[0], pad=0.01, fraction=0.03); cb.set_label("median grade,\nlog$_{10}$ c/yd$^3$", fontsize=6.5)
    Pm = np.where(ins, P, np.nan)
    im2 = axs[1].imshow(Pm, origin="lower", extent=ext, aspect="auto", cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    axs[1].imshow(np.where(mined, 1.0, np.nan), origin="lower", extent=ext, aspect="auto", cmap="Greys",
                  vmin=0, vmax=1.5, alpha=0.45, interpolation="nearest")
    axs[1].contour(S, N, mined.astype(float), levels=[0.5], colors="#7fd3ff", linewidths=0.6)
    cb2 = fig.colorbar(im2, ax=axs[1], pad=0.01, fraction=0.03); cb2.set_label("P(grade > 20 c/yd$^3$)", fontsize=6.5)
    for ax, l in zip(axs, "ab"):
        ax.set_ylabel("n (m), + left limit"); panel(ax, l)
        ax.spines["top"].set_visible(True); ax.spines["right"].set_visible(True)
    axs[1].set_xlabel("s, distance down the valley (m)")
    fig.savefig(OUT / "fig4_dominion.pdf", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 5
def fig_twins():
    s11 = load_module("s11", ROOT / "scripts" / "11_twin_holes.py")
    p = pd.read_csv(ROOT / "reports" / "twins" / "pairs.csv")
    pos = p[p.both_pos]
    ff = pd.read_csv(ROOT / "reports" / "twins" / "field_fits.csv")
    rng = np.random.default_rng(7)
    scen = [("grains only, $m_{\\rm eff}$ = 10 mg", dict(mbar=3.85, alpha=0.58, sd_hole=0.0, tau_geo=0.0), C["blue"]),
            ("grains only, $m_{\\rm eff}$ = 90 mg", dict(mbar=15.0, alpha=0.2, sd_hole=0.0, tau_geo=0.0), C["purple"]),
            ("10 mg grains + per-hole $\\sigma_h$ = 0.8", dict(mbar=3.85, alpha=0.58, sd_hole=0.8, tau_geo=0.0), C["green"])]
    fig, axs = plt.subplots(1, 2, figsize=(W2, W2 * 0.42), gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axs[0]
    ax.scatter(pos.gold_mg_pair, np.abs(pos.logratio), s=9, color=C["k"], lw=0, alpha=0.8, label="observed twin pairs")
    bins = np.logspace(np.log10(pos.gold_mg_pair.min()), np.log10(pos.gold_mg_pair.max()), 8)
    for lab, v, col in scen:
        sim = s11.simulate_twins(pos, rng=rng, **v)
        idx = np.digitize(sim.gold, bins)
        xs, med, hi = [], [], []
        for i in range(1, len(bins)):
            q = np.abs(sim.lr[idx == i])
            if len(q) > 50:
                xs.append(np.sqrt(bins[i - 1] * bins[i])); med.append(np.median(q)); hi.append(np.quantile(q, 0.9))
        ax.plot(xs, med, "-", color=col, lw=1.4, label=lab + " (median)")
        ax.plot(xs, hi, "--", color=col, lw=0.8)
    ax.set_xscale("log"); ax.set_xlabel("gold in the drill sample, pair mean (mg)")
    ax.set_ylabel("|log($G_1/G_2$)|, holes 3–12 m apart")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.03), ncol=2, fontsize=6)
    ax.set_ylim(0, 5.2); panel(ax, "a")
    ax = axs[1]
    piv = ff.pivot(index="map_id", columns="model", values="logZ")
    dz = (piv["grain + per-hole"] - piv["grain only"]).sort_values()
    sd = ff[ff.model == "grain + per-hole"].set_index("map_id").sd_drill
    names = {"YCGC26009": "Bear Cr. 1912", "YCGC26341": "Gold Run 1941", "YCGC26438": "lower Hunker",
             "YCGC25041": "dredge area 1933"}
    ax.barh(range(len(dz)), dz.values, color=C["green"], height=0.6)
    ax.set_yticks(range(len(dz)))
    ax.set_yticklabels([f"{names.get(m, m)}\n$\\hat\\sigma_h$ = {sd[m]:.2f}" for m in dz.index], fontsize=6.5)
    ax.set_xlabel("Δ log marginal likelihood\n(grains + per-hole − grains only)")
    panel(ax, "b")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "fig5_twins.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 6
def fig_identifiability():
    r = pd.read_csv(ROOT / "reports" / "identifiability" / "results.csv")
    fig, axs = plt.subplots(1, 2, figsize=(W2, W2 * 0.3))
    combos = [(0.0, "lines"), (0.0, "twins"), (0.8, "lines"), (0.8, "twins")]
    labels = ["$\\sigma_h$=0\nlines", "$\\sigma_h$=0\nlines+twins", "$\\sigma_h$=0.8\nlines", "$\\sigma_h$=0.8\nlines+twins"]
    for j, (metric, lab) in enumerate([("m_eff", "estimated $m_{\\rm eff}$ (mg)"), ("rmse_latent", "RMSE of latent field")]):
        ax = axs[j]
        for k, (model, col, off) in enumerate([("grain only", C["grey"], -0.12), ("grain + per-hole", C["blue"], 0.12)]):
            for i, (sd, des) in enumerate(combos):
                g = r[(r.sd_hole_true == sd) & (r.design == des) & (r.model == model)][metric]
                ax.scatter(np.full(len(g), i + off), g, s=10, color=col, lw=0, label=model if i == 0 else None)
                ax.plot([i + off - 0.08, i + off + 0.08], [g.mean()] * 2, color=col, lw=1.4)
        if metric == "m_eff":
            ax.axhline(10.49, color=C["red"], lw=0.7, ls="--"); ax.text(3.45, 10.7, "true", color=C["red"], fontsize=6, ha="right")
        ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=6.3); ax.set_ylabel(lab); panel(ax, "ab"[j])
    axs[0].legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "fig6_identifiability.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 7
def fig_profiles():
    s = pd.read_csv(ROOT / "reports" / "regional" / "grain_profiles_summary.csv")
    fig, axs = plt.subplots(1, 2, figsize=(W2, W2 * 0.33))
    ax = axs[0]
    jit = np.random.default_rng(1).uniform(-0.6, 0.6, len(s))
    for _, r_ in s.iterrows():
        pass
    x = s.median_ds_ft + jit
    col = np.where(s.closed, C["blue"], C["red"])
    ax.vlines(x, s.mbar_lo, s.mbar_hi, color=col, lw=0.9)
    ax.scatter(x, s.mbar_best, s=np.clip(s.n / 5, 6, 60), color=col, zorder=3, lw=0)
    dom = s.creek == "Dominion"
    ax.scatter(x[dom], s.mbar_best[dom], s=50, facecolor="none", edgecolor=C["k"], lw=0.6, zorder=4, label="Dominion campaigns")
    ax.scatter([], [], s=12, color=C["blue"], label="closed 95% interval")
    ax.scatter([], [], s=12, color=C["red"], label="open interval")
    ax.set_yscale("log"); ax.set_xlabel("median dredge section of the campaign (ft)")
    ax.set_ylabel("mean grain mass $\\bar m$ (mg)")
    ax.legend(frameon=False, loc="upper right"); panel(ax, "a")
    ax = axs[1]
    ax.scatter(s.I_zero.clip(0.05), s.range_logZ, s=np.clip(s.n / 5, 6, 60), color=col, lw=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("zero-channel information on log $\\bar m$ (known-$c$ bound)")
    ax.set_ylabel("depth of profile (Δ log Z over grid)")
    panel(ax, "b")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "fig7_profiles.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 8
def fig_sulphur():
    b = pd.read_csv(ROOT / "reports" / "sulphur" / "dredge_blocks.csv")
    sc = pd.read_csv(ROOT / "reports" / "error_bound" / "meff_scan.csv")
    fig, axs = plt.subplots(1, 2, figsize=(W2, W2 * 0.32), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axs[0]
    yr = b.YEAR.astype(int)
    ax.plot(yr, b.act_c, "-o", color=C["k"], ms=3, label="recovered by the dredge")
    ax.plot(yr, b.est_c, "--s", color=C["grey"], ms=3, label="YCGC pre-mining estimate")
    ax.plot(yr, b.holes_mean, ":^", color=C["purple"], ms=3, label="mean of holes in block")
    ax.plot(yr, b.cpg_lbo, "-o", color=C["orange"], ms=3, label="grain-counting field, block held out")
    ax.plot(yr, b.kriging_lbo, "-", color=C["blue"], ms=3, marker="d", label="kriging, block held out")
    ax.set_ylabel("grade (c/yd$^3$ at \\$20.67)"); ax.set_xlabel("dredge No. 9 season")
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=6); ax.set_ylim(0, 140); panel(ax, "a")
    ax = axs[1]
    ax.plot(sc.m_eff_mg, sc.pred_robust_sd, "-o", color=C["orange"], ms=3, label="predicted (model + grains)")
    ax.axhspan(sc.obs_lo90.iloc[0], sc.obs_hi90.iloc[0], color=C["light"], label="observed, 90% bootstrap")
    ax.axhline(sc.obs_robust_sd.iloc[0], color=C["k"], lw=1.0, label="observed")
    ax.set_xscale("log"); ax.set_xlabel("effective grain mass $m_{\\rm eff}$ (mg)")
    ax.set_ylabel("robust sd, log(holes mean / recovered)")
    ax.legend(frameon=False, loc="upper left"); panel(ax, "b")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "fig8_sulphur.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 9
def fig_design():
    s15 = load_module("s15", ROOT / "scripts" / "15_design.py")
    gold = np.logspace(0, 3.3, 300)
    fig, ax = plt.subplots(figsize=(W1, W1 * 0.75))
    for m_eff, ls in [(10.0, "-"), (50.0, "--")]:
        for sd_h, col in [(0.6, C["blue"]), (0.9, C["red"])]:
            ax.plot(gold, s15.kappa2(sd_h, m_eff, gold) / 0.04, ls, color=col,
                    label=f"$m_{{\\rm eff}}$ {m_eff:.0f} mg, $\\sigma_h$ {sd_h}")
        ax.axvline(m_eff / np.expm1(0.7 ** 2), color=C["grey"], ls=ls, lw=0.6)
    tab = pd.read_csv(ROOT / "reports" / "design" / "holes_needed.csv")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("gold in one drill sample, $cV$ (mg)")
    ax.set_ylabel("holes per block for ±20% (1 sd)")
    ax.legend(frameon=False, fontsize=6)
    fig.savefig(OUT / "fig9_design.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = holes()
    fig_overview(h)
    print("price ratio pairs:", fig_price(h))
    fig_synthetic(); fig_dominion(); fig_twins(); fig_identifiability(); fig_profiles(); fig_sulphur(); fig_design()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
