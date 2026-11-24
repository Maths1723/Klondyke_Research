"""Profile likelihood of the mean grain mass, per drilling campaign.

For each campaign (same set as 10_regional_grain.py), fix the number-mean
grain mass m̄ on a grid and refit everything else (field, grain shape) by
Laplace. The width of the profile says whether that campaign's assays carry
information on grain size at all, and how that depends on the expected
number of grains per sample.

Each campaign's profile is set against the information its holes can carry
about grain size even if the concentration at every hole were known
(sampling.grain_information, reference m̄ = 3 mg, α = 0.6, non-grain log
variance v0 = 0.5).

Usage: python scripts/12_grain_profiles.py [--summarise-only]
Outputs: reports/regional/grain_profiles.csv, grain_profiles_summary.csv, grain_profiles.png
"""
from __future__ import annotations

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
GRID = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
NOT_GRADE = re.compile(r"thaw|strip|progress|workings|tailing|dredge no\.? ?\d+ (area|workings)", re.I)


def load_holes():
    h = pd.read_csv(ROOT / "data" / "interim" / "ycgc_holes.csv", low_memory=False)
    return h[h.usable & ~h.map_title.fillna("").str.contains(NOT_GRADE)]


def information(h, maps, mbar=3.0, alpha=0.6, v0=0.5):
    from klondyke import units
    rows = []
    for m in maps:
        g = h[h.map_id == m]
        c = units.cents_per_yd3_to_mg_per_m3(g.grade_cents_2067.values)
        pos = c[c > 0]
        c = np.where(c > 0, c, np.percentile(pos, 10) if len(pos) else 1.0)   # zeros: low plug-in level
        V = sampling.sample_volume_m3(g.dredge_section_ft.values)
        Iz, Id = sampling.grain_information(c, V, mbar, alpha, v0)
        rows.append({"map_id": m, "median_gold_mg": float(np.median(c * V)),
                     "median_grains_ref": float(np.median(c * V / mbar)),
                     "I_zero": float(Iz.sum()), "I_disp": float(Id.sum()),
                     "sd_log_mbar_bound": float(1 / np.sqrt(Iz.sum() + 1e-12)),
                     "sd_log_meff_bound": float(1 / np.sqrt(Id.sum() + 1e-12))})
    return pd.DataFrame(rows)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarise-only", action="store_true")
    a = ap.parse_args()
    res = pd.read_csv(OUT / "grain_by_map.csv")
    h = load_holes()
    if a.summarise_only:
        summarise(pd.read_csv(OUT / "grain_profiles.csv"), res, h)
        return
    rows = []
    for m in res.map_id:
        g = h[h.map_id == m]
        P = g[["x", "y"]].to_numpy(); c0 = P.mean(0)
        U = np.linalg.svd(P - c0, full_matrices=False)[2]; X = (P - c0) @ U.T
        kind, s, st, V = sampling.observations_from_grade(g.grade_cents_raw.values, g.dredge_section_ft.values,
                                                          price=g.gold_price.values)
        for mb in GRID:
            t0 = time.time()
            try:
                f = physical.CPGField(maxiter=50, fixed={"mbar": float(mb)},
                                      init={"ls_s": 150.0, "ls_n": 40.0}).fit(X, kind, s, st, V)
                r = f.summary()
                rows.append({"map_id": m, "mbar_fixed": mb, "logZ": r["logZ"], "alpha": r["alpha"],
                             "m_eff": r["m_eff"], "sigma": r["sigma"]})
            except Exception as e:
                rows.append({"map_id": m, "mbar_fixed": mb, "logZ": np.nan})
            print(time.strftime("%H:%M:%S"), m, mb, round(rows[-1]["logZ"], 1), f"{time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "grain_profiles.csv", index=False)
    summarise(pd.DataFrame(rows), res, h)


def summarise(pr, res, h):
    summ = []
    for m, g in pr.groupby("map_id"):
        g = g.dropna(subset=["logZ"]).sort_values("mbar_fixed")
        best = g.logZ.max(); ok = g[g.logZ >= best - 1.92]       # ~95% profile interval on the grid
        summ.append({"map_id": m, "mbar_best": float(g.loc[g.logZ.idxmax(), "mbar_fixed"]),
                     "mbar_lo": float(ok.mbar_fixed.min()), "mbar_hi": float(ok.mbar_fixed.max()),
                     "open_low": bool(ok.mbar_fixed.min() == GRID.min()), "open_high": bool(ok.mbar_fixed.max() == GRID.max()),
                     "range_logZ": float(best - g.logZ.min())})
    s = pd.DataFrame(summ).merge(res[["map_id", "creek", "n", "zero_share", "median_grade_c", "median_ds_ft", "dist_ksd_km"]], on="map_id")
    s = s.merge(information(h, s.map_id), on="map_id")
    s["closed"] = ~s.open_low & ~s.open_high
    s.to_csv(OUT / "grain_profiles_summary.csv", index=False)
    print(s.round(3).to_string())
    from scipy.stats import mannwhitneyu, spearmanr
    print("closed intervals:", int(s.closed.sum()), "of", len(s))
    print("profile depth vs zero-channel information: Spearman",
          spearmanr(s.range_logZ, s.I_zero).statistic.round(2), spearmanr(s.range_logZ, s.I_zero).pvalue.round(4))
    print("profile depth vs median section:", spearmanr(s.range_logZ, s.median_ds_ft).statistic.round(2))
    if s.closed.any() and (~s.closed).any():
        print("I_zero closed vs open: median", s[s.closed].I_zero.median().round(2), s[~s.closed].I_zero.median().round(2),
              "MWU p", mannwhitneyu(s[s.closed].I_zero, s[~s.closed].I_zero).pvalue.round(4))
    plot(pr, s)


def plot(pr, s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    a = ax[0]
    cm = plt.get_cmap("viridis")
    lo, hi = np.log10(s.I_zero.clip(0.1)).min(), np.log10(s.I_zero.clip(0.1)).max()
    for m, g in pr.dropna(subset=["logZ"]).groupby("map_id"):
        r = s[s.map_id == m].iloc[0]
        col = cm((np.log10(max(r.I_zero, 0.1)) - lo) / (hi - lo + 1e-9))
        a.plot(g.mbar_fixed, g.logZ - g.logZ.max(), "-o", ms=3, color=col, alpha=0.8)
    a.axhline(-1.92, color="k", ls="--", lw=0.8)
    a.set_xscale("log"); a.set_ylim(-40, 2)
    a.set_xlabel("mean grain mass m̄ fixed at (mg)"); a.set_ylabel("profile log marginal likelihood (rel. to max)")
    a.set_title("Grain-mass profiles, 33 campaigns (colour: zero-channel information)")
    b = ax[1]
    b.scatter(s.I_zero.clip(0.05), s.range_logZ.clip(0.05), c=np.where(s.closed, "tab:green", "tab:red"), s=np.clip(s.n / 4, 12, 90))
    for _, r in s.iterrows():
        b.annotate(r.creek[:8], (max(r.I_zero, 0.05), max(r.range_logZ, 0.05)), fontsize=6, alpha=0.7)
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlabel("information on log m̄ from zero assays (known-c bound)")
    b.set_ylabel("depth of profile (Δ log Z over grid)")
    b.set_title("Green: closed 95% interval; red: open")
    fig.tight_layout(); fig.savefig(OUT / "grain_profiles.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
