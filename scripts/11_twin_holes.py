"""Twin holes: what makes two drill holes a few metres apart disagree?

YCGC sometimes drilled holes 3-12 m apart on the same map (close-spaced test
lines, re-drills). At that spacing the gravel is nearly the same, so the
disagreement between the two grades measures the nugget directly, which
line-spaced drilling (60 m x 150 m) cannot do.

Three candidate sources of the nugget, with different signatures:
  * grain counting: relative variance m_eff / (c V), shrinks as the expected
    gold in the sample grows, and produces zeros at low counts;
  * per-hole multiplicative error (core recovery, volume factor, bedrock
    penetration, metre-scale geology): constant variance on the log scale;
  * spatial geology: small at 3-12 m if the pay streak is tens of metres wide.

The script
  1. extracts twin pairs (same map, < 12 m, distinct hole ids, not double records);
  2. compares the observed spread of log(G1/G2) with simulated twins under
     grain counting alone and under grain counting + per-hole error;
  3. fits the full latent-field model with and without a per-hole error term on
     every map with >= 9 twin pairs and compares marginal likelihoods.

Usage: python scripts/11_twin_holes.py
Outputs: reports/twins/pairs.csv, field_fits.csv, summary.json, twins.png
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from klondyke import physical, sampling, units  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "reports" / "twins"
RADIUS = 12.0
NOT_GRADE = "thaw|strip|progress|workings"


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def twin_pairs(h: pd.DataFrame, radius=RADIUS) -> pd.DataFrame:
    ok = (h.grade_cents_raw.notna() & h.x.notna() & h.dredge_section_ft.gt(0)
          & ~h.flag_not_drillhole.fillna(False) & ~h.flag_grade_ambiguous.fillna(False)
          & ~h.map_title.fillna("").str.contains(NOT_GRADE, case=False))
    d = h[ok].reset_index(drop=True)
    P = np.array(sorted(cKDTree(d[["x", "y"]].values).query_pairs(radius)))
    a, b = d.iloc[P[:, 0]].reset_index(drop=True), d.iloc[P[:, 1]].reset_index(drop=True)
    same_map = a.map_id.values == b.map_id.values
    double = ((np.abs(a.grade_cents_raw - b.grade_cents_raw).values < 0.05)
              & (np.abs(a.dredge_section_ft - b.dredge_section_ft).values < 0.05))
    same_id = (a.hole_id.astype(str).values == b.hole_id.astype(str).values) & a.hole_id.notna().values
    keep = same_map & ~double & ~same_id
    log(f"pairs < {radius:.0f} m: {len(P)}; same map {same_map.sum()}; dropped double records "
        f"{(same_map & double).sum()}, same id {(same_map & same_id).sum()}; kept {keep.sum()}")
    a, b = a[keep].reset_index(drop=True), b[keep].reset_index(drop=True)
    out = pd.DataFrame({"map_id": a.map_id, "title": a.map_title, "year": a.map_year,
                        "id1": a.hole_id, "id2": b.hole_id, "uid1": a.uid, "uid2": b.uid,
                        "dist_m": np.hypot(a.x - b.x, a.y - b.y),
                        "g1": a.grade_cents_2067, "g2": b.grade_cents_2067,
                        "ds1": a.dredge_section_ft, "ds2": b.dredge_section_ft,
                        "ob1": a.overburden_ft, "ob2": b.overburden_ft,
                        "notes1": a.notes, "notes2": b.notes})
    ow = out.notes1.fillna("").str.contains(r"O\.?W", regex=True) | out.notes2.fillna("").str.contains(r"O\.?W", regex=True)
    out["old_workings"] = ow
    c1 = units.cents_per_yd3_to_mg_per_m3(out.g1.values); c2 = units.cents_per_yd3_to_mg_per_m3(out.g2.values)
    out["V1"] = sampling.sample_volume_m3(out.ds1.values); out["V2"] = sampling.sample_volume_m3(out.ds2.values)
    out["c1"], out["c2"] = c1, c2
    out["both_pos"] = (out.g1 > 0.15) & (out.g2 > 0.15)
    out["logratio"] = np.where(out.both_pos, np.log(np.maximum(c1, 1e-9) / np.maximum(c2, 1e-9)), np.nan)
    out["gold_mg_pair"] = (c1 * out.V1 + c2 * out.V2) / 2   # mean gold mass per sample
    return out


def robust_var(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float((1.4826 * np.median(np.abs(x - np.median(x)))) ** 2)


def simulate_twins(p: pd.DataFrame, mbar, alpha, sd_hole, tau_geo, rng, reps=200):
    """Simulated log-ratios for the observed pairs (same volumes, same level), positive pairs only."""
    c = np.sqrt(np.maximum(p.c1.values, 1.0) * np.maximum(p.c2.values, 1.0))   # common level
    out = []
    for r in range(reps):
        dgeo = rng.normal(0, tau_geo, len(c))
        e1 = rng.normal(0, sd_hole, len(c)); e2 = rng.normal(0, sd_hole, len(c))
        S1, _ = sampling.simulate(c * np.exp(dgeo / 2 + e1), p.V1.values, mbar, alpha, rng)
        S2, _ = sampling.simulate(c * np.exp(-dgeo / 2 + e2), p.V2.values, mbar, alpha, rng)
        G1, G2 = S1 / p.V1.values, S2 / p.V2.values
        tr = units.cents_per_yd3_to_mg_per_m3(0.15)
        ok = (G1 > tr) & (G2 > tr)
        out.append(pd.DataFrame({"lr": np.log(G1[ok] / G2[ok]), "gold": p.gold_mg_pair.values[ok], "rep": r}))
    return pd.concat(out)


def field_fits(h, maps):
    rows = []
    for m in maps:
        g = h[(h.map_id == m) & h.usable]
        P = g[["x", "y"]].to_numpy(); c0 = P.mean(0)
        U = np.linalg.svd(P - c0, full_matrices=False)[2]; X = (P - c0) @ U.T
        kind, s, st, V = sampling.observations_from_grade(g.grade_cents_raw.values, g.dredge_section_ft.values,
                                                          price=g.gold_price.values)
        for dn in (False, True):
            t0 = time.time()
            f = physical.CPGField(maxiter=80, drill_noise=dn, init={"ls_s": 150.0, "ls_n": 40.0}).fit(X, kind, s, st, V)
            r = f.summary()
            rows.append({"map_id": m, "title": g.map_title.iloc[0][:60], "n": len(g),
                         "model": "grain + per-hole" if dn else "grain only",
                         **{k: r[k] for k in ["logZ", "m_eff", "mbar", "alpha", "sigma", "ls_s", "ls_n", "sd_drill"]},
                         "median_gold_mg": float(np.median(s[kind == 2]))})
            log(f"{m} {rows[-1]['model']:16s} logZ={r['logZ']:.1f} m_eff={r['m_eff']:.1f} sd_hole={r['sd_drill']:.2f} "
                f"ls=({r['ls_s']:.0f},{r['ls_n']:.0f}) ({time.time()-t0:.0f}s)")
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = pd.read_csv(ROOT / "data" / "interim" / "ycgc_holes.csv", low_memory=False)
    p = twin_pairs(h)
    p.to_csv(OUT / "pairs.csv", index=False)
    pos = p[p.both_pos]
    by_map = p.groupby("map_id").agg(n=("dist_m", "size"), title=("title", "first"), median_dist=("dist_m", "median"))
    log("pairs by map:\n" + by_map.sort_values("n", ascending=False).to_string())

    # 1. observed spread, all and without old-workings holes, and by expected gold per sample
    summ = {"n_pairs": int(len(p)), "n_both_positive": int(len(pos)), "n_with_zero": int((~p.both_pos).sum()),
            "median_dist_m": float(p.dist_m.median()),
            "robust_var_logratio": robust_var(pos.logratio),
            "robust_var_logratio_no_old_workings": robust_var(pos[~pos.old_workings].logratio),
            "n_no_old_workings": int((~pos.old_workings).sum()),
            "median_abs_logratio": float(np.median(np.abs(pos.logratio)))}
    edges = np.quantile(pos.gold_mg_pair, [0, 1 / 3, 2 / 3, 1])
    pos = pos.assign(tercile=np.digitize(pos.gold_mg_pair, edges[1:-1]))
    summ["by_gold_tercile"] = [{"gold_mg_range": [float(edges[i]), float(edges[i + 1])],
                                "n": int((pos.tercile == i).sum()),
                                "robust_var": robust_var(pos.logratio[pos.tercile == i])} for i in range(3)]
    log(json.dumps(summ, indent=1))

    # 2. simulated twins under alternative nugget models
    rng = np.random.default_rng(7)
    scen = {"grains only, m_eff 10 mg (mid-Dominion fit)": dict(mbar=3.85, alpha=0.58, sd_hole=0.0, tau_geo=0.0),
            "grains only, m_eff 90 mg (coarse, Sulphur)": dict(mbar=15.0, alpha=0.2, sd_hole=0.0, tau_geo=0.0),
            "grains m_eff 10 mg + per-hole sd 0.8": dict(mbar=3.85, alpha=0.58, sd_hole=0.8, tau_geo=0.0)}
    sims = {}
    for k, v in scen.items():
        sim = simulate_twins(pos, rng=rng, **v)
        sims[k] = sim
        summ.setdefault("simulated_robust_var", {})[k] = robust_var(sim.lr)
        summ.setdefault("simulated_by_tercile", {})[k] = [robust_var(sim.lr[np.digitize(sim.gold, edges[1:-1]) == i]) for i in range(3)]
    log("simulated robust var: " + json.dumps(summ["simulated_robust_var"], indent=1))

    # 3. full field model with / without per-hole error on twin-rich maps
    maps = by_map[by_map.n >= 4].index.tolist()
    ff = field_fits(h, maps)
    ff.to_csv(OUT / "field_fits.csv", index=False)
    piv = ff.pivot(index="map_id", columns="model", values="logZ")
    summ["delta_logZ_per_hole_minus_grain_only"] = (piv["grain + per-hole"] - piv["grain only"]).round(2).to_dict()
    summ["sd_hole_fitted"] = ff[ff.model == "grain + per-hole"].set_index("map_id").sd_drill.round(2).to_dict()
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2))
    plot(pos, sims, ff, edges)


def plot(pos, sims, ff, edges):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.4, 1]})
    a = ax[0]
    a.scatter(pos.gold_mg_pair, np.abs(pos.logratio), s=18, c=np.where(pos.old_workings, "tab:orange", "k"),
              alpha=0.7, label="observed twin pairs (orange: old workings)")
    bins = np.logspace(np.log10(pos.gold_mg_pair.min()), np.log10(pos.gold_mg_pair.max()), 8)
    cols = ["tab:blue", "tab:purple", "tab:green"]
    for (k, sim), col in zip(sims.items(), cols):
        idx = np.digitize(sim.gold, bins)
        xs, med, hi = [], [], []
        for i in range(1, len(bins)):
            v = np.abs(sim.lr[idx == i])
            if len(v) > 50:
                xs.append(np.sqrt(bins[i - 1] * bins[i])); med.append(np.median(v)); hi.append(np.quantile(v, 0.9))
        a.plot(xs, med, "-", color=col, lw=2, label=f"{k}: median")
        a.plot(xs, hi, "--", color=col, lw=1)
    a.set_xscale("log"); a.set_xlabel("gold in the drill sample (mg, pair mean)")
    a.set_ylabel("|log(G1 / G2)| for holes 3-12 m apart")
    a.set_title("Twin holes disagree far more than grain counting predicts")
    a.legend(fontsize=7, loc="upper right")
    b = ax[1]
    piv = ff.pivot(index="map_id", columns="model", values="logZ")
    d = (piv["grain + per-hole"] - piv["grain only"]).sort_values()
    sd = ff[ff.model == "grain + per-hole"].set_index("map_id").sd_drill
    b.barh(range(len(d)), d.values, color="tab:green")
    b.set_yticks(range(len(d))); b.set_yticklabels([f"{m} (sd={sd[m]:.2f})" for m in d.index], fontsize=7)
    b.axvline(0, color="k", lw=0.8)
    b.set_xlabel("Δ log marginal likelihood (per-hole error − grains only)")
    b.set_title("Adding a per-hole error term, maps with twins")
    fig.tight_layout(); fig.savefig(OUT / "twins.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
