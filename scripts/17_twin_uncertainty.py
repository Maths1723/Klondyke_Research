"""Uncertainty and sensitivity of the twin-hole results (scripts/11_twin_holes.py).

Twin pairs cluster by map (38 of 78 on one 1912 Bear Creek map), so every
interval here resamples whole maps (cluster bootstrap), and the trend test
permutes gold content only within maps, so that a contrast between maps
cannot produce it.

  1. Robust variance of log(G1/G2), observed and simulated under grain counting
     (m_eff 10 mg and 90 mg), with cluster-bootstrap intervals for their ratio.
  2. Trend of |log ratio| with gold per sample: Spearman rho with a cluster-
     bootstrap interval; within-map permutation p-value; slope of log|lr| on
     log gold with map fixed effects (grain counting predicts about -1/2), and
     the slope simulated under grain counting alone for m_eff from 10 to 300 mg.
  3. The same without Bear Creek (map 26009) and without holes through old
     workings.

Usage: python scripts/17_twin_uncertainty.py
Outputs: reports/twins/uncertainty.json
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from klondyke import sampling, units  # noqa: E402

OUT = ROOT / "reports" / "twins"
RNG = np.random.default_rng(2026)
B = 2000
REPS = 300


def robust_var(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float((1.4826 * np.median(np.abs(x - np.median(x)))) ** 2)


def simulate_pairs(p, mbar, alpha):
    """REPS simulated log-ratios per pair (grain counting only, same volumes and level)."""
    c = np.sqrt(np.maximum(p.c1.values, 1.0) * np.maximum(p.c2.values, 1.0))
    tr = units.cents_per_yd3_to_mg_per_m3(0.15)
    out = np.full((len(p), REPS), np.nan)
    for r in range(REPS):
        S1, _ = sampling.simulate(c, p.V1.values, mbar, alpha, RNG)
        S2, _ = sampling.simulate(c, p.V2.values, mbar, alpha, RNG)
        G1, G2 = S1 / p.V1.values, S2 / p.V2.values
        ok = (G1 > tr) & (G2 > tr)
        out[ok, r] = np.log(G1[ok] / G2[ok])
    return out


def fe_slope(p):
    """Slope of log|lr| on log(gold) with map fixed effects (OLS on within-map deviations)."""
    y = np.log(np.abs(p.logratio.values) + 1e-3); x = np.log(p.gold_mg_pair.values)
    g = p.map_id.values
    yd = y - pd.Series(y).groupby(g).transform("mean").values
    xd = x - pd.Series(x).groupby(g).transform("mean").values
    return float((xd * yd).sum() / (xd ** 2).sum()) if (xd ** 2).sum() > 0 else np.nan


def within_map_perm_p(p, n=5000):
    """One-sided p for a positive association, permuting gold within maps (sum of within-map Spearman)."""
    def stat(gold):
        tot = 0.0
        for _, idx in p.groupby("map_id").indices.items():
            if len(idx) >= 3:
                tot += len(idx) * spearmanr(np.abs(p.logratio.values[idx]), gold[idx]).statistic
        return tot
    obs = stat(p.gold_mg_pair.values)
    gold = p.gold_mg_pair.values.copy()
    groups = list(p.groupby("map_id").indices.values())
    cnt = 0
    for _ in range(n):
        g2 = gold.copy()
        for idx in groups:
            g2[idx] = RNG.permutation(gold[idx])
        cnt += stat(g2) >= obs
    return float((cnt + 1) / (n + 1)), float(obs)


def summarise(p, sims, label):
    maps = p.map_id.unique()
    idx_by_map = {m: np.where(p.map_id.values == m)[0] for m in maps}
    obs = robust_var(p.logratio)
    sim10 = robust_var(sims["10"].ravel()); sim90 = robust_var(sims["90"].ravel())
    rho = spearmanr(np.abs(p.logratio), p.gold_mg_pair).statistic
    boots = []
    for _ in range(B):
        take = np.concatenate([idx_by_map[m] for m in RNG.choice(maps, len(maps), replace=True)])
        q = p.iloc[take]
        o = robust_var(q.logratio)
        boots.append((o / robust_var(sims["10"][take].ravel()), o / robust_var(sims["90"][take].ravel()),
                      spearmanr(np.abs(q.logratio), q.gold_mg_pair).statistic, fe_slope(q), o))
    boots = np.array(boots)
    ci = lambda k: [float(v) for v in np.nanpercentile(boots[:, k], [5, 95])]
    perm_p, _ = within_map_perm_p(p)
    return {"label": label, "n_pairs": int(len(p)), "n_maps": int(len(maps)),
            "robust_var_obs": obs, "robust_var_obs_ci90": ci(4),
            "robust_var_sim_10mg": sim10, "robust_var_sim_90mg": sim90,
            "ratio_obs_to_10mg": obs / sim10, "ratio_obs_to_10mg_ci90": ci(0),
            "ratio_obs_to_90mg": obs / sim90, "ratio_obs_to_90mg_ci90": ci(1),
            "spearman": float(rho), "spearman_ci90": ci(2),
            "fe_slope_log_abs_lr_on_log_gold": fe_slope(p), "fe_slope_ci90": ci(3),
            "within_map_permutation_p": perm_p}


def main():
    p = pd.read_csv(OUT / "pairs.csv")
    p = p[p.both_pos].reset_index(drop=True)
    sims = {"10": simulate_pairs(p, 3.85, 0.58), "90": simulate_pairs(p, 15.0, 0.2)}
    # grain-counting reference for the trend: slope under simulation
    sim_df = p.assign(logratio=sims["10"][:, 0])
    ok = np.isfinite(sim_df.logratio)
    res = {"all": summarise(p, sims, "all 72 pairs with gold in both holes"),
           "sim_fe_slope_10mg": fe_slope(sim_df[ok])}
    keep = (p.map_id != "YCGC26009").values
    res["without_bear_creek"] = summarise(p[keep].reset_index(drop=True), {k: v[keep] for k, v in sims.items()},
                                          "without Bear Creek 1912 (map 26009)")
    keep2 = (~p.old_workings).values
    res["without_old_workings"] = summarise(p[keep2].reset_index(drop=True), {k: v[keep2] for k, v in sims.items()},
                                            "without holes through old workings")
    # trend expected under grain counting alone, over a range of grain masses (computed last,
    # so that the random streams of the results above do not change)
    global REPS
    REPS = 100
    res["sim_fe_slope_by_m_eff"] = {}
    for meff, alpha in [(10, 0.58), (30, 0.4), (90, 0.2), (300, 0.1)]:
        S = simulate_pairs(p, meff / (1 + 1 / alpha), alpha)
        sl = [fe_slope(p.assign(logratio=S[:, r])[np.isfinite(S[:, r])]) for r in range(S.shape[1])]
        res["sim_fe_slope_by_m_eff"][str(meff)] = {"mean": float(np.mean(sl)), "q05": float(np.percentile(sl, 5)),
                                                    "q95": float(np.percentile(sl, 95)),
                                                    "robust_var": robust_var(S.ravel())}
    (OUT / "uncertainty.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
