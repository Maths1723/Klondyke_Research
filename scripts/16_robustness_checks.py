"""Robustness checks quoted in the paper that are not produced by scripts 05-15.

1. Mid-Dominion: profile over a fixed per-hole error sigma_h (grain mass, fit).
2. Mid-Dominion: profile over the structural-zero probability pi0, and the
   "fine gold + barren holes" solution at pi0 = 0.1.
3. Campaigns: grain mass under structural zeros and a per-hole error.
4. Twin holes: rank correlation of disagreement with gold per sample; offsets
   between drilling series on lower Hunker and Gold Run.
5. Mid-Dominion targets: agreement of target probabilities with sigma_h = 0 and 0.7
   (needs 06_dominion_targets.py run with and without --sd-hole 0.7).

Usage: python scripts/16_robustness_checks.py
Outputs: reports/robustness/*.csv, summary.json
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from klondyke import physical, sampling  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "reports" / "robustness"
NOT_GRADE = re.compile(r"thaw|strip|progress|workings|tailing|dredge no\.? ?\d+ (area|workings)", re.I)
BASE = {"sigma": 1.212, "ls_s": 129.64, "ls_n": 32.75, "beta0": 5.642, "mbar": 3.85, "alpha": 0.58}


def summary(f):
    r = f.summary()
    return {k: float(r[k]) for k in ["logZ", "m_eff", "mbar", "alpha", "sigma", "ls_s", "ls_n", "sd_drill", "pi0"]}


def dominion():
    d = pd.read_csv(ROOT / "reports" / "dominion" / "holes_curated.csv")
    kind, s, st, V = sampling.observations_from_grade(d.c.values, d.ds.values)
    X = d[["s", "n"]].to_numpy()
    rows = []
    for sd in [0.0, 0.3, 0.6, 0.8, 1.0, 1.2]:
        kw = dict(drill_noise=True, fixed={"sd_drill": sd}) if sd > 0 else {}
        rows.append({"check": "sd_hole_profile", "value": sd, **summary(physical.CPGField(maxiter=80, **kw).fit(X, kind, s, st, V))})
        print(rows[-1], flush=True)
    for pi in [0.0, 0.01, 0.02, 0.05, 0.1]:
        if pi == 0:
            f = physical.CPGField(maxiter=100, init=BASE).fit(X, kind, s, st, V)
        else:
            f = physical.CPGField(maxiter=100, zero_infl=True, init=BASE, fixed={"pi0": pi}).fit(X, kind, s, st, V)
        rows.append({"check": "pi0_profile", "value": pi, **summary(f)})
        print(rows[-1], flush=True)
    f = physical.CPGField(maxiter=100, zero_infl=True, init={"pi0": 0.1, "mbar": 0.5}).fit(X, kind, s, st, V)
    rows.append({"check": "fine_gold_with_barren_holes", "value": np.nan, **summary(f)})
    print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def campaigns():
    h = pd.read_csv(ROOT / "data" / "interim" / "ycgc_holes.csv", low_memory=False)
    h = h[h.usable & ~h.map_title.fillna("").str.contains(NOT_GRADE)]
    rows = []
    for m in ["YCGC26743", "YCGC26683", "YCGC26741", "YCGC26199", "YCGC26222", "YCGC26340", "YCGC26346", "YCGC26461"]:
        g = h[h.map_id == m]
        P = g[["x", "y"]].to_numpy(); c0 = P.mean(0)
        U = np.linalg.svd(P - c0, full_matrices=False)[2]; X = (P - c0) @ U.T
        kind, s, st, V = sampling.observations_from_grade(g.grade_cents_raw.values, g.dredge_section_ft.values,
                                                          price=g.gold_price.values)
        for name, kw in [("grains", {}), ("grains + zeros", {"zero_infl": True}),
                         ("grains + per-hole 0.7 + zeros", {"zero_infl": True, "drill_noise": True, "fixed": {"sd_drill": 0.7}})]:
            f = physical.CPGField(maxiter=80, init={"ls_s": 150.0, "ls_n": 40.0}, **kw).fit(X, kind, s, st, V)
            rows.append({"map_id": m, "model": name, "n": len(g), "zero_share": float((kind == 0).mean()), **summary(f)})
            print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def twins():
    p = pd.read_csv(ROOT / "reports" / "twins" / "pairs.csv")
    pos = p[p.both_pos]
    tr = 19.68 * 0.15
    lr0 = np.log(np.maximum(p.c1, tr) / np.maximum(p.c2, tr))
    out = {"spearman_abs_logratio_vs_gold_positive_pairs": list(map(float, spearmanr(np.abs(pos.logratio), pos.gold_mg_pair))),
           "spearman_abs_logratio_vs_gold_all_pairs_zeros_at_trace": list(map(float, spearmanr(np.abs(lr0), p.gold_mg_pair))),
           "n_positive_pairs": int(len(pos)), "n_pairs": int(len(p))}

    def num(x):
        mm = re.match(r"(\d+)", str(x))
        return int(mm.group(1)) if mm else np.nan
    for m in ["YCGC26438", "YCGC26341"]:
        g = p[p.map_id == m]
        n1, n2 = g.id1.map(num), g.id2.map(num)
        lr = np.log(np.maximum(g.c1, tr) / np.maximum(g.c2, tr))
        sgn = np.where(n1 >= 70, 1, np.where(n2 >= 70, -1, np.nan))   # later series numbered from 70 up
        o = (lr * sgn).dropna() if hasattr(lr, "dropna") else lr * sgn
        o = np.asarray(o)[np.isfinite(o)]
        out[f"series_offset_{m}"] = {"n": int(len(o)), "median": float(np.median(o)), "share_positive": float(np.mean(o > 0))}
    return out


def targets():
    a = np.load(ROOT / "reports" / "dominion" / "target_grid.npz")
    b = np.load(ROOT / "reports" / "dominion" / "target_grid_sdhole0.7.npz")
    um = a["unmined"]
    pa, pb = a["P"][um], b["P"][um]
    k = int(0.2 * um.sum())
    top = len(set(np.argsort(-pa)[:k]) & set(np.argsort(-pb)[:k])) / k
    rel = um & (a["S"] < 6800)
    ta, tb = a["P"][rel] >= 0.5, b["P"][rel] >= 0.5
    return {"spearman_P_unmined": float(spearmanr(pa, pb).statistic), "top_fifth_overlap": float(top),
            "jaccard_targets_reliable_frame": float((ta & tb).sum() / (ta | tb).sum()),
            "target_ha_base": float(ta.sum() * 100 / 1e4), "target_ha_sdhole": float(tb.sum() * 100 / 1e4)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summ = {"twins": twins(), "targets": targets()}
    print(json.dumps(summ, indent=1), flush=True)
    dominion().to_csv(OUT / "dominion_profiles.csv", index=False)
    campaigns().to_csv(OUT / "campaign_alternatives.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
