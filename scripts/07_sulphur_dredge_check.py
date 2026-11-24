"""Upper Sulphur Creek: do the models predict what the dredge actually recovered?

Ground truth: 9 annual dredge-No.9 blocks (1950-1958) with YCGC's pre-dredging
estimate AND the recovered value, both at $20.67 (maps YCGC26682 / 25152).
Every estimator sees the same drill holes YCGC had and predicts each block's
grade (cents/yd3 over the sluiced section):

  ycgc_recorded    YCGC's own estimate, from the map
  holes_mean       arithmetic mean of the drill holes inside the block
  kriging_block    ordinary kriging of log(1+grade), lognormal back-transform,
                   averaged over the block (nugget excluded)
  cpg_block        grain-counting latent field, E[c] averaged over the block

Two settings: 'all holes' (as YCGC had), and 'leave-block-out' (holes inside
the block removed: predicting a block from surrounding drilling only).

Price basis: maps 26683 and 26743 carry identical grades for 84 co-located holes
although 26743 records $35 and 26683 nothing; YCGC valued these blocks at
$20.67. Primary run: both at $20.67. Sensitivity: both at $35 (--price35).

Usage: python scripts/07_sulphur_dredge_check.py [--price35] [--radius 40]
Outputs: reports/sulphur/dredge_check[_price35].csv, figure, holes used.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MPath
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from klondyke import baselines as B  # noqa: E402
from klondyke import physical, sampling, units, valley  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "reports" / "sulphur"
MAPS = ["YCGC25153", "YCGC26682", "YCGC26683", "YCGC26743", "YCGC26781", "YCGC26782", "YCGC26783",
        "YCGC26744", "YCGC26701a"]
C1 = units.cents_per_yd3_to_mg_per_m3(1.0)


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def load(price35: bool):
    h = pd.read_csv(ROOT / "data" / "interim" / "ycgc_holes.csv", low_memory=False)
    h = h[h.usable & h.map_id.isin(MAPS)].copy()
    h = h[h.x.between(183500, 188500) & h.y.between(1042500, 1051000)]
    p = 35.0 if price35 else 20.67
    sel = h.map_id.isin(["YCGC26683", "YCGC26743"])
    h.loc[sel, "gold_price"] = p
    h["c"] = h.grade_cents_raw * 20.67 / h.gold_price
    return h.reset_index(drop=True)


def blocks():
    d = pd.read_pickle(ROOT / "data" / "interim" / "ycgc_dredge.pkl")
    b = d[d.MAP_ID.isin(["YCGC26682", "YCGC25152"]) & d.VALUE.notna()].sort_values("YEAR").reset_index(drop=True)
    b["est_c"] = 100 * b.ESTIMATED_VALUE / b.ESTIMATED_YARDAGE
    b["act_c"] = 100 * b.VALUE / b.YARDAGE_SLUICED
    return b


def cells_in(rings, step=5.0):
    allp = np.vstack(rings)
    xs = np.arange(allp[:, 0].min(), allp[:, 0].max(), step)
    ys = np.arange(allp[:, 1].min(), allp[:, 1].max(), step)
    G = np.array(np.meshgrid(xs, ys)).reshape(2, -1).T
    m = np.zeros(len(G), bool)
    for r in rings:
        m |= MPath(r).contains_points(G)
    return G[m]


def frame(h, radius=None):
    nn = cKDTree(h[["x", "y"]].to_numpy()).query(h[["x", "y"]].to_numpy(), k=2)[0][:, 1]
    r = float(np.clip(1.15 * np.percentile(nn, 75), 40, 90))   # robust to twin holes / 30 m infill
    if radius is not None:                                     # sensitivity: fixed line radius
        r = float(radius)
    poly, lab = valley.reconstruct_frame(h.x.values, h.y.values, radius=r, max_angle_deg=90.0)
    s, n = valley.project_to_centreline(h.x.values, h.y.values, poly)
    return poly, s, n, lab, r


def fit_predict(h, X, train, Q_by_block):
    """Fit both models on rows `train`; return block predictions (cents/yd3)."""
    kind, sm, st, V = sampling.observations_from_grade(h.c.values, h.dredge_section_ft.values)
    cpg = physical.CPGField(maxiter=60).fit(X[train], kind[train], sm[train], st[train], V[train])
    y = np.log1p(h.c.values)
    ok = B.OrdinaryKriging(anisotropy="auto").fit(X[train], y[train])
    out = []
    for Q in Q_by_block:
        mu, sd = cpg.predict_latent(Q)
        cpg_c = float(np.mean(np.exp(mu + 0.5 * sd ** 2)) / C1)
        km, ks = ok.predict_latent(Q)
        kr_c = float(np.mean(np.exp(km + 0.5 * ks ** 2) - 1.0))
        out.append((cpg_c, kr_c))
    return out, cpg.summary()


def score(pred, act):
    lr = np.log(pred / act)
    return {"bias_total": float(pred.sum() / act.sum() - 1), "mean_log_err": float(lr.mean()),
            "rmse_log": float(np.sqrt(np.mean(lr ** 2))), "spearman": float(spearmanr(pred, act).statistic),
            "pearson": float(np.corrcoef(pred, act)[0, 1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price35", action="store_true")
    ap.add_argument("--radius", type=float, default=None, help="fixed drill-line radius (m); default from hole spacing")
    a = ap.parse_args()
    tag = ("_price35" if a.price35 else "") + (f"_radius{a.radius:g}" if a.radius else "")
    OUT.mkdir(parents=True, exist_ok=True)
    h = load(a.price35)
    b = blocks()
    poly, s, n, lab, r = frame(h, a.radius)
    h["s"], h["n"], h["line"] = s - s.min(), n, lab
    X = h[["s", "n"]].to_numpy()
    off = s.min()
    log(f"{len(h)} holes on {h.map_id.nunique()} maps, {len(np.unique(lab))} lines (radius {r:.0f} m)")

    Q = []
    for rings in b.rings:
        G = cells_in(rings)
        qs, qn = valley.project_to_centreline(G[:, 0], G[:, 1], poly)
        Q.append(np.column_stack([qs - off, qn]))
    inside = np.full(len(h), -1)
    for k, rings in enumerate(b.rings):
        m = np.zeros(len(h), bool)
        for r_ in rings:
            m |= MPath(r_).contains_points(h[["x", "y"]].to_numpy())
        inside[m & (inside < 0)] = k
    h["block"] = inside
    b["n_holes"] = [(inside == k).sum() for k in range(len(b))]
    b["holes_mean"] = [h.c[inside == k].mean() if (inside == k).any() else np.nan for k in range(len(b))]

    log("fit on all holes")
    preds, summ = fit_predict(h, X, np.ones(len(h), bool), Q)
    b["cpg_block"] = [p[0] for p in preds]; b["kriging_block"] = [p[1] for p in preds]
    log(f"  CPG: {json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in summ.items()})}")

    lbo_c, lbo_k = [], []
    for k in range(len(b)):
        t0 = time.time()
        p, _ = fit_predict(h, X, inside != k, [Q[k]])
        lbo_c.append(p[0][0]); lbo_k.append(p[0][1])
        log(f"  leave-block-out {int(b.YEAR[k])}: cpg {p[0][0]:.1f}  kriging {p[0][1]:.1f}  actual {b.act_c[k]:.1f} ({time.time()-t0:.0f}s)")
    b["cpg_lbo"], b["kriging_lbo"] = lbo_c, lbo_k

    act = b.act_c.values
    rows = []
    for name, col in [("YCGC recorded estimate", "est_c"), ("mean of holes in block", "holes_mean"),
                      ("kriging (all holes)", "kriging_block"), ("grain-counting field (all holes)", "cpg_block"),
                      ("kriging (leave-block-out)", "kriging_lbo"), ("grain-counting field (leave-block-out)", "cpg_lbo")]:
        p = b[col].values.astype(float)
        ok = np.isfinite(p)
        sc = score(p[ok], act[ok]); sc.update(estimator=name, n_blocks=int(ok.sum())); rows.append(sc)
    res = pd.DataFrame(rows)[["estimator", "n_blocks", "bias_total", "mean_log_err", "rmse_log", "spearman", "pearson"]]
    res.to_csv(OUT / f"dredge_check{tag}.csv", index=False)
    b.drop(columns=["rings"]).to_csv(OUT / f"dredge_blocks{tag}.csv", index=False)
    h.to_csv(OUT / f"holes_used{tag}.csv", index=False)
    (OUT / f"cpg_fit{tag}.json").write_text(json.dumps(summ, indent=2))
    print(b[["YEAR", "n_holes", "est_c", "holes_mean", "kriging_block", "cpg_block", "kriging_lbo", "cpg_lbo", "act_c"]].round(1).to_string(index=False))
    print(res.round(3).to_string(index=False))
    plot(b, tag)


def plot(b, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    yrs = b.YEAR.astype(int).values
    ax.plot(yrs, b.act_c, "ko-", lw=2, label="recovered by dredge")
    ax.plot(yrs, b.est_c, "s--", color="#999", label="YCGC estimate (recorded)")
    ax.plot(yrs, b.kriging_block, "^-", color="#1f77b4", alpha=0.8, label="kriging, all holes")
    ax.plot(yrs, b.cpg_block, "o-", color="#d95f02", label="grain-counting field, all holes")
    ax.plot(yrs, b.cpg_lbo, "o:", color="#d95f02", alpha=0.6, label="grain-counting, leave-block-out")
    ax.set_ylabel("grade, cents/yd³ at $20.67"); ax.set_xlabel("dredge No. 9 season")
    ax.set_title("Upper Sulphur Creek: predicted vs recovered grade, 9 annual dredge blocks")
    ax.legend(fontsize=8, ncol=2); fig.tight_layout()
    fig.savefig(OUT / f"dredge_check{tag}.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
