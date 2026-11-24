"""Does grain counting predict how wrong drill-based block estimates are?

For each mined block we simulate, from the fitted model, the joint behaviour of
  * the true block grade (mean of exp(f) over the block), and
  * the drill estimate (mean of the assays of the holes inside the block, each
    a compound Poisson draw of grains from its own sample volume),
using the ACTUAL hole positions and dredge-section thicknesses. The spread of
log(estimate / truth) is the predicted error of a drill-based block estimate,
split into a spatial part (holes are points in a block) and a grain-counting part.

Compared with the observed spread of drill estimate vs dredge recovery:
  upper Sulphur  9 annual blocks (YCGC26682/25152), recorded estimate and recovery
  mid-Dominion   van Loon (2016): 10 annual blocks, 6 under-estimated by 23% on average,
                 4 over-estimated by 14% -> sd of log ratio >= 0.17-0.21 (lower bound:
                 within-group scatter is not reported)

Only the SPREAD is compared: the mean offset (YCGC's estimation factor, dredge
recovery) is not part of the drilling physics.

Usage: python scripts/08_error_bound.py
Outputs: reports/error_bound/{blocks.csv, summary.json, error_bound.png}
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from klondyke import physical, sampling, valley  # noqa: E402

OUT = ROOT / "reports" / "error_bound"
X0, Y0 = 197000.0, 1040000.0
RNG = np.random.default_rng(2026)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def cells(rings, step=10.0, cap=700):
    allp = np.vstack(rings)
    xs = np.arange(allp[:, 0].min(), allp[:, 0].max(), step)
    ys = np.arange(allp[:, 1].min(), allp[:, 1].max(), step)
    G = np.array(np.meshgrid(xs, ys)).reshape(2, -1).T
    m = np.zeros(len(G), bool)
    for r in rings:
        m |= MPath(r).contains_points(G)
    G = G[m]
    if len(G) > cap:
        G = G[RNG.choice(len(G), cap, replace=False)]
    return G


def simulate_block(cell_sn, hole_sn, hole_V, p, n_sims=1500):
    """Return arrays of log(est/truth) with grain counting, and spatial-only."""
    P = np.vstack([cell_sn, hole_sn])
    K = physical.matern32(P, P, np.array([p["ls_s"], p["ls_n"]]), p["sigma"] ** 2)
    K[np.diag_indices_from(K)] += 1e-6
    L = np.linalg.cholesky(K)
    nc = len(cell_sn)
    F = p["beta0"] + (L @ RNG.standard_normal((len(P), n_sims))).T      # (sims, points)
    C = np.exp(F)
    truth = C[:, :nc].mean(1)
    ch = C[:, nc:]
    S, _ = sampling.simulate(ch, np.broadcast_to(hole_V, ch.shape), p["mbar"], p["alpha"], RNG)
    est = (S / hole_V[None, :]).mean(1)
    est_sp = ch.mean(1)
    ok = est > 0
    return np.log(est[ok] / truth[ok]), np.log(est_sp / truth), float(1 - ok.mean())


def robust_sd(x):
    q1, q3 = np.percentile(x, [25, 75])
    return float((q3 - q1) / 1.349)


def sulphur():
    s07 = _load_module("s07", ROOT / "scripts" / "07_sulphur_dredge_check.py")
    h = pd.read_csv(ROOT / "reports" / "sulphur" / "holes_used.csv", low_memory=False)
    poly, s, n, lab, r = s07.frame(h)
    off = s.min()
    b = s07.blocks()
    hm = pd.read_csv(ROOT / "reports" / "sulphur" / "dredge_blocks.csv")
    b["holes_mean"] = hm.holes_mean.values
    p = json.loads((ROOT / "reports" / "sulphur" / "cpg_fit.json").read_text())
    rows = []
    for k, rr in enumerate(b.rings):
        G = cells(rr)
        cs, cn = valley.project_to_centreline(G[:, 0], G[:, 1], poly)
        hk = h[h.block == k]
        if len(hk) == 0:
            continue
        V = sampling.sample_volume_m3(hk.dredge_section_ft.values)
        e, e_sp, p_allzero = simulate_block(np.column_stack([cs - off, cn]), hk[["s", "n"]].to_numpy(), V, p)
        obs = -float(np.log(b.act_c[k] / b.holes_mean[k]))      # log(estimate / recovered), same sign as e
        rows.append({"creek": "upper Sulphur", "block": int(b.YEAR[k]), "n_holes": len(hk),
                     "pred_sd_log": float(e.std()), "pred_sd_log_spatial_only": float(e_sp.std()),
                     "pred_robust_sd": robust_sd(e), "pred_robust_sd_spatial_only": robust_sd(e_sp),
                     "_sims": e, "_obs": obs,
                     "pred_mean_log": float(e.mean()), "p_all_holes_zero": p_allzero,
                     "obs_log_act_over_ycgc": float(np.log(b.act_c[k] / b.est_c[k])),
                     "obs_log_act_over_holes": float(np.log(b.act_c[k] / b.holes_mean[k]))})
    return pd.DataFrame(rows), p


def dominion():
    bm = pickle.load(open(ROOT / "reports" / "dominion" / "best_model.pkl", "rb"))
    poly, off, model = bm["poly"], bm["off"], bm["model"]
    p = model.p_
    d = pd.read_csv(ROOT / "reports" / "dominion" / "holes_curated.csv", dtype={"mapid": str, "hid": str})
    polys = []
    for line in (ROOT / "data" / "interim" / "dominion_mid_dredge_polys.txt").read_text().splitlines():
        head, geom = line.split(":", 1)
        if head[0] != "D" or len(head) < 5:
            continue
        rings = [np.array([[float(v) for v in q.split()] for q in r.split(";")]) + [X0, Y0] for r in geom.split("|")]
        polys.append((int(head[1:]), rings))
    by_year = {}
    for yr, rings in polys:
        by_year.setdefault(yr, []).extend(rings)
    rows = []
    for yr, rings in sorted(by_year.items()):
        inside = np.zeros(len(d), bool)
        for r in rings:
            inside |= MPath(r).contains_points(d[["x", "y"]].to_numpy())
        hk = d[inside]
        if len(hk) < 3:
            continue
        G = cells(rings)
        cs, cn = valley.project_to_centreline(G[:, 0], G[:, 1], poly)
        V = sampling.sample_volume_m3(hk.ds.values)
        e, e_sp, p_allzero = simulate_block(np.column_stack([cs - off, cn]), hk[["s", "n"]].to_numpy(), V, p)
        rows.append({"creek": "mid-Dominion", "block": yr, "n_holes": len(hk),
                     "pred_sd_log": float(e.std()), "pred_sd_log_spatial_only": float(e_sp.std()),
                     "pred_robust_sd": robust_sd(e), "pred_robust_sd_spatial_only": robust_sd(e_sp),
                     "pred_mean_log": float(e.mean()), "p_all_holes_zero": p_allzero})
    return pd.DataFrame(rows), p


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    su, ps = sulphur()
    do, pd_ = dominion()
    # calibration of the observed mean-of-holes errors (offset unknown -> centre both on their medians)
    pit = []
    obs_c = su._obs - su._obs.median()
    for (_, r), oc in zip(su.iterrows(), obs_c):
        sims = r._sims - np.median(r._sims)
        pit.append(float((sims < oc).mean()))
    su["pit_centered"] = pit
    su_obs_robust = robust_sd(su._obs)
    su = su.drop(columns=["_sims", "_obs"])
    allb = pd.concat([su, do], ignore_index=True)
    allb.to_csv(OUT / "blocks.csv", index=False)

    def rms(x):
        return float(np.sqrt(np.mean(np.asarray(x) ** 2)))

    vl_sd = {"lower": float(np.sqrt(6 * 4 / (10 * 9)) * (np.log(1.23) - np.log(0.86))),
             "upper": float(np.sqrt(6 * 4 / (10 * 9)) * (np.log(1 / 0.77) - np.log(1 / 1.14)))}
    summ = {
        "upper_Sulphur": {
            "m_eff_mg": ps["m_eff"], "n_blocks": len(su), "median_holes_per_block": float(su.n_holes.median()),
            "predicted_sd_log_error": rms(su.pred_sd_log), "predicted_sd_spatial_only": rms(su.pred_sd_log_spatial_only),
            "observed_sd_log_act_over_ycgc": float(su.obs_log_act_over_ycgc.std(ddof=1)),
            "observed_sd_log_act_over_holes_mean": float(su.obs_log_act_over_holes.std(ddof=1)),
            "predicted_robust_sd": float(np.median(su.pred_robust_sd)),
            "predicted_robust_sd_spatial_only": float(np.median(su.pred_robust_sd_spatial_only)),
            "observed_robust_sd_holes_mean": su_obs_robust,
            "pit_centered": [round(v, 3) for v in su.pit_centered],
            "share_obs_within_central_50": float(np.mean(np.abs(np.array(su.pit_centered) - 0.5) <= 0.25)),
            "share_obs_within_central_90": float(np.mean(np.abs(np.array(su.pit_centered) - 0.5) <= 0.45)),
        },
        "mid_Dominion": {
            "m_eff_mg": float(physical.sampling.grain_params(pd_["mbar"], pd_["alpha"])[1]),
            "n_blocks": len(do), "median_holes_per_block": float(do.n_holes.median()),
            "predicted_sd_log_error": rms(do.pred_sd_log), "predicted_sd_spatial_only": rms(do.pred_sd_log_spatial_only),
            "observed_sd_log_vanLoon_lower_bound_range": [vl_sd["lower"], vl_sd["upper"]],
            "predicted_robust_sd": float(np.median(do.pred_robust_sd)),
            "predicted_robust_sd_spatial_only": float(np.median(do.pred_robust_sd_spatial_only)),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2))
    print(allb.round(3).to_string(index=False))
    print(json.dumps(summ, indent=2))
    plot(allb, summ)


def plot(allb, summ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, (creek, key) in enumerate([("mid-Dominion", "mid_Dominion"), ("upper Sulphur", "upper_Sulphur")]):
        g = allb[allb.creek == creek]
        ax.scatter(np.full(len(g), i) + np.linspace(-0.12, 0.12, len(g)), g.pred_sd_log, color="#d95f02", s=18,
                   label="predicted per block (grain counting + spatial)" if i == 0 else None)
        ax.scatter(np.full(len(g), i) + np.linspace(-0.12, 0.12, len(g)), g.pred_sd_log_spatial_only, color="#999",
                   s=14, marker="x", label="predicted, spatial only (no grain noise)" if i == 0 else None)
        sm = summ[key]
        if key == "upper_Sulphur":
            ax.hlines(sm["observed_sd_log_act_over_ycgc"], i - 0.3, i + 0.3, color="k", lw=2.5,
                      label="observed: drill estimate vs dredge recovery")
        else:
            lo, hi = sm["observed_sd_log_vanLoon_lower_bound_range"]
            ax.fill_between([i - 0.3, i + 0.3], lo, hi, color="k", alpha=0.35, label="observed (van Loon 2016), lower bound")
        ax.text(i, -0.06, f"m_eff ≈ {sm['m_eff_mg']:.0f} mg", ha="center", fontsize=9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["mid-Dominion\n(fine, travelled gold)", "upper Sulphur\n(coarse, angular gold)"])
    ax.set_ylim(-0.1, None)
    ax.set_ylabel("sd of log(drill estimate / true block grade)")
    ax.set_title("Grain counting predicts the error of drill-based block estimates")
    ax.legend(fontsize=7.5, loc="upper left"); fig.tight_layout()
    fig.savefig(OUT / "error_bound.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
