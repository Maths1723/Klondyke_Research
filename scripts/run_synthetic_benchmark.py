"""Benchmark every method on a synthetic valley with known truth.

Runs the full pipeline the real data will go through:
  drill-hole map coordinates -> estimated centreline -> (s, n)
  -> leave-one-line-out and along-valley block CV -> scores.

Usage:  python scripts/run_synthetic_benchmark.py [--seeds 3]
Outputs: reports/synthetic_benchmark.csv, reports/figures/synthetic_*.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from klondyke import experiment, models, synthetic, valley  # noqa: E402


def build(seed):
    df, truth = synthetic.make_dataset(synthetic.SynthParams(seed=seed))
    poly = valley.centreline_from_drill_lines(df.x.values, df.y.values, df.claim_no.values)
    s, n = valley.project_to_centreline(df.x.values, df.y.values, poly)
    offset = s.min()
    df["s"], df["n"] = s - offset, n
    df["halfwidth"] = valley.valley_halfwidth(df.s.values, n, df.line_id.values)
    return df, truth, poly, offset


def save_figures(df, truth, poly, offset, gp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = ROOT / "reports" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    sc = ax.scatter(df.x, df.y, c=df.y_log, s=9, cmap="YlOrBr", vmin=0)
    ax.plot(poly[:, 0], poly[:, 1], "k-", lw=0.8, label="valley axis estimated from drill lines")
    ax.set_aspect("equal")
    ax.set_title("Synthetic valley: YCGC-style drill lines, coloured by log1p(accumulation)")
    ax.legend(loc="lower right"); fig.colorbar(sc, ax=ax, shrink=0.7, label="log1p(mg/m²)")
    fig.tight_layout(); fig.savefig(figdir / "synthetic_layout.png", dpi=150); plt.close(fig)

    sg = np.linspace(df.s.min(), df.s.max(), 400)
    ng = np.linspace(df.n.min(), df.n.max(), 120)
    S, N = np.meshgrid(sg, ng)
    mu, sd = gp.predict(np.column_stack([S.ravel(), N.ravel()]))
    # truth in the ESTIMATED frame: grid -> map xy -> true (s, n)
    qx, qy = valley.sn_to_xy(S.ravel(), N.ravel(), poly, s_offset=offset)
    true_poly = synthetic.valley_centreline(np.arange(-500, 12000, 5.0))
    st, nt = valley.project_to_centreline(qx, qy, true_poly)
    T = truth["latent_fn"](st - 500.0, nt).reshape(S.shape)
    fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for a, Z, t in zip(axs, [T, mu.reshape(S.shape), sd.reshape(S.shape)],
                       ["truth (latent log accumulation)", "GP mean, valley coordinates",
                        "GP predictive sd (new measurement)"]):
        im = a.pcolormesh(S, N, Z, shading="auto", cmap="viridis" if "sd" in t else "YlOrBr")
        a.scatter(df.s, df.n, s=2, c="k", alpha=0.4)
        a.set_title(t); a.set_ylabel("n (m, + = left limit)")
        fig.colorbar(im, ax=a, shrink=0.8)
    axs[-1].set_xlabel("s, distance down valley (m)")
    fig.tight_layout(); fig.savefig(figdir / "synthetic_gp_vs_truth.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    all_rows, infos = [], []
    for seed in range(1, a.seeds + 1):
        df, truth, poly, offset = build(seed)
        print(f"--- seed {seed}: {len(df)} holes on {df.line_id.nunique()} lines")
        res, _ = experiment.run_all(df, seed=seed)
        all_rows.append(res)
        gp = models.AnisotropicGP(n_restarts=2).fit(df[["s", "n"]].to_numpy(), df.y_log.values)
        infos.append({"seed": seed,
                      "median_abs_axis_error_m": float(np.median(np.abs(df.n - df.n_true))),
                      "gp_full_fit": gp.params()})
        if seed == 1:
            save_figures(df, truth, poly, offset, gp)
    res = pd.concat(all_rows)
    rep = ROOT / "reports"
    res.to_csv(rep / "synthetic_benchmark_raw.csv", index=False)
    summ = experiment.summarise(res)
    summ.to_csv(rep / "synthetic_benchmark.csv")
    (rep / "synthetic_benchmark_info.json").write_text(json.dumps(infos, indent=2))
    print("\n", summ.to_string())


if __name__ == "__main__":
    main()
