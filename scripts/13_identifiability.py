"""When can historic assays separate grain counting from a per-hole nugget?

Synthetic creeks with known truth. Assays are grain draws (compound Poisson-
gamma) from a latent concentration that also carries an optional per-hole
log-normal error (sd_hole: core recovery, volume factor, metre-scale traps).

Designs
  lines : YCGC-style lines (150 m apart, holes 40 m apart on a line)
  twins : the same plus a twin 8 m from 15% of the holes (as on Bear,
          Gold Run and Hunker creeks)

Fitted models: grain only; grain + per-hole error. Recorded: grain mass
estimates, per-hole sd, length scales, marginal likelihood, and the error of
the latent field at the holes.

Usage: python scripts/13_identifiability.py [--seeds 21,22,23]
Outputs: reports/identifiability/results.csv, summary.csv
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from klondyke import physical, sampling, synthetic, units  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "reports" / "identifiability"
MBAR, ALPHA = 3.85, 0.58          # mid-Dominion fit: m_eff = 10.5 mg


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def make(seed, sd_hole, twins, twin_frac=0.15, twin_dist=8.0):
    pp = synthetic.PhysParams(seed=seed)
    sp = synthetic.SynthParams(seed=seed, patch_sd=pp.richness_sd, length_m=4500.0)
    df, truth = synthetic.make_dataset(sp)
    s, n = df.s_true.to_numpy(), df.n_true.to_numpy()
    rng = np.random.default_rng(seed + 77)
    if twins:
        k = rng.choice(len(s), int(twin_frac * len(s)), replace=False)
        ang = rng.uniform(0, 2 * np.pi, len(k))
        s = np.concatenate([s, s[k] + twin_dist * np.cos(ang)])
        n = np.concatenate([n, n[k] + twin_dist * np.sin(ang)])
    logc = truth["logc_fn"](s, n) if "logc_fn" in truth else None
    if logc is None:
        # make_dataset returns geometry only; build the grain-counting truth as make_physical_dataset does
        ch = np.interp(s, truth["sg"], truth["n_c"]); rich = np.interp(s, truth["sg"], truth["richness"])
        logc = pp.log_c_background + (pp.log_c_peak_add + rich) * np.exp(-0.5 * ((n - ch) / sp.channel_width_m) ** 2)
    sec = np.clip(np.round(np.exp(np.log(pp.section_ft_median) + rng.normal(0, pp.section_ft_logsd, len(s)))), 2, 40)
    V = sampling.sample_volume_m3(sec)
    c_site = np.exp(logc + rng.normal(0, sd_hole, len(s)))
    S, _ = sampling.simulate(c_site, V, MBAR, ALPHA, rng)
    cents = (S / V) / units.cents_per_yd3_to_mg_per_m3(1.0)
    rec = np.round(cents, 1); rec = np.where((cents > 0) & (rec < 0.1), 0.1, rec)
    return np.column_stack([s, n]), rec, sec, logc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="21,22,23")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in [int(x) for x in a.seeds.split(",")]:
        for sd_hole in (0.0, 0.8):
            for design in ("lines", "twins"):
                X, rec, sec, logc = make(seed, sd_hole, design == "twins")
                kind, sm, st, V = sampling.observations_from_grade(rec, sec)
                for model in ("grain only", "grain + per-hole"):
                    t0 = time.time()
                    f = physical.CPGField(maxiter=80, drill_noise=(model != "grain only")).fit(X, kind, sm, st, V)
                    r = f.summary()
                    mu = f.predict_latent(X)[0]
                    rows.append({"seed": seed, "sd_hole_true": sd_hole, "design": design, "model": model, "n": len(rec),
                                 "zeros": float(np.mean(rec == 0)),
                                 **{k: r[k] for k in ["logZ", "m_eff", "mbar", "alpha", "sigma", "ls_s", "ls_n", "sd_drill"]},
                                 "rmse_latent": float(np.sqrt(np.mean((mu - logc) ** 2))),
                                 "secs": round(time.time() - t0)})
                    log(f"seed {seed} sd_hole {sd_hole} {design:5s} {model:16s} m_eff={r['m_eff']:6.2f} (true 10.49) "
                        f"sd_hat={r['sd_drill']:.2f} ls=({r['ls_s']:.0f},{r['ls_n']:.0f}) logZ={r['logZ']:.1f} "
                        f"rmse={rows[-1]['rmse_latent']:.3f} ({rows[-1]['secs']}s)")
                pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)
    res = pd.DataFrame(rows)
    piv = res.pivot_table(index=["seed", "sd_hole_true", "design"], columns="model", values="logZ")
    res["dlogZ_perhole"] = res.set_index(["seed", "sd_hole_true", "design"]).index.map(
        (piv["grain + per-hole"] - piv["grain only"]).to_dict())
    g = res.groupby(["sd_hole_true", "design", "model"]).agg(
        m_eff_mean=("m_eff", "mean"), m_eff_sd=("m_eff", "std"), sd_hat=("sd_drill", "mean"),
        ls_n=("ls_n", "median"), rmse_latent=("rmse_latent", "mean"), dlogZ=("dlogZ_perhole", "mean"))
    g.to_csv(OUT / "summary.csv")
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()
