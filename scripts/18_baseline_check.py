"""Is the kriging baseline of the synthetic benchmark a straw man?

Kriging log-assays estimates E[log G], which sits below log c when samples
hold few grains. The standard remedy in geostatistics is a log-normal
correction: if an assay is c times a noise factor of mean one with
log-variance tau^2 (the nugget), then E[log G] = log c - tau^2/2, so
log c is estimated by the kriged mean plus half the nugget. The same applies
to the Gaussian-process baseline with its white-noise variance.

This script repeats the held-out-line (LOLO) part of the synthetic benchmark
(same creeks, frame and folds as run_physical_benchmark.py, seeds 11-13) for
  kriging, kriging + nugget/2, GP, GP + noise/2,
and sets them against the saved grain-counting results.

Usage: python scripts/18_baseline_check.py
Outputs: reports/baseline_check.csv, reports/baseline_check_summary.csv
"""

from __future__ import annotations

import importlib.util
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from klondyke import baselines as B  # noqa: E402
from klondyke import cv, models, sampling, synthetic, units, valley  # noqa: E402

warnings.filterwarnings("ignore")
spec = importlib.util.spec_from_file_location("bench", ROOT / "scripts" / "run_physical_benchmark.py")
bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)


def run(seed, folds=10):
    df, _ = synthetic.make_physical_dataset(synthetic.PhysParams(seed=seed))
    poly = valley.centreline_from_drill_lines(df.x.values, df.y.values, df.claim_no.values)
    sv, nv = valley.project_to_centreline(df.x.values, df.y.values, poly)
    sv -= sv.min()
    X = np.column_stack([sv, nv])
    y = np.log1p(units.cents_per_yd3_to_mg_per_m3(df.grade_cents_yd3.values))
    T = df.logc_true.values
    names = ["kriging", "kriging + nugget/2", "GP", "GP + noise/2"]
    mu = {k: np.full(len(y), np.nan) for k in names}
    sd = {k: np.full(len(y), np.nan) for k in names}
    nug = []
    for i, (tr, te) in enumerate(cv.grouped_line_folds(df.line_id.values, n_folds=folds, seed=seed)):
        t0 = time.time()
        k = B.OrdinaryKriging(anisotropy="auto").fit(X[tr], y[tr])
        m_, s_ = k.predict_latent(X[te]); tau2 = k.vario_["nugget"]; nug.append(tau2)
        mu["kriging"][te], sd["kriging"][te] = m_, s_
        mu["kriging + nugget/2"][te], sd["kriging + nugget/2"][te] = m_ + tau2 / 2, s_
        g = models.AnisotropicGP(n_restarts=0).fit(X[tr], y[tr])
        m_, s_ = g.predict_latent(X[te])
        noise = g.gp.kernel_.k2.noise_level * g.gp._y_train_std ** 2
        mu["GP"][te], sd["GP"][te] = m_, s_
        mu["GP + noise/2"][te], sd["GP + noise/2"][te] = m_ + noise / 2, s_
        print(time.strftime("%H:%M:%S"), f"seed {seed} fold {i+1}: nugget {tau2:.2f}, GP noise {noise:.2f} ({time.time()-t0:.0f}s)",
              flush=True)
    rows = []
    for k in names:
        r = bench.score(T, mu[k], sd[k]); r.update(method=k, seed=seed, mean_nugget=float(np.mean(nug)))
        rows.append(r)
    return rows


def main():
    rows = []
    for seed in (11, 12, 13):
        rows += run(seed)
        saved = pd.read_csv(ROOT / "reports" / f"physical_benchmark_seed{seed}.csv")
        r = saved[(saved.method == "cpg_field") & (saved.scheme == "LOLO")].iloc[0].to_dict()
        r.update(method="grain-counting field (saved)", seed=seed)
        rows.append({k: r[k] for k in ["method", "seed", "rmse", "bias", "crps", "cov90", "top20_gold", "spearman_in_channel"]})
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "reports" / "baseline_check.csv", index=False)
    summ = res.groupby("method")[["rmse", "bias", "crps", "cov90", "top20_gold", "spearman_in_channel"]].agg(["mean", "std"])
    summ.to_csv(ROOT / "reports" / "baseline_check_summary.csv")
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
