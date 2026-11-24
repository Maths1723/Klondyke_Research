"""Is the Laplace estimate of grain mass biased? An MCMC check.

Conditional on the fitted spatial hyperparameters (σ, ℓ_s, ℓ_n, σ_h), sample
the posterior of the latent field and of the grain parameters (β₀, m̄, α)
exactly:
  * latent field f = β₀ + ε, ε ~ N(0, K): elliptical slice sampling
    (Murray, Adams & MacKay 2010), which needs no tuning;
  * (log m̄, log α, β₀): random-walk Metropolis given ε, flat priors on the
    log scale within the fitting bounds.
The posterior of m_eff = (1 + 1/α) m̄ is compared with the Laplace point
estimate, on a synthetic creek with known m_eff and on mid-Dominion.

Usage: python scripts/14_mcmc_check.py [--iters 20000] [--cases synthetic,dominion]
Outputs: reports/mcmc/summary.json, traces_*.npz
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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
OUT = ROOT / "reports" / "mcmc"


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def loglik(model, f, mbar, alpha):
    lam = np.exp(np.clip(f + np.log(model.V) - np.log(mbar), -40, 40))
    ll, _, _ = sampling.count_posterior(lam, model.kind, model.s, model.s_tr, mbar, alpha)
    return float(ll.sum())


def run_mcmc(model, iters, rng, burn=0.3, start="mode"):
    p = dict(model.p_)
    K = model._K(p)                                  # includes per-hole variance on the diagonal
    L = np.linalg.cholesky(K)
    n = len(model.V)
    eps = model.post_["f"] - p["beta0"]              # start at the Laplace mode
    b0, lmb, lal = p["beta0"], np.log(p["mbar"]), np.log(p["alpha"])
    if start == "dispersed":                         # far from the mode: field at its prior mean,
        eps = np.zeros(n)                            # grain mass x3, shape x2
        lmb, lal = lmb + np.log(3.0), lal + np.log(2.0)
    cur = loglik(model, b0 + eps, np.exp(lmb), np.exp(lal))
    step = np.array([0.05, 0.08, 0.15])              # β₀, log m̄, log α
    acc = np.zeros(3); tr = []
    lo_a, hi_a = physical.SPEC["alpha"][2]; lo_m, hi_m = physical.SPEC["mbar"][2]
    for it in range(iters):
        # elliptical slice sampling for eps
        nu = L @ rng.standard_normal(n)
        logy = cur + np.log(rng.random())
        th = rng.uniform(0, 2 * np.pi); lo, hi = th - 2 * np.pi, th
        while True:
            prop = eps * np.cos(th) + nu * np.sin(th)
            llp = loglik(model, b0 + prop, np.exp(lmb), np.exp(lal))
            if llp > logy:
                eps, cur = prop, llp
                break
            if th < 0:
                lo = th
            else:
                hi = th
            th = rng.uniform(lo, hi)
        # Metropolis on each grain / level parameter in turn
        for j in range(3):
            q = [b0, lmb, lal]
            q[j] += step[j] * rng.standard_normal()
            if not (lo_m <= q[1] <= hi_m and lo_a <= q[2] <= hi_a):
                continue
            llq = loglik(model, q[0] + eps, np.exp(q[1]), np.exp(q[2]))
            if np.log(rng.random()) < llq - cur:
                b0, lmb, lal = q; cur = llq; acc[j] += 1
        if it < iters * burn and (it + 1) % 100 == 0:   # adapt steps during burn-in only
            rate = acc / 100
            step *= np.exp(np.clip(rate - 0.35, -0.5, 0.5))
            acc[:] = 0
        mb, al = np.exp(lmb), np.exp(lal)
        tr.append((b0, mb, al, (1 + 1 / al) * mb, cur))
        if (it + 1) % 500 == 0:
            log(f"  iter {it+1}: m_eff={tr[-1][3]:.2f} mbar={mb:.2f} alpha={al:.2f} beta0={b0:.2f} ll={cur:.1f}")
    tr = np.array(tr)
    return tr[int(burn * iters):]


def ess(x):
    """Effective sample size from the initial positive sequence of autocorrelations."""
    x = np.asarray(x) - np.mean(x); n = len(x)
    f = np.fft.rfft(x, 2 * n); ac = np.fft.irfft(f * np.conj(f))[:n]; ac /= ac[0]
    s = 0.0
    for k in range(1, n - 1, 2):
        pair = ac[k] + ac[k + 1]
        if pair < 0:
            break
        s += pair
    return float(n / (1 + 2 * s))


def rhat(chains):
    """Split-free Gelman-Rubin potential scale reduction for chains of equal length."""
    n = min(len(c) for c in chains)
    C = np.array([c[-n:] for c in chains])
    W = C.var(1, ddof=1).mean(); Bv = n * C.mean(1).var(ddof=1)
    return float(np.sqrt(((n - 1) / n * W + Bv / n) / W))


def synthetic_case():
    spec = importlib.util.spec_from_file_location("s13", ROOT / "scripts" / "13_identifiability.py")
    s13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s13)
    X, rec, sec, logc = s13.make(21, 0.0, False)
    kind, sm, st, V = sampling.observations_from_grade(rec, sec)
    return X, kind, sm, st, V, {"m_eff": (1 + 1 / s13.ALPHA) * s13.MBAR, "mbar": s13.MBAR, "alpha": s13.ALPHA}


def lowcount_case():
    """The benchmark creek of run_physical_benchmark.py (seed 11): m̄ 1 mg, median 5 grains per sample."""
    from klondyke import synthetic
    pp = synthetic.PhysParams(seed=11)
    df, _ = synthetic.make_physical_dataset(pp)
    kind, sm, st, V = sampling.observations_from_grade(df.grade_cents_yd3.values, df.dredge_section_ft.values)
    return (df[["s_true", "n_true"]].to_numpy(), kind, sm, st, V,
            {"m_eff": (1 + 1 / pp.alpha) * pp.mbar_mg, "mbar": pp.mbar_mg, "alpha": pp.alpha})


def dominion_case():
    d = pd.read_csv(ROOT / "reports" / "dominion" / "holes_curated.csv")
    kind, sm, st, V = sampling.observations_from_grade(d.c.values, d.ds.values)
    return d[["s", "n"]].to_numpy(), kind, sm, st, V, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--cases", default="synthetic,dominion")
    ap.add_argument("--start", default="mode", choices=["mode", "dispersed"])
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    tag = "" if a.start == "mode" else "_dispersed"
    OUT.mkdir(parents=True, exist_ok=True)
    sp = OUT / "summary.json"
    summ = json.loads(sp.read_text()) if sp.exists() else {}      # keep cases not rerun
    for case in a.cases.split(","):
        X, kind, sm, st, V, truth = {"synthetic": synthetic_case, "lowcount": lowcount_case,
                                     "dominion": dominion_case}[case]()
        t0 = time.time()
        f = physical.CPGField(maxiter=80).fit(X, kind, sm, st, V)
        lap = f.summary()
        log(f"{case}: Laplace m_eff={lap['m_eff']:.2f} mbar={lap['mbar']:.2f} alpha={lap['alpha']:.2f} ({time.time()-t0:.0f}s)")
        tr = run_mcmc(f, a.iters, np.random.default_rng(a.seed), start=a.start)
        np.savez(OUT / f"traces_{case}{tag}.npz", trace=tr, cols=np.array(["beta0", "mbar", "alpha", "m_eff", "loglik"]))
        q = lambda k, c: [float(v) for v in np.quantile(tr[:, k], [0.05, 0.5, 0.95])]
        h = len(tr) // 2
        summ[case + tag] = {"laplace": {k: lap[k] for k in ["m_eff", "mbar", "alpha", "beta0"]},
                      "mcmc_m_eff_q05_50_95": q(3, "m_eff"), "mcmc_mbar_q05_50_95": q(1, "mbar"),
                      "mcmc_alpha_q05_50_95": q(2, "alpha"), "mcmc_beta0_q05_50_95": q(0, "beta0"),
                      "ess_m_eff": ess(np.log(tr[:, 3])), "n_kept": len(tr), "truth": truth,
                      "m_eff_median_first_half": float(np.median(tr[:h, 3])),
                      "m_eff_median_second_half": float(np.median(tr[h:, 3])),
                      "secs": round(time.time() - t0)}
        other = OUT / f"traces_{case}{'' if tag else '_dispersed'}.npz"
        if other.exists():                            # Gelman-Rubin across the two chains
            tr2 = np.load(other)["trace"]
            summ[case + tag]["rhat_log_m_eff"] = rhat([np.log(tr[:, 3]), np.log(tr2[:, 3])])
            w = len(tr) // 4                          # last quarter: after a dispersed start has arrived
            summ[case + tag]["rhat_log_m_eff_last_quarter"] = rhat([np.log(tr[-w:, 3]), np.log(tr2[-w:, 3])])
            pooled = np.r_[tr[-w:, 3], tr2[-w:, 3]]
            summ[case + tag]["pooled_last_quarter_m_eff_q05_50_95"] = [float(v) for v in np.quantile(pooled, [0.05, 0.5, 0.95])]
        log(json.dumps(summ[case + tag], indent=1))
        (OUT / "summary.json").write_text(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
