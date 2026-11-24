"""How many holes does a block need, and when does grain counting matter?

For n holes in one block (ignoring spatial variation inside the block), the
estimate's log error has variance

    Var[log(mean of n assays / c)] ≈ κ² / n,   κ² = (1 + h²)(1 + g²) − 1      (delta method)

with per-hole relative variance h² = exp(σ_h²) − 1 (log-normal factor of mean
one) and grain-counting relative variance g² = m_eff / (c V). Hence:
  * grain counting dominates only while a sample holds less gold than
    cV* = m_eff / h²  (the crossover; ≈ 16 mg for m_eff 10 mg, σ_h 0.7);
  * the holes needed for a relative precision s are n ≈ κ² / s².
The script checks the formula by Monte Carlo (exact grain draws, log-normal
per-hole error, median-unbiased convention) and draws the regime map with the
settings of the creeks studied here.

Usage: python scripts/15_design.py
Outputs: reports/design/holes_needed.csv, mc_check.csv, regime.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from klondyke import sampling, units  # noqa: E402

OUT = ROOT / "reports" / "design"
RNG = np.random.default_rng(15)


def kappa2(sd_hole, m_eff, gold_mg):
    h2 = np.expm1(sd_hole ** 2); g2 = m_eff / gold_mg
    return (1 + h2) * (1 + g2) - 1


def approx_var(n, sd_hole, m_eff, gold_mg):
    return kappa2(sd_hole, m_eff, gold_mg) / n


def mc_var(n, sd_hole, mbar, alpha, c, V, reps=20000):
    e = RNG.normal(0, sd_hole, (reps, n))
    S, _ = sampling.simulate(c * np.exp(e - sd_hole ** 2 / 2), V, mbar, alpha, RNG)   # mean-one per-hole factor
    est = (S / V).mean(1)
    ok = est > 0
    r = np.log(est[ok] / c)
    q1, q3 = np.percentile(r, [25, 75])
    return float(r.var()), float(((q3 - q1) / 1.349) ** 2), float(1 - ok.mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Monte Carlo check of the approximation
    rows = []
    for m_eff, alpha in [(10.0, 0.6), (50.0, 0.3)]:
        mbar = m_eff / (1 + 1 / alpha)
        for gold in [5.0, 20.0, 100.0, 500.0]:
            V = sampling.sample_volume_m3(20.0); c = gold / V
            for sd_h in [0.0, 0.7]:
                for n in [4, 16, 64]:
                    v_mc, v_rob, p0 = mc_var(n, sd_h, mbar, alpha, c, V)
                    rows.append({"m_eff": m_eff, "gold_mg_per_sample": gold, "sd_hole": sd_h, "n": n,
                                 "var_formula": approx_var(n, sd_h, m_eff, gold), "var_mc": v_mc,
                                 "var_mc_robust": v_rob, "p_all_zero": p0})
    mc = pd.DataFrame(rows)
    mc["ratio_mc_over_formula"] = mc.var_mc / mc.var_formula
    mc.to_csv(OUT / "mc_check.csv", index=False)
    print(mc.round(3).to_string())

    # holes needed for ±20% and ±30% (one sd on the log scale)
    tab = []
    cases = {
        "mid-Dominion pay (20 c/yd³, 20 ft, m_eff 10 mg)": (20.0, 20.0, 10.0),
        "mid-Dominion lean (5 c/yd³, 20 ft, m_eff 10 mg)": (5.0, 20.0, 10.0),
        "upper Sulphur bench (5 c/yd³, 7 ft, m_eff 50 mg)": (5.0, 7.0, 50.0),
        "upper Sulphur bench, rich (20 c/yd³, 7 ft, m_eff 50 mg)": (20.0, 7.0, 50.0),
        "Gold Run pay (34 c/yd³, 15 ft, m_eff 45 mg)": (34.0, 15.0, 45.0),
    }
    for name, (cents, ft, m_eff) in cases.items():
        V = sampling.sample_volume_m3(ft); gold = units.cents_per_yd3_to_mg_per_m3(cents) * V
        for sd_h in (0.6, 0.9):
            per_hole = kappa2(sd_h, m_eff, gold)
            h2, g2 = np.expm1(sd_h ** 2), m_eff / gold
            tab.append({"case": name, "gold_mg_per_sample": round(gold, 1), "sd_hole": sd_h,
                        "grain_share": round(g2 / (g2 + h2), 2),
                        "crossover_gold_mg": round(m_eff / h2, 1),
                        "holes_for_20pct": int(np.ceil(per_hole / 0.2 ** 2)),
                        "holes_for_30pct": int(np.ceil(per_hole / 0.3 ** 2))})
    tab = pd.DataFrame(tab)
    tab.to_csv(OUT / "holes_needed.csv", index=False)
    print(tab.to_string(index=False))
    plot(tab)


def plot(tab):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gold = np.logspace(0, 3.3, 200)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for m_eff, ls in [(10, "-"), (50, "--")]:
        for sd_h, col in [(0.6, "tab:blue"), (0.9, "tab:red")]:
            n20 = kappa2(sd_h, m_eff, gold) / 0.04
            ax.plot(gold, n20, ls, color=col, label=f"m_eff {m_eff} mg, σ_h {sd_h}")
        ax.axvline(m_eff / np.expm1(0.7 ** 2), color="grey", ls=ls, lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("gold in one drill sample (mg) = grade × sample volume")
    ax.set_ylabel("holes per block for ±20% (1 sd)")
    ax.set_title("Grain counting dominates only left of the crossover (grey, σ_h = 0.7)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "regime.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
