"""Which effective grain masses are consistent with the observed spread of drill-vs-dredge errors?

Re-runs the block simulation of 08_error_bound.py for upper Sulphur with the
grain mass scaled (shape alpha fixed, so m_eff scales with mbar), and compares
the predicted robust spread of log(mean-of-holes / block grade) with the
observed one, using a bootstrap over the 9 blocks for the observed uncertainty.

Usage: python scripts/09_meff_from_production.py
Outputs: reports/error_bound/meff_scan.csv, meff_scan.png
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from klondyke import sampling, valley  # noqa: E402

spec = importlib.util.spec_from_file_location("eb", ROOT / "scripts" / "08_error_bound.py")
eb = importlib.util.module_from_spec(spec); spec.loader.exec_module(eb)
s07 = eb._load_module("s07", ROOT / "scripts" / "07_sulphur_dredge_check.py")
OUT = ROOT / "reports" / "error_bound"

h = pd.read_csv(ROOT / "reports" / "sulphur" / "holes_used.csv", low_memory=False)
poly, s, n, lab, r = s07.frame(h); off = s.min()
b = s07.blocks(); hm = pd.read_csv(ROOT / "reports" / "sulphur" / "dredge_blocks.csv")
obs = np.log(hm.holes_mean.values / b.act_c.values)          # log(estimate / recovered)
p0 = json.loads((ROOT / "reports" / "sulphur" / "cpg_fit.json").read_text())

geo = []
for k, rr in enumerate(b.rings):
    G = eb.cells(rr)
    cs, cn = valley.project_to_centreline(G[:, 0], G[:, 1], poly)
    hk = h[h.block == k]
    geo.append((np.column_stack([cs - off, cn]), hk[["s", "n"]].to_numpy(), sampling.sample_volume_m3(hk.dredge_section_ft.values)))

rng = np.random.default_rng(7)
boot = [eb.robust_sd(rng.choice(obs, len(obs))) for _ in range(4000)]
obs_rsd = eb.robust_sd(obs); lo, hi = np.percentile(boot, [5, 95])
rows = []
for meff in [0.3, 1, 3, 10, 20, 40, 80, 160]:
    p = dict(p0); p["mbar"] = p0["mbar"] * meff / p0["m_eff"]
    errs = []
    for cell_sn, hole_sn, V in geo:
        e, _, _ = eb.simulate_block(cell_sn, hole_sn, V, p, n_sims=800)
        errs.append(e - np.median(e))
    pooled = np.concatenate(errs)
    rows.append({"m_eff_mg": meff, "pred_robust_sd": eb.robust_sd(pooled), "pred_sd": float(pooled.std()),
                 "pred_q90_abs": float(np.percentile(np.abs(pooled), 90))})
    print(rows[-1], flush=True)
res = pd.DataFrame(rows)
res["obs_robust_sd"] = obs_rsd; res["obs_lo90"] = lo; res["obs_hi90"] = hi
res.to_csv(OUT / "meff_scan.csv", index=False)
ok = res[(res.pred_robust_sd >= lo) & (res.pred_robust_sd <= hi)]
print(f"observed robust sd {obs_rsd:.3f} (90% bootstrap {lo:.3f}-{hi:.3f}); consistent m_eff: {ok.m_eff_mg.tolist()}")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.semilogx(res.m_eff_mg, res.pred_robust_sd, "o-", color="#d95f02", label="predicted (spatial + grain counting)")
ax.axhspan(lo, hi, color="k", alpha=0.15, label="observed, 90% bootstrap band")
ax.axhline(obs_rsd, color="k", lw=1.5, label="observed (9 dredge blocks)")
ax.axvline(p0["m_eff"], color="#d95f02", ls=":", label=f"assay-only estimate ({p0['m_eff']:.0f} mg)")
ax.set_xlabel("effective grain mass m_eff (mg)"); ax.set_ylabel("robust sd of log(drill estimate / block)")
ax.set_title("Upper Sulphur: grain mass consistent with production"); ax.legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(OUT / "meff_scan.png", dpi=140)
