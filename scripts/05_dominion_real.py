"""Mid-Dominion Creek, real YCGC data: curation, valley frame, model comparison.

Input: data/interim/dominion_mid_browser_extract.csv  (890 records from the YGS map
service, de-duplicated across maps; SHA-256 verified against the browser export).
Coordinates are offsets from (197000, 1040000) in EPSG:3578; thickness and
elevations in feet; grade in cents/yd3.

Curation rules (evidence in DECISIONS.md / docs/SCIENCE.md):
  keep   YCGC26199 (plan 1103, 1936), 26222/26223/26225a/26225b (1955 redraws of the
         1936 plans 1104-1105 drilling, $20.67), 26232 (1958, $20.67)
  drop   code 'A' (digitisers doubt the number is a grade); No.10 thaw/strip layouts
         (26151, 26752: third number is not a grade - see check_thaw_layouts);
         25097 (1960 $35 revaluation of 1103 holes); 26309 (location unknown);
         25086 (1959, price basis unknown); 25084 (1914 prospect shafts)
  fix    85-4: public record duplicates 85-8; restored from van Loon (2016) App. 1
         (Frozen, 893/883 ft, OB 1.52 m = 5 ft, DS 1.83 m = 6 ft, 2.7 c/yd3)
  drop   within-map duplicates; records with dredge section 0

Usage: python scripts/05_dominion_real.py [--variants base,drill,zero,full] [--cv]
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
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from klondyke import baselines as B  # noqa: E402
from klondyke import cv, metrics, models, physical, sampling, units, valley  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "reports" / "dominion"
X0, Y0 = 197000.0, 1040000.0
KEEP = {"26199", "26222", "26223", "26225a", "26225b", "26232"}


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def load():
    d = pd.read_csv(ROOT / "data" / "interim" / "dominion_mid_browser_extract.csv", dtype={"map": str, "hid": str})
    d = d.rename(columns={"map": "mapid"})
    d["x"] += X0; d["y"] += Y0
    d["ambiguous"] = d.code.str.contains("A")
    d["ground"] = d.code.str.extract(r"([FTP])", expand=False)
    return d


def check_thaw_layouts(d):
    """Evidence that the No.10 thaw-layout numbers are not grades: compare with 1103 holes <10 m away."""
    ref = d[d.mapid == "26199"]
    tree = cKDTree(ref[["x", "y"]].to_numpy())
    rows = []
    for m in ["26151", "26752"]:
        t = d[d.mapid == m]
        dist, j = tree.query(t[["x", "y"]].to_numpy())
        close = dist < 10
        for (_, r), jj in zip(t[close].iterrows(), j[close]):
            rows.append((m, r.hid, r.c, ref.iloc[jj].hid, ref.iloc[jj].c))
    return pd.DataFrame(rows, columns=["map", "hid", "value", "hole_1103", "grade_1103"])


def curate(d):
    log_ = []
    n0 = len(d)
    d = d[d.mapid.isin(KEEP)].copy(); log_.append(("keep 1936-derived maps", n0, len(d)))
    n = len(d); d = d[~d.ambiguous]; log_.append(("drop ambiguous", n, len(d)))
    n = len(d)
    d = d.loc[~(d.duplicated(["mapid", "hid", "c", "ob", "ds"]))]; log_.append(("drop within-map duplicates", n, len(d)))
    fix = (d.mapid == "26199") & (d.hid == "85-4")
    if fix.any():
        d.loc[fix, ["ob", "ds", "se", "be", "c", "ground"]] = [5, 6, 893, 883, 2.7, "F"]
        log_.append(("restore 85-4 from van Loon (2016)", int(fix.sum()), int(fix.sum())))
    d.loc[(d.be > d.se), "be"] = np.nan
    n = len(d); d = d[d.ds > 0]; log_.append(("drop zero dredge section", n, len(d)))
    d["claim"] = pd.to_numeric(d.hid.str.extract(r"^(\d+)[A-Z]?-", expand=False), errors="coerce")
    return d.reset_index(drop=True), log_


def frame(d):
    # real YCGC lines: holes ~60 m apart, lines ~150 m apart, not always perpendicular
    # to the valley (they follow claim baselines) -> radius-only graph, 70 m
    poly, lab = valley.reconstruct_frame(d.x.values, d.y.values, radius=70.0, max_angle_deg=90.0)
    s, n = valley.project_to_centreline(d.x.values, d.y.values, poly)
    # orient downstream: claims numbered below discovery increase downstream
    ok = d.claim.notna()
    if np.corrcoef(s[ok], d.claim[ok])[0, 1] < 0:
        poly = poly[::-1]
        s, n = valley.project_to_centreline(d.x.values, d.y.values, poly)
        lab = lab.max() - lab
    off = s.min()
    d = d.assign(s=s - off, n=n, line=lab)
    return d, poly, off


SD_HOLE_FIXED = 0.7   # per-hole log-sd from twin holes on other creeks (reports/twins); not identifiable here


def variant_configs():
    """Model variants. 'drillfix' fixes the per-hole error at the twin-hole value."""
    return {"base": {}, "drill": {"drill_noise": True}, "zero": {"zero_infl": True},
            "full": {"drill_noise": True, "zero_infl": True},
            "drillfix": {"drill_noise": True, "fixed": {"sd_drill": SD_HOLE_FIXED}}}


def fit_variants(X, kind, sm, st, V, variants):
    cfg = variant_configs()
    out = {}
    for v in variants:
        t0 = time.time()
        kw = {k_: v_ for k_, v_ in cfg[v].items() if k_ != "fixed"}
        m = physical.CPGField(maxiter=80, fixed=cfg[v].get("fixed"), **kw).fit(X, kind, sm, st, V)
        sm_ = m.summary(); sm_["secs"] = round(time.time() - t0)
        out[v] = (m, sm_)
        log(f"variant {v}: logZ={sm_['logZ']:.1f} ls_s={sm_['ls_s']:.0f} ls_n={sm_['ls_n']:.0f} "
            f"sigma={sm_['sigma']:.2f} mbar={sm_['mbar']:.3g} alpha={sm_['alpha']:.3g} m_eff={sm_['m_eff']:.3g} "
            f"sd_drill={sm_['sd_drill']:.3f} pi0={sm_['pi0']:.4f} ({sm_['secs']}s)")
    return out


def run_cv(d, X, kind, sm, st, V, best, variant_cfg, seed=0):
    y = np.log1p(units.cents_per_yd3_to_mg_per_m3(d.c.values))
    accum = units.cents_per_yd3_to_mg_per_m3(d.c.values) * d.ds.values * 0.3048
    schemes = {"LOLO": list(cv.grouped_line_folds(d.line.values, n_folds=10, seed=seed)),
               "block1000m": list(cv.along_valley_blocks(d.s.values, 1000.0, 150.0))}
    fs = best.summary()
    rows = []
    rng = np.random.default_rng(seed)
    for sch, splits in schemes.items():
        P = {k: [np.full(len(y), np.nan), np.full(len(y), np.nan)] for k in ["kriging", "gp", "cpg"]}
        crps_cpg = np.full(len(y), np.nan); cov_cpg = []
        latent_cpg = np.full(len(y), np.nan)
        for i, (tr, te) in enumerate(splits):
            t0 = time.time()
            k = B.OrdinaryKriging(anisotropy="auto").fit(X[tr], y[tr]); m_, s_ = k.predict(X[te])
            P["kriging"][0][te], P["kriging"][1][te] = m_, s_
            g = models.AnisotropicGP(n_restarts=0).fit(X[tr], y[tr]); m_, s_ = g.predict(X[te])
            P["gp"][0][te], P["gp"][1][te] = m_, s_
            fixed = {"mbar": fs["mbar"], "alpha": fs["alpha"], **variant_cfg.get("fixed", {})}
            kw = {k_: v_ for k_, v_ in variant_cfg.items() if k_ != "fixed"}
            c = physical.CPGField(maxiter=30, fixed=fixed, **kw).fit(
                X[tr], kind[tr], sm[tr], st[tr], V[tr],
                theta0={k_: fs[k_] for k_ in ("sigma", "ls_s", "ls_n", "beta0", "sd_drill", "pi0") if k_ in fs})
            Sd = c.sample_assays(X[te], V[te], n_draws=600, rng=rng)
            Yd = np.log1p(Sd / V[te][None, :])
            crps_cpg[te] = metrics.crps_samples(y[te], Yd)
            P["cpg"][0][te] = np.median(Yd, 0)
            latent_cpg[te] = c.predict_latent(X[te])[0]
            lo, hi = np.quantile(Yd, [0.05, 0.95], axis=0)
            cov_cpg.append(((y[te] >= lo) & (y[te] <= hi)))
            log(f"{sch} fold {i+1}/{len(splits)} ({time.time()-t0:.0f}s)")
        for name in ["kriging", "gp"]:
            m_, s_ = P[name]
            sc = metrics.score_all(y, m_, s_, y_linear=accum); sc.update(method=name, scheme=sch); rows.append(sc)
        sc = {"crps": float(np.nanmean(crps_cpg)), "mae": float(np.nanmean(np.abs(y - P["cpg"][0]))),
              "spearman": float(spearmanr(y, P["cpg"][0]).statistic),
              "cov90": float(np.concatenate(cov_cpg).mean()),
              "top20": metrics.capture_curve(accum, latent_cpg)["top20"],
              "method": "cpg", "scheme": sch}
        rows.append(sc)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="base,full")
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--cv-variant", default=None, help="variant to cross-validate (default: best by logZ)")
    ap.add_argument("--out-suffix", default="", help="write CV table as cv_results<suffix>.csv")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    d = load()
    thaw = check_thaw_layouts(d)
    thaw.to_csv(OUT / "evidence_thaw_layouts_not_grades.csv", index=False)
    log(f"thaw-layout check: {len(thaw)} co-located pairs; layout values median {thaw.value.median():g}, "
        f"1103 grades median {thaw.grade_1103.median():g}")
    d, steps = curate(d)
    for s_, a0, a1 in steps:
        log(f"  {s_}: {a0} -> {a1}")
    d, poly, off = frame(d)
    log(f"{len(d)} holes, {d.line.nunique()} reconstructed lines, valley {d.s.max():.0f} m, "
        f"zeros {np.mean(d.c==0):.1%}, trace {np.mean(np.isclose(d.c,0.1)):.1%}")
    d.to_csv(OUT / "holes_curated.csv", index=False)
    pd.DataFrame(poly, columns=["x", "y"]).assign(s_offset=off).to_csv(OUT / "axis.csv", index=False)

    kind, sm, st, V = sampling.observations_from_grade(d.c.values, d.ds.values)
    X = d[["s", "n"]].to_numpy()
    res = fit_variants(X, kind, sm, st, V, a.variants.split(","))
    summ = {k: v[1] for k, v in res.items()}
    (OUT / f"model_comparison{a.out_suffix}.json").write_text(json.dumps(summ, indent=2))
    best_name = max(summ, key=lambda k: summ[k]["logZ"])
    log(f"best by marginal likelihood: {best_name}")
    best = res[best_name][0]
    import pickle
    if not a.out_suffix:
        pickle.dump({"model": best, "poly": poly, "off": off}, open(OUT / "best_model.pkl", "wb"))

    if a.cv:
        cv_name = a.cv_variant or best_name
        if cv_name != best_name:
            best = res[cv_name][0]
        r = run_cv(d, X, kind, sm, st, V, best, variant_configs()[cv_name])
        r.to_csv(OUT / f"cv_comparison{a.out_suffix}.csv", index=False)
        print(r.round(3).to_string())


if __name__ == "__main__":
    main()
