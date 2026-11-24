"""Check every quantitative statement in the manuscript against the analysis outputs.

Each entry pairs a number as printed in klondike_placer_drilling.tex with the
value recomputed from data/ and reports/. A number passes if the recomputed
value rounds to the printed one. Run after run_pipeline.bat:

    python paper/check_numbers.py

Exit code 1 if any check fails.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
sys.path.insert(0, str(ROOT / "src"))
from klondyke import sampling, units  # noqa: E402

RESULTS = []


def chk(section, label, printed, value, nd=0, scale=1.0):
    """Pass if value*scale rounds (half away from zero, nd decimals) to the printed number."""
    v = float(value) * scale
    tol = 0.5 * 10 ** (-nd) + 1e-9
    ok = abs(v - printed) <= tol
    RESULTS.append((section, label, printed, round(v, nd + 2), ok))


def js(p):
    return json.loads((R / p).read_text())


# ------------------------------------------------------------------ data
def data_section():
    h = pd.read_csv(ROOT / "data" / "interim" / "ycgc_holes.csv", low_memory=False)
    u = h[h.usable]
    S = "2 data"
    chk(S, "point records", 13358, len(h)); chk(S, "source maps with points", 340, h.map_id.nunique())
    pt = h.point_type.value_counts()
    chk(S, "drill holes", 12447, pt["drill hole"]); chk(S, "shafts", 670, pt["shaft"])
    chk(S, "bedrock test holes", 231, pt["bedrock test hole"])
    chk(S, "records with a grade", 8630, h.grade_cents_raw.notna().sum())
    chk(S, "records with bedrock elevation", 2720, h.bedrock_elev_ft.notna().sum())
    chk(S, "dredge-cut and thaw polygons", 5352, len(json.loads((ROOT / "raw" / "ycgc_layer1_dredge_limits_cuts.json").read_text())["features"]))
    chk(S, "other-workings polygons", 2057, len(json.loads((ROOT / "raw" / "ycgc_layer2_other_workings.json").read_text())["features"]))
    chk(S, "map index entries", 933, len(json.loads((ROOT / "raw" / "ycgc_layer3_map_index.json").read_text())["features"]))
    chk(S, "thickness inconsistent >35%", 209, h.flag_thickness_inconsistent.fillna(False).sum())
    chk(S, "price recorded, % of holes", 14, h.gold_price_recorded.notna().mean(), 0, 100)
    chk(S, "redrawn duplicates removed", 667, h.flag_duplicate.fillna(False).sum())
    chk(S, "grades marked doubtful", 835, h.flag_grade_ambiguous.fillna(False).sum())
    chk(S, "usable holes", 6179, len(u)); chk(S, "usable maps", 122, u.map_id.nunique())
    chk(S, "first year", 1907, u.map_year.min()); chk(S, "last year", 1961, u.map_year.max())
    chk(S, "median section ft", 14.5, u.dredge_section_ft.median(), 1)
    chk(S, "median grade c/yd3", 9.1, u.grade_cents_2067.median(), 1)
    chk(S, "zero share %", 6.7, (u.grade_cents_raw == 0).mean(), 1, 100)
    ok = h.grade_cents_raw.gt(0.5) & h.x.notna() & ~h.flag_not_drillhole.fillna(False)
    d = h[ok].reset_index(drop=True)
    P = np.array(sorted(cKDTree(d[["x", "y"]].values).query_pairs(25.0)))
    a, b = d.iloc[P[:, 0]], d.iloc[P[:, 1]]
    diff = a.map_id.values != b.map_id.values
    g1, g2 = a.grade_cents_raw.values[diff], b.grade_cents_raw.values[diff]
    r = np.maximum(g1, g2) / np.minimum(g1, g2)
    chk(S, "cross-map pairs <25 m", 1982, diff.sum()); chk(S, "identical grades", 721, np.sum(np.abs(r - 1) < 0.005))
    chk(S, "ratio 1.690", 68, np.sum(np.abs(r - 35 / 20.67) < 0.01)); chk(S, "ratio 1.863", 75, np.sum(np.abs(r - 38.5 / 20.67) < 0.01))
    dom = pd.read_csv(R / "dominion" / "holes_curated.csv")
    chk(S, "mid-Dominion holes", 612, len(dom)); chk(S, "mid-Dominion lines", 71, dom.line.nunique())
    chk(S, "mid-Dominion zeros %", 11.8, (dom.c == 0).mean(), 1, 100)
    chk(S, "mid-Dominion valley km", 12.6, np.ptp(dom.s) / 1000, 1)
    sul = pd.read_csv(R / "sulphur" / "holes_used.csv", low_memory=False)
    chk(S, "upper Sulphur holes", 376, len(sul)); chk(S, "upper Sulphur maps", 9, sul.map_id.nunique())
    camp = pd.read_csv(R / "regional" / "grain_by_map.csv")
    chk(S, "campaigns", 33, len(camp)); chk(S, "campaign min holes", 62, camp.n.min()); chk(S, "campaign max holes", 393, camp.n.max())


# ------------------------------------------------------------------ synthetic
def synthetic_section():
    S = "5 simulation"
    d = pd.concat([pd.read_csv(R / f"physical_benchmark_seed{s}.csv") for s in (11, 12, 13)]).dropna(subset=["rmse"])
    g = d[d.scheme == "LOLO"].groupby("method")
    m, sd = g.mean(numeric_only=True), g.std(numeric_only=True)
    for meth, vals in {"kriging_loggrade": (0.848, 0.015, -0.34, 0.508, 0.019, 0.97, 0.70, 0.31, 0.02),
                       "gp_loggrade": (0.837, 0.012, -0.32, 0.460, 0.006, 0.91, 0.68, 0.33, 0.06),
                       "cpg_field": (0.719, 0.004, -0.10, 0.411, 0.003, 0.96, 0.69, 0.35, 0.08)}.items():
        rm, rs, bi, cr, cs, cv, tg, rh, rhs = vals
        chk(S, f"{meth} RMSE", rm, m.loc[meth, "rmse"], 3); chk(S, f"{meth} RMSE sd", rs, sd.loc[meth, "rmse"], 3)
        chk(S, f"{meth} bias", bi, m.loc[meth, "bias"], 2); chk(S, f"{meth} CRPS", cr, m.loc[meth, "crps"], 3)
        chk(S, f"{meth} CRPS sd", cs, sd.loc[meth, "crps"], 3); chk(S, f"{meth} cov90", cv, m.loc[meth, "cov90"], 2)
        chk(S, f"{meth} top20", tg, m.loc[meth, "top20_gold"], 2)
        chk(S, f"{meth} rho in channel", rh, m.loc[meth, "spearman_in_channel"], 2)
        chk(S, f"{meth} rho sd", rhs, sd.loc[meth, "spearman_in_channel"], 2)
    bc = pd.read_csv(R / "baseline_check.csv")
    gb = bc.groupby("method")
    bm, bs = gb.mean(numeric_only=True), gb.std(numeric_only=True)
    for meth, (rm, rs, bi, bis, cr, cs, cv) in {"kriging + nugget/2": (0.832, 0.037, -0.27, 0.13, 0.501, 0.031, 0.98),
                                                 "GP + noise/2": (0.798, 0.042, 0.19, 0.12, 0.453, 0.027, 0.92)}.items():
        chk(S, f"{meth} RMSE", rm, bm.loc[meth, "rmse"], 3); chk(S, f"{meth} RMSE sd", rs, bs.loc[meth, "rmse"], 3)
        chk(S, f"{meth} bias", bi, bm.loc[meth, "bias"], 2); chk(S, f"{meth} bias sd", bis, bs.loc[meth, "bias"], 2)
        chk(S, f"{meth} CRPS", cr, bm.loc[meth, "crps"], 3); chk(S, f"{meth} CRPS sd", cs, bs.loc[meth, "crps"], 3)
        chk(S, f"{meth} cov90", cv, bm.loc[meth, "cov90"], 2)
    chk(S, "RMSE gain vs kriging %", 15, 1 - m.loc["cpg_field", "rmse"] / m.loc["kriging_loggrade", "rmse"], 0, 100)
    chk(S, "RMSE gain vs GP + noise/2 %", 10, 1 - m.loc["cpg_field", "rmse"] / bm.loc["GP + noise/2", "rmse"], 0, 100)
    me = [js(f"physical_benchmark_seed{s}.json")["cpg_full_fit"]["m_eff"] for s in (11, 12, 13)]
    chk(S, "benchmark m_eff min", 0.8, min(me), 1); chk(S, "benchmark m_eff max", 1.2, max(me), 1)
    idf = pd.read_csv(R / "identifiability" / "summary.csv").set_index(["sd_hole_true", "design", "model"])
    rows = {(0.0, "lines", "grain only"): (8.2, 0.9, 0.62), (0.0, "lines", "grain + per-hole"): (8.2, 0.9, 0.62),
            (0.0, "twins", "grain only"): (9.4, 0.6, 0.59), (0.0, "twins", "grain + per-hole"): (9.2, 0.5, 0.59),
            (0.8, "lines", "grain only"): (8.4, 3.3, 0.83), (0.8, "lines", "grain + per-hole"): (8.1, 3.1, 0.70),
            (0.8, "twins", "grain only"): (11.3, 1.9, 0.80), (0.8, "twins", "grain + per-hole"): (8.1, 0.1, 0.68)}
    for k, (me_, msd, rm) in rows.items():
        chk(S, f"ident {k} m_eff", me_, idf.loc[k, "m_eff_mean"], 1); chk(S, f"ident {k} m_eff sd", msd, idf.loc[k, "m_eff_sd"], 1)
        chk(S, f"ident {k} RMSE", rm, idf.loc[k, "rmse_latent"], 2)
    for k, v in {(0.0, "lines"): 0.0, (0.0, "twins"): 0.1, (0.8, "lines"): 0.4, (0.8, "twins"): 8.2}.items():
        chk(S, f"ident {k} dlogZ", v, idf.loc[(*k, "grain only"), "dlogZ"], 1)
    for k, v in {(0.0, "lines"): 0.08, (0.0, "twins"): 0.15, (0.8, "lines"): 0.58, (0.8, "twins"): 0.68}.items():
        chk(S, f"ident {k} sd_hat", v, idf.loc[(*k, "grain + per-hole"), "sd_hat"], 2)
    chk(S, "RMSE cost of omitting per-hole %", 18,
        idf.loc[(0.8, "lines", "grain only"), "rmse_latent"] / idf.loc[(0.8, "lines", "grain + per-hole"), "rmse_latent"] - 1, 0, 100)


# ------------------------------------------------------------------ Dominion
def dominion_section():
    S = "6.1 Dominion"
    mc = js("dominion/model_comparison.json")
    b = mc["base"]
    chk(S, "logZ base", -2775.0, b["logZ"], 1); chk(S, "ls_s", 130, b["ls_s"]); chk(S, "ls_n", 33, b["ls_n"])
    chk(S, "m_eff", 10.5, b["m_eff"], 1); chk(S, "mbar", 3.85, b["mbar"], 2); chk(S, "alpha", 0.58, b["alpha"], 2)
    chk(S, "logZ + per-hole", -2775.0, mc["drill"]["logZ"], 1); chk(S, "sd_hat per-hole", 0.003, mc["drill"]["sd_drill"], 3)
    df = js("dominion/model_comparison_drillfix.json")["drillfix"]
    chk(S, "logZ sd_h=0.7", -2778.6, df["logZ"], 1); chk(S, "ls_s sd_h=0.7", 214, df["ls_s"])
    chk(S, "ls_n sd_h=0.7", 49, df["ls_n"]); chk(S, "m_eff sd_h=0.7", 10.3, df["m_eff"], 1)
    chk(S, "cost of sd_h=0.7", 3.5, b["logZ"] - df["logZ"], 1)
    rp = pd.read_csv(R / "robustness" / "dominion_profiles.csv")
    sdp = rp[rp.check == "sd_hole_profile"].set_index("value")
    for v, me in {0.0: 10.5, 0.3: 10.0, 0.6: 10.0, 0.8: 10.5, 1.0: 8.4, 1.2: 6.1}.items():
        chk(S, f"m_eff at sd_h={v}", me, sdp.loc[v, "m_eff"], 1)
    pp = rp[rp.check == "pi0_profile"].set_index("value")
    chk(S, "logZ pi0=0.01", -2785.8, pp.loc[0.01, "logZ"], 1); chk(S, "m_eff pi0=0.01", 11.5, pp.loc[0.01, "m_eff"], 1)
    chk(S, "logZ pi0=0.1", -2808.2, pp.loc[0.1, "logZ"], 1); chk(S, "m_eff pi0=0.1", 10.8, pp.loc[0.1, "m_eff"], 1)
    chk(S, "ls pi0=0.01", 146, pp.loc[0.01, "ls_s"]); chk(S, "ls pi0=0.1", 131, pp.loc[0.1, "ls_s"])
    chk(S, "min m_eff over pi0 profile", 10.3, pp.m_eff.min(), 1); chk(S, "max m_eff over pi0 profile", 11.5, pp.m_eff.max(), 1)
    chk(S, "drop pi0 0 -> 0.01", 10.7, pp.loc[0.0, "logZ"] - pp.loc[0.01, "logZ"], 1)
    fg = rp[rp.check == "fine_gold_with_barren_holes"].iloc[0]
    chk(S, "fine-gold m_eff", 2.3, fg.m_eff, 1); chk(S, "fine-gold gap at pi0=0.1", 9.4, pp.loc[0.1, "logZ"] - fg.logZ, 1)
    cvd = pd.read_csv(R / "dominion" / "cv_comparison.csv").set_index(["method", "scheme"])
    exp = {("kriging", "LOLO"): (1.210, 0.34, 0.88, 27.8), ("gp", "LOLO"): (1.189, 0.32, 0.86, 32.1),
           ("cpg", "LOLO"): (1.115, 0.38, 0.915, 34.9), ("kriging", "block1000m"): (1.301, 0.06, 0.87, 25.5),
           ("gp", "block1000m"): (1.279, -0.02, 0.85, 25.0), ("cpg", "block1000m"): (1.190, 0.13, 0.913, 27.9)}
    for k, (c, rho, cov, top) in exp.items():
        chk(S, f"CV {k} CRPS", c, cvd.loc[k, "crps"], 3); chk(S, f"CV {k} rho", rho, cvd.loc[k, "spearman"], 2)
        chk(S, f"CV {k} cov90", cov, cvd.loc[k, "cov90"], 3 if cov in (0.915, 0.913) else 2)
        chk(S, f"CV {k} top20 %", top, cvd.loc[k, "top20"], 1, 100)
    cvf = pd.read_csv(R / "dominion" / "cv_comparison_drillfix.csv").set_index(["method", "scheme"])
    k = ("cpg", "LOLO")
    chk(S, "CV sd_h=0.7 CRPS", 1.107, cvf.loc[k, "crps"], 3); chk(S, "CV sd_h=0.7 rho", 0.38, cvf.loc[k, "spearman"], 2)
    chk(S, "CV sd_h=0.7 cov90", 0.913, cvf.loc[k, "cov90"], 3); chk(S, "CV sd_h=0.7 top20 %", 34.1, cvf.loc[k, "top20"], 1, 100)
    t = js("dominion/targets_summary.json")
    chk(S, "floor mined %", 45, t["share_of_drilled_floor_mined"], 0, 100)
    chk(S, "drill gold in mined ground %", 78, t["share_of_drill_gold_inside_mined_ground"], 0, 100)
    chk(S, "target rank correlation", 0.94, js("robustness/summary.json")["targets"]["spearman_P_unmined"], 2)


# ------------------------------------------------------------------ twins
def twins_section():
    S = "6.2 twins"
    h = pd.read_csv(ROOT / "data" / "interim" / "ycgc_holes.csv", low_memory=False)
    ok = (h.grade_cents_raw.notna() & h.x.notna() & h.dredge_section_ft.gt(0) & ~h.flag_not_drillhole.fillna(False)
          & ~h.flag_grade_ambiguous.fillna(False) & ~h.map_title.fillna("").str.contains("thaw|strip|progress|workings", case=False))
    d = h[ok].reset_index(drop=True)
    P = np.array(sorted(cKDTree(d[["x", "y"]].values).query_pairs(12.0)))
    a, b = d.iloc[P[:, 0]].reset_index(drop=True), d.iloc[P[:, 1]].reset_index(drop=True)
    same = a.map_id.values == b.map_id.values
    double = (np.abs(a.grade_cents_raw - b.grade_cents_raw).values < 0.05) & (np.abs(a.dredge_section_ft - b.dredge_section_ft).values < 0.05)
    sid = (a.hole_id.astype(str).values == b.hole_id.astype(str).values) & a.hole_id.notna().values
    chk(S, "pairs <12 m", 706, len(P)); chk(S, "same map", 85, same.sum())
    chk(S, "double records", 5, (same & double).sum()); chk(S, "same hole number", 6, (same & sid).sum())
    p = pd.read_csv(R / "twins" / "pairs.csv")
    vc = p.map_id.value_counts()
    chk(S, "twin pairs", 78, len(p)); chk(S, "maps", 13, p.map_id.nunique()); chk(S, "Bear Creek pairs", 38, vc["YCGC26009"])
    chk(S, "Gold Run pairs", 15, vc["YCGC26341"]); chk(S, "lower Hunker pairs", 9, vc["YCGC26438"])
    chk(S, "median distance m", 8.3, p.dist_m.median(), 1); chk(S, "pairs with gold in both", 72, p.both_pos.sum())
    s = js("twins/summary.json")
    chk(S, "median |log ratio|", 1.12, s["median_abs_logratio"], 2); chk(S, "median factor", 3.06, np.exp(s["median_abs_logratio"]), 2)
    chk(S, "robust variance", 2.87, s["robust_var_logratio"], 2)
    chk(S, "robust variance, no old workings", 2.59, s["robust_var_logratio_no_old_workings"], 2)
    chk(S, "pairs excluded for old workings", 11, s["n_both_positive"] - s["n_no_old_workings"])
    chk(S, "ratio to 10 mg (text)", 13, s["robust_var_logratio"] / s["simulated_robust_var"]["grains only, m_eff 10 mg (mid-Dominion fit)"])
    for i, v in enumerate([0.58, 2.45, 5.14]):
        chk(S, f"observed tercile {i}", v, s["by_gold_tercile"][i]["robust_var"], 2)
    for i, (lo, hi) in enumerate([(6, 88), (88, 181), (181, 1259)]):
        chk(S, f"tercile {i} lower mg", lo, s["by_gold_tercile"][i]["gold_mg_range"][0]); chk(S, f"tercile {i} upper mg", hi, s["by_gold_tercile"][i]["gold_mg_range"][1])
    sims = {"grains only, m_eff 10 mg (mid-Dominion fit)": (0.23, 0.81, 0.22, 0.09),
            "grains only, m_eff 90 mg (coarse, Sulphur)": (1.59, 3.99, 2.27, 0.86),
            "grains m_eff 10 mg + per-hole sd 0.8": (1.69, 2.17, 1.67, 1.39)}
    for k, (tot, t1, t2, t3) in sims.items():
        chk(S, f"sim {k[:22]} all", tot, s["simulated_robust_var"][k], 2)
        for i, v in enumerate((t1, t2, t3)):
            chk(S, f"sim {k[:22]} tercile {i}", v, s["simulated_by_tercile"][k][i], 2)
    u = js("twins/uncertainty.json")
    for key, (n, nm, rv, rvlo, rvhi, r10, r10lo, r10hi, r90, r90lo, r90hi, sl, sllo, slhi, pp) in {
            "all": (72, 11, 2.87, 0.97, 4.34, 12.7, 3.7, 23.5, 1.8, 0.6, 2.9, 0.26, -0.04, 0.60, 0.045),
            "without_bear_creek": (37, 10, 2.20, 0.62, 4.54, 11.4, 2.2, 27.6, 1.4, 0.4, 3.2, 0.17, -0.13, 0.88, 0.12),
            "without_old_workings": (61, 11, 2.59, 0.97, 4.33, 13.7, 4.0, 24.7, 1.8, 0.6, 3.0, 0.30, -0.03, 0.60, 0.09)}.items():
        x = u[key]
        chk(S, f"{key} n", n, x["n_pairs"]); chk(S, f"{key} maps", nm, x["n_maps"])
        chk(S, f"{key} robust var", rv, x["robust_var_obs"], 2)
        chk(S, f"{key} rv lo", rvlo, x["robust_var_obs_ci90"][0], 2); chk(S, f"{key} rv hi", rvhi, x["robust_var_obs_ci90"][1], 2)
        chk(S, f"{key} ratio10", r10, x["ratio_obs_to_10mg"], 1)
        chk(S, f"{key} ratio10 lo", r10lo, x["ratio_obs_to_10mg_ci90"][0], 1); chk(S, f"{key} ratio10 hi", r10hi, x["ratio_obs_to_10mg_ci90"][1], 1)
        chk(S, f"{key} ratio90", r90, x["ratio_obs_to_90mg"], 1)
        chk(S, f"{key} ratio90 lo", r90lo, x["ratio_obs_to_90mg_ci90"][0], 1); chk(S, f"{key} ratio90 hi", r90hi, x["ratio_obs_to_90mg_ci90"][1], 1)
        chk(S, f"{key} slope", sl, x["fe_slope_log_abs_lr_on_log_gold"], 2)
        chk(S, f"{key} slope lo", sllo, x["fe_slope_ci90"][0], 2); chk(S, f"{key} slope hi", slhi, x["fe_slope_ci90"][1], 2)
        chk(S, f"{key} perm p", pp, x["within_map_permutation_p"], 3 if pp == 0.045 else 2)
    sm = u["sim_fe_slope_by_m_eff"]
    chk(S, "grain-only slope, most negative mean", -0.51, min(v["mean"] for v in sm.values()), 2)
    chk(S, "grain-only slope, least negative mean", -0.38, max(v["mean"] for v in sm.values()), 2)
    chk(S, "grain-only slope, max 95th pct", 0.04, max(v["q95"] for v in sm.values()), 2)
    chk(S, "Bear Creek pairs with gold in both", 35, u["all"]["n_pairs"] - u["without_bear_creek"]["n_pairs"])
    ff = pd.read_csv(R / "twins" / "field_fits.csv")
    piv = ff.pivot(index="map_id", columns="model", values="logZ")
    dz = sorted((piv["grain + per-hole"] - piv["grain only"]).round(1).tolist())
    for v, w in zip([1.7, 2.2, 9.0, 15.7], dz):
        chk(S, "field-fit dlogZ", v, w, 1)
    sdv = ff[ff.model == "grain + per-hole"].sd_drill
    chk(S, "sd_hat min", 0.62, sdv.min(), 2); chk(S, "sd_hat max", 1.23, sdv.max(), 2)
    g = ff[ff.model == "grain only"].set_index("map_id"); hh = ff[ff.model == "grain + per-hole"].set_index("map_id")
    short = g.ls_n[g.ls_n < 20]
    chk(S, "maps with short ls_n", 3, len(short)); chk(S, "short ls_n min", 5, short.min()); chk(S, "short ls_n max", 14, short.max())
    chk(S, "ls_n with per-hole min", 16, hh.ls_n.min()); chk(S, "ls_n with per-hole max", 934, hh.ls_n.max())
    chk(S, "Gold Run m_eff grains", 88, g.loc["YCGC26341", "m_eff"]); chk(S, "Gold Run m_eff per-hole", 45, hh.loc["YCGC26341", "m_eff"])
    chk(S, "Hunker m_eff grains", 59, g.loc["YCGC26438", "m_eff"]); chk(S, "Hunker m_eff per-hole", 4.5, hh.loc["YCGC26438", "m_eff"], 1)
    rb = js("robustness/summary.json")["twins"]
    chk(S, "Hunker series offset", -0.09, rb["series_offset_YCGC26438"]["median"], 2)
    chk(S, "Hunker positive pairs", 4, rb["series_offset_YCGC26438"]["share_positive"] * rb["series_offset_YCGC26438"]["n"])
    chk(S, "Gold Run series offset", 0.56, rb["series_offset_YCGC26341"]["median"], 2)
    chk(S, "Gold Run positive pairs", 7, rb["series_offset_YCGC26341"]["share_positive"] * rb["series_offset_YCGC26341"]["n"])
    pos = p[p.both_pos]
    chk(S, "median gold per pair mg", 140, pos.gold_mg_pair.median(), -1)


# ------------------------------------------------------------------ regional
def regional_section():
    S = "6.3 campaigns"
    s = pd.read_csv(R / "regional" / "grain_profiles_summary.csv")
    chk(S, "closed intervals", 29, s.closed.sum()); chk(S, "profile depth min", 17, s.range_logZ.min())
    chk(S, "profile depth max", 285, s.range_logZ.max())
    chk(S, "depth vs info rho", 0.78, spearmanr(s.range_logZ, s.I_zero).statistic, 2)
    chk(S, "mbar min", 0.1, s.mbar_best.min(), 1); chk(S, "mbar max", 10, s.mbar_best.max())
    dmb = s[s.creek == "Dominion"].mbar_best
    chk(S, "Dominion campaigns", 6, len(dmb)); chk(S, "Dominion mbar min", 0.1, dmb.min(), 1); chk(S, "Dominion mbar max", 10, dmb.max())
    for c, (rho, pv) in {"median_ds_ft": (-0.49, 0.004), "zero_share": (0.54, 0.001), "dist_ksd_km": (-0.09, 0.62)}.items():
        r = spearmanr(s.mbar_best, s[c])
        chk(S, f"rho mbar vs {c}", rho, r.statistic, 2); chk(S, f"p mbar vs {c}", pv, r.pvalue, 3 if pv < 0.01 else 2)
    ca = pd.read_csv(R / "robustness" / "campaign_alternatives.csv").set_index(["map_id", "model"])
    chk(S, "26222 holes", 149, ca.loc[("YCGC26222", "grains"), "n"])
    chk(S, "26222 m_eff grains", 20, ca.loc[("YCGC26222", "grains"), "m_eff"])
    chk(S, "26222 m_eff alt", 1.4, ca.loc[("YCGC26222", "grains + per-hole 0.7 + zeros"), "m_eff"], 1)
    chk(S, "26222 dlogZ", 1.6, ca.loc[("YCGC26222", "grains + per-hole 0.7 + zeros"), "logZ"] - ca.loc[("YCGC26222", "grains"), "logZ"], 1)
    a = pd.read_csv(R / "regional" / "grain_by_map.csv").set_index("map_id")
    b = pd.read_csv(R / "regional" / "grain_by_map_sdhole0.7.csv").set_index("map_id")
    chk(S, "26681 holes", 393, a.loc["YCGC26681", "n"]); chk(S, "26681 m_eff grains", 40, a.loc["YCGC26681", "m_eff"])
    chk(S, "26681 m_eff per-hole", 3.5, b.loc["YCGC26681", "m_eff"], 1)
    chk(S, "26681 dlogZ", 17, b.loc["YCGC26681", "logZ"] - a.loc["YCGC26681", "logZ"])
    up = [a.loc[m, "m_eff"] for m in ["YCGC26743", "YCGC26741", "YCGC26683"]]
    chk(S, "upper Sulphur m_eff min", 29, min(up)); chk(S, "upper Sulphur m_eff max", 84, max(up))


# ------------------------------------------------------------------ production, design, MCMC
def production_section():
    S = "6.4 Sulphur"
    b = pd.read_csv(R / "sulphur" / "dredge_blocks.csv")
    chk(S, "value recovered above estimate %", 23, b.VALUE.sum() / b.ESTIMATED_VALUE.sum() - 1, 0, 100)
    chk(S, "gravel above estimate %", 9, b.YARDAGE_SLUICED.sum() / b.ESTIMATED_YARDAGE.sum() - 1, 0, 100)
    chk(S, "min yardage Myd3", 0.39, min(b.ESTIMATED_YARDAGE.min(), b.YARDAGE_SLUICED.min()) / 1e6, 2)
    chk(S, "max yardage Myd3", 0.61, max(b.ESTIMATED_YARDAGE.max(), b.YARDAGE_SLUICED.max()) / 1e6, 2)
    chk(S, "recovered grade min", 17, b.act_c.min()); chk(S, "recovered grade max", 38, b.act_c.max())
    chk(S, "YCGC estimate / holes mean", 0.5, b.est_c.sum() / b.holes_mean.sum(), 1)
    for tag, col in [("", "primary"), ("_radius40", "radius 40"), ("_price35", "price 35")]:
        f = R / "sulphur" / f"dredge_check{tag}.csv"
        if not f.exists():
            RESULTS.append((S, f"{col}: file missing", np.nan, np.nan, False)); continue
        d = pd.read_csv(f).set_index("estimator")
        exp = {"": {"YCGC recorded estimate": (0.37, -12), "mean of holes in block": (0.64, 74),
                    "kriging (leave-block-out)": (0.65, 75), "grain-counting field (leave-block-out)": (0.62, 70)},
               "_radius40": {"YCGC recorded estimate": (0.37, -12), "mean of holes in block": (0.64, 74),
                             "kriging (leave-block-out)": (1.09, -48), "grain-counting field (leave-block-out)": (0.42, 29)},
               "_price35": {"YCGC recorded estimate": (0.37, -12), "mean of holes in block": (0.62, 70),
                            "kriging (leave-block-out)": (0.43, -10), "grain-counting field (leave-block-out)": (0.56, 55)}}[tag]
        for est, (rm, bi) in exp.items():
            chk(S, f"{col}: {est} RMSE", rm, d.loc[est, "rmse_log"], 2); chk(S, f"{col}: {est} bias %", bi, d.loc[est, "bias_total"], 0, 100)
        if tag == "":
            chk(S, "YCGC pearson r", 0.06, d.loc["YCGC recorded estimate", "pearson"], 2)
    e = js("error_bound/summary.json")
    us, md = e["upper_Sulphur"], e["mid_Dominion"]
    chk(S, "pred sd Sulphur", 0.90, us["predicted_sd_log_error"], 2); chk(S, "pred robust Sulphur", 0.64, us["predicted_robust_sd"], 2)
    chk(S, "spatial only Sulphur", 0.29, us["predicted_sd_spatial_only"], 2); chk(S, "pred sd Dominion", 0.44, md["predicted_sd_log_error"], 2)
    chk(S, "Dominion blocks", 19, md["n_blocks"]); chk(S, "obs sd holes mean", 0.45, us["observed_sd_log_act_over_holes_mean"], 2)
    chk(S, "obs robust", 0.29, us["observed_robust_sd_holes_mean"], 2)
    chk(S, "van Loon lower", 0.18, md["observed_sd_log_vanLoon_lower_bound_range"][0], 2)
    chk(S, "van Loon upper", 0.20, md["observed_sd_log_vanLoon_lower_bound_range"][1], 2)
    sc = pd.read_csv(R / "error_bound" / "meff_scan.csv")
    chk(S, "scan band low", 0.08, sc.obs_lo90.iloc[0], 2); chk(S, "scan band high", 0.80, sc.obs_hi90.iloc[0], 2)

    S = "6.5 design"
    t = pd.read_csv(R / "design" / "holes_needed.csv")
    rows = {"mid-Dominion pay": (68, 0.25, 0.10, 17, 40, 8, 18), "mid-Dominion lean": (17, 0.57, 0.32, 32, 65, 15, 29),
            "Gold Run pay": (87, 0.54, 0.29, 30, 61, 14, 27), "upper Sulphur bench, rich": (24, 0.83, 0.63, 86, 149, 39, 67),
            "upper Sulphur bench (": (6, 0.95, 0.87, 311, 501, 138, 223)}
    for k, (gold, g06, g09, h20a, h20b, h30a, h30b) in rows.items():
        q = t[t.case.str.startswith(k)]
        a, b_ = q[q.sd_hole == 0.6].iloc[0], q[q.sd_hole == 0.9].iloc[0]
        chk(S, f"{k} gold", gold, a.gold_mg_per_sample); chk(S, f"{k} grain share 0.6", g06, a.grain_share, 2)
        chk(S, f"{k} grain share 0.9", g09, b_.grain_share, 2)
        chk(S, f"{k} holes 20% lo", h20a, a.holes_for_20pct); chk(S, f"{k} holes 20% hi", h20b, b_.holes_for_20pct)
        chk(S, f"{k} holes 30% lo", h30a, a.holes_for_30pct); chk(S, f"{k} holes 30% hi", h30b, b_.holes_for_30pct)
    chk(S, "crossover 10 mg", 16, 10 / np.expm1(0.49)); chk(S, "crossover 50 mg", 80, 50 / np.expm1(0.49), -1)
    mc = pd.read_csv(R / "design" / "mc_check.csv")
    a = mc[(mc.n >= 16) & (mc.gold_mg_per_sample >= mc.m_eff)].ratio_mc_over_formula
    b_ = mc[(mc.n >= 16) & (mc.gold_mg_per_sample < mc.m_eff)].ratio_mc_over_formula
    chk(S, "MC ratio min (enough gold)", 0.79, a.min(), 2); chk(S, "MC ratio max (enough gold)", 1.02, a.max(), 2)
    chk(S, "MC ratio min (little gold)", 0.69, b_.min(), 2); chk(S, "MC ratio max (little gold)", 2.05, b_.max(), 2)

    S = "6.6 MCMC"
    m = js("mcmc/summary.json")
    for case, (lap, med, lo, hi) in {"synthetic": (9.2, 9.4, 7.5, 12.0), "dominion": (10.5, 10.3, 8.9, 11.8)}.items():
        x = m[case]
        chk(S, f"{case} Laplace", lap, x["laplace"]["m_eff"], 1); chk(S, f"{case} MCMC median", med, x["mcmc_m_eff_q05_50_95"][1], 1)
        chk(S, f"{case} q05", lo, x["mcmc_m_eff_q05_50_95"][0], 1); chk(S, f"{case} q95", hi, x["mcmc_m_eff_q05_50_95"][2], 1)
    for case in ("synthetic", "dominion"):
        chk(S, f"{case} Laplace within 3% of MCMC median", 3,
            max(3.0, abs(m[case]["laplace"]["m_eff"] / m[case]["mcmc_m_eff_q05_50_95"][1] - 1) * 100))
    chk(S, "dominion ESS below 20", 1, float(m["dominion"]["ess_m_eff"] < 20))
    a = np.load(R / "mcmc" / "traces_dominion.npz")["trace"][:, 3]
    b = np.load(R / "mcmc" / "traces_dominion_dispersed.npz")["trace"][:, 3]
    w = len(b) // 4
    dd = m["dominion_dispersed"]
    chk(S, "R-hat last quarter", 1.04, dd["rhat_log_m_eff_last_quarter"], 2)
    chk(S, "last-quarter median, mode chain", 9.9, np.median(a[-w:]), 1)
    chk(S, "last-quarter median, dispersed chain", 10.2, np.median(b[-w:]), 1)
    chk(S, "pooled median", 10.0, dd["pooled_last_quarter_m_eff_q05_50_95"][1], 1)
    chk(S, "pooled q05", 8.8, dd["pooled_last_quarter_m_eff_q05_50_95"][0], 1)
    chk(S, "pooled q95", 11.4, dd["pooled_last_quarter_m_eff_q05_50_95"][2], 1)
    chk(S, "Laplace within 5% of pooled", 5, max(5.0, (m["dominion"]["laplace"]["m_eff"] / dd["pooled_last_quarter_m_eff_q05_50_95"][1] - 1) * 100))
    chk(S, "last-quarter window", 3500, w)
    idf = pd.read_csv(R / "identifiability" / "summary.csv")
    idf = idf[~((idf.sd_hole_true == 0.8) & (idf.design == "twins") & (idf.model == "grain only"))]
    chk(S, "shortfall min %", 10, (1 - idf.m_eff_mean.max() / 10.49) * 100)
    chk(S, "shortfall max %", 23, (1 - idf.m_eff_mean.min() / 10.49) * 100)


def main():
    for f in (data_section, synthetic_section, dominion_section, twins_section, regional_section, production_section):
        try:
            f()
        except Exception as e:            # a missing output is a failed check, not a crash
            RESULTS.append((f.__name__, f"ERROR {type(e).__name__}: {e}", np.nan, np.nan, False))
    df = pd.DataFrame(RESULTS, columns=["section", "statement", "printed", "recomputed", "ok"])
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 500)
    bad = df[~df.ok]
    print(f"{df.ok.sum()} of {len(df)} numbers match.")
    if len(bad):
        print(bad.to_string(index=False))
    df.to_csv(ROOT / "paper" / "number_check.csv", index=False)
    sys.exit(0 if bad.empty else 1)


if __name__ == "__main__":
    main()
