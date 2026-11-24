"""Shared experiment definitions: the method line-up, features and the CV loop.

Used identically by the synthetic benchmark and the real-data runs, so the
two are directly comparable.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import baselines as B
from . import cv, metrics, models

# Target: log1p of accumulation in mg/m2.
#   zero assays -> 0; trace (0.1 c/yd3 over ~2 m) -> ~1.4; typical pay -> ~7-9.
# Chosen over grade because accumulation (grade x thickness) is what is
# mineable per unit area and it removes the thick-gravel dilution effect
# van Loon (2017) discusses for lower Sulphur Creek.
TARGET = "y_log"


def make_target(accum_mg_m2):
    return np.log1p(np.asarray(accum_mg_m2, float))


def method_table():
    """(name, feature set, factory)."""
    return [
        ("global_median", "xy", lambda: B.GlobalMedian()),
        ("nn_xy", "xy", lambda: B.NearestNeighbour()),
        ("idw_xy", "xy", lambda: B.IDW()),
        ("kriging_xy", "xy", lambda: B.OrdinaryKriging(anisotropy=1.0, name="kriging_xy")),
        ("idw_sn_aniso5", "sn", lambda: B.IDW(scale=np.array([1.0, 5.0]))),
        ("kriging_sn_auto", "sn", lambda: B.OrdinaryKriging(anisotropy="auto", name="kriging_sn")),
        ("gp_sn", "sn", lambda: models.AnisotropicGP(n_restarts=1)),
        ("gbm_sn", "sn_feat", lambda: models.GBMQuantile()),
    ]


def features(df: pd.DataFrame, which: str) -> np.ndarray:
    if which == "xy":
        return df[["x", "y"]].to_numpy(float)
    if which == "sn":
        return df[["s", "n"]].to_numpy(float)
    if which == "sn_feat":
        cols = [df.s, df.n, np.abs(df.n), df.n / df.halfwidth]
        for extra in ("bedrock_rel_m", "overburden_m", "thawed"):
            if extra in df:
                cols.append(df[extra])
        return np.column_stack(cols).astype(float)
    raise ValueError(which)


def splits_for(df: pd.DataFrame, seed: int = 0, n_folds: int = 10, block: float = 1000.0, buffer: float = 150.0):
    return {
        "LOLO": list(cv.grouped_line_folds(df.line_id.values, n_folds=n_folds, seed=seed)),
        f"block{int(block)}m": list(cv.along_valley_blocks(df.s.values, block=block, buffer=buffer)),
    }


def run_all(df: pd.DataFrame, seed: int = 0, methods=None, verbose=True, **split_kw):
    y = df[TARGET].to_numpy(float)
    ylin = np.expm1(y)
    rows, preds = [], {}
    for name, feat, fac in (methods or method_table()):
        X = features(df, feat)
        for sch, splits in splits_for(df, seed=seed, **split_kw).items():
            t0 = time.time()
            mu, sd, _ = cv.run_cv(fac, X, y, splits)
            sd = None if np.all(np.isnan(sd)) else sd
            sc = metrics.score_all(y, mu, sd, y_linear=ylin)
            sc.update(method=name, scheme=sch, seed=seed, secs=round(time.time() - t0, 1))
            rows.append(sc)
            preds[(name, sch)] = (mu, sd)
            if verbose:
                c90 = sc["cov90"]
                print(f"{sch:10s} {name:16s} crps={sc['crps']:.3f} rho={sc['spearman']:.2f} "
                      f"cov90={c90:.2f} top20={sc['top20']:.2f}" if c90 == c90 else
                      f"{sch:10s} {name:16s} crps={sc['crps']:.3f} rho={sc['spearman']:.2f} "
                      f"cov90=  - top20={sc['top20']:.2f}")
    return pd.DataFrame(rows), preds


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    cols = ["crps", "mae", "spearman", "cov50", "cov90", "interval_score90", "top20", "secs"]
    return res.groupby(["scheme", "method"])[cols].mean().round(3).sort_values(["scheme", "crps"])
