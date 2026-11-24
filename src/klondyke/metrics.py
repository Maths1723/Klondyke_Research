"""Scores for probabilistic spatial predictions.

All scores are computed on the modelling scale (log accumulation) unless
stated. CRPS is the main score: it rewards accuracy AND honest uncertainty,
and for a deterministic prediction it reduces to the absolute error, so
point predictors (NN, IDW) and probabilistic ones are on the same footing.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, spearmanr


def crps_gaussian(y, mu, sd):
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    if sd is None or np.all(np.isnan(sd)):
        return np.abs(y - mu)
    sd = np.asarray(sd, float)
    z = (y - mu) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def crps_samples(y, S):
    """CRPS from predictive draws. y: (n,), S: (n_draws, n). Uses the sorted-sample identity
    E|X-X'| = (2/m^2) sum_i (2i - m - 1) x_(i)."""
    y = np.asarray(y, float)
    S = np.sort(np.asarray(S, float), axis=0)
    m = S.shape[0]
    t1 = np.abs(S - y[None, :]).mean(0)
    w = (2 * np.arange(1, m + 1) - m - 1)[:, None]
    t2 = (2.0 / m ** 2) * (w * S).sum(0)
    return t1 - 0.5 * t2


def coverage_samples(y, S, level=0.9):
    lo, hi = np.quantile(S, [(1 - level) / 2, (1 + level) / 2], axis=0)
    y = np.asarray(y)
    return float(np.mean((y >= lo) & (y <= hi)))


def coverage(y, mu, sd, level=0.9):
    if sd is None or np.all(np.isnan(sd)):
        return np.nan
    q = norm.ppf(0.5 + level / 2)
    return float(np.mean(np.abs(np.asarray(y) - mu) <= q * np.asarray(sd)))


def interval_score(y, mu, sd, alpha=0.1):
    """Gneiting & Raftery (2007) interval score for the central (1-alpha) interval."""
    if sd is None or np.all(np.isnan(sd)):
        return np.nan
    q = norm.ppf(1 - alpha / 2)
    lo, hi = mu - q * sd, mu + q * sd
    y = np.asarray(y)
    return float(np.mean((hi - lo) + 2 / alpha * (lo - y) * (y < lo) + 2 / alpha * (y - hi) * (y > hi)))


def capture_curve(y_true_linear, score, fractions=(0.1, 0.2, 0.3, 0.5)):
    """Share of total gold (linear units) contained in the top-k% ranked locations.

    The operator's question: if I only test the best-ranked 20% of ground,
    what share of the gold do I reach? Random ranking gives ~k%.
    """
    y = np.asarray(y_true_linear, float)
    order = np.argsort(-np.asarray(score))
    cum = np.cumsum(y[order]) / y.sum()
    n = len(y)
    return {f"top{int(f*100)}": float(cum[max(int(np.ceil(f * n)) - 1, 0)]) for f in fractions}


def score_all(y, mu, sd, y_linear=None):
    y = np.asarray(y, float)
    out = {
        "crps": float(np.mean(crps_gaussian(y, mu, sd))),
        "mae": float(np.mean(np.abs(y - mu))),
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
        "spearman": float(spearmanr(y, mu).statistic),
        "cov50": coverage(y, mu, sd, 0.5),
        "cov90": coverage(y, mu, sd, 0.9),
        "interval_score90": interval_score(y, mu, sd, 0.1),
    }
    if y_linear is not None:
        out.update(capture_curve(y_linear, mu))
    return out
