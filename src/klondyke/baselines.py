"""Baseline spatial predictors.

Every model exposes fit(X, y) and predict(X) -> (mean, sd or None).
X columns are the spatial coordinates the model should use (map x,y or
valley s,n); scaling of axes is the caller's choice unless stated.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


class GlobalMedian:
    name = "global_median"

    def fit(self, X, y):
        self.m = float(np.median(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.m), None


class NearestNeighbour:
    name = "nearest_neighbour"

    def __init__(self, scale=None):
        self.scale = scale

    def _s(self, X):
        return X * self.scale if self.scale is not None else X

    def fit(self, X, y):
        self.tree = cKDTree(self._s(X))
        self.y = np.asarray(y)
        return self

    def predict(self, X):
        _, j = self.tree.query(self._s(X), k=1)
        return self.y[j], None


class IDW:
    """Inverse distance weighting, the method used in the YGS studies."""

    name = "idw"

    def __init__(self, power: float = 2.0, k: int = 12, scale=None):
        self.power, self.k, self.scale = power, k, scale

    def _s(self, X):
        return X * self.scale if self.scale is not None else X

    def fit(self, X, y):
        self.tree = cKDTree(self._s(X))
        self.y = np.asarray(y)
        return self

    def predict(self, X):
        k = min(self.k, len(self.y))
        d, j = self.tree.query(self._s(X), k=k)
        d = np.maximum(d, 1e-6)
        w = 1.0 / d ** self.power
        return (w * self.y[j]).sum(1) / w.sum(1), None


# ---------------------------------------------------------------- variograms
def exp_variogram(h, nugget, psill, rng):
    return nugget + psill * (1.0 - np.exp(-3.0 * h / rng))  # practical range


def empirical_variogram(X, y, n_bins=15, max_dist=None, mask_fn=None):
    """Classical (Matheron) estimator. mask_fn(dX) -> bool selects pair directions."""
    i, j = np.triu_indices(len(y), 1)
    dX = X[j] - X[i]
    h = np.hypot(dX[:, 0], dX[:, 1])
    g = 0.5 * (y[j] - y[i]) ** 2
    keep = np.ones(len(h), bool) if mask_fn is None else mask_fn(dX)
    h, g = h[keep], g[keep]
    max_dist = max_dist or np.percentile(h, 50)
    edges = np.linspace(0, max_dist, n_bins + 1)
    idx = np.digitize(h, edges) - 1
    hc, gc, nc = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() >= 10:
            hc.append(h[m].mean()); gc.append(g[m].mean()); nc.append(m.sum())
    return np.array(hc), np.array(gc), np.array(nc)


def fit_exp_variogram(hc, gc, nc):
    var0 = gc.max()
    def resid(p):
        model = exp_variogram(hc, *p)
        return np.sqrt(nc) * (gc - model) / np.maximum(model, 1e-9)  # Cressie weights
    p0 = [0.2 * var0, 0.8 * var0, hc.max() / 2]
    lb = [0, 1e-9, max(hc.min(), 1e-3)]
    ub = [var0 * 2, var0 * 3, hc.max() * 10]
    r = least_squares(resid, p0, bounds=(lb, ub))
    return dict(nugget=r.x[0], psill=r.x[1], range=r.x[2])


def directional_anisotropy(X, y, tol_deg=22.5, max_dist=None):
    """Ratio of variogram ranges along axis 0 vs axis 1 (e.g. s vs n).

    Returns ratio >= 1 when correlation persists further along axis 0.
    """
    tol = np.tan(np.deg2rad(tol_deg))
    along = lambda d: np.abs(d[:, 1]) <= tol * np.abs(d[:, 0])
    across = lambda d: np.abs(d[:, 0]) <= tol * np.abs(d[:, 1])
    out = {}
    for name, fn in (("along", along), ("across", across)):
        hc, gc, nc = empirical_variogram(X, y, mask_fn=fn, max_dist=max_dist)
        out[name] = fit_exp_variogram(hc, gc, nc) if len(hc) >= 4 else None
    if out["along"] is None or out["across"] is None:
        return 1.0, out
    return float(out["along"]["range"] / out["across"]["range"]), out


class OrdinaryKriging:
    """Global-neighbourhood ordinary kriging with an exponential variogram.

    anisotropy: factor applied to axis 1 before computing distances
    ('auto' estimates it from directional variograms). The predictive sd is
    for a NEW measurement at the location (includes the nugget), which is the
    right comparison against held-out drill holes.
    """

    def __init__(self, anisotropy=1.0, max_dist=None, name="ordinary_kriging"):
        self.anisotropy = anisotropy
        self.max_dist = max_dist
        self.name = name

    def _s(self, X):
        Z = np.array(X, dtype=float, copy=True)
        Z[:, 1] *= self.ratio_
        return Z

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        if self.anisotropy == "auto":
            r, _ = directional_anisotropy(X, y, max_dist=self.max_dist)
            self.ratio_ = float(np.clip(r, 1.0, 50.0))
        else:
            self.ratio_ = float(self.anisotropy)
        Z = self._s(X)
        hc, gc, nc = empirical_variogram(Z, y, max_dist=self.max_dist)
        self.vario_ = fit_exp_variogram(hc, gc, nc)
        self.Z, self.y = Z, y
        n = len(y)
        sill = self.vario_["nugget"] + self.vario_["psill"]
        D = np.hypot(*(Z[:, None, :] - Z[None, :, :]).transpose(2, 0, 1))
        C = sill - exp_variogram(D, **self._vp())
        np.fill_diagonal(C, sill)
        K = np.ones((n + 1, n + 1)); K[:n, :n] = C; K[n, n] = 0.0
        self.K_inv_ = np.linalg.pinv(K)
        self.sill_ = sill
        return self

    def _vp(self):
        v = self.vario_
        return dict(nugget=v["nugget"], psill=v["psill"], rng=v["range"])

    def predict(self, X):
        Zq = self._s(np.asarray(X, float))
        D = np.hypot(*(Zq[:, None, :] - self.Z[None, :, :]).transpose(2, 0, 1))
        c = self.sill_ - exp_variogram(D, **self._vp())
        # nugget is a discontinuity at h=0: cross-covariances to other locations exclude it
        c = np.where(D < 1e-9, self.sill_, c)
        rhs = np.hstack([c, np.ones((len(Zq), 1))])
        W = rhs @ self.K_inv_.T
        w, mu = W[:, :-1], W[:, -1]
        mean = w @ self.y
        var = self.sill_ - (w * c).sum(1) - mu
        self._last_var_ = var
        return mean, np.sqrt(np.maximum(var, 1e-12))

    def predict_latent(self, X):
        """Mean and sd of the smooth field, nugget removed.

        Approximates a mining block whose size averages out the nugget
        (small-scale, uncorrelated) variance. Proper block kriging over a
        defined block geometry is on the roadmap.
        """
        mean, _ = self.predict(X)
        var = self._last_var_ - self.vario_["nugget"]
        return mean, np.sqrt(np.maximum(var, 1e-12))
