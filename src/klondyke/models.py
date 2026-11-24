"""Probabilistic models beyond the baselines.

AnisotropicGP
    Gaussian process with a Matern-3/2 kernel and separate length scales per
    input (automatic relevance determination). Fed valley coordinates (s, n),
    the fitted length scales ARE the along- and across-valley correlation
    ranges - estimated by maximum marginal likelihood rather than eyeballed
    from a variogram. This is the probabilistic core of the project.

GBMQuantile
    Gradient boosting on hand-built geological features, with 5/50/95%
    quantile models to give an uncertainty band. The "ML" comparison that
    most prospectivity work would use.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler


class AnisotropicGP:
    name = "anisotropic_gp"

    def __init__(self, nu: float = 1.5, n_restarts: int = 3, init_ls=None, ls_bounds=(1e-2, 1e2), seed: int = 0):
        self.nu, self.n_restarts, self.init_ls, self.ls_bounds, self.seed = nu, n_restarts, init_ls, ls_bounds, seed

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.sx = StandardScaler().fit(X)
        Xs = self.sx.transform(X)
        d = X.shape[1]
        ls0 = np.ones(d) if self.init_ls is None else np.asarray(self.init_ls, float)
        kernel = (ConstantKernel(1.0, (1e-3, 1e2))
                  * Matern(length_scale=ls0, length_scale_bounds=self.ls_bounds, nu=self.nu)
                  + WhiteKernel(0.5, (1e-4, 1e1)))
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                           n_restarts_optimizer=self.n_restarts,
                                           random_state=self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.gp.fit(Xs, y)
        return self

    def predict(self, X):
        Xs = self.sx.transform(np.asarray(X, float))
        mu, sd = self.gp.predict(Xs, return_std=True)  # includes white noise: new-measurement sd
        return mu, sd

    def predict_latent(self, X):
        """Smooth-field mean and sd with the white-noise (nugget) variance removed."""
        mu, sd = self.predict(X)
        noise = self.gp.kernel_.k2.noise_level * self.gp._y_train_std ** 2
        return mu, np.sqrt(np.maximum(sd ** 2 - noise, 1e-12))

    def length_scales_m(self):
        """Fitted correlation length per input, back in original units (m)."""
        k = self.gp.kernel_.k1.k2
        return np.asarray(k.length_scale) * self.sx.scale_

    def params(self):
        k = self.gp.kernel_
        return {
            "length_scales": self.length_scales_m().tolist(),
            "signal_var_std_units": float(k.k1.k1.constant_value),
            "noise_var_std_units": float(k.k2.noise_level),
            "log_marginal_likelihood": float(self.gp.log_marginal_likelihood_value_),
        }


class GBMQuantile:
    name = "gbm_quantile"

    def __init__(self, max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10, seed=0):
        self.kw = dict(max_iter=max_iter, learning_rate=learning_rate,
                       max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                       random_state=seed)

    def fit(self, X, y):
        self.mean_ = HistGradientBoostingRegressor(loss="squared_error", **self.kw).fit(X, y)
        self.lo_ = HistGradientBoostingRegressor(loss="quantile", quantile=0.05, **self.kw).fit(X, y)
        self.hi_ = HistGradientBoostingRegressor(loss="quantile", quantile=0.95, **self.kw).fit(X, y)
        return self

    def predict(self, X):
        mu = self.mean_.predict(X)
        lo, hi = self.lo_.predict(X), self.hi_.predict(X)
        sd = np.maximum((hi - lo) / (2 * 1.6449), 1e-6)  # Gaussian-equivalent spread
        return mu, sd
