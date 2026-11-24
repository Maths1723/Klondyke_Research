"""Latent gold-concentration field with a physical measurement model.

    f(x) = log c(x)            c: in-situ gold concentration (mg/m3), the geology
    f ~ GP(beta0, K)           K = sigma^2 Matern-3/2(ds/ls_s, dn/ls_n)        (valley coordinates)
                               [+ tau^2 1{same source map}]                    campaign bias, optional
    log lam_i = f_i + log V_i - log mbar + e_i,   e_i ~ N(0, sd_drill^2)       core-recovery error, optional
    assay_i | lam_i  ~  pi0 * 1{S=0}  +  (1 - pi0) * CompoundPoissonGamma(lam_i; alpha, mbar/alpha)
                                                                               structural zeros, optional

Three distinct sources of scatter between neighbouring assays are separated:
  * grain counting  - variance m_eff / (c V): falls with grade and sample volume;
  * drilling error  - e_i: a random effective sample volume (core recovery vs
                      the assumed shoe volume), constant on the log scale;
  * structural zero - pi0: a sample that failed or missed the gravel entirely.
What remains spatially correlated in f is the geology a mining block sees.

Because e_i is iid and enters log lam additively, it is exactly a diagonal
(white) term in the latent covariance at the data, and is excluded when
predicting c at new locations.

Inference: Laplace approximation (Rasmussen & Williams 2006, Alg. 3.1) with a
Fisher-type safeguard: the log-likelihood curvature in log lam can be
positive, so the Newton weight is W = max(-curvature, eps). Hyperparameters
maximise the approximate marginal likelihood (L-BFGS-B); nested variants are
compared by that marginal likelihood.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize

from . import sampling


def matern32(A, B, ls, sigma2):
    d = np.sqrt((((A[:, None, :] - B[None, :, :]) / ls) ** 2).sum(-1))
    r = np.sqrt(3.0) * d
    return sigma2 * (1 + r) * np.exp(-r)


# name: (to-internal, from-internal, default, bounds-on-internal)
_LOG = (np.log, np.exp)
_ID = (lambda v: v, lambda v: v)
_LOGIT = (lambda v: np.log(v / (1 - v)), lambda v: 1 / (1 + np.exp(-v)))
SPEC = {
    "sigma":    (_LOG, 1.0, (np.log(0.05), np.log(5.0))),
    "ls_s":     (_LOG, 200.0, (np.log(5.0), np.log(5000.0))),
    "ls_n":     (_LOG, 40.0, (np.log(2.0), np.log(1000.0))),
    "beta0":    (_ID, None, (-5.0, 15.0)),
    "mbar":     (_LOG, 1.0, (np.log(1e-3), np.log(1e3))),
    "alpha":    (_LOG, 1.0, (np.log(0.05), np.log(50.0))),
    "tau":      (_LOG, 0.3, (np.log(1e-3), np.log(3.0))),
    "sd_drill": (_LOG, 0.3, (np.log(1e-3), np.log(3.0))),
    "pi0":      (_LOGIT, 0.02, (-9.0, 0.0)),
}


class CPGField:
    name = "cpg_field"

    def __init__(self, init=None, fit_grain=True, group_effect=False, drill_noise=False,
                 zero_infl=False, maxiter=60, jitter=1e-6, fixed=None):
        self.init = dict(init or {})
        self.fixed = dict(fixed or {})
        if not fit_grain:
            self.fixed.setdefault("mbar", self.init.get("mbar", 1.0))
            self.fixed.setdefault("alpha", self.init.get("alpha", 1.0))
        self.group_effect, self.drill_noise, self.zero_infl = group_effect, drill_noise, zero_infl
        self.maxiter, self.jitter = maxiter, jitter

    # --------------------------------------------------------------- params
    def _active(self):
        names = ["sigma", "ls_s", "ls_n", "beta0", "mbar", "alpha"]
        if self.group_effect:
            names.append("tau")
        if self.drill_noise:
            names.append("sd_drill")
        if self.zero_infl:
            names.append("pi0")
        return [n for n in names if n not in self.fixed]

    def _unpack(self, th):
        p = {"tau": 0.0, "sd_drill": 0.0, "pi0": 0.0}
        p.update(self.fixed)
        for name, v in zip(self.names_, th):
            p[name] = SPEC[name][0][1](v)
        return p

    def _K(self, p, train=True):
        K = matern32(self.X, self.X, np.array([p["ls_s"], p["ls_n"]]), p["sigma"] ** 2)
        if self.group_effect:
            K = K + p["tau"] ** 2 * (self.G[:, None] == self.G[None, :])
        K[np.diag_indices_from(K)] += self.jitter + (p["sd_drill"] ** 2 if train else 0.0)
        return K

    # ----------------------------------------------------------- likelihood
    def _lik(self, f, p, logV_m):
        """log-lik per obs, gradient and (negated, safeguarded) curvature in f."""
        lam = np.exp(np.clip(f + logV_m, -40.0, 40.0))
        ll, EN, VN = sampling.count_posterior(lam, self.kind, self.s, self.s_tr, p["mbar"], p["alpha"])
        g = EN - lam
        h = VN - lam
        pi0 = p["pi0"]
        if pi0 > 0:
            z = self.kind == 0
            q = (1 - pi0) * np.exp(-lam[z])
            Z = pi0 + q
            r = q / Z
            ll[z] = np.log(Z)
            g[z] = -lam[z] * r
            h[z] = -lam[z] * r + lam[z] ** 2 * r * (1 - r)
            ll[~z] += np.log1p(-pi0)
        return ll, g, np.maximum(-h, 1e-8)

    def _laplace(self, p, a0=None, tol=1e-7, max_newton=60):
        K = self._K(p)
        m = p["beta0"]
        n = len(self.V)
        logV_m = np.log(self.V) - np.log(p["mbar"])

        def psi_of(f, a):
            ll, g, W = self._lik(f, p, logV_m)
            val = ll.sum() - 0.5 * a @ (f - m)
            return (val if np.isfinite(val) else -np.inf), g, W

        a = np.zeros(n); f = K @ a + m
        psi, g, W = psi_of(f, a)
        if a0 is not None:
            fw = K @ a0 + m
            pw = psi_of(fw, a0)
            if pw[0] > psi:
                a, f, (psi, g, W) = a0.copy(), fw, pw
        for _ in range(max_newton):
            sW = np.sqrt(W)
            L = np.linalg.cholesky(np.eye(n) + sW[:, None] * K * sW[None, :])
            b = W * (f - m) + g
            a_new = b - sW * cho_solve((L, True), sW * (K @ b))
            step = 1.0
            while True:
                a_t = a + step * (a_new - a)
                f_t = K @ a_t + m
                psi_t, g_t, W_t = psi_of(f_t, a_t)
                if psi_t >= psi - 1e-10 or step < 1e-4:
                    break
                step *= 0.5
            conv = abs(psi_t - psi) < tol * (1 + abs(psi))
            a, f, psi, g, W = a_t, f_t, psi_t, g_t, W_t
            if conv:
                break
        sW = np.sqrt(W)
        L = np.linalg.cholesky(np.eye(n) + sW[:, None] * K * sW[None, :])
        return dict(f=f, a=a, W=W, sW=sW, L=L, logZ=psi - np.log(np.diag(L)).sum(), m=m)

    # ------------------------------------------------------------------ API
    def fit(self, X, kind, s, s_tr, V, groups=None, theta0=None):
        self.X = np.asarray(X, float)
        self.kind, self.s, self.s_tr, self.V = map(np.asarray, (kind, s, s_tr, V))
        self.G = np.asarray(groups) if groups is not None else np.zeros(len(self.V))
        self.names_ = self._active()
        pos = self.s[self.kind == 2] / self.V[self.kind == 2]
        defaults = {k: SPEC[k][1] for k in SPEC}
        defaults["beta0"] = float(np.log(np.median(pos))) if len(pos) else 0.0
        defaults.update(self.init)
        th0 = np.array([SPEC[k][0][0](defaults[k]) for k in self.names_], float)
        if theta0 is not None:                     # dict of natural-scale values (warm start)
            th0 = np.array([SPEC[k][0][0](theta0[k]) if k in theta0 else th0[i]
                            for i, k in enumerate(self.names_)], float)
        bounds = [SPEC[k][2] for k in self.names_]
        th0 = np.clip(th0, [b[0] for b in bounds], [b[1] for b in bounds])
        self._acache = None

        def nlml(th):
            try:
                r = self._laplace(self._unpack(th), a0=self._acache)
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                return 1e10
            if not np.isfinite(r["logZ"]):
                return 1e10
            self._acache = r["a"]
            return -r["logZ"]

        res = minimize(nlml, th0, method="L-BFGS-B", bounds=bounds,
                       options=dict(maxiter=self.maxiter, eps=1e-3))
        self.theta_, self.opt_ = res.x, res
        self.p_ = self._unpack(res.x)
        self.post_ = self._laplace(self.p_)
        return self

    def predict_latent(self, Xq, groups_q=None):
        """Mean and sd of log c at new locations (geology only: no drilling or grain noise)."""
        p, r = self.p_, self.post_
        ls = np.array([p["ls_s"], p["ls_n"]])
        Ks = matern32(np.asarray(Xq, float), self.X, ls, p["sigma"] ** 2)
        kss = p["sigma"] ** 2
        if self.group_effect and groups_q is not None:
            Ks = Ks + p["tau"] ** 2 * (np.asarray(groups_q)[:, None] == self.G[None, :])
            kss = kss + p["tau"] ** 2
        mu = r["m"] + Ks @ r["a"]
        v = solve_triangular(r["L"], r["sW"][:, None] * Ks.T, lower=True)
        return mu, np.sqrt(np.maximum(kss - (v ** 2).sum(0), 1e-12))

    def sample_assays(self, Xq, Vq, n_draws=400, rng=None, groups_q=None):
        """Predictive draws of recovered mass S (mg) for new samples of volume Vq."""
        rng = rng or np.random.default_rng(0)
        p = self.p_
        mu, sd = self.predict_latent(Xq, groups_q)
        tot = np.sqrt(sd ** 2 + p["sd_drill"] ** 2)
        f = mu[None, :] + tot[None, :] * rng.standard_normal((n_draws, len(mu)))
        S, _ = sampling.simulate(np.exp(f), np.broadcast_to(Vq, f.shape), p["mbar"], p["alpha"], rng)
        if p["pi0"] > 0:
            S = np.where(rng.random(S.shape) < p["pi0"], 0.0, S)
        return S

    def summary(self):
        p = self.p_
        _, meff = sampling.grain_params(p["mbar"], p["alpha"])
        out = {k: float(p[k]) for k in ["sigma", "ls_s", "ls_n", "beta0", "mbar", "alpha", "tau", "sd_drill", "pi0"]}
        out.update(m_eff=float(meff), logZ=float(self.post_["logZ"]), n_params=len(self.names_),
                   converged=bool(self.opt_.success), n_iter=int(self.opt_.nit))
        return out
