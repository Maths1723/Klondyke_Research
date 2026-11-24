"""Physical measurement model for historic placer drill assays.

A churn-drill sample of volume V taken where the gold concentration is c
(mg/m3) contains a random number N of gold grains with random masses m_k:

    N ~ Poisson(lam),     lam = c V / mbar,       m_k ~ Gamma(alpha, theta) iid,
    S = sum_{k<=N} m_k    (gold mass recovered, mg),  observed grade G = S / V.

S is compound Poisson-gamma. Its moments follow directly:

    E[S]   = lam * mbar                        = c V
    Var[S] = lam * E[m^2] = c V * m_eff,       m_eff = E[m^2] / E[m] = (alpha+1) theta
    CV^2   = m_eff / (c V)                     (relative variance of one assay)
    P(S=0) = exp(-lam)                         (a zero assay is "no grain in the sample")

So the nugget effect of placer drilling is not an unexplained constant: it
is predicted to shrink with richer ground (c) and bigger samples (V, i.e.
thicker dredge sections), and the share of zero assays is tied to the same
parameters. This is the Poisson argument of Clifton, Hunter & Phillips
(1969, USGS PP 625-C) extended to unequal grain masses.

Only lam = c V / mbar enters the likelihood together with the grain shape
alpha, so an error in the assumed drill volume V is absorbed one-for-one by
mbar: grain-mass estimates are relative to the assumed shoe area.

Observation types handled
    zero   : S = 0                  P = exp(-lam)
    trace  : 0 < S < s_trace        (YCGC "TR", digitised as 0.1 c/yd3)
    value  : S = s > 0              density of the compound Poisson-gamma
For each, d log p / d log lam = E[N | obs] - lam and
d2 log p / d(log lam)^2 = Var[N | obs] - lam, so the posterior over the
latent grain count N is all the inference needs.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaincc, gammainc, gammaln

from . import units

GOLD_DENSITY_MG_MM3 = 17.5  # placer gold of fineness ~800 (pure gold 19.3)

# Typical churn-drill geometry (Placer Examination, BLM Tech. Bull. 4, 1969):
# 6-inch drive pipe with a 7.5-inch shoe; theoretical shoe area 0.3068 ft2.
SHOE_AREA_FT2_7_5IN = np.pi / 4 * (7.5 / 12) ** 2
# Radford factor: 0.27 ft3 per foot of drive (~1/100 yd3), an arbitrary upgrade.
RADFORD_FT3_PER_FT = 0.27


def sample_volume_m3(section_ft, area_ft2: float = SHOE_AREA_FT2_7_5IN):
    """Drill-sample volume for a section of the given thickness (ft)."""
    return np.asarray(section_ft, float) * area_ft2 * 0.3048 ** 3


def flake_mass_mg(diameter_mm, thickness_ratio: float = 0.2, density=GOLD_DENSITY_MG_MM3):
    """Mass of a disc-shaped flake (thickness = ratio x diameter)."""
    d = np.asarray(diameter_mm, float)
    return np.pi / 4 * d ** 2 * (thickness_ratio * d) * density


def grain_params(mbar_mg: float, alpha: float):
    """Gamma(alpha, theta) with mean mbar. Returns theta and m_eff."""
    theta = mbar_mg / alpha
    return theta, (alpha + 1) * theta


def assay_cv(c_mg_m3, V_m3, m_eff_mg):
    """Relative sd of a single assay from grain counting alone."""
    return np.sqrt(m_eff_mg / (np.asarray(c_mg_m3) * np.asarray(V_m3)))


def log_sd_from_cv(cv):
    """sd of log(assay) implied by a relative sd (lognormal moment match)."""
    return np.sqrt(np.log1p(np.asarray(cv) ** 2))


def p_zero(c_mg_m3, V_m3, mbar_mg):
    return np.exp(-np.asarray(c_mg_m3) * np.asarray(V_m3) / mbar_mg)


def simulate(c_mg_m3, V_m3, mbar_mg, alpha, rng):
    """Draw recovered gold mass S (mg) for each sample."""
    c, V = np.broadcast_arrays(np.asarray(c_mg_m3, float), np.asarray(V_m3, float))
    lam = c * V / mbar_mg
    N = rng.poisson(lam)
    theta = mbar_mg / alpha
    S = np.where(N > 0, rng.gamma(np.maximum(N, 1) * alpha, theta), 0.0)
    return S, N


# ------------------------------------------------------------------ likelihood
def _mode_value(lam, s, alpha, theta, iters=60):
    """Continuous mode n* of log[Pois(n; lam) Gamma(s; n alpha, theta)] and its curvature.

    d/dn = log lam - psi(n+1) + alpha log(s/theta) - alpha psi(n alpha); decreasing in n.
    """
    from scipy.special import polygamma, psi
    lo = np.full(len(lam), 1e-3); hi = np.full(len(lam), 1e7)
    c0 = np.log(lam) + alpha * np.log(s / theta)
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        d = c0 - psi(mid + 1) - alpha * psi(mid * alpha)
        up = d > 0
        lo = np.where(up, mid, lo); hi = np.where(up, hi, mid)
    n = np.sqrt(lo * hi)
    curv = polygamma(1, n + 1) + alpha ** 2 * polygamma(1, n * alpha)
    return n, 1.0 / np.sqrt(curv)


def _windows(lo, hi, cap=6000):
    lo = np.maximum(1, np.floor(lo)).astype(int)
    hi = np.maximum(lo, np.ceil(hi)).astype(int)
    width = int(min(cap, (hi - lo).max() + 1))
    n = lo[:, None] + np.arange(width)[None, :]
    valid = n <= hi[:, None]
    return n, valid


def count_posterior(lam, kind, s, s_trace, mbar, alpha):
    """Posterior moments of the grain count N and log p(obs | lam).

    lam   : (n,) expected grains
    kind  : (n,) 0 = zero, 1 = trace (0 < S < s_trace), 2 = value S = s
    s     : (n,) observed mass (mg) for kind 2 (ignored otherwise)
    s_trace : (n,) upper bound (mg) for kind 1
    Returns loglik (n,), E[N|obs] (n,), Var[N|obs] (n,).
    """
    lam = np.asarray(lam, float)
    kind = np.asarray(kind)
    s = np.asarray(s, float)
    theta = mbar / alpha
    ll = np.empty(len(lam)); EN = np.zeros(len(lam)); VN = np.zeros(len(lam))

    z = kind == 0
    ll[z] = -lam[z]

    s_trace = np.broadcast_to(np.asarray(s_trace, float), lam.shape)
    for k in (2, 1):
        sel = kind == k
        if not sel.any():
            continue
        L = lam[sel]
        if k == 2:
            x = s[sel]
            nstar, sd = _mode_value(L, x, alpha, theta)
            n, valid = _windows(nstar - 12 * sd - 10, nstar + 12 * sd + 10)
            a = n * alpha
            logterm = (n * np.log(L[:, None]) - gammaln(n + 1) - L[:, None]
                       + (a - 1) * np.log(x[:, None]) - x[:, None] / theta - gammaln(a) - a * np.log(theta))
        else:
            x = s_trace[sel]
            # posterior over N: Poisson prior times P(S < x | N); mass sits at small n
            top = np.minimum(L + 12 * np.sqrt(L) + 20, 3 * x / mbar + 12 * np.sqrt(x / mbar) + 30)
            n, valid = _windows(np.ones_like(L), top)
            a = n * alpha
            logterm = (n * np.log(L[:, None]) - gammaln(n + 1) - L[:, None]
                       + np.log(np.maximum(gammainc(a, x[:, None] / theta), 1e-300)))
        logterm = np.where(valid, logterm, -np.inf)
        mx = logterm.max(1, keepdims=True)
        w = np.exp(logterm - mx)
        tot = w.sum(1)
        ll[sel] = mx[:, 0] + np.log(tot)
        p = w / tot[:, None]
        EN[sel] = (p * n).sum(1)
        VN[sel] = (p * n.astype(float) ** 2).sum(1) - EN[sel] ** 2
    return ll, EN, VN


def observations_from_grade(cents_yd3, section_ft, price=units.HISTORIC_PRICE_PER_OZ,
                            area_ft2=SHOE_AREA_FT2_7_5IN, trace_cents=units.TRACE_CENTS):
    """Convert YCGC grade records into (kind, s_mg, s_trace_mg, V_m3).

    Grade is normalised to the $20.67 basis before conversion to mass, so maps
    valued at $35/oz are not read as 1.69x richer.
    """
    g = np.asarray(cents_yd3, float) * (units.HISTORIC_PRICE_PER_OZ / np.asarray(price, float))
    V = sample_volume_m3(section_ft, area_ft2)
    c = units.cents_per_yd3_to_mg_per_m3(g)
    s = c * V
    s_tr = units.cents_per_yd3_to_mg_per_m3(trace_cents * 1.5) * V   # upper bound for "trace"
    kind = np.where(g <= 0, 0, np.where(np.isclose(np.asarray(cents_yd3, float), trace_cents), 1, 2))
    return kind, s, s_tr, V


def grain_information(c_mg_m3, V_m3, mbar_mg, alpha, v0=0.5):
    """Per-sample Fisher information on grain size, for a known concentration c.

    Two channels:
      zero channel  -- the indicator 1{S = 0}, P = exp(-lam), lam = cV/mbar:
                       I(log mbar) = lam^2 e^{-lam} / (1 - e^{-lam}), largest near lam = 1.6
                       and vanishing once samples hold many grains;
      dispersion    -- log-assay variance v = v0 + log(1 + m_eff/(cV)), where v0 collects
                       every non-grain source (per-hole error, unresolved geology):
                       I(log m_eff) = 0.5 * [x/(1+x)]^2 / v^2, x = m_eff/(cV)  (Gaussian approx.).
    Known c makes these upper bounds on what real assays (unknown c) carry.
    Returns (I_zero, I_disp) arrays.
    """
    c, V = np.broadcast_arrays(np.asarray(c_mg_m3, float), np.asarray(V_m3, float))
    lam = c * V / mbar_mg
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        I_zero = np.where(lam > 1e-9, lam ** 2 * np.exp(-lam) / -np.expm1(-lam), 0.0)
    m_eff = (1.0 + 1.0 / alpha) * mbar_mg
    x = m_eff / np.maximum(c * V, 1e-12)
    v = v0 + np.log1p(x)
    I_disp = 0.5 * (x / (1 + x)) ** 2 / v ** 2
    return np.nan_to_num(I_zero), I_disp
