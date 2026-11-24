"""Synthetic placer valley with a known paleochannel, sampled like YCGC drilling.

Purpose: before trusting any method on real data, check it recovers a known
truth. The generator mimics the structure described by van Loon (2016, 2017):
a narrow (~25-50 m), sinuous pay channel offset from the valley axis, patchy
grade along it, a large nugget component, ~10-15% zero assays, sampled on
cross-valley drill lines about one claim (~150 m) apart with holes ~30-50 m
apart.

Nothing here is calibrated to real Klondike values yet; the parameters are
plausible shapes, not estimates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SynthParams:
    length_m: float = 9000.0
    halfwidth_m: float = 300.0
    line_spacing_m: float = 150.0
    hole_spacing_m: float = 40.0
    channel_width_m: float = 35.0       # sd of the Gaussian cross profile
    channel_offset_frac: float = 0.35   # mean offset of the pay channel (left limit)
    channel_meander_amp_frac: float = 0.3
    channel_meander_wl_m: float = 1800.0
    patch_corr_m: float = 600.0         # along-channel correlation of richness
    patch_sd: float = 0.8
    base_log: float = 3.0               # log1p(accum mg/m2) background
    peak_log: float = 4.0               # added on the channel axis
    nugget_sd: float = 0.7
    zero_threshold: float = 2.0         # latent below this -> assay of zero
    seed: int = 1


def _raw_curve(u: np.ndarray) -> np.ndarray:
    return np.column_stack([u, 400 * np.sin(2 * np.pi * u / 4000.0)
                            + 150 * np.sin(2 * np.pi * u / 1300.0 + 0.7)])


_U = np.linspace(-500, 15000, 200_001)
_P = _raw_curve(_U)
_ARC = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(_P, axis=0).T))])
_ARC -= np.interp(0.0, _U, _ARC)  # arc length 0 at u = 0


def valley_centreline(s: np.ndarray) -> np.ndarray:
    """Centreline point at ARC LENGTH s (so s is true distance down the valley)."""
    u = np.interp(np.asarray(s, float), _ARC, _U)
    return _raw_curve(u)


def _gp_1d(s, corr, sd, rng):
    # exponential-covariance process on a grid via AR(1) recursion
    order = np.argsort(s)
    ss = s[order]
    out = np.empty_like(ss)
    out[0] = rng.normal(0, sd)
    for i in range(1, len(ss)):
        rho = np.exp(-(ss[i] - ss[i - 1]) / corr)
        out[i] = rho * out[i - 1] + np.sqrt(1 - rho ** 2) * rng.normal(0, sd)
    res = np.empty_like(out)
    res[order] = out
    return res


def make_dataset(p: SynthParams = SynthParams()) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(p.seed)
    # fine grid along s for the latent channel
    sg = np.arange(0, p.length_m + 1, 10.0)
    richness = _gp_1d(sg, p.patch_corr_m, p.patch_sd, rng)
    phase = rng.uniform(0, 2 * np.pi)
    n_c = p.halfwidth_m * (p.channel_offset_frac
                           + p.channel_meander_amp_frac * np.sin(2 * np.pi * sg / p.channel_meander_wl_m + phase))

    def latent(s, n):
        r = np.interp(s, sg, richness)
        c = np.interp(s, sg, n_c)
        return p.base_log + (p.peak_log + r) * np.exp(-0.5 * ((n - c) / p.channel_width_m) ** 2)

    rows = []
    line_s = np.arange(p.line_spacing_m / 2, p.length_m, p.line_spacing_m)
    for k, s0 in enumerate(line_s):
        s0 = s0 + rng.normal(0, 15)
        hw = p.halfwidth_m * rng.uniform(0.6, 0.95)  # lines often stop short of the valley edges
        nn = np.arange(-hw, hw + 1, p.hole_spacing_m) + rng.normal(0, 5)
        for h, n in enumerate(nn):
            rows.append((f"{k + 1}-{h + 1}", k + 1, s0, n))
    df = pd.DataFrame(rows, columns=["hole_id", "claim_no", "s_true", "n_true"])

    # map coordinates: offset n perpendicular to the curving centreline
    C = valley_centreline(df.s_true.to_numpy())
    C2 = valley_centreline(df.s_true.to_numpy() + 1.0)
    t = C2 - C
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    left = np.column_stack([-t[:, 1], t[:, 0]])
    xy = C + left * df.n_true.to_numpy()[:, None]
    df["x"], df["y"] = xy[:, 0], xy[:, 1]

    lat = latent(df.s_true.to_numpy(), df.n_true.to_numpy())
    obs = lat + rng.normal(0, p.nugget_sd, len(df))
    obs = np.where(obs < p.zero_threshold, 0.0, obs)
    df["latent"] = lat
    df["y_log"] = obs                        # log1p(accum mg/m2)
    df["accum_mg_m2"] = np.expm1(obs)
    df["line_id"] = df.claim_no.astype(str)
    truth = {"sg": sg, "n_c": n_c, "richness": richness, "latent_fn": latent, "params": p}
    return df, truth


@dataclass
class PhysParams:
    """Grain-counting synthetic: the assay is a draw of gold grains from the sample volume."""
    log_c_background: float = 4.0     # ln(mg/m3): ~55 mg/m3 ~ 2.8 c/yd3
    log_c_peak_add: float = 3.5       # added on the channel axis: x33 -> ~1800 mg/m3 ~ 90 c/yd3
    richness_sd: float = 0.6
    micro_sd: float = 0.0             # optional extra small-scale geological noise (ln units)
    mbar_mg: float = 1.0              # mean grain mass
    alpha: float = 0.6                # Gamma shape: m_eff = (1+1/alpha) mbar
    section_ft_median: float = 8.0
    section_ft_logsd: float = 0.4
    seed: int = 11


def make_physical_dataset(pp: PhysParams = PhysParams(), sp: SynthParams | None = None):
    from . import sampling, units
    sp = sp or SynthParams(seed=pp.seed, patch_sd=pp.richness_sd)
    df, truth = make_dataset(sp)
    rng = np.random.default_rng(pp.seed + 1000)
    shape = np.exp(-0.5 * ((df.n_true.to_numpy() - np.interp(df.s_true, truth["sg"], truth["n_c"]))
                           / sp.channel_width_m) ** 2)
    rich = np.interp(df.s_true, truth["sg"], truth["richness"])
    logc = pp.log_c_background + (pp.log_c_peak_add + rich) * shape
    logc_obs_site = logc + rng.normal(0, pp.micro_sd, len(df)) if pp.micro_sd > 0 else logc
    section_ft = np.clip(np.round(np.exp(np.log(pp.section_ft_median) + rng.normal(0, pp.section_ft_logsd, len(df)))), 2, 40)
    V = sampling.sample_volume_m3(section_ft)
    S, N = sampling.simulate(np.exp(logc_obs_site), V, pp.mbar_mg, pp.alpha, rng)
    grade_mg = S / V
    cents = grade_mg / units.cents_per_yd3_to_mg_per_m3(1.0)
    rec = np.round(cents, 1)
    rec = np.where((cents > 0) & (rec < 0.1), 0.1, rec)          # gold seen but below 0.1 c -> "trace"
    out = df[["hole_id", "claim_no", "line_id", "s_true", "n_true", "x", "y"]].copy()
    out["dredge_section_ft"] = section_ft
    out["dredge_section_m"] = section_ft * 0.3048
    out["grade_cents_yd3"] = rec
    out["grains"] = N
    out["logc_true"] = logc
    truth.update(logc_fn=lambda s, n: pp.log_c_background + (pp.log_c_peak_add + np.interp(s, truth["sg"], truth["richness"]))
                 * np.exp(-0.5 * ((n - np.interp(s, truth["sg"], truth["n_c"])) / sp.channel_width_m) ** 2),
                 phys=pp)
    return out, truth
