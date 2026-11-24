# Paper outline (draft, 24 Nov 2026)

**Working title:** *What limits placer drilling? Grain counting, per-hole error
and a century of Klondike drill holes*

**Target:** an applied statistics / mathematical geoscience journal
(Mathematical Geosciences; Natural Resources Research; Spatial Statistics;
Applied Earth Science – Trans. IMM B). Length ~25 pages with appendix.

## Abstract (draft)
Placer gold is evaluated by drilling. Each hole recovers ~0.1–0.3 m³ of gravel
that holds only a handful of gold grains, and a single hole's value is
notoriously unreliable. We model drill-hole assays as a compound Poisson-gamma
draw of grains from the drilled volume under a latent Gaussian field of
log-concentration in valley coordinates, with an optional per-hole
multiplicative error, and fit it by a Laplace approximation with a safeguard
for the non-log-concave likelihood. We apply it to the public record of the
Yukon Consolidated Gold Corporation (13,358 holes, 1907–1966), after
recovering units, unrecorded gold-price bases and duplicated holes from the
data themselves.
- On synthetic creeks, the grain-counting model removes the ~30% downward bias
  of kriged log-grades and cuts RMSE by 15%.
- On Middle Dominion Creek it gives calibrated predictions on held-out drill
  lines and beats kriging (CRPS 1.11 vs 1.21).
- Seventy-eight pairs of holes drilled 3–12 m apart disagree by a typical
  factor of three, about thirteen times the variance grain counting predicts,
  growing as the gold in the sample increases. This reveals a per-hole
  error (σ ≈ 0.6–1.2 on the log scale) that line-spaced drilling cannot
  detect.
- Synthetic experiments show this error is invisible to the marginal
  likelihood without twin holes, yet degrades field estimates by 18% if
  omitted. Grain-mass estimates from assays alone are informative only where
  samples hold few grains, which we quantify with a simple Fisher-information
  argument and confirm on 33 drilling campaigns.
- Against nine dredge blocks with recorded recovery, no model beats the
  company's own estimates; block-level error is dominated by unrecorded
  conventions.
- A closed-form design rule follows. Var[log block mean] ≈ κ²/n, with
  κ² = (1+h²)(1+g²) − 1, places the crossover at which grain counting stops
  mattering, about 16 mg of gold per sample for Dominion gold. It shows that
  lean bench ground with coarse gold needs hundreds of holes per block.

Historic placer drilling can rank ground, but its precision is limited by
per-hole error rather than grain counting. Twinning 10–15% of holes is the
cheapest way to measure it.

## 1. Introduction
- Placer evaluation: churn and sonic drilling, Radford-type volume factors,
  why a single hole is unreliable (BLM 1969).
- Sampling theory for gold: the Poisson/grain-count view (Clifton et al. 1969;
  Gy; Ingamells; François-Bongarçon). These views need duplicate samples, and
  placer drill holes have none.
- Geostatistics of placers: kriging with a constant nugget. The nugget mixes
  grain counting, drilling error and metre-scale geology, and kriging cannot
  tell them apart.
- The YCGC archive is public, large and almost unused statistically.
- Contributions:
  1. an exact measurement model and inference;
  2. curation findings for the archive;
  3. a twin-hole decomposition of the nugget;
  4. identifiability results with design advice;
  5. a production check.

## 2. Data
- YCGC archive (YGS map service): layers, 933 maps, units (feet), price bases
  (recorded, map-level, inferred from co-located redrawn holes at exact
  ratios 1.690/1.708/1.863), cross-map duplicates (redraws), null-as-zero in
  the 2020 shapefile, non-grade layers (thaw/strip layouts).
- Study sets: mid-Dominion (612 holes, 1936); upper Sulphur (376 holes, 9
  dredge blocks with recorded estimate and recovery); twin maps (78 pairs);
  33 campaigns with ≥ 60 usable holes.

## 3. Model
- 3.1 Grain counting: N ~ Poisson(cV/m̄), Gamma(α) grain masses; moments, zero
  probability, trace censoring; score and curvature through E[N|obs],
  Var[N|obs]; evaluation by windowed sums around the mode.
- 3.2 Per-hole error e_i ~ N(0, σ_h²) on log λ.
- 3.3 Latent field: anisotropic Matérn-3/2 in valley coordinates; valley
  frame and drill lines reconstructed from hole geometry.
- 3.4 Laplace approximation with a Fisher-type weight; marginal likelihood
  for hyperparameters.
- 3.5 Signatures and identifiability:
  - grain variance m_eff/(cV) vs constant σ_h²;
  - zero channel information λ²e^{−λ}/(1−e^{−λ});
  - dispersion channel information ½(g/(g+v₀))² per hole;
  - confounding of σ_h with sub-spacing geology.

## 4. Simulation studies
- 4.1 Benchmark vs kriging and GP on log-grades (3 creeks, held-out lines and
  1 km blocks): bias, RMSE, CRPS, coverage, ranking.
- 4.2 Identifiability: σ_h ∈ {0, 0.8} × design {lines, lines + 15% twins}.

## 5. Results on the Klondike record
- 5.1 Mid-Dominion: parameters (ℓ_s 130 m, ℓ_n 33 m, m_eff ≈ 10 mg),
  held-out CV, robustness to σ_h, mined vs unmined ground (45% of floor, 78% of
  gold).
- 5.2 Twin holes: observed vs simulated disagreement; field fits with and
  without σ_h on the four twin-rich maps.
- 5.3 Grain mass across 33 campaigns: profile intervals vs predicted
  information; no distance trend; what can and cannot be said.
- 5.4 Production: nine upper-Sulphur dredge blocks; predicted vs observed
  block error; why nine blocks cannot measure grain size.
- 5.5 Design rule: κ²/n, crossover m_eff/h², holes per block by ground type;
  Monte Carlo check (±30% for n ≥ 16).

## 6. Discussion
- What limits placer drilling: per-hole error, then the conventions
  (estimation factor, recovery), then grain counting.
- Use of historic data: ranking and locating ground, not forecasting grade.
- Design: twin 10–15% of holes; report volume recovery per hole.
- Limitations:
  - Laplace checked by MCMC at Dominion-like counts; not at very low counts;
  - twin pairs from four maps;
  - valley frame;
  - unrecorded conventions.

## 7. Conclusion

## Appendices
- A. Likelihood: derivation, derivatives, numerical verification.
- B. Price-basis inference.
- C. Drill-line reconstruction (radius graph; ARI 1.0 on plan 1103).
- D. Data and code availability.

## Figures
1. Study area: campaigns, mid-Dominion, upper Sulphur, twin maps.
2. Archive audit: price-ratio histogram of co-located redrawn holes.
3. Synthetic benchmark (bias and RMSE by method).
4. Mid-Dominion: unrolled valley strip with drill grades, model, mined ground.
5. Twin holes: |log ratio| vs gold per sample, with simulated envelopes; Δ log Z.
6. Identifiability: estimates by design and model.
7. Grain-mass profiles by campaign vs predicted information.
8. Upper Sulphur: predicted vs recovered block grades; error-bound scan.
9. Design regime map: holes per block for ±20% vs gold per sample.

## Still to do before writing
- [x] MCMC check of the Laplace grain mass (agrees within 3% on mid-Dominion and synthetic)
- [x] Target-ranking sensitivity to σ_h on mid-Dominion (rank corr. 0.94)
- [ ] Figures in a consistent publication style
