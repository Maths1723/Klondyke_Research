# Roadmap

## Done (24 Nov 2026)
- [x] Located the full YCGC dataset behind the YGS web app (map service); exported all layers
- [x] Audit: units, price basis (incl. inferred), cross-map duplicates, non-grade fields, transcription errors
- [x] Compound Poisson grain-counting likelihood + latent field (Laplace), verified numerically
- [x] Synthetic validation against known truth (grain-counting assays)
- [x] Mid-Dominion: curated 612 holes, lines from geometry, model selection, held-out-line CV, target map
- [x] Full export ingested (6,179 usable holes); official 2020 release compared (older, null-as-zero)
- [x] Production check, 9 upper-Sulphur dredge blocks: no model beats YCGC's estimate; absolute grade not robust
- [x] Synthetic benchmark over 3 seeds (mean ± sd)
- [x] Error bound for drill-based block estimates; grain-mass scan against production (underpowered)
- [x] Twin holes (78 pairs): the nugget is mostly a per-hole error, not grain counting
- [x] Identifiability study: lines vs lines + twins, with / without per-hole error
- [x] Grain mass across 33 campaigns: profile likelihoods + information bound (not identified regionally)
- [x] Mid-Dominion robustness: forced per-hole error (CV unchanged), structural-zero alternative (loses by ~30 log units)

## Next
- [ ] **Post-1966 disturbance check** of every mid-Dominion target on satellite imagery; drop mined ground
- [ ] Profile the structural-zero probability π₀ (the nested fit scored lower, so the optimiser needs checking)
- [ ] Recover van Loon's elevations / ground condition for the 1955-redrawn holes (match on grade + OB + DS)
- [x] MCMC check of the Laplace grain mass: mid-Dominion 10.3 (8.9–11.8) vs Laplace 10.5 mg; synthetic covers truth
- [x] Target-ranking sensitivity to the per-hole error (σ_h = 0.7): rank correlation 0.94, top target unchanged
- [ ] Paper: see `docs/PAPER_OUTLINE.md`
- [ ] Fix the valley frame at the lower S-bend (s > 6.8 km): local axis from the 1958 lines

## Independent validation
- [ ] Reconstruct YCGC's block-estimation method (archive notes for YCGC25152 / 26682); recovery factor
- [ ] Make the valley frame and anisotropy robust (sensitivity of block results to the axis is unacceptable)
- [ ] **Transfer test:** parameters from Dominion, predictions on Sulphur (three campaigns, recorded and inferred prices)
- [ ] **Modern ground truth:** 2021–22 sonic/RC holes and test pit on Sulphur (YMEP22-048)

## Model
- [ ] Block kriging of the latent field over mining-block geometry (targets as tonnes and grams, not points)
- [ ] Campaign/map random effect (τ) for multi-campaign creeks
- [ ] Covariates with a physical rationale: tributary mouths, bedrock gradient (where elevations exist)
- [ ] Non-stationary anisotropy following the valley direction (SPDE), only if residuals require it

## Deliverables
- [ ] 2-page technical summary for operators: what the data say, the validation, the targets, the caveats
- [ ] Target sheet: location (lat/lon), probability, grade range, overburden, distance to data, suggested test
