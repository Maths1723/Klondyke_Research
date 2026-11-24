# Research decision log

Each entry: what was decided, why, and what would change it. Dates are when
the decision was made.

---

### 2026-11-02 — Research question
**Can 1923–1966 YCGC churn-drill data, modelled with honest spatial
statistics, predict placer gold accumulation at drill locations the model has
not seen, well enough to rank ground for an operator?**
Stated as a prediction-and-validation question, not "make a prospectivity map",
because a map without a validation score cannot be checked.

### 2026-11-02 — Start with mid-Dominion Creek (claims 76–177)
- ~450 holes over ~9 km from one 1936 campaign (maps 1103–1105): one set of
  drilling biases rather than several.
- Bedrock elevation recorded for nearly every hole.
- A published attribute table (van Loon 2016, App. 1) to cross-check our data.
- Dredge estimate vs actual production for 10 blocks: an independent check.
Sulphur Creek (1,134 holes, 3 campaigns) is kept for the **transfer test**;
lower Hunker (1,005 holes) is a third creek. *Would change if* the GDB lacks
Dominion drill holes or Sulphur turns out to be much cleaner.

### 2026-11-02 — Target = log1p(accumulation, mg/m²), not dollars or grade
Accumulation (grade × dredge-section thickness) is gold per unit area, what an
operator mines; it removes the thick-gravel dilution van Loon (2017) discusses
at the mouth of Sulphur Creek. Fixed-price dollars are converted to mass
(1 c/yd3 at $20.67 = 19.68 mg/m3). log1p keeps zeros and compresses the
nugget-driven tail. *Would change if* miners say gold is concentrated in the
bottom ~1 m regardless of section (then bottom-metre grade is the target).

### 2026-11-02 — Elevations treated as feet on a local datum
Checked on 20 holes from van Loon (2016) App. 1: thickness / elevation
difference has median 0.317 vs 0.3048 m/ft. The paper does not say so. The
check is automated (`qc.infer_elevation_unit`). Until a datum tie exists, only
datum-free bedrock features are used (bedrock relative to the lowest bedrock on
the same drill line).

### 2026-11-02 — Validation is spatial, never random
Random hold-out leaks: holes on one drill line are ~30–50 m apart. Two schemes:
leave-one-line-out (interpolation between drilled lines) and 1 km along-valley
blocks with a 150 m buffer (extrapolation). Scores: CRPS (accuracy + honest
uncertainty), 50/90% interval coverage, Spearman rank correlation, and share of
total gold captured in the top 20% of ranked holes.

### 2026-11-02 — Baselines before models
Global median, nearest neighbour, IDW (the YGS method), kriging in map
coordinates. A model counts as an improvement only if it beats these on the
same splits.

### 2026-11-24 — Valley coordinates built from the drill lines themselves
Each YCGC line crosses the valley, so the sequence of line midpoints traces
the valley axis without any external stream layer. Every hole gets s (distance
down valley) and n (signed offset, + = left limit). On synthetic data the
estimated axis is within ~25 m of the truth (median |n error|).

### 2026-11-24 — Test every method on synthetic data with known truth first
Synthetic valley: sinuous ~35 m-wide pay channel offset to the left limit,
patchy along-channel richness (600 m correlation), large nugget, 4–8% zero
assays (fewer than Sulphur's 12.5%), sampled on YCGC-style lines. Result (3 seeds):
- **Interpolation (LOLO):** kriging / GP in valley coordinates best
  (CRPS ≈ 0.67 vs 0.77 for kriging in map coordinates and 0.98 for IDW),
  with calibrated 90% intervals (coverage 0.90).
- **Extrapolation (1 km blocks):** every method is close to useless
  (Spearman ≤ 0.16). Expected when the channel meanders on a ~1.8 km
  wavelength: data cannot tell you where it goes 500 m beyond the last line.
- **Gradient boosting** is worse than kriging and its intervals under-cover
  (0.67–0.87 for a nominal 0.90). Not a primary model.
**Consequence:** claims about ground far from drilling must come from
geology (tributaries, bedrock form), not interpolation, and the maps say so.

### 2026-11-24 — Kriging in valley coordinates is the workhorse; GP is the check
They score the same; kriging is ~15× faster. The GP's maximum-likelihood
length scales give the along/across correlation ranges directly (synthetic:
~200 m along vs ~50 m across). Non-stationary SPDE models (R-INLA) are the
next step only if real-data residuals show the anisotropy direction changing.

### 2026-11-24 — Targets use block-level, not single-sample, uncertainty
A drill sample carries nugget variance that averages out over a mining
block. Target probabilities use the smooth-field variance (nugget removed);
`--point` gives the single-sample version. Proper block kriging over a defined
block geometry is on the roadmap.

### 2026-11-24 — Gold threshold for targets is a placeholder
50 c/yd3 over 2 m (≈1,970 mg/m²). The economic cut-off is an operator
question (OPEN_QUESTIONS #10); the code takes it as a parameter.

### 2026-11-24 — Data come from the YGS map service, not the zip downloads
The open-data zips sit behind a bot check that automated tools must not
bypass. The same data are published as a public ArcGIS map service
(`deptweb.gov.yk.ca/.../YGS_HistoricPlacer/Historic_placer_data/MapServer`),
found through the official web app. Layers 0–3 exported in full (13,358 holes,
5,352 polygons, 933 maps). Transfers into the analysis environment are
verified by SHA-256 against the browser copy.

### 2026-11-24 — Price basis is recovered, not assumed
Grades are normalised to $20.67 using the recorded price, else the map's
recorded price, else a price **inferred** from co-located redrawn holes
(ratio 1.690/1.708/1.863). Without this, YCGC25097 (Dominion benches, 1960)
reads 69% richer than the same 1936 holes. *Would change if* the archive
documents a different revaluation convention.

### 2026-11-24 — Mid-Dominion dataset = 1936 plan 1103 + 1955 redraws of plans 1104–1105
Verified hole-by-hole against van Loon (2016) App. 1. Thaw/strip layouts
excluded (values shown not to be grades); 1960 $35 revaluation excluded as a
duplicate; "unknown location", undated-price and 1914 shaft maps excluded.
612 holes after curation; every rule is in `scripts/05_dominion_real.py`.

### 2026-11-24 — Drill lines are recovered from geometry, not labels
1955 maps number holes without line prefixes. Holes on a line are ~60 m apart
(90th pct 61 m), lines ~150 m apart, and lines follow claim baselines (up to
~30° off perpendicular to the valley). A 70 m radius graph without an angle
rule reproduces the plan-1103 line labels exactly (adjusted Rand index 1.0).
Lines define the cross-validation folds, so this matters for honest scores.

### 2026-11-24 — Measurement model: compound Poisson grain counting
Assays are modelled as gold grains drawn from the drill sample (Clifton et al.
1969 generalised to unequal grains) under a latent log-concentration field.
Exact likelihood, verified numerically; Laplace inference with a Fisher-type
safeguard because the likelihood is not log-concave. Chosen because it makes
three testable predictions that a constant-nugget model does not, and because
it separates block-level geology from single-sample noise, which is what a
target map needs.

### 2026-11-24 — Model components chosen by marginal likelihood on the real data
Grain counting alone beats adding a drilling (core-recovery) error and a
structural-zero probability on mid-Dominion. The all-maps pair statistics
that motivated those components mixed campaigns and creeks; on one campaign
they are not needed. Re-test on Sulphur (several campaigns).

### 2026-11-24 — Targets are reported only where the valley frame is reliable
Downstream of s ≈ 6.8 km the frame folds at an S-bend and holes are sparse;
targets there are flagged, not ranked with the rest. Every target carries its
distance to the nearest hole and its predicted overburden.

### 2026-11-24 — The live map service, not the official download, is the source of record
The open-data zip is the 2020-01-07 release: 2,632 fewer holes, and the
shapefile stores nulls as zeros (80% of its zero grades are unassayed holes).
Values agree exactly where both exist, so the service is a strict superset.
Anyone reproducing this work from the zip will get different, and worse,
results; `01b_ingest_service.py` documents the comparison.

### 2026-11-24 — Absolute grades are not reported as forecasts until calibrated against production
On nine upper-Sulphur dredge blocks, no model beats YCGC's own estimate, and
model block grades move by tens of percent with defensible analysis choices
(valley axis, disputed price basis). Targets are therefore ranked by
probability relative to local drilling; absolute grade and tonnage wait for a
local calibration (YCGC estimation factor, dredge or plant recovery), ideally
from a modern test pit. *Would change if* the archive documents YCGC's
estimation method and recovery, or a test pit gives a direct check.

### 2026-11-24 — Price conflict on 26683/26743 resolved by evidence, with a sensitivity run
84 co-located holes carry identical grades on both maps, although 26743
records $35 and 26683 nothing; YCGC valued the 1950–58 blocks at $20.67.
Primary analysis uses $20.67 for both; the $35 alternative is reported
alongside.

### 2026-11-24 — Twin holes are the evidence for the nugget, not the line-spaced data
78 pairs of holes 3–12 m apart on the same map (Bear 1912, Gold Run 1941, lower
Hunker, a 1933 dredge-area map) disagree by a typical factor ~3. That is about
13× the variance grain counting predicts, and the disagreement rises with the
gold in the sample, where grain counting requires it to fall. A per-hole
multiplicative error (σ_h ≈ 0.6–1.2) is therefore part of the measurement
model (`physical.CPGField(drill_noise=True)`). *Would change if* the archive
shows these pairs were drilled with a different rig or factor convention, or
were re-drills chosen because the first hole looked wrong.

### 2026-11-24 — Mid-Dominion: base model stays primary, σ_h = 0.7 run reported alongside
Line-spaced drilling cannot detect a per-hole error (synthetic: Δ log Z 0.4 for
a real σ_h = 0.8), so its absence on Dominion is not evidence. Forcing the twin
value changes neither the grain mass (10.3 vs 10.5 mg) nor the held-out scores
(CRPS 1.107 vs 1.115). The base model stays primary because the targets were
built and checked with it. Checked: under σ_h = 0.7 the target probabilities
on unmined ground keep rank correlation 0.94, 84% of the top-fifth cells are
shared, and the top target is unchanged. *Would change if* twin holes are
found on Dominion itself and give a larger σ_h.

### 2026-11-24 — No regional grain-size map from assays alone
Per-campaign grain-mass estimates (33 campaigns, ≥ 60 holes) range over three
orders of magnitude and track dredge-section thickness (ρ = −0.49, p = 0.004)
and zero share, not distance from King Solomon Dome (ρ = −0.09). That is the
signature of weak identifiability. Grain size is reported only where profile
intervals are closed. Gold-grain studies (Chapman et al.) remain the source
for regional grain size.

### 2026-11-24 — Recommendation for any new drilling: twin 10–15% of holes
On synthetic creeks with σ_h = 0.8, twins at 15% of holes make the per-hole
error detectable (Δ log Z 8.2) and stabilise the grain mass (sd 3.1 → 0.1 mg).
It is the cheapest way to measure what limits block estimates.

### 2026-11-24 — Laplace inference is kept; MCMC is the check, not the method
With the spatial hyperparameters fixed, an elliptical-slice / Metropolis
sampler gives m_eff 10.3 mg (90%: 8.9–11.8) on mid-Dominion against the
Laplace 10.5 mg. On a synthetic creek it gives 9.4 (7.5–12.0) against Laplace
9.2 and truth 10.5. Laplace is ~1000× faster, and that speed is what makes
cross-validation and 33-campaign profiles feasible. *Would change if* a
low-count setting (thin sections, coarse gold) shows a larger gap. That check
has not been run.

### 2026-11-24 — Design rule reported as a lower bound
Holes per block n ≈ κ²/s² ignores spatial variation inside the block, so it
is a lower bound. It is within ±30% of Monte Carlo for n ≥ 16 and optimistic
by up to 2× when samples hold less gold than one effective grain mass.

### 2026-11-24 — Recorded "prices" that are not gold prices are ignored
Maps 26412 ($1.00) and 26438 ($7.60, $15.60) carry recorded values that are
not historical gold prices. They are now ignored, and those maps fall back to
$20.67 like any map without a recorded basis. Before this fix, 26438's grades
were multiplied by 2.7. Twin-hole ratios are unaffected. The fix changes which
copy of 36 redrawn holes is kept, and shifts the regional correlations by at
most 0.04. *Would change if* the source maps show what these numbers are.

### 2026-11-24 — Paper: Natural Resources Research
Target journal chosen for fit (quantitative resource evaluation, IAMG
journal, case studies welcome) and for odds of acceptance. The manuscript is in
`paper/` (LaTeX, figures from `paper/make_figures.py`).

### 2026-11-24 — Review-driven checks before submission
- **Twin inference resamples whole maps.** 38 of the 78 pairs are on one map,
  so pair-level tests overstate certainty. Cluster bootstrap and a within-map
  permutation test are reported (`scripts/17_twin_uncertainty.py`). The excess
  over Dominion-like grain counting survives (12.7×, 90% interval 3.7–23.5;
  11× without Bear Creek). On size alone, very coarse gold is not excluded;
  the trend with gold content excludes it.
- **Baselines get the standard log-normal correction** (`scripts/18_baseline_check.py`).
  It barely moves kriging and over-corrects the GP. The grain-counting field
  stays 10% better in RMSE than the best corrected baseline.
- **Two MCMC chains on mid-Dominion**, one from a dispersed start: R-hat 1.04
  over the converged window. The low-count benchmark case did not converge and
  is reported as open.
- **Every number in the manuscript is checked by `paper/check_numbers.py`**
  against the outputs. The first run found ten mis-rounded or stale values,
  all now corrected.
