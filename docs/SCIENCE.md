# Technical note: from churn-drill assays to in-situ gold

*Working document, 24 Nov 2026.*

## 1. Geological setting: what the drill holes sample

**Dominion, Sulphur and Gold Run creeks** drain the south side of King Solomon
Dome into the Indian River, in unglaciated terrain. They have produced over
2.6 Moz since 1896 (Froese et al. 2001). Four gravel units matter
(Froese, Enkin & Smith 2001):

| Unit | Age | Setting | Relevance |
|---|---|---|---|
| White Channel gravel | Pliocene, >2.6 Ma | high benches 20–40 m above the valley floor | bench placers (YCGC "Dominion bench" maps) |
| Ross gravel | early Pleistocene, >780 ka | incised-valley fill **below** modern creek level, ~80% vein quartz | the oldest and largest placer source on Dominion |
| Dominion Creek gravel | Brunhes, <780 ka (¹⁴C 47–6 ka) | lateral-accretion gravels of a wandering stream over the Ross gravel | reworks older gold |
| Creek / gulch gravel + "muck" | late Pleistocene–Holocene | up to 15 m of loess and organic silt over gravel | the "overburden" in YCGC logs |

Consequences for the model:

- **The pay is inherited.** Little gold has come off bedrock on Dominion in
  the last ~800 ka; the modern creek has reworked an older channel's lag. This
  explains van Loon's (2016) observation that the Dominion pay channel sits off
  the bedrock low: its position was set by an earlier thalweg. **Bedrock
  topography should therefore be a weaker predictor than along-valley
  continuity**, which the fitted correlation lengths can test.
- **The gold is fine, flat and travelled** (mostly <2 mm, well rounded,
  ~10–15 km transport; fineness 800–900 on Dominion, rising downstream).
  Corey shape factors at lower Dominion average ~0.1, i.e. very flaky, and gold
  <125 µm is negligible (Chapman, Bond & LeBarge 2010). Flake mass ≈
  ρ·(π/4)d²·t: ~0.3 mg at d = 0.5 mm, ~2 mg at 1 mm, ~15–40 mg at 2–3 mm.
- **Coarse-gold anomalies** (mouth of Brimstone Gulch; Laskey Creek for Gold
  Run) point to local sources, so tributary mouths are physically motivated
  covariates, not just pattern-matching.

## 2. The measurement: what a YCGC grade is

YCGC churn-drilled with Keystone-type rigs (casing and a drive shoe in thawed
ground, open hole in frozen ground; Hester 1962 in van Loon 2017). A grade in
cents per cubic yard is the value of gold panned from the dredge-section
interval, divided by a calculated sample volume and valued at a fixed price.
Three things sit inside that number:

1. **Volume convention.** The theoretical volume of a 7.5-inch shoe is 0.3068
   ft³ per foot of drive; the widely used **Radford factor** takes 0.27 ft³
   (≈1/100 yd³), an arbitrary ~12% upgrade (BLM Tech. Bull. 4, 1969). Which one
   YCGC used is an open question (OPEN_QUESTIONS #2–3). It is a single
   multiplicative factor and does not affect rankings; it does affect absolute
   grade and the inferred grain mass (§3).
2. **Price basis.** The map service records $20.67, $35, $35.30 and $38.50 on
   different maps. Grades are normalised to $20.67 before use; maps with no
   recorded price are assumed $20.67 and flagged.
3. **Grain counting.** A sample of ~0.03–0.3 m³ from ground grading tens of
   cents per yard contains only a handful to a few hundred gold grains. That
   is the nugget effect, and it has a known form.

## 3. Measurement model: compound Poisson grain counting

Let c(x) be the in-situ gold concentration (mg/m³) and V the sample volume.

$$N \sim \text{Poisson}(\lambda),\quad \lambda = c\,V/\bar m,\qquad
m_k \overset{iid}{\sim} \text{Gamma}(\alpha, \bar m/\alpha),\qquad
S = \sum_{k=1}^{N} m_k,\qquad G = S/V.$$

**Moments.** By the law of total variance,
$E[S] = cV$ and $\operatorname{Var}[S] = \lambda E[m^2] = cV\,m_{\text{eff}}$ with
$m_{\text{eff}} = E[m^2]/E[m] = (1+1/\alpha)\,\bar m$. Hence

$$\text{CV}^2(G) = \frac{m_{\text{eff}}}{c\,V},\qquad P(G=0) = e^{-cV/\bar m}.$$

This is the Poisson argument of Clifton, Hunter & Phillips (1969) (relative
error ≈ X^{-1/2} for X grains; ≥20 grains for ±50% at 95%) generalised to
unequal grains, where the effective grain mass $m_{\text{eff}}$ — dominated by
the coarse fraction — replaces the grain count. Three testable predictions
follow, none of which a constant-nugget kriging model makes:

- **P1.** Assay scatter shrinks with grade: residual variance on the log scale
  ≈ log(1 + m_eff/(cV)).
- **P2.** Assay scatter shrinks with section thickness (bigger V).
- **P3.** Zero assays concentrate where cV/m̄ is small, at a rate e^{-λ}.

**Likelihood.** S is compound Poisson-gamma. For an observed mass s > 0,

$$p(s\mid\lambda) = \sum_{n\ge1} \frac{e^{-\lambda}\lambda^n}{n!}\,
\text{Gamma}(s;\,n\alpha,\,\bar m/\alpha),$$

a "trace" assay is the censored event 0 < S < s_tr (Gamma CDF in place of the
density), and a zero is e^{-λ}. Differentiating in log λ,

$$\frac{\partial \log p}{\partial \log\lambda} = E[N\mid\text{obs}] - \lambda,\qquad
\frac{\partial^2 \log p}{\partial (\log\lambda)^2} = \operatorname{Var}[N\mid\text{obs}] - \lambda,$$

so inference needs only the posterior over the unseen grain count. The sum is
evaluated on a window of ±12 posterior sd around the mode n*, found from
log λ + α log(s/θ) = ψ(n+1) + α ψ(nα) by bisection. Verified: probabilities
sum to 1 to 10⁻⁶, analytic derivatives match finite differences.

**Identifiability.** λ involves V/m̄, so an error in the assumed drill volume
is absorbed by m̄: inferred grain masses are relative to the assumed shoe
area. The ratio m_eff/m̄ = 1 + 1/α (grain-size spread) is volume-free.

Note the second derivative can be positive (the likelihood is not log-concave
in log λ when α is small), so the Laplace approximation below uses the
Fisher-type weight W = max(λ − Var[N|obs], ε).

**Per-hole error.** A drill hole is not only a grain draw. The volume actually
sampled differs from the theoretical shoe volume (caving, packing, boulders),
engineers corrected each hole by judgement, the BLM (1969) bulletin describes
upward corrections of up to 2×, and a hole that catches or misses a bedrock
crevice samples different gravel from one 5 m away. All of this multiplies
the concentration seen by one hole and not its neighbour:

$$\log\lambda_i = f(x_i) + \log V_i - \log\bar m + e_i,\qquad e_i \sim N(0,\sigma_h^2).$$

Grain counting and the per-hole error have different signatures. Grain-count
variance falls as m_eff/(cV) when there is more gold in the sample; per-hole
variance is the same at every grade and section. Spatially, both are
uncorrelated between holes, so with holes 40–60 m apart they are also
confounded with geology at scales below the hole spacing. §6.5–6.6 show that
only close-spaced (twin) holes separate them.

The model puts the per-hole factor e^{e_i} at median one, so the field f is
a *median*. Suppose the extra gold is real (metre-scale geology), or the
factor is an unbiased measurement error (mean one). Then a block's mean
concentration is exp(f + σ_h²/2), 28% above the median field for σ_h = 0.7.
Only a factor biased high on average, such as sloughing of gravel into the
casing, would need no correction. The data cannot tell these apart, which is
one more reason absolute grades are not reported until they are calibrated
against production (§6.3).

## 4. Spatial model

$$f(x) = \log c(x) \sim \mathcal{GP}\big(\beta_0,\ \sigma^2\,\text{Matérn}_{3/2}(\Delta s/\ell_s, \Delta n/\ell_n)\big)\ [+\ \tau^2\,\mathbb 1\{\text{same source map}\}]$$

in valley coordinates (s down-valley, n across, + = left limit). The optional
map effect absorbs campaign-level biases (price basis, volume convention,
drilling practice). Hyperparameters (σ, ℓ_s, ℓ_n, β₀, m̄, α, τ) maximise the
Laplace-approximate marginal likelihood (Rasmussen & Williams 2006, Alg. 3.1,
with backtracking line search).

**Why this, and not kriging of log-grades.** Kriging log1p(grade) treats
grain-count noise as a constant nugget and estimates E[log1p G], which for
low-count samples is biased below log c (Jensen, and zeros). The latent model
estimates c itself, the quantity a mining block sees, and assigns each assay
its own noise level from its grade and volume.

**Relation to other fields.** Fisheries catch data have the same structure:
many zeros, a skewed positive part, spatial anisotropy. There it is handled
with Tweedie (compound Poisson-gamma) likelihoods on SPDE random fields
(sdmTMB, Anderson et al.). Our likelihood differs in one physical respect: the
grain-mass distribution is fixed by geology, not tied to the mean, so
Var[S] ∝ E[S] (variance power 1), not the Tweedie μ^p with 1<p<2.

## 5. What the data look like (map-service audit, 24 Nov 2026)

The YGS "Historic placer data" map service holds 13,358 point records on 340
source maps, 5,352 dredge/thaw/cut polygons and a 933-map index with placement
confidence. Findings, each checked against the records themselves:

- **Units are feet** throughout; (OB + DS)/(surface − bedrock) = 1.04–1.10 on
  every map with all four fields, the excess being bedrock dug as part of the
  dredge section.
- **Duplicates across maps.** 675 holes appear on two maps with identical
  non-zero grade, median 4.8 m apart (90th pct 10.6 m): a direct estimate of
  relative georeferencing error.
- **Unrecorded price bases, recovered from internal evidence.** Redrawn holes
  on a map revalued at $35 / $35.30 / $38.50 read exactly 1.690× / 1.708× /
  1.863× the $20.67 original. Henderson Creek maps with recorded prices
  confirm the method (61 pairs at 1.863×, 38 at 1.690×); it then assigns
  **YCGC26343 = $38.50** and **YCGC25097 = $35** although neither records a price.
  Unnormalised, those maps would read 69–86% richer for the same gold.
- **The official download is an older, lossy release.** The zip linked from
  the open-data page contains a geodatabase dated 2020-01-07 with 10,726 holes;
  the live service has 13,358 (105 more maps, e.g. the 1941 Dominion–Gold Run
  examination with 300 holes). Where both hold a value they agree exactly. But
  the shapefile writes nulls as zero: **3,339 of its 4,157 zero grades (80%) are
  holes that were never assayed**, stored as the literal `0.00000000000e+00`,
  plus null-as-zero in 2,405 dredge sections, 1,611 overburdens and ~7,000
  elevations. A model built on the official download would see a creek about
  five times more barren than the records say.
- **Middle Dominion.** Plan 1103 (YCGC26199, 1936) matches van Loon (2016)
  App. 1 exactly, and its elevations (median 906 ft = 276 m) sit on a **local
  datum** far below the real valley floor. Plans 1104–1105 are not indexed under
  their own titles, but the **1955 "dredge limit" maps (YCGC26222/26223/26225a/b)
  are redraws of that 1936 drilling**: van Loon's holes 125-6 (105.2 c, 4/6 ft),
  128-4 (275.3 c) and 134-4 (368.7 c) appear on them unchanged. Her full
  ~450-hole dataset is therefore recoverable from public data.
- **Values that are not grades.** The dredge No. 10 thaw/strip layouts
  (1941–62) carry small integers in the grade field; at 21 locations within
  10 m of a 1936 hole they read median 2 against median 33.4 c/yd³. Excluded.
  Map YCGC26722 likewise disagrees with co-located holes (4 vs 110 c/yd³).
- **Transcription errors.** In the public record, hole 85-4 carries hole 85-8's
  values (136.6 c, 17/9 ft, 907/885 ft); van Loon gives 85-4 = 2.7 c. One copy of
  82-7 has bedrock 987 ft above a surface of 909 ft; a 1958-map hole has surface
  282.5 ft for ~780 ft. Each is corrected or nulled and logged.

## 6. Results

### 6.1 Synthetic creek with grain-counting assays (known truth)
720 holes on YCGC-style lines, median 5 grains per sample. Scored against the
true log c at held-out drill lines; mean ± sd over 3 independent creeks
(seeds 11–13):

| method | RMSE | bias | CRPS | 90% cov. | gold in top 20% | ρ within channel |
|---|---|---|---|---|---|---|
| kriging log-grade | 0.848 ± 0.015 | −0.35 ± 0.00 | 0.508 ± 0.019 | 0.97 | 0.70 | 0.31 ± 0.02 |
| GP log-grade | 0.837 ± 0.012 | −0.33 ± 0.01 | 0.460 ± 0.006 | 0.91 | 0.68 | 0.33 ± 0.07 |
| **grain-counting field** | **0.719 ± 0.004** | **−0.10 ± 0.04** | **0.411 ± 0.003** | 0.96 | 0.70 | **0.35 ± 0.08** |

Kriging log-grades **understates in-situ concentration by ~30%** (bias −0.34
in ln c): with few grains per sample, the log of a noisy assay is biased low
(Jensen, plus zeros). The grain-counting model removes most of the bias and
cuts RMSE by 15%; the ranking of rich ground (gold in the top 20%) is the same
for all three.

**Fairer baselines** (`scripts/18_baseline_check.py`). The standard log-normal
correction adds half the nugget to a kriged log-grade.
- Kriging: the variograms fitted at 40 m spacing have little or no nugget, so
  the correction barely helps (bias −0.27, RMSE 0.832).
- GP: the white-noise variance is 0.75–1.06, and the correction over-corrects
  (bias +0.19, RMSE 0.798 ± 0.042). Grain-counting noise on the log scale is
  skewed, not Gaussian.
- The grain-counting field stays 10% better in RMSE than the best corrected
  baseline (0.719) and better in CRPS (0.411 vs 0.453). Over 1 km held-out blocks no method extrapolates the meandering
channel (within-channel ρ 0.03–0.05 for every method). The Laplace estimate of
m_eff is ~2× low in this low-count setting (1.2 vs 2.7 mg; not yet checked by
MCMC). At Dominion-like grain masses it is 10–20% low, and §6.9 shows that
shortfall is not caused by the approximation.

### 6.2 Real data: Middle Dominion Creek, 1936 drilling
612 holes (plan 1103 + the 1955 redraws of plans 1104–1105, $20.67), 71 drill
lines recovered from geometry alone (a 70 m radius graph reproduces the plan
1103 line labels exactly, adjusted Rand index 1.0), 12.5 km of valley, 11.8%
zero assays.

**Fitted model (selected by marginal likelihood):** grain counting only.
- correlation lengths **ℓ_s ≈ 130 m along, ℓ_n ≈ 33 m across** the valley (4:1),
  consistent with van Loon's ~45 m-wide pay channel;
- **effective grain mass m_eff ≈ 10.5 mg** (mean grain m̄ ≈ 3.9 mg, Gamma shape
  α ≈ 0.58), for a 7.5-inch shoe. This is inferred from assay statistics alone
  and falls in the 5–20 mg range implied independently by the published
  Dominion grain sizes and shapes (§1). Given the ~2× low bias seen on synthetic
  data, the true value is plausibly ~10–20 mg;
- a separate per-hole error is **not detectable** here (σ_h → 0.003, identical
  marginal likelihood). That is expected: line-spaced drilling cannot see it
  (§6.6). Twin holes on other creeks put it at σ_h ≈ 0.6–1.2 (§6.5).
  Forcing σ_h = 0.7 leaves the grain mass unchanged (m_eff 10.3 mg), lengthens
  the correlation lengths (ℓ_s 214 m, ℓ_n 49 m), and leaves held-out scores
  unchanged or slightly better: LOLO CRPS 1.107 vs 1.115, ρ 0.38, 90% coverage
  0.913, gold in top 20% 34.1%. The Dominion results do not depend on this
  unresolved component;
- structural zeros are not supported: on a profile over π₀, log Z is largest
  at π₀ = 0 (−2775.1; −2785.8 at 0.01; −2808.2 at 0.1) and m_eff stays at
  10.3–11.5 mg. The abrupt drop between π₀ = 0 and 0.01 reflects the poorer
  Laplace approximation once a zero assay's likelihood is flat in λ, so the
  comparison is approximate. At π₀ = 0.1 the fine-gold solution (m_eff 2.3 mg)
  is 9.4 log units below the coarse one.

**Held-out drill lines (10-fold leave-lines-out):**

| method | CRPS | Spearman | 90% coverage | gold in top 20% |
|---|---|---|---|---|
| kriging (log-grade) | 1.210 | 0.34 | 0.88 | 27.8% |
| Gaussian process | 1.189 | 0.32 | 0.86 | 32.1% |
| **grain-counting field** | **1.115** | **0.38** | **0.915** | **34.9%** |

Over 1 km held-out blocks: CRPS 1.19 vs 1.30, ρ 0.13 vs 0.06: still near chance.
Absolute skill is modest: ranking held-out ground by the model puts ~35% of its
gold in the top fifth (1.75× random).

**History check.** YCGC-era dredge cuts and thaw blocks cover 45% of the
drilled floor but contain 78% of the drill-indicated gold: the company mined
what its drilling found, and the mask and grades agree.

**Targets.** At a placeholder threshold of 20 c/yd³ ($20.67; ≈0.39 g/m³) and
P > 0.5, the model leaves 63 patches of ground outside YCGC-era mining; 39
(31.8 ha) lie where the valley frame is reliable (s < 6.8 km). Most are side
pay on the margins of the dredge cuts, mainly on the left limit, with
overburden 8–33 ft. **Ground mined after 1966 is not in the polygon layer**, so
every target must be checked against modern disturbance before it means
anything. Figures: `reports/dominion/strip_targets.png`, `map_targets.png`;
table: `reports/dominion/targets.csv`.

**Sensitivity to the per-hole error.** Refitting with σ_h = 0.7 (§6.5) and
redrawing the target map (`06_dominion_targets.py --sd-hole 0.7`):
- target probabilities on unmined ground have rank correlation 0.94 with the
  base model;
- 84% of the top-fifth unmined cells are shared;
- the target area where the frame is reliable overlaps with Jaccard index 0.73
  (33.3 vs 29.6 ha);
- the smoother field merges patches (42 instead of 63);
- the top-ranked target is the same.


### 6.3 Production check: upper Sulphur Creek, 9 annual dredge blocks (1950–58)
Ground truth: dredge No. 9 blocks on maps YCGC26682/25152, each with YCGC's
pre-dredging estimate and the value recovered, both at $20.67. 376 curated
holes from five campaigns; every estimator sees the drilling YCGC had.

**The historical baseline.** YCGC's own block estimates have essentially no
year-to-year skill (r = 0.06 against recovered grade) and, in total, the dredge
recovered 23% more value than estimated. Recovered grades span only 17–38
c/yd³, so with nine blocks no estimator can show ranking skill (the standard
error of a correlation is ~0.35).

**Result (log-RMSE against recovered grade; total bias):**

| estimator | primary run | sensitivity: line radius 40 m | sensitivity: 26683/26743 at $35 |
|---|---|---|---|
| YCGC recorded estimate | 0.37; −12% | 0.37; −12% | 0.37; −12% |
| mean of holes in block | 0.64; +74% | 0.64; +74% | 0.62; +70% |
| kriging, leave-block-out | 0.65; +75% | 1.09; −48% | 0.43; −10% |
| grain-counting field, leave-block-out | 0.62; +70% | 0.42; +29% | 0.56; +55% |

**Reading.** This is a negative result and it is reported as one:

1. **No model beats YCGC's own estimate at block scale.** The held-out-line
   advantage (§6.2) does not carry over to production forecasts here.
2. **Absolute block grades are not robust to defensible analysis choices.**
   Changing only how the valley axis is built moves kriging from −48% to +75%,
   and the disputed price basis moves it again. The grain-counting field is
   more stable but still spans +29% to +70%.
3. **The dominant uncertainty is in the conventions, not the interpolator.**
   YCGC's block estimates are about half the raw mean of the holes inside the
   blocks, so the company applied a weighting or deduction factor that the
   public data do not record. Dredge recovery is also unknown: a model of gold
   *in the ground* that is 30–70% above recovered gold is consistent with
   recoveries of 60–80%.
4. **The grain-size contrast agrees with the geology, with caveats.** The
   effective grain mass is ~80–95 mg on upper Sulphur against ~10 mg on
   mid-Dominion. Van Loon (2017) describes upper Sulphur gold as coarse and
   angular, up to several grams, and Froese et al. (2001) describe Dominion
   gold as fine, flat and well travelled. Separately, the three upper-Sulphur
   bench campaigns give 29–84 mg, stable under a per-hole error and structural
   zeros (thin sections, many zeros: high information, §6.7). The production
   blocks cannot confirm it (§6.4), and on most other campaigns the same
   estimate is not robust (§6.7).

**Consequence for use.** Model outputs should be used to *rank and locate*
ground relative to the drilling, with absolute grades reported only once a
local conversion (YCGC estimation factor × recovery) has been calibrated
against production. This is the calibration a modern test pit provides.

### 6.4 How wrong should a drill-based block estimate be?
`scripts/08_error_bound.py`, `09_meff_from_production.py`. For each mined
block, simulate from the fitted model the true block grade (mean of e^f over
the block) together with the mean of the drill assays at the *actual* hole
positions and section thicknesses. The spread of log(estimate / truth) is the
predicted error of a drill-based estimate. Only the spread is compared with
production: the mean offset (estimation factor, recovery) is not drilling
physics.

| | m_eff | holes per block | predicted sd | of which spatial only | observed |
|---|---|---|---|---|---|
| mid-Dominion (19 blocks) | 10 mg | 14 | 0.44 | 0.36 | ≥ 0.18–0.20 (van Loon 2016; lower bound) |
| upper Sulphur (9 blocks) | 80 mg | 7 | 0.90 (robust 0.64) | 0.29 (robust 0.27) | 0.36–0.45 (robust 0.29) |

- The *ordering* is right: coarse Sulphur gold sampled by thin bench sections
  should give block errors about twice Dominion's, and it does.
- The *size* is not: on Sulphur the grains-only model over-predicts the error
  about twofold. Scanning m_eff with everything else fixed, the observed robust
  spread (0.29) is best matched by m_eff ≲ 10 mg, but its 90% bootstrap band
  (0.08–0.80) is consistent with any m_eff from 0.3 to ~100 mg. **Nine blocks
  cannot measure grain size.** Taken together with §6.5, the 80 mg Sulphur
  figure is best read as an upper bound that also absorbs per-hole error.

### 6.5 Twin holes: the nugget is mostly not grain counting
`scripts/11_twin_holes.py`. On a few maps YCGC drilled holes 3–12 m apart:
78 pairs on the same map with distinct hole numbers (median 8.3 m apart; 38 on
Bear Creek 1912, 15 on Gold Run 1941, 9 on lower Hunker). Most come from whole
check lines drilled alongside earlier lines, or from lettered infill holes
(37, 37B, 37C …), not from re-drilling single suspicious holes, so the pairs are
not selected for disagreement. There is no systematic offset between the two
drilling series on a map (Hunker: median log ratio −0.09, 4 of 9 positive;
Gold Run: +0.56, 7 of 10 positive, sign test p = 0.34).

| | robust variance of log(G₁/G₂) |
|---|---|
| observed, 72 pairs with gold in both | **2.87** (median factor between twins ≈ 3) |
| observed, excluding holes through old workings | 2.59 |
| simulated: grains only, m_eff 10 mg (Dominion fit) | 0.23 |
| simulated: grains only, m_eff 90 mg (coarse) | 1.59 |
| simulated: grains 10 mg + per-hole error sd 0.8 | 1.69 |

Two observations rule out grain counting as the main source:
1. **Size.** Twins disagree ~13× more (in variance) than grain counting with a
   Dominion-like grain mass predicts, and nearly twice as much as even 90 mg gold.
2. **Shape.** Grain-count disagreement must fall as the gold in the sample
   rises (simulated robust variance 0.81 → 0.22 → 0.09 across terciles of gold
   per sample, pairs with gold in both holes). Observed: 0.58 → 2.45 → 5.14.
   The rank correlation between |log ratio| and gold per sample is +0.26
   (p = 0.03): the disagreement grows with the gold in the sample.

Fitting the full latent-field model on the four maps with ≥ 4 twin pairs,
adding the per-hole term (§3) raises the marginal likelihood on every one
(Δ log Z = +1.7, +2.2, +9.0, +15.7) with σ_h = 0.62–1.23. Without it, on
three of the four maps the model mimics the nugget with an implausibly short
cross-valley correlation length (ℓ_n = 5–14 m); with it, ℓ_n is 16–934 m. The
grain mass estimates move substantially (Gold Run 88 → 45 mg; lower Hunker
59 → 4.5 mg).

**Reading.** The hole-to-hole nugget of churn drilling is dominated by a
per-hole multiplicative error of about a factor 2–3 (σ_h ≈ 0.6–1.2): volume
recovery and its correction, and metre-scale traps at the bedrock surface that
one hole catches and its twin misses. Grain counting is a floor that matters
only where samples hold little gold (thin sections, low grade, coarse gold).
It is not the whole story. The twin maps are Bear Creek (1912), Gold Run (1941),
lower Hunker and a 1933 dredge-area prospecting map. Whether fine, travelled
gold (Dominion) has a smaller per-hole error cannot be tested directly
because Dominion has no twins.

**Uncertainty** (`scripts/17_twin_uncertainty.py`). Pairs cluster by map, so
the intervals resample whole maps (2,000 bootstrap samples, 90% intervals):

| pairs | robust var. | ratio to 10 mg grains | ratio to 90 mg | FE slope | within-map p |
|---|---|---|---|---|---|
| all 72 (11 maps) | 2.87 (0.97–4.34) | 12.7 (3.7–23.5) | 1.8 (0.6–2.9) | +0.26 (−0.04–0.60) | 0.045 |
| without Bear Creek, 37 (10) | 2.20 (0.62–4.54) | 11.4 (2.2–27.6) | 1.4 (0.4–3.2) | +0.17 (−0.13–0.88) | 0.12 |
| without old workings, 61 (11) | 2.59 (0.97–4.33) | 13.7 (4.0–24.7) | 1.8 (0.6–3.0) | +0.30 (−0.03–0.60) | 0.09 |

- FE slope: log|log ratio| regressed on log gold per sample, with map fixed
  effects. Grain counting alone gives −0.51 to −0.38 for m_eff 10–300 mg, and
  its 95th percentile is at most +0.04.
- **On size alone, coarse gold (90 mg) cannot be excluded. The trend excludes
  it.**
- Without Bear Creek, the excess over Dominion-like grain counting remains,
  but the trend is not significant on its own.

### 6.6 Identifiability: what each drilling design can tell apart
`scripts/13_identifiability.py`. Synthetic creeks (3 seeds, 355 holes on
YCGC-style lines), true m_eff = 10.5 mg, per-hole error σ_h = 0 or 0.8, with
and without twins 8 m from 15% of the holes:

| true σ_h | design | model | m_eff (mean ± sd) | σ̂_h | Δ log Z (per-hole − grains) | RMSE of latent field |
|---|---|---|---|---|---|---|
| 0 | lines | grains only | 8.2 ± 0.9 | – | 0.0 | 0.62 |
| 0 | lines + twins | grains only | 9.4 ± 0.6 | – | 0.1 | 0.59 |
| 0.8 | lines | grains only | 8.4 ± 3.3 | – | | 0.83 |
| 0.8 | lines | grains + per-hole | 8.1 ± 3.1 | 0.58 | **0.4** | **0.70** |
| 0.8 | lines + twins | grains only | 11.3 ± 1.9 | – | | 0.80 |
| 0.8 | lines + twins | grains + per-hole | **8.1 ± 0.1** | 0.68 | **8.2** | **0.68** |

1. **Line-spaced drilling cannot detect a per-hole error** (Δ log Z 0.4 for a
   real σ_h = 0.8), but leaving it out still costs 18% in field RMSE. Absence
   of evidence in the marginal likelihood is not evidence of absence. This is
   exactly the mid-Dominion situation (§6.2).
2. **Twins identify it** (Δ log Z 8.2) and make the grain mass estimate
   stable (sd 0.1 instead of 3), as on the real twin maps.
3. Without a per-hole term, the grain mass absorbs whatever nugget it can.
   It is erratic on lines alone (sd 3.3 mg) and biased up once twins expose
   the nugget (11.3 vs 8.1 mg).
4. All estimates sit ~20% below the true 10.5 mg, with or without twins. The
   MCMC check (§6.9) shows it is not caused by the Laplace approximation. The
   likely cause is the field prior: the synthetic channel is not a Gaussian
   process.

**Consequences.**
- The mid-Dominion model should carry a per-hole error even though the data
  there cannot see it. Forcing σ_h from 0 to 0.8 lowers log Z by only 4.4, and
  the grain mass is stable (m_eff 10.5 → 10.0 → 10.0 → 10.5 mg at σ_h = 0,
  0.3, 0.6, 0.8; 8.4 and 6.1 mg at 1.0 and 1.2). The per-hole error trades
  against the spatial field (ℓ_s 130 → 283 m), not against grain mass.
- **A cheap design recommendation for modern drilling:** twin 10–15% of
  holes. That is the only way to separate sampling error from geology, and the
  per-hole error is what limits block estimates.

### 6.7 Grain mass across 33 campaigns: precise, but not accurate
`scripts/10_regional_grain.py`, `12_grain_profiles.py`. Every campaign with
≥ 60 usable grades (33 maps, 62–393 holes) was fitted separately. The mean
grain mass m̄ was then profiled on a grid of 0.1–100 mg.

**How much a campaign can say.** For a hole with known concentration, the
Fisher information on grain size comes through two channels
(`sampling.grain_information`):

$$I_{\text{zero}}(\log\bar m) = \frac{\lambda^2 e^{-\lambda}}{1-e^{-\lambda}},\qquad
I_{\text{disp}}(\log m_{\text{eff}}) \approx \frac{1}{2}\left(\frac{x/(1+x)}{v_0+\log(1+x)}\right)^2,\quad x=\frac{m_{\text{eff}}}{cV}.$$

The zero channel peaks at λ ≈ 1.6 grains per sample and vanishes once samples
hold more than ~10 grains. The dispersion channel vanishes once grain variance
is small next to the non-grain variance v₀. So thick, rich sections say almost
nothing about grain size; thin, lean sections say a lot.

**What the campaigns show.**
- **The likelihood is confident.** The profiles drop by 17–285 log units
  across the grid, and 29 of 33 have a closed 95% interval. The depth of each
  profile tracks the zero-channel information bound (Spearman ρ = 0.78).
- **The answers do not agree.** Best-fitting m̄ ranges from 0.1 to 10 mg
  between campaigns *on the same creek* (Dominion: six campaigns, 0.1–10 mg).
  It tracks section thickness (ρ = −0.49, p = 0.004) and zero share
  (ρ = +0.54), not distance from King Solomon Dome (ρ = −0.09, p = 0.62).
- **On a single campaign the answer depends on the model.** On 26222
  (Dominion, 149 holes), allowing structural zeros and a per-hole error
  moves m_eff from 20 mg to 1.4 mg and *raises* log Z by 1.6. "Coarse gold"
  and "fine gold plus barren holes plus per-hole error" explain the same
  assays about equally well. On 26681 (Sulphur, 393 holes) a per-hole error
  alone moves m_eff from 40 mg to 3.5 mg (Δ log Z +17).
- **Mid-Dominion is the exception.** With 612 holes from one campaign, the
  "fine gold plus barren holes" alternative (m_eff ≈ 2.5 mg, 11% structural
  zeros) loses by Δ log Z ≈ 30–40. The ~10 mg estimate is stable under a
  forced per-hole error (§6.6).

**Reading.** Grain size can be read from assay statistics only with large
single-campaign data sets (hundreds of holes) and checks against the
alternatives that mimic coarse gold: barren ground, per-hole error and a
heavier grain tail. A regional grain-size map from the historic assays is not
supportable. The sharp profiles are an instance of a general hazard: a
confident likelihood is not an identified quantity when the model omits a
component that produces the same data.

### 6.8 A design rule: how many holes, and when grain counting matters
`scripts/15_design.py`. Ignore spatial variation inside a block. The mean of n
assays then has log error variance

$$\operatorname{Var}\big[\log(\bar G/c)\big] \approx \frac{\kappa^2}{n},\qquad
\kappa^2 = (1+h^2)(1+g^2)-1,\quad h^2 = e^{\sigma_h^2}-1,\quad g^2 = \frac{m_{\text{eff}}}{cV}.$$

Two consequences:
- **Crossover.** Grain counting dominates the per-hole variance only while a
  sample holds less gold than cV* = m_eff / h². That is ≈ 16 mg for
  Dominion-like gold (m_eff 10 mg, σ_h 0.7) and ≈ 80 mg for coarse gold
  (50 mg). The twin pairs hold a median ~140 mg, on the per-hole side of the
  crossover, which is why their disagreement does not fall with gold content
  (§6.5).
- **Holes needed** for a relative precision s: n ≈ κ²/s².

Monte Carlo with exact grain draws confirms the formula within ±30% for
n ≥ 16 (median ratio 1.00). Where samples hold fewer grains than one
effective grain mass, it is optimistic by up to 2×.

| ground (grade, section, m_eff) | gold per sample | grain share of κ² (σ_h 0.6–0.9) | holes for ±20% | holes for ±30% |
|---|---|---|---|---|
| mid-Dominion pay (20 c/yd³, 20 ft, 10 mg) | 68 mg | 0.25–0.10 | 17–40 | 8–18 |
| mid-Dominion lean (5 c/yd³, 20 ft, 10 mg) | 17 mg | 0.57–0.32 | 32–65 | 15–29 |
| Gold Run pay (34 c/yd³, 15 ft, 45 mg) | 87 mg | 0.54–0.29 | 30–61 | 14–27 |
| upper Sulphur bench, rich (20 c/yd³, 7 ft, 50 mg) | 24 mg | 0.83–0.63 | 86–149 | 39–67 |
| upper Sulphur bench (5 c/yd³, 7 ft, 50 mg) | 6 mg | 0.95–0.87 | 311–501 | 138–223 |

These counts are lower bounds: spatial variation inside the block adds to them.
**Lean bench ground with coarse gold cannot be valued to ±20% by drilling at
any practical density.** That is the regime where the BLM (1969) bulletin
sends the engineer to bulk sampling, now with a number attached. On
valley-bottom pay with fine gold, 15–40 holes per block suffice, and the
per-hole error, not grain counting, sets the count.

### 6.9 Is the Laplace approximation the problem? An MCMC check
`scripts/14_mcmc_check.py`. Kernel hyperparameters were fixed at their fitted
values. The latent field was sampled by elliptical slice sampling (Murray,
Adams & MacKay 2010), and (β₀, log m̄, log α) by random-walk Metropolis.
Each chain ran 20,000 iterations and kept the last 14,000.

| case | Laplace m_eff | MCMC median (90% interval) | truth |
|---|---|---|---|
| synthetic creek (seed 21, lines) | 9.2 mg | 9.4 (7.5–12.0) mg | 10.5 mg |
| mid-Dominion (612 holes) | 10.5 mg | 10.3 (8.9–11.8) mg | – |

- The Laplace estimate agrees with the exact conditional posterior to within
  3% on both, and the posterior interval covers the true value on the
  synthetic creek.
- The ~20% shortfall across synthetic seeds (§6.6) is therefore not a Laplace
  artefact.
- On mid-Dominion, the grain mass is 10 mg with a ±15% posterior interval,
  conditional on the spatial hyperparameters.
- Mixing is slow (effective sample size ≈ 18–19 for m_eff), so the intervals
  are approximate. The two halves of the Dominion chain agree (medians 10.4
  and 10.2 mg).
- A 6,000-iteration chain had not yet left the Laplace mode's neighbourhood
  (8.4 mg). Long chains are needed.
- A second Dominion chain was started far from the mode (field at its prior
  mean, grain mass ×3). It reached the first chain's region after ~15,000
  iterations. Over the last 3,500 iterations of both chains, R-hat for
  log m_eff is 1.04 and the pooled posterior is 10.0 mg (8.8–11.4), within 5%
  of the Laplace estimate.
- The low-count benchmark case (median five grains per sample) did not
  converge within 11,000 iterations and remains open.

## References

- Chapman, R.J., Bond, D.P.G., LeBarge, W. (2010). Particle-size distribution of gold within the Sulphur and Dominion creek drainages… *Yukon Exploration and Geology 2009*, 135–142.
- Clifton, H.E., Hunter, R.E., Phillips, R.L. (1969). Sample size and meaningful gold analysis. *USGS Professional Paper 625-C*.
- Froese, D.G., Enkin, R.J., Smith, D.G. (2001). Placer depositional settings and their ages along Dominion Creek, Klondike area, Yukon. *Yukon Exploration and Geology 2000*, 159–169.
- Murray, I., Adams, R.P., MacKay, D.J.C. (2010). Elliptical slice sampling. *Proceedings of AISTATS 2010*, JMLR W&CP 9, 541–548.
- Rasmussen, C.E., Williams, C.K.I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- US Bureau of Land Management (1969). *Placer Examination – Principles and Practice*, Technical Bulletin 4.
- van Loon, S. (2016, 2017). *Yukon Exploration and Geology 2015*, 253–280; *2016*, 225–242.
- Yukon Geological Survey. Historic placer data (map service and open data), Open Government Licence – Yukon.
