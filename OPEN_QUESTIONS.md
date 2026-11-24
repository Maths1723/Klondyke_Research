# Open questions for placer miners and geologists

Questions the data cannot answer. Each one changes a modelling choice; the
affected file or decision is noted.

## Data and units
1. **Elevation datum.** What datum and benchmark did YCGC use on the 1936
   Dominion drill maps (1103–1105)? Elevations are in feet on a local datum;
   tying them to modern lidar needs one known point. → `qc.py`, bedrock surface.
2. **Price basis.** Were all YCGC drill values reported at $20.67/oz even after
   the 1934 price change to $35? Was the value for fine gold or for crude gold
   at a given fineness? → `units.py` conversion.
3. **What "dredge section" includes.** How much bedrock was counted in the
   dredge section, and was it consistent between campaigns (1936 vs 1943 vs 1955)?
   → accumulation target.
4. **Claim length.** Are the creek claims on these maps 500 ft or 250 ft along
   the baseline? Needed to place holes along the valley if coordinates are missing.

13. **YCGC drill geometry.** What shoe diameter did YCGC use on the 1936
    Dominion program (7.5-inch?), and did grades use the theoretical volume or a
    Radford-type factor? The inferred grain mass scales directly with it.
14. **Post-1966 mining.** Which of the ground outside the YCGC dredge limits on
    mid-Dominion (claims ~77–150 below Lower Discovery) has been mined since the
    1970s? The current target list assumes none, which is surely wrong in places.
15. **Grain size from operators.** What is the typical size of gold recovered on
    mid-Dominion (mesh distribution, largest pieces)? The model infers an
    effective grain mass of ~10–20 mg; a sieve analysis would test that directly.

16. **How YCGC turned drill holes into block estimates.** On upper Sulphur its
    block estimates are about half the raw mean of the holes inside each block.
    Was this area weighting, a drill factor, a recovery deduction, or all three?
    The archive documents for maps YCGC25152 / 26682 may say.
17. **Dredge recovery.** What share of in-situ gold did YCGC's dredges recover
    on Sulphur (fine vs coarse gold)? Needed to compare a model of gold in the
    ground with recorded production.
18. **26683 vs 26743.** Was the undated Sulphur map 26683 drawn at $20.67 or $35?
    The two maps share identical grades for 84 holes but only 26743 records a price.

## Drilling bias
5. How large is the churn-drill bias in **frozen vs thawed** ground? Should
   ground condition be a correction on the measurement rather than a predictor?
6. Is the ~8% under-estimate vs dredge recovery on mid-Dominion typical, and
   does it depend on gold coarseness (nugget effect in mid-Sulphur)?

## Geology
7. How much of the gold sits in the **bottom ~1 m** of gravel and the top of
   bedrock, versus spread through the section, on Dominion specifically?
   Decides whether grade × section thickness is the right target.
8. Why does the pay channel sit **upstream** of left-limit tributaries on
   Dominion (van Loon 2016) but **downstream** of them on Sulphur (van Loon 2017)?
9. Are buried **bench / terrace** gravels expected along these reaches, and at
   what height above the valley floor? A 2-D model of the valley floor ignores them.

## Economics and ground truth
10. What **thickness of overburden and gravel**, and what **grade over what
    thickness**, makes ground worth mining today? Sets the threshold in
    `04_target_map.py` (currently a placeholder: 50 c/yd3 over 2 m).
11. Which areas inside the dredge limits have been **re-mined since the 1980s**?
    Is there a public record beyond satellite imagery?
12. Would an operator share **test-pit or production results** on a reach in
    exchange for the model's predictions there? This is the validation that
    matters most.

## Drilling practice (new, from the twin-hole analysis)
19. On Bear Creek (1912), Gold Run (1941) and lower Hunker, YCGC drilled holes
    3–12 m apart. Were these **check lines** drilled as routine, or
    **re-drills** of holes whose first result looked wrong? If re-drills, the
    twin disagreement is inflated by selection. *Affects:* the size of the
    per-hole error (§6.5).
20. How did YCGC handle **core recovery** hole by hole? Was the volume
    correction applied per hole (as the BLM 1969 bulletin describes), capped,
    or replaced by a fixed factor? *Affects:* whether the per-hole error is
    mostly drilling or mostly geology.
21. In a modern sonic or RC programme on these creeks, how far apart would
    two holes need to be before an experienced driller expects **different
    pay**, because of bedrock crevices, boulders or pay at the rim? *Affects:*
    the twin spacing we recommend and how the per-hole error is interpreted.
