# Data sources

All Yukon government data is used under the **Open Government Licence – Yukon**
(commercial reuse allowed with attribution). Put downloads in `raw/`; zips are
extracted automatically by `scripts/01_inspect_raw.py`. Nothing in `raw/` is
modified.

| # | Dataset | Use | Where | Status |
|---|---|---|---|---|
| 1 | YCGC Historic Placer data, GDB | drill holes, grades, dredge limits | https://ygsftp.gov.yk.ca/ApexUploads/Compilations/43/YCGC_Historic_Data.gdb.zip | needed |
| 2 | YCGC Historic Placer data, SHP | same, cross-check | https://ygsftp.gov.yk.ca/ApexUploads/Compilations/43/YCGC_Historic_Data.shp.zip | needed |
| 3 | van Loon (2016) Dominion Creek pilot, YEG 2015 | Appendix 1 attribute table (cross-check), methods | https://emrlibrary.gov.yk.ca/ygs/yeg/2015/16_vanLoon.pdf | needed |
| 4 | van Loon (2017) Sulphur & lower Dominion, YEG 2016 | transfer test, methods | https://emrlibrary.gov.yk.ca/ygs/yeg/2016/13_vanLoon.pdf | later |
| 5 | Lower Hunker Creek & Klondike River, YEG 2018 | third creek | https://ygsftp.gov.yk.ca/publications/yeg/yeg18/papers/7_vanLoon.pdf | later |
| 6 | Placer streams 1:250k | stream lines, tributary mouths | https://open.yukon.ca/data/placer-streams-250k | needed |
| 7 | Placer claims (GeoYukon / mineral tenure) | overlay targets on current tenure | https://yukon.ca/en/science-and-natural-resources/mining/find-information-mineral-tenure | later |
| 8 | HRDEM / ArcticDEM | elevation datum tie, valley morphology | https://open.canada.ca/data/en/dataset/957782bf-847c-4644-a757-e383c0057995 | later |
| 9 | YMEP22-048 Sulphur Creek report (2023) | modern sonic/RC drilling and test pit: independent validation | https://emrlibrary.gov.yk.ca/emrlibrary/ygs/YMIP_YEIP/2022/2022-048.pdf | later |

Attribution line for any output: *Contains information licensed under the Open
Government Licence – Yukon.*

## Known properties of the YCGC data (from the papers)

- Grade: cents per cubic yard at a fixed **$20.67/oz** (van Loon 2016 says CDN, 2017 says USD).
- Trace assays recorded as **0.1 c/yd3** by the digitiser.
- Thicknesses in the papers' tables are already converted to metres.
- **Elevations in van Loon (2016) Appendix 1 are in feet on a local datum**, not
  metres a.s.l. as the paper implies: surface − bedrock ≈ (overburden + dredge
  section) / 0.3048 for every hole checked. `qc.infer_elevation_unit` tests this
  on every dataset.
- Drilling under-estimated dredge recovery by ~8% on average (6 blocks under by
  ~23%, 4 over by ~14%) on mid-Dominion.
- Hole 134-8 was originally 2132.7 c/yd3; the paper replaced it with a
  neighbour average, and Appendix 1 prints the replaced value (237). If the GDB
  holds the original, we keep it and flag it rather than smoothing it away;
  either way the choice is recorded in DECISIONS.md.
