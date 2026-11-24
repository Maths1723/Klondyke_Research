# YCGC historic placer data: audit

Source: YGS map service export, ycgc_layer0_drillholes.json etc.

## Contents

- records: 13,358 on 340 source maps (12,447 drill holes, 670 shafts, 231 bedrock test holes)
- with a grade: 8,630; with bedrock elevation: 2,720
- usable for grade modelling (grade + section, unambiguous, de-duplicated, drill holes): **6,179**
- dredge/thaw/cut polygons: 5,352; with estimated AND actual value: 10

## Problems found

- **Units:** thickness and elevation fields are in feet; median (OB+DS)/(surface-bedrock) = 1.079 (dredge section includes some bedrock). 209 holes are inconsistent by >35%.
- **Gold price basis:** recorded for 14% of holes; values [1.0, 7.6, 15.6, 20.67, 35.0, 35.3, 38.5]. Assumed $20.67 for 86% (maps with no price anywhere).
- **Unrecorded price bases detected** from co-located redrawn holes (grade ratio 1.690 / 1.708 / 1.863): YCGC26345 = $38.5 (61 pairs, confirms recorded); YCGC26344 = $35.0 (38 pairs, confirms recorded); YCGC26343 = $38.5 (16 pairs, NEW (unrecorded)); YCGC25097 = $35.0 (14 pairs, NEW (unrecorded)).
- **Ambiguous grades** (digitiser notes doubt the number is cents/yd3): 835.
- **Duplicates across maps** (same grade, <15 m apart): 667 records removed; 653 clusters.
- **Trace assays:** 397 (notes or 0.1 c/yd3).
- **Already mined:** 8568 holes fall inside a dredge cut, thaw block or other recorded working.

## Largest source maps

```
                                                                                                       title    year confidence    n  n_usable  n_bedrock  price  median_thickness_ratio  median_grade_2067
map_id                                                                                                                                                                                                     
YCGC26681                         Sulphur Creek drill holes claims 61-116A below Discovery. File 2. Oct 1935  1935.0     Medium  394       393         11  20.67                1.042553            12.7000
YCGC26018                     Plan of Creek Claims from No. 61 Below to 105 Below Discovery on Bonanza Creek     NaN       Good  371         0          0  20.67                     NaN            32.9500
YCGC26349      Plan of Placer mining claims from No. 39 Below Disc. to No. 65 Below Disc. on Hunker Creek YT     NaN       Good  346       136          0  20.67                     NaN            18.0600
YCGC26528                             Plan of Klondike Valley portion of Hydraulic Mining Lease No. 18, 1917  1917.0     Medium  329         1          0  20.67                     NaN             5.0000
YCGC26201                                                   Examination Dominion  - Gold Run, April 10, 1941  1941.0       Good  300       261        184  20.67                1.060606             7.4000
YCGC26009                          Plan of Bear Creek claims 3 Above to 20 Below Discovery ... Plan No. 6759  1912.0       Good  299       133          0  20.67                     NaN            11.0000
YCGC25041                                                       Dredge Canadian no.3 area - Prospecting 1933  1933.0       Good  260       226          0  20.67                     NaN             6.6500
YCGC26461                                                     Indian Creek Drill Holes, Claims 273 to 4 B.C.     NaN     Medium  243       241        241  20.67                1.083333             8.4500
YCGC25159         Arlington project - General map showing examination results and proposed method of working  1934.0       Good  242       156          0  20.67                     NaN             7.0000
YCGC26199     Middle Dominion drill holes claims 77 to 99 below Lower Discovery, Sept 7, 1936 - Plan No 1103  1936.0       Good  212       209        202  20.67                1.088235            20.2000
YCGC26525                General Map, Dredge Canadian No. 2 Area, Right limit Klondike Valley, November 1934  1934.0     Medium  196       106          0  20.67                     NaN             1.9000
YCGC26743   Position of dredge 9 fall 1958 to end of reserves Drill holes and limits upper Sulphur 7A to A23  1958.0     Medium  186       158        139  35.00                1.055556             2.8938
YCGC26340                                 Blueprint: Gold Run Examination Claims 35-66, 1942 - Plan no. 1153  1942.0     Medium  183       174        182  20.67                1.066667             4.1000
YCGC26223                                                 Middle Dominion Original and Revised Dredge Limits  1935.0       Good  181       178          0  20.67                     NaN            10.1000
YCGC26593                                                                           Quartz Creek drill lines  1937.0       Good  177        17          0  20.67                     NaN             8.2000
YCGC26341                                            Gold Run Examination  Claims 8-35, 1941 - Plan no. 1154  1941.0       Good  170       168          0  20.67                     NaN            33.0000
YCGC26056                                                                            Dredge No. 3 Area, 1950  1950.0       Good  167         0          0  20.67                     NaN                NaN
YCGC26324                                                         Eureka Creek drill holes on Claims 1 to 15     NaN       Good  165       116        157  20.67                1.057983             8.2000
YCGC26438        Plan of Placer Mining Claims from No 63 Below Discovery to Anderson Concession Hunker Creek     NaN       Good  163       136          0  20.67                     NaN            19.8000
YCGC26225a                                                                                               NaN     NaN        NaN  160        47          0  20.67                     NaN            19.2000
YCGC25126                                            Henderson Creek examination - Drill line 73 to A4 incl.  1941.0       Good  151       151        149  20.67                1.100000            10.3000
YCGC26358                                Dredge No. 4 Progress Map, Arlington Area, Season 1937 - Plan no. 2  1937.0       Good  150         0          0  20.67                     NaN                NaN
YCGC26683                                   Sulphur Creek drill holes, Claims Discovery to 22 Above, no date     NaN       Good  149        83        141  20.67                1.050000             4.6000
YCGC26222                                                 Middle Dominion Original and Revised Dredge Limits  1935.0       Good  149       149          0  20.67                     NaN            12.6000
YCGC26299                                                                            Dredge 6 Workings, 1915  1915.0       Good  126        29          0  20.67                     NaN            20.3000
YCGC26374               Plan of placer mining claims from No. 41 to No.55 below Discovery on Hunker Creek YT     NaN     Medium  125        20          0  20.67                     NaN            35.6000
YCGC26457                       Eureka Examination Drill Holes on Indian Ck at mouth of Eureka Ck. Dec. 1940  1940.0     Medium  120       116        110  20.67                1.117647             6.1000
YCGC26482                                                                     Klondike Left Limit Area, 1934  1934.0     Medium  117        86          0  20.67                     NaN             4.0000
YCGC26263         Plan Showing Drilling Results and Area Dredge on Upper Dominion Creek, Y.T. Dec. 31, 1924.  1924.0       Good  108        86          0  20.67                     NaN            11.5000
YCGC26463                                              Indian Creek drill holes, mouth of Quartz Creek, 1941  1941.0       Good  103       103          0  20.67                     NaN             4.0000
```