# Does winning or losing move player stats? (2026-10-22)

Residual = actual − our out-of-sample projection, 2019-25 (≈150k player-games). Code: `nba/analysis/win_effect.py`.

## Verdict
- **Same game (hindsight): yes, a lot.** Starters (30+ projected min) beat their PRA projection by +1.4 in wins and miss by −1.1 in losses.
  Blowouts cut starters' minutes ~3 either way; blowout losses cost starters −4.5 PRA, blowout wins ≈ 0 (fewer minutes, efficient ones).
  Close games (within 5): starters +1.9 min, +1.7 PRA. You can't know this before tip, so it isn't usable.
- **Before tip (expected result from the winner model): no usable effect.** Starters on big favorites/underdogs (8+) play ~0.3 min less,
  PRA moves ≤0.3. A correction fit on earlier seasons improved next-season squared error by 0.05% (PRA) / 0.13% (minutes) vs placebo
  0.02% / 0.01%, mixed by season. The player model already uses team point differential, so blowout risk is mostly priced in.
- **Previous game result (bounce-back / momentum): no usable effect.** Starters after a 15+ point game either way run +0.4-0.6 PRA
  (fewer minutes in the blowout dragged their recent averages). Correction gain 0.06% PRA vs placebo −0.02%; mixed seasons.
- Decision: no change to the model.

## Raw tables
```

== same_game_win
                           n   min   pts   reb   ast   pra
role            win                                       
bench <20       False  20171 -0.42 -0.44 -0.22 -0.12 -0.73
                True   20977 -0.16  0.04  0.05  0.01  0.16
rotation 20-30  False  32619 -0.26 -0.56 -0.24 -0.15 -0.94
                True   31023 -0.05  0.45  0.15  0.09  0.71
starter 30+ min False  22257  0.06 -0.68 -0.24 -0.21 -1.12
                True   23496  0.03  0.95  0.19  0.22  1.38

== same_game_margin
                               n   min   pts   reb   ast   pra
role            final                                         
bench <20       lost 15+    7175 -0.12 -0.23 -0.22 -0.06 -0.46
                lost 6-15   9315 -0.68 -0.58 -0.26 -0.15 -0.93
                within 5    8362 -0.19 -0.33 -0.08 -0.10 -0.45
                won 6-15    9019 -0.34 -0.27 -0.02 -0.07 -0.30
                won 15+     7277  0.01  0.57  0.19  0.16  0.97
rotation 20-30  lost 15+   10784 -1.39 -1.26 -0.56 -0.30 -2.11
                lost 6-15  15464  0.17 -0.27 -0.15 -0.10 -0.50
                within 5   14211  0.59  0.06  0.11 -0.01  0.18
                won 6-15   14130  0.32  0.45  0.19  0.06  0.71
                won 15+     9053 -1.15  0.71  0.08  0.22  1.03
starter 30+ min lost 15+    6827 -2.73 -2.93 -0.87 -0.72 -4.50
                lost 6-15  10803  1.00 -0.02 -0.06 -0.04 -0.09
                within 5   10708  1.94  1.24  0.30  0.12  1.69
                won 6-15   10789  0.85  1.30  0.36  0.29  1.96
                won 15+     6626 -3.03  0.01 -0.24  0.21  0.01

== pregame_expected
                                n   min   pts   reb   ast   pra
role            expect                                         
bench <20       big dog 8+   4442 -0.15 -0.07 -0.09 -0.01 -0.12
                dog 3-8      8131 -0.35 -0.24 -0.12 -0.08 -0.39
                toss-up     13970 -0.32 -0.25 -0.09 -0.07 -0.35
                fav 3-8      8825 -0.38 -0.25 -0.08 -0.06 -0.34
                big fav 8+   5780 -0.09 -0.01 -0.01  0.01  0.04
rotation 20-30  big dog 8+   9043 -0.35 -0.18 -0.14 -0.08 -0.37
                dog 3-8     13568 -0.09 -0.10 -0.04 -0.04 -0.16
                toss-up     20410 -0.06  0.00 -0.04 -0.02 -0.03
                fav 3-8     12680 -0.14 -0.04 -0.06 -0.04 -0.12
                big fav 8+   7941 -0.31 -0.10 -0.01 -0.01 -0.10
starter 30+ min big dog 8+   5009 -0.33 -0.21 -0.05 -0.05 -0.26
                dog 3-8      9217  0.04  0.13 -0.03 -0.03  0.08
                toss-up     15959  0.23  0.26 -0.02  0.04  0.29
                fav 3-8      9845  0.15  0.23  0.02  0.04  0.31
                big fav 8+   5723 -0.33  0.11 -0.07  0.01  0.09

== previous_game
                              n   min   pts   reb   ast   pra
role            prev                                         
bench <20       lost 15+   5916 -0.41 -0.21 -0.11 -0.04 -0.32
                lost <15  13917 -0.04 -0.07 -0.03 -0.02 -0.07
                won <15   14723 -0.24 -0.21 -0.08 -0.05 -0.29
                won 15+    5979 -0.78 -0.40 -0.15 -0.12 -0.63
rotation 20-30  lost 15+  11004 -0.05 -0.02 -0.04 -0.01 -0.05
                lost <15  21263 -0.20 -0.00 -0.07 -0.03 -0.07
                won <15   21435 -0.13 -0.10 -0.05 -0.05 -0.17
                won 15+    9004 -0.14 -0.13 -0.03 -0.05 -0.19
starter 30+ min lost 15+   6487  0.38  0.42  0.07  0.11  0.64
                lost <15  15412 -0.17  0.02 -0.06  0.00 -0.01
                won <15   16826 -0.03  0.10 -0.03 -0.02  0.07
                won 15+    6517  0.39  0.36  0.02  0.02  0.40

== test_pregame
      mean   min   max
stat                  
ast   0.01 -0.06  0.06
min   0.13 -0.07  0.26
pra   0.05 -0.07  0.16
pts   0.02 -0.14  0.17
reb  -0.05 -0.21  0.12

== seasons_pregame
stat     ast   min   pra   pts   reb
season                              
2020    0.01  0.19 -0.07 -0.07 -0.17
2021    0.06  0.26  0.16  0.07 -0.11
2022   -0.06  0.05  0.08  0.17  0.08
2023   -0.03 -0.07 -0.06 -0.14 -0.21
2024    0.03  0.11  0.09  0.05 -0.03
2025    0.04  0.25  0.08  0.06  0.12

== test_pregame_placebo
      mean   min   max
stat                  
ast  -0.00 -0.05  0.06
min   0.01 -0.20  0.28
pra   0.02 -0.10  0.18
pts   0.00 -0.17  0.14
reb  -0.07 -0.23  0.08

== seasons_pregame_placebo
stat     ast   min   pra   pts   reb
season                              
2020    0.06  0.12 -0.02 -0.02 -0.14
2021    0.03  0.28  0.18  0.10 -0.12
2022   -0.04 -0.05  0.07  0.14  0.07
2023   -0.05 -0.20 -0.10 -0.17 -0.23
2024   -0.00  0.01  0.02  0.00 -0.06
2025   -0.01 -0.08 -0.04 -0.05  0.08

== test_previous
      mean   min   max
stat                  
ast   0.02 -0.04  0.08
min   0.13 -0.07  0.43
pra   0.06 -0.07  0.21
pts   0.03 -0.17  0.20
reb  -0.06 -0.22  0.09

== seasons_previous
stat     ast   min   pra   pts   reb
season                              
2020    0.01  0.19  0.03  0.03 -0.22
2021    0.08  0.43  0.21  0.11 -0.04
2022    0.04  0.06  0.15  0.20  0.07
2023   -0.04 -0.07 -0.07 -0.17 -0.21
2024   -0.01  0.09  0.00 -0.02 -0.03
2025    0.01  0.10  0.02  0.01  0.09

== test_previous_placebo
      mean   min   max
stat                  
ast  -0.02 -0.09  0.06
min  -0.08 -0.27  0.26
pra  -0.02 -0.16  0.17
pts  -0.03 -0.22  0.12
reb  -0.09 -0.24  0.07

== seasons_previous_placebo
stat     ast   min   pra   pts   reb
season                              
2020   -0.05 -0.09 -0.06 -0.04 -0.24
2021    0.06  0.26  0.17  0.10 -0.10
2022   -0.07 -0.18  0.03  0.12  0.05
2023   -0.09 -0.27 -0.16 -0.22 -0.24
2024    0.01 -0.09 -0.05 -0.05 -0.07
2025   -0.00 -0.13 -0.08 -0.08  0.07

```
