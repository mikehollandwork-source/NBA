# Confluence / safety-filter backtest (70%+ leans, 2019-25, out-of-sample)

gap = hit rate minus stated probability; diff = gap(flag on) - gap(flag off); placebo_sd = spread of diff when the flag is shuffled within season (|diff| should be well above it).

```
                                     flag  share_on  n_on  hit_on  gap_on  gap_off   diff  seasons_same_sign  placebo_sd
                         3/3 models agree     0.984 43322   0.777   0.002   -0.066  0.068                  5       0.015
                                  starter     0.429 18890   0.801   0.029   -0.020  0.049                  7       0.003
          last-5 average agrees with lean     0.858 37764   0.781   0.006   -0.030  0.036                  7       0.005
role stable (proj min within 3 of last-5)     0.586 25823   0.771   0.010   -0.012  0.022                  7       0.004
           models tightly agree (top 25%)     0.246 10834   0.782   0.013   -0.003  0.017                  7       0.005
                     team on back-to-back     0.253 11154   0.783   0.003   -0.000  0.004                  3       0.004
                                home game     0.506 22287   0.777   0.002   -0.001  0.003                  4       0.003
                  teammates out (n_out>0)     0.788 34706   0.779   0.001    0.001 -0.001                  4       0.005
       blowout risk (|proj margin| >= 10)     0.176  7765   0.780   0.000    0.001 -0.001                  3       0.004
     steady minutes (min SD below median)     0.462 20332   0.763  -0.001    0.002 -0.003                  3       0.004
```

## Combined confluence score (70%+ leans)
Score = count of: 3/3 models agree, starter, last-5 average agrees with the lean, role stable (proj min within 3 of last-5).

| score | n | stated | hit | gap |
|---|---|---|---|---|
| 0 | 553 | 79.3% | 73.6% | -5.7 |
| 1 | 2,987 | 78.0% | 74.2% | -3.9 |
| 2 | 10,154 | 79.3% | 76.5% | -2.8 |
| 3 | 18,860 | 77.1% | 77.5% | +0.4 |
| 4 | 11,481 | 76.3% | 79.7% | +3.4 |

Score 4 beats stated in 7/7 seasons (2019 +0.2, 2020-25 +3.4 to +5.5); ~10 such leans per night in 2025.
Flags that did nothing (|diff| within ~1 placebo SD or inconsistent): teammates out, blowout risk, steady minutes, home, team back-to-back.

## Correction: same-game starter flag is look-ahead
The `starter` flag above is whether he started THAT game (not known at the 1-hour lock). With pregame start rate (last 5 >= 0.6):
score 0: -6.4, 1: -3.1, 2: -0.5, 3: +0.4, 4: +1.8 (gap = hit - stated). Score <= 2 under stated in 7/7 seasons. Live rule uses this version.
