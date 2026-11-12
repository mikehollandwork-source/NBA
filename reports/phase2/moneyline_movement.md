# Moneyline: winner model vs line movement (evaluation only)

Data: ESPN core API open/close moneylines (DraftKings/Caesars/ESPN BET preferred), available 2023-25 only
(2019-22 have no open/close). 3,686 games matched to the out-of-sample winner model. Betting splits
(ticket %/handle %) have no free historical source, so they were not tested.

## 70%+ winner predictions (1,380 games)
| Line move (no-vig, our team) | n | won | ROI at close | ROI at open |
|---|---|---|---|---|
| moved against us (2+ pts) | 250 | 78.0% | +7.1% | -1.1% |
| flat | 585 | 82.7% | -0.1% | -0.1% |
| moved toward us (2+ pts) | 545 | 82.0% | +1.1% | +10.8% |
All 70%+: won 81.6%, ROI +1.7% at close, +4.0% at open. Movement direction does not change who wins;
ROI gap toward vs against -5.9 pts vs placebo SD 4.5 (noise).

## Model vs closing market (any confidence, bet the side the model rates higher)
| model minus no-vig close | bets | ROI | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|
| >= 3 pts | 2,520 | +0.2% | -0.2% | +2.4% | -1.5% |
| >= 5 pts | 1,906 | +1.1% | +1.3% | +1.7% | +0.2% |
| >= 8 pts | 1,231 | +5.2% | +8.7% | +3.3% | +3.7% |
| >= 10 pts | 860 | +6.5% | -2.1% | +8.4% | +13.3% |

Caveat: the backtest's missing-player input uses who actually sat (late scratches are known to the model
but may not be at the 1-hour lock); only 3 seasons. Treat as a paper-track candidate, not a pick rule.

## Our win % above vs below the market (user 2026-11-12), every team-side, 2023-25 (3,686 games)
gap = our win % minus the market's no-vig closing win %; bet that team at the closing (and opening) price.

| Our win % vs market | n | ours | market | won | ROI close | ROI open | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|
| we're 10+ lower | 860 | 48% | 65% | 62% | -10.1% | -5.9% | -2.8% | -12.2% | -15.4% |
| 5-10 lower | 1,046 | 52% | 59% | 58% | -7.4% | -3.9% | -10.4% | -6.4% | -5.4% |
| 2-5 lower | 991 | 51% | 54% | 54% | -7.7% | -4.5% | -11.9% | -7.1% | -3.9% |
| about equal | 1,578 | 50% | 50% | 50% | -6.6% | -7.0% | -5.8% | -8.3% | -5.7% |
| 2-5 higher | 991 | 49% | 46% | 46% | -5.2% | -5.5% | -2.3% | -1.7% | -12.2% |
| 5-10 higher | 1,046 | 48% | 41% | 42% | -3.4% | -2.8% | +4.2% | -3.5% | -10.9% |
| 10+ higher | 860 | 52% | 35% | 38% | **+6.5%** | +1.1% | -2.1% | +8.4% | +13.3% |

- ROI rises steadily from "we're lower" to "we're higher": the model has some information the market misses,
  but only the 10+ higher bucket clears the vig. Placebo (gap shuffled within season): 8+ higher ROI +5.2% vs
  placebo -4.9% (SD 2.9).
- Within 10+ higher: market favorites 154 bets, +7.9% close / +10.8% open; market underdogs 706, +6.2% / -1.0%.
- When we're LOWER than the market the market is right: those teams win close to the market's rate, and betting
  them loses 7-10% (every season).
- Caveat: the backtest's missing-player input uses who actually sat; live edges will be smaller. 3 seasons only.

## Against the spread (user 2026-11-12), 2023-25, 3,683 games, closing spread + its price (-110 if missing)
Our projected margin (winner model) vs the closing spread; bet the side we favour by `edge` points.

| Our margin vs spread | n | cover | ROI | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|
| < 1 pt | 932 | 48.9% | -6.7% | -3.5% | -9.6% | -6.9% |
| 1-2 | 847 | 51.8% | -1.2% | +3.5% | -4.5% | -2.4% |
| 2-3 | 642 | 52.3% | -0.2% | +3.3% | +2.5% | -6.3% |
| 3-4 | 491 | 50.9% | -2.9% | -4.6% | -0.4% | -4.3% |
| 4-6 | 469 | 50.1% | -4.7% | -4.5% | -5.6% | -4.2% |
| **6+** | **302** | **62.3%** | **+19.0%** | +7.0% | +22.0% | +24.0% |
Laying points with 3+ edge: 287 bets, 58% cover, +10.6%; getting points: 975, 52%, -1.0%.

### Look-ahead check (model WITHOUT the missing-player input, so no knowledge of who sat)
- ATS 6+: 464 bets, 57.5% cover, **+9.7%** (2023 +5.6%, 2024 +9.9%, 2025 +12.5%); 1-6 pt edges all lose.
- Moneyline 10+ higher: 1,144 bets, **+2.7%** (+5.3%, +1.3%, +1.3%).
The live model knows the injury report at the lock (not late scratches), so expect live results between the two
versions: ATS 6+ roughly +10-19%, moneyline 10+ roughly +3-6%. Paper-track before treating either as a pick.
