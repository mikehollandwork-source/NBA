# Ranking backtest — which leans are "strongest"? (2026-10-22)

Question: should leans be ranked by model probability alone, or by a blend with the player's
past hit rate at that same line (last season, last 10 games) plus penalties (swing > 0.40,
new team, star first-game under)? Code: `nba/analysis/rank_backtest.py`.

Setup: out-of-sample leans 2019-25 (calibrations fit on other seasons), players passing the
live filters (swing ≤ 0.45, ≥10 projected min). Each night, take the top k leans by each ranking.
Grade vs the player's own baseline line at -120. History uses prior games only.

| Rank by (top 5/night) | 2019-24 hit | 2025 hit | 2025 ROI |
|---|---|---|---|
| Model probability | 76.6% | **78.1%** | +43% |
| Logistic fit of all signals | 76.9% | 78.1% | +43% |
| Hand blend (model + last season + last 10, penalties) | 73.8% | 74.1% | +36% |
| History only | 71.6% | 72.0% | +32% |
| Random (control) | 64.8% | 63.6% | +17% |

- Model beat the hand blend in 7/7 seasons; the fitted blend only tied the model → no gain from history.
- Model ≥65% with history disagreeing still hit 70.0% (vs 72.2% when history agrees).
- 2025 by stated probability: 65-70% → 68.8%, 70%+ → 75.8% (~10 leans/night).
- Star (last-season ≥22 PTS) first-game unders at 70%+: 66.7% hit vs 77.8% stated (n=42) → flagged ⚠️ on the board.
- Kalshi agreement could not be tested (no price history).
- Random leans hit 64%: the reference is a soft line (player's own baseline). These are not edges vs sportsbooks.

Decision: board shows leans with model probability ≥ 70% (`STRONG_MIN`), ranked by model probability.
