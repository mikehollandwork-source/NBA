# Game-winner model — backtest (2026-10-22)

Code: `nba/analysis/winners.py` (live: `live_predict`, board via `live_pipeline.winners_board`). Shown on the
board as **📈 PREDICTION (not a pick)**; graded straight up in `reports/live/winner_record.csv` (user's choice).

## Method
Projected home margin = home edge + team strength gap + last-10 trend gap + rest/back-to-back + missing rotation value.
- Strength = season-to-date margin blended with 60% of last season's margin (last season weighs as 20 games).
- Last-10 trend shrunk by n/(n+2) while a team has few games (tuned by CV on 2016-18).
- Missing value = prior 20-game Game Score average of rotation players (3+ of the team's last 10 games, 15+ min)
  who aren't playing; active after a team's 3rd game (same rule as the player model). Live: ESPN "Out" list at the lock.
- Linear fit, walk-forward (train on all earlier seasons). Win % = share of 20,000 numpy draws of
  margin ~ Normal(projection, 13.4) above 0.
- Why not sum the player sims: each player is simulated alone (no shared pace/OT/off nights) and opponent
  defense adds ~nothing at player level — it decides half of every game.

## Results (2019-25, 8,138 games, out-of-sample)
| Method | Winners right | Brier | Log loss |
|---|---|---|---|
| **Model** | **65.9%** | **0.2126** | **0.613** |
| Same, no missing-player terms (control) | 65.3% | 0.2170 | 0.623 |
| Team rating (season margin + home edge) | 64.9% | 0.2184 | 0.626 |
| Missing values shuffled (placebo) | 64.1% | 0.2223 | 0.635 |
| Elo (538-style) | 64.3% | 0.2224 | 0.636 |
| Always home team | 55.2% | 0.2478 | 0.689 |

- Best Brier in 7/7 seasons. Missing-player terms are real signal (placebo falls below the no-injury control).
- Calibrated: stated 70-75% → won 73.6%; 75-80% → 79.1%; 80%+ → 85.5%.
- Opening week is ~3-4 pts over-confident (team's 1st game: stated 61% → won 58%, n=111). Separate early-season
  weights were tried and didn't help.
- Lineups in the backtest = who actually played; live uses the injury report at the lock, so late scratches are missed.
- Known data quirk (training and live alike): game dates are UTC, so some late-game rest days are off by one.
- Public models pick ~66-69% of winners; the betting market already prices injuries. No edge vs the market is claimed.
