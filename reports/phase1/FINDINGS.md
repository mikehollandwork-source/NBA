# Phase 1 findings — 2025-26 regular season

Sample: 349 players (≥20 games, ≥15 mpg), 21,064 player-games. Regular season only.
Ground truth = ESPN official box scores, cross-checked against nba.com play-by-play
(≥99.8% exact agreement per stat; 103 rows ESPN dropped were filled from PBP).
No betting lines or third-party projections are used anywhere.

## Test A — which stats are consistent / predictable
Pregame prediction = player's season-to-date average (prior games only), scored from game 11 on.
`nMAE` = mean absolute error ÷ stat average (lower = more predictable).

| Rank | Stat | Avg | Median player CV | Pregame nMAE | Ceiling nMAE* | Var explained by minutes |
|---|---|---|---|---|---|---|
| 1 | MIN | 25.1 | 0.24 | 0.20 | 0.19 | – |
| 2 | FGA | 9.5 | 0.42 | 0.32 | 0.31 | 39% |
| 3 | PRA | 19.7 | 0.44 | 0.33 | 0.32 | 43% |
| 4 | PR | 16.9 | 0.47 | 0.35 | 0.34 | 39% |
| 5 | PA | 15.2 | 0.50 | 0.36 | 0.35 | 36% |
| 6 | RA | 7.4 | 0.49 | 0.37 | 0.36 | 29% |
| 7 | PTS | 12.4 | 0.55 | 0.40 | 0.39 | 30% |
| 8 | 3PA | 3.9 | 0.58 | 0.42 | 0.40 | 21% |
| 9 | REB | 4.5 | 0.58 | 0.43 | 0.42 | 22% |
| 11 | AST | 2.9 | 0.73 | 0.51 | 0.50 | 14% |
| 12 | TOV | 1.5 | 0.93 | 0.66 | 0.65 | 6% |
| 13 | 3PM | 1.4 | 0.94 | 0.68 | 0.67 | 10% |
| 16 | OREB | 1.2 | 1.15 | 0.77 | 0.75 | 7% |
| 17 | STL | 0.9 | 1.15 | 0.89 | 0.87 | 6% |
| 18 | BLK | 0.5 | 1.70 | 1.09 | 1.07 | 3% |

*Ceiling = using the player's full-season average (knowing the future). Season-to-date is
within 1–3% of it, so **better averaging cannot create edge** — remaining error is game-level.

Other Test A facts:
- Stat *levels* are very stable between players (split-half reliability 0.87–0.99). Who is good is easy; what happens tonight is the hard part.
- Streakiness is near zero for everything except minutes (lag-1 autocorr: MIN 0.33, FGA 0.17, PTS 0.09, STL/BLK ~0.02). Role changes persist; "hot hands" mostly don't.
- Distributions: PTS / combos are over-dispersed (variance ≈ 3.2× mean) → need negative-binomial-type sims, not Poisson. STL/BLK/TOV/3PM/OREB are ≈ Poisson.
- Per-36 rates are steadier than raw totals for PTS/PRA/FGA → project minutes × rate.

## Test B — which pregame inputs actually help
LightGBM, train Oct 21–Jan 31, test Feb 1–Apr 12 (8,387 out-of-sample player-games).
Gains are % reduction in squared error vs. the season-to-date average, with 95% bootstrap CI.

| Stat | Model gain vs season avg | Share of remaining error caused by not knowing minutes |
|---|---|---|
| MIN | **31.7%** [29.0, 34.6] | – |
| PRA | **15.2%** [13.5, 16.9] | 40% |
| PTS | **10.5%** [9.0, 11.9] | 27% |
| REB | 6.1% [4.7, 7.7] | 22% |
| AST | 6.0% [4.3, 7.8] | 13% |
| TOV | 5.2% [3.7, 6.8] | 4% |
| 3PM | 0.9% [-0.2, 2.1] — not significant | 12% |
| STL | 0.3% [-0.9, 1.4] — not significant | 5% |
| BLK | -1.7% [-3.1, -0.3] — model worse | 3% |

Unique value of each input group (drop-one loss, significant only):
- **Teammates out** — the biggest unique driver: MIN 9.9%, PRA 4.7%, PTS 3.7%, TOV 2.2%, REB 1.7%, AST 1.7%.
- **Minutes/role** (recent minutes, starter, volatility) — 0.9–2.4% across MIN/PTS/REB/AST/PRA.
- **Recent form (L5/L10)** — matters for MIN and AST, not PTS once role is known.
- **Opponent defense** — small: AST 0.6%, BLK 1.1%, TOV 1.1%, MIN 0.5%. Not significant for PTS/REB/PRA.
- **Pace** — not significant for any stat.
- **Rest / home** — MIN 1.5%, PRA 0.5%.
- **Game script** (team strength gap, our blowout proxy) — REB 0.5% only.

## What this means for the build
1. Project **minutes first**, as its own model. It's the most predictable stat and accounts for 27–40% of PTS/PRA error.
2. **Injury/rotation data is the edge source.** Teammates-out is the single most valuable input. A good pregame inactive/injury feed matters more than any opponent stat.
3. Project stats as **minutes × per-minute rate**, with usage redistribution when teammates sit.
4. **PRA, PTS, REB, AST** are where modeling adds signal. **3PM, STL, BLK** are close to pure noise beyond a season average. Treat them as distributions around the average, not as targets with an edge.
5. Opponent/pace adjustments are small (≤1%). Worth including, but they will not create edge on their own.

## Caveats
- One season, one train/test split. Effects under ~1% need confirming on more seasons (Phase 2: 2016-17 onward).
- "Teammates out" uses who actually played. Pregame this comes from injury reports; late scratches will make live results slightly worse than this backtest.
- Opponent-defense features are basic (team allowed by position group). Matchup-level data (primary defender) is downloaded but not modeled yet.
