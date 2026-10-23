# Winner model: offense vs defense, fouls/free throws, referees (2026-10-23)

Code: `nba/analysis/winners_ext.py`; referees: `nba/ingest/espn_officials.py` (ESPN gameInfo.officials,
12,016 games 2016-26, 99.3% with crews). Walk-forward, test seasons 2020-25 (player projections start 2019).
Crew features use each ref's PRIOR games only (last 3 seasons, 20+ games); crew known in 95.9% of games.

| Method | Winners right | Brier | Log loss |
|---|---|---|---|
| Current model | 65.9% | 0.2121 | 0.611 |
| + player projection points | 65.9% | 0.2119 | 0.611 |
| + referees (crew home lean, crew FTA × team FTA gap) | 65.9% | 0.2121 | 0.611 |
| Referees placebo (crews shuffled) | 65.9% | 0.2121 | 0.612 |
| + fouls / free throws | 66.1% | 0.2123 | 0.612 |
| + everything | 65.9% | 0.2123 | 0.612 |
| Player projection placebo | 66.0% | 0.2130 | 0.614 |
| Offense vs defense, numpy sim + player projections | 64.8% | 0.2174 | 0.625 |
| Offense vs defense, numpy sim | 64.1% | 0.2219 | 0.635 |

- Offense-vs-defense simulation is clearly worse: points for/against carry the same information as margin
  but lose rest, injuries and the recent trend, and add noise from pace.
- Referees and fouls/free throws: no effect on winners (identical to placebo).
- Player projection points: +0.0002 Brier, better in 3 of 6 seasons → noise; not added.

## Referees and player stats (props)
Out-of-sample player residuals (actual − our projection) by crew whistle rate (crew's prior total-FTA deviation):

| Crew (quintile) | Crew FTA vs league | PTS residual | PRA residual |
|---|---|---|---|
| fewest FTA | −2.0 | −0.11 | −0.15 |
| 2 | −1.1 | −0.09 | −0.15 |
| 3 | −0.5 | −0.08 | −0.13 |
| 4 | +0.2 | 0.00 | −0.03 |
| most FTA | +1.3 | +0.10 | +0.06 |

Monotonic, ~30k player-games per group (≈5 standard errors top vs bottom): real but small — about ±0.1 point per player.
Untested as a model input yet (would need the crew at the lock; NBA posts assignments ~9 AM ET).
