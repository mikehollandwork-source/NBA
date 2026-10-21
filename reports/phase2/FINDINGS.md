# Phase 2 findings — 10 seasons (2016-17 to 2025-26)

No betting lines or outside projections are used anywhere. Outcomes are scored against official box scores.

## Data
- 252,000+ player-games, every official regular-season game (12,099) matched in all 10 seasons.
- Truth = ESPN box scores. 326 games where ESPN's box is blank (2016-18) were filled from Basketball-Reference; spot checks matched ESPN exactly.
- 2020-21 onward cross-checked against nba.com play-by-play: ≥99.0% exact agreement on every stat, every season (`data_audit_espn_vs_pbp_by_season.csv`).
- nba.com's public matchup data leaves `switches_on` / `help_*` empty, so coverage is proxied (see "matchup_coverage" below).

## Test A — predictability holds across all 10 seasons
Ranking is near-identical every season (`testA_stability_summary.csv`):
MIN (nMAE 0.20) > FGA > PRA (0.32) > PR > PA > RA > PTS (0.40) > 3PA > REB (0.43) > DREB > AST (0.51) > TOV > 3PM (0.69) > FTA > STOCKS > OREB > STL (0.91) > BLK (1.08).
Season-to-date average is within ~1% of the full-season (future-knowing) average every season.

## Test B — what improves projections (walk-forward: train on earlier seasons, test 2019-20 … 2025-26)
Full model vs season-to-date average, % squared-error reduction (significant in 7/7 seasons for every stat):

| MIN | PRA | PTS | AST | REB | TOV | STL | 3PM | BLK |
|---|---|---|---|---|---|---|---|---|
| 25.4 | 10.9 | 7.7 | 6.4 | 5.9 | 5.0 | 3.0 | 2.9 | 2.3 |

Correction to Phase 1: with 10 seasons of training, 3PM/STL/BLK *are* beatable — but only by 2–3%.

Inputs that hold up across seasons (drop-one loss significant in ≥ half the seasons and positive in ≥6/7):
- **Teammates out** — #1 for every stat (MIN 6.5%, PRA 3.5%, PTS 2.5%, AST 1.8%, REB 1.4%, TOV 1.2%).
- **Minutes/role** — MIN, PTS, REB, AST, PRA.
- **Recent form** — MIN (4.8%) and AST only.
- **Teammates returning** — MIN, PRA.
- **Last season** — REB, AST, TOV.
- **Opponent defense** — TOV only.
- Not holding up anywhere: pace, game script, opponent absences, rest/home, matchup/coverage proxies.

## Projection system (`projection_accuracy_summary.csv`)
Stats projected: MIN, PTS, REB, AST, TOV and PRA/PR/PA/RA (the predictable ones).
Ensemble (avg of minutes×rate, direct, and season-avg+deviation models) is best for every stat, both scenarios, all 7 seasons.

| % error reduction vs season avg | MIN | PRA | PR | PA | PTS | RA | AST | REB | TOV |
|---|---|---|---|---|---|---|---|---|---|
| Injury report known | 25.4 | 11.6 | 9.9 | 10.3 | 8.2 | 8.8 | 6.9 | 6.6 | 5.5 |
| + starting lineup known | 30.9 | 13.0 | 11.0 | 11.4 | 9.1 | 10.2 | 7.7 | 7.7 | 5.8 |
| Worst season (injury report) | 21.5 | 9.0 | 7.8 | 7.7 | 6.1 | 6.4 | 4.9 | 4.5 | 4.7 |

Distributions: negative binomial with variance ∝ μ^p fit out-of-sample (p ≈ 0.7–1.05; constant-dispersion NB was overconfident and was replaced).
50%/80% intervals cover 48–50% / 77–80%. P(stat > season avg) is calibrated within ~1–2 pts in every bucket and beats the season-average distribution on Brier/log-loss for every stat.

## When the projection disagrees with the season average
(`disagreement_by_size.csv`, `disagreement_by_factor.csv`) — out-of-sample, 7 seasons.

How often reality moved off the season average in the direction the model said:

| Size of disagreement | share of games | MIN | PTS | PRA | REB | AST |
|---|---|---|---|---|---|---|
| < 0.1 SD | ~28% | 52% | 52% | 52% | 52% | 51% |
| 0.1–0.25 SD | ~32% | 58% | 56% | 56% | 56% | 55% |
| 0.25–0.5 SD | ~25% | 64% | 61% | 62% | 61% | 59% |
| 0.5–1 SD | ~12% | 76% | 68% | 71% | 67% | 64% |
| > 1 SD | ~3% | 86% | 76% | 80% | 75% | 71% |

Realized move ≈ 0.9–0.97× the predicted move — size is about right, slightly too big.

Which factors to trust (`disagreement_trust_weights.csv`; 1.0 = moves exactly as projected, 0 = noise, controlled for all other factors):
- **Trust (≈1.0):** minutes/role, teammates out, teammates back, return from absence, recent form, opponent defense, season timing (≥1: late-season effects are under-weighted).
- **Discount (0.5–0.75):** last season (PTS/PRA/MIN), rest/home (except MIN 1.65), usage profile (REB 0.41).
- **Ignore (≈0 or negative):** matchup/coverage proxies, pace, opponent absences, game script (PTS/REB).
When the "last season" factor is the main thing pulling a projection away from the current average, it is right only ~49% of the time.

Re-weighting each factor by its historical trust (fit on other seasons) improves every stat slightly (+0.2–0.4 pts), including every stat's worst season (`disagreement_trust_weighted.csv`).

## Caveats
- Teammates-out / lineup features use who actually played; live you rely on injury reports and announced lineups, so live accuracy will be a bit lower, especially for late scratches.
- "Season average as the line" is a stand-in. Real edge needs comparison to closing lines (agreed: evaluation only).
- Players filtered to ≥20 games & ≥15 mpg in a season (uses full-season info for the filter, not for predictions).

---

# Similar-defense signal (backtest 2019-20 to 2025-26)

**Question:** when a player faces defense D, does how he did (vs our projection) against defenses *similar to D* predict tonight?

**Defense fingerprint** (prior games only, blended with last season early on; `nba/analysis/defense_profiles.py`), four equally weighted groups:
- Size: minute-weighted height (all / guards / forwards / bigs), weight — ESPN bios.
- Speed (proxies — real tracking speed isn't free/reachable): age, steals & turnovers forced, block rate, fast-break share of shots allowed (2020-21+), defensive rebounding.
- Coverage: shot locations allowed (rim / paint / mid / corner 3 / above-break 3), shot types allowed (pull-up / drive / catch-and-shoot / cut / post / putback), switching spread, bigs-on-guards, foul rate, assisted share.
- What they allow: pts/100, rim FG%, mid FG%, 3P%, eFG%, share of points to G / F / C.
End-of-2025-26 cards with best / worst / most-similar teams: `defense_cards_2025.csv` (e.g. OKC ≈ DET, NYK; SAS ≈ BOS; GSW ≈ PHX, CLE, MIA).

**Signal:** the player's 8 most similar earlier matchups (this + last season, similarity ≥ 0.3), similarity-weighted average of (actual − our out-of-sample projection), shrunk toward 0.
**Control:** the same player's average miss vs *all* earlier opponents ("general bias").
**Placebo:** the player's 8 *least* similar earlier matchups.

| | PTS | REB | AST | PRA | MIN |
|---|---|---|---|---|---|
| Correlation with tonight's miss — similar defenses | 0.024 | 0.020 | 0.025 | 0.026 | 0.043 |
| — placebo (least similar) | 0.020 | 0.013 | 0.010 | 0.020 | 0.035 |
| Error reduction — similar defenses alone | 0.05% | 0.04% | 0.07% | 0.06% | 0.22% |
| — general player bias alone | 0.19% | 0.11% | 0.11% | 0.20% | 0.47% |
| — both | 0.20% | 0.11% | 0.13% | 0.21% | 0.51% |

Vs the season-average stand-in line (strongest 30% of signals): projection side wins 59.8% when the history agrees vs 56.5% when it disagrees (PTS). But using general player bias instead gives a bigger spread (60.5 vs 56.2), and the similar-defense part *minus* general bias gives 58.0 vs 57.4 — noise.
Variations (4 or 15 neighbours, stricter/looser similarity, excluding the same opponent) all give the same picture (`similar_defense_robustness.csv`).

**Verdict:** the similar-defense signal has no usable edge beyond a player's general tendency to beat or miss projections. Small samples (8 games) are dominated by outliers (e.g. one 83-point game drove a +7 signal; the next game vs a similar defense he scored 14).

**Were projections accurate in those games?** Yes — out-of-sample projections exist for every game in the window: MAE PTS 4.85 vs season average 5.03, bias ≈ 0. Even in the 10% of games where the similar-defense history was most strongly "over", projections were low by only 0.34 pts on average (PRA 0.33); strongest "under" → projections high by 0.24 (PRA 0.46). Examples with the projection and actual for each similar game: `similar_defense_examples_pts_2025.json`.

**What did work (side finding):** a per-player bias correction (shrunk average miss vs our projection) is small but consistent: +0.1–0.5% error reduction and a ~4–5 pt hit-rate spread vs the season-average line. Worth adding to the model.

---

# Season simulation (NumPy Monte Carlo, 2020-21 to 2025-26, 1,861 player-seasons)

Setup (`nba/analysis/season_sim.py`): every player who was a rotation player last season (≥20 g, ≥15 mpg) and played this season, over his team's real schedule (follows trades; teams meet 2×/3×/4× per season ≈ 51% / 16% / 32% of opponents). 1,000 simulated seasons each.
- Game outcome: negative binomial, variance curve fit on other seasons; PTS/REB/AST correlated within a game (Gaussian copula, scaled so PRA spread matches reality: PRA 80% range covers 80.4%).
- Season-long shift: players run ±5–7% vs game-day projections for a whole season (±24–36% vs a preseason forecast); simulated as one random multiplier per season.
- Games played: actual, or simulated — availability drawn from what happened to players with similar last-season availability (other seasons), plus streaky absences (out today → out next game 77%). Simulated avg 56.6 games vs 56.4 actual; 80% range covers 89%.

Versions compared: **base** (our projection), **simdef** (+ history vs similar defenses), **h2h** (average vs that exact team), **last_szn** (last season's average), plus **bias** (player's general miss) and a **placebo** (least similar defenses) as controls.

## Per matchup (game-day forecasts, 104,872 games)
| Version | PTS avg miss | PRA avg miss | PTS CRPS | PTS 80% range coverage |
|---|---|---|---|---|
| Our projection | **4.80** | **6.12** | **3.37** | 82% |
| + similar defense | 4.91 | 6.26 | 3.47 | 81% |
| Last season avg | 5.46 | 7.23 | 3.84 | 76% |
| Head-to-head avg | 5.99 | 7.85 | 4.31 | 71% |
Examples (SGA, Jokic, Brunson 2025-26, each version + 10/50/90% range + actual): `season_sim_matchup_examples_2025.csv`.

## Season totals — true preseason forecast (everything frozen before opening night)
Avg error in season total (actual games played / simulated games played):

| Version | PTS | PRA | PTS with simulated games |
|---|---|---|---|
| Our preseason projection | 147 | 215 | 233 |
| + similar defense | 147 | 215 | 230 |
| Placebo (least similar) | 148 | 215 | 229 |
| Player bias | 148 | 215 | 230 |
| Head-to-head (last season) | 138 | 198 | 234 |
| Last season avg | 139 | 199 | 236 |

- Similar defense = placebo → no defense-similarity effect at season level either.
- Head-to-head ≈ last-season average (a player's averages vs all 29 teams add back up to his season average) — no head-to-head effect.
- Both beat our preseason projection *only because* our "preseason projection" is the first-game projection, which under-projects stars (~1.3 pts early in the season) and carries one-night context. A dedicated preseason model is needed.
- Games played dominates season-total error: simulating availability raises error ~60% (147 → 233 PTS) for every version.

Invalid comparison avoided: summing game-day projections over a season reuses in-season results (a "last 8 games miss" adjustment scores 16 vs 45 purely mechanically), so season totals are judged only with preseason-frozen forecasts.
