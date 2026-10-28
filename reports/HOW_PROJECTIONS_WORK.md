# How the projections are made — complete walkthrough

Everything below is what the live board actually runs (`scripts/live.py` → `nba/live_pipeline.py`). Numbers are the current fitted values.

---

## 0. What is never used
- No betting lines, odds or other sites' projections as inputs — anywhere. ESPN's game data includes odds; the downloader deletes them before saving.
- Lines appear only in the evaluation code (`lines_backtest.py`) and in the record at an *assumed* -120 price.

---

## 1. Data that goes in

| Source | What | Seasons |
|---|---|---|
| ESPN | Official box scores (the "truth"), schedule, rosters, injury report, height/weight/birthdate | 2016-17 → today |
| nba.com (via public mirror) | Play-by-play (exact minutes, cross-check), shot locations, defender matchups | 2016-17 → 2025-26 |
| Basketball-Reference | 326 games ESPN left blank | 2016-18 |

- Every official game is matched in every season; ESPN and nba.com agree on ≥99% of player stat lines (2020-25).
- Current season (2026-27): ESPN only (the nba.com mirror hasn't published it yet), so the defender-matchup and shot-location inputs are blank this season.

---

## 2. Who gets a projection tonight
1. Every player on tonight's rosters (ESPN).
2. Minus anyone ESPN's injury report lists as **Out**. Day-to-day players stay in and are flagged "(DTD)".
3. Minus anyone failing the **volatility rule** (§8) — they're listed as "No projection".
4. Only players projected for **10+ minutes** appear on the board.

---

## 3. The inputs (features) — all built from games *before* tonight only
Season-to-date values reset each season. "Usage" = FGA + 0.44×FTA + turnovers.

| Group | Exact inputs |
|---|---|
| Season so far | stat average this season, games played, per-36-minute rate this season |
| Last season | minutes/game, games played, share of games started, per-36 rate of the stat |
| Recent form | last-5 average (needs 3+ games), last-10 average (needs 5+), last-10 per-36 rate |
| Minutes & role | minutes: season avg, last 3/5/10, game-to-game SD; started last game; start rate last 5 and season |
| Usage | usage season & last 10; usage share of team; FGA, 3PA, FTA averages |
| Return from absence | team games missed right before tonight (max 15); first game back flag |
| **Teammates out** | Rotation = teammates who played 3+ of the team's previous 10 games averaging 15+ min (traded players excluded). Missing tonight → their recent minutes, usage, assists, rebounds; how many were starters; minutes missing at his position; missing usage as share of team usage; his usage share × that; count missing. *Zero for a team's first 3 games.* |
| Teammates back | minutes & usage of rotation players returning tonight after missing the last game |
| Opponent defense | what the opponent has allowed of this stat per game ÷ league average to date (overall and to his position G/F/C); opponent net rating |
| Opponent absences | opponent's missing rotation minutes and usage |
| Matchup / coverage | defender-quality score, switching tendency, bigs-on-guards share, shot locations allowed (blank this season; tested ≈ no value) |
| Pace | expected pace (avg of both teams' possessions/game), each team's pace |
| Game script | team's point differential/game, difference vs opponent, absolute difference |
| Rest / home | days of rest (max 10), back-to-back, opponent back-to-back, home/away |
| Season timing | team's game number |
| **Preseason prior** | the preseason projection for this stat and for minutes (§4) |

Minutes model uses the same groups minus stat-specific ones.

---

## 4. The preseason prior (feeds every game-day model)
For each player before the season:
- **Inputs:** last 3 seasons (per-game, per-36, totals, games, availability, starts, absence episodes, longest absence, whether he finished last season out, injury-tagged absences), age, height, weight, position, whether he changed teams, and roster context (usage/minutes his team lost, usage of teammates on the current roster).
- **Model:** LightGBM (500 trees, learning rate 0.03, 15 leaves), trained on all past seasons, players with a previous season and 10+ games, weighted by √games.
- **Final number = 50% model + 50% last season's per-game average** (beat last season alone and the standard "Marcel" method on every stat in 2018-25 backtests).
- Rookies have no preseason projection.

---

## 5. The game-day models
- **Algorithm:** LightGBM gradient-boosted trees — 300 trees, learning rate 0.06, 31 leaves, ≥100 games per leaf, 80% row/feature sampling, L2 penalty 5, seed 7.
- **Training data:** 217,005 player-games, 2016-17 → 2025-26 — players who were rotation players this season (20+ games, 15+ mpg) or last season. Trained once, saved, reused nightly.
- **Minutes:** one model predicts tonight's minutes.
- **Three projections per stat (PTS, REB, AST, TOV, PRA, PR, PA, RA):**
  1. *Minutes × rate:* predicted minutes × predicted per-minute rate (rate model trained on games with 3+ minutes, weighted by minutes). Combos = sum of their parts.
  2. *Direct:* model predicts the stat itself.
  3. *Season average + adjustment:* model predicts tonight's difference from his season-to-date average.
- **Final projection = average of the three** (beat each one alone in all 7 test seasons).
- 3PM, STL, BLK use the direct model only (least predictable stats).

---

## 6. Minutes check (team must total 240)
The models were trained on players who played, so they over-give minutes to deep bench players. Fix:
1. Players with no NBA history get **12.6 min** (historical average for such players in a team's first 5 games, DNPs counted as 0).
2. Each team's **top 8 keep their model minutes** (max 40); players 9+ are scaled down to fill the remainder to 240. *(Rule, not backtested.)*
3. Every counting stat is scaled by the same ratio as the player's minutes.

---

## 7. The 80% ranges (numpy simulation, `nba/sim.py`)
- **20,000 simulated games per player.** Each one:
  1. draws tonight's **minutes** from the real spread of actual ÷ projected minutes for players with similar projections (captures early exits, foul trouble, blowouts);
  2. scales the PTS/REB/AST projections to those minutes;
  3. draws PTS, REB, AST from negative binomials (variance = k × mean^p, fit on 150k out-of-sample games) **scaled by the player's own past miss size** (last 250 games, blended with the league curve at the weight of 100 games);
  4. links the three with a Gaussian copula (correlation fit after minutes are drawn, tuned so PRA's spread matches reality).
- Range = 10th to 90th percentile. Backtest 2023-25: 80% ranges contain the result ~81% of the time for steady and swingy players alike (before the player-specific spread: 85% vs 78%).
- The same draws give the Kalshi rung probabilities. Details: `reports/phase2/sim_improvements.md`.

---

## 8. Volatility rule ("baseline")
- Swing = last season's PRA standard deviation ÷ PRA average (20+ games).
- **Swing > 0.45 → no projection.** No qualifying last season → no projection.
- Backtest: typical miss falls from 36% to 29% of the projection; ~53% of players kept.

---

## 9. The Lean (which stat the model points to)
For PTS, REB, AST, PRA, PR, PA, RA:
1. **Baseline** = season-to-date average once he has 5+ games, otherwise last season's average.
2. **Reference line** = the half-point line where a distribution centred on that baseline is closest to 50/50 (fixes the "average lines favour the under" skew).
3. **Probability** of over/under that line from our projection's distribution, then recalibrated per stat (logistic, fit on other seasons).
4. **Pick** the stat with the largest distance from 50%. Each stat's swing is already in its probability (more swing → wider distribution → closer to 50%). Picking the least-swing stat instead was worse (top 5/night 75.6% vs 77.3%, 2019-25; `reports/phase2/ranking_backtest.md`).
5. **Second calibration** on picked leans only (picking the max of 7 inflates confidence), per stat.
- Backtest 2019-25 (149k leans, each season tested on calibrations fit to other seasons): stated 55-60% → 57.5% hit; 60-65% → 62.4%; 65-70% → 67.9%; 70%+ → 77.6%.
- **Picks must be bettable at DraftKings:** a 70%+ lean only appears as a ✅ PICK if DraftKings (via ESPN's public data) offers the same line or a better one (higher for unders, lower for overs). The board shows the DraftKings line and odds to bet; the pick is graded at exactly that line and price, frozen at the 1-hour lock. Leans with no DraftKings line yet show as ⏳ waiting; leans where DraftKings' line is worse show as ⏸. The record was reset on Oct 24; Oct 22-23 were backfilled at DraftKings' closing line. The old own-average record is kept as a reference only. DraftKings lines never feed the model.
- **Board = 70%+ leans only**, ranked by model probability (2025: 75.8% hit, ~10/night). Ranking by model + the player's past hit rates at that line was worse in 7/7 seasons (`reports/phase2/ranking_backtest.md`).
- ⚠️ = star (last season 22+ PTS) under in his first game: these hit 66.7% vs 77.8% stated (n=42).
- Full projections for every player stay in the CSV.
- **The reference is the player's own baseline, not a sportsbook line.**

---

## 10. Injuries and news
- ESPN injury report + roster injury tags, re-pulled every hourly run. Out → removed; DTD → flagged.
- Teammates-out and returning-teammates inputs (after a team's 3rd game).
- Not captured: news after the last run before tip, unannounced minutes limits, in-game injuries/ejections/blowouts.

---

## 11. Locking, grading, record
- Runs every 30 minutes. Each run rewrites a game's projections until **1 hour before tip**; the first run inside that hour writes the **final lock** (`reports/live/locked/<date>/<game>.csv`, never changed again). A run that starts within 80 minutes of a lock waits for it inside the job, because GitHub's schedule can be late or skip runs. If no run lands in that hour, the last earlier version stands.
- After the game is final, each board lean is graded from ESPN's box score: **win +0.833u, loss −1u (-120), push 0, did not play = void.**
- Two records, kept separately: **strongest (70%+, the board)** and **all leans** — each with last night, shown on the board and in Telegram. Break-even at -120 = 54.5%.

---

## 12. Accuracy so far (out-of-sample, 2019-25)
| | MIN | PTS | REB | AST | PRA |
|---|---|---|---|---|---|
| Error reduction vs season average | 25% | 8% | 7% | 7% | 12% |
| Typical miss (game-day, all games) | 4.6 | 4.7 | 1.9 | 1.4 | 6.1 |

Known weaknesses: stars run ~1.9 pts low in their first game; minutes rule untested; no real book lines yet to measure market edge.

---

## 13. Kalshi paper tracking (not picks)
- Every run before tip pulls Kalshi's player ladders (PTS, REB, AST, PRA — e.g. "Jokic: 25+ points") from Kalshi's free public API.
- For every rung: our probability P(stat ≥ k) from the same 20,000-game simulation as the ranges (§7), vs Kalshi's price (mid of bid/ask).
- "Paper picks" = best rung per player-stat where buying at the ask plus Kalshi's fee (0.07 × price × (1 − price)) has ≥5% expected return. **Shown for tracking only, not as bets** — on Oct 21 this rule went 53-187 (−19% ROI) at real prices.
- Graded after each game at the real price paid; the record also reports a Brier score for model vs market on every rung (lower = better). If the model never beats Kalshi on that score, there is no edge.
- **🔁 Fade tracking:** each paper leg is also graded on the opposite side (if the model bought YES, the fade buys NO) at the real price + fee frozen at the lock. Tests whether betting *against* the model's disagreements pays. Oct 22-26: paper −15.8% ROI, fade −3.3% — both lose; the bid/ask spread and fee cover the gap. Paper only, not picks.
- Prices are never used as model inputs.

## 14. Game-winner predictions (📈 tracked, not picks)
- Projected point margin from: home edge (~2.6 pts), each team's strength (this season's margin blended with last season's), last-10 trend, rest and back-to-backs, and the **value of rotation players who are out** (from the injury report at the lock).
- numpy draws 20,000 margins around that projection (spread 13.4 pts); win % = share where the team wins.
- Backtest 2019-25: 65.9% of winners right, best of every baseline (team rating 64.9%, Elo 64.3%, home team 55.2%), well calibrated (70-75% → 73.6% won). Opening week runs ~3-4 pts over-confident.
- Each prediction also shows **player-sum team totals** (sum of every player's projected points) for reference only: as a winner method it picked 64.8% in 2020-25 backtests vs 65.9% for the margin model, and it ran ~11 pts high per team in opening week.
- Locked with the same 1-hour rule; graded straight up (no price). Details: `reports/phase2/winners_backtest.md`.

## 15. Where it lives in the code
`nba/analysis/features.py` (inputs) · `groups.py` (input groups) · `drivers.py` (model settings) · `preseason.py` · `projection.py` · `lean.py` · `nba/live_pipeline.py` (tonight) · `nba/record.py` (grading) · `nba/kalshi.py` (paper tracking) · `nba/analysis/winners.py` (game winners) · `scripts/telegram_notify.py`

## 15. 🔥 Scoring watch (Telegram, not picks)
- Top 5 players on the slate by a mix of **ceiling** (90th-percentile points from the 20,000 sims — a 1-in-10 good night) and **upside** (ceiling minus the player's own average). Players projected under 20 minutes or listed Out/Doubtful are skipped.
- Early-season usage trends were tested (2019-25, 3,813 games) and predicted nothing beyond our projection, so they are not used.

## 16. Daily channel posts (Discord)
- **Morning:** the pick record (DraftKings line + odds) and 🔥 **On fire**: of the 15 players who beat their own average by the most last game (20+ projected minutes), the 5 our data leans on hardest today. Backtest 2019-25: these leans hit 79.3% vs 76.7% stated, better than stated in every season. Unders right after a hot game hit only 64% vs 70% stated, so they carry ⚠️.
- **At each game's lock (1 hour before tip):** that game's ✅ picks, or just the teams and start time if there are none.
- **After games:** results for games that had picks, then the updated record.
- **Disagreement cap:** a 70%+ lean is not a pick if our probability is more than 20 points above DraftKings' implied probability for that price. When we disagree that much, the line has usually known something (Oct 22-27: 41% hit, n=54).
