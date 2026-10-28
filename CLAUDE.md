# NBA player projection system — instructions for Claude

## Working with this user
- Verdict first, then reasoning. No filler. If an idea is weak, say so and give a probability; give one concrete alternative.
- **Ask before assuming.** When a request is ambiguous (thresholds, data sources, scope), ask with options and a recommendation before building.
- Rank options by likelihood of success. State confidence when uncertain.
- The user reads on a phone: short tables, outcome first.
- Every result must be out-of-sample (walk-forward by season, prior games only). Always run a control / placebo before calling something an edge.

## Hard rules
- **No betting lines or third-party projections as model inputs — ever.** Lines are allowed only in `nba/analysis/lines_backtest.py` for evaluation. ESPN summaries contain odds; `nba/ingest/espn.py` strips them (`KEEP_KEYS`).
- User's pricing assumption when no odds exist: **-120 both sides** (break-even 54.55%).
- No real historical prop lines are in the repo yet (no free source found). Proxy-line results (season/L10 averages) are **not** edge estimates — see FINDINGS §4.

## Environment facts
- stats.nba.com is blocked from cloud IPs → nba.com data comes from the public mirror `github.com/shufinskiy/nba_data` (sparse git clone in `nba/ingest/bulk.py`; raw downloads 403).
- Basketball-Reference rejects Python `requests` (Cloudflare) → `nba/ingest/bbref.py` uses `curl`, 1 request / 3.5 s.
- ESPN team codes differ from nba.com: GS/GSW, NY/NYK, SA/SAS, NO/NOP, UTAH/UTA, WSH/WAS. Always map to nba.com codes (`live_pipeline.TEAM`).
- Season label = starting year (2025 = 2025-26). `LIVE_SEASON = 2026` in `nba/config.py`.
- `data/` is gitignored. A new session has **no data and no trained models** — rebuild first.

## Rebuild from scratch (new session)
```bash
pip install -r requirements.txt          # pinned versions; SessionStart hook also does this
python scripts/rebuild.py                # ~2.5-3 h fresh; skips finished steps; --force to redo
python scripts/live.py YYYY-MM-DD        # first run trains + caches models in data/models/ (~10 min)
python scripts/score_live.py YYYY-MM-DD  # after games are final
```
Rebuild order matters (see `scripts/rebuild.py` docstring): ESPN boxes → reconcile/BBRef backfill → bios → features → preseason model → features again (adds preseason prior) → out-of-sample projections (used for range calibration) → live models.
Full research reruns: `python scripts/run_phase2.py [features|testA|testB|project|disagree|similar|season|preseason]` (hours; not needed for daily use).

## Automation (GitHub Actions → Telegram)
- `.github/workflows/bootstrap.yml` (manual): rebuilds data + trains models (~3-4 h), saves Actions cache `nba-data-v1-*`. Re-run after model/data-code changes, or when the season changes (cache paths include `matchups_2025` — bump to the previous season each year).
- `.github/workflows/live.yml`: every 30 min 16:05–03:35 UTC (noon–11:30 pm ET in EDT). Runs `scripts/live.py` (date = today ET), `scripts/telegram_notify.py`, commits + pushes `reports/live` to `main`. GitHub cron is unreliable (2026-10-22: 4 hourly runs skipped, next 42 min late), so after each run `scripts/wait_for_lock.py` sleeps in-job until the next lock time if it is ≤80 min away and runs again (up to 3 locks per job, timeout 180 min).
- **Channel posts (user's choice 2026-10-28: clean and simple):** morning (first run of the day) = record line (DK picks + last night) + 🔥 On fire: of the 15 players (≥20 proj min) furthest above their own average last game, the 5 with the strongest lean (backtest 2019-25: stated 76.7% → hit 79.3%, 7/7 seasons; unders after a hot game 64% vs 70% stated → ⚠️). Then nothing until each game's final lock: one post per game with ✅ picks, or just teams + start time if none. After games: per-game pick results (games with picks only) + record line. Kalshi paper (no Kalshi in any message), winner predictions, Scoring watch (`scoring_watch()`, kept in code) and mid-day change updates are still tracked/graded but NOT posted. Usage trends tested and add nothing. **Delivery = Discord webhook (user's choice 2026-10-28, Telegram kept chat-not-found):** secret `DISCORD_WEBHOOK_URL`; when set, `scripts/telegram_notify.py` posts there (HTML → markdown, split under 2,000 chars, CSV attached) and Telegram is skipped. Telegram fallback secrets: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`. With neither the notifier dry-runs (prints) and marks nothing as sent.
- `BBREF_OPTIONAL=1` in bootstrap: if Basketball-Reference blocks the runner, the 326 old games are dropped instead of failing (slightly different training data — note it if it happens).

## Data sources
| Source | Use |
|---|---|
| ESPN site API | Box scores (truth), schedule, rosters, injury report, player bios |
| shufinskiy/nba_data | nba.com play-by-play (CDN 2020+), shot charts 2016+, defender matchups 2017+ |
| Basketball-Reference | 326 games ESPN left blank (2016-18) |
| ESPN core API propBets (provider 100 = DraftKings) | DraftKings player-prop lines + odds, DISPLAY ONLY next to picks (`nba/ingest/espn_props.py`; first price of each pair = over, verified on 1,132 lines). Posted game-day morning; ESPN keeps only recent games |
Reconciliation: ESPN vs nba.com play-by-play agree ≥99% on every stat 2020-25; all official games matched every season.

## Model (what the live sheet uses)
- Game-day ensemble = mean of (minutes model × per-minute rate), direct model, and season-avg + deviation model. LightGBM, `drivers.PARAMS`, `random_state=7`. Features in `nba/analysis/groups.py`.
- Preseason projection = 50/50 blend of preseason LightGBM and last season's per-game average (`preseason.final_projection`); fed to game-day models as `pre_*` prior.
- Ranges + Kalshi probabilities: `nba/sim.py`, 20,000 sims, one shared draw. Minutes drawn first (empirical actual/projected ratio), then negative binomial PTS/REB/AST with variance k·μ^p × player spread multiplier (last 250 games, shrunk with weight 100), Gaussian copula fit after minutes. Backtest: `reports/phase2/sim_improvements.md` (player-specific stat links tested, dropped). Leans don't use the sims.
- Live minutes: team total 240; players with no NBA history get 12.6 min (historical early-season average); top-8 keep model minutes, bench squeezed (rule, not backtested).
- **Volatility baseline:** no projection if last season's PRA swing (SD / avg, ≥20 games) > `VOL_MAX = 0.45` (`nba/config.py`), or no qualifying last season.
- Teammates-out only activates after a team's 3rd game of the season.
- **Locking (user rule, 2026-10-22): final lock 1 hour before tip** (`LOCK_LEAD_MIN = 60`). Runs before tip−60 rewrite `reports/live/locked/<date>/<away>@<home>.csv` (pre-lock); the first run at/after tip−60 writes the final file (`lock_final=True`), never touched again. If no run lands in that hour, the last pre-lock file stands at tip. Lock time is judged at run start (when injuries were pulled). Scoring (`score_live.py`) uses locked files only. `scripts/live.py` auto-commits `reports/live` — push after each run so locks survive new sessions. 2026-10-21 locks were backfilled (last version generated before each tip).
- **Record @ -120:** `scripts/grade.py` → `nba/record.py` grades every locked lean once final (win +0.833u, loss −1u, push 0, DNP void) into `reports/live/record.csv` + `record_summary.md`, reported separately for **strongest (≥70%, the board)** and **all leans** (user's choice); shown on the sheet and Telegram (daily header + ✅/❌ results per game). Reference = player's own baseline, not book lines — always say so.
- **Kalshi = PAPER tracking only (user's choice, 2026-10-22).** `nba/kalshi.py` pulls Kalshi ladders (PTS/REB/AST/PRA; free public API), scores every rung: model P(stat ≥ k) vs market mid, and flags would-be picks (best rung per player-stat, exp. return ≥5% at ask + 0.07·p·(1−p) fee). Locked per game in `<game>.kalshi.csv`; graded in `reports/live/kalshi_record.csv`. Board/Telegram/results label every model lean "✅ PICK" and every Kalshi line "🚫 NOT A PICK" (user's choice, 2026-10-22: keep Kalshi on the board, make the difference obvious). Never present these as picks: on 2026-10-21 the rule went 53-187, −19% ROI at real prices. Key metric = Brier model vs market on all rungs. **Fade tracking (user's choice, 2026-10-27):** every paper leg is also graded on the OPPOSITE side at its real lock price + fee (`record.add_fade`, `fade_*` columns in `kalshi_record.csv`, 🔁 line in report/Telegram; paper only, not picks). Oct 22-26: paper 178-538 −15.8%, fade 538-178 −3.3% — both lose; spread + fee eat it, market is better calibrated. Kalshi has a maintenance window (503s, exchange inactive) early Thursday mornings ET — the pipeline falls back to projections only. Polymarket DOES list nightly player props (PTS/REB/AST/3PM O/U inside each game event, zero fee). A market-check filter (publish picks only if Kalshi/Polymarket/DraftKings offered the line at a price we beat) was built and then removed on 2026-10-23 at the user's request (they can't bet unders on prediction markets) — see git history (PRs #17-#18) if it's wanted again.
- **Game winners = 📈 PREDICTION, not a pick (user's choice, 2026-10-22).** `nba/analysis/winners.py`: linear margin model (home edge, strength = season margin blended with last season, last-10 trend shrunk n/(n+2), rest/b2b, missing rotation Game Score value) → 20,000 numpy margin draws → win %. Model saved in `nba/winners_model.json` (committed; refit with `winners.train_final()` after `winners.build()`). Locked as `<game>.winner.csv`, graded straight up in `reports/live/winner_record.csv`. Backtest 2019-25: 65.9% right, Brier 0.2126, beats team rating/Elo/home 7/7 seasons; opening week ~3-4 pts over-confident. Lock-file readers must skip `.winner.csv` and `.kalshi.csv`.
- Full plain-English method: `reports/HOW_PROJECTIONS_WORK.md` — keep it in sync when the model changes.
- **Disagreement cap (user rule, 2026-10-28):** a pick is also dropped when our probability beats DraftKings' implied probability (vig included) by more than `MAX_DISAGREE = 0.20` (`espn_props.ok_mask`; Oct 22-27 leans 20+ pts above DK hit 41%, −21%, n=54 — small sample). Applies to the record from `CAP_FROM = 2026-10-28` on.
- **Picks must be bettable at DraftKings (user rule, 2026-10-24):** a 70%+ lean is a ✅ PICK only if DraftKings' main line (ESPN propBets, frozen at the lock) is ours or better (under: DK line ≥ ours; over: DK line ≤ ours). Graded at DK's line + American odds (`record.dk_view`, `dk_*` ledger columns). Record reset 2026-10-24: earlier days backfilled at DK's CLOSING line (`reports/live/dk_backfill.csv`; ESPN keeps only recent games, so keep that file committed). The own-average @ -120 record stays as reference only. Board shows ⏳ waiting (no DK line yet) / ⏸ DK line worse. DK lines for 70%+ leans are logged every run to `reports/live/dk_lines/<date>.csv` to compare morning vs lock lines (user's choice: keep the 1-hour lock). Lines never feed the model. Scraping DraftKings directly is blocked (403) and against their terms.
- **Board = strongest only:** leans with calibrated probability ≥ `STRONG_MIN = 0.70`, ranked by model probability alone. Ranking backtest (`nba/analysis/rank_backtest.py`, `reports/phase2/ranking_backtest.md`): blending in past hit rates made it worse 7/7 seasons; 70%+ hit 75.8% in 2025. ⚠️ flags star first-game unders (66.7% vs 77.8% stated, n=42).
- **Lean column:** per player, the stat with the strongest calibrated over/under probability vs his OWN baseline (`nba/analysis/lean.py`; needs `data/processed/lean_pick_calibration.json` from `lean.backtest_calibrated()`). Backtest: stated 70%+ → 77.6% hit. Not a book-line test.

## Key findings (details: `reports/phase2/FINDINGS.md`)
- Predictability (10 seasons, stable): MIN > FGA > PRA > PR > PA > RA > PTS > REB > AST > TOV > 3PM > STL > BLK.
- Biggest inputs: teammates out, minutes/role. Opponent defense, pace, matchup/coverage proxies ≈ no value.
- Ensemble beats season average in 7/7 seasons: MIN 25%, PRA 12%, PTS 8%, REB/AST ~7% (squared error).
- Similar-defense signal, head-to-head averages: no edge (placebo-tested). Player-bias correction: small real gain.
- Known gaps: stars under-projected ~1.9 pts in game 1; player spread uses games through 2025-26 only (current-season misses not added yet); preseason sim misses injuries (games played is the largest season-total error).

## Conventions
- `main` is the live branch (scheduled Actions run from it; Actions commits there hourly — pull before pushing). Feature work on `phase1-predictability`, merge via PR.
- New analyses go in `nba/analysis/`, reports in `reports/phase2/`, live sheets in `reports/live/`.
- Don't use `pgrep -f <name>` wait loops (they match themselves) — use background tasks / Monitor.
