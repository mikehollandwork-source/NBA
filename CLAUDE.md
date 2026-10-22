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
- `.github/workflows/live.yml`: hourly 16:05–03:05 UTC (noon–11 pm ET in EDT). Runs `scripts/live.py` (date = today ET), `scripts/telegram_notify.py`, commits `reports/live` (locks + Telegram state) back to `main`.
- Telegram: first run of the day posts the full board (strong leans ≥65% marked 🔥, listed first) + CSV; later runs post only changes (status, lean flips/±5 pts, added/removed players, new locks). Secrets: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`. Without them the notifier dry-runs (prints).
- `BBREF_OPTIONAL=1` in bootstrap: if Basketball-Reference blocks the runner, the 326 old games are dropped instead of failing (slightly different training data — note it if it happens).

## Data sources
| Source | Use |
|---|---|
| ESPN site API | Box scores (truth), schedule, rosters, injury report, player bios |
| shufinskiy/nba_data | nba.com play-by-play (CDN 2020+), shot charts 2016+, defender matchups 2017+ |
| Basketball-Reference | 326 games ESPN left blank (2016-18) |
Reconciliation: ESPN vs nba.com play-by-play agree ≥99% on every stat 2020-25; all official games matched every season.

## Model (what the live sheet uses)
- Game-day ensemble = mean of (minutes model × per-minute rate), direct model, and season-avg + deviation model. LightGBM, `drivers.PARAMS`, `random_state=7`. Features in `nba/analysis/groups.py`.
- Preseason projection = 50/50 blend of preseason LightGBM and last season's per-game average (`preseason.final_projection`); fed to game-day models as `pre_*` prior.
- Ranges: negative binomial, variance = k·μ^p fit out-of-sample; PTS/REB/AST linked by a Gaussian copula scaled so PRA spread matches reality.
- Live minutes: team total 240; players with no NBA history get 12.6 min (historical early-season average); top-8 keep model minutes, bench squeezed (rule, not backtested).
- **Volatility baseline:** no projection if last season's PRA swing (SD / avg, ≥20 games) > `VOL_MAX = 0.45` (`nba/config.py`), or no qualifying last season.
- Teammates-out only activates after a team's 3rd game of the season.
- **Locking:** each run (re)writes a game's projections to `reports/live/locked/<date>/<away>@<home>.csv` only while the game is Scheduled and before tip; after tip the file is never overwritten. Scoring (`score_live.py`) uses locked files only. `scripts/live.py` auto-commits `reports/live` — push after each run so locks survive new sessions. 2026-10-21 locks were backfilled (last version generated before each tip).
- **Lean column:** per player, the stat with the strongest calibrated over/under probability vs his OWN baseline (`nba/analysis/lean.py`; needs `data/processed/lean_pick_calibration.json` from `lean.backtest_calibrated()`). Backtest: stated 70%+ → 77.6% hit. Not a book-line test.

## Key findings (details: `reports/phase2/FINDINGS.md`)
- Predictability (10 seasons, stable): MIN > FGA > PRA > PR > PA > RA > PTS > REB > AST > TOV > 3PM > STL > BLK.
- Biggest inputs: teammates out, minutes/role. Opponent defense, pace, matchup/coverage proxies ≈ no value.
- Ensemble beats season average in 7/7 seasons: MIN 25%, PRA 12%, PTS 8%, REB/AST ~7% (squared error).
- Similar-defense signal, head-to-head averages: no edge (placebo-tested). Player-bias correction: small real gain.
- Known gaps: stars under-projected ~1.9 pts in game 1; ranges not player-specific; preseason sim misses injuries (games played is the largest season-total error).

## Conventions
- `main` is the live branch (scheduled Actions run from it; Actions commits there hourly — pull before pushing). Feature work on `phase1-predictability`, merge via PR.
- New analyses go in `nba/analysis/`, reports in `reports/phase2/`, live sheets in `reports/live/`.
- Don't use `pgrep -f <name>` wait loops (they match themselves) — use background tasks / Monitor.
