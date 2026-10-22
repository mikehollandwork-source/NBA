# NBA player projection system

Built from scratch on free data. **No betting lines or third-party projections are used as
inputs.** Closing lines will be used later only to evaluate results.

## Data sources (free)
| Source | What | Notes |
|---|---|---|
| [shufinskiy/nba_data](https://github.com/shufinskiy/nba_data) | nba.com play-by-play (CDN + v3), defender matchups, shot charts | stats.nba.com blocks cloud IPs; this mirror doesn't |
| ESPN site API | Official box scores (ground truth) | odds fields are stripped before saving |
| Basketball-Reference | Box scores for 326 games ESPN left blank (2016-18) | fetched with curl, 1 request / 3.5 s |

## Run (10 seasons, 2016-17 to 2025-26)
```bash
pip install -r requirements.txt
python scripts/fetch_espn_all.py   # ESPN box scores (cached, ~10 min)
python scripts/reconcile_all.py    # bulk nba.com data + reconciliation + BBRef backfill (~30 min first time)
python scripts/run_phase2.py all   # features, Test A, Test B, projections, disagreement (~1.5 h)
```
Results: `reports/phase2/FINDINGS.md` (Phase 1, single season: `reports/phase1/`).

## Layout
- `nba/ingest/` — downloaders, play-by-play box score + minutes builder, ESPN↔nba.com reconciliation
- `nba/analysis/consistency.py` — Test A: stat consistency / predictability
- `nba/analysis/features.py` — pregame features (no future data)
- `nba/analysis/matchups.py` — defender-matchup difficulty and coverage proxies
- `nba/analysis/drivers.py` — Test B: which inputs improve projections (walk-forward by season, bootstrap CIs)
- `nba/analysis/projection.py` — projection system (minutes x rate, direct, residual, ensemble) + calibrated distributions
- `nba/analysis/disagreement.py` — why projections differ from the season average, and which factors to trust

## Daily projections (live season)
```bash
python scripts/live.py 2026-10-22        # game-day model for that date's slate -> reports/live/game_day_<date>.md/.csv
python scripts/score_live.py 2026-10-21  # after games are final: accuracy vs box scores
```
First run trains and caches models in `data/models/` (~10 min); `--retrain` refreshes them.
Uses ESPN box scores for this season, current rosters and the ESPN injury report. No betting lines.

## Automation: GitHub Actions + Telegram
1. Create a bot: message **@BotFather** on Telegram → `/newbot` → copy the token.
2. Get your chat ID: send any message to your bot, then open `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy `chat.id` (for a group/channel, add the bot first; channel IDs start with `-100`).
3. GitHub repo → Settings → Secrets and variables → Actions → add `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.
4. Actions tab → **bootstrap-data** → Run workflow (once, ~3-4 h).
5. **live-projections** then runs hourly from noon ET; you can also run it manually with a date.
