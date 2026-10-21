# NBA player projection system

Built from scratch on free data. **No betting lines or third-party projections are used as
inputs.** Closing lines will be used later only to evaluate results.

## Data sources (free)
| Source | What | Notes |
|---|---|---|
| [shufinskiy/nba_data](https://github.com/shufinskiy/nba_data) | nba.com play-by-play (CDN + v3), defender matchups, shot charts | stats.nba.com blocks cloud IPs; this mirror doesn't |
| ESPN site API | Official box scores (ground truth) | odds fields are stripped before saving |

## Run
```bash
pip install -r requirements.txt
python scripts/run_phase1.py   # downloads into data/ (gitignored), writes reports/phase1/
```

## Layout
- `nba/ingest/` — downloaders, play-by-play box score + minutes builder, ESPN↔nba.com reconciliation
- `nba/analysis/consistency.py` — Test A: stat consistency / predictability
- `nba/analysis/features.py` — pregame features (no future data)
- `nba/analysis/drivers.py` — Test B: which inputs improve projections (time-split, bootstrap CIs)
- `reports/phase1/FINDINGS.md` — results
