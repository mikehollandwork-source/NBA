"""Shared paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
REPORTS = ROOT / "reports"

# Season label = starting year (2025 -> 2025-26), matching shufinskiy/nba_data naming.
SEASON = 2025
SEASON_START = "2025-10-21"
SEASON_END = "2026-04-12"

SEASONS = list(range(2016, 2026))  # 2016-17 .. 2025-26
# Regular-season date windows (UTC dates; COVID seasons shifted).
SEASON_WINDOWS = {y: (f"{y}-10-01", f"{y + 1}-04-30") for y in SEASONS}
SEASON_WINDOWS[2019] = ("2019-10-01", "2020-08-20")
SEASON_WINDOWS[2020] = ("2020-12-15", "2021-05-20")
SEASON_WINDOWS[2026] = ("2026-10-01", "2027-04-30")
LIVE_SEASON = 2026

# Phase 1 player filter (agreed with user).
MIN_GAMES = 20
MIN_MPG = 15.0

STATS = ["min", "pts", "reb", "ast", "fg3m", "stl", "blk", "tov",
         "oreb", "dreb", "fga", "fgm", "fta", "ftm", "fg3a", "pf"]
COMBOS = {"pra": ["pts", "reb", "ast"], "pr": ["pts", "reb"],
          "pa": ["pts", "ast"], "ra": ["reb", "ast"], "stocks": ["stl", "blk"]}

# Volatility baseline for live projections: no projection if a player's PRA swung
# too much last season (coefficient of variation = game-to-game SD / average, >=20 games).
# Backtest 2019-25: no filter -> 36% typical miss (share of projection), 0.45 -> 29% (keeps ~53%),
# 0.40 -> 27% (keeps ~40%). Players without a qualifying last season get no projection.
VOL_MAX = 0.45
VOL_MIN_GAMES = 20
