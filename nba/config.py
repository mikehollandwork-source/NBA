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

# Phase 1 player filter (agreed with user).
MIN_GAMES = 20
MIN_MPG = 15.0

STATS = ["min", "pts", "reb", "ast", "fg3m", "stl", "blk", "tov",
         "oreb", "dreb", "fga", "fgm", "fta", "ftm", "fg3a", "pf"]
COMBOS = {"pra": ["pts", "reb", "ast"], "pr": ["pts", "reb"],
          "pa": ["pts", "ast"], "ra": ["reb", "ast"], "stocks": ["stl", "blk"]}
