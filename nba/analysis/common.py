"""Shared loading for analysis: qualified players, played games, combo stats."""
import pandas as pd

from nba.config import PROC, MIN_GAMES, MIN_MPG, COMBOS

TARGETS = ["min", "pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "oreb", "dreb", "fga", "fg3a", "fta",
           "pra", "pr", "pa", "ra", "stocks"]


def load_games(qualified_only: bool = True) -> pd.DataFrame:
    t = pd.read_parquet(PROC / "player_games.parquet")
    t = t[t["min"] > 0].copy()
    t["game_date"] = pd.to_datetime(t.game_date)
    t["min"] = t.min_pbp.fillna(t["min"])  # exact minutes from play-by-play
    for k, cols in COMBOS.items():
        t[k] = t[cols].sum(axis=1)
    t = t.sort_values(["nba_id", "game_date", "gameId"]).reset_index(drop=True)
    if qualified_only:
        g = t.groupby("nba_id").agg(gp=("gameId", "size"), mpg=("min", "mean"))
        keep = g[(g.gp >= MIN_GAMES) & (g.mpg >= MIN_MPG)].index
        t = t[t.nba_id.isin(keep)].copy()
    return t
