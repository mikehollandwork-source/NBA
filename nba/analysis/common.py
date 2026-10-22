"""Shared loading for analysis: all seasons, played games, combo stats."""
import pandas as pd

from nba.config import PROC, MIN_GAMES, MIN_MPG, COMBOS, SEASONS

TARGETS = ["min", "pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "oreb", "dreb", "fga", "fg3a", "fta",
           "pra", "pr", "pa", "ra", "stocks"]
POS_MAP = {"PG": "G", "SG": "G", "G": "G", "SF": "F", "PF": "F", "F": "F", "C": "C"}


def load_games(seasons=SEASONS, qualified_only: bool = True) -> pd.DataFrame:
    t = pd.concat([pd.read_parquet(PROC / f"player_games_{y}.parquet") for y in seasons], ignore_index=True)
    t = t[t["min"] > 0].copy()
    t["pid"] = t.espn_id.astype(str)
    t["game_date"] = pd.to_datetime(t.game_date)
    t["pos"] = t.pos.map(POS_MAP)
    t["pos"] = t.pos.fillna(t.groupby("pid").pos.transform(lambda x: x.mode().iloc[0] if x.notna().any() else "F"))
    for k, cols in COMBOS.items():
        t[k] = t[cols].sum(axis=1)
    t = t.sort_values(["season", "pid", "game_date", "gameId"]).reset_index(drop=True)
    if qualified_only:
        t = qualify(t)
    return t


def qualify(t: pd.DataFrame) -> pd.DataFrame:
    g = t.groupby(["season", "pid"]).agg(gp=("gameId", "size"), mpg=("min", "mean"))
    keep = g[(g.gp >= MIN_GAMES) & (g.mpg >= MIN_MPG)].index
    return t[t.set_index(["season", "pid"]).index.isin(keep)].copy()
