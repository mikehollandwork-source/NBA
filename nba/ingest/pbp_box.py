"""Build per-player-game box scores (with exact minutes and on-court possessions)
from nba.com CDN play-by-play. Used to cross-check ESPN and to add NBA player IDs.
"""
import re

import numpy as np
import pandas as pd

from nba.config import PROC

PRESENCE_COLS = ["personId", "assistPersonId", "stealPersonId", "blockPersonId", "foulDrawnPersonId"]


def _clock_sec(s: pd.Series) -> np.ndarray:
    m = s.str.extract(r"PT(\d+)M([\d.]+)S").astype(float)
    return (m[0] * 60 + m[1]).to_numpy()


def _period_start(p):
    return np.where(p <= 4, (p - 1) * 720, 2880 + (p - 5) * 300)


def _period_len(p):
    return np.where(p <= 4, 720, 300)


def load(path) -> pd.DataFrame:
    c = pd.read_csv(path, low_memory=False)
    c = c[c.periodType.isin(["REGULAR", "OVERTIME"])].copy()
    c["t"] = _period_start(c.period) + _period_len(c.period) - _clock_sec(c.clock)
    return c.sort_values(["gameId", "orderNumber"]).reset_index(drop=True)


def counting_stats(c: pd.DataFrame) -> pd.DataFrame:
    made = c.shotResult.eq("Made")
    ev = []
    def add(mask, col, pid="personId", val=1):
        d = c.loc[mask, ["gameId", pid]].rename(columns={pid: "nba_id"})
        d["stat"], d["v"] = col, val if np.isscalar(val) else val[mask]
        ev.append(d)
    fg2, fg3, ft = c.actionType.eq("2pt"), c.actionType.eq("3pt"), c.actionType.eq("freethrow")
    add(fg2 | fg3, "fga"); add((fg2 | fg3) & made, "fgm")
    add(fg3, "fg3a"); add(fg3 & made, "fg3m")
    add(ft, "fta"); add(ft & made, "ftm")
    add(c.actionType.eq("rebound") & c.subType.eq("offensive"), "oreb")
    add(c.actionType.eq("rebound") & c.subType.eq("defensive"), "dreb")
    add((fg2 | fg3) & made & c.assistPersonId.notna(), "ast", "assistPersonId")
    add(c.actionType.eq("turnover"), "tov")
    add(c.stealPersonId.notna() & c.actionType.eq("turnover"), "stl", "stealPersonId")
    add(c.blockPersonId.notna() & (fg2 | fg3), "blk", "blockPersonId")
    add(c.actionType.eq("foul") & c.subType.isin(["personal", "offensive"]), "pf")
    e = pd.concat(ev)
    e = e[(e.nba_id.fillna(0) > 0) & (e.nba_id < 1_000_000_000)]
    box = e.pivot_table(index=["gameId", "nba_id"], columns="stat", values="v", aggfunc="sum", fill_value=0)
    box["reb"] = box.oreb + box.dreb
    box["pts"] = 2 * box.fgm + box.fg3m + box.ftm
    return box.reset_index()


def stints(c: pd.DataFrame) -> pd.DataFrame:
    """On-court intervals [t_in, t_out) per player using subs + period-starter inference."""
    rows, warnings = [], 0
    pteam = c[["gameId", "personId", "teamId"]].dropna()
    pteam = pteam[pteam.personId > 0].drop_duplicates(["gameId", "personId"]).set_index(["gameId", "personId"]).teamId
    for (g, p), d in c.groupby(["gameId", "period"], sort=False):
        t0 = float(_period_start(np.array([p]))[0]); t1 = t0 + float(_period_len(np.array([p]))[0])
        first = {}
        for r in d.itertuples(index=False):
            if r.actionType == "foul" and r.subType == "technical":
                continue  # technicals can be on coaches / team
            if r.actionType == "substitution":
                pid = r.personId
                if pid not in first:
                    first[pid] = r.subType
            else:
                for col in PRESENCE_COLS:
                    pid = getattr(r, col)
                    if pid == pid and pid and pid > 0 and pid not in first:
                        first[pid] = "action"
        on = {pid: t0 for pid, k in first.items() if k in ("out", "action")}
        for r in d[d.actionType == "substitution"].itertuples(index=False):
            if r.subType == "out" and r.personId in on:
                rows.append((g, r.personId, on.pop(r.personId), r.t))
            elif r.subType == "in":
                on[r.personId] = r.t
        for pid, tin in on.items():
            rows.append((g, pid, tin, t1))
    s = pd.DataFrame(rows, columns=["gameId", "nba_id", "t_in", "t_out"])
    s = s[s.nba_id < 1_000_000_000]  # drop team ids that appear in person columns
    s["teamId"] = [pteam.get((g, p), np.nan) for g, p in zip(s.gameId, s.nba_id)]
    return s


def build(cdn_path) -> pd.DataFrame:
    c = load(cdn_path)
    box = counting_stats(c)
    st = stints(c)
    mins = st.assign(sec=st.t_out - st.t_in).groupby(["gameId", "nba_id"]).sec.sum().div(60).rename("min_pbp")
    # on-court check: each team-second should have 5 players
    tri = c[["teamId", "teamTricode"]].dropna().drop_duplicates("teamId").set_index("teamId").teamTricode
    info = st.groupby(["gameId", "nba_id"]).teamId.first().to_frame()
    info["team"] = info.teamId.map(tri)
    out = box.set_index(["gameId", "nba_id"]).join(mins, how="outer").join(info, how="left").fillna(
        {k: 0 for k in box.columns if k not in ("gameId", "nba_id")} | {"min_pbp": 0}).reset_index()
    games = c.groupby("gameId").agg(date=("timeActual", lambda x: str(x.iloc[0])[:10]))
    out = out.join(games, on="gameId")
    PROC.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PROC / "pbp_box.parquet", index=False)
    st.to_parquet(PROC / "stints.parquet", index=False)
    return out
