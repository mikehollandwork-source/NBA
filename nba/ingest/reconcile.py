"""Join ESPN box scores to nba.com play-by-play box scores and audit differences."""
import unicodedata

import numpy as np
import pandas as pd

from nba.config import PROC, RAW

ESPN_TO_NBA = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS", "PHX": "PHX"}
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm_last(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower().replace(".", "")
    toks = [t for t in s.replace("-", " ").split() if t not in SUFFIXES]
    return toks[-1] if toks else ""


def _assign_ids(e: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    """Per team-game optimal assignment on (last-name mismatch, minutes, points),
    then a season-wide majority vote so each ESPN id maps to one nba.com id."""
    from scipy.optimize import linear_sum_assignment
    pairs = []
    pg = {k: d for k, d in p.groupby(["gameId", "team"])}
    for k, de in e.groupby(["gameId", "team"]):
        dp = pg.get(k)
        if dp is None:
            continue
        cost = ((de.ln.to_numpy()[:, None] != dp.ln.to_numpy()[None, :]) * 100
                + np.abs(de["min"].to_numpy()[:, None] - dp.min_pbp.to_numpy()[None, :])
                + np.abs(de.pts.to_numpy()[:, None] - dp.pts.to_numpy()[None, :]))
        r, c = linear_sum_assignment(cost)
        ok = cost[r, c] < 100  # require last-name agreement
        pairs.append(pd.DataFrame({"espn_id": de.espn_id.to_numpy()[r[ok]], "nba_id": dp.nba_id.to_numpy()[c[ok]]}))
    votes = pd.concat(pairs).value_counts().rename("n").reset_index()
    return votes.drop_duplicates("espn_id").drop_duplicates("nba_id")


def build() -> pd.DataFrame:
    e = pd.read_parquet(PROC / "espn_box.parquet")
    e["team"] = e.team.replace(ESPN_TO_NBA); e["opp"] = e.opp.replace(ESPN_TO_NBA)
    p = pd.read_parquet(PROC / "pbp_box.parquet")
    names = pd.read_csv(RAW / "bulk/cdnnba_2025/cdnnba_2025.csv", usecols=["gameId", "personId", "playerName"],
                        low_memory=False).dropna().drop_duplicates(["gameId", "personId"])
    p = p.merge(names.rename(columns={"personId": "nba_id"}), on=["gameId", "nba_id"], how="left")
    # game match: same team pair, date within 1 day (both are UTC dates)
    eg = e.groupby("event_id").agg(d=("game_date", "first"), teams=("team", lambda x: tuple(sorted(set(x)))),
                                   tot=("pts", "sum")).reset_index()
    pg = p.groupby("gameId").agg(d=("date", "first"), teams=("team", lambda x: tuple(sorted(set(x.dropna())))),
                                 tot=("pts", "sum")).reset_index()
    gm = eg.merge(pg, on="teams", suffixes=("_e", "_p"))
    gm["dd"] = (pd.to_datetime(gm.d_e) - pd.to_datetime(gm.d_p)).abs().dt.days
    gm["score"] = gm.dd * 1000 + (gm.tot_e - gm.tot_p).abs()  # tie-break same-pair games by total points
    gm = gm[gm.dd <= 1].sort_values("score")
    # one-to-one: teams sometimes meet on back-to-back days
    gm = gm.drop_duplicates("event_id").drop_duplicates("gameId")[["event_id", "gameId"]]
    e = e.merge(gm, on="event_id", how="inner")  # drops All-Star / non-NBA-regular-season games
    e["ln"] = e.player.map(norm_last); p["ln"] = p.playerName.map(norm_last)
    # player id map by majority vote over last-name matches within team-games
    idmap = _assign_ids(e[e.dnp == 0], p)
    e = e.merge(idmap[["espn_id", "nba_id"]], on="espn_id", how="left")
    played = e[e.dnp == 0]
    print(f"ESPN games matched: {e.event_id.nunique()} | played rows unmapped: {played.nba_id.isna().sum()}")
    m = played.merge(p, on=["gameId", "nba_id"], how="outer", suffixes=("", "_pbp"), indicator=True)
    return e, p, m


STAT_COLS = ["pts", "reb", "oreb", "dreb", "ast", "stl", "blk", "tov", "fga", "fgm", "fg3a", "fg3m", "fta", "ftm", "pf"]


def truth_table(e: pd.DataFrame, p: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    """Final outcome table. ESPN official box = truth; nba.com PBP fills players ESPN
    dropped (e.g. missing athlete ids). Exact PBP minutes kept alongside."""
    both = m[m._merge == "both"].copy()
    both["source"] = "espn"
    fill = m[(m._merge == "right_only") & (m.min_pbp > 0) & m.playerName.notna()].copy()
    fill["team"] = fill.team_pbp
    for s in STAT_COLS:
        fill[s] = fill[f"{s}_pbp"]
    fill["min"] = fill.min_pbp.round()
    fill["source"] = "pbp_fill"
    game_info = e.groupby("gameId").agg(game_date=("game_date", "first")).reset_index()
    fill = fill.drop(columns=["game_date"], errors="ignore").merge(game_info, on="gameId", how="left")
    teams = e.drop_duplicates(["gameId", "team"]).set_index(["gameId", "team"])[["opp", "home"]]
    fill = fill.drop(columns=["opp", "home"]).join(teams, on=["gameId", "team"])
    fill["player"] = fill.playerName
    fill["starter"] = np.nan
    cols = ["gameId", "game_date", "nba_id", "espn_id", "player", "team", "opp", "home", "starter", "min", "min_pbp",
            "plus_minus", "source"] + STAT_COLS
    t = pd.concat([both[cols], fill.reindex(columns=cols)], ignore_index=True)
    t["nba_id"] = t.nba_id.astype("int64")
    t.to_parquet(PROC / "player_games.parquet", index=False)
    return t


def audit(m: pd.DataFrame) -> pd.DataFrame:
    both = m[m._merge == "both"]
    rows = []
    for s in ["pts", "reb", "oreb", "dreb", "ast", "stl", "blk", "tov", "fga", "fgm", "fg3a", "fg3m", "fta", "ftm", "pf"]:
        diff = both[s] - both[f"{s}_pbp"]
        rows.append({"stat": s, "exact_match_pct": round(100 * (diff == 0).mean(), 2),
                     "rows_off": int((diff != 0).sum()), "mean_abs_diff": round(diff.abs().mean(), 4)})
    d = both["min"] - both["min_pbp"]
    rows.append({"stat": "min", "exact_match_pct": round(100 * (d.abs() <= 1).mean(), 2),
                 "rows_off": int((d.abs() > 1).sum()), "mean_abs_diff": round(d.abs().mean(), 4)})
    a = pd.DataFrame(rows)
    a.attrs["only_espn"] = int((m._merge == "left_only").sum())
    a.attrs["only_pbp"] = int((m._merge == "right_only").sum())
    return a
