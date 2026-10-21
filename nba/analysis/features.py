"""Pregame feature table for Test B. Every feature uses only games before the
one being predicted (shift(1) everywhere).

Leakage note: 'teammates_out' uses who actually played. In practice this is
known pregame from injury reports, except for rare late scratches.
"""
import numpy as np
import pandas as pd

from nba.analysis.common import load_games
from nba.config import PROC, RAW

STAT_FEATURE_TARGETS = ["min", "pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "pra"]


def _positions(allg: pd.DataFrame) -> pd.Series:
    """G/F/C: listed starting position where available, otherwise nearest
    starter-centroid on per-36 ast/reb/3PA/blk profile."""
    m = pd.read_csv(RAW / "bulk/matchups_2025/matchups_2025.csv", usecols=["person_id", "position"]).dropna()
    listed = m.groupby("person_id").position.agg(lambda x: x.mode().iloc[0])
    prof = allg.groupby("nba_id")[["ast", "reb", "fg3a", "blk", "min"]].sum()
    prof = prof[["ast", "reb", "fg3a", "blk"]].div(prof["min"].clip(lower=1), axis=0) * 36
    z = (prof - prof.mean()) / prof.std()
    known = z.join(listed.rename("pos"), how="inner")
    cent = known.groupby("pos")[["ast", "reb", "fg3a", "blk"]].mean()
    d = ((z.to_numpy()[:, None, :] - cent.to_numpy()[None, :, :]) ** 2).sum(-1)
    inferred = pd.Series(cent.index[d.argmin(1)], index=z.index)
    return listed.reindex(z.index).fillna(inferred)


def _team_games(allg: pd.DataFrame) -> pd.DataFrame:
    tg = allg.groupby(["gameId", "team"]).agg(
        game_date=("game_date", "first"), opp=("opp", "first"), home=("home", "first"),
        pts=("pts", "sum"), fga=("fga", "sum"), oreb=("oreb", "sum"), tov=("tov", "sum"), fta=("fta", "sum")).reset_index()
    tg["poss"] = tg.fga - tg.oreb + tg.tov + 0.44 * tg.fta
    o = tg[["gameId", "team", "pts", "poss"]].rename(columns={"team": "opp", "pts": "opp_pts", "poss": "opp_poss"})
    tg = tg.merge(o, on=["gameId", "opp"])
    tg["pace"] = (tg.poss + tg.opp_poss) / 2
    tg["net"] = tg.pts - tg.opp_pts
    tg = tg.sort_values(["team", "game_date", "gameId"])
    g = tg.groupby("team")
    tg["pace_prior"] = g.pace.transform(lambda v: v.shift().expanding().mean())
    tg["net_prior"] = g.net.transform(lambda v: v.shift().expanding().mean())
    tg["team_gnum"] = g.cumcount()
    tg["team_rest"] = g.game_date.diff().dt.days
    return tg


def _opp_allowed(allg: pd.DataFrame, tg: pd.DataFrame, stats) -> pd.DataFrame:
    """Prior per-game amount of each stat a defense allowed, overall and to each
    position group, as a ratio to the league prior average."""
    a = allg.groupby(["gameId", "opp", "pos"])[stats].sum().unstack("pos", fill_value=0)
    a.columns = [f"{s}_{p}" for s, p in a.columns]
    tot = allg.groupby(["gameId", "opp"])[stats].sum()
    a = a.join(tot).reset_index().rename(columns={"opp": "def_team"})
    a = a.merge(tg[["gameId", "team", "game_date"]].rename(columns={"team": "def_team"}), on=["gameId", "def_team"])
    a = a.sort_values(["def_team", "game_date", "gameId"])
    cols = [c for c in a.columns if c not in ("gameId", "def_team", "game_date")]
    prior = a.groupby("def_team")[cols].transform(lambda v: v.shift().expanding().mean())
    # league prior average by date (all games strictly before the date)
    daily = a.groupby("game_date")[cols].mean()
    cnt = a.groupby("game_date").size()
    league = (daily.mul(cnt, axis=0).cumsum().shift()).div(cnt.cumsum().shift(), axis=0)
    ratio = prior / league.reindex(a.game_date).to_numpy()
    ratio.columns = [f"oppallow_{c}" for c in cols]
    return pd.concat([a[["gameId", "def_team"]], ratio], axis=1)


def _teammates_out(allg: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    """For each team-game: prior minutes / usage of rotation players who did not play.
    Rotation = played >=3 of the team's previous 10 games averaging >=15 min.
    Players whose next appearance is for another team (traded) are not counted as out."""
    allg = allg.copy()
    allg["usg"] = allg.fga + 0.44 * allg.fta + allg.tov
    nxt_team = allg.sort_values("game_date").groupby("nba_id").team.shift(-1)
    nxt_team = nxt_team.reindex(allg.index)
    allg["traded_after"] = nxt_team.notna() & (nxt_team != allg.team)
    out = []
    for team, d in tg.groupby("team"):
        games = d.sort_values("team_gnum").gameId.to_list()
        pg = allg[allg.team == team]
        mins = pg.pivot_table(index="gameId", columns="nba_id", values="min", aggfunc="sum").reindex(games)
        usg = pg.pivot_table(index="gameId", columns="nba_id", values="usg", aggfunc="sum").reindex(games)
        last_traded = pg[pg.traded_after].groupby("nba_id").gameId.max()
        pos = pg.groupby("nba_id").pos.first()
        for i, gid in enumerate(games):
            if i < 3:
                out.append((gid, team, 0.0, 0.0, 0.0, 0.0, 0.0)); continue
            w, wu = mins.iloc[max(0, i - 10):i], usg.iloc[max(0, i - 10):i]
            gp, avg, uavg = w.notna().sum(), w.mean(), wu.mean()
            rot = gp[(gp >= 3) & (avg >= 15)].index
            gone = [p for p in rot if p in last_traded.index and games.index(last_traded[p]) < i]
            missing = [p for p in rot if pd.isna(mins.iloc[i].get(p)) and p not in gone]
            mm = avg[missing].sum(); mu = uavg[missing].sum()
            by = {k: avg[[p for p in missing if pos.get(p) == k]].sum() for k in ("G", "F", "C")}
            out.append((gid, team, mm, mu, by["G"], by["F"], by["C"]))
    return pd.DataFrame(out, columns=["gameId", "team", "out_min", "out_usg", "out_min_G", "out_min_F", "out_min_C"])


def build() -> pd.DataFrame:
    allg = load_games(qualified_only=False)
    pos = _positions(allg)
    allg["pos"] = allg.nba_id.map(pos).fillna("F")
    stats = [s for s in STAT_FEATURE_TARGETS if s != "min"] + ["min", "fga", "fg3a", "fta"]
    tg = _team_games(allg)
    opp = _opp_allowed(allg, tg, [s for s in STAT_FEATURE_TARGETS if s != "min"])
    tout = _teammates_out(allg, tg)

    t = allg.copy()
    t["usg"] = t.fga + 0.44 * t.fta + t.tov
    g = t.groupby("nba_id")
    t["gp"] = g.cumcount()
    t["days_rest"] = g.game_date.diff().dt.days.clip(upper=10)
    for s in stats + ["usg"]:
        sh = g[s].shift()
        t[f"{s}_szn"] = sh.groupby(t.nba_id).transform(lambda v: v.expanding().mean())
        t[f"{s}_l5"] = sh.groupby(t.nba_id).transform(lambda v: v.rolling(5, min_periods=3).mean())
        t[f"{s}_l10"] = sh.groupby(t.nba_id).transform(lambda v: v.rolling(10, min_periods=5).mean())
    tot_min = g["min"].transform(lambda v: v.shift().expanding().sum())
    for s in stats:
        if s != "min":
            t[f"{s}_p36_szn"] = g[s].transform(lambda v: v.shift().expanding().sum()) / tot_min * 36
    t["min_l3"] = g["min"].transform(lambda v: v.shift().rolling(3, min_periods=2).mean())
    t["min_std"] = g["min"].transform(lambda v: v.shift().expanding().std())
    t["starter_last"] = g.starter.shift()
    t["start_rate_l5"] = g.starter.transform(lambda v: v.shift().rolling(5, min_periods=3).mean())

    tgi = tg.set_index(["gameId", "team"])
    t = t.join(tgi[["pace_prior", "net_prior", "team_rest"]], on=["gameId", "team"])
    o = tgi[["pace_prior", "net_prior"]].rename(columns=lambda c: "opp_" + c)
    t = t.join(o, on=["gameId", "opp"], rsuffix="_o")
    t["exp_pace"] = (t.pace_prior + t.opp_pace_prior) / 2
    t["net_diff"] = t.net_prior - t.opp_net_prior
    t["abs_net_diff"] = t.net_diff.abs()
    t["b2b"] = (t.team_rest == 1).astype(int)
    t = t.merge(opp.rename(columns={"def_team": "opp"}), on=["gameId", "opp"], how="left")
    t = t.merge(tout, on=["gameId", "team"], how="left")
    t["out_min_same_pos"] = np.select([t.pos == "G", t.pos == "F", t.pos == "C"], [t.out_min_G, t.out_min_F, t.out_min_C])
    for s in [x for x in STAT_FEATURE_TARGETS if x != "min"]:
        t[f"oppallow_{s}_mypos"] = np.select([t.pos == k for k in ("G", "F", "C")],
                                              [t[f"oppallow_{s}_{k}"] for k in ("G", "F", "C")])
    t.to_parquet(PROC / "features.parquet", index=False)
    return t


if __name__ == "__main__":
    import time
    t0 = time.time(); f = build(); print(f.shape, round(time.time() - t0), "s")
    print(f[["out_min", "out_usg", "exp_pace", "net_diff", "oppallow_pts_mypos", "pts_l10", "days_rest"]].describe().round(2))
