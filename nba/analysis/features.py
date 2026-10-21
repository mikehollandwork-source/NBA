"""Pregame feature table, all seasons. Every feature uses only games before the
one being predicted (in-season features reset each season; last-season features
come from the previous season's totals).

Leakage notes
* teammates_out / opp_out / match_diff use who actually played. Pregame this
  comes from injury reports; late scratches will make live results a bit worse.
* starter_tonight is only used in the "lineup-known" scenario (lineups are
  announced ~30 min before tip).
"""
import numpy as np
import pandas as pd

from nba.analysis.common import load_games, qualify
from nba.analysis import matchups
from nba.config import PROC, SEASONS

ROLL_STATS = ["min", "pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "pra", "pr", "pa", "ra",
              "fga", "fg3a", "fta", "oreb", "dreb", "usg"]
RATE_STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "fga", "fg3a", "fta", "oreb", "dreb", "usg"]
OPP_STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "pra"]


def _team_games(allg):
    tg = allg.groupby(["season", "gameId", "team"]).agg(
        game_date=("game_date", "first"), opp=("opp", "first"), home=("home", "first"), pts=("pts", "sum"),
        fga=("fga", "sum"), oreb=("oreb", "sum"), tov=("tov", "sum"), fta=("fta", "sum"), usg=("usg", "sum")).reset_index()
    tg["poss"] = tg.fga - tg.oreb + tg.tov + 0.44 * tg.fta
    o = tg[["gameId", "team", "pts", "poss"]].rename(columns={"team": "opp", "pts": "opp_pts", "poss": "opp_poss"})
    tg = tg.merge(o, on=["gameId", "opp"])
    tg["pace"] = (tg.poss + tg.opp_poss) / 2
    tg["net"] = tg.pts - tg.opp_pts
    tg = tg.sort_values(["season", "team", "game_date", "gameId"])
    g = tg.groupby(["season", "team"])
    for c in ("pace", "net", "usg"):
        tg[f"{c}_prior"] = g[c].transform(lambda v: v.shift().expanding().mean())
    tg["team_gnum"] = g.cumcount()
    tg["team_rest"] = g.game_date.diff().dt.days
    return tg


def _opp_allowed(allg, tg):
    a = allg.groupby(["gameId", "opp", "pos"])[OPP_STATS].sum().unstack("pos", fill_value=0)
    a.columns = [f"{s}_{p}" for s, p in a.columns]
    a = a.join(allg.groupby(["gameId", "opp"])[OPP_STATS].sum()).reset_index().rename(columns={"opp": "def_team"})
    a = a.merge(tg[["season", "gameId", "team", "game_date"]].rename(columns={"team": "def_team"}), on=["gameId", "def_team"])
    a = a.sort_values(["season", "def_team", "game_date", "gameId"])
    cols = [c for c in a.columns if c not in ("gameId", "def_team", "game_date", "season")]
    prior = a.groupby(["season", "def_team"])[cols].transform(lambda v: v.shift().expanding().mean())
    out = []
    for y, d in a.groupby("season"):
        daily = d.groupby("game_date")[cols].mean(); cnt = d.groupby("game_date").size()
        league = daily.mul(cnt, axis=0).cumsum().shift().div(cnt.cumsum().shift(), axis=0)
        out.append(prior.loc[d.index] / league.reindex(d.game_date).to_numpy())
    ratio = pd.concat(out).loc[a.index]
    ratio.columns = [f"oppallow_{c}" for c in cols]
    return pd.concat([a[["gameId", "def_team"]], ratio], axis=1)


def _absences(allg, tg):
    """Per team-game: who from the recent rotation is missing tonight, and who is back.
    Rotation = played >=3 of the team's previous 10 games averaging >=15 min.
    Players whose next appearance is for another team (traded) are not 'out'."""
    nxt = allg.sort_values(["pid", "game_date"]).groupby("pid").team.shift(-1).reindex(allg.index)
    allg = allg.assign(traded_after=nxt.notna() & (nxt != allg.team))
    rows = []
    for (season, team), d in tg.groupby(["season", "team"]):
        games = d.sort_values("team_gnum").gameId.to_list()
        gi = {g: i for i, g in enumerate(games)}
        pg = allg[(allg.season == season) & (allg.team == team)]
        P = {c: pg.pivot_table(index="gameId", columns="pid", values=c, aggfunc="sum").reindex(games)
             for c in ("min", "usg", "ast", "reb", "starter")}
        last_traded = pg[pg.traded_after].groupby("pid").gameId.max().map(gi)
        pos = pg.groupby("pid").pos.first()
        for i, gid in enumerate(games):
            if i < 3:
                rows.append((gid, team) + (0.0,) * 11); continue
            w = slice(max(0, i - 10), i)
            wm = P["min"].iloc[w]
            gp, avg = wm.notna().sum(), wm.mean()
            rot = gp[(gp >= 3) & (avg >= 15)].index
            rot = [p for p in rot if not (p in last_traded.index and last_traded[p] < i)]
            tonight = P["min"].iloc[i]
            missing = [p for p in rot if pd.isna(tonight.get(p))]
            back = [p for p in rot if pd.notna(tonight.get(p)) and pd.isna(P["min"].iloc[i - 1].get(p))]
            mu = P["usg"].iloc[w].mean(); ma = P["ast"].iloc[w].mean(); mr = P["reb"].iloc[w].mean()
            sr = P["starter"].iloc[w].mean()
            by = {k: avg[[p for p in missing if pos.get(p) == k]].sum() for k in ("G", "F", "C")}
            rows.append((gid, team, avg[missing].sum(), mu[missing].sum(), ma[missing].sum(), mr[missing].sum(),
                         float((sr[missing] >= 0.5).sum()), by["G"], by["F"], by["C"], avg[back].sum(), mu[back].sum(),
                         float(len(missing))))
    return pd.DataFrame(rows, columns=["gameId", "team", "out_min", "out_usg", "out_ast", "out_reb", "out_starters",
                                       "out_min_G", "out_min_F", "out_min_C", "back_min", "back_usg", "n_out"])


ZONES = {"Restricted Area": "rim", "In The Paint (Non-RA)": "paint", "Mid-Range": "mid",
         "Left Corner 3": "c3", "Right Corner 3": "c3", "Above the Break 3": "ab3"}


def _shot_profile_allowed(seasons):
    """Share of opponent shots by zone a defense has allowed, prior-to-date."""
    from nba.config import RAW
    from nba.ingest.reconcile import official_games
    out = []
    for y in seasons:
        path = RAW / f"bulk/shotdetail_{y}/shotdetail_{y}.csv"
        if not path.exists():  # live season: nba.com mirror not available yet
            continue
        s = pd.read_csv(path, usecols=["GAME_ID", "HTM", "VTM", "TEAM_ID", "SHOT_ZONE_BASIC", "TEAM_NAME"])
        og = official_games(y)[["gameId", "date"]]
        s["zone"] = s.SHOT_ZONE_BASIC.map(ZONES)
        s = s.dropna(subset=["zone"])
        # shooting team tricode: whichever of HTM/VTM matches most of that TEAM_ID's games
        tid = s.groupby(["TEAM_ID", "HTM", "VTM"]).size().reset_index()
        cand = pd.concat([tid[["TEAM_ID", "HTM", 0]].rename(columns={"HTM": "tri"}), tid[["TEAM_ID", "VTM", 0]].rename(columns={"VTM": "tri"})])
        tmap = cand.groupby(["TEAM_ID", "tri"])[0].sum().reset_index().sort_values(0).drop_duplicates("TEAM_ID", keep="last").set_index("TEAM_ID").tri
        s["shooter"] = s.TEAM_ID.map(tmap)
        s["def_team"] = np.where(s.shooter == s.HTM, s.VTM, s.HTM)
        z = s.pivot_table(index=["GAME_ID", "def_team"], columns="zone", values="HTM", aggfunc="size", fill_value=0).reset_index()
        z = z.rename(columns={"GAME_ID": "gameId"}).merge(og, on="gameId").sort_values("date")
        zc = ["rim", "paint", "mid", "c3", "ab3"]
        c = z.groupby("def_team")[zc].cumsum() - z[zc]
        tot = c.sum(axis=1).replace(0, np.nan)
        for k in zc:
            z[f"opp_allow_{k}"] = c[k] / tot
        out.append(z[["gameId", "def_team"] + [f"opp_allow_{k}" for k in zc]])
    return pd.concat(out) if out else pd.DataFrame()


def build(seasons=SEASONS, out_path=None) -> pd.DataFrame:
    allg = load_games(seasons, qualified_only=False)
    allg["usg"] = allg.fga + 0.44 * allg.fta + allg.tov
    tg = _team_games(allg)
    print("team games", len(tg), flush=True)
    opp = _opp_allowed(allg, tg)
    ab = _absences(allg, tg)
    print("absences done", flush=True)

    t = allg
    g = t.groupby(["season", "pid"])
    t["gp"] = g.cumcount()
    t["days_rest"] = g.game_date.diff().dt.days.clip(upper=10)
    for s in ROLL_STATS:
        sh = g[s].shift()
        gs = sh.groupby([t.season, t.pid])
        t[f"{s}_szn"] = gs.transform(lambda v: v.expanding().mean())
        t[f"{s}_l5"] = gs.transform(lambda v: v.rolling(5, min_periods=3).mean())
        t[f"{s}_l10"] = gs.transform(lambda v: v.rolling(10, min_periods=5).mean())
        t[f"{s}_sd"] = gs.transform(lambda v: v.expanding(min_periods=3).std())
    t = t.copy(); g = t.groupby(["season", "pid"])
    tot_min = g["min"].transform(lambda v: v.shift().expanding().sum())
    for s in RATE_STATS:
        t[f"{s}_p36_szn"] = g[s].transform(lambda v: v.shift().expanding().sum()) / tot_min * 36
        t[f"{s}_p36_l10"] = g[s].transform(lambda v: v.shift().rolling(10, min_periods=5).sum()) / \
            g["min"].transform(lambda v: v.shift().rolling(10, min_periods=5).sum()) * 36
    t["min_l3"] = g["min"].transform(lambda v: v.shift().rolling(3, min_periods=2).mean())
    t["starter_last"] = g.starter.shift()
    t["start_rate_l5"] = g.starter.transform(lambda v: v.shift().rolling(5, min_periods=3).mean())
    t["start_rate_szn"] = g.starter.transform(lambda v: v.shift().expanding().mean())

    # last season (by player, any team)
    ps = allg.groupby(["season", "pid"]).agg(prev_gp=("gameId", "size"), prev_mpg=("min", "mean"),
                                             prev_start_rate=("starter", "mean"),
                                             **{f"prev_{s}_tot": (s, "sum") for s in RATE_STATS}, prev_min_tot=("min", "sum"))
    for s in RATE_STATS:
        ps[f"prev_{s}_p36"] = ps[f"prev_{s}_tot"] / ps.prev_min_tot * 36
    ps = ps[[c for c in ps.columns if not c.endswith("_tot")]].reset_index()
    ps["season"] += 1
    t = t.merge(ps, on=["season", "pid"], how="left")

    # team context
    tgi = tg.set_index(["gameId", "team"])
    t = t.join(tgi[["pace_prior", "net_prior", "usg_prior", "team_rest", "team_gnum"]], on=["gameId", "team"])
    o = tgi[["pace_prior", "net_prior", "team_rest"]].rename(columns=lambda c: "opp_" + c)
    t = t.join(o, on=["gameId", "opp"])
    t["exp_pace"] = (t.pace_prior + t.opp_pace_prior) / 2
    t["net_diff"] = t.net_prior - t.opp_net_prior
    t["abs_net_diff"] = t.net_diff.abs()
    t["b2b"] = (t.team_rest == 1).astype(int)
    t["opp_b2b"] = (t.opp_team_rest == 1).astype(int)
    t["usg_share"] = t.usg_szn / t.usg_prior

    # games missed right before tonight (returning from absence)
    tgn = tg.set_index(["gameId", "team"]).team_gnum
    t["tg_now"] = [tgn.get((a, b), np.nan) for a, b in zip(t.gameId, t.team)]
    prev_tg = t.groupby(["season", "pid"]).tg_now.shift()
    same_team = t.groupby(["season", "pid"]).team.shift() == t.team
    t["missed_before"] = np.where(same_team, t.tg_now - prev_tg - 1, 0).clip(0, 15)
    t["first_game_back"] = (t.missed_before >= 1).astype(int)

    t = t.copy()
    # opponent defense + absences
    t = t.merge(opp.rename(columns={"def_team": "opp"}), on=["gameId", "opp"], how="left")
    for s in OPP_STATS:
        t[f"oppallow_{s}_mypos"] = np.select([t.pos == k for k in ("G", "F", "C")], [t[f"oppallow_{s}_{k}"] for k in ("G", "F", "C")])
    t = t.merge(ab, on=["gameId", "team"], how="left")
    oab = ab[["gameId", "team", "out_min", "out_usg"]].rename(columns={"team": "opp", "out_min": "opp_out_min", "out_usg": "opp_out_usg"})
    t = t.merge(oab, on=["gameId", "opp"], how="left")
    t["out_min_same_pos"] = np.select([t.pos == k for k in ("G", "F", "C")], [t[f"out_min_{k}"] for k in ("G", "F", "C")])
    t["out_usg_share"] = t.out_usg / t.usg_prior
    t["usg_x_out"] = t.usg_share * t.out_usg_share

    t = t.copy()
    # defender matchups (2017-18+)
    pos_by_nba = t.dropna(subset=["nba_id"]).groupby("nba_id").pos.agg(lambda x: x.mode().iloc[0])
    mds, scs = [], []
    for y in seasons:
        r = matchups.build_season(y, pos_by_nba)
        if r is not None:
            mds.append(r[0]); scs.append(r[1])
    if not mds:
        mds = [pd.DataFrame(columns=["gameId", "def_team", "pos", "match_diff"])]
        scs = [pd.DataFrame(columns=["gameId", "def_team", "opp_switch_proxy", "opp_cross_share"])]
    md, sc = pd.concat(mds), pd.concat(scs)
    t = t.merge(md.rename(columns={"def_team": "opp"}), on=["gameId", "opp", "pos"], how="left")
    t = t.merge(sc.rename(columns={"def_team": "opp"}), on=["gameId", "opp"], how="left")
    sp = _shot_profile_allowed(seasons)
    if len(sp):
        t = t.merge(sp.rename(columns={"def_team": "opp"}), on=["gameId", "opp"], how="left")
    t = add_preseason(t)
    t.to_parquet(out_path or PROC / "features_all.parquet", index=False)
    return t


def add_preseason(t: pd.DataFrame) -> pd.DataFrame:
    """Merge preseason projections (nba/analysis/preseason.py) as a prior."""
    path = PROC / "preseason_final.parquet"
    if not path.exists():
        return t
    pre = pd.read_parquet(path)
    t = t.drop(columns=[c for c in pre.columns if c.startswith("pre_") and c in t.columns])
    return t.merge(pre, on=["season", "pid"], how="left")


if __name__ == "__main__":
    import time
    t0 = time.time(); f = build(); print(f.shape, round(time.time() - t0), "s")
    q = qualify(f)
    print(q.groupby("season")[["out_min", "back_min", "missed_before", "match_diff", "opp_switch_rate", "prev_mpg"]].mean().round(3))
