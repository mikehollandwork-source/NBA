"""Defender-matchup features from nba.com matchup data (2017-18+).

match_diff : expected defensive difficulty for a player tonight. Each opposing
             defender who plays gets a prior-to-date rating (points allowed per
             partial possession ABOVE what the scorers he guarded usually score,
             shrunk toward 0), weighted by how much he usually guards this
             player's position group.
opp_switch_proxy : 1 - concentration (HHI) of defender time on each scorer;
             switch-heavy schemes spread a scorer across many defenders.
opp_cross_share  : share of possessions where a center guards a guard or vice
             versa (bigs switched onto guards).
(nba.com's switches_on / help_* columns are empty in the public data.)
All values use only games before the one being predicted.
"""
import numpy as np
import pandas as pd

from nba.config import RAW
from nba.ingest.reconcile import official_games

K_OFF, K_DEF = 150.0, 300.0


def _cum_prior(df, keys, cols, order):
    df = df.sort_values(order)
    c = df.groupby(keys)[cols].cumsum() - df[cols]
    return c


def build_season(season: int, pos_by_nba: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    path = RAW / f"bulk/matchups_{season}/matchups_{season}.csv"
    if not path.exists():
        return None
    cols = {"game_id": "gameId", "team_tricode": "off_team", "person_id": "off", "matchups_person_id": "dfn",
            "partial_possessions": "poss", "player_points": "pts", "switches_on": "sw",
            "help_field_goals_attempted": "help_fga", "matchup_field_goals_attempted": "fga"}
    m = pd.read_csv(path, usecols=list(cols)).rename(columns=cols)
    og = official_games(season)
    m = m.merge(og[["gameId", "date", "htm", "vtm"]], on="gameId")
    m["def_team"] = np.where(m.off_team == m.htm, m.vtm, m.htm)
    m["off_pos"] = m.off.map(pos_by_nba).fillna("F")
    m["def_pos"] = m.dfn.map(pos_by_nba).fillna("F")
    # league pts per partial possession, prior-to-date
    daily = m.groupby("date")[["pts", "poss"]].sum().sort_index()
    cum = daily.cumsum().shift()
    league = (cum.pts / cum.poss).bfill().fillna(daily.pts.sum() / daily.poss.sum())
    m["L"] = m.date.map(league)
    # offensive player's prior scoring rate per partial possession (shrunk to league)
    og_ = m.groupby(["off", "gameId", "date"])[["pts", "poss"]].sum().reset_index().sort_values("date")
    c = og_.groupby("off")[["pts", "poss"]].cumsum() - og_[["pts", "poss"]]
    og_["off_cpts"], og_["off_cposs"] = c.pts, c.poss
    m = m.merge(og_[["off", "gameId", "off_cpts", "off_cposs"]], on=["off", "gameId"])
    m["exp"] = m.poss * (m.off_cpts + K_OFF * m.L) / (m.off_cposs + K_OFF)
    # defender rating (prior-to-date), and positional workload shares
    dg = m.groupby(["dfn", "gameId", "date", "def_team"]).agg(pts=("pts", "sum"), exp=("exp", "sum"), poss=("poss", "sum")).reset_index()
    pp = m.pivot_table(index=["dfn", "gameId"], columns="off_pos", values="poss", aggfunc="sum", fill_value=0)
    pp = pp.reindex(columns=["G", "F", "C"], fill_value=0).add_prefix("p_").reset_index()
    dg = dg.merge(pp, on=["dfn", "gameId"]).sort_values("date")
    cols = ["pts", "exp", "poss", "p_G", "p_F", "p_C"]
    c = dg.groupby("dfn")[cols].cumsum() - dg[cols]
    dg["rating"] = (c.pts - c.exp) / (c.poss + K_DEF)
    ngames = dg.groupby("dfn").cumcount()
    for k in ("G", "F", "C"):
        dg[f"w_{k}"] = c[f"p_{k}"] / ngames.replace(0, np.nan)  # prior avg possessions vs this position per game
    dg = dg.fillna({"w_G": 0, "w_F": 0, "w_C": 0})
    rows = []
    for k in ("G", "F", "C"):
        g = dg.assign(wr=dg[f"w_{k}"] * dg.rating).groupby(["gameId", "def_team"])[["wr", f"w_{k}"]].sum()
        rows.append((g.wr / g[f"w_{k}"].replace(0, np.nan)).rename(k))
    md = pd.concat(rows, axis=1).stack().rename("match_diff").reset_index()
    md.columns = ["gameId", "def_team", "pos", "match_diff"]
    # scheme proxies (team, prior-to-date)
    share = m.poss / m.groupby(["gameId", "off"]).poss.transform("sum").replace(0, np.nan)
    m["hhi_w"] = share ** 2 * m.groupby(["gameId", "off"]).poss.transform("sum")
    m["cross"] = m.poss * (((m.def_pos == "C") & (m.off_pos == "G")) | ((m.def_pos == "G") & (m.off_pos == "C")))
    tg = m.groupby(["def_team", "gameId", "date"]).agg(hhi_w=("hhi_w", "sum"), cross=("cross", "sum"),
                                                       poss=("poss", "sum")).reset_index().sort_values("date")
    c = tg.groupby("def_team")[["hhi_w", "cross", "poss"]].cumsum() - tg[["hhi_w", "cross", "poss"]]
    tg["opp_switch_proxy"] = 1 - c.hhi_w / c.poss.replace(0, np.nan)
    tg["opp_cross_share"] = c.cross / c.poss.replace(0, np.nan)
    return md, tg[["gameId", "def_team", "opp_switch_proxy", "opp_cross_share"]]
