"""Defense fingerprints, as of each game (prior games only).

Four groups, equally weighted in similarity:
  size      : minute-weighted height (all / G / F / C) and weight of the defenders who played
  speed     : athleticism proxies — age, steals & turnovers forced per 100, block rate,
              transition (fast-break) share of shots allowed [2020-21+], defensive rebounding
  coverage  : where & how opponents shoot (rim / paint / mid / corner 3 / above-break 3;
              pull-up / drive / catch-and-shoot / cut / post / putback), switching proxy,
              bigs-on-guards share, foul rate, assisted share allowed
  allowed   : efficiency allowed — pts per 100, rim FG%, mid FG%, 3P%, eFG%, share of
              points scored by opposing G / F / C
True speed / distance (tracking) is not available from free sources reachable here.
Early season: blended with last season's final profile (pseudo-count K games).
"""
import numpy as np
import pandas as pd

from nba.analysis.common import load_games
from nba.config import PROC, RAW, SEASONS
from nba.ingest.reconcile import official_games

K = 10.0
ZONES = {"Restricted Area": "rim", "In The Paint (Non-RA)": "paint", "Mid-Range": "mid",
         "Left Corner 3": "c3", "Right Corner 3": "c3", "Above the Break 3": "ab3"}


def shot_family(a: str) -> str:
    a = a.lower()
    if "putback" in a or "tip" in a:
        return "putback"
    if "cutting" in a or "alley oop" in a:
        return "cut"
    if "pull" in a or "step back" in a or "fadeaway" in a and "turnaround" not in a:
        return "pullup"
    if "turnaround" in a or "hook" in a or "post" in a:
        return "post"
    if "driving" in a or "running" in a or "floating" in a:
        return "drive"
    if a in ("jump shot",):
        return "catch"
    return "other"


FEATURES = {
    "size": ["hgt", "hgt_G", "hgt_F", "hgt_C", "wgt"],
    "speed": ["age", "stl100", "tovf100", "blk_rate", "fastbreak_share", "dreb_pct"],
    "coverage": ["z_rim", "z_paint", "z_mid", "z_c3", "z_ab3", "t_pullup", "t_drive", "t_catch", "t_cut", "t_post",
                 "t_putback", "switch_proxy", "cross_share", "fta_rate", "ast_rate"],
    "allowed": ["drtg", "fg_rim", "fg_mid", "fg3", "efg", "pts_G", "pts_F", "pts_C"],
}
LABELS = {"hgt": "overall height", "hgt_G": "guard height", "hgt_F": "wing/forward height", "hgt_C": "big height",
          "wgt": "weight", "age": "age", "stl100": "steals forced", "tovf100": "turnovers forced",
          "blk_rate": "shot blocking", "fastbreak_share": "transition shots allowed", "dreb_pct": "defensive rebounding",
          "z_rim": "rim attempts allowed", "z_paint": "paint (non-rim) attempts allowed", "z_mid": "midrange allowed",
          "z_c3": "corner 3s allowed", "z_ab3": "above-break 3s allowed", "t_pullup": "pull-ups allowed",
          "t_drive": "drives allowed", "t_catch": "catch-and-shoot allowed", "t_cut": "cuts/lobs allowed",
          "t_post": "post-ups allowed", "t_putback": "putbacks allowed", "switch_proxy": "switching",
          "cross_share": "bigs on guards", "fta_rate": "fouling", "ast_rate": "assisted shots allowed",
          "drtg": "points per 100 allowed", "fg_rim": "rim FG% allowed", "fg_mid": "mid FG% allowed",
          "fg3": "3P% allowed", "efg": "eFG% allowed", "pts_G": "share of pts to guards",
          "pts_F": "share of pts to forwards", "pts_C": "share of pts to centers"}
# direction where higher = better defense (for best/worst lists); 0 = descriptive only
GOOD = {"stl100": 1, "tovf100": 1, "blk_rate": 1, "fastbreak_share": -1, "dreb_pct": 1, "z_rim": -1, "t_cut": -1,
        "t_putback": -1, "fta_rate": -1, "drtg": -1, "fg_rim": -1, "fg_mid": -1, "fg3": -1, "efg": -1, "z_c3": -1}


def _counts(seasons) -> pd.DataFrame:
    """Raw per (defense, game) counts that profile ratios are built from."""
    t = load_games(seasons, qualified_only=False)
    bio = pd.read_parquet(PROC / "player_bio.parquet")
    t = t.merge(bio, on="pid", how="left")
    t["age_y"] = (t.game_date - t.dob).dt.days / 365.25
    # offense (opponent of defense) totals, credited to the defense = t.opp
    off = t.groupby(["season", "gameId", "opp"]).agg(
        o_pts=("pts", "sum"), o_fga=("fga", "sum"), o_fgm=("fgm", "sum"), o_fg3a=("fg3a", "sum"), o_fg3m=("fg3m", "sum"),
        o_fta=("fta", "sum"), o_oreb=("oreb", "sum"), o_tov=("tov", "sum"), o_ast=("ast", "sum")).reset_index()
    for k in ("G", "F", "C"):
        x = t[t.pos == k].groupby(["gameId", "opp"]).pts.sum().rename(f"o_pts_{k}")
        off = off.join(x, on=["gameId", "opp"])
    off = off.rename(columns={"opp": "team"})
    t["mh"] = t["min"] * t.height_in
    t["mw"] = t["min"] * t.weight_lb
    t["ma"] = t["min"] * t.age_y
    t["mok"] = t["min"] * t.height_in.notna()
    dfn = t.groupby(["gameId", "team"]).agg(stl=("stl", "sum"), blk=("blk", "sum"), dreb=("dreb", "sum"),
                                            mh=("mh", "sum"), mw=("mw", "sum"), ma=("ma", "sum"), mok=("mok", "sum"))
    for k in ("G", "F", "C"):
        x = t[t.pos == k].groupby(["gameId", "team"]).agg(**{f"mh_{k}": ("mh", "sum"), f"mok_{k}": ("mok", "sum")})
        dfn = dfn.join(x)
    c = off.join(dfn, on=["gameId", "team"]).fillna(0)
    dates = t.groupby("gameId").game_date.first()
    c["date"] = c.gameId.map(dates)
    return c


def _shots(seasons) -> pd.DataFrame:
    out = []
    for y in seasons:
        s = pd.read_csv(RAW / f"bulk/shotdetail_{y}/shotdetail_{y}.csv",
                        usecols=["GAME_ID", "HTM", "VTM", "TEAM_ID", "SHOT_ZONE_BASIC", "ACTION_TYPE", "SHOT_MADE_FLAG"])
        tid = s.groupby(["TEAM_ID", "HTM", "VTM"]).size().reset_index()
        cand = pd.concat([tid[["TEAM_ID", "HTM", 0]].rename(columns={"HTM": "tri"}),
                          tid[["TEAM_ID", "VTM", 0]].rename(columns={"VTM": "tri"})])
        tmap = cand.groupby(["TEAM_ID", "tri"])[0].sum().reset_index().sort_values(0) \
            .drop_duplicates("TEAM_ID", keep="last").set_index("TEAM_ID").tri
        s["shooter"] = s.TEAM_ID.map(tmap)
        s["team"] = np.where(s.shooter == s.HTM, s.VTM, s.HTM)  # defense
        s["zone"] = s.SHOT_ZONE_BASIC.map(ZONES)
        s["fam"] = s.ACTION_TYPE.astype(str).map(shot_family)
        z = s.pivot_table(index=["GAME_ID", "team"], columns="zone", values="SHOT_MADE_FLAG", aggfunc=["size", "sum"], fill_value=0)
        z.columns = [f"{'n' if a == 'size' else 'm'}_{b}" for a, b in z.columns]
        f = s.pivot_table(index=["GAME_ID", "team"], columns="fam", values="SHOT_MADE_FLAG", aggfunc="size", fill_value=0)
        f.columns = [f"f_{c}" for c in f.columns]
        out.append(z.join(f).reset_index().rename(columns={"GAME_ID": "gameId"}))
    return pd.concat(out).fillna(0)


def _pbp_transition(seasons) -> pd.DataFrame:
    out = []
    for y in seasons:
        p = RAW / f"bulk/cdnnba_{y}/cdnnba_{y}.csv"
        if not p.exists():
            continue
        c = pd.read_csv(p, usecols=["gameId", "actionType", "qualifiers", "teamTricode"], low_memory=False)
        c = c[c.actionType.isin(["2pt", "3pt"])]
        c["fb"] = c.qualifiers.fillna("").str.contains("fastbreak")
        og = official_games(y)[["gameId", "htm", "vtm"]]
        c = c.merge(og, on="gameId")
        c["team"] = np.where(c.teamTricode == c.htm, c.vtm, c.htm)
        out.append(c.groupby(["gameId", "team"]).agg(pbp_fga=("fb", "size"), fb_fga=("fb", "sum")).reset_index())
    return pd.concat(out)


def _switch(seasons) -> pd.DataFrame:
    from nba.analysis import matchups
    f = pd.read_parquet(PROC / "features_all.parquet", columns=["gameId", "opp", "opp_switch_proxy", "opp_cross_share"])
    # these are already prior-to-date (cumulative) values; keep one row per defense-game
    return f.drop_duplicates(["gameId", "opp"]).rename(columns={"opp": "team", "opp_switch_proxy": "switch_proxy",
                                                                "opp_cross_share": "cross_share"})


COUNT_COLS = None


def _ratios(c: pd.DataFrame) -> pd.DataFrame:
    poss = c.o_fga - c.o_oreb + c.o_tov + 0.44 * c.o_fta
    sh = c[[x for x in c.columns if x.startswith("n_")]].sum(axis=1).replace(0, np.nan)
    fam = c[[x for x in c.columns if x.startswith("f_")]].sum(axis=1).replace(0, np.nan)
    r = pd.DataFrame(index=c.index)
    r["hgt"] = c.mh / c.mok
    for k in ("G", "F", "C"):
        r[f"hgt_{k}"] = c[f"mh_{k}"] / c[f"mok_{k}"].replace(0, np.nan)
    r["wgt"] = c.mw / c.mok
    r["age"] = c.ma / c.mok
    r["stl100"] = c.stl / poss * 100
    r["tovf100"] = c.o_tov / poss * 100
    r["blk_rate"] = c.blk / (c.o_fga - c.o_fg3a).replace(0, np.nan)
    r["fastbreak_share"] = c.fb_fga / c.pbp_fga.replace(0, np.nan)
    r["dreb_pct"] = c.dreb / (c.dreb + c.o_oreb).replace(0, np.nan)
    for z in ("rim", "paint", "mid", "c3", "ab3"):
        r[f"z_{z}"] = c[f"n_{z}"] / sh
    for f in ("pullup", "drive", "catch", "cut", "post", "putback"):
        r[f"t_{f}"] = c[f"f_{f}"] / fam
    r["fta_rate"] = c.o_fta / c.o_fga
    r["ast_rate"] = c.o_ast / c.o_fgm
    r["drtg"] = c.o_pts / poss * 100
    r["fg_rim"] = c.m_rim / c.n_rim.replace(0, np.nan)
    r["fg_mid"] = c.m_mid / c.n_mid.replace(0, np.nan)
    r["fg3"] = c.o_fg3m / c.o_fg3a.replace(0, np.nan)
    r["efg"] = (c.o_fgm + 0.5 * c.o_fg3m) / c.o_fga
    for k in ("G", "F", "C"):
        r[f"pts_{k}"] = c[f"o_pts_{k}"] / c.o_pts
    return r


def build(seasons=SEASONS) -> pd.DataFrame:
    c = _counts(seasons)
    c = c.merge(_shots(seasons), on=["gameId", "team"], how="left")
    c = c.merge(_pbp_transition(seasons), on=["gameId", "team"], how="left")
    c[["pbp_fga", "fb_fga"]] = c[["pbp_fga", "fb_fga"]].astype(float)  # NaN before 2020-21
    cols = [x for x in c.columns if x not in ("season", "gameId", "team", "date")]
    c = c.sort_values(["team", "date", "gameId"]).reset_index(drop=True)
    g = c.groupby(["season", "team"])
    cum = g[cols].cumsum() - c[cols]  # prior games only
    n = g.cumcount()
    # last season's per-game averages as a K-game prior
    prev = c.groupby(["season", "team"])[cols].mean().reset_index()
    prev["season"] += 1
    pv = c[["season", "team"]].merge(prev, on=["season", "team"], how="left")[cols]
    blended = cum.add(pv.fillna(0).mul(K), fill_value=0)
    has_prev = pv.notna().any(axis=1)
    ok = (n >= 1) | has_prev
    prof = _ratios(blended)
    prof[["season", "gameId", "team", "date", "games"]] = c[["season", "gameId", "team", "date"]].assign(games=n)
    prof = prof[ok.to_numpy()]
    prof = prof.merge(_switch(seasons), on=["gameId", "team"], how="left")
    # z-score within season across all defense-games (descriptive scaling)
    feats = [f for v in FEATURES.values() for f in v]
    z = prof.groupby("season")[feats].transform(lambda x: (x - x.mean()) / x.std())
    prof[[f"zz_{f}" for f in feats]] = z.fillna(0).to_numpy()
    prof.to_parquet(PROC / "defense_profiles.parquet", index=False)
    return prof


def vectors(prof: pd.DataFrame) -> np.ndarray:
    """Group-balanced vector: each group contributes equally to distance."""
    parts = []
    for grp, feats in FEATURES.items():
        parts.append(prof[[f"zz_{f}" for f in feats]].to_numpy() / np.sqrt(len(feats)))
    return np.hstack(parts)


def card(prof_row: pd.Series, n=3) -> dict:
    """Best / worst / most distinctive traits of one defense profile."""
    feats = [f for v in FEATURES.values() for f in v]
    z = {f: prof_row[f"zz_{f}"] for f in feats}
    good = {f: z[f] * s for f, s in GOOD.items()}
    best = sorted(good, key=good.get, reverse=True)[:n]
    worst = sorted(good, key=good.get)[:n]
    dist = sorted(z, key=lambda f: abs(z[f]), reverse=True)[:n]
    fmt = lambda f: f"{LABELS[f]} ({z[f]:+.1f}σ)"
    return {"best": [fmt(f) for f in best], "worst": [fmt(f) for f in worst], "distinctive": [fmt(f) for f in dist]}
