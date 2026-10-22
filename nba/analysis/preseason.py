"""Preseason projections (everything known before opening night of season S).

1. Per-game averages for S: MIN, PTS, REB, AST, 3PM, STL, BLK, TOV
   - LightGBM on the last 3 seasons, age, size, role, team change and roster
     context (usage/minutes vacated on his team, usage competition on the
     opening roster), walk-forward by season.
   - Benchmarks: last season's average, and Marcel (3-season weighted per-minute
     rate 5/4/3, regressed to league average with 1,000 minutes, x weighted mpg).
2. Availability (share of team games played) — injury/absence history from the
   real schedule (episodes, longest absence, finished last season out), ESPN
   injury tags where present, age, size, workload.

Opening roster = players who appear for the team in its first 5 games (known
from preseason rosters in practice).
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from nba.analysis.common import load_games
from nba.config import PROC, SEASONS
from nba.ingest import reconcile

STATS = ["min", "pts", "reb", "ast", "fg3m", "stl", "blk", "tov"]
RATE = [s for s in STATS if s != "min"]
LOWER = ("KNEE", "ANKLE", "FOOT", "ACHILLES", "HAMSTRING", "CALF", "HIP", "TOE", "HEEL", "LEG", "GROIN", "QUAD")
NON_INJURY = ("COACH", "REST", "SUSPEND", "PERSONAL", "NOT WITH TEAM", "G LEAGUE", "G-LEAGUE", "TWO-WAY", "TRADE",
              "PROTOCOL", "ILLNESS", "LOAD MANAGEMENT", "INACTIVE", "BEREAVEMENT", "CONDITIONING")
PARAMS = dict(objective="regression", learning_rate=0.03, n_estimators=500, num_leaves=15, min_child_samples=40,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, random_state=7, n_jobs=4)


def _dnp_reasons(seasons) -> pd.DataFrame:
    out = []
    for y in seasons:
        e = reconcile._load_espn(y)
        og = reconcile.official_games(y)
        e = e.merge(reconcile._match_games(e, og), on="event_id")
        d = e[e.dnp == 1][["gameId", "espn_id", "dnp_reason"]].rename(columns={"espn_id": "pid"})
        d["pid"] = d.pid.astype(str)
        r = d.dnp_reason.fillna("").str.upper()
        d["inj"] = (r != "") & ~r.str.contains("|".join(NON_INJURY))
        d["rest"] = r.str.contains("REST|LOAD MANAGEMENT")
        d["lower"] = d.inj & r.str.contains("|".join(LOWER))
        out.append(d.assign(season=y))
    return pd.concat(out)


def schedules(seasons=SEASONS) -> pd.DataFrame:
    """Every team game for every player who appeared that season (follows trades)."""
    allg = load_games(seasons, qualified_only=False)
    tg = allg.drop_duplicates(["gameId", "team"])[["season", "gameId", "team", "game_date"]]
    tg["tgn"] = tg.sort_values("game_date").groupby(["season", "team"]).cumcount()
    out = []
    for S in seasons:
        played = allg[allg.season == S][["pid", "gameId", "team", "game_date"]].sort_values("game_date")
        pairs = played[["pid", "team"]].drop_duplicates()
        cand = pairs.merge(tg[tg.season == S], on="team").sort_values("game_date")
        at = pd.merge_asof(cand, played.rename(columns={"team": "team_at"})[["pid", "game_date", "team_at"]],
                           on="game_date", by="pid", direction="backward")
        at["team_at"] = at.team_at.fillna(at.pid.map(played.groupby("pid").team.first()))
        x = at[at.team == at.team_at].drop(columns="team_at")
        x["played"] = x.set_index(["pid", "gameId"]).index.isin(played.set_index(["pid", "gameId"]).index)
        out.append(x)
    return pd.concat(out, ignore_index=True).sort_values(["season", "pid", "game_date"]), allg, tg


def _runs(v: np.ndarray) -> tuple[int, int]:
    """(number of absence episodes, longest absence) in a played/missed sequence."""
    m = ~v
    if not m.any():
        return 0, 0
    d = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return len(starts), int((ends - starts).max())


def player_seasons(seasons=SEASONS) -> pd.DataFrame:
    sch, allg, tg = schedules(seasons)
    bio = pd.read_parquet(PROC / "player_bio.parquet")
    allg = allg.assign(usg=allg.fga + 0.44 * allg.fta + allg.tov)
    g = allg.groupby(["season", "pid"])
    ps = g.agg(player=("player", "last"), pos=("pos", "first"), gp=("gameId", "size"), starter=("starter", "mean"),
               team_first=("team", "first"), team_last=("team", "last"),
               **{f"{s}_tot": (s, "sum") for s in STATS + ["usg", "fga", "fta"]})
    for s in STATS + ["usg"]:
        ps[f"{s}_pg"] = ps[f"{s}_tot"] / ps.gp
    for s in RATE + ["usg"]:
        ps[f"{s}_p36"] = ps[f"{s}_tot"] / ps.min_tot.clip(lower=1) * 36
    # availability + absence structure from the schedule
    ab = []
    for (S, pid), d in sch.groupby(["season", "pid"]):
        v = d.played.to_numpy()
        ep, longest = _runs(v)
        ab.append((S, pid, len(v), ep, longest, int((~v[-5:]).any()), int((~v[-10:]).sum())))
    ab = pd.DataFrame(ab, columns=["season", "pid", "team_games", "out_episodes", "out_longest", "out_at_end", "out_last10"])
    ps = ps.join(ab.set_index(["season", "pid"]))
    ps["avail"] = ps.gp / ps.team_games
    dn = _dnp_reasons(seasons).groupby(["season", "pid"]).agg(inj_dnp=("inj", "sum"), rest_dnp=("rest", "sum"),
                                                                lower_dnp=("lower", "sum"))
    ps = ps.join(dn).fillna({"inj_dnp": 0, "rest_dnp": 0, "lower_dnp": 0}).reset_index()
    ps = ps.merge(bio, on="pid", how="left")
    ps["age"] = (pd.to_datetime(ps.season.astype(str) + "-10-15") - ps.dob).dt.days / 365.25
    return ps, allg, tg


def _roster_context(ps: pd.DataFrame, allg: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    """Opening roster (first 5 team games) -> usage/minutes vacated & competition."""
    first5 = tg[tg.tgn < 5][["gameId", "team"]]
    roster = allg.merge(first5, on=["gameId", "team"])[["season", "team", "pid"]].drop_duplicates()
    prev = ps[["season", "pid", "team_last", "usg_tot", "min_tot", "usg_pg", "min_pg"]].copy()
    prev["season"] += 1
    r = roster.merge(prev.drop(columns="team_last"), on=["season", "pid"], how="left").fillna(0)
    comp = r.groupby(["season", "team"])[["usg_pg", "min_pg"]].sum().rename(columns=lambda c: "roster_" + c)
    # vacated: last season's team totals from players not on this opening roster
    lt = prev.rename(columns={"team_last": "team"})
    lt["kept"] = lt.set_index(["season", "team", "pid"]).index.isin(roster.set_index(["season", "team", "pid"]).index)
    tot = lt.groupby(["season", "team"])[["usg_tot", "min_tot"]].sum()
    gone = lt[~lt.kept].groupby(["season", "team"])[["usg_tot", "min_tot"]].sum()
    vac = (gone / tot).rename(columns={"usg_tot": "usg_vacated", "min_tot": "min_vacated"})
    return comp.join(vac).reset_index()


def feature_table(ps, allg, tg) -> pd.DataFrame:
    keep = ["gp", "avail", "starter", "out_episodes", "out_longest", "out_at_end", "out_last10", "inj_dnp", "rest_dnp",
            "lower_dnp", "team_last"] + [f"{s}_pg" for s in STATS + ["usg"]] + [f"{s}_p36" for s in RATE + ["usg"]] \
        + [f"{s}_tot" for s in STATS]
    base = ps[["season", "pid", "player", "pos", "team_first", "height_in", "weight_lb", "age", "gp", "avail"]
              + [f"{s}_pg" for s in STATS]].rename(columns={"gp": "y_gp", "avail": "y_avail", **{f"{s}_pg": f"y_{s}" for s in STATS}})
    f = base
    for k in (1, 2, 3):
        h = ps[["season", "pid"] + keep].copy()
        h["season"] += k
        f = f.merge(h.rename(columns={c: f"{c}_{k}" for c in keep}), on=["season", "pid"], how="left")
    f["team_change"] = (f.team_first != f.team_last_1).astype(float).where(f.team_last_1.notna())
    f = f.merge(_roster_context(ps, allg, tg).rename(columns={"team": "team_first"}), on=["season", "team_first"], how="left")
    own = f.usg_pg_1.fillna(0)
    f["roster_usg_others"] = f.roster_usg_pg - np.where(f.team_change == 0, own, 0)
    f["roster_min_others"] = f.roster_min_pg - np.where(f.team_change == 0, f.min_pg_1.fillna(0), 0)
    f["pos_c"] = f.pos.map({"G": 0, "F": 1, "C": 2})
    for c in ("team_first", "team_last_1", "team_last_2", "team_last_3", "pos"):
        f = f.drop(columns=c, errors="ignore")
    return f


def marcel(f: pd.DataFrame, s: str, league_rate: float) -> pd.Series:
    w = {1: 5, 2: 4, 3: 3}
    num = sum(w[k] * f[f"min_tot_{k}"].fillna(0) for k in w)
    if s == "min":
        gpw = sum(w[k] * f[f"gp_{k}"].fillna(0) for k in w)
        return (num / gpw.replace(0, np.nan))
    tot = sum(w[k] * f[f"{s}_tot_{k}"].fillna(0) for k in w)
    rate = (tot + 1000 * league_rate) / (num + 1000)
    return rate * marcel(f, "min", 0)


def features(f):
    drop = {"season", "pid", "player", "y_gp", "y_avail"} | {f"y_{s}" for s in STATS}
    return [c for c in f.columns if c not in drop and not c.startswith("pre_") and not c.startswith("marcel_")]


def walk_forward(f: pd.DataFrame, test_seasons=range(2018, 2026)) -> pd.DataFrame:
    """OOS preseason projections + availability for each test season."""
    f = f.copy()
    # vectorised marcel
    for s in STATS:
        lrate = (f.groupby("season")[f"{s}_tot_1"].sum() / f.groupby("season")["min_tot_1"].sum()) if s != "min" else None
        f[f"marcel_{s}"] = marcel(f, s, f.season.map(lrate) if lrate is not None else 0)
    X = features(f)
    out = []
    for S in test_seasons:
        tr = f[(f.season < S) & f.gp_1.notna()]
        te = f[(f.season == S) & f.gp_1.notna()].copy()
        for s in STATS:
            t2 = tr[tr.y_gp >= 10]
            m = lgb.LGBMRegressor(**PARAMS).fit(t2[X], t2[f"y_{s}"], sample_weight=np.sqrt(t2.y_gp))
            te[f"pre_{s}"] = np.clip(m.predict(te[X]), 0, None)
        m = lgb.LGBMRegressor(**PARAMS).fit(tr[X], tr.y_avail.clip(0, 1))
        te["pre_avail"] = np.clip(m.predict(te[X]), 0.02, 1)
        out.append(te)
        print("preseason", S, len(te), flush=True)
    return pd.concat(out, ignore_index=True)


def evaluate(p: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-game average accuracy (players with >=10 games this season, rotation last
    season) and availability accuracy (rotation last season)."""
    rot = p[(p.gp_1 >= 20) & (p.min_pg_1 >= 15)]
    rows = []
    for s in STATS:
        d = rot[rot.y_gp >= 10]
        w = d.y_gp
        for name, col in (("preseason model", f"pre_{s}"), ("Marcel", f"marcel_{s}"), ("last season avg", f"{s}_pg_1")):
            e = (d[f"y_{s}"] - d[col]).abs()
            rows.append({"stat": s, "method": name, "player_seasons": len(d), "mae_per_game": (e * w).sum() / w.sum(),
                         "bias": ((d[f"y_{s}"] - d[col]) * w).sum() / w.sum()})
    acc = pd.DataFrame(rows)
    a = []
    for name, col in (("availability model", "pre_avail"), ("last season availability", "avail_1"),
                      ("league average", None)):
        pred = rot[col] if col else pd.Series(rot.y_avail.mean(), index=rot.index)
        a.append({"method": name, "mae_games": ((rot.y_avail - pred).abs() * rot.y_gp.div(rot.y_avail).clip(upper=82)).mean(),
                  "mae_rate": (rot.y_avail - pred).abs().mean()})
    return acc, pd.DataFrame(a)


def final_projection(p: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    """Preseason projection = 50/50 blend of the model and last season's average
    (best for every stat out-of-sample; flat between 0.5 and 0.7). Seasons before
    the model can be trained (2017-18) fall back to last season's average."""
    out = p[["season", "pid"] + [f"pre_{s}" for s in STATS] + ["pre_avail"]].copy()
    p2 = p.set_index(["season", "pid"])
    for s in STATS:
        last = p2[f"{s}_pg_1"].to_numpy()
        out[f"pre_{s}"] = np.where(np.isfinite(last), 0.5 * p2[f"pre_{s}"].to_numpy() + 0.5 * last, p2[f"pre_{s}"].to_numpy())
    early = f[(f.season < p.season.min()) & f.gp_1.notna()][["season", "pid"] + [f"{s}_pg_1" for s in STATS] + ["avail_1"]]
    early = early.rename(columns={**{f"{s}_pg_1": f"pre_{s}" for s in STATS}, "avail_1": "pre_avail"})
    out = pd.concat([early, out], ignore_index=True)
    out["pre_pra"] = out.pre_pts + out.pre_reb + out.pre_ast
    out["pre_pr"] = out.pre_pts + out.pre_reb
    out["pre_pa"] = out.pre_pts + out.pre_ast
    out["pre_ra"] = out.pre_reb + out.pre_ast
    out.to_parquet(PROC / "preseason_final.parquet", index=False)
    return out
