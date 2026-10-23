"""Winner-model extensions tested against the current model (2026-10-23, user ideas):
  A. offense vs defense: points scored vs opponent points allowed (both blended with last season),
     plus our player-projection points sum; both scores simulated with numpy (bivariate normal)
  B. fouls / free throws: FTA a team draws and fouls it commits (and its opponent's)
  C. referees: crew whistle rate and crew home lean from the refs' PRIOR games only
     (+ interaction with how much each team lives at the line); placebo = crews shuffled
Walk-forward 2019-25 like nba/analysis/winners.py. Also: do crews move player points / FTA?
Run: python -m nba.analysis.winners_ext   Report: reports/phase2/winners_ext.md
"""
import numpy as np
import pandas as pd

from nba.analysis import winners as W
from nba.analysis.common import load_games
from nba.config import PROC, SEASONS

RNG = np.random.default_rng(41)
K, REG, K2 = W.K_PRIOR_FINAL


def team_box(g: pd.DataFrame) -> pd.DataFrame:
    t = g.groupby(["season", "gameId", "game_date", "team", "opp", "home"])[["pts", "fta", "pf", "fga", "tov", "oreb"]] \
        .sum().reset_index()
    o = t[["gameId", "team", "pts", "fta", "pf"]].rename(columns={"team": "opp", "pts": "opp_pts", "fta": "opp_fta",
                                                                  "pf": "opp_pf"})
    return t.merge(o, on=["gameId", "opp"]).sort_values(["team", "game_date", "gameId"]).reset_index(drop=True)


def blended(T: pd.DataFrame, col: str) -> pd.Series:
    """Season-to-date mean (prior games) blended with last season's mean, like team strength."""
    gb = T.groupby(["season", "team"])[col]
    szn = gb.transform(lambda x: x.shift(1).expanding().mean())
    last = T.groupby(["season", "team"])[col].mean().rename("last").reset_index()
    last["season"] += 1
    lm = T[["season", "team"]].merge(last, on=["season", "team"], how="left")["last"].to_numpy()
    lg = T.groupby("season")[col].transform("mean")
    lm = np.where(np.isnan(lm), lg, lm)
    gp = T.groupby(["season", "team"]).cumcount()
    f = gp / (gp + K)
    return pd.Series(np.where(np.isnan(szn), lm, f * szn + (1 - f) * lm), index=T.index)


def proj_points() -> pd.DataFrame:
    """Sum of our out-of-sample player point projections per team-game (rotation players)."""
    P = pd.read_parquet(PROC / "projections_all_games_oos.parquet")
    return P.groupby(["gameId", "team"]).pts_ensemble.sum().rename("proj_pts").reset_index()


def ref_features(T: pd.DataFrame) -> pd.DataFrame:
    """Per game: crew's prior average total fouls and total FTA per game and prior home margin
    edge vs league (refs' own earlier games, last 3 seasons). Needs data/processed/officials.parquet."""
    O = pd.read_parquet(PROC / "officials.parquet")
    O = O[O.refs.apply(lambda r: isinstance(r, (list, np.ndarray)) and len(r) > 0)].copy()
    tm = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}
    O["home"] = O.home.map(lambda x: tm.get(x, x))
    O["game_date"] = pd.to_datetime(O.game_date)
    H = T[T.home == 1][["season", "gameId", "game_date", "team", "pf", "opp_pf", "fta", "opp_fta", "pts", "opp_pts"]]
    H = H.rename(columns={"team": "home"})
    H = H.merge(O[["game_date", "home", "refs"]], on=["game_date", "home"], how="left")
    H["tot_pf"] = H.pf + H.opp_pf
    H["tot_fta"] = H.fta + H.opp_fta
    H["hmargin"] = H.pts - H.opp_pts
    H = H.sort_values(["game_date", "gameId"]).reset_index(drop=True)
    lg = H.groupby("season")[["tot_pf", "tot_fta", "hmargin"]].transform(lambda x: x.shift(1).expanding().mean())
    hist = {}   # ref -> list of (season, dev_pf, dev_fta, dev_margin)
    out = []
    for i, q in H.iterrows():
        refs = list(q.refs) if isinstance(q.refs, (list, np.ndarray)) else []
        vals = []
        for r in refs:
            h = [x for x in hist.get(r, []) if x[0] >= q.season - 2]
            if len(h) >= 20:
                a = np.array([x[1:] for x in h])
                vals.append(a.mean(0))
        crew = np.mean(vals, 0) if vals else [np.nan, np.nan, np.nan]
        out.append((q.gameId, len(vals), *crew))
        if refs and not np.isnan(lg.loc[i, "tot_pf"]):
            dev = (q.season, q.tot_pf - lg.loc[i, "tot_pf"], q.tot_fta - lg.loc[i, "tot_fta"],
                   q.hmargin - lg.loc[i, "hmargin"])
            for r in refs:
                hist.setdefault(r, []).append(dev)
    R = pd.DataFrame(out, columns=["gameId", "n_refs_known", "crew_pf", "crew_fta", "crew_home"])
    return R


def build() -> pd.DataFrame:
    g = load_games(list(SEASONS), qualified_only=False)
    T = pd.read_parquet(PROC / "winners_team_games.parquet")
    B = team_box(g)
    for c in ["pts", "opp_pts", "fta", "opp_fta", "pf"]:
        B[f"b_{c}"] = blended(B, c)
    T = T.merge(B[["gameId", "team"] + [f"b_{c}" for c in ["pts", "opp_pts", "fta", "opp_fta", "pf"]]],
                on=["gameId", "team"], how="left")
    T = T.merge(proj_points(), on=["gameId", "team"], how="left")
    G = W.game_rows(T, K, REG, K2)
    a = T[T.home == 0].set_index("gameId")[["b_pts", "b_opp_pts", "b_fta", "b_opp_fta", "b_pf", "proj_pts"]]
    G = G.join(a, on="gameId", rsuffix="_a")
    G["home_pts"] = G.pts
    G["away_pts"] = G.opp_pts
    # B: fouls / free throws (team FTA drawn, fouls committed; diff home - away)
    G["d_fta"] = G.b_fta - G.b_fta_a
    G["d_pf"] = G.b_pf - G.b_pf_a
    G["d_projpts"] = G.proj_pts - G.proj_pts_a
    try:
        R = ref_features(B)
        G = G.merge(R, on="gameId", how="left")
        G["crew_known"] = G.n_refs_known.fillna(0) >= 2
        for c in ["crew_pf", "crew_fta", "crew_home"]:
            G[c] = G[c].where(G.crew_known, 0).fillna(0)
        G["ref_fta_x"] = G.crew_fta * (G.d_fta)          # whistle-heavy crew helps the team that lives at the line
    except FileNotFoundError:
        print("officials.parquet missing: referee variant skipped")
    G["y"] = (G.margin > 0).astype(int)
    G.to_parquet(PROC / "winners_ext_games.parquet", index=False)
    return G


def od_sim(trn, tst, n=20000, use_proj=False):
    """Offense vs defense: regress each side's points on its offense and the opponent's points allowed
    (+ home), simulate both scores jointly (bivariate normal with fitted residual correlation)."""
    def X(D, side):
        if side == "home":
            cols = [D.b_pts, D.b_opp_pts_a] + ([D.proj_pts.fillna(D.b_pts)] if use_proj else [])
        else:
            cols = [D.b_pts_a, D.b_opp_pts] + ([D.proj_pts_a.fillna(D.b_pts_a)] if use_proj else [])
        return np.column_stack([np.ones(len(D))] + [c.to_numpy(float) for c in cols])
    bh = np.linalg.lstsq(X(trn, "home"), trn.home_pts.to_numpy(float), rcond=None)[0]
    ba = np.linalg.lstsq(X(trn, "away"), trn.away_pts.to_numpy(float), rcond=None)[0]
    rh = trn.home_pts - X(trn, "home") @ bh
    ra = trn.away_pts - X(trn, "away") @ ba
    C = np.cov(np.vstack([rh, ra]))
    mh, ma = X(tst, "home") @ bh, X(tst, "away") @ ba
    Z = RNG.multivariate_normal([0, 0], C, size=(n,))
    sh = mh[None, :] + Z[:, 0:1]
    sa = ma[None, :] + Z[:, 1:2]
    return pd.Series((sh > sa).mean(0), index=tst.index), np.corrcoef(rh, ra)[0, 1]


def backtest(test_seasons=range(2020, 2026)):  # 2020+: player projections need a prior season
    G = pd.read_parquet(PROC / "winners_ext_games.parquet")
    has_ref = "crew_pf" in G
    sets = {"current": W.FEATS,
            "+ fouls/FT": W.FEATS + ["d_fta", "d_pf"],
            "+ player proj pts": W.FEATS + ["d_projpts"]}
    if has_ref:
        sets["+ referees"] = W.FEATS + ["crew_home", "ref_fta_x"]
        sets["+ everything"] = W.FEATS + ["d_fta", "d_pf", "d_projpts", "crew_home", "ref_fta_x"]
    rows = []
    for y in test_seasons:
        trn, tst = G[G.season < y].copy(), G[G.season == y].copy()
        if y >= 2020:   # projections exist from 2019, so proj features need a 2019+ training season
            pass
        for name, feats in sets.items():
            ok_tr = trn.dropna(subset=feats)
            ok_te = tst.copy()
            for f in feats:
                ok_te[f] = ok_te[f].fillna(0)
            m = W.fit(ok_tr, feats)
            p = W.predict(m, ok_te).p_home
            rows.append({"season": y, "method": name, **W.metrics(p, tst.y)})
        if has_ref:  # placebo: crews shuffled within season
            plc = tst.copy()
            perm = RNG.permutation(len(plc))
            plc[["crew_home", "ref_fta_x"]] = plc[["crew_home", "ref_fta_x"]].to_numpy()[perm]
            m = W.fit(trn.dropna(subset=sets["+ referees"]), sets["+ referees"])
            rows.append({"season": y, "method": "referees placebo", **W.metrics(W.predict(m, plc.fillna(0)).p_home, tst.y)})
        p, rho = od_sim(trn.dropna(subset=["b_pts", "b_opp_pts_a"]), tst)
        rows.append({"season": y, "method": "offense vs defense sim", **W.metrics(p, tst.y)})
        if y >= 2020:
            tr2 = trn[trn.season >= 2019].dropna(subset=["proj_pts", "proj_pts_a"])
            p, _ = od_sim(tr2, tst, use_proj=True)
            rows.append({"season": y, "method": "offense vs defense sim + player proj", **W.metrics(p, tst.y)})
    return pd.DataFrame(rows)


def ref_player_effect():
    """Do crews move player points / FTA vs our projections? Residual by crew whistle rate, out-of-sample."""
    G = pd.read_parquet(PROC / "winners_ext_games.parquet")
    P = pd.read_parquet(PROC / "projections_all_games_oos.parquet")
    P = P.merge(G[["gameId", "crew_fta", "crew_pf", "crew_known"]], on="gameId")
    P = P[P.crew_known]
    P["res_pts"] = P.pts - P.pts_ensemble
    P["res_pra"] = P.pra - P.pra_ensemble
    P["crew"] = pd.qcut(P.crew_fta, 5, labels=["fewest FTA", "2", "3", "4", "most FTA"])
    return P.groupby("crew", observed=True).agg(n=("res_pts", "size"), crew_fta_dev=("crew_fta", "mean"),
                                               pts=("res_pts", "mean"), pra=("res_pra", "mean")).round(3)


if __name__ == "__main__":
    build()
    R = backtest()
    print(R.groupby("method")[["acc", "brier", "logloss"]].mean().sort_values("brier").round(4).to_string())
    print(R.pivot(index="season", columns="method", values="brier").round(4).to_string())
    try:
        print(ref_player_effect().to_string())
    except Exception as ex:
        print("ref player effect skipped:", ex)
