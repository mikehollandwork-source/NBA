"""Game winners: projected point margin -> numpy-simulated win probability.

Team-level model (player sims alone have no shared game factors and ignore defense):
  home margin = HCA + a·(strength diff) + b·(last-10 diff) + rest terms
                + c·(missing rotation value, home) + d·(missing rotation value, away)
  strength      = blend of season-to-date margin and last season's margin (weight shifts to this
                  season as games are played)
  missing value = prior game-score average of rotation players who are out (rotation = 3+ of the
                  team's last 10 games at 15+ min; active after a team's 3rd game, like the player model)
Linear fit, walk-forward by season (train on all earlier seasons). Win probability = share of
20,000 numpy draws of margin ~ Normal(projection, residual SD) above 0.
Baselines: always home, team rating (season margin + home edge), Elo. Control: no missing-player
terms; placebo: missing-player values shuffled within season.
Run: python -m nba.analysis.winners   Report: reports/phase2/winners_backtest.md
"""
import bisect
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from nba.analysis.common import load_games
from nba.config import PROC, SEASONS

N_SIMS = 20000
K_PRIOR = 10       # games of weight on last season's margin (tuned below on train seasons)
LAST_REG = 0.6     # last season's margin regressed toward 0
K_L10 = 2          # last-10 trend weight = n/(n+K_L10), n = games in it (tuned on 2016-18 by CV)
RNG = np.random.default_rng(23)


def game_score(g: pd.DataFrame) -> pd.Series:
    return (g.pts + 0.4 * g.fgm - 0.7 * g.fga - 0.4 * (g.fta - g.ftm) + 0.7 * g.oreb + 0.3 * g.dreb
            + g.stl + 0.7 * g.ast + 0.7 * g.blk - 0.4 * g.pf - g.tov)


def team_games(g: pd.DataFrame) -> pd.DataFrame:
    t = g.groupby(["season", "gameId", "game_date", "team", "opp", "home"]).pts.sum().reset_index()
    o = t[["gameId", "team", "pts"]].rename(columns={"team": "opp", "pts": "opp_pts"})
    t = t.merge(o, on=["gameId", "opp"])
    t["margin"] = t.pts - t.opp_pts
    return t.sort_values(["team", "game_date", "gameId"]).reset_index(drop=True)


def player_values(g: pd.DataFrame):
    """pid -> (sorted dates, prior-inclusive rolling game score avg over last 20, rolling mpg)."""
    g = g.sort_values(["pid", "game_date", "gameId"]).copy()
    g["gs"] = game_score(g)
    r = g.groupby("pid")
    g["gs20"] = r.gs.transform(lambda x: x.rolling(20, min_periods=1).mean())
    out = {}
    for pid, d in g.groupby("pid"):
        out[pid] = (d.game_date.to_numpy(), d.gs20.to_numpy())
    return out


def value_before(vals, pid, date):
    v = vals.get(pid)
    if v is None:
        return 0.0
    i = bisect.bisect_left(v[0], np.datetime64(date)) - 1
    return float(v[1][i]) if i >= 0 else 0.0


def missing_values(g: pd.DataFrame, T: pd.DataFrame, vals) -> pd.DataFrame:
    """Per team-game: summed prior game-score avg of rotation players not playing (team game 4+)."""
    played = g.groupby(["team", "gameId"]).apply(lambda d: dict(zip(d.pid, d["min"])), include_groups=False)
    rows = []
    for (season, team), d in T.groupby(["season", "team"]):
        hist = deque(maxlen=10)
        for _, q in d.iterrows():
            now = played.get((team, q.gameId), {})
            miss, n_miss = 0.0, 0
            if len(hist) >= 3:
                cnt, mins = defaultdict(int), defaultdict(float)
                for h in hist:
                    for p, m in h.items():
                        cnt[p] += 1
                        mins[p] += m
                rot = [p for p in cnt if cnt[p] >= 3 and mins[p] / cnt[p] >= 15]
                for p in rot:
                    if p not in now:
                        miss += max(value_before(vals, p, q.game_date), 0.0)
                        n_miss += 1
            rows.append((q.gameId, team, miss, n_miss))
            hist.append(now)
    return pd.DataFrame(rows, columns=["gameId", "team", "miss_gs", "n_miss"])


def team_features(T: pd.DataFrame, M: pd.DataFrame) -> pd.DataFrame:
    T = T.merge(M, on=["gameId", "team"], how="left")
    gb = T.groupby(["season", "team"])
    T["gp"] = gb.cumcount()
    T["szn_margin"] = gb.margin.transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
    T["l10_margin"] = gb.margin.transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean()).fillna(0)
    last = T.groupby(["season", "team"]).margin.mean().rename("last_margin").reset_index()
    last["season"] += 1
    T = T.merge(last, on=["season", "team"], how="left")
    T["last_margin"] = T.last_margin.fillna(0)
    prev = T.groupby("team").game_date.shift(1)
    T["rest"] = (T.game_date - prev).dt.days.clip(upper=7).fillna(7)
    T["b2b"] = (T.rest == 1).astype(float)
    return T


def strength(T: pd.DataFrame, k=K_PRIOR, reg=LAST_REG) -> pd.Series:
    f = T.gp / (T.gp + k)
    return f * T.szn_margin + (1 - f) * reg * T.last_margin


def game_rows(T: pd.DataFrame, k=K_PRIOR, reg=LAST_REG, k_l10=K_L10) -> pd.DataFrame:
    T = T.copy()
    T["str"] = strength(T, k, reg)
    n10 = T.gp.clip(upper=10)
    T["l10_margin"] = T.l10_margin * n10 / (n10 + k_l10) if k_l10 else T.l10_margin
    h = T[T.home == 1]
    a = T[T.home == 0].set_index("gameId")
    G = h.set_index("gameId").join(a[["team", "str", "l10_margin", "rest", "b2b", "miss_gs", "n_miss", "gp"]],
                                   rsuffix="_a").reset_index()
    G = G.rename(columns={"team": "home_team", "team_a": "away_team"})
    G["d_str"] = G["str"] - G.str_a
    G["d_l10"] = G.l10_margin - G.l10_margin_a
    G["d_rest"] = G.rest.clip(upper=3) - G.rest_a.clip(upper=3)
    return G


FEATS = ["d_str", "d_l10", "d_rest", "b2b", "b2b_a", "miss_gs", "miss_gs_a"]
FEATS_NO_MISS = ["d_str", "d_l10", "d_rest", "b2b", "b2b_a"]


def fit(G: pd.DataFrame, feats) -> dict:
    X = np.column_stack([np.ones(len(G))] + [G[f].to_numpy(float) for f in feats])
    y = G.margin.to_numpy(float)
    lam = np.eye(X.shape[1]) * 1.0
    lam[0, 0] = 0
    beta = np.linalg.solve(X.T @ X + lam, X.T @ y)
    sd = float(np.std(y - X @ beta))
    return {"beta": beta, "sd": sd, "feats": list(feats)}


def predict(m: dict, G: pd.DataFrame, n_sims=N_SIMS) -> pd.DataFrame:
    X = np.column_stack([np.ones(len(G))] + [G[f].to_numpy(float) for f in m["feats"]])
    mu = X @ m["beta"]
    sims = mu[None, :] + m["sd"] * RNG.standard_normal((n_sims, len(G)))
    return pd.DataFrame({"proj_margin": mu, "p_home": (sims > 0).mean(0)}, index=G.index)


def elo(T_games: pd.DataFrame, k=20, hca=100, carry=0.75) -> pd.Series:
    """538-style Elo with margin-of-victory multiplier; returns pre-game P(home win) per gameId."""
    R, out, last_season = defaultdict(lambda: 1505.0), {}, None
    for _, q in T_games.sort_values(["game_date", "gameId"]).iterrows():
        if q.season != last_season:
            for t in list(R):
                R[t] = 1505 + carry * (R[t] - 1505)
            last_season = q.season
        rh, ra = R[q.home_team], R[q.away_team]
        p = 1 / (1 + 10 ** (-(rh + hca - ra) / 400))
        out[q.gameId] = p
        mov = abs(q.margin)
        diff = (rh + hca - ra) * (1 if q.margin > 0 else -1)
        mult = ((mov + 3) ** 0.8) / (7.5 + 0.006 * diff)
        s = 1.0 if q.margin > 0 else 0.0
        R[q.home_team] += k * mult * (s - p)
        R[q.away_team] -= k * mult * (s - p)
    return pd.Series(out)


def metrics(p, y) -> dict:
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    y = np.asarray(y, float)
    return {"acc": float(((p > 0.5) == (y == 1)).mean()), "brier": float(((p - y) ** 2).mean()),
            "logloss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()), "n": len(y)}


def build(seasons=SEASONS) -> pd.DataFrame:
    g = load_games(list(seasons), qualified_only=False)
    T = team_games(g)
    M = missing_values(g, T, player_values(g))
    T = team_features(T, M)
    T.to_parquet(PROC / "winners_team_games.parquet", index=False)
    return T


def backtest(T: pd.DataFrame | None = None, test_seasons=range(2019, 2026)) -> pd.DataFrame:
    T = pd.read_parquet(PROC / "winners_team_games.parquet") if T is None else T
    # tune the prior weight on seasons before the first test season only
    tr = T[T.season < min(test_seasons)]
    best = min(((k, r) for k in (5, 10, 20, 30) for r in (0.4, 0.6, 0.8)),
               key=lambda kr: ((game_rows(tr, *kr).margin - game_rows(tr, *kr).d_str) ** 2).mean())
    def cv(k2):
        out = []
        for y in (2017, 2018):
            Gc = game_rows(T[T.season <= y], *best, k2)
            Gc["y"] = (Gc.margin > 0).astype(int)
            m = fit(Gc[Gc.season < y], FEATS)
            p = predict(m, Gc[Gc.season == y], 4000).p_home
            out.append(((p - Gc[Gc.season == y].y) ** 2).mean())
        return np.mean(out)
    k_l10 = min((0, 2, 5, 10, 20), key=cv)
    print("last-10 shrink k:", k_l10, flush=True)
    best = (*best, k_l10)
    G = game_rows(T, *best)
    G["y"] = (G.margin > 0).astype(int)
    G["elo"] = G.gameId.map(elo(G))
    rows = []
    for y in test_seasons:
        trn, tst = G[G.season < y], G[G.season == y].copy()
        full, ctrl = fit(trn, FEATS), fit(trn, FEATS_NO_MISS)
        tst["model"] = predict(full, tst).p_home
        tst["no_injury"] = predict(ctrl, tst).p_home
        plc = tst.copy()
        perm = RNG.permutation(len(plc))
        plc[["miss_gs", "miss_gs_a"]] = plc[["miss_gs", "miss_gs_a"]].to_numpy()[perm]
        tst["placebo"] = predict(full, plc).p_home
        rate = fit(trn, ["d_str"])
        tst["team_rating"] = predict(rate, tst).p_home
        tst["home"] = trn.y.mean()
        for name in ["model", "no_injury", "placebo", "team_rating", "elo", "home"]:
            rows.append({"season": y, "method": name, **metrics(tst[name], tst.y)})
        G.loc[tst.index, ["model", "no_injury", "placebo", "team_rating", "home"]] = \
            tst[["model", "no_injury", "placebo", "team_rating", "home"]]
    R = pd.DataFrame(rows)
    G.to_parquet(PROC / "winners_backtest.parquet", index=False)
    return R, G, best


if __name__ == "__main__":
    T = build()
    R, G, best = backtest(T)
    print("prior weight (k, reg):", best)
    print(R.groupby("method")[["acc", "brier", "logloss"]].mean().sort_values("brier").round(4).to_string())
    print(R.pivot(index="season", columns="method", values="acc").round(3).to_string())


# ---------------------------------------------------------------- live
K_PRIOR_FINAL = (20, 0.6, K_L10)  # from backtest(): prior weight, last-season regression, last-10 shrink
MODEL_PATH = Path(__file__).resolve().parents[1] / "winners_model.json"  # committed (tiny); retrain: train_final()


def train_final(T: pd.DataFrame | None = None) -> dict:
    """Fit on every completed season (k, reg tuned on pre-2019 seasons) and save."""
    import json
    T = pd.read_parquet(PROC / "winners_team_games.parquet") if T is None else T
    k, reg, k_l10 = K_PRIOR_FINAL
    m = fit(game_rows(T, k, reg, k_l10), FEATS)
    out = {"beta": [float(b) for b in m["beta"]], "sd": m["sd"], "feats": m["feats"], "k": k, "reg": reg,
           "k_l10": k_l10}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(out))
    return out


def load_model() -> dict:
    import json
    if not MODEL_PATH.exists():
        if not (PROC / "winners_team_games.parquet").exists():
            build()
        train_final()
    m = json.loads(MODEL_PATH.read_text())
    m["beta"] = np.array(m["beta"])
    return m


def live_predict(G: pd.DataFrame, R: pd.DataFrame, season: int, d) -> pd.DataFrame:
    """Tonight's games (G: event_id, home, away, tip_utc) with roster/injuries R (pid, team, status).
    Uses last season + this season's completed games (player_games parquets, written by live run)."""
    m = load_model()
    g = load_games([season - 1, season], qualified_only=False)
    # completed games only: drop tonight's placeholder rows (dates are UTC, like the training data)
    g = g[(g.source != "pending") & ~g.gameId.isin(G.event_id.astype("int64"))]
    tip_day = {}
    for _, q in G.iterrows():
        day = pd.Timestamp(pd.Timestamp(q.tip_utc).tz_convert("UTC").date())
        tip_day[q.home] = tip_day[q.away] = day
    T = team_games(g)
    vals = player_values(g)
    T = team_features(T, missing_values(g, T, vals))
    active = R[R.status != "Out"].groupby("team").pid.apply(set).to_dict()
    rows = []
    for team in set(G.home) | set(G.away):
        t = T[(T.team == team)].sort_values("game_date")
        cur = t[t.season == season]
        last_m = t[t.season == season - 1].margin.mean() if len(t[t.season == season - 1]) else 0.0
        gp = len(cur)
        szn = cur.margin.mean() if gp else 0.0
        n10 = min(gp, 10)
        l10 = cur.margin.tail(10).mean() * n10 / (n10 + m.get("k_l10", K_L10)) if gp else 0.0
        rest = min((tip_day[team] - t.game_date.max()).days, 7) if len(t) else 7
        miss, n_miss, names = 0.0, 0, []
        if gp >= 3:
            gt = g[(g.team == team) & g.gameId.isin(cur.gameId.tail(10))]
            c = gt.groupby("pid").agg(n=("gameId", "size"), mpg=("min", "mean"))
            rot = c[(c.n >= 3) & (c.mpg >= 15)].index
            for p in rot:
                if p not in active.get(team, set()):
                    miss += max(value_before(vals, p, tip_day[team]), 0.0)
                    n_miss += 1
                    names.append(g[g.pid == p].player.iloc[-1])
        rows.append({"team": team, "gp": gp, "szn_margin": szn, "l10_margin": l10, "last_margin": last_m,
                     "rest": rest, "b2b": float(rest == 1), "miss_gs": miss, "n_miss": n_miss,
                     "missing": ", ".join(names)})
    F = pd.DataFrame(rows).set_index("team")
    F["str"] = strength(F.reset_index(), m["k"], m["reg"]).to_numpy()
    out = []
    for _, q in G.iterrows():
        h, a = F.loc[q.home], F.loc[q.away]
        x = {"d_str": h["str"] - a["str"], "d_l10": h.l10_margin - a.l10_margin,
             "d_rest": min(h.rest, 3) - min(a.rest, 3), "b2b": h.b2b, "b2b_a": a.b2b,
             "miss_gs": h.miss_gs, "miss_gs_a": a.miss_gs}
        X = pd.DataFrame([x])
        p = predict(m, X).iloc[0]
        fav = q.home if p.p_home >= 0.5 else q.away
        out.append({"event_id": q.event_id, "gameId": int(q.event_id), "home": q.home, "away": q.away,
                    "proj_margin_home": round(float(p.proj_margin), 1), "p_home": round(float(p.p_home), 3),
                    "pick_team": fav, "p_pick": round(float(max(p.p_home, 1 - p.p_home)), 3),
                    "home_missing": h.missing, "away_missing": a.missing})
    return pd.DataFrame(out)
