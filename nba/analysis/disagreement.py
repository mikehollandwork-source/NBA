"""When the projection disagrees with the season-to-date average, which
game-specific factors drive it, and are they right?

A deviation model predicts (tonight - season avg). LightGBM's exact per-row
feature contributions (TreeSHAP) split each predicted deviation into feature
groups: predicted_dev = bias + sum(group contributions). Out-of-sample we then
check, overall and per factor:
  * hit rate  : % of games where the actual result moved off the season average
                in the direction the model said (ties excluded)
  * capture   : realised move / predicted move (1.0 = moves exactly as much as
                projected; <1 = factor is over-trusted; ~0 = noise)
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from nba.analysis.drivers import load, PARAMS, TEST_SEASONS
from nba.analysis.groups import groups, all_features
from nba.config import PROC

STATS = ["min", "pts", "reb", "ast", "pra", "pr", "pa", "ra"]


def contributions(stats=STATS, test_seasons=TEST_SEASONS):
    f = load()
    out = []
    for s in stats:
        G = groups(s)
        feats = all_features(s)
        f2g = {c: g for g, cols in G.items() for c in cols}
        for y in test_seasons:
            tr = f[f.season < y].dropna(subset=[s, f"{s}_szn"])
            te = f[f.season == y].dropna(subset=[s, f"{s}_szn"])
            m = lgb.LGBMRegressor(**PARAMS).fit(tr[feats], tr[s] - tr[f"{s}_szn"])
            c = m.predict(te[feats], pred_contrib=True)
            cdf = pd.DataFrame(c[:, :-1], columns=feats, index=te.index)
            gdf = cdf.T.groupby(lambda x: f2g[x]).sum().T
            r = pd.DataFrame({"stat": s, "season": y, "pid": te.pid, "player": te.player, "gameId": te.gameId,
                              "szn": te[f"{s}_szn"], "sd": te[f"{s}_sd"], "actual": te[s],
                              "pred_dev": c.sum(1), "bias": c[:, -1]}, index=te.index)
            out.append(pd.concat([r, gdf.add_prefix("g_")], axis=1))
        print("contrib", s, flush=True)
    d = pd.concat(out, ignore_index=True)
    d["actual_dev"] = d.actual - d.szn
    d.to_parquet(PROC / "disagreement_oos.parquet", index=False)
    return d


def _hit(pred, act):
    m = act != 0
    return np.mean(np.sign(pred[m]) == np.sign(act[m])) if m.any() else np.nan


def by_size(d):
    rows = []
    for s, x in d.groupby("stat"):
        z = (x.pred_dev / x.sd.replace(0, np.nan)).abs()
        x = x.assign(z=pd.cut(z, [0, .1, .25, .5, 1, 99], labels=["<0.1 sd", "0.1-0.25 sd", "0.25-0.5 sd", "0.5-1 sd", ">1 sd"]))
        for b, y in x.groupby("z", observed=True):
            rows.append({"stat": s, "disagreement": b, "n": len(y), "share_of_games": len(y) / len(x),
                         "avg_pred_move": y.pred_dev.abs().mean(),
                         "hit_rate": _hit(y.pred_dev.to_numpy(), y.actual_dev.to_numpy()),
                         "capture": (y.actual_dev * np.sign(y.pred_dev)).mean() / y.pred_dev.abs().mean()})
    return pd.DataFrame(rows)


def by_factor(d, top_share=0.10):
    """For each factor: games where that factor moves the projection most
    (top 10% by |contribution| within the stat) — does reality follow?"""
    gcols = [c for c in d.columns if c.startswith("g_")]
    rows = []
    for s, x in d.groupby("stat"):
        # multivariable capture: regress actual deviation on all group contributions
        X = x[gcols].fillna(0).to_numpy(); X1 = np.column_stack([np.ones(len(X)), X])
        beta = np.linalg.lstsq(X1, x.actual_dev.to_numpy() - x.bias.to_numpy(), rcond=None)[0][1:]
        for g, b in zip(gcols, beta):
            c = x[g].fillna(0)
            if (c == 0).all():
                continue
            thr = c.abs().quantile(1 - top_share)
            y = x[c.abs() >= max(thr, 1e-9)]
            rows.append({"stat": s, "factor": g[2:], "n_top": len(y),
                         "avg_push_top": y[g].abs().mean(),
                         "hit_rate_top": _hit(y[g].to_numpy(), y.actual_dev.to_numpy() - y.bias.to_numpy()),
                         "capture_top": ((y.actual_dev - y.bias) * np.sign(y[g])).mean() / y[g].abs().mean(),
                         "capture_all_controlled": b,
                         "mean_abs_push_all": c.abs().mean()})
    return pd.DataFrame(rows).sort_values(["stat", "mean_abs_push_all"], ascending=[True, False])


def examples(d, stat="pts", n=15):
    x = d[d.stat == stat].copy()
    gcols = [c for c in d.columns if c.startswith("g_")]
    x["main_factor"] = x[gcols].abs().idxmax(axis=1).str[2:]
    x["main_push"] = [x.at[i, "g_" + f] for i, f in zip(x.index, x.main_factor)]
    return x.reindex(x.pred_dev.abs().sort_values(ascending=False).index).head(n)[
        ["season", "player", "szn", "pred_dev", "actual", "main_factor", "main_push"]]


def trust_weighted(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-weight each factor's push by how much it has historically been right.
    For each held-out season: fit actual_dev ~ bias + sum(beta_g * contrib_g) on
    the OTHER seasons, apply to the held-out one. Compare squared error vs the
    raw deviation model and vs the season average."""
    gcols = [c for c in d.columns if c.startswith("g_")]
    rows, betas = [], []
    for s, x in d.groupby("stat"):
        for y in sorted(x.season.unique()):
            tr, te = x[x.season != y], x[x.season == y]
            Xtr = np.column_stack([np.ones(len(tr)), tr[gcols].fillna(0)])
            b = np.linalg.lstsq(Xtr, tr.actual_dev - tr.bias, rcond=None)[0]
            pred = te.bias + b[0] + te[gcols].fillna(0).to_numpy() @ b[1:]
            e0 = (te.actual_dev ** 2).mean()
            e_raw = ((te.actual_dev - te.pred_dev) ** 2).mean()
            e_tw = ((te.actual_dev - pred) ** 2).mean()
            rows.append({"stat": s, "season": y, "gain_raw_pct": 100 * (1 - e_raw / e0),
                         "gain_trust_weighted_pct": 100 * (1 - e_tw / e0)})
            betas.append(pd.Series(b[1:], index=[c[2:] for c in gcols], name=(s, y)))
    B = pd.DataFrame(betas)
    B.index = pd.MultiIndex.from_tuples(B.index, names=["stat", "season"])
    return pd.DataFrame(rows), B.groupby("stat").agg(["mean", "std"]).round(2)
