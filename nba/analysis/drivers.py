"""Test B across seasons: which pregame inputs improve projections.

Walk-forward: for each test season S, train on all seasons before S, test on S
(players with >=5 prior games that season). Score = squared error vs actual box
score; gains are % MSE reduction with paired-bootstrap 95% CIs.
For each feature group: drop-one loss (unique value the full model loses
without it), per season, then summarised across seasons.
"""
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from nba.analysis.common import qualify
from nba.analysis.groups import groups, all_features
from nba.config import PROC

MIN_PRIOR_GAMES = 5
TEST_SEASONS = list(range(2019, 2026))
STATS = ["min", "pts", "reb", "ast", "pra", "tov", "fg3m", "stl", "blk"]
PARAMS = dict(objective="regression", learning_rate=0.06, n_estimators=300, num_leaves=31, min_child_samples=100,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, random_state=7,
              n_jobs=int(os.environ.get("NBA_JOBS", 4)))
RNG = np.random.default_rng(7)


def load():
    f = pd.read_parquet(PROC / "features_all.parquet")
    f = qualify(f)
    return f[f.gp >= MIN_PRIOR_GAMES].copy()


def fit_predict(tr, te, feats, y, params=PARAMS, weight=None):
    m = lgb.LGBMRegressor(**params)
    m.fit(tr[feats], tr[y], sample_weight=None if weight is None else tr[weight])
    return m.predict(te[feats]), m


def boot(err_a, err_b, n=500):
    d = err_a - err_b
    idx = RNG.integers(0, len(d), (n, len(d)))
    bs = d[idx].mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def run(stats=STATS, test_seasons=TEST_SEASONS, ablate=True):
    f = load()
    S, R = [], []
    for s in stats:
        G = groups(s)
        feats = all_features(s)
        for y in test_seasons:
            tr = f[(f.season < y)].dropna(subset=[s])
            te = f[(f.season == y)].dropna(subset=[s, f"{s}_szn"])
            yv = te[s].to_numpy()
            e_szn = (yv - te[f"{s}_szn"].to_numpy()) ** 2
            p, _ = fit_predict(tr, te, feats, s)
            e_full = (yv - p) ** 2
            d, lo, hi = boot(e_szn, e_full)
            S.append({"stat": s, "season": y, "test_rows": len(te), "rmse_season_avg": np.sqrt(e_szn.mean()),
                      "rmse_model": np.sqrt(e_full.mean()), "gain_pct": 100 * d / e_szn.mean(),
                      "ci_lo": 100 * lo / e_szn.mean(), "ci_hi": 100 * hi / e_szn.mean()})
            print(s, y, round(S[-1]["gain_pct"], 1), flush=True)
            if not ablate:
                continue
            for name, cols in G.items():
                if not cols or name == "season_avg":
                    continue
                pd_, _ = fit_predict(tr, te, [c for c in feats if c not in cols], s)
                dd, dlo, dhi = boot((yv - pd_) ** 2, e_full)
                R.append({"stat": s, "season": y, "group": name, "drop_loss_pct": 100 * dd / e_full.mean(),
                          "ci_lo": 100 * dlo / e_full.mean(), "ci_hi": 100 * dhi / e_full.mean()})
    S, R = pd.DataFrame(S), pd.DataFrame(R)
    return S, R


def summarise(S, R):
    s = S.groupby("stat").agg(gain_mean=("gain_pct", "mean"), gain_min=("gain_pct", "min"), gain_max=("gain_pct", "max"),
                              seasons_sig_positive=("ci_lo", lambda x: int((x > 0).sum())), seasons=("season", "size"))
    r = R.groupby(["stat", "group"]).agg(loss_mean=("drop_loss_pct", "mean"), loss_min=("drop_loss_pct", "min"),
                                         loss_max=("drop_loss_pct", "max"),
                                         seasons_sig=("ci_lo", lambda x: int((x > 0).sum())),
                                         seasons_positive=("drop_loss_pct", lambda x: int((x > 0).sum())),
                                         seasons=("season", "size")).reset_index()
    r["holds_across_seasons"] = (r.seasons_positive >= r.seasons - 1) & (r.seasons_sig >= r.seasons / 2)
    return s.round(2), r.sort_values(["stat", "loss_mean"], ascending=[True, False]).round(3)
