"""Test B: which pregame inputs actually improve projections, per stat.

Train on games before SPLIT_DATE, test on games after (out-of-sample, forward in
time). Baselines are the player's season-to-date and last-10 averages. Feature
groups are measured two ways:
  * add-one: season-average features + this group (what does it add alone?)
  * drop-one: full model minus this group (what is lost without it?)
Scoring is squared error (MSE): projections are meant to be the expected value,
and MAE would reward predicting the median (e.g. 0 blocks). Gains are % MSE
reduction; a paired bootstrap on test rows gives the 95% CI.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from nba.config import PROC, MIN_GAMES, MIN_MPG
from nba.analysis.features import STAT_FEATURE_TARGETS

SPLIT_DATE = "2026-02-01"
MIN_PRIOR_GAMES = 5
RNG = np.random.default_rng(7)


def groups(s: str) -> dict[str, list[str]]:
    g = {
        "season_avg": [f"{s}_szn", "gp"] + ([f"{s}_p36_szn"] if s != "min" else []),
        "recent_form": [f"{s}_l5", f"{s}_l10"],
        "minutes_role": ["min_szn", "min_l5", "min_l3", "min_std", "starter_last", "start_rate_l5"],
        "usage_profile": ["usg_szn", "usg_l10", "fga_szn", "fg3a_szn", "fta_szn"],
        "opponent_defense": ([f"oppallow_{s}", f"oppallow_{s}_mypos"] if s != "min" else []) + ["opp_net_prior"],
        "pace": ["exp_pace", "pace_prior", "opp_pace_prior"],
        "game_script": ["net_prior", "net_diff", "abs_net_diff"],
        "rest_home": ["days_rest", "b2b", "home"],
        "teammates_out": ["out_min", "out_usg", "out_min_same_pos"],
    }
    if s == "min":
        g["season_avg"] = ["min_szn", "gp"]
        g["minutes_role"] = ["min_l3", "min_std", "starter_last", "start_rate_l5"]
        g["recent_form"] = ["min_l5", "min_l10"]
    return g


PARAMS = dict(objective="regression", learning_rate=0.03, n_estimators=600, num_leaves=15,
              min_child_samples=60, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              reg_lambda=5.0, verbose=-1, random_state=7)


def _fit_predict(tr, te, feats, y):
    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(tr[feats], tr[y])
    return m.predict(te[feats]), m


def _boot_ci(err_a, err_b, n=1000):
    """95% CI of mean(err_a - err_b) via paired bootstrap; >0 means b is better."""
    d = err_a - err_b
    idx = RNG.integers(0, len(d), (n, len(d)))
    bs = d[idx].mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def load():
    f = pd.read_parquet(PROC / "features.parquet")
    q = f.groupby("nba_id").agg(gp=("gameId", "size"), mpg=("min", "mean"))
    keep = q[(q.gp >= MIN_GAMES) & (q.mpg >= MIN_MPG)].index
    f = f[f.nba_id.isin(keep) & (f.gp >= MIN_PRIOR_GAMES)].copy()
    return f[f.game_date < SPLIT_DATE], f[f.game_date >= SPLIT_DATE]


def run_stat(s, tr, te):
    G = groups(s)
    allf = sorted({c for v in G.values() for c in v})
    tr, te = tr.dropna(subset=[s, f"{s}_szn"]), te.dropna(subset=[s, f"{s}_szn"])
    y = te[s].to_numpy()
    sq = lambda p: (y - p) ** 2
    e_szn = sq(te[f"{s}_szn"].to_numpy())
    e_l10 = sq(te[f"{s}_l10"].fillna(te[f"{s}_szn"]).to_numpy())
    p_full, model = _fit_predict(tr, te, allf, s)
    e_full = sq(p_full)
    p_base, _ = _fit_predict(tr, te, G["season_avg"], s)
    e_base = sq(p_base)
    rm = lambda e: round(float(np.sqrt(e.mean())), 3)
    summary = {"stat": s, "test_rows": len(te), "mean": round(y.mean(), 2),
               "rmse_season_avg": rm(e_szn), "rmse_last10": rm(e_l10),
               "rmse_model_season_only": rm(e_base), "rmse_full_model": rm(e_full),
               "mae_season_avg": round(np.abs(y - te[f"{s}_szn"]).mean(), 3),
               "mae_full_model": round(np.abs(y - p_full).mean(), 3),
               "corr_full_model": round(np.corrcoef(y, p_full)[0, 1], 3)}
    d, lo, hi = _boot_ci(e_szn, e_full)
    summary.update(mse_gain_vs_season_avg_pct=round(100 * d / e_szn.mean(), 1),
                   gain_ci95=f"[{100*lo/e_szn.mean():.1f}, {100*hi/e_szn.mean():.1f}]")
    if s != "min":  # ceiling: what if we knew tonight's minutes exactly?
        p_or, _ = _fit_predict(tr, te, allf + ["min"], s)
        e_or = sq(p_or)
        summary["rmse_if_minutes_known"] = rm(e_or)
        summary["share_of_error_from_minutes_pct"] = round(100 * (e_full.mean() - e_or.mean()) / e_full.mean(), 1)
    rows = []
    for name, cols in G.items():
        if name == "season_avg" or not cols:
            continue
        p_add, _ = _fit_predict(tr, te, sorted(set(G["season_avg"] + cols)), s)
        a, alo, ahi = _boot_ci(e_base, sq(p_add))
        p_drop, _ = _fit_predict(tr, te, [c for c in allf if c not in cols], s)
        dd, dlo, dhi = _boot_ci(sq(p_drop), e_full)
        rows.append({"stat": s, "group": name,
                     "add_one_gain_pct": round(100 * a / e_base.mean(), 2),
                     "add_one_ci95": f"[{100*alo/e_base.mean():.2f}, {100*ahi/e_base.mean():.2f}]",
                     "add_one_significant": bool(alo > 0),
                     "drop_one_loss_pct": round(100 * dd / e_full.mean(), 2),
                     "drop_one_ci95": f"[{100*dlo/e_full.mean():.2f}, {100*dhi/e_full.mean():.2f}]",
                     "drop_one_significant": bool(dlo > 0)})
    imp = pd.DataFrame({"stat": s, "feature": allf, "gain": model.booster_.feature_importance("gain")})
    imp["gain_share"] = (imp.gain / imp.gain.sum()).round(4)
    return summary, rows, imp.sort_values("gain_share", ascending=False).drop(columns="gain")


def run(stats=STAT_FEATURE_TARGETS):
    tr, te = load()
    print(f"train {len(tr)} rows (< {SPLIT_DATE}), test {len(te)} rows")
    S, R, I = [], [], []
    for s in stats:
        a, b, c = run_stat(s, tr, te)
        print(a)
        S.append(a); R += b; I.append(c)
    return pd.DataFrame(S), pd.DataFrame(R), pd.concat(I)


if __name__ == "__main__":
    s, r, i = run()
    pd.set_option("display.width", 250)
    print(s.to_string(index=False)); print(r.to_string(index=False))
