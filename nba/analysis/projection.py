"""Projection system for the predictable stats: MIN, PTS, REB, AST (+TOV) and
the combos PRA / PR / PA / RA.

Three methods, compared out-of-sample (walk-forward by season):
  two_stage : minutes model x per-minute rate model (rate model sees teammates-out,
              usage share etc., so it learns usage redistribution)
  direct    : model predicts the stat directly
  residual  : season-to-date average + model of tonight's deviation from it
  ensemble  : mean of the three
Two scenarios: 'injury_report' (no starting lineup) and 'lineup_known'.

Distributions: negative binomial (counts) with dispersion fit on other seasons'
out-of-sample residuals; minutes use a normal with spread binned by projection.
"""
import numpy as np
import pandas as pd
from scipy import stats as st

from nba.analysis.drivers import load, fit_predict, boot, TEST_SEASONS
from nba.analysis.groups import all_features
from nba.config import PROC

COMPONENTS = ["pts", "reb", "ast", "tov"]
COMBOS = {"pra": ["pts", "reb", "ast"], "pr": ["pts", "reb"], "pa": ["pts", "ast"], "ra": ["reb", "ast"]}
METHODS = ["two_stage", "direct", "residual", "ensemble"]


def project(test_seasons=TEST_SEASONS, scenarios=("injury_report", "lineup_known")):
    f = load()
    out = []
    for scen in scenarios:
        lk = scen == "lineup_known"
        for y in test_seasons:
            tr, te = f[f.season < y].copy(), f[f.season == y].copy()
            r = te[["season", "gameId", "game_date", "pid", "player", "team", "opp", "min", "min_szn", "min_sd"]
                   + COMPONENTS + list(COMBOS) + [f"{s}_szn" for s in COMPONENTS + list(COMBOS)]
                   + [f"{s}_sd" for s in COMPONENTS + list(COMBOS)]].copy()
            r["scenario"] = scen
            r["proj_min"], _ = fit_predict(tr, te, all_features("min", lk), "min")
            trr = tr[tr["min"] >= 3].copy()
            for s in COMPONENTS + list(COMBOS):
                feats = all_features(s, lk)
                if s in COMPONENTS:
                    trr[f"{s}_rate"] = trr[s] / trr["min"]
                    rate, _ = fit_predict(trr, te, feats, f"{s}_rate", weight="min")
                    r[f"{s}_two_stage"] = r.proj_min * np.clip(rate, 0, None)
                else:
                    r[f"{s}_two_stage"] = sum(r[f"{c}_two_stage"] for c in COMBOS[s])
                r[f"{s}_direct"], _ = fit_predict(tr, te, feats, s)
                trd = tr.assign(_dev=tr[s] - tr[f"{s}_szn"])
                dev, _ = fit_predict(trd, te, feats, "_dev")
                r[f"{s}_residual"] = te[f"{s}_szn"].to_numpy() + dev
                r[f"{s}_ensemble"] = r[[f"{s}_{m}" for m in ("two_stage", "direct", "residual")]].mean(axis=1)
            out.append(r)
            print(scen, y, "done", flush=True)
    p = pd.concat(out, ignore_index=True)
    p.to_parquet(PROC / "projections_oos.parquet", index=False)
    return p


def accuracy(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scen, y), d in p.groupby(["scenario", "season"]):
        for s in ["min"] + COMPONENTS + list(COMBOS):
            yv = d[s].to_numpy()
            base = (yv - d[f"{s}_szn"].to_numpy()) ** 2
            meths = ["proj_min"] if s == "min" else [f"{s}_{m}" for m in METHODS]
            for col in meths:
                e = (yv - d[col].to_numpy()) ** 2
                g, lo, hi = boot(base, e, n=300)
                rows.append({"scenario": scen, "season": y, "stat": s, "method": col.replace(f"{s}_", "") if s != "min" else "model",
                             "rmse": np.sqrt(e.mean()), "rmse_season_avg": np.sqrt(base.mean()),
                             "mae": np.abs(yv - d[col]).mean(), "gain_pct": 100 * g / base.mean(),
                             "sig": lo > 0})
    return pd.DataFrame(rows)


def _nb_alpha(y, mu):
    mu = np.clip(mu, 0.05, None)
    return max(((y - mu) ** 2 - mu).sum() / (mu ** 2).sum(), 1e-4)


def nb_dist(mu, alpha):
    mu = np.clip(mu, 0.05, None)
    n = 1 / alpha
    return st.nbinom(n, n / (n + mu))


def fit_variance(y, mu, bins=20):
    """Var(y | mu) = k * mu^p, fit on binned out-of-sample residuals (log-log)."""
    d = pd.DataFrame({"mu": np.clip(mu, 0.05, None), "v": (y - mu) ** 2})
    g = d.groupby(pd.qcut(d.mu, bins, duplicates="drop"), observed=True).mean()
    p, logk = np.polyfit(np.log(g.mu), np.log(g.v), 1)
    return float(np.exp(logk)), float(p)


def dist(mu, k, p):
    """Negative binomial whose variance follows k*mu^p (falls back to ~Poisson
    when that variance is below the mean)."""
    mu = np.clip(mu, 0.05, None)
    var = k * mu ** p
    alpha = np.clip((var - mu) / mu ** 2, 1e-4, None)
    n = 1 / alpha
    return st.nbinom(n, n / (n + mu))


def _logit(x):
    x = np.clip(x, 1e-4, 1 - 1e-4)
    return np.log(x / (1 - x))


def calibration(p: pd.DataFrame, method: str = "ensemble", scenario: str = "injury_report"):
    """Interval coverage and calibration of P(stat > season-to-date average).
    Variance curve and probability recalibration are fit on all OTHER seasons'
    out-of-sample results, then applied to the held-out season."""
    from sklearn.linear_model import LogisticRegression
    p = p[p.scenario == scenario]
    rows, rel = [], []
    for s in COMPONENTS + list(COMBOS):
        col = f"{s}_{method}"
        for y in sorted(p.season.unique()):
            te, oth = p[p.season == y], p[p.season != y]
            if oth.empty:
                oth = te
            k, pw = fit_variance(oth[s].to_numpy(), oth[col].to_numpy())
            k0, pw0 = fit_variance(oth[s].to_numpy(), oth[f"{s}_szn"].to_numpy())
            d, d0 = dist(te[col].to_numpy(), k, pw), dist(te[f"{s}_szn"].to_numpy(), k0, pw0)
            yv = te[s].to_numpy()
            u = np.random.default_rng(1).uniform(size=len(yv))
            pit = d.cdf(yv - 1) + u * d.pmf(yv)
            line = te[f"{s}_szn"].to_numpy()  # stand-in "line": season-to-date average
            p_raw = 1 - d.cdf(np.floor(line))
            p0 = 1 - d0.cdf(np.floor(line))
            # recalibrate P(over) with logistic regression on logit(p), fit on other seasons
            do = dist(oth[col].to_numpy(), k, pw)
            po = 1 - do.cdf(np.floor(oth[f"{s}_szn"].to_numpy()))
            lr = LogisticRegression().fit(_logit(po)[:, None], (oth[s] > oth[f"{s}_szn"]).astype(int))
            p_cal = lr.predict_proba(_logit(p_raw)[:, None])[:, 1]
            over = (yv > line).astype(float)
            ll = lambda q: -np.mean(over * np.log(np.clip(q, 1e-6, 1)) + (1 - over) * np.log(np.clip(1 - q, 1e-6, 1)))
            rows.append({"stat": s, "season": y, "var_k": k, "var_power": pw,
                         "cov50": np.mean((pit > 0.25) & (pit < 0.75)), "cov80": np.mean((pit > 0.1) & (pit < 0.9)),
                         "brier_season_avg": np.mean((p0 - over) ** 2), "brier_model_raw": np.mean((p_raw - over) ** 2),
                         "brier_model_cal": np.mean((p_cal - over) ** 2),
                         "logloss_season_avg": ll(p0), "logloss_model_raw": ll(p_raw), "logloss_model_cal": ll(p_cal)})
            b = pd.cut(p_cal, [0, .3, .4, .45, .5, .55, .6, .7, 1.0])
            rel.append(pd.DataFrame({"stat": s, "bin": b.astype(str), "p": p_cal, "o": over}))
    cal = pd.DataFrame(rows)
    rel = pd.concat(rel).groupby(["stat", "bin"]).agg(n=("o", "size"), predicted=("p", "mean"), actual=("o", "mean")).reset_index()
    return cal, rel
