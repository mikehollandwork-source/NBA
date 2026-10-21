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


def calibration(p: pd.DataFrame, method: str = "ensemble", scenario: str = "injury_report") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Interval coverage and calibration of P(stat > season-to-date average).
    Dispersion is fit on all OTHER seasons' out-of-sample residuals."""
    p = p[p.scenario == scenario]
    rows, rel = [], []
    for s in COMPONENTS + list(COMBOS):
        col = f"{s}_{method}"
        for y in sorted(p.season.unique()):
            te, oth = p[p.season == y], p[p.season != y]
            if oth.empty:
                oth = te
            a = _nb_alpha(oth[s].to_numpy(), oth[col].to_numpy())
            a0 = _nb_alpha(oth[s].to_numpy(), oth[f"{s}_szn"].to_numpy())
            d, d0 = nb_dist(te[col].to_numpy(), a), nb_dist(te[f"{s}_szn"].to_numpy(), a0)
            yv = te[s].to_numpy()
            # randomized PIT for discrete outcomes
            u = np.random.default_rng(1).uniform(size=len(yv))
            pit = d.cdf(yv - 1) + u * (d.pmf(yv))
            line = te[f"{s}_szn"].to_numpy()  # stand-in "line": season-to-date average
            p_over = 1 - d.cdf(np.floor(line))
            p_over0 = 1 - d0.cdf(np.floor(line))
            over = (yv > line).astype(float)
            rows.append({"stat": s, "season": y, "alpha": a,
                         "cov50": np.mean((pit > 0.25) & (pit < 0.75)), "cov80": np.mean((pit > 0.1) & (pit < 0.9)),
                         "brier_model": np.mean((p_over - over) ** 2), "brier_season_avg": np.mean((p_over0 - over) ** 2),
                         "logloss_model": -np.mean(over * np.log(np.clip(p_over, 1e-6, 1)) + (1 - over) * np.log(np.clip(1 - p_over, 1e-6, 1))),
                         "logloss_season_avg": -np.mean(over * np.log(np.clip(p_over0, 1e-6, 1)) + (1 - over) * np.log(np.clip(1 - p_over0, 1e-6, 1)))})
            b = pd.cut(p_over, [0, .3, .4, .45, .5, .55, .6, .7, 1.0])
            rel.append(pd.DataFrame({"stat": s, "bin": b.astype(str), "p": p_over, "o": over}))
    cal = pd.DataFrame(rows)
    rel = pd.concat(rel).groupby(["stat", "bin"]).agg(n=("o", "size"), predicted=("p", "mean"), actual=("o", "mean")).reset_index()
    return cal, rel
