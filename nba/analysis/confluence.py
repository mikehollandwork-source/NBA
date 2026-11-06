"""Confluence / safety-filter backtest (user request 2026-11-06).

For every out-of-sample 70%+ lean (2019-25, nba/analysis/lean.py), test whether an extra independent
signal agreeing (or disagreeing) with the lean changes the hit rate beyond the calibrated probability.
Metric per flag: hit - stated when the flag is on vs off, sign consistency across seasons, and a placebo
(flag shuffled within season) so a "gain" from chance is visible.

Run: python -m nba.analysis.confluence   Report: reports/phase2/confluence.md
"""
import numpy as np
import pandas as pd

from nba.analysis import lean
from nba.config import PROC, ROOT

COMBOS = {"pra": ["pts", "reb", "ast"], "pr": ["pts", "reb"], "pa": ["pts", "ast"], "ra": ["reb", "ast"]}
RNG = np.random.default_rng(11)


def pick_col(X, suffix):
    """Value of <lean_stat><suffix> per row."""
    return np.array([X[f"{s}{suffix}"].to_numpy() for s in lean.STATS]).T[np.arange(len(X)),
                                                                            [lean.STATS.index(s) for s in X.lean_stat]]


def build() -> pd.DataFrame:
    P = lean.baselines(pd.read_parquet(PROC / "projections_all_games_oos.parquet"))
    R = pd.concat([lean.pick(lean.probabilities(P[P.season == y], fit=P, exclude_season=y))
                   for y in sorted(P.season.unique())], ignore_index=True)
    B = pd.read_parquet(PROC / "lean_backtest.parquet")
    keep = ["season", "pid", "gameId", "lean_stat", "lean_ref", "proj_min", "team", "opp"] + \
           [f"{s}_{m}" for s in lean.STATS for m in ("two_stage", "direct", "residual", "ensemble")]
    X = B.merge(R[keep], on=["season", "pid", "gameId", "lean_stat"])
    over = (X.lean_side == "over").to_numpy()
    comps = np.vstack([pick_col(X, f"_{m}") for m in ("two_stage", "direct", "residual")])
    ref = X.lean_ref.to_numpy()
    X["votes"] = np.where(over, comps > ref, comps < ref).sum(0)
    X["spread"] = comps.std(0) / np.maximum(np.abs(comps.mean(0)), 1)
    F = pd.read_parquet(PROC / "features_all.parquet")
    F["pid"] = F.pid.astype(str)
    for s, parts in COMBOS.items():
        if f"{s}_l5" not in F:
            F[f"{s}_l5"] = sum(F[f"{p}_l5"] for p in parts)
    fcols = ["min_sd", "min_l5", "starter", "home", "b2b", "n_out", "out_min"] + [f"{s}_l5" for s in lean.STATS]
    X["pid"] = X.pid.astype(str)
    X = X.merge(F[["season", "gameId", "pid"] + fcols].drop_duplicates(["season", "gameId", "pid"]),
                on=["season", "gameId", "pid"], how="left")
    l5 = pick_col(X, "_l5")
    X["l5_agrees"] = np.where(over, l5 > ref, l5 < ref)
    # projected game margin for the player's team (winner model, out-of-sample)
    W = pd.read_parquet(PROC / "winners_backtest.parquet")[["gameId", "home_team", "away_team", "model"]]
    X = X.merge(W, on="gameId", how="left")
    from scipy.stats import norm
    m_home = norm.ppf(X.model.clip(0.01, 0.99)) * 13.4          # back out the projected home margin
    X["team_margin"] = np.where(X.team == X.home_team, m_home, -m_home)
    return X


def flags(X) -> dict:
    q = X.spread.quantile(0.25)
    return {
        "3/3 models agree": X.votes == 3,
        "models tightly agree (top 25%)": X.spread <= q,
        "last-5 average agrees with lean": X.l5_agrees,
        "steady minutes (min SD below median)": X.min_sd <= X.min_sd.median(),
        "role stable (proj min within 3 of last-5)": (X.proj_min - X.min_l5).abs() <= 3,
        "teammates out (n_out>0)": X.n_out > 0,
        "starter": X.starter == 1,
        "home game": X.home == 1,
        "team on back-to-back": X.b2b == 1,
        "blowout risk (|proj margin| >= 10)": X.team_margin.abs() >= 10,
    }


def test(X: pd.DataFrame, min_prob=0.70) -> pd.DataFrame:
    X = X[X.lean_prob_cal >= min_prob].copy()
    rows = []
    for name, f in flags(X).items():
        f = f.fillna(False).astype(bool)
        on, off = X[f], X[~f]
        gap_on = on.lean_hit.mean() - on.lean_prob_cal.mean()
        gap_off = off.lean_hit.mean() - off.lean_prob_cal.mean()
        per = X.assign(f=f).groupby("season").apply(
            lambda g: (g[g.f].lean_hit.mean() - g[g.f].lean_prob_cal.mean())
            - (g[~g.f].lean_hit.mean() - g[~g.f].lean_prob_cal.mean()))
        plc = []
        for _ in range(20):
            sh = X.assign(f=X.assign(f=f).groupby("season").f.transform(lambda s: RNG.permutation(s.to_numpy())))
            a, b = sh[sh.f], sh[~sh.f]
            plc.append((a.lean_hit.mean() - a.lean_prob_cal.mean()) - (b.lean_hit.mean() - b.lean_prob_cal.mean()))
        rows.append({"flag": name, "share_on": round(f.mean(), 3), "n_on": len(on),
                     "hit_on": round(on.lean_hit.mean(), 3), "gap_on": round(gap_on, 3),
                     "gap_off": round(gap_off, 3), "diff": round(gap_on - gap_off, 3),
                     "seasons_same_sign": int((np.sign(per) == np.sign(gap_on - gap_off)).sum()),
                     "placebo_sd": round(float(np.std(plc)), 3)})
    return pd.DataFrame(rows).sort_values("diff", ascending=False)


if __name__ == "__main__":
    f = PROC / "confluence.parquet"
    X = pd.read_parquet(f) if f.exists() else build()
    X.to_parquet(f, index=False)
    T = test(X)
    print(T.to_string(index=False))
    (ROOT / "reports/phase2/confluence.md").write_text(
        "# Confluence / safety-filter backtest (70%+ leans, 2019-25, out-of-sample)\n\n"
        "gap = hit rate minus stated probability; diff = gap(flag on) - gap(flag off); "
        "placebo_sd = spread of diff when the flag is shuffled within season (|diff| should be well above it).\n\n"
        + "```\n" + T.to_string(index=False) + "\n```\n")
