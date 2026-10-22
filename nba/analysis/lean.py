"""Per player: which stat does the model lean on hardest?

For each stat, the reference is the player's own baseline (season-to-date average
once he has 5+ games, otherwise last season's per-game average), turned into a
half-point line placed where a distribution centred on that baseline splits 50/50
(removes the skew trap: average-based lines favour the under by default).
Model probability of going over/under that line comes from the calibrated
negative binomial, then a per-stat logistic recalibration fit on OTHER seasons.
The stat with the largest |P - 50%| is the player's lean.
No betting lines are used; with real lines, swap the reference line for the book's.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from nba.analysis.common import load_games
from nba.analysis.projection import fit_variance, dist
from nba.config import PROC

STATS = ["pts", "reb", "ast", "pra", "pr", "pa", "ra"]


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def reference_line(base: np.ndarray, k: float, p: float) -> np.ndarray:
    """Half-point line where a distribution centred on `base` is closest to 50/50."""
    d = dist(np.clip(base, 0.05, None), k, p)
    med = d.ppf(0.5)
    cands = np.stack([med - 1.5, med - 0.5, med + 0.5, med + 1.5], 1)
    cands = np.clip(cands, 0.5, None)
    pov = np.stack([1 - d.cdf(np.floor(cands[:, j])) for j in range(4)], 1)
    return cands[np.arange(len(base)), np.abs(pov - 0.5).argmin(1)]


def baselines(P: pd.DataFrame) -> pd.DataFrame:
    """season-to-date avg if gp>=5, else last season's per-game avg."""
    last = load_games(qualified_only=False).groupby(["season", "pid"])[STATS].mean().reset_index()
    last["season"] += 1
    P = P.merge(last.rename(columns={s: f"{s}_lastszn" for s in STATS}), on=["season", "pid"], how="left")
    for s in STATS:
        P[f"{s}_base"] = np.where(P.gp >= 5, P[f"{s}_szn"], P[f"{s}_lastszn"])
    return P


def probabilities(P: pd.DataFrame, fit: pd.DataFrame | None = None, exclude_season=None) -> pd.DataFrame:
    """Calibrated P(over reference line) per stat. `fit` = out-of-sample history used to
    fit variance curves and recalibration (excluding `exclude_season`)."""
    fit = P if fit is None else fit
    if exclude_season is not None:
        fit = fit[fit.season != exclude_season]
    out = P.copy()
    for s in STATS:
        f = fit.dropna(subset=[s, f"{s}_ensemble", f"{s}_base"])
        k, pw = fit_variance(f[s].to_numpy(), f[f"{s}_ensemble"].to_numpy())
        kb, pb = fit_variance(f[s].to_numpy(), f[f"{s}_base"].to_numpy())
        # recalibration: logistic on logit(raw P) using the fit set
        fl = reference_line(f[f"{s}_base"].to_numpy(), kb, pb)
        fraw = 1 - dist(f[f"{s}_ensemble"].to_numpy(), k, pw).cdf(np.floor(fl))
        y = (f[s].to_numpy() > fl).astype(int)
        lr = LogisticRegression().fit(_logit(fraw)[:, None], y)
        ok = out[f"{s}_base"].notna() & out[f"{s}_ensemble"].notna()
        line = reference_line(out.loc[ok, f"{s}_base"].to_numpy(), kb, pb)
        raw = 1 - dist(out.loc[ok, f"{s}_ensemble"].to_numpy(), k, pw).cdf(np.floor(line))
        out.loc[ok, f"{s}_ref"] = line
        out.loc[ok, f"{s}_p_over"] = lr.predict_proba(_logit(raw)[:, None])[:, 1]
    return out


def pick(out: pd.DataFrame) -> pd.DataFrame:
    strength = np.column_stack([np.abs(out[f"{s}_p_over"] - 0.5).fillna(-1) for s in STATS])
    j = strength.argmax(1)
    out = out.copy()
    out["lean_stat"] = np.array(STATS)[j]
    out["lean_p_over"] = np.array([out[f"{s}_p_over"].to_numpy() for s in STATS]).T[np.arange(len(out)), j]
    out["lean_side"] = np.where(out.lean_p_over >= 0.5, "over", "under")
    out["lean_prob"] = np.maximum(out.lean_p_over, 1 - out.lean_p_over)
    out["lean_ref"] = np.array([out[f"{s}_ref"].to_numpy() for s in STATS]).T[np.arange(len(out)), j]
    out["lean_proj"] = np.array([out[f"{s}_ensemble"].to_numpy() for s in STATS]).T[np.arange(len(out)), j]
    return out


def backtest() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    P = pd.read_parquet(PROC / "projections_all_games_oos.parquet")
    P = baselines(P)
    res = []
    for y in sorted(P.season.unique()):
        res.append(pick(probabilities(P[P.season == y], fit=P, exclude_season=y)))
    R = pd.concat(res, ignore_index=True)
    # outcome of the picked lean
    act = np.array([R[s].to_numpy() for s in STATS]).T[np.arange(len(R)), [STATS.index(s) for s in R.lean_stat]]
    R["lean_hit"] = np.where(R.lean_side == "over", act > R.lean_ref, act < R.lean_ref)
    # every stat's lean (for comparison: is the top stat better than the rest?)
    allrows = []
    for s in STATS:
        d = R.dropna(subset=[f"{s}_p_over", s])
        side_over = d[f"{s}_p_over"] >= 0.5
        hit = np.where(side_over, d[s] > d[f"{s}_ref"], d[s] < d[f"{s}_ref"])
        allrows.append(pd.DataFrame({"stat": s, "prob": np.maximum(d[f"{s}_p_over"], 1 - d[f"{s}_p_over"]), "hit": hit,
                                     "season": d.season, "is_top": d.lean_stat == s}))
    A = pd.concat(allrows)
    b = pd.cut(R.lean_prob, [0.5, 0.55, 0.6, 0.65, 0.7, 1.0])
    by_strength = R.groupby(b, observed=True).agg(games=("lean_hit", "size"), model_prob=("lean_prob", "mean"),
                                                  hit_rate=("lean_hit", "mean")).reset_index()
    top_vs_rest = A.groupby("is_top").agg(n=("hit", "size"), model_prob=("prob", "mean"), hit_rate=("hit", "mean")).reset_index()
    by_stat = R.groupby("lean_stat").agg(games=("lean_hit", "size"), model_prob=("lean_prob", "mean"),
                                         hit_rate=("lean_hit", "mean")).reset_index()
    seasons = R[R.lean_prob >= 0.6].groupby("season").agg(games=("lean_hit", "size"), hit_rate=("lean_hit", "mean")).reset_index()
    return by_strength, top_vs_rest, by_stat, seasons


def fit_pick_calibration(R: pd.DataFrame) -> dict:
    """Per-stat logistic: P(hit) ~ logit(lean_prob), fit on picked leans only.
    Fixes winner's-curse overconfidence from taking the max over stats."""
    cal = {}
    for s, d in R.groupby("lean_stat"):
        lr = LogisticRegression().fit(_logit(d.lean_prob.to_numpy())[:, None], d.lean_hit.astype(int))
        cal[s] = (float(lr.intercept_[0]), float(lr.coef_[0][0]))
    return cal


def apply_pick_calibration(out: pd.DataFrame, cal: dict) -> pd.DataFrame:
    out = out.copy()
    a = np.array([cal.get(s, (0.0, 1.0))[0] for s in out.lean_stat])
    b = np.array([cal.get(s, (0.0, 1.0))[1] for s in out.lean_stat])
    out["lean_prob_cal"] = 1 / (1 + np.exp(-(a + b * _logit(out.lean_prob.to_numpy()))))
    return out


def backtest_calibrated():
    """Leave-one-season-out: pick calibration fit on other seasons' picked leans."""
    P = baselines(pd.read_parquet(PROC / "projections_all_games_oos.parquet"))
    R = pd.concat([pick(probabilities(P[P.season == y], fit=P, exclude_season=y)) for y in sorted(P.season.unique())],
                  ignore_index=True)
    R = R.dropna(subset=["lean_prob"]).reset_index(drop=True)  # no baseline (rookies) -> no lean
    act = np.array([R[s].to_numpy() for s in STATS]).T[np.arange(len(R)), [STATS.index(s) for s in R.lean_stat]]
    R["lean_hit"] = np.where(R.lean_side == "over", act > R.lean_ref, act < R.lean_ref)
    R = pd.concat([apply_pick_calibration(R[R.season == y], fit_pick_calibration(R[R.season != y]))
                   for y in sorted(R.season.unique())], ignore_index=True)
    R[["season", "pid", "gameId", "lean_stat", "lean_side", "lean_prob", "lean_hit", "lean_prob_cal"]].to_parquet(
        PROC / "lean_backtest.parquet", index=False)
    b = pd.cut(R.lean_prob_cal, [0.4, 0.55, 0.6, 0.65, 0.7, 1.0])
    by_strength = R.groupby(b, observed=True).agg(games=("lean_hit", "size"), stated=("lean_prob_cal", "mean"),
                                                  hit_rate=("lean_hit", "mean")).reset_index()
    by_stat = R.groupby("lean_stat").agg(games=("lean_hit", "size"), stated=("lean_prob_cal", "mean"),
                                         hit_rate=("lean_hit", "mean")).reset_index()
    return R, by_strength, by_stat
