"""Test A: which stats are most consistent / predictable game-to-game.

Every pregame number uses only games BEFORE the one being predicted.
"""
import numpy as np
import pandas as pd

from nba.analysis.common import TARGETS, load_games

WARMUP = 10  # need 10 prior games before scoring a pregame prediction


def _spearman_brown(r):
    return 2 * r / (1 + r)


def stat_table(t: pd.DataFrame) -> pd.DataFrame:
    g = t.groupby("nba_id")
    t = t.assign(k=g.cumcount())
    rows = []
    for s in TARGETS:
        x = t[s].astype(float)
        pm = g[s].transform("mean")
        dev = x - pm
        # season-to-date and last-10 means using only prior games
        prior = g[s].transform(lambda v: v.shift().expanding().mean())
        l10 = g[s].transform(lambda v: v.shift().rolling(10, min_periods=5).mean())
        ev = t.k >= WARMUP
        a, p, p10, orc = x[ev], prior[ev], l10[ev], pm[ev]
        # per-player game-to-game CV
        ps = g[s].agg(["mean", "std"])
        cv = (ps["std"] / ps["mean"]).replace([np.inf], np.nan)
        # split-half reliability across players (odd vs even games)
        oe = t.assign(h=t.k % 2).groupby(["nba_id", "h"])[s].mean().unstack()
        rel = _spearman_brown(oe[0].corr(oe[1]))
        # share of within-player variance explained by that night's minutes
        dm = t["min"] - g["min"].transform("mean")
        r2_min = np.corrcoef(dev, dm)[0, 1] ** 2 if s != "min" else np.nan
        # lag-1 autocorrelation of within-player deviations (streakiness)
        lag = dev.groupby(t.nba_id).shift()
        ac1 = dev.corr(lag)
        # dispersion: variance / mean per player (1 = Poisson)
        disp = (ps["std"] ** 2 / ps["mean"]).replace([np.inf], np.nan).median()
        rows.append({
            "stat": s, "avg": round(x.mean(), 2),
            "median_player_cv": round(cv.median(), 3),
            "pregame_corr_season_avg": round(a.corr(p), 3),
            "pregame_corr_last10": round(a.corr(p10), 3),
            "pregame_nmae_season_avg": round((a - p).abs().mean() / a.mean(), 3),
            "oracle_nmae_full_season_avg": round((a - orc).abs().mean() / a.mean(), 3),
            "within_player_var_share": round(dev.var() / x.var(), 3),
            "split_half_reliability": round(rel, 3),
            "within_var_explained_by_minutes": None if np.isnan(r2_min) else round(r2_min, 3),
            "lag1_autocorr": round(ac1, 3),
            "dispersion_var_over_mean": round(disp, 2),
        })
    out = pd.DataFrame(rows)
    out["predictability_rank"] = out.pregame_nmae_season_avg.rank().astype(int)
    return out.sort_values("pregame_nmae_season_avg")


def per36_table(t: pd.DataFrame) -> pd.DataFrame:
    """Same consistency view on per-36-minute rates (strips out minutes noise)."""
    rows = []
    t = t[t["min"] >= 10].copy()
    g = t.groupby("nba_id")
    for s in [x for x in TARGETS if x != "min"]:
        r = t[s] / t["min"] * 36
        cvr = r.groupby(t.nba_id).std() / r.groupby(t.nba_id).mean()
        cvraw = g[s].std() / g[s].mean()
        rows.append({"stat": s, "median_cv_raw": round(cvraw.median(), 3), "median_cv_per36": round(cvr.median(), 3)})
    return pd.DataFrame(rows)


def player_table(t: pd.DataFrame) -> pd.DataFrame:
    """Per-player CV for each stat (lower = more consistent)."""
    g = t.groupby("nba_id")
    info = g.agg(player=("player", "last"), team=("team", "last"), gp=("gameId", "size"), mpg=("min", "mean"))
    for s in TARGETS:
        info[f"{s}_avg"] = g[s].mean().round(2)
        info[f"{s}_cv"] = (g[s].std() / g[s].mean()).round(3)
    return info.reset_index()


def run():
    t = load_games()
    return t, stat_table(t), per36_table(t), player_table(t)


if __name__ == "__main__":
    t, a, b, c = run()
    pd.set_option("display.width", 250)
    print(t.nba_id.nunique(), "qualified players,", len(t), "player-games")
    print(a.to_string(index=False)); print(b.to_string(index=False))
