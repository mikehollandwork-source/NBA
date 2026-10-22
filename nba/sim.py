"""Live outcome simulation (ranges + Kalshi rung probabilities). Leans don't use this.

Backtested in nba/analysis/sim_improve.py (reports/phase2/sim_improvements.md):
  - minutes first: each sim draws tonight's minutes from the real spread of
    (actual minutes / projected minutes) for similar projections, then stats given minutes;
  - player spread: each stat's game-to-game spread scaled by the player's own past misses
    (last 250 games), shrunk toward the league curve (weight of 100 games);
  - 20,000 sims (probability noise ~0.3 pts).
Player-specific PTS/REB/AST links were tested and dropped (no gain beyond noise).
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from nba.analysis import sim_improve as si
from nba.analysis.projection import fit_variance
from nba.config import PROC

N_SIMS = 20000
_CACHE: dict = {}  # pid -> (n_sims, 3) sims from the latest simulate() call


@lru_cache(maxsize=1)
def params():
    """Fit on every out-of-sample season: minutes pools, conditional variance + correlation,
    league variance curves and each player's spread multiplier (from his last 250 games)."""
    P = si.load()
    F = P.copy()
    for s in si.COMP:
        F[f"{s}_mu"] = F[f"{s}_ensemble"].clip(lower=0.05)
    prm = si.fit_variant(F, None, "minutes")
    mult = {}
    last = P.groupby("pid").tail(si.SPREAD_WINDOW)
    for s in si.COMP:
        k, p = fit_variance(P[s].to_numpy(), P[f"{s}_ensemble"].clip(lower=0.05).to_numpy())
        mu = last[f"{s}_ensemble"].clip(lower=0.05)
        z2 = (last[s] - mu) ** 2 / (k * mu ** p)
        g = z2.groupby(last.pid).agg(["sum", "count"])
        mult[s] = ((g["sum"] + si.SPREAD_N0) / (g["count"] + si.SPREAD_N0)).rename(s)
    prm["mult"] = pd.concat(mult.values(), axis=1)
    prm["mult"].index = prm["mult"].index.astype(str)
    return prm


def simulate(r: pd.DataFrame, n_sims: int = N_SIMS, chunk: int = 100) -> np.ndarray:
    """r needs pid, proj_min, proj_pts/reb/ast. Returns (n_sims, len(r), 3) int32; also caches by pid."""
    prm = params()
    D = pd.DataFrame({"proj_min": r.proj_min.clip(lower=1).to_numpy()})
    for s in si.COMP:
        D[f"{s}_mu"] = np.clip(r[f"proj_{s}"].to_numpy(dtype=float), 0.05, None)
    mm = prm["mult"].reindex(r.pid.astype(str)).fillna(1.0)[si.COMP].to_numpy()
    L = np.linalg.cholesky(prm["C"])
    out = np.empty((n_sims, len(r), 3), dtype=np.int32)
    for a in range(0, len(r), chunk):
        d = D.iloc[a:a + chunk]
        mu, var = si.minutes_means(d, prm, n_sims)
        out[:, a:a + chunk] = si.draw_general(mu, var * mm[None, a:a + chunk], L, n_sims)
    for j, pid in enumerate(r.pid.astype(str)):
        _CACHE[pid] = out[:, j]
    return out


def cached(pid: str):
    return _CACHE.get(str(pid))
