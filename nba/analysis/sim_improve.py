"""Improving the outcome distributions (ranges, rung probabilities, leans).

Four upgrades, each tested out-of-sample against the current setup (2026-10-22):
  1. player_spread  : per-player variance multiplier from his own past misses, shrunk
                      toward the league curve (prior games only, curves fit on other seasons)
  2. minutes_first  : simulate minutes first (empirical early-exit / blowout tail), then stats
  3. player_corr    : per-player PTS/REB/AST correlation, shrunk toward the league
  4. n_sims         : Monte Carlo noise vs number of simulations
Run: python -m nba.analysis.sim_improve [spread|minutes|corr|sims|leans]
Report: reports/phase2/sim_improvements.md
"""
import sys

import numpy as np
import pandas as pd
import scipy.stats as st

from nba.analysis.projection import fit_variance
from nba.config import PROC

STATS = ["pts", "reb", "ast", "pra", "pr", "pa", "ra"]
COMP = ["pts", "reb", "ast"]
RUNGS = {"pts": [10, 15, 20, 25, 30, 35], "reb": [4, 6, 8, 10, 12], "ast": [2, 4, 6, 8, 10],
         "pra": [15, 20, 25, 30, 35, 40, 45, 50], "pr": [15, 20, 25, 30, 35, 40], "pa": [15, 20, 25, 30, 35, 40],
         "ra": [6, 8, 10, 12, 15, 18]}


def load() -> pd.DataFrame:
    P = pd.read_parquet(PROC / "projections_all_games_oos.parquet")
    P["game_date"] = pd.to_datetime(P.game_date)
    return P.sort_values(["pid", "game_date"]).reset_index(drop=True)


def nb_from_var(mu, var):
    mu = np.clip(mu, 0.05, None)
    alpha = np.clip((var - mu) / mu ** 2, 1e-4, None)
    n = np.minimum(1 / alpha, 1e4)
    return st.nbinom(n, n / (n + mu))


# ---------------------------------------------------------------- 1. player spread
def spread_history(P: pd.DataFrame, season: int, window: int) -> pd.DataFrame:
    """For rows of `season`: base variance (curve fit on other seasons) and the player's
    prior sum/count of z2 = miss^2 / base variance over his last `window` games (any season)."""
    out = pd.DataFrame(index=P.index[P.season == season])
    fit = P[P.season != season]
    for s in STATS:
        k, p = fit_variance(fit[s].to_numpy(), fit[f"{s}_ensemble"].to_numpy())
        mu = P[f"{s}_ensemble"].clip(lower=0.05)
        v0 = k * mu ** p
        z2 = (P[s] - mu) ** 2 / v0
        cs = z2.groupby(P.pid).cumsum()
        cn = P.groupby("pid").cumcount() + 1
        prev_s = cs - z2                      # sum of all prior games
        lag = cs.groupby(P.pid).shift(window + 1).fillna(0)  # sum up to game i-window-1
        lag_n = (cn - 1 - window).clip(lower=0)
        out[f"{s}_v0"] = v0[out.index]
        out[f"{s}_zs"] = (prev_s - lag)[out.index]
        out[f"{s}_zn"] = ((cn - 1) - lag_n)[out.index]
    return out


def multiplier(h: pd.DataFrame, s: str, n0: float) -> np.ndarray:
    return ((h[f"{s}_zs"] + n0) / (h[f"{s}_zn"] + n0)).to_numpy()


def score(y, mu, var, s, full=True):
    """NB log-lik (+ 80% coverage and rung Brier if full) for one stat."""
    d = nb_from_var(mu, var)
    ll = d.logpmf(y)
    if not full:
        return ll, None, None
    cov = (d.cdf(y) >= 0.1) & (d.cdf(y - 1) < 0.9)   # y inside [10th, 90th] percentile
    br = np.mean([((1 - d.cdf(k - 1)) - (y >= k)) ** 2 for k in RUNGS[s]], axis=0)
    return ll, cov, br


def run_spread(seasons=range(2020, 2026)):
    P = load()
    rows, keep = [], {}
    for window in (40, 100, 250):
        H = {y: spread_history(P, y, window) for y in seasons}
        for n0 in (10, 25, 50, 100, 200):
            for y in seasons:
                h = H[y]
                T = P.loc[h.index]
                for s in STATS:
                    y_, mu = T[s].to_numpy(), T[f"{s}_ensemble"].to_numpy()
                    v0 = h[f"{s}_v0"].to_numpy()
                    m = multiplier(h, s, n0)
                    for name, var in (("league", v0), ("player", v0 * m)):
                        ll, _, _ = score(y_, mu, var, s, full=False)
                        rows.append({"window": window, "n0": n0, "season": y, "stat": s, "model": name,
                                     "ll": ll.mean(), "n": len(T)})
            print("done window", window, "n0", n0, flush=True)
    R = pd.DataFrame(rows)
    R.to_parquet(PROC / "sim_improve_spread.parquet", index=False)
    return R


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "spread"
    if what == "spread":
        run_spread()


# ---------------------------------------------------------------- simulation harness (2, 3, 4)
from scipy import special  # noqa: E402

RNG = np.random.default_rng(11)


def nb_n(mu, var):
    mu = np.clip(mu, 0.05, None)
    alpha = np.clip((var - mu) / mu ** 2, 1e-4, None)
    return np.minimum(1 / alpha, 1e4)


def draw_general(mu, var, L, n_sims):
    """mu, var: (n_sims, M, 3) or (M, 3); L: (3,3) or (M,3,3) Cholesky of latent corr.
    Gamma-Poisson with Gaussian copula on the gamma draws -> (n_sims, M, 3) ints."""
    M = mu.shape[-2]
    Z = RNG.standard_normal((n_sims, M, 3))
    Z = Z @ L.T if L.ndim == 2 else np.einsum("smj,mkj->smk", Z, L)
    U = np.clip(special.ndtr(Z), 1e-9, 1 - 1e-9)
    mu = np.broadcast_to(mu, (n_sims, M, 3))
    var = np.broadcast_to(var, (n_sims, M, 3))
    n = nb_n(mu, var)
    G = special.gammaincinv(n, U) / n
    return RNG.poisson(np.clip(mu, 0.05, None) * G).astype(np.int32)


def crps(x, y):
    xs = np.sort(x, axis=0).astype(float)
    n = xs.shape[0]
    a = np.abs(xs - y[None, :]).mean(0)
    i = np.arange(1, n + 1)[:, None]
    return a - ((2 * i - n - 1) * xs).sum(0) / (n * n)


def sim_metrics(sims, T):
    """sims (n_sims, M, 3) -> per-row metrics for pts/reb/ast/pra/pr/pa/ra."""
    out = {}
    parts = {"pts": [0], "reb": [1], "ast": [2], "pra": [0, 1, 2], "pr": [0, 1], "pa": [0, 2], "ra": [1, 2]}
    for s, ix in parts.items():
        x = sims[:, :, ix].sum(2)
        y = T[s].to_numpy()
        lo, hi = np.percentile(x, [10, 90], axis=0)
        out[f"{s}_crps"] = crps(x, y)
        out[f"{s}_cov"] = (y >= lo) & (y <= hi)
        out[f"{s}_brier"] = np.mean([((x >= k).mean(0) - (y >= k)) ** 2 for k in RUNGS[s]], axis=0)
    return pd.DataFrame(out, index=T.index)


def latent_corr(T, var_fn):
    """Latent PTS/REB/AST correlation from randomized PIT of the marginal NBs."""
    u = RNG.uniform(size=len(T))
    z = []
    for j, s in enumerate(COMP):
        mu = T[f"{s}_mu"].to_numpy()
        d = nb_from_var(mu, var_fn(T, s))
        y = T[s].to_numpy()
        z.append(special.ndtri(np.clip(d.cdf(y - 1) + u * d.pmf(y), 1e-6, 1 - 1e-6)))
    return np.corrcoef(np.vstack(z)), np.vstack(z).T


def scale_corr(C0, T, var_fn, mu_fn=None, n=60):
    """Scale off-diagonals so simulated PRA miss variance matches reality (as season_sim.fit_params)."""
    sub = T.sample(min(len(T), 20000), random_state=1)
    mu = sub[[f"{s}_mu" for s in COMP]].to_numpy()
    target = ((sub[COMP].sum(1).to_numpy() - mu.sum(1)) ** 2).mean()
    var = np.column_stack([var_fn(sub, s) for s in COMP])
    best, C = np.inf, C0
    for c in (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5):
        Cc = np.eye(3) + np.clip(c * (C0 - np.eye(3)), -0.95, 0.95)
        if np.linalg.eigvalsh(Cc).min() <= 1e-6:
            continue
        if mu_fn is None:
            x = draw_general(mu, var, np.linalg.cholesky(Cc), n)
        else:
            m2, v2 = mu_fn(sub, n)
            x = draw_general(m2, v2, np.linalg.cholesky(Cc), n)
        v = ((x.sum(2) - mu.sum(1)[None]) ** 2).mean()
        if abs(v - target) < best:
            best, C = abs(v - target), Cc
    return C


SPREAD_WINDOW, SPREAD_N0 = 250, 100   # chosen on 2020-22 log-likelihood


def spread_detail(seasons=range(2023, 2026)):
    """Coverage by swing group + rung Brier, league vs player spread (held-out seasons)."""
    P = load()
    rows = []
    for y in seasons:
        h = spread_history(P, y, SPREAD_WINDOW)
        T = P.loc[h.index]
        for s in STATS:
            yv, mu, v0 = T[s].to_numpy(), T[f"{s}_ensemble"].to_numpy(), h[f"{s}_v0"].to_numpy()
            m = multiplier(h, s, SPREAD_N0)
            for name, var in (("league", v0), ("player", v0 * m)):
                ll, cov, br = score(yv, mu, var, s)
                rows.append(pd.DataFrame({"season": y, "stat": s, "model": name, "ll": ll, "cov": cov,
                                          "brier": br, "mult": m, "pid": T.pid.to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def lean_variant(mult: bool, seasons=range(2019, 2026)):
    """Lean backtest (same as lean.backtest_calibrated) with optional player spread multiplier
    on each stat's projection distribution. Returns picked leans with pick-calibrated prob."""
    from nba.analysis import lean as L
    from nba.analysis.projection import dist  # noqa: F401
    P = L.baselines(load())
    if mult:
        H = pd.concat([spread_history(P, y, SPREAD_WINDOW) for y in seasons])
        for s in STATS:
            P.loc[H.index, f"{s}_vm"] = multiplier(H, s, SPREAD_N0)
    for s in STATS:
        if f"{s}_vm" not in P:
            P[f"{s}_vm"] = 1.0
    out = []
    for y in seasons:
        fit, T = P[P.season != y], P[P.season == y].copy()
        for s in STATS:
            f = fit.dropna(subset=[s, f"{s}_ensemble", f"{s}_base"])
            k, pw = fit_variance(f[s].to_numpy(), f[f"{s}_ensemble"].to_numpy())
            kb, pb = fit_variance(f[s].to_numpy(), f[f"{s}_base"].to_numpy())
            fl = L.reference_line(f[f"{s}_base"].to_numpy(), kb, pb)
            mu_f = f[f"{s}_ensemble"].clip(lower=0.05).to_numpy()
            fraw = 1 - nb_from_var(mu_f, f[f"{s}_vm"].to_numpy() * k * mu_f ** pw).cdf(np.floor(fl))
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression().fit(L._logit(fraw)[:, None], (f[s].to_numpy() > fl).astype(int))
            ok = T[f"{s}_base"].notna() & T[f"{s}_ensemble"].notna()
            line = L.reference_line(T.loc[ok, f"{s}_base"].to_numpy(), kb, pb)
            mu = T.loc[ok, f"{s}_ensemble"].clip(lower=0.05).to_numpy()
            raw = 1 - nb_from_var(mu, T.loc[ok, f"{s}_vm"].to_numpy() * k * mu ** pw).cdf(np.floor(line))
            T.loc[ok, f"{s}_ref"] = line
            T.loc[ok, f"{s}_p_over"] = lr.predict_proba(L._logit(raw)[:, None])[:, 1]
        out.append(L.pick(T))
    R = pd.concat(out, ignore_index=True).dropna(subset=["lean_prob"]).reset_index(drop=True)
    act = np.array([R[s].to_numpy() for s in L.STATS]).T[np.arange(len(R)), [L.STATS.index(s) for s in R.lean_stat]]
    R["lean_hit"] = np.where(R.lean_side == "over", act > R.lean_ref, act < R.lean_ref)
    R = pd.concat([L.apply_pick_calibration(R[R.season == y], L.fit_pick_calibration(R[R.season != y]))
                   for y in sorted(R.season.unique())], ignore_index=True)
    return R


def lean_summary(R: pd.DataFrame, elig: pd.DataFrame) -> dict:
    """Board-eligible leans (swing <= 0.45, >= 10 proj min): 70%+ hit, top 5/night."""
    x = R.merge(elig, on=["season", "pid", "gameId"])
    x = x[(x.cv <= 0.45) & (x.proj_min >= 10)]
    x["date"] = pd.to_datetime(x.game_date).dt.date
    s70 = x[x.lean_prob_cal.round(2) >= 0.70]
    t5 = x.sort_values("lean_prob_cal", ascending=False).groupby("date").head(5)
    res = {"all": x.lean_hit.mean(), "n70": len(s70), "hit70": s70.lean_hit.mean(), "stated70": s70.lean_prob_cal.mean(),
           "top5": t5.lean_hit.mean()}
    for y in (2023, 2024, 2025):
        res[f"top5_{y}"] = t5[t5.season == y].lean_hit.mean()
        res[f"hit70_{y}"] = s70[s70.season == y].lean_hit.mean()
    return res


# ---------------------------------------------------------------- 2/3: simulation variants
def _prep(P, y, n_rows, seed=5):
    """Held-out rows of season y (sample) + training rows (other seasons), with PIT inputs."""
    T = P[P.season == y]
    T = T.sample(min(n_rows, len(T)), random_state=seed).copy()
    F = P[(P.season != y) & (P.season < 2026)].copy()
    for D in (T, F):
        for s in COMP:
            D[f"{s}_mu"] = D[f"{s}_ensemble"].clip(lower=0.05)
    return T, F


def fit_variant(F, h_all, variant):
    """Fit everything a variant needs on training rows F. Returns dict of params."""
    prm = {"variant": variant}
    prm["var"] = {s: fit_variance(F[s].to_numpy(), F[f"{s}_mu"].to_numpy()) for s in COMP}
    if "minutes" in variant:
        ratio = (F["min"] / F.proj_min.clip(lower=1)).to_numpy()
        edges = np.quantile(F.proj_min, [0, .2, .4, .6, .8, 1])
        b = np.clip(np.searchsorted(edges, F.proj_min, side="right") - 1, 0, 4)
        prm["edges"], prm["pools"] = edges, [ratio[b == i] for i in range(5)]
        sc = (F["min"] / F.proj_min.clip(lower=1)).to_numpy()
        prm["cvar"] = {s: fit_variance(F[s].to_numpy(), (F[f"{s}_mu"] * sc).clip(lower=0.05).to_numpy()) for s in COMP}
    # latent correlation (conditional on minutes for minutes variants)
    if "minutes" in variant:
        sc = (F["min"] / F.proj_min.clip(lower=1)).to_numpy()
        G = F.copy()
        for s in COMP:
            G[f"{s}_mu"] = (F[f"{s}_mu"] * sc).clip(lower=0.05)
        vf = lambda D, s: prm["cvar"][s][0] * D[f"{s}_mu"].to_numpy() ** prm["cvar"][s][1]  # noqa: E731
        C0, _ = latent_corr(G.sample(min(len(G), 60000), random_state=2), vf)
        prm["C"] = scale_corr(C0, F, None or (lambda D, s: prm["var"][s][0] * D[f"{s}_mu"].to_numpy() ** prm["var"][s][1]),
                              mu_fn=lambda D, n: minutes_means(D, prm, n))
    else:
        vf = lambda D, s: prm["var"][s][0] * D[f"{s}_mu"].to_numpy() ** prm["var"][s][1]  # noqa: E731
        C0, _ = latent_corr(F.sample(min(len(F), 60000), random_state=2), vf)
        prm["C"] = scale_corr(C0, F, vf)
    prm["C0"] = C0
    return prm


def minutes_means(D, prm, n):
    """Per-sim means/variances when minutes are drawn first: (n, M, 3) each."""
    b = np.clip(np.searchsorted(prm["edges"], D.proj_min.to_numpy(), side="right") - 1, 0, 4)
    r = np.empty((n, len(D)))
    for i in range(5):
        cols = np.where(b == i)[0]
        if len(cols):
            pool = prm["pools"][i]
            r[:, cols] = pool[RNG.integers(len(pool), size=(n, len(cols)))]
    mu = D[[f"{s}_mu" for s in COMP]].to_numpy()[None] * r[:, :, None]
    mu = np.clip(mu, 0.05, None)
    var = np.stack([prm["cvar"][s][0] * mu[:, :, j] ** prm["cvar"][s][1] for j, s in enumerate(COMP)], 2)
    return mu, var


def player_corr(P, y, C_league0, c_scale, window=250, n0=100):
    """Per-row latent corr from the player's prior PIT z-scores, shrunk to the league; then
    off-diagonals scaled like the league matrix. Returns (len(rows of y), 3, 3)."""
    F = P[P.season != y]
    var = {s: fit_variance(F[s].to_numpy(), F[f"{s}_ensemble"].clip(lower=0.05).to_numpy()) for s in COMP}
    u = np.random.default_rng(3).uniform(size=len(P))
    Z = {}
    for s in COMP:
        mu = P[f"{s}_ensemble"].clip(lower=0.05).to_numpy()
        d = nb_from_var(mu, var[s][0] * mu ** var[s][1])
        yv = P[s].to_numpy()
        Z[s] = special.ndtri(np.clip(d.cdf(yv - 1) + u * d.pmf(yv), 1e-6, 1 - 1e-6))
    def prior(x):
        x = pd.Series(x, index=P.index)
        cs = x.groupby(P.pid).cumsum()
        return ((cs - x) - cs.groupby(P.pid).shift(window + 1).fillna(0)).to_numpy()
    cn = P.groupby("pid").cumcount().to_numpy()
    n = np.minimum(cn, window)
    sxx = {s: prior(Z[s] ** 2) for s in COMP}
    pairs = [(0, 1), (0, 2), (1, 2)]
    m = (P.season == y).to_numpy()
    Cs = np.repeat(np.eye(3)[None], m.sum(), 0)
    for i, j in pairs:
        a, b = COMP[i], COMP[j]
        sxy = prior(Z[a] * Z[b])
        r = np.where(n > 2, sxy / np.sqrt(np.maximum(sxx[a] * sxx[b], 1e-9)), 0)
        rho = (n * r + n0 * C_league0[i, j]) / (n + n0)
        rho = np.clip(c_scale * rho, -0.95, 0.95)[m]
        Cs[:, i, j] = Cs[:, j, i] = rho
    bad = np.linalg.eigvalsh(Cs).min(1) <= 1e-6
    Cl = np.eye(3) + np.clip(c_scale * (C_league0 - np.eye(3)), -0.95, 0.95)
    Cs[bad] = Cl
    return pd.DataFrame({"idx": P.index[m]}), Cs


def run_sims(seasons=(2023, 2024, 2025), n_rows=12000, n_sims=1000, chunk=3000):
    from nba.config import PROC as _P
    P = load()
    out = []
    for y in seasons:
        T, F = _prep(P, y, n_rows)
        h = spread_history(P, y, SPREAD_WINDOW)
        mult = {s: pd.Series(multiplier(h, s, SPREAD_N0), index=h.index) for s in COMP}
        base = fit_variant(F, None, "league")
        mins = fit_variant(F, None, "minutes")
        c_scale = (base["C"][0, 1] / base["C0"][0, 1])
        idx, Cs = player_corr(P, y, base["C0"], c_scale)
        Cmap = pd.Series(range(len(idx)), index=idx.idx)
        for k0 in range(0, len(T), chunk):
            D = T.iloc[k0:k0 + chunk]
            mu = D[[f"{s}_mu" for s in COMP]].to_numpy()
            v0 = np.column_stack([base["var"][s][0] * mu[:, j] ** base["var"][s][1] for j, s in enumerate(COMP)])
            mm = np.column_stack([mult[s][D.index].to_numpy() for s in COMP])
            Lb = np.linalg.cholesky(base["C"])
            Lp = np.linalg.cholesky(Cs[Cmap[D.index].to_numpy()])
            m2, v2 = minutes_means(D, mins, n_sims)
            vm = {"current": (mu, v0, Lb),
                  "1 player spread": (mu, v0 * mm, Lb),
                  "2 minutes first": (m2, v2, np.linalg.cholesky(mins["C"])),
                  "3 player links": (mu, v0, Lp),
                  "1+2": (m2, v2 * mm[None], np.linalg.cholesky(mins["C"])),
                  "1+3": (mu, v0 * mm, Lp)}
            for name, (a, b, L) in vm.items():
                met = sim_metrics(draw_general(a, b, L, n_sims), D)
                met["variant"], met["season"] = name, y
                out.append(met)
            print(y, k0, flush=True)
    R = pd.concat(out)
    R.to_parquet(_P / "sim_improve_sims.parquet")
    return R
