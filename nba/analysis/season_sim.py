"""Monte Carlo season simulation (NumPy) per player, over his real schedule.

For every team game a player could play (his team's schedule, following trades):
  mean per game (three versions, compared):
    base     : our out-of-sample pregame projection (ensemble). Games he missed get
               the projection carried from his nearest played game.
    bias     : base + his average miss vs our projection over ALL earlier games
               (this + last season, shrunk) — control for simdef
    simdef   : base + his history vs defenses similar to tonight's opponent
               (shrunk average miss vs projection in his 8 most similar earlier matchups)
    h2h      : his average vs that exact team in earlier meetings (this + last season);
               falls back to base before the first meeting
    last_szn : last season's per-game average (simple reference, no model)
  game outcome: negative binomial (variance curve fit on OTHER seasons), PTS/REB/AST
               correlated within a game via a Gaussian copula -> PRA = sum.
  season-long shift: one random multiplier per player per simulated season
               (players beat / miss projections persistently); size fit on other seasons
  games played:
    actual    : the games he really played
    simulated : availability rate drawn from what happened to players with similar
               last-season availability (other seasons), then a two-state Markov chain
               for absence streaks
Everything is fit without the season being simulated.
"""
import numpy as np
import pandas as pd
from scipy import special, stats as st

from nba.analysis.common import load_games
from nba.analysis.defense_profiles import vectors
from nba.analysis.projection import fit_variance
from nba.config import PROC

COMP = ["pts", "reb", "ast"]
VARIANTS = ["base", "bias", "simdef", "h2h", "last_szn"]
N_SIMS = 1000
RNG = np.random.default_rng(11)
TOP_K, MIN_SIM, SHRINK = 8, 0.30, 2.0


# ---------------------------------------------------------------- targets
def build_targets(seasons=range(2020, 2026)) -> pd.DataFrame:
    allg = load_games(qualified_only=False)
    P = pd.read_parquet(PROC / "projections_all_games_oos.parquet")
    tg = allg.drop_duplicates(["gameId", "team"])[["season", "gameId", "team", "opp", "game_date"]]
    ps = allg.groupby(["season", "pid"]).agg(gp=("gameId", "size"), mpg=("min", "mean"), last_team=("team", "last"))
    team_games = tg.groupby(["season", "team"]).size()
    out = []
    for S in seasons:
        prev = ps.loc[S - 1]
        rot = prev[(prev.gp >= 20) & (prev.mpg >= 15)]
        cur = allg[(allg.season == S) & allg.pid.isin(rot.index)]
        played = cur[["pid", "gameId", "team", "game_date"]].sort_values("game_date")
        pairs = played[["pid", "team"]].drop_duplicates()
        cand = pairs.merge(tg[tg.season == S], on="team").sort_values("game_date")
        at = pd.merge_asof(cand, played.rename(columns={"team": "team_at"})[["pid", "game_date", "team_at"]],
                           on="game_date", by="pid", direction="backward")
        first = played.groupby("pid").team.first()
        at["team_at"] = at.team_at.fillna(at.pid.map(first))
        x = at[at.team == at.team_at].drop(columns="team_at").copy()
        x["played"] = x.set_index(["pid", "gameId"]).index.isin(played.set_index(["pid", "gameId"]).index)
        prv = rot.reindex(x.pid.unique())
        x["prev_avail"] = x.pid.map(prv.gp / team_games.loc[S - 1].reindex(prv.last_team).to_numpy())
        out.append(x)
    T = pd.concat(out, ignore_index=True)
    keep = ["season", "gameId", "pid", "player", "gp"] + COMP + ["pra"] + [f"{s}_ensemble" for s in COMP + ["pra"]] \
        + [f"{s}_szn" for s in COMP + ["pra"]]
    T = T.merge(P[keep], on=["season", "gameId", "pid"], how="left")
    T = T.sort_values(["season", "pid", "game_date", "gameId"]).reset_index(drop=True)
    g = T.groupby(["season", "pid"])
    for c in [f"{s}_ensemble" for s in COMP + ["pra"]] + ["player"]:
        T[c] = g[c].transform(lambda v: v.ffill().bfill())
    for s in COMP + ["pra"]:
        T[s] = T[s].where(T.played, np.nan)
    # last-season per-game average (reference)
    last = allg.groupby(["season", "pid"])[COMP].mean().reset_index()
    last["season"] += 1
    T = T.merge(last.rename(columns={s: f"{s}_last" for s in COMP}), on=["season", "pid"], how="left")
    return T, P


# ---------------------------------------------------------------- adjustments
def add_variants(T: pd.DataFrame, P: pd.DataFrame) -> pd.DataFrame:
    prof = pd.read_parquet(PROC / "defense_profiles.parquet")
    V = vectors(prof)
    V = V / np.linalg.norm(V, axis=1, keepdims=True).clip(1e-9)
    key = pd.Series(np.arange(len(prof)), index=pd.MultiIndex.from_arrays([prof.gameId, prof.team]))
    vi = lambda g, o: np.array([key.get((a, b), -1) for a, b in zip(g, o)])
    H = P[P.pid.isin(T.pid.unique())].copy()  # played history with projections
    H["vi"] = vi(H.gameId, H.opp)
    T["vi"] = vi(T.gameId, T.opp)
    for s in COMP:
        H[f"res_{s}"] = H[s] - H[f"{s}_ensemble"]
        T[f"sig_{s}"] = 0.0
        T[f"gen_{s}"] = 0.0
        T[f"h2h_{s}"] = np.nan
    Hg = {k: d for k, d in H.groupby("pid")}
    for pid, idx in T.groupby("pid").indices.items():
        h = Hg.get(pid)
        if h is None:
            continue
        t = T.iloc[idx]
        hd, td = h.game_date.to_numpy(), t.game_date.to_numpy()
        valid = (hd[None, :] < td[:, None]) & (h.season.to_numpy()[None, :] >= t.season.to_numpy()[:, None] - 1)
        hv, tv = h.vi.to_numpy(), t.vi.to_numpy()
        S = V[tv] @ V[hv].T
        S[(tv < 0)[:, None] | (hv < 0)[None, :]] = -9
        S[~valid] = -9
        same = (t.opp.to_numpy()[:, None] == h.opp.to_numpy()[None, :]) & valid
        for s in COMP:
            res, act = h[f"res_{s}"].to_numpy(), h[s].to_numpy()
            n = same.sum(1)
            T.loc[t.index, f"h2h_{s}"] = np.where(n > 0, (same * act).sum(1) / np.maximum(n, 1), np.nan)
        nv = valid.sum(1)
        for s in COMP:
            T.loc[t.index, f"gen_{s}"] = (valid * h[f"res_{s}"].to_numpy()[None, :]).sum(1) / (nv + 5.0)
        order = np.argsort(-S, axis=1)[:, :TOP_K]
        w = np.take_along_axis(S, order, 1)
        w = np.where(w >= MIN_SIM, w, 0)
        for s in COMP:
            r = h[f"res_{s}"].to_numpy()[order]
            T.loc[t.index, f"sig_{s}"] = (w * r).sum(1) / (w.sum(1) + SHRINK)
    for s in COMP:
        T[f"{s}_mu_base"] = T[f"{s}_ensemble"]
        T[f"{s}_mu_bias"] = T[f"{s}_ensemble"] + T[f"gen_{s}"]
        T[f"{s}_mu_simdef"] = T[f"{s}_ensemble"] + T[f"sig_{s}"]
        T[f"{s}_mu_h2h"] = T[f"h2h_{s}"].fillna(T[f"{s}_ensemble"])
        T[f"{s}_mu_last_szn"] = T[f"{s}_last"].fillna(T[f"{s}_ensemble"])
        for v in VARIANTS:
            T[f"{s}_mu_{v}"] = T[f"{s}_mu_{v}"].clip(lower=0.05)
    return T


# ---------------------------------------------------------------- distributions
def fit_params(P: pd.DataFrame, exclude_season: int):
    """Variance curves + within-game PTS/REB/AST correlation, fit on other seasons."""
    o = P[(P.season != exclude_season)].dropna(subset=COMP + [f"{s}_ensemble" for s in COMP])
    var = {s: fit_variance(o[s].to_numpy(), o[f"{s}_ensemble"].to_numpy()) for s in COMP}
    z = []
    u = RNG.uniform(size=len(o))
    for s in COMP:
        d = nb(o[f"{s}_ensemble"].to_numpy(), *var[s])
        y = o[s].to_numpy()
        pit = np.clip(d.cdf(y - 1) + u * d.pmf(y), 1e-6, 1 - 1e-6)
        z.append(special.ndtri(pit))
    C0 = np.corrcoef(np.vstack(z))
    # Poisson noise dilutes the latent correlation; scale it so simulated PRA
    # variance matches the real PRA miss variance (fit on these other seasons).
    sub = o.sample(min(len(o), 30000), random_state=1)
    mu = sub[[f"{s}_ensemble" for s in COMP]].to_numpy()
    target = ((sub[COMP].sum(1) - mu.sum(1)) ** 2).mean()
    best, C = np.inf, C0
    for c in (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5):
        Cc = np.eye(3) + np.clip(c * (C0 - np.eye(3)), -0.95, 0.95)
        if np.linalg.eigvalsh(Cc).min() <= 1e-6:
            continue
        v = draw(mu, var, Cc, n_sims=60).sum(2).var(0).mean()
        if abs(v - target) < best:
            best, C = abs(v - target), Cc
    return var, C


def _n_alpha(mu, k, p):
    mu = np.clip(mu, 0.05, None)
    alpha = np.clip((k * mu ** p - mu) / mu ** 2, 1e-4, None)
    return np.minimum(1 / alpha, 1e4)


def nb(mu, k, p):
    n = _n_alpha(mu, k, p)
    mu = np.clip(mu, 0.05, None)
    return st.nbinom(n, n / (n + mu))


def draw(mu: np.ndarray, var: dict, C: np.ndarray, n_sims=N_SIMS) -> np.ndarray:
    """mu: (M, 3) means for PTS/REB/AST -> sims (n_sims, M, 3) integer stat lines.
    Gamma-Poisson (= negative binomial) with a Gaussian copula on the gamma draws."""
    M = mu.shape[0]
    L = np.linalg.cholesky(C)
    Z = RNG.standard_normal((n_sims, M, 3)) @ L.T
    U = special.ndtr(Z)
    out = np.empty((n_sims, M, 3), dtype=np.int32)
    for j, s in enumerate(COMP):
        n = _n_alpha(mu[:, j], *var[s])
        G = special.gammaincinv(n[None, :], U[:, :, j]) / n[None, :]
        out[:, :, j] = RNG.poisson(mu[None, :, j] * G)
    return out


def season_shift_sd(P: pd.DataFrame, exclude_season: int) -> dict:
    """SD of a player's season-long relative miss vs projection, beyond game noise
    (other seasons). E.g. 0.08 = players typically run +/-8% vs projection all season."""
    o = P[P.season != exclude_season].dropna(subset=COMP + [f"{s}_ensemble" for s in COMP])
    out = {}
    for s in COMP:
        g = o.groupby(["season", "pid"])
        mu, y = g[f"{s}_ensemble"].sum(), g[s].sum()
        n = g.size()
        keep = n >= 20
        rel = (y - mu)[keep] / mu[keep]
        k, p = fit_variance(o[s].to_numpy(), o[f"{s}_ensemble"].to_numpy())
        noise = (o.assign(v=k * o[f"{s}_ensemble"].clip(lower=0.05) ** p).groupby(["season", "pid"]).v.sum()[keep]
                 / mu[keep] ** 2)
        out[s] = float(np.sqrt(max(rel.var() - noise.mean(), 0)))
    return out


def draw_shifted(mu, shift, var, C, n_sims=N_SIMS):
    """Like draw(), but each simulated season scales the player's means by a
    season-long shift (n_sims, 3)."""
    M = mu.shape[0]
    L = np.linalg.cholesky(C)
    Z = RNG.standard_normal((n_sims, M, 3)) @ L.T
    U = special.ndtr(Z)
    out = np.empty((n_sims, M, 3), dtype=np.int32)
    for j, s in enumerate(COMP):
        m = mu[None, :, j] * shift[:, j:j + 1]
        n = _n_alpha(m, *var[s])
        G = special.gammaincinv(n, U[:, :, j]) / n
        out[:, :, j] = RNG.poisson(m * G)
    return out


def availability_pool(T: pd.DataFrame, exclude_season: int) -> pd.DataFrame:
    """(last-season availability, this-season availability) pairs from other seasons."""
    o = T[T.season != exclude_season]
    return o.groupby(["season", "pid"]).agg(prev=("prev_avail", "first"), cur=("played", "mean")).dropna()


def draw_availability_rate(prev: float, pool: pd.DataFrame, n_sims=N_SIMS) -> np.ndarray:
    """Empirical: this-season availability of the 150 pool players whose
    last-season availability was closest to this player's."""
    if not np.isfinite(prev):
        return RNG.choice(pool.cur.to_numpy(), n_sims)
    near = pool.iloc[np.argsort(np.abs(pool.prev.to_numpy() - prev))[:150]]
    return RNG.choice(near.cur.to_numpy(), n_sims)


def availability_params(T: pd.DataFrame, exclude_season: int):
    """League persistence of absences (P(out next | out now)) and mean availability."""
    o = T[T.season < exclude_season] if (T.season < exclude_season).any() else T[T.season != exclude_season]
    out_now = ~o.played.to_numpy()
    same = (o.pid.to_numpy()[1:] == o.pid.to_numpy()[:-1]) & (o.season.to_numpy()[1:] == o.season.to_numpy()[:-1])
    q_mm = (out_now[1:] & out_now[:-1] & same).sum() / (out_now[:-1] & same).sum()
    p_league = o.groupby(["season", "pid"]).played.mean().mean()
    return q_mm, p_league


def simulate_availability(n_games: int, p_play, q_mm: float, n_sims=N_SIMS) -> np.ndarray:
    """p_play: scalar or (n_sims,) availability rates; Markov chain keeps absences streaky."""
    p_play = np.clip(np.broadcast_to(np.asarray(p_play, float), (n_sims,)), 0.02, 0.995)
    q_mm = min(q_mm, 0.995)
    q_eff = np.minimum(q_mm, 1 - 1e-3)
    p_out_after_play = np.clip((1 - p_play) * (1 - q_eff) / p_play, 0, 1)
    s = np.empty((n_sims, n_games), dtype=bool)
    s[:, 0] = RNG.uniform(size=n_sims) < p_play
    u = RNG.uniform(size=(n_sims, n_games))
    for g in range(1, n_games):
        s[:, g] = np.where(s[:, g - 1], u[:, g] >= p_out_after_play, u[:, g] >= q_eff)
    # very low availability (season-ending injuries) needs longer streaks than q_mm allows
    return s


# ---------------------------------------------------------------- run
def crps_samples(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS from samples. x: (n_sims, M), y: (M,). Lower is better."""
    xs = np.sort(x, axis=0).astype(float)
    n = xs.shape[0]
    a = np.abs(xs - y[None, :]).mean(0)
    i = np.arange(1, n + 1)[:, None]
    b = ((2 * i - n - 1) * xs).sum(0) / (n * n)
    return a - b


def run(seasons=range(2020, 2026)):
    T, P = build_targets(seasons)
    T = add_variants(T, P)
    T.to_parquet(PROC / "season_sim_targets.parquet", index=False)
    games, seasons_out = [], []
    for S in seasons:
        var, C = fit_params(P, S)
        q_mm, p_league = availability_params(T, S)
        tau = season_shift_sd(P, S)
        pool = availability_pool(T, S)
        print(S, "season shift sd", {k: round(v, 3) for k, v in tau.items()}, flush=True)
        TS = T[T.season == S]
        print(S, "players", TS.pid.nunique(), "q_mm", round(q_mm, 3), "p_league", round(p_league, 3), flush=True)
        for pid, d in TS.groupby("pid"):
            played = d.played.to_numpy()
            rates = draw_availability_rate(d.prev_avail.iloc[0], pool)
            avail = simulate_availability(len(d), rates, q_mm)
            z = RNG.standard_normal(N_SIMS)  # season-long shift, shared across stats & variants
            shift = np.column_stack([np.clip(1 + tau[s] * z, 0.05, None) for s in COMP])  # (N, 3)
            act = {s: d[s].to_numpy() for s in COMP}
            row = {"season": S, "pid": pid, "player": d.player.iloc[0], "team_games": len(d),
                   "games_actual": int(played.sum()), "games_sim_mean": avail.sum(1).mean(),
                   "games_sim_p10": np.percentile(avail.sum(1), 10), "games_sim_p90": np.percentile(avail.sum(1), 90)}
            for s in COMP + ["pra"]:
                row[f"{s}_actual_total"] = (d[s].fillna(0).sum() if s != "pra" else sum(d[c].fillna(0).sum() for c in COMP))
            for v in VARIANTS:
                mu = np.column_stack([d[f"{s}_mu_{v}"].to_numpy() for s in COMP])
                sims = draw_shifted(mu, shift, var, C)        # (N, G, 3)
                pra = sims.sum(2)
                for mode, mask in (("actual", np.broadcast_to(played, avail.shape)), ("simulated", avail)):
                    tot = np.concatenate([(sims * mask[:, :, None]).sum(1), (pra * mask).sum(1)[:, None]], 1)  # (N, 4)
                    for j, s in enumerate(COMP + ["pra"]):
                        y = row[f"{s}_actual_total"]
                        x = tot[:, j]
                        q = np.percentile(x, [10, 25, 50, 75, 90])
                        row[f"{s}_{v}_{mode}_mean"] = x.mean()
                        row[f"{s}_{v}_{mode}_p10"], row[f"{s}_{v}_{mode}_p90"] = q[0], q[4]
                        row[f"{s}_{v}_{mode}_in50"] = q[1] <= y <= q[3]
                        row[f"{s}_{v}_{mode}_in80"] = q[0] <= y <= q[4]
                        row[f"{s}_{v}_{mode}_crps"] = crps_samples(x[:, None], np.array([y]))[0]
                # per-matchup (played games only)
                if played.any():
                    sp = sims[:, played, :]
                    g = d[played][["season", "gameId", "game_date", "pid", "player", "opp"] + COMP].copy()
                    g["variant"] = v
                    for j, s in enumerate(COMP + ["pra"]):
                        x = sp[:, :, j] if s != "pra" else sp.sum(2)
                        y = g[s].to_numpy() if s != "pra" else g[COMP].sum(1).to_numpy()
                        g[f"{s}_mean"] = x.mean(0)
                        q = np.percentile(x, [10, 50, 90], axis=0)
                        g[f"{s}_p10"], g[f"{s}_p50"], g[f"{s}_p90"] = q
                        g[f"{s}_crps"] = crps_samples(x, y)
                        g[f"{s}_in80"] = (q[0] <= y) & (y <= q[2])
                    games.append(g)
            seasons_out.append(row)
    G = pd.concat(games, ignore_index=True)
    R = pd.DataFrame(seasons_out)
    G.to_parquet(PROC / "season_sim_games.parquet", index=False)
    R.to_parquet(PROC / "season_sim_seasons.parquet", index=False)
    return T, G, R


def summarise(G: pd.DataFrame, R: pd.DataFrame):
    g = []
    for v, d in G.groupby("variant"):
        for s in COMP + ["pra"]:
            y = d[s] if s != "pra" else d[COMP].sum(1)
            g.append({"variant": v, "stat": s, "games": len(d), "mae_mean": (y - d[f"{s}_mean"]).abs().mean(),
                      "crps": d[f"{s}_crps"].mean(), "coverage80": d[f"{s}_in80"].mean()})
    per_game = pd.DataFrame(g)
    r = []
    for mode in ("actual", "simulated"):
        for v in VARIANTS:
            for s in COMP + ["pra"]:
                y = R[f"{s}_actual_total"]
                m = R[f"{s}_{v}_{mode}_mean"]
                r.append({"games_mode": mode, "variant": v, "stat": s, "player_seasons": len(R),
                          "mae_total": (y - m).abs().mean(), "mape_total": ((y - m).abs() / y.clip(lower=1)).median(),
                          "mae_per_game_avg": ((y - m).abs() / R.games_actual.clip(lower=1)).mean() if mode == "actual" else np.nan,
                          "crps_total": R[f"{s}_{v}_{mode}_crps"].mean(),
                          "coverage50": R[f"{s}_{v}_{mode}_in50"].mean(), "coverage80": R[f"{s}_{v}_{mode}_in80"].mean()})
    season = pd.DataFrame(r)
    games_played = pd.DataFrame({"mae_games": [(R.games_actual - R.games_sim_mean).abs().mean()],
                                 "coverage80_games": [((R.games_actual >= R.games_sim_p10) & (R.games_actual <= R.games_sim_p90)).mean()],
                                 "actual_avg_games": [R.games_actual.mean()], "sim_avg_games": [R.games_sim_mean.mean()]})
    return per_game, season, games_played


# ================================================================ preseason
PRE_VARIANTS = ["base", "bias", "simdef", "simdef_placebo", "h2h", "last_szn"]


def preseason_means(T: pd.DataFrame, P: pd.DataFrame) -> pd.DataFrame:
    """Everything frozen before opening night of season S:
      base     : the player's first-game projection (uses only last-season / preseason info for him)
      bias     : base + his average miss vs our projections last season (shrunk)
      simdef   : base + his last-season miss vs the 8 defenses most similar to each opponent,
                 using each opponent's preseason fingerprint
      simdef_placebo : same, but the 8 LEAST similar defenses (equal weights)
      h2h      : his last-season average vs that team (base if they didn't meet)
      last_szn : last season's per-game average"""
    prof = pd.read_parquet(PROC / "defense_profiles.parquet")
    V = vectors(prof)
    V = V / np.linalg.norm(V, axis=1, keepdims=True).clip(1e-9)
    prof["vi"] = np.arange(len(prof))
    first_vi = prof.sort_values("date").groupby(["season", "team"]).vi.first()   # preseason fingerprint
    game_vi = pd.Series(prof.vi.to_numpy(), index=pd.MultiIndex.from_arrays([prof.gameId, prof.team]))
    T = T.copy()
    T["vi_pre"] = [first_vi.get((s, o), -1) for s, o in zip(T.season, T.opp)]
    first_proj = T[T.played].sort_values("game_date").groupby(["season", "pid"])[[f"{s}_ensemble" for s in COMP]].first()
    T = T.join(first_proj.add_suffix("_pre0"), on=["season", "pid"])
    H = P[P.pid.isin(T.pid.unique())].copy()
    H["vi"] = [game_vi.get((a, b), -1) for a, b in zip(H.gameId, H.opp)]
    for s in COMP:
        H[f"res_{s}"] = H[s] - H[f"{s}_ensemble"]
        for v in ("bias", "simdef", "simdef_placebo", "h2h"):
            T[f"{s}_pre_{v}"] = np.nan
    Hg = {k: d for k, d in H.groupby(["season", "pid"])}
    for (S, pid), idx in T.groupby(["season", "pid"]).indices.items():
        t = T.iloc[idx]
        h = Hg.get((S - 1, pid))
        base = {s: t[f"{s}_ensemble_pre0"].to_numpy() for s in COMP}
        if h is None or len(h) == 0:
            for s in COMP:
                for v in ("bias", "simdef", "simdef_placebo", "h2h"):
                    T.iloc[idx, T.columns.get_loc(f"{s}_pre_{v}")] = base[s]
            continue
        Sm = V[t.vi_pre.to_numpy()] @ V[h.vi.to_numpy()].T
        Sm[(t.vi_pre.to_numpy() < 0)[:, None] | (h.vi.to_numpy() < 0)[None, :]] = -9
        order = np.argsort(-Sm, axis=1)[:, :TOP_K]
        w = np.take_along_axis(Sm, order, 1)
        w = np.where(w >= MIN_SIM, w, 0)
        same = t.opp.to_numpy()[:, None] == h.opp.to_numpy()[None, :]
        Sp = np.where(Sm == -9, 9, Sm)
        least = np.argsort(Sp, axis=1)[:, :TOP_K]  # placebo: 8 LEAST similar, equal weights
        for s in COMP:
            r = h[f"res_{s}"].to_numpy()
            T.iloc[idx, T.columns.get_loc(f"{s}_pre_bias")] = base[s] + r.sum() / (len(r) + 5.0)
            T.iloc[idx, T.columns.get_loc(f"{s}_pre_simdef")] = base[s] + (w * r[order]).sum(1) / (w.sum(1) + SHRINK)
            T.iloc[idx, T.columns.get_loc(f"{s}_pre_simdef_placebo")] = base[s] + r[least].sum(1) / (least.shape[1] + SHRINK)
            n = same.sum(1)
            h2h = (same * h[s].to_numpy()[None, :]).sum(1) / np.maximum(n, 1)
            T.iloc[idx, T.columns.get_loc(f"{s}_pre_h2h")] = np.where(n > 0, h2h, base[s])
    for s in COMP:
        T[f"{s}_pre_base"] = T[f"{s}_ensemble_pre0"]
        T[f"{s}_pre_last_szn"] = T[f"{s}_last"].fillna(T[f"{s}_ensemble_pre0"])
        for v in PRE_VARIANTS:
            T[f"{s}_pre_{v}"] = T[f"{s}_pre_{v}"].clip(lower=0.05)
    return T


def preseason_shift_sd(T: pd.DataFrame, var: dict, exclude_season: int, v: str) -> dict:
    """Season-long relative miss of the PRESEASON forecast, beyond game noise (other seasons)."""
    o = T[(T.season != exclude_season) & T.played]
    out = {}
    for s in COMP:
        mu = o[f"{s}_pre_{v}"]
        g = o.assign(mu=mu, v=var[s][0] * mu.clip(lower=0.05) ** var[s][1]).groupby(["season", "pid"])
        a = g.agg(y=(s, "sum"), mu=("mu", "sum"), v=("v", "sum"), n=("mu", "size"))
        a = a[a.n >= 20]
        rel = (a.y - a.mu) / a.mu
        # robust spread: guard against a few huge role changes dominating
        sd = 1.4826 * np.median(np.abs(rel - rel.median()))
        out[s] = float(np.sqrt(max(sd ** 2 - (a.v / a.mu ** 2).mean(), 0)))
        out[s + "_bias"] = float(rel.median())
    return out


def run_preseason(seasons=range(2020, 2026)):
    T = pd.read_parquet(PROC / "season_sim_targets.parquet")
    P = pd.read_parquet(PROC / "projections_all_games_oos.parquet")
    T = preseason_means(T, P)
    T.to_parquet(PROC / "season_sim_preseason_targets.parquet", index=False)
    rows = []
    for S in seasons:
        var, C = fit_params(P, S)
        q_mm, _ = availability_params(T, S)
        pool = availability_pool(T, S)
        taus = {v: preseason_shift_sd(T, var, S, v) for v in PRE_VARIANTS}
        print(S, {v: {k: round(x, 3) for k, x in t.items()} for v, t in taus.items() if v in ("base", "simdef")}, flush=True)
        for pid, d in T[T.season == S].groupby("pid"):
            played = d.played.to_numpy()
            avail = simulate_availability(len(d), draw_availability_rate(d.prev_avail.iloc[0], pool), q_mm)
            z = RNG.standard_normal(N_SIMS)
            row = {"season": S, "pid": pid, "player": d.player.iloc[0], "games_actual": int(played.sum()),
                   "games_sim_mean": avail.sum(1).mean()}
            for s in COMP + ["pra"]:
                row[f"{s}_actual_total"] = d[s].fillna(0).sum() if s != "pra" else sum(d[c].fillna(0).sum() for c in COMP)
            for v in PRE_VARIANTS:
                tau = taus[v]
                shift = np.column_stack([np.clip(1 + tau[s] * z, 0.05, None) for s in COMP])
                mu = np.column_stack([d[f"{s}_pre_{v}"].to_numpy() for s in COMP])
                sims = draw_shifted(mu, shift, var, C)
                pra = sims.sum(2)
                for mode, mask in (("actual", np.broadcast_to(played, avail.shape)), ("simulated", avail)):
                    tot = np.concatenate([(sims * mask[:, :, None]).sum(1), (pra * mask).sum(1)[:, None]], 1)
                    for j, s in enumerate(COMP + ["pra"]):
                        y, x = row[f"{s}_actual_total"], tot[:, j]
                        q = np.percentile(x, [10, 25, 75, 90])
                        row[f"{s}_{v}_{mode}_mean"] = x.mean()
                        row[f"{s}_{v}_{mode}_in50"] = q[1] <= y <= q[2]
                        row[f"{s}_{v}_{mode}_in80"] = q[0] <= y <= q[3]
                        row[f"{s}_{v}_{mode}_crps"] = crps_samples(x[:, None], np.array([y]))[0]
                        row[f"{s}_{v}_{mode}_p10"], row[f"{s}_{v}_{mode}_p90"] = q[0], q[3]
            rows.append(row)
    R = pd.DataFrame(rows)
    R.to_parquet(PROC / "season_sim_preseason.parquet", index=False)
    out = []
    for mode in ("actual", "simulated"):
        for v in PRE_VARIANTS:
            for s in COMP + ["pra"]:
                y, m = R[f"{s}_actual_total"], R[f"{s}_{v}_{mode}_mean"]
                out.append({"games_mode": mode, "variant": v, "stat": s, "player_seasons": len(R),
                            "mae_total": (y - m).abs().mean(),
                            "mae_per_game_avg": ((y - m).abs() / R.games_actual.clip(lower=1)).mean() if mode == "actual" else np.nan,
                            "crps_total": R[f"{s}_{v}_{mode}_crps"].mean(),
                            "coverage50": R[f"{s}_{v}_{mode}_in50"].mean(), "coverage80": R[f"{s}_{v}_{mode}_in80"].mean()})
    return T, R, pd.DataFrame(out)
