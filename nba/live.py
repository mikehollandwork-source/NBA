"""Tonight's projections (opening week): preseason model + current rosters + injuries.

Per-game mean = preseason projection (model trained on all seasons to date,
blended 50/50 with last season's per-game average). Ranges come from the
calibrated game distribution plus season-level uncertainty of a preseason
forecast. Players listed OUT get no projection.

Known gap: teammates-out (the biggest game-day factor) is NOT applied in a
team's first games — the game-day model only learns it from recent team games.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from nba.analysis import preseason as pre
from nba.analysis import season_sim as ss
from nba.config import PROC


def season_features(R: pd.DataFrame, season: int) -> pd.DataFrame:
    ps, allg, tg = pre.player_seasons()
    f_hist = pre.feature_table(ps, allg, tg)
    keep = ["gp", "avail", "starter", "out_episodes", "out_longest", "out_at_end", "out_last10", "inj_dnp", "rest_dnp",
            "lower_dnp", "team_last"] + [f"{s}_pg" for s in pre.STATS + ["usg"]] + [f"{s}_p36" for s in pre.RATE + ["usg"]] \
        + [f"{s}_tot" for s in pre.STATS]
    f = R[["pid", "player", "team", "height_in", "weight_lb", "dob", "pos"]].copy()
    f["season"] = season
    f["age"] = (pd.Timestamp(f"{season}-10-15") - pd.to_datetime(f.dob, errors="coerce")).dt.days / 365.25
    for k in (1, 2, 3):
        h = ps[ps.season == season - k][["pid"] + keep]
        f = f.merge(h.rename(columns={c: f"{c}_{k}" for c in keep}), on="pid", how="left")
    f["team_change"] = (f.team != f.team_last_1).astype(float).where(f.team_last_1.notna())
    # roster context: current roster (excluding players out for the opener)
    act = R[R.inj_status != "Out"]
    last = ps[ps.season == season - 1][["pid", "usg_pg", "min_pg", "usg_tot", "min_tot", "team_last"]]
    r = act[["team", "pid"]].merge(last[["pid", "usg_pg", "min_pg"]], on="pid", how="left").fillna(0)
    comp = r.groupby("team")[["usg_pg", "min_pg"]].sum().rename(columns=lambda c: "roster_" + c)
    lt = last.rename(columns={"team_last": "team"})
    lt["kept"] = lt.set_index(["team", "pid"]).index.isin(R.set_index(["team", "pid"]).index)
    tot = lt.groupby("team")[["usg_tot", "min_tot"]].sum()
    gone = lt[~lt.kept].groupby("team")[["usg_tot", "min_tot"]].sum()
    vac = (gone / tot).rename(columns={"usg_tot": "usg_vacated", "min_tot": "min_vacated"})
    f = f.join(comp, on="team").join(vac, on="team")
    f["roster_usg_others"] = f.roster_usg_pg - np.where(f.team_change == 0, f.usg_pg_1.fillna(0), 0)
    f["roster_min_others"] = f.roster_min_pg - np.where(f.team_change == 0, f.min_pg_1.fillna(0), 0)
    f["pos_c"] = f.pos.map({"PG": 0, "SG": 0, "G": 0, "SF": 1, "PF": 1, "F": 1, "C": 2})
    return f, f_hist


def preseason_projection(f: pd.DataFrame, f_hist: pd.DataFrame) -> pd.DataFrame:
    X = pre.features(f_hist)
    tr = f_hist[f_hist.gp_1.notna()]
    out = f.copy()
    for s in pre.STATS:
        t2 = tr[tr.y_gp >= 10]
        m = lgb.LGBMRegressor(**pre.PARAMS).fit(t2[X], t2[f"y_{s}"], sample_weight=np.sqrt(t2.y_gp))
        model = np.clip(m.predict(out.reindex(columns=X)), 0, None)
        last = out[f"{s}_pg_1"].to_numpy()
        out[f"proj_{s}"] = np.where(np.isfinite(last), 0.5 * model + 0.5 * last, np.nan)  # rookies: no projection
    m = lgb.LGBMRegressor(**pre.PARAMS).fit(tr[X], tr.y_avail.clip(0, 1))
    out["proj_avail"] = np.clip(m.predict(out.reindex(columns=X)), 0.02, 1)
    return out


def ranges(out: pd.DataFrame, n_sims=4000) -> pd.DataFrame:
    """10/50/90% outcomes for PTS/REB/AST/PRA: calibrated per-game distribution
    plus the season-level uncertainty of a preseason forecast."""
    P = pd.read_parquet(PROC / "projections_all_games_oos.parquet")
    var, C = ss.fit_params(P, -1)
    T = pd.read_parquet(PROC / "season_sim_preseason_targets.parquet")
    pre_f = pd.read_parquet(PROC / "preseason_final.parquet")
    T = T.merge(pre_f[["season", "pid", "pre_pts", "pre_reb", "pre_ast"]], on=["season", "pid"], how="left")
    for s in ss.COMP:
        T[f"{s}_pre_model"] = T[f"pre_{s}"].fillna(T[f"{s}_pre_base"]).clip(lower=0.05)
    tau = ss.preseason_shift_sd(T, var, -1, "model")
    ok = out.proj_pts.notna()
    mu = out.loc[ok, ["proj_pts", "proj_reb", "proj_ast"]].to_numpy()
    z = ss.RNG.standard_normal((n_sims, 1))
    shift = np.concatenate([np.clip(1 + tau[s] * z, 0.05, None) for s in ss.COMP], axis=1)
    sims = np.empty((n_sims, mu.shape[0], 3), dtype=np.int32)
    # independent season shift per player: loop in chunks over players
    for i in range(mu.shape[0]):
        zi = ss.RNG.standard_normal(n_sims)
        sh = np.column_stack([np.clip(1 + tau[s] * zi, 0.05, None) for s in ss.COMP])
        sims[:, i:i + 1, :] = ss.draw_shifted(mu[i:i + 1], sh, var, C, n_sims=n_sims)
    for j, s in enumerate(ss.COMP + ["pra"]):
        x = sims[:, :, j] if s != "pra" else sims.sum(2)
        q = np.percentile(x, [10, 50, 90], axis=0)
        out.loc[ok, f"{s}_p10"], out.loc[ok, f"{s}_p50"], out.loc[ok, f"{s}_p90"] = q
    out["proj_pra"] = out.proj_pts + out.proj_reb + out.proj_ast
    out.attrs["tau"] = tau
    return out
