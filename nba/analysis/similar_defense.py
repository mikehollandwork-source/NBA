"""Similar-defense signal.

For a player tonight vs defense D: find the player's earlier games (this season
and last) against the defenses whose fingerprint (as of that game) was most
similar to D's fingerprint today. Signal = how much the player beat / missed
our projection in those games (similarity-weighted, shrunk toward 0).

Control: the same player's average miss vs ALL earlier opponents ("general
bias"). The similar-defense signal only matters if it adds beyond that.
Everything uses games before the one being predicted; projections are the
out-of-sample ensemble (injury-report scenario).
"""
import numpy as np
import pandas as pd

from nba.analysis.defense_profiles import vectors
from nba.config import PROC

STATS = ["min", "pts", "reb", "ast", "pra", "tov"]
TOP_K = 8          # most similar prior games used
MIN_SIM = 0.30     # ignore weakly similar defenses
SHRINK = 2.0       # pseudo-games at 0 for the similar-defense signal
SHRINK_ALL = 5.0   # pseudo-games at 0 for the general-bias control


def load():
    p = pd.read_parquet(PROC / "projections_oos.parquet")
    p = p[p.scenario == "injury_report"].copy()
    p = p.rename(columns={"proj_min": "min_ensemble"})
    prof = pd.read_parquet(PROC / "defense_profiles.parquet")
    v = vectors(prof)
    v = v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-9)
    key = pd.Series(np.arange(len(prof)), index=pd.MultiIndex.from_arrays([prof.gameId, prof.team]))
    p["vi"] = [key.get((g, o), -1) for g, o in zip(p.gameId, p.opp)]
    p = p[p.vi >= 0].sort_values(["pid", "game_date", "gameId"]).reset_index(drop=True)
    return p, v, prof


def signals(p: pd.DataFrame, v: np.ndarray, exclude_same_team: bool = False, least: bool = False) -> pd.DataFrame:
    res = {s: (p[s] - p[f"{s}_ensemble"]).to_numpy() for s in STATS}
    out = {f"sig_{s}": np.full(len(p), np.nan) for s in STATS}
    out.update({f"gen_{s}": np.full(len(p), np.nan) for s in STATS})
    out.update({"n_similar": np.zeros(len(p)), "mean_sim": np.full(len(p), np.nan), "similar_idx": [None] * len(p)})
    for pid, idx in p.groupby("pid").indices.items():
        if len(idx) < 2:
            continue
        V = v[p.vi.to_numpy()[idx]]
        S = V @ V.T
        seas = p.season.to_numpy()[idx]
        opp = p.opp.to_numpy()[idx]
        for a in range(1, len(idx)):
            cand = np.arange(a)
            cand = cand[seas[cand] >= seas[a] - 1]
            if exclude_same_team:
                cand = cand[opp[cand] != opp[a]]
            if len(cand) == 0:
                continue
            i = idx[a]
            for s in STATS:
                out[f"gen_{s}"][i] = res[s][idx[cand]].sum() / (len(cand) + SHRINK_ALL)
            sims = S[a, cand]
            if least:  # placebo: the LEAST similar defenses, equal weights
                top = cand[np.argsort(sims)[:TOP_K]]
                w = np.ones(len(top))
            else:
                keep = cand[sims >= MIN_SIM]
                if len(keep) == 0:
                    continue
                top = keep[np.argsort(-S[a, keep])[:TOP_K]]
                w = S[a, top]
            out["n_similar"][i] = len(top)
            out["mean_sim"][i] = w.mean()
            out["similar_idx"][i] = idx[top]
            for s in STATS:
                out[f"sig_{s}"][i] = (w * res[s][idx[top]]).sum() / (w.sum() + SHRINK)
    o = pd.DataFrame(out, index=p.index)
    return pd.concat([p, o], axis=1)


def backtest(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Does the signal predict tonight's miss vs our projection?
    Adjustment weights are fit on the OTHER seasons (leave-one-season-out)."""
    rows, buckets, line = [], [], []
    for s in STATS:
        x = d.dropna(subset=[f"sig_{s}", f"gen_{s}"]).copy()
        x["r"] = x[s] - x[f"{s}_ensemble"]
        for y in sorted(x.season.unique()):
            tr, te = x[x.season != y], x[x.season == y]
            def fit(cols):
                A = np.column_stack([np.ones(len(tr))] + [tr[c] for c in cols])
                b = np.linalg.lstsq(A, tr.r, rcond=None)[0]
                B = np.column_stack([np.ones(len(te))] + [te[c] for c in cols])
                return B @ b, b
            e0 = (te.r ** 2).mean()
            p_gen, _ = fit([f"gen_{s}"])
            p_sig, bs = fit([f"sig_{s}"])
            p_both, bb = fit([f"gen_{s}", f"sig_{s}"])
            rows.append({"stat": s, "season": y, "n": len(te),
                         "corr_signal": np.corrcoef(te[f"sig_{s}"], te.r)[0, 1],
                         "corr_general_bias": np.corrcoef(te[f"gen_{s}"], te.r)[0, 1],
                         "gain_general_bias_pct": 100 * (1 - ((te.r - p_gen) ** 2).mean() / e0),
                         "gain_similar_def_pct": 100 * (1 - ((te.r - p_sig) ** 2).mean() / e0),
                         "gain_both_pct": 100 * (1 - ((te.r - p_both) ** 2).mean() / e0),
                         "weight_similar_def_in_both": bb[2]})
        # buckets of signal strength (pooled seasons)
        x["bucket"] = pd.qcut(x[f"sig_{s}"], 10, labels=False, duplicates="drop")
        b = x.groupby("bucket").agg(n=("r", "size"), signal=(f"sig_{s}", "mean"), miss_tonight=("r", "mean"),
                                    beat_projection=("r", lambda r: (r > 0).mean())).reset_index()
        b.insert(0, "stat", s)
        buckets.append(b)
        # vs season-average stand-in line: do projection and similar-defense history agree?
        if s != "min":
            ln = x[f"{s}_szn"]
            proj_side = np.sign(x[f"{s}_ensemble"] - ln)
            hist_side = np.sign(x[f"{s}_ensemble"] + x[f"sig_{s}"] - ln) * (x[f"sig_{s}"].abs() > 0.5 * x[f"{s}_sd"] * 0.25)
            agree = np.sign(x[f"sig_{s}"]) == proj_side
            strong = x[f"sig_{s}"].abs() >= x[f"sig_{s}"].abs().quantile(0.7)
            for name, m in (("all games", np.ones(len(x), bool)), ("history agrees with projection", agree & strong),
                            ("history disagrees with projection", (~agree) & strong)):
                y_ = x[m]
                over = (y_[s] > y_[f"{s}_szn"])
                side = proj_side[m]
                ok = (y_[s] != y_[f"{s}_szn"]) & (side != 0)
                hit = ((over == (side > 0)) & ok).sum() / ok.sum()
                line.append({"stat": s, "group": name, "n": int(ok.sum()), "projection_side_hit_rate": hit})
    return pd.DataFrame(rows), pd.concat(buckets), pd.DataFrame(line)


def accuracy_vs_similar(d: pd.DataFrame) -> pd.DataFrame:
    """How accurate were our projections in games vs defenses similar to each
    opponent, compared with all games (same players)."""
    rows = []
    for s in ["pts", "reb", "ast", "pra"]:
        e = d[s] - d[f"{s}_ensemble"]
        rows.append({"stat": s, "games": len(d), "mae_projection": e.abs().mean(), "bias_projection": e.mean(),
                     "mae_season_avg": (d[s] - d[f"{s}_szn"]).abs().mean()})
    return pd.DataFrame(rows)


def examples(d: pd.DataFrame, prof: pd.DataFrame, stat="pts", season=2025, n=6) -> list[dict]:
    """Biggest-signal games in a season: tonight's opponent, the similar defenses,
    what we projected and what happened in each."""
    from nba.analysis.defense_profiles import card
    x = d[(d.season == season) & d[f"sig_{stat}"].notna() & (d[f"{stat}_szn"] >= 15)]
    x = x.reindex(x[f"sig_{stat}"].abs().sort_values(ascending=False).index).drop_duplicates("pid").head(n)
    out = []
    pkey = prof.set_index(["gameId", "team"])
    for _, r in x.iterrows():
        sims = d.loc[r.similar_idx]
        out.append({
            "player": r.player, "date": str(r.game_date.date()), "opp": r.opp,
            "opp_card": card(pkey.loc[(r.gameId, r.opp)]),
            "season_avg": round(r[f"{stat}_szn"], 1), "projection": round(r[f"{stat}_ensemble"], 1),
            "similar_defense_signal": round(r[f"sig_{stat}"], 1), "actual": int(r[stat]),
            "similar_games": [{"date": str(q.game_date.date()), "opp": q.opp, "projection": round(q[f"{stat}_ensemble"], 1),
                               "actual": int(q[stat])} for _, q in sims.iterrows()]})
    return out


def robustness(p, v):
    """Same backtest under alternative settings + a placebo using the LEAST
    similar defenses. If the placebo works as well, similarity adds nothing."""
    import nba.analysis.similar_defense as m
    base = (m.TOP_K, m.MIN_SIM)
    variants = {"default (8 most similar, sim>=0.3)": dict(k=8, ms=0.3, ex=False, flip=False),
                "4 most similar, sim>=0.5": dict(k=4, ms=0.5, ex=False, flip=False),
                "15 most similar, sim>=0.0": dict(k=15, ms=0.0, ex=False, flip=False),
                "exclude same opponent": dict(k=8, ms=0.3, ex=True, flip=False),
                "PLACEBO: 8 least similar (equal weights)": dict(k=8, ms=-1.0, ex=False, flip=True),
                "default but equal weights": dict(k=8, ms=0.3, ex=False, flip=False, eq=True)}
    rows = []
    for name, c in variants.items():
        m.TOP_K, m.MIN_SIM = c["k"], c["ms"]
        d = signals(p, v, exclude_same_team=c["ex"], least=c["flip"])
        bt, _, _ = backtest(d)
        g = bt.groupby("stat")[["corr_signal", "gain_similar_def_pct", "gain_both_pct", "gain_general_bias_pct"]].mean()
        g["variant"] = name
        rows.append(g.reset_index())
    m.TOP_K, m.MIN_SIM = base
    return pd.concat(rows)
