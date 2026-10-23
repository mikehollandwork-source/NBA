"""Does a team winning or losing move its players' stats?

Measured as player residual = actual − our out-of-sample projection (2019-25), so a player's
normal level and every input the model already uses are taken out.
  1. same game (hindsight): residual by win/loss and final margin — real but not usable pregame
  2. pregame: residual by the winner model's PROJECTED margin (blowout risk) — usable; tested as a
     correction fit on earlier seasons, applied to the next, vs a placebo (margins shuffled)
  3. carry-over: residual by the team's PREVIOUS game result (bounce-back / momentum)
Run: python -m nba.analysis.win_effect   Report: reports/phase2/win_effect.md
"""
import numpy as np
import pandas as pd

from nba.analysis import winners as W
from nba.config import PROC

STATS = ["min", "pts", "reb", "ast", "pra"]
RNG = np.random.default_rng(31)


def game_margins(seasons=range(2019, 2026)) -> pd.DataFrame:
    """Per team-game: out-of-sample projected margin, actual margin, previous game's result."""
    T = pd.read_parquet(PROC / "winners_team_games.parquet")
    k, reg, k2 = W.K_PRIOR_FINAL
    G = W.game_rows(T, k, reg, k2)
    G["proj"] = np.nan
    for y in seasons:
        m = W.fit(G[G.season < y], W.FEATS)
        G.loc[G.season == y, "proj"] = W.predict(m, G[G.season == y], 2000).proj_margin
    h = G[["gameId", "home_team", "proj", "margin"]].rename(columns={"home_team": "team"})
    a = G[["gameId", "away_team", "proj", "margin"]].rename(columns={"away_team": "team"})
    a[["proj", "margin"]] = -a[["proj", "margin"]]
    M = pd.concat([h, a]).dropna(subset=["proj"])
    T = T.sort_values(["team", "game_date"])
    T["prev_margin"] = T.groupby(["season", "team"]).margin.shift(1)
    return M.merge(T[["gameId", "team", "prev_margin", "gp"]], on=["gameId", "team"])


def load() -> pd.DataFrame:
    P = pd.read_parquet(PROC / "projections_all_games_oos.parquet")
    P["proj_pra"] = P.pra_ensemble
    for s in ["pts", "reb", "ast"]:
        P[f"proj_{s}"] = P[f"{s}_ensemble"]
    P["res_min"] = P["min"] - P.proj_min
    for s in ["pts", "reb", "ast", "pra"]:
        P[f"res_{s}"] = P[s] - P[f"proj_{s}"]
    P = P.merge(game_margins(), on=["gameId", "team"])
    P["role"] = np.where(P.proj_min >= 30, "starter 30+ min", np.where(P.proj_min >= 20, "rotation 20-30", "bench <20"))
    P["win"] = P.margin > 0
    return P


def table(P, by) -> pd.DataFrame:
    return P.groupby(by, observed=True).agg(n=("res_pts", "size"), **{f"{s}": (f"res_{s}", "mean") for s in STATS}).round(2)


def correction_test(P: pd.DataFrame, col: str, seasons=range(2020, 2026), placebo=False) -> pd.DataFrame:
    """Fit residual ~ role × (col, |col|) on earlier seasons; apply to the next; % squared-error change."""
    rows = []
    for y in seasons:
        tr, te = P[(P.season < y)].dropna(subset=[col]), P[P.season == y].dropna(subset=[col]).copy()
        if placebo:
            te[col] = te.groupby("role")[col].transform(lambda x: x.sample(frac=1, random_state=y).to_numpy())
        for s in STATS:
            base = ((te[f"res_{s}"]) ** 2).mean()
            pred = np.zeros(len(te))
            for r, d in tr.groupby("role"):
                X = np.column_stack([np.ones(len(d)), d[col], d[col].abs()])
                b = np.linalg.lstsq(X, d[f"res_{s}"].to_numpy(), rcond=None)[0]
                mt = (te.role == r).to_numpy()
                pred[mt] = np.column_stack([np.ones(mt.sum()), te.loc[mt, col], te.loc[mt, col].abs()]) @ b
            new = ((te[f"res_{s}"] - pred) ** 2).mean()
            rows.append({"season": y, "stat": s, "gain_pct": 100 * (1 - new / base)})
    return pd.DataFrame(rows)


def run():
    P = load()
    P.to_parquet(PROC / "win_effect.parquet", index=False)
    out = {}
    out["same_game_win"] = table(P, ["role", "win"])
    P["final"] = pd.cut(P.margin, [-200, -15, -5, 5, 15, 200], labels=["lost 15+", "lost 6-15", "within 5", "won 6-15", "won 15+"])
    out["same_game_margin"] = table(P, ["role", "final"])
    P["expect"] = pd.cut(P.proj, [-50, -8, -3, 3, 8, 50], labels=["big dog 8+", "dog 3-8", "toss-up", "fav 3-8", "big fav 8+"])
    out["pregame_expected"] = table(P, ["role", "expect"])
    P["prev"] = pd.cut(P.prev_margin, [-200, -15, 0, 15, 200], labels=["lost 15+", "lost <15", "won <15", "won 15+"])
    out["previous_game"] = table(P, ["role", "prev"])
    for name, col in (("pregame", "proj"), ("previous", "prev_margin")):
        for plc in (False, True):
            c = correction_test(P, col, placebo=plc)
            out[f"test_{name}{'_placebo' if plc else ''}"] = c.groupby("stat").gain_pct.agg(["mean", "min", "max"]).round(2)
            out[f"seasons_{name}{'_placebo' if plc else ''}"] = c.pivot(index="season", columns="stat", values="gain_pct").round(2)
    return out


if __name__ == "__main__":
    for k, v in run().items():
        print(f"\n== {k}\n{v.to_string()}")
