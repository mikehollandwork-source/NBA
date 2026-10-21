"""Closing-line backtest — lines are used ONLY here, for evaluation. They never
feed any projection.

Input: CSV files in data/lines/ with columns
    date (YYYY-MM-DD, local game date), player, stat, line, over_odds, under_odds
    optional: book, team
  stat in {pts, reb, ast, pra, pr, pa, ra}; odds American (e.g. -115, +105).
  (scripts/fetch_odds_api.py writes this format from The Odds API historical
  endpoint if ODDS_API_KEY is set — paid plans only.)

What it measures (all out-of-sample projections + calibrated distributions):
  * model P(over) vs no-vig market P(over): Brier / log loss
  * information test: logistic regression of the result on market logit and
    model logit. A model coefficient reliably > 0 = edge the market misses.
  * betting sim: bet the side where model prob exceeds no-vig prob by >= X;
    hit rate, ROI at the actual odds, by stat / edge bucket / season.
"""
import glob
import unicodedata

import numpy as np
import pandas as pd

from nba.analysis.projection import fit_variance, dist
from nba.config import PROC, ROOT

LINES_DIR = ROOT / "data" / "lines"
STATS = ["pts", "reb", "ast", "pra", "pr", "pa", "ra"]


def _key(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = s.replace(" jr", "").replace(" sr", "").replace(" iii", "").replace(" ii", "").replace(" iv", "")
    return "".join(ch for ch in s if ch.isalpha())


def american_to_prob(o):
    o = np.asarray(o, float)
    return np.where(o < 0, -o / (-o + 100), 100 / (o + 100))


def payout(o):
    """Profit per 1 unit staked on a win."""
    o = np.asarray(o, float)
    return np.where(o < 0, 100 / -o, o / 100)


def load_lines(path_glob=None) -> pd.DataFrame:
    files = sorted(glob.glob(str(path_glob or LINES_DIR / "*.csv")))
    if not files:
        raise FileNotFoundError(f"no line files in {LINES_DIR}")
    L = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    L["date"] = pd.to_datetime(L.date)
    L["k"] = L.player.map(_key)
    L = L[L.stat.isin(STATS)].dropna(subset=["line", "over_odds", "under_odds"])
    po, pu = american_to_prob(L.over_odds), american_to_prob(L.under_odds)
    L["p_mkt"] = po / (po + pu)  # no-vig
    return L


def attach_projections(L: pd.DataFrame, proj_path=None) -> pd.DataFrame:
    P = pd.read_parquet(proj_path or PROC / "projections_all_games_oos.parquet")
    if "scenario" in P:
        P = P[P.scenario == "injury_report"]
    P["k"] = P.player.map(_key)
    # local game date: projections carry UTC-ish ESPN dates; allow +-1 day
    rows = []
    for off in (0, -1, 1):
        q = P.assign(date=pd.to_datetime(P.game_date) + pd.Timedelta(days=off))
        rows.append(L.merge(q, on=["k", "date"], how="inner", suffixes=("", "_p")).assign(off=abs(off)))
    m = pd.concat(rows).sort_values("off").drop_duplicates(subset=["k", "date", "stat", "line"] + (["book"] if "book" in L else []))
    return m


def model_probs(m: pd.DataFrame, P: pd.DataFrame) -> pd.DataFrame:
    """P(over line) from our calibrated negative binomial; variance curve fit on
    OTHER seasons of out-of-sample projections."""
    m = m.copy()
    m["p_model"] = np.nan
    m["actual"] = np.nan
    for s in STATS:
        for y in m.season.unique():
            idx = (m.stat == s) & (m.season == y)
            if not idx.any():
                continue
            o = P[(P.season != y)].dropna(subset=[s, f"{s}_ensemble"])
            k, pw = fit_variance(o[s].to_numpy(), o[f"{s}_ensemble"].to_numpy())
            d = dist(m.loc[idx, f"{s}_ensemble"].to_numpy(), k, pw)
            line = m.loc[idx, "line"].to_numpy()
            p_over = 1 - d.cdf(np.floor(line))
            # whole-number lines: a push refunds; condition on no push
            push = np.where(line == np.floor(line), d.pmf(line), 0)
            m.loc[idx, "p_model"] = np.where(push > 0, (1 - d.cdf(line)) / (1 - push), p_over)
            m.loc[idx, "actual"] = m.loc[idx, s].to_numpy()
    m["result"] = np.where(m.actual > m.line, 1.0, np.where(m.actual < m.line, 0.0, np.nan))
    return m.dropna(subset=["result", "p_model"])


def information_test(m: pd.DataFrame) -> pd.DataFrame:
    """result ~ logit(p_market) + logit(p_model). Bootstrap CI on the model weight."""
    from sklearn.linear_model import LogisticRegression
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    out = []
    rng = np.random.default_rng(3)
    for s, d in list(m.groupby("stat")) + [("all", m)]:
        X = np.column_stack([lg(d.p_mkt), lg(d.p_model)])
        y = d.result.to_numpy()
        coef = LogisticRegression(C=1e6).fit(X, y).coef_[0]
        bs = []
        for _ in range(200):
            i = rng.integers(0, len(d), len(d))
            bs.append(LogisticRegression(C=1e6).fit(X[i], y[i]).coef_[0][1])
        out.append({"stat": s, "n": len(d), "w_market": coef[0], "w_model": coef[1],
                    "w_model_ci_lo": np.percentile(bs, 2.5), "w_model_ci_hi": np.percentile(bs, 97.5),
                    "brier_market": np.mean((d.p_mkt - y) ** 2), "brier_model": np.mean((d.p_model - y) ** 2)})
    return pd.DataFrame(out)


def betting(m: pd.DataFrame, thresholds=(0.0, 0.02, 0.04, 0.06, 0.08)) -> pd.DataFrame:
    edge_over = m.p_model - m.p_mkt
    side_over = edge_over > 0
    p_side = np.where(side_over, m.p_model, 1 - m.p_model)
    odds = np.where(side_over, m.over_odds, m.under_odds)
    win = np.where(side_over, m.result == 1, m.result == 0)
    profit = np.where(win, payout(odds), -1.0)
    rows = []
    for t in thresholds:
        sel = np.abs(edge_over) >= t
        for s in ["all"] + STATS:
            k = sel & ((m.stat == s) if s != "all" else True)
            if k.sum() == 0:
                continue
            rows.append({"min_edge": t, "stat": s, "bets": int(k.sum()), "hit_rate": win[k].mean(),
                         "roi": profit[k].mean(), "avg_model_prob": p_side[k].mean()})
    return pd.DataFrame(rows)


def run(path_glob=None):
    L = load_lines(path_glob)
    P = pd.read_parquet(PROC / "projections_all_games_oos.parquet")
    m = model_probs(attach_projections(L), P)
    return m, information_test(m), betting(m)
