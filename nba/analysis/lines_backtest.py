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


BREAK_EVEN_120 = 120 / 220  # 54.55%


def proxy_backtest(odds: int = -120, proxies=("szn", "l10", "l5"), min_gp: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """UPPER-BOUND test without real lines: the 'line' is a public average
    (season-to-date / last 10 / last 5) rounded to a half point, priced at
    `odds` on both sides. Real books set sharper lines, so real results will be
    worse. Bet the side whose model probability clears a threshold."""
    P = pd.read_parquet(PROC / "projections_all_games_oos.parquet")
    F = pd.read_parquet(PROC / "features_all.parquet",
                        columns=["season", "gameId", "pid"] + [f"{s}_{w}" for s in STATS for w in ("l10", "l5")])
    P = P.merge(F, on=["season", "gameId", "pid"], how="left")
    P = P[P.gp >= min_gp]
    frames = []
    for px in proxies:
        for s in STATS:
            src = P[f"{s}_{px}"] if px != "szn" else P[f"{s}_szn"]
            d = P[src.notna()].copy()
            d["stat"], d["line"], d["proxy"] = s, np.floor(src[src.notna()]) + 0.5, px
            frames.append(d)
    m = pd.concat(frames, ignore_index=True)
    m["over_odds"] = m["under_odds"] = odds
    m["p_mkt"] = 0.5
    out = []
    for px, d in m.groupby("proxy"):
        out.append(model_probs(d, P).assign(proxy=px))
    m = pd.concat(out, ignore_index=True)
    p_side = np.maximum(m.p_model, 1 - m.p_model)
    side_over = m.p_model >= 0.5
    win = np.where(side_over, m.result == 1, m.result == 0)
    m["p_side"], m["win"] = p_side, win
    m["profit"] = np.where(win, payout(odds), -1.0)
    rows = []
    for t in (BREAK_EVEN_120, 0.56, 0.58, 0.60, 0.65):
        k = m[m.p_side >= t]
        for (px, s), d in list(k.groupby(["proxy", "stat"])) + [((px, "all"), d) for px, d in k.groupby("proxy")]:
            rows.append({"proxy_line": px, "stat": s, "min_model_prob": round(t, 4), "bets": len(d),
                         "bets_per_season": len(d) / d.season.nunique(), "hit_rate": d.win.mean(), "roi": d.profit.mean(),
                         "avg_model_prob": d.p_side.mean()})
    seasons = (m[m.p_side >= 0.58].groupby(["proxy", "season"])
               .agg(bets=("win", "size"), hit_rate=("win", "mean"), roi=("profit", "mean")).reset_index())
    return pd.DataFrame(rows), seasons


# ---------------------------------------------------------------- moneyline movement (evaluation only)
ML_CACHE = ROOT / "data" / "raw" / "espn_odds"
TEAM_FIX = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}
PREF = ("DraftKings", "Caesars", "ESPN BET", "MGM", "consensus")


def _american(x):
    try:
        return float(str(x).replace("+", ""))
    except (TypeError, ValueError):
        return np.nan


def _ml_event(eid: str) -> dict | None:
    """Open/close moneylines for one ESPN event (cached). Evaluation only, never a model input."""
    import json
    import requests
    f = ML_CACHE / f"{eid}.json"
    if f.exists():
        return json.loads(f.read_text()) or None
    u = f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{eid}/competitions/{eid}/odds"
    for _ in range(3):
        try:
            j = requests.get(u, timeout=30, headers={"User-Agent": "Mozilla/5.0 nba-research"}).json()
            break
        except Exception:
            j = None
    if j is None:
        return None
    best = None
    for it in j.get("items", []):
        h, a = it.get("homeTeamOdds") or {}, it.get("awayTeamOdds") or {}
        ho, hc = (h.get("open") or {}).get("moneyLine"), (h.get("close") or {}).get("moneyLine")
        ao, ac = (a.get("open") or {}).get("moneyLine"), (a.get("close") or {}).get("moneyLine")
        if not (ho and hc and ao and ac):
            continue
        row = {"provider": it.get("provider", {}).get("name", ""),
               "home_open": _american(ho.get("american")), "home_close": _american(hc.get("american")),
               "away_open": _american(ao.get("american")), "away_close": _american(ac.get("american"))}
        rank = next((i for i, p in enumerate(PREF) if p.lower() in row["provider"].lower()), len(PREF))
        if best is None or rank < best[0]:
            best = (rank, row)
    out = best[1] if best else {}
    ML_CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out))
    return out or None


def fetch_moneylines(seasons=range(2019, 2026), workers=8) -> pd.DataFrame:
    """ESPN event -> (ET date, home, away) from cached summaries, plus open/close moneylines."""
    import gzip
    import json
    from concurrent.futures import ThreadPoolExecutor
    from nba.config import RAW
    ev = []
    for y in seasons:
        for p in sorted((RAW / "espn" / str(y)).glob("*.json.gz")):
            try:
                c = json.loads(gzip.decompress(p.read_bytes()))["header"]["competitions"][0]
            except Exception:
                continue
            side = {x["homeAway"]: TEAM_FIX.get(x["team"]["abbreviation"], x["team"]["abbreviation"])
                    for x in c["competitors"]}
            ev.append({"season": y, "eid": p.name.split(".")[0], "home": side.get("home"), "away": side.get("away"),
                       "date": pd.Timestamp(c["date"]).tz_convert("America/New_York").date()})
    E = pd.DataFrame(ev)
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(_ml_event, E.eid))
    M = pd.concat([E, pd.DataFrame([r or {} for r in res])], axis=1)
    M.to_parquet(PROC / "moneylines.parquet", index=False)
    return M


def _imp(a):
    a = np.asarray(a, float)
    return np.where(a < 0, -a / (-a + 100), 100 / (a + 100))


def moneyline_movement_backtest(min_p=0.70) -> dict:
    """Winner model at >= min_p (out-of-sample): does line movement toward / against our team change
    the hit rate or ROI? Placebo = movement shuffled within season."""
    M = pd.read_parquet(PROC / "moneylines.parquet").dropna(subset=["home_open", "home_close"])
    W = pd.read_parquet(PROC / "winners_backtest.parquet")
    W = W[W.season >= 2019].dropna(subset=["model"]).copy()
    W["date"] = pd.to_datetime(W.game_date).dt.date
    X = W.merge(M[["date", "home", "home_open", "home_close", "away_open", "away_close", "provider"]],
                left_on=["date", "home_team"], right_on=["date", "home"], how="inner")
    home_pick = X.model >= 0.5
    X["p"] = np.where(home_pick, X.model, 1 - X.model)
    X["won"] = np.where(home_pick, X.y == 1, X.y == 0)
    for t in ("open", "close"):
        h, a = _imp(X[f"home_{t}"]), _imp(X[f"away_{t}"])
        X[f"mkt_{t}"] = np.where(home_pick, h / (h + a), a / (h + a))      # no-vig, our team
        X[f"price_{t}"] = np.where(home_pick, X[f"home_{t}"], X[f"away_{t}"])
    X["move"] = X.mkt_close - X.mkt_open                                      # + = market moved toward us
    X["edge_close"] = X.p - X.mkt_close
    pay = lambda a: np.where(a < 0, 100 / -a, a / 100)
    X["u_close"] = np.where(X.won, pay(X.price_close), -1.0)
    X["u_open"] = np.where(X.won, pay(X.price_open), -1.0)
    S = X[X.p >= min_p].copy()
    S["mv"] = pd.cut(S.move, [-1, -0.02, 0.02, 1], labels=["moved against us", "flat (<2 pts)", "moved toward us"])
    agg = dict(n=("won", "size"), stated=("p", "mean"), market=("mkt_close", "mean"), won=("won", "mean"),
               roi_close=("u_close", "mean"), roi_open=("u_open", "mean"))
    out = {"n_matched": len(X), "overall": S.agg({"won": "mean", "p": "mean", "u_close": "mean", "u_open": "mean"}),
           "by_move": S.groupby("mv", observed=True).agg(**agg),
           "by_season": S.groupby(["season", "mv"], observed=True).agg(n=("won", "size"), won=("won", "mean"),
                                                                          roi_close=("u_close", "mean")),
           "by_edge": S.groupby(pd.cut(S.edge_close, [-1, -0.05, 0, 0.05, 1]), observed=True).agg(**agg)}
    rng = np.random.default_rng(5)
    plc = []
    for _ in range(50):
        sh = S.groupby("season").move.transform(lambda s: rng.permutation(s.to_numpy()))
        plc.append(S[sh > 0.02].u_close.mean() - S[sh < -0.02].u_close.mean())
    out["placebo_roi_gap_sd"] = float(np.std(plc))
    out["roi_gap_toward_minus_against"] = float(S[S.move > 0.02].u_close.mean() - S[S.move < -0.02].u_close.mean())
    return out
