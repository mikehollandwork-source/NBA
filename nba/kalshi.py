"""Kalshi player-prop prices -> model edges, picks and grading.

Prices are used ONLY to choose/grade picks — never as model inputs.

Kalshi lists ladders ("Player: 25+ points" = YES if he scores 25 or more).
For every rung we compare our probability P(stat >= k) (same simulation as the
board's ranges) with the price you'd actually pay:
    YES cost = yes_ask + fee        NO cost = (1 - yes_bid) + fee
    fee per contract = 0.07 * price * (1 - price)   (Kalshi quadratic fee)
Return per $1 staked if it wins = (1 - cost) / cost; loss = -1.
A pick needs an expected return >= MIN_ROI after fees; only the best rung per
player and stat is kept (ladder rungs are near-duplicates). Inactive players
are settled at a "fair price" by Kalshi -> counted as void.
"""
import time
import unicodedata

import numpy as np
import pandas as pd
import requests

from nba.analysis import season_sim as ss
from nba.config import PROC

B = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = {"pts": "KXNBAPTS", "reb": "KXNBAREB", "ast": "KXNBAAST", "pra": "KXNBAPRA"}  # 3PM: no calibrated spread yet
FEE = 0.07
MIN_ROI = 0.05      # starting rule, not backtested
N_SIMS = 20000
S = requests.Session()


def _get(path, pause=0.15, **params):
    for t in range(10):
        r = S.get(f"{B}/{path}", params=params, timeout=30)
        if r.status_code == 200:
            time.sleep(pause)
            return r.json()
        time.sleep(min(60, 2 ** t))
    raise RuntimeError(f"kalshi {path} {r.status_code} {r.text[:200]}")


def key(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    for suf in (" jr", " sr", " iii", " ii", " iv"):
        s = s.replace(suf, "")
    return "".join(ch for ch in s if ch.isalpha())


def date_code(d) -> str:
    return pd.Timestamp(d).strftime("%y%b%d").upper()


def markets(d, games: pd.DataFrame) -> pd.DataFrame:
    """All rungs for the date's games (games: away/home nba.com codes, event_id, tip_utc)."""
    rows = []
    for _, g in games.iterrows():
        for stat, ser in SERIES.items():
            ev = f"{ser}-{date_code(d)}{g.away}{g.home}"
            for m in _get("markets", event_ticker=ev, limit=1000).get("markets", []):
                rows.append({"gameId": int(g.event_id), "game": f"{g.away}@{g.home}", "tip_utc": g.tip_utc,
                             "stat": stat, "series": ser, "ticker": m["ticker"], "title": m["title"],
                             "player_key": key(m["title"].split(":")[0]), "k": int(round(float(m["floor_strike"]) + 0.5)),
                             "yes_bid": float(m.get("yes_bid_dollars") or 0), "yes_ask": float(m.get("yes_ask_dollars") or 0),
                             "volume": float(m.get("volume_fp") or 0), "status": m.get("status")})
    return pd.DataFrame(rows)


def pre_tip_quotes(mk: pd.DataFrame, cache=None) -> pd.DataFrame:
    """For finished games: last 1-minute candle quote before tip (bid/ask close).
    Slow on purpose (Kalshi rate limits); results cached per ticker so reruns resume."""
    import json
    cache = cache or PROC.parent / "live" / "kalshi_candles_cache.json"
    seen = json.loads(cache.read_text()) if cache.exists() else {}
    mk = mk.copy()
    for n, (i, r) in enumerate(mk.iterrows()):
        tip = int(pd.Timestamp(r.tip_utc).timestamp())
        if r.ticker in seen:
            c = seen[r.ticker]
        else:
            c = _get(f"series/{r.series}/markets/{r.ticker}/candlesticks", pause=0.6, start_ts=tip - 3 * 3600,
                     end_ts=tip, period_interval=1).get("candlesticks", [])
            c = [{"yes_bid": cd.get("yes_bid", {}), "yes_ask": cd.get("yes_ask", {})} for cd in c]
            seen[r.ticker] = c
            if n % 25 == 0:
                cache.write_text(json.dumps(seen))
        bid = ask = np.nan
        for cd in c:  # chronological; keep the last valid quote before tip
            yb, ya = cd.get("yes_bid", {}), cd.get("yes_ask", {})
            b = yb.get("close_dollars", yb.get("close"))  # live candles use *_dollars, historical use plain keys
            a = ya.get("close_dollars", ya.get("close"))
            if b is not None and a is not None:
                bid, ask = float(b), float(a)
        mk.at[i, "yes_bid"], mk.at[i, "yes_ask"] = bid, ask
    cache.write_text(json.dumps(seen))
    return mk


def model_sims(proj: pd.DataFrame) -> dict:
    """pid -> dict(stat -> simulated outcomes), from locked/live projections."""
    P = pd.read_parquet(PROC / "projections_all_games_oos.parquet")
    var, C = ss.fit_params(P, -1)
    out = {}
    for _, r in proj.iterrows():
        mu = np.array([[r.proj_pts, r.proj_reb, r.proj_ast]], float)
        if not np.isfinite(mu).all():
            continue
        x = ss.draw(np.clip(mu, 0.05, None), var, C, n_sims=N_SIMS)[:, 0, :]
        out[str(r.pid)] = {"pts": x[:, 0], "reb": x[:, 1], "ast": x[:, 2], "pra": x.sum(1)}
    return out


def fee(p):
    return FEE * p * (1 - p)


def edges(mk: pd.DataFrame, proj: pd.DataFrame) -> pd.DataFrame:
    """Every priced rung with model probability and expected return of both sides."""
    proj = proj.copy()
    proj["player_key"] = proj.player.map(key)
    m = mk.merge(proj[["pid", "player", "team", "player_key", "proj_pts", "proj_reb", "proj_ast", "proj_pra"]],
                 left_on=["player_key"], right_on=["player_key"], how="inner")
    m = m[m.ticker.str.split("-").str[2].str[:3] == m.team]  # same team
    sims = model_sims(proj[proj.pid.astype(str).isin(m.pid.astype(str))])
    m["p_model"] = [float((sims[str(p)][s] >= k).mean()) if str(p) in sims else np.nan
                    for p, s, k in zip(m.pid, m.stat, m.k)]
    ok = (m.yes_bid > 0) & (m.yes_ask > 0) & (m.yes_ask < 1) & m.p_model.notna()
    m = m[ok].copy()
    m["cost_yes"] = m.yes_ask + fee(m.yes_ask)
    m["cost_no"] = (1 - m.yes_bid) + fee(1 - m.yes_bid)
    m["roi_yes"] = m.p_model / m.cost_yes - 1
    m["roi_no"] = (1 - m.p_model) / m.cost_no - 1
    m["side"] = np.where(m.roi_yes >= m.roi_no, "YES", "NO")
    m["price"] = np.where(m.side == "YES", m.yes_ask, 1 - m.yes_bid)
    m["cost"] = np.where(m.side == "YES", m.cost_yes, m.cost_no)
    m["p_side"] = np.where(m.side == "YES", m.p_model, 1 - m.p_model)
    m["exp_roi"] = np.maximum(m.roi_yes, m.roi_no)
    return m


def picks(e: pd.DataFrame, min_roi: float = MIN_ROI) -> pd.DataFrame:
    x = e[e.exp_roi >= min_roi].sort_values("exp_roi", ascending=False)
    return x.drop_duplicates(["pid", "stat"]).reset_index(drop=True)


def paper(e: pd.DataFrame, min_roi: float = MIN_ROI) -> pd.DataFrame:
    """Keep every priced rung (for model-vs-market scoring) and flag the would-be
    picks (best rung per player-stat with exp. return >= min_roi). Paper only."""
    e = e.copy()
    e["market_p"] = (e.yes_bid + e.yes_ask) / 2
    best = e[e.exp_roi >= min_roi].sort_values("exp_roi", ascending=False).drop_duplicates(["pid", "stat"]).index
    e["paper_pick"] = e.index.isin(best)
    return e


def describe(q) -> str:
    rung = f"{q.k}+" if q.side == "YES" else f"under {q.k}"
    mkt = (1 - q.market_p) if q.side == "NO" and "market_p" in q else q.get("market_p", float("nan"))
    return f"{q.stat.upper()} {rung} @ {q.price * 100:.0f}¢ · model {q.p_side:.0%} vs market {mkt:.0%} · exp {q.exp_roi:+.0%}"


def grade(pk: pd.DataFrame, box: pd.DataFrame) -> pd.DataFrame:
    """box: pid, gameId, dnp, pts, reb, ast, pra."""
    g = pk.merge(box[["pid", "gameId", "dnp", "pts", "reb", "ast", "pra"]].assign(pid=lambda b: b.pid.astype(str)),
                 left_on=[pk.pid.astype(str), "gameId"], right_on=["pid", "gameId"], how="left", suffixes=("", "_b"))
    act = np.array([r[r.stat] if pd.notna(r.get(r.stat)) else np.nan for _, r in g.iterrows()], float)
    hit = np.where(g.side == "YES", act >= g.k, act < g.k)
    void = np.isnan(act) | (g.dnp == 1)
    g["actual"] = act
    g["result"] = np.where(void, "void", np.where(hit, "win", "loss"))
    g["units"] = np.where(void, 0.0, np.where(hit, (1 - g.cost) / g.cost, -1.0))
    return g
