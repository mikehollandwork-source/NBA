"""DraftKings moneylines from ESPN's core odds API — DISPLAY / PAPER TRACKING ONLY, never a model input.

📈 Value (user 2026-11-12, paper only): a team our winner model rates 10+ points above DraftKings' no-vig
win % (VALUE_GAP). Backtest 2023-25 closing lines (reports/phase2/moneyline_movement.md): 10+ higher +6.5% ROI
(860 bets; 2023 -2%, 2024 +8%, 2025 +13%); 8+ higher +5.2% vs placebo -4.9% (SD 2.9). Every smaller gap lost.
"""
import numpy as np
import pandas as pd
import requests

URL = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{e}/competitions/{e}/odds"
VALUE_GAP = 0.10
# 📏 ATS paper tracking (user 2026-11-12): our projected margin beats DraftKings' spread by >= ATS_EDGE points.
# Backtest 2023-25 closing spreads: 6+ -> 62% cover, +19% (302); without the injury input still 57.5%, +9.7% (464),
# positive all 3 seasons; every edge under 6 lost.
ATS_EDGE = 6.0


def _item(event_id):
    try:
        j = requests.get(URL.format(e=event_id), timeout=30, headers={"User-Agent": "Mozilla/5.0 nba-research"}).json()
    except Exception:
        return None
    return next((i for i in j.get("items", []) if "draftkings" in i.get("provider", {}).get("name", "").lower()), None)


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def fetch_all(event_id) -> dict:
    """DraftKings moneylines, home spread and spread prices (nan when missing)."""
    it = _item(event_id)
    if not it:
        return {}
    h, a = it.get("homeTeamOdds") or {}, it.get("awayTeamOdds") or {}
    return {"home_ml": _f(h.get("moneyLine")), "away_ml": _f(a.get("moneyLine")), "home_spread": _f(it.get("spread")),
            "home_spread_price": _f(h.get("spreadOdds")), "away_spread_price": _f(a.get("spreadOdds"))}


def fetch(event_id) -> tuple:
    """(home_ml, away_ml) from DraftKings, or (nan, nan)."""
    try:
        j = requests.get(URL.format(e=event_id), timeout=30, headers={"User-Agent": "Mozilla/5.0 nba-research"}).json()
    except Exception:
        return np.nan, np.nan
    it = next((i for i in j.get("items", []) if "draftkings" in i.get("provider", {}).get("name", "").lower()), None)
    if not it:
        return np.nan, np.nan
    h, a = (it.get("homeTeamOdds") or {}).get("moneyLine"), (it.get("awayTeamOdds") or {}).get("moneyLine")
    try:
        return float(h), float(a)
    except (TypeError, ValueError):
        return np.nan, np.nan


def _imp(a):
    return -a / (-a + 100) if a < 0 else 100 / (a + 100)


def attach(W: pd.DataFrame) -> pd.DataFrame:
    """Add DK moneylines, the market's no-vig home win %, and the 📈 value team (if any) to winner rows."""
    W = W.copy()
    for c in ("dk_ml_home", "dk_ml_away", "mkt_home", "value_team", "value_gap", "value_price",
              "dk_spread_home", "ats_team", "ats_line", "ats_price", "ats_edge"):
        W[c] = None if c in ("value_team", "ats_team") else np.nan
    for i, q in W.iterrows():
        o = fetch_all(q.event_id)
        sp = o.get("home_spread", np.nan)
        if not np.isnan(sp):
            W.loc[i, "dk_spread_home"] = sp
            edge = float(q.proj_margin_home) + sp                  # > 0: we like home against the number
            if abs(edge) >= ATS_EDGE:
                home = edge > 0
                price = o.get("home_spread_price" if home else "away_spread_price", np.nan)
                W.loc[i, ["ats_team", "ats_line", "ats_price", "ats_edge"]] = [
                    q.home if home else q.away, sp if home else -sp,
                    price if not np.isnan(price) and abs(price) >= 100 else -110.0, round(abs(edge), 1)]
        h, a = o.get("home_ml", np.nan), o.get("away_ml", np.nan)
        if np.isnan(h) or np.isnan(a) or abs(h) < 100 or abs(a) < 100:
            continue
        mh = _imp(h) / (_imp(h) + _imp(a))
        gh = float(q.p_home) - mh
        W.loc[i, ["dk_ml_home", "dk_ml_away", "mkt_home"]] = [h, a, round(mh, 3)]
        if gh >= VALUE_GAP:
            W.loc[i, ["value_team", "value_gap", "value_price"]] = [q.home, round(gh, 3), h]
        elif -gh >= VALUE_GAP:
            W.loc[i, ["value_team", "value_gap", "value_price"]] = [q.away, round(-gh, 3), a]
    return W
