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
    for c in ("dk_ml_home", "dk_ml_away", "mkt_home", "value_team", "value_gap", "value_price"):
        W[c] = np.nan if c != "value_team" else None
    for i, q in W.iterrows():
        h, a = fetch(q.event_id)
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
