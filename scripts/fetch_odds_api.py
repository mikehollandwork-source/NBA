"""Fetch historical NBA player-prop closing lines from The Odds API (paid plans
only; props from May 2023) into data/lines/odds_api_<date>.csv.

    ODDS_API_KEY=... python scripts/fetch_odds_api.py 2024-10-22 2025-04-13

Snapshot is taken 5 minutes before each game's scheduled tip (≈ closing line).
UNTESTED against the live API (no key in this environment) — check the first
day's output before a full run; each call costs API credits.
"""
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nba.config import ROOT  # noqa: E402

KEY = os.environ.get("ODDS_API_KEY")
BASE = "https://api.the-odds-api.com/v4/historical/sports/basketball_nba"
MARKETS = {"player_points": "pts", "player_rebounds": "reb", "player_assists": "ast",
           "player_points_rebounds_assists": "pra", "player_points_rebounds": "pr",
           "player_points_assists": "pa", "player_rebounds_assists": "ra"}


def day(d: date):
    snap = datetime(d.year, d.month, d.day, 16, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    ev = requests.get(f"{BASE}/events", params={"apiKey": KEY, "date": snap}, timeout=30).json()
    rows = []
    for e in ev.get("data", []):
        tip = datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
        if tip.date() not in (d, d + timedelta(days=1)):
            continue
        close = (tip - timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
        r = requests.get(f"{BASE}/events/{e['id']}/odds", params={
            "apiKey": KEY, "date": close, "regions": "us", "markets": ",".join(MARKETS), "oddsFormat": "american"},
            timeout=30).json()
        for b in r.get("data", {}).get("bookmakers", []):
            for mk in b.get("markets", []):
                by = {}
                for o in mk.get("outcomes", []):
                    by.setdefault((o.get("description"), o.get("point")), {})[o["name"].lower()] = o["price"]
                for (player, line), px in by.items():
                    if "over" in px and "under" in px:
                        rows.append({"date": d.isoformat(), "player": player, "stat": MARKETS[mk["key"]], "line": line,
                                     "over_odds": px["over"], "under_odds": px["under"], "book": b["key"]})
        time.sleep(0.2)
    return rows


if __name__ == "__main__":
    if not KEY:
        sys.exit("set ODDS_API_KEY")
    d0, d1 = (date.fromisoformat(a) for a in sys.argv[1:3])
    out = ROOT / "data" / "lines"
    out.mkdir(parents=True, exist_ok=True)
    d = d0
    while d <= d1:
        rows = day(d)
        if rows:
            pd.DataFrame(rows).to_csv(out / f"odds_api_{d}.csv", index=False)
        print(d, len(rows), flush=True)
        d += timedelta(days=1)
