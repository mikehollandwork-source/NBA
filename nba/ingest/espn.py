"""Official box scores from ESPN's public site API (ground truth for outcomes).

IMPORTANT: ESPN's summary payload also contains betting odds ("pickcenter",
"odds", "againstTheSpread"). We deliberately keep ONLY the box score and game
header so no betting information enters the system.
"""
import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd
import requests

from nba.config import RAW, PROC, LIVE_SEASON

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
KEEP_KEYS = ("boxscore", "header")  # never store odds/pickcenter/againstTheSpread
S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0 nba-projection-research"


def _get(url, params=None, tries=7):
    """Retry with backoff (~2 min total): ESPN sometimes refuses GitHub runners for a minute or two."""
    last = None
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}"
        except requests.RequestException as ex:
            last = type(ex).__name__
        time.sleep(min(2 ** i, 30))
    raise RuntimeError(f"failed: {url} {params} ({last})")


def _scoreboard(day: str) -> dict:
    """Scoreboard for YYYYMMDD. Finished past days are cached on disk (they never change), so a
    flaky ESPN can't stop the live run over a day that's already final."""
    f = RAW / "espn" / str(LIVE_SEASON) / "scoreboard" / f"{day}.json"  # inside the Actions ESPN cache path
    if f.exists():
        return json.loads(f.read_text())
    b = _get(f"{BASE}/scoreboard", {"dates": day})
    ev = b.get("events", [])
    if day < (date.today() - timedelta(1)).strftime("%Y%m%d") and ev and all(
            e["status"]["type"]["completed"] or e["status"]["type"].get("name") == "STATUS_POSTPONED" for e in ev):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"events": [{k: e[k] for k in ("id", "date", "name", "season", "status")} for e in ev]}))
    return b


def regular_season_event_ids(start: str, end: str, season: int | None = None) -> list[dict]:
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    days = [(d0 + timedelta(n)).strftime("%Y%m%d") for n in range((d1 - d0).days + 1)]
    with ThreadPoolExecutor(4) as ex:
        boards = list(ex.map(_scoreboard, days))
    events = []
    for b in boards:
        for e in b.get("events", []):
            if (e["season"]["type"] == 2 and e["status"]["type"]["completed"]
                    and (season is None or e["season"]["year"] == season + 1)):
                events.append({"event_id": e["id"], "date": e["date"][:10], "name": e["name"]})
    return events


def fetch_summary(event_id: str, season: int) -> dict:
    cache = RAW / "espn" / str(season) / f"{event_id}.json.gz"
    if cache.exists():
        return json.loads(gzip.decompress(cache.read_bytes()))
    d = _get(f"{BASE}/summary", {"event": event_id})
    d = {k: d[k] for k in KEEP_KEYS if k in d}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(gzip.compress(json.dumps(d).encode()))
    return d


def _split(v):
    a, b = v.split("-")
    return int(a), int(b)


def parse_box(d: dict) -> list[dict]:
    comp = d["header"]["competitions"][0]
    teams = {c["team"]["id"]: c for c in comp["competitors"]}
    rows = []
    for t in d["boxscore"]["players"]:
        tid = t["team"]["id"]
        opp = next(c for k, c in teams.items() if k != tid)
        for blk in t["statistics"]:
            keys = blk["keys"]
            for a in blk["athletes"]:
                base = {"event_id": d["header"]["id"], "game_date": comp["date"][:10],
                        "team": t["team"]["abbreviation"], "opp": opp["team"]["abbreviation"],
                        "home": int(teams[tid]["homeAway"] == "home"),
                        "espn_id": a["athlete"].get("id", "x_" + a["athlete"].get("shortName", "")),
                        "player": a["athlete"].get("displayName", a["athlete"].get("shortName", "")),
                        "starter": int(a.get("starter", False)),
                        "pos": a["athlete"].get("position", {}).get("abbreviation", ""),
                        "dnp": int(a.get("didNotPlay", False) or not a.get("stats")
                                   or not str(a["stats"][0]).isdigit() or a["stats"][0] == "0"),
                        "dnp_reason": a.get("reason", "")}
                if not base["dnp"]:
                    s = dict(zip(keys, a["stats"]))
                    fgm, fga = _split(s["fieldGoalsMade-fieldGoalsAttempted"])
                    tpm, tpa = _split(s["threePointFieldGoalsMade-threePointFieldGoalsAttempted"])
                    ftm, fta = _split(s["freeThrowsMade-freeThrowsAttempted"])
                    base.update(min=float(s["minutes"]), pts=int(s["points"]), fgm=fgm, fga=fga,
                                fg3m=tpm, fg3a=tpa, ftm=ftm, fta=fta, reb=int(s["rebounds"]),
                                ast=int(s["assists"]), tov=int(s["turnovers"]), stl=int(s["steals"]),
                                blk=int(s["blocks"]), oreb=int(s["offensiveRebounds"]),
                                dreb=int(s["defensiveRebounds"]), pf=int(s["fouls"]),
                                plus_minus=int(str(s["plusMinus"]).replace("+", "") or 0))
                rows.append(base)
    return rows


def build_boxscores(season: int) -> pd.DataFrame:
    from nba.config import SEASON_WINDOWS
    out = PROC / f"espn_box_{season}.parquet"
    if out.exists():
        return pd.read_parquet(out)
    events = regular_season_event_ids(*SEASON_WINDOWS[season], season=season)
    print(f"{season}: {len(events)} completed regular-season games (ESPN)", flush=True)
    with ThreadPoolExecutor(8) as ex:
        sums = list(ex.map(lambda e: fetch_summary(e["event_id"], season), events))
    rows = [r for s in sums for r in parse_box(s)]
    df = pd.DataFrame(rows)
    df["season"] = season
    PROC.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df
