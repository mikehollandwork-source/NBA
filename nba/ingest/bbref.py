"""Basketball-Reference box scores — fallback for games where ESPN's box is
blank (163 games each in 2016-17 and 2017-18). Polite: one request per 3.5 s."""
import gzip
import io
import subprocess
import time

import pandas as pd

from nba.config import RAW

TO_BBREF = {"BKN": "BRK", "CHA": "CHO", "PHX": "PHO"}


def _html(date: pd.Timestamp, home: str) -> str:
    code = f"{date:%Y%m%d}0{TO_BBREF.get(home, home)}"
    cache = RAW / "bbref" / f"{code}.html.gz"
    if cache.exists():
        return gzip.decompress(cache.read_bytes()).decode()
    # curl, not requests: Cloudflare rejects Python's TLS fingerprint here.
    url = f"https://www.basketball-reference.com/boxscores/{code}.html"
    for i in range(4):
        r = subprocess.run(["curl", "-s", "-A", "Mozilla/5.0", "-w", "\n%{http_code}", url], capture_output=True, text=True)
        body, _, status = r.stdout.rpartition("\n")
        time.sleep(3.5)
        if status == "200":
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(gzip.compress(body.encode()))
            return body
        if status == "429":
            time.sleep(60)
    raise RuntimeError(f"bbref failed {code} {status}")


def _mp(v):
    try:
        m, s = str(v).split(":")
        return int(m) + int(s) / 60
    except ValueError:
        return 0.0


def box(date: pd.Timestamp, home: str, away: str) -> pd.DataFrame:
    h = _html(date, home).replace("<!--", "").replace("-->", "")
    rows = []
    for team, opp, is_home in ((home, away, 1), (away, home, 0)):
        t = pd.read_html(io.StringIO(h), attrs={"id": f"box-{TO_BBREF.get(team, team)}-game-basic"}, header=1)[0]
        t = t.rename(columns={"Starters": "player"})
        t["starter"] = (t.index < 5).astype(int)
        t = t[~t.player.isin(["Reserves", "Team Totals"])]
        t["min"] = t.MP.map(_mp)
        t = t[t["min"] > 0]
        num = lambda c: pd.to_numeric(t[c], errors="coerce").fillna(0).astype(int)
        rows.append(pd.DataFrame({
            "player": t.player, "team": team, "opp": opp, "home": is_home, "starter": t.starter, "min": t["min"],
            "pts": num("PTS"), "fgm": num("FG"), "fga": num("FGA"), "fg3m": num("3P"), "fg3a": num("3PA"),
            "ftm": num("FT"), "fta": num("FTA"), "oreb": num("ORB"), "dreb": num("DRB"), "reb": num("TRB"),
            "ast": num("AST"), "stl": num("STL"), "blk": num("BLK"), "tov": num("TOV"), "pf": num("PF"),
            "plus_minus": pd.to_numeric(t["+/-"].astype(str).str.replace("+", ""), errors="coerce").fillna(0).astype(int)}))
    return pd.concat(rows, ignore_index=True)
