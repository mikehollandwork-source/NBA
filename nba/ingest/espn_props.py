"""DraftKings player props from ESPN's public core API (provider 100 = DraftKings). DISPLAY ONLY:
shown next to our picks so the board matches what the sportsbook offers; never a model input.

Each player/stat/line comes as two prices; the first is the OVER (verified 2026-10-24 on 1,132
lines from Oct 22-23: when the first price was the favorite players went over 49% vs 42%).
Props appear the morning of game day; ESPN keeps only recent games (no historical archive).
"""
import requests
import pandas as pd

URL = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{e}/competitions/{e}/odds/100/propBets"
TYPES = {"Total Points": "pts", "Total Rebounds": "reb", "Total Assists": "ast",
         "Total Points, Rebounds, and Assists": "pra", "Total Points and Rebounds": "pr",
         "Total Points and Assists": "pa", "Total Assists and Rebounds": "ra"}


def fetch(event_id) -> pd.DataFrame:
    """-> pid, stat, line, over, under (American odds as int), updated."""
    try:
        r = requests.get(URL.format(e=event_id), params={"limit": 1000}, timeout=30)
        items = r.json().get("items", []) if r.ok else []
    except Exception as ex:
        print("ESPN props unavailable:", ex, flush=True)
        return pd.DataFrame()
    grp = {}
    for x in items:
        stat = TYPES.get(x.get("type", {}).get("name"))
        if stat is None or "athlete" not in x:
            continue
        try:
            pid = x["athlete"]["$ref"].split("/athletes/")[1].split("?")[0]
            line = float(x["odds"]["total"]["value"])
            price = int(float(x["odds"]["american"]["value"]))
        except (KeyError, ValueError, TypeError):
            continue
        grp.setdefault((pid, stat, line), []).append((price, x.get("lastUpdated")))
    rows = [{"pid": pid, "stat": s, "line": line, "over": v[0][0], "under": v[1][0], "updated": v[0][1]}
            for (pid, s, line), v in grp.items() if len(v) == 2]
    return pd.DataFrame(rows)


def attach(r: pd.DataFrame, G: pd.DataFrame, writable: set) -> pd.DataFrame:
    """Add dk_line / dk_price (for the lean's side) / dk_updated to rows of games still writable."""
    r = r.copy()
    for c in ("dk_line", "dk_price"):
        if c not in r:
            r[c] = float("nan")
    if "dk_updated" not in r:
        r["dk_updated"] = pd.Series([None] * len(r), index=r.index, dtype=object)
    for _, g in G[G.event_id.isin(writable)].iterrows():
        dk = fetch(g.event_id)
        if dk.empty:
            continue
        rows = r.index[(r.gameId == int(g.event_id)) & r.lean_stat.apply(lambda v: isinstance(v, str))]
        for i in rows:
            q = r.loc[i]
            m = dk[(dk.pid == str(q.pid)) & (dk.stat == q.lean_stat)]
            if m.empty:
                continue
            # main line = the one with prices closest to even; show it even if it differs from ours
            m = m.assign(gap=(m.over.abs() - 100).abs() + (m.under.abs() - 100).abs()).sort_values("gap")
            b = m.iloc[0]
            r.loc[i, "dk_line"] = b.line
            r.loc[i, "dk_price"] = b.over if q.lean_side == "over" else b.under
            r.loc[i, "dk_updated"] = b.updated
    return r


def describe(q) -> str:
    """'DK u4.5 -140' for the lean's side, or 'DK: no line'."""
    line, price = q.get("dk_line"), q.get("dk_price")
    if line is None or pd.isna(line) or pd.isna(price):
        return "DK: no line yet"
    return f"DK {'o' if q.lean_side == 'over' else 'u'}{float(line):g} {int(price):+d}"
