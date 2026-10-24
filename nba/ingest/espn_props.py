"""DraftKings player props from ESPN's public core API (provider 100 = DraftKings). Never a model input.
User rule (2026-10-24): a 70%+ lean is a ✅ PICK only if DraftKings' main line is the same as ours or
better (under: DK line >= ours; over: DK line <= ours); it is graded at DK's line and odds frozen at the
lock. Past days are backfilled from DK's closing line (ESPN keeps recent games only).

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
    items, page, pages = [], 1, 1
    try:
        while page <= pages:  # ESPN pages at 1,000 items; busy games have 1,200+ (page 2 was being missed)
            r = requests.get(URL.format(e=event_id), params={"limit": 1000, "page": page}, timeout=30)
            if not r.ok:
                break
            j = r.json()
            items += j.get("items", [])
            pages = int(j.get("pageCount", 1) or 1)
            page += 1
    except Exception as ex:
        print("ESPN props unavailable:", ex, flush=True)
        if not items:
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
    r["dk_ok"] = ok_mask(r)
    return r


def ok_mask(r: pd.DataFrame) -> pd.Series:
    """DK line is ours or better for the lean's side."""
    if "dk_line" not in r:
        return pd.Series(False, index=r.index)
    line = pd.to_numeric(r.dk_line, errors="coerce")
    ok = ((r.lean_side == "under") & (line >= r.lean_ref)) | ((r.lean_side == "over") & (line <= r.lean_ref))
    return ok & line.notna() & pd.to_numeric(r.get("dk_price"), errors="coerce").notna()


def payout(american) -> float:
    a = float(american)
    return 100 / -a if a < 0 else a / 100


def log_snapshot(r: pd.DataFrame, d, now, strong_min: float):
    """Append DK line/price for every 70%+ lean at this run (morning-vs-lock comparison later)."""
    from nba.config import ROOT
    x = r[(r.lean_prob_cal.round(2) >= strong_min) & r.get("dk_line", pd.Series(dtype=float)).notna()]
    if x.empty:
        return
    out = ROOT / "reports" / "live" / "dk_lines" / f"{d}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    snap = x[["gameId", "pid", "player", "lean_stat", "lean_side", "lean_ref", "lean_prob_cal", "dk_line", "dk_price"]].copy()
    snap.insert(0, "run_utc", pd.Timestamp(now).isoformat(timespec="seconds"))
    snap.to_csv(out, mode="a", header=not out.exists(), index=False)


def backfill(day: str, lock_files) -> pd.DataFrame:
    """DK CLOSING line/price for each lean in lock files that predate the DK columns."""
    rows = []
    for f in lock_files:
        L = pd.read_csv(f, dtype={"pid": str})
        if "dk_line" in L or "gameId" not in L or "lean_stat" not in L:
            continue
        L = L[L.lean_stat.notna()]
        if L.empty:
            continue
        dk = fetch(int(L.gameId.iloc[0]))
        if dk.empty:
            continue
        for _, q in L.iterrows():
            m = dk[(dk.pid == str(q.pid)) & (dk.stat == q.lean_stat)]
            if m.empty:
                continue
            m = m.assign(gap=(m.over.abs() - 100).abs() + (m.under.abs() - 100).abs()).sort_values("gap")
            b = m.iloc[0]
            rows.append({"date": day, "gameId": int(q.gameId), "pid": str(q.pid), "stat": q.lean_stat,
                         "dk_line": b.line, "dk_price": b.over if q.lean_side == "over" else b.under,
                         "source": "DK closing (backfill)"})
    return pd.DataFrame(rows)


def describe(q) -> str:
    """'DK u4.5 -140' for the lean's side, or 'DK: no line'."""
    line, price = q.get("dk_line"), q.get("dk_price")
    if line is None or pd.isna(line) or pd.isna(price):
        return "DK: no line yet"
    return f"DK {'o' if q.lean_side == 'over' else 'u'}{float(line):g} {int(price):+d}"
