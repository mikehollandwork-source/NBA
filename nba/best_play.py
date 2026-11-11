"""⭐ Play of the day: the single best bettable lean (user request 2026-11-11; replaces 🔥 On fire,
which lost 9% over 70 bets at DraftKings' lines).

Rule (chosen on Oct 22 - Nov 10 live data, 19 days, see reports/phase2/best_play.md):
  among leans DraftKings offers at our number or better (espn_props.ok_mask: line, 20-pt disagreement cap,
  3+ of 4 confluence checks), with 20+ projected minutes, an active player, a game that has not tipped,
  calibrated probability >= MIN_PROB and a price no worse than MIN_PRICE, take the highest expected
  return at DraftKings' odds:  EV = prob * (1 + payout) - 1.
It is a LEAN, not a 70% board pick. The pick is saved when posted (reports/live/best_play/<date>.json) and
graded at the line and odds shown then -> reports/live/best_play_record.csv. Lines never feed the model.
"""
import json

import numpy as np
import pandas as pd

from nba.config import ROOT

MIN_PROB = 0.60
MIN_PRICE = -160
MIN_MINUTES = 20
DIR = ROOT / "reports" / "live" / "best_play"
LEDGER = ROOT / "reports" / "live" / "best_play_record.csv"
BAD_STATUS = {"out", "doubtful", "questionable", "day-to-day"}
COMBOS = {"pra": ["pts", "reb", "ast"], "pr": ["pts", "reb"], "pa": ["pts", "ast"], "ra": ["reb", "ast"]}


def payout(american) -> np.ndarray:
    a = np.asarray(american, float)
    return np.where(a < 0, 100 / -a, a / 100)


def candidates(r: pd.DataFrame, G: pd.DataFrame | None = None, now=None) -> pd.DataFrame:
    """Every lean that passes the rule, best first."""
    from nba.ingest import espn_props
    x = r[r.lean_stat.apply(lambda v: isinstance(v, str))].copy()
    if x.empty or "dk_line" not in x:
        return x.iloc[0:0]
    if "projected" in x:
        x = x[x.projected.astype(str).str.lower() == "true"]
    if "status" in x:
        x = x[~x.status.fillna("").str.lower().isin(BAD_STATUS)]
    if G is not None and len(G):  # only games that have not tipped
        now = now or pd.Timestamp.now(tz="UTC")
        open_ids = set(G[pd.to_datetime(G.tip_utc, utc=True) > now].event_id.astype(int))
        x = x[x.gameId.astype(int).isin(open_ids)]
    x = x[(x.proj_min >= MIN_MINUTES) & espn_props.ok_mask(x)]
    price = pd.to_numeric(x.dk_price, errors="coerce")
    x = x[(x.lean_prob_cal >= MIN_PROB) & (price >= MIN_PRICE)].copy()
    x["ev"] = x.lean_prob_cal * (1 + payout(x.dk_price)) - 1
    return x.sort_values(["ev", "lean_prob_cal"], ascending=False)


def pick(r: pd.DataFrame, G: pd.DataFrame | None = None, now=None) -> dict | None:
    c = candidates(r, G, now)
    if c.empty:
        return None
    q = c.iloc[0]
    return {"gameId": int(q.gameId), "pid": str(q.pid), "player": q.player, "team": q.team, "opp": q.opp,
            "stat": q.lean_stat, "side": q.lean_side, "line": float(q.dk_line), "price": int(q.dk_price),
            "prob": round(float(q.lean_prob_cal), 3), "proj": round(float(q.get(f"proj_{q.lean_stat}", np.nan)), 1),
            "ev": round(float(q.ev), 3), "conf": None if pd.isna(q.get("conf")) else int(q.conf)}


def save(d, p: dict, now=None) -> None:
    """Freeze the posted play (first post of the day wins; later runs never overwrite it)."""
    f = DIR / f"{d}.json"
    if f.exists():
        return
    DIR.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({**p, "date": str(d),
                             "posted_utc": (now or pd.Timestamp.now(tz="UTC")).isoformat(timespec="seconds")}))


def load(d) -> dict | None:
    f = DIR / f"{d}.json"
    return json.loads(f.read_text()) if f.exists() else None


def grade() -> pd.DataFrame:
    """Grade every saved play at the line and odds it was posted with."""
    from nba import record
    rows = []
    for f in sorted(DIR.glob("*.json")) if DIR.exists() else []:
        p = json.loads(f.read_text())
        box = record._box(p["date"])
        b = box[(box.pid == str(p["pid"])) & (box.gameId == int(p["gameId"]))] if len(box) else box
        if b.empty:
            continue  # game not final yet
        b = b.iloc[0]
        actual = float(sum(b[c] for c in COMBOS.get(p["stat"], [p["stat"]])))
        if b.get("dnp", 0) == 1:
            res, u = "void", 0.0
        elif actual == p["line"]:
            res, u = "push", 0.0
        else:
            win = actual > p["line"] if p["side"] == "over" else actual < p["line"]
            res, u = ("win", float(payout(p["price"]))) if win else ("loss", -1.0)
        rows.append({**{k: p[k] for k in ("date", "player", "team", "opp", "stat", "side", "line", "price", "prob")},
                     "actual": actual, "result": res, "units": round(u, 4)})
    led = pd.DataFrame(rows, columns=["date", "player", "team", "opp", "stat", "side", "line", "price", "prob",
                                      "actual", "result", "units"])
    led.to_csv(LEDGER, index=False)
    return led


def summary(led: pd.DataFrame | None = None) -> str:
    if led is None:
        led = pd.read_csv(LEDGER) if LEDGER.exists() else pd.DataFrame()
    if led.empty:
        return "no graded plays yet"
    w, l = (led.result == "win").sum(), (led.result == "loss").sum()
    if not w + l:
        return "no graded plays yet"
    q = led.iloc[-1]
    icon = {"win": "✅", "loss": "❌", "push": "➖", "void": "⚪"}[q.result]
    last = f" · last: {icon} {q.player} {q.stat.upper()} {q.side} {q.line:g} → {q.actual:g}"
    return f"{w}-{l} ({w / (w + l):.0%}) · {led.units.sum():+.2f}u{last}"
