"""Record + ROI of the board's leans at an assumed -120 price.

Every lean shown on the board (projected player, >=10 projected minutes, lean
present) in a LOCKED file is graded once its game is final:
  win  -> +100/120 = +0.833 units      loss -> -1 unit
  push (actual == reference) -> 0      player did not play -> void (not counted)
The reference is the player's own baseline line (nba/analysis/lean.py), NOT a
sportsbook line, so this record is a tracking tool, not proof of a market edge.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from nba.config import ROOT, LIVE_SEASON
from nba.ingest import espn

LOCKS = ROOT / "reports" / "live" / "locked"
LEDGER = ROOT / "reports" / "live" / "record.csv"
WIN = 100 / 120
STRONG = 0.65


def _box(d: str) -> pd.DataFrame:
    ev = espn.regular_season_event_ids(d, d, season=LIVE_SEASON)  # completed games only
    rows = [r for e in ev for r in espn.parse_box(espn.fetch_summary(e["event_id"], LIVE_SEASON))]
    b = pd.DataFrame(rows)
    if b.empty:
        return b
    b["pid"] = b.espn_id.astype(str)
    b["gameId"] = b.event_id.astype("int64")
    b["pra"], b["pr"] = b.pts + b.reb + b.ast, b.pts + b.reb
    b["pa"], b["ra"] = b.pts + b.ast, b.reb + b.ast
    return b


def grade_all() -> pd.DataFrame:
    rows = []
    for day in sorted(p for p in LOCKS.iterdir() if p.is_dir()):
        L = pd.concat([pd.read_csv(f, dtype={"pid": str}).assign(game=f.stem) for f in day.glob("*.csv")],
                      ignore_index=True)
        if "lean_stat" not in L:
            continue  # locks from before leans existed
        L = L[L.lean_stat.notna() & (L.proj_min >= 10)]
        if "projected" in L:
            L = L[L.projected.astype(str).str.lower() == "true"]
        if L.empty:
            continue
        box = _box(day.name)
        if box.empty:
            continue
        final_games = set(box.gameId)
        L = L[L.gameId.isin(final_games)]
        m = L.merge(box[["pid", "gameId", "dnp", "pts", "reb", "ast", "pra", "pr", "pa", "ra"]],
                    on=["pid", "gameId"], how="left")
        for _, q in m.iterrows():
            actual = q.get(q.lean_stat)
            if pd.isna(actual) or q.dnp == 1:
                res, profit = "void", 0.0
            elif actual == q.lean_ref:
                res, profit = "push", 0.0
            else:
                hit = actual > q.lean_ref if q.lean_side == "over" else actual < q.lean_ref
                res, profit = ("win", WIN) if hit else ("loss", -1.0)
            rows.append({"date": day.name, "game": q.game, "player": q.player, "team": q.team,
                         "stat": q.lean_stat, "side": q.lean_side, "ref": q.lean_ref,
                         "prob": round(float(q.lean_prob_cal), 3), "strong": bool(q.lean_prob_cal >= STRONG - 0.005),
                         "actual": actual, "result": res, "units": round(profit, 4)})
    led = pd.DataFrame(rows, columns=["date", "game", "player", "team", "stat", "side", "ref", "prob", "strong",
                                      "actual", "result", "units"])
    led.to_csv(LEDGER, index=False)
    return led


def summarize(led: pd.DataFrame, mask=None) -> dict:
    x = led if mask is None else led[mask]
    x = x[x.result != "void"]
    w, l, p = (x.result == "win").sum(), (x.result == "loss").sum(), (x.result == "push").sum()
    risked = w + l  # pushes return the stake
    units = x.units.sum()
    return {"w": int(w), "l": int(l), "p": int(p), "hit": w / risked if risked else np.nan,
            "units": units, "roi": units / risked if risked else np.nan,
            "stated": x.prob.mean() if len(x) else np.nan}


def fmt(s: dict) -> str:
    if not (s["w"] + s["l"]):
        return "no graded leans yet"
    push = f"-{s['p']}" if s["p"] else ""
    return (f"{s['w']}-{s['l']}{push} ({s['hit']:.1%}) · {s['units']:+.2f}u · ROI {s['roi']:+.1%}"
            f" · model said {s['stated']:.0%}")


def report(led: pd.DataFrame) -> str:
    if led.empty:
        return "Record @ -120: no graded leans yet (starts Oct 22)."
    last = led.date.max()
    lines = ["**Record @ -120 (vs player's own baseline, not book lines; break-even 54.5%)**",
             f"- All leans: {fmt(summarize(led))}",
             f"- 🔥 Strong (65%+): {fmt(summarize(led, led.strong))}",
             f"- Last night ({last}): {fmt(summarize(led, led.date == last))}"]
    by = []
    for s, x in led.groupby("stat"):
        t = summarize(x)
        if t["w"] + t["l"]:
            by.append(f"{s.upper()} {t['w']}-{t['l']}")
    if by:
        lines.append("- By stat: " + " · ".join(by))
    return "\n".join(lines)
