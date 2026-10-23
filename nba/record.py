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

from nba.config import ROOT, LIVE_SEASON, STRONG_MIN
from nba.ingest import espn

LOCKS = ROOT / "reports" / "live" / "locked"
LEDGER = ROOT / "reports" / "live" / "record.csv"
WIN = 100 / 120
STRONG = STRONG_MIN  # board shows only these; record keeps all leans AND strongest separately


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


def _result(actual, dnp, side, line, win_units):
    """(result, units) for one bet at `line`; win pays win_units per 1u staked."""
    if pd.isna(actual) or dnp == 1:
        return "void", 0.0
    if actual == line:
        return "push", 0.0
    hit = actual > line if side == "over" else actual < line
    return ("win", win_units) if hit else ("loss", -1.0)


def pub_view(led: pd.DataFrame) -> pd.DataFrame:
    """Published picks as a ledger graded at the market line and real price."""
    if "pub" not in led:
        return led.iloc[0:0]
    x = led[led.pub.astype(str).str.lower().eq("true")].copy()
    x["result"], x["units"] = x.pub_result, x.pub_units
    return x


def grade_all() -> pd.DataFrame:
    rows = []
    for day in sorted(p for p in LOCKS.iterdir() if p.is_dir()):
        files = [f for f in day.glob("*.csv") if not f.name.endswith((".kalshi.csv", ".winner.csv"))]
        if not files:
            continue
        L = pd.concat([pd.read_csv(f, dtype={"pid": str}).assign(game=f.stem) for f in files], ignore_index=True)
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
            res, profit = _result(actual, q.dnp, q.lean_side, q.lean_ref, WIN)
            pub = str(q.get("pub", "")).lower() == "true"
            pres, pprofit = _result(actual, q.dnp, q.lean_side, float(q.pub_line), float(q.pub_payout)) \
                if pub else ("", 0.0)
            rows.append({"date": day.name, "game": q.game, "player": q.player, "team": q.team,
                         "stat": q.lean_stat, "side": q.lean_side, "ref": q.lean_ref,
                         "prob": round(float(q.lean_prob_cal), 3), "strong": bool(q.lean_prob_cal >= STRONG - 0.005),
                         "actual": actual, "result": res, "units": round(profit, 4),
                         "pub": pub, "pub_venue": q.get("pub_venue") if pub else "",
                         "pub_line": q.get("pub_line") if pub else np.nan, "pub_price": q.get("pub_price") if pub else "",
                         "pub_ev": q.get("pub_ev") if pub else np.nan, "pub_result": pres,
                         "pub_units": round(pprofit, 4)})
    led = pd.DataFrame(rows, columns=["date", "game", "player", "team", "stat", "side", "ref", "prob", "strong",
                                      "actual", "result", "units", "pub", "pub_venue", "pub_line", "pub_price",
                                      "pub_ev", "pub_result", "pub_units"])
    led.to_csv(LEDGER, index=False)
    return led


KLEDGER = ROOT / "reports" / "live" / "kalshi_record.csv"


def grade_kalshi() -> pd.DataFrame:
    """Grade locked Kalshi picks at the price paid + fee (win (1-cost)/cost, loss -1, void 0)."""
    from nba import kalshi
    out = []
    for day in sorted(p for p in LOCKS.iterdir() if p.is_dir()):
        frames = []
        for f in day.glob("*.kalshi.csv"):
            try:
                x = pd.read_csv(f, dtype={"pid": str})
            except pd.errors.EmptyDataError:
                continue
            if len(x):
                frames.append(x.assign(game=f.name.replace(".kalshi.csv", "")))
        if not frames:
            continue
        pk = pd.concat(frames, ignore_index=True)
        box = _box(day.name)
        if box.empty:
            continue
        pk = pk[pk.gameId.isin(set(box.gameId))]
        if pk.empty:
            continue
        g = kalshi.grade(pk, box)
        g["date"] = day.name
        if "paper_pick" not in g:
            g["paper_pick"] = True
        if "market_p" not in g:
            g["market_p"] = (g.yes_bid + g.yes_ask) / 2
        g["yes_hit"] = (g.actual >= g.k).astype(float).where(g.result != "void")
        out.append(g[["date", "game", "player", "team", "stat", "side", "k", "price", "cost", "p_model", "market_p",
                      "p_side", "exp_roi", "paper_pick", "actual", "yes_hit", "result", "units"]])
    led = pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["date", "game", "player", "team", "stat", "side", "k", "price", "cost", "p_model", "market_p", "p_side",
                 "exp_roi", "paper_pick", "actual", "yes_hit", "result", "units"])
    led.to_csv(KLEDGER, index=False)
    return led


WLEDGER = ROOT / "reports" / "live" / "winner_record.csv"


def grade_winners() -> pd.DataFrame:
    """Grade locked game-winner predictions (not picks) once final: straight-up result."""
    rows = []
    for day in sorted(p for p in LOCKS.iterdir() if p.is_dir()):
        files = list(day.glob("*.winner.csv"))
        if not files:
            continue
        box = _box(day.name)
        if box.empty:
            continue
        pts = box.groupby(["gameId", "team"]).pts.sum()
        for f in files:
            w = pd.read_csv(f).iloc[0]
            gid = int(w.event_id)
            if gid not in box.gameId.values:
                continue
            hp, ap = pts.get((gid, w.home)), pts.get((gid, w.away))
            if hp is None or ap is None or hp == ap:
                continue
            winner = w.home if hp > ap else w.away
            rows.append({"date": day.name, "game": f.name.replace(".winner.csv", ""), "pick_team": w.pick_team,
                         "p_pick": float(w.p_pick), "proj_margin_home": float(w.proj_margin_home),
                         "home_pts": int(hp), "away_pts": int(ap), "winner": winner,
                         "result": "win" if winner == w.pick_team else "loss"})
    led = pd.DataFrame(rows, columns=["date", "game", "pick_team", "p_pick", "proj_margin_home", "home_pts",
                                      "away_pts", "winner", "result"])
    led.to_csv(WLEDGER, index=False)
    return led


def summarize_w(x: pd.DataFrame) -> str:
    if x.empty:
        return "none graded yet"
    w, l = (x.result == "win").sum(), (x.result == "loss").sum()
    y = (x.result == "win").astype(float)
    brier = ((x.p_pick - y) ** 2).mean()
    return f"{w}-{l} ({w / (w + l):.0%}) · model said {x.p_pick.mean():.0%} · Brier {brier:.3f} (coin flip 0.250)"


def model_vs_market(x: pd.DataFrame) -> str:
    """Brier score on every priced rung (lower = better probabilities)."""
    x = x.dropna(subset=["yes_hit", "p_model", "market_p"])
    if x.empty:
        return "no graded rungs yet"
    bm, bk = ((x.p_model - x.yes_hit) ** 2).mean(), ((x.market_p - x.yes_hit) ** 2).mean()
    who = "model better" if bm < bk else "market better"
    return f"Brier model {bm:.4f} vs Kalshi {bk:.4f} ({who}) · {len(x)} rungs"


def summarize_k(x: pd.DataFrame) -> str:
    if "paper_pick" in x:
        x = x[x.paper_pick.astype(str).str.lower().eq("true")]
    x = x[x.result != "void"]
    w, l = (x.result == "win").sum(), (x.result == "loss").sum()
    if not w + l:
        return "no graded picks yet"
    u = x.units.sum()
    return f"{w}-{l} ({w / (w + l):.1%}) · {u:+.2f}u · ROI {u / (w + l):+.1%} · expected {x.exp_roi.mean():+.1%}"


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


def report(led: pd.DataFrame, kled: pd.DataFrame | None = None, wled: pd.DataFrame | None = None) -> str:
    """✅ Picks record first; Kalshi paper tracking (🚫 not picks) after."""
    lines = []
    if led.empty:
        lines.append("**✅ PICKS record:** no graded picks yet.")
    else:
        last = led.date.max()
        strong = led.prob >= STRONG - 0.005  # recomputed from the stated prob (old ledgers used 65%)
        pv = pub_view(led)
        lines += ["**✅ PICKS record (market-checked since Oct 23: real line + real price)**",
                  f"- Picks: {fmt(summarize(pv)) if len(pv) else 'none graded yet'}"
                  + (f" · last night: {fmt(summarize(pv, pv.date == last))}" if len(pv) else ""),
                  "**Reference (vs player's own average @ -120, not real prices)**",
                  f"- {STRONG:.0%}+ leans: {fmt(summarize(led, strong))}",
                  f"  - last night ({last}): {fmt(summarize(led, strong & (led.date == last)))}",
                  f"- All leans: {fmt(summarize(led))}",
                  f"  - last night ({last}): {fmt(summarize(led, led.date == last))}"]
        by = []
        for s_, x in led[strong].groupby("stat"):
            t = summarize(x)
            if t["w"] + t["l"]:
                by.append(f"{s_.upper()} {t['w']}-{t['l']}")
        if by:
            lines.append("- Picks by stat: " + " · ".join(by))
    if wled is not None:
        lines.append("**📈 Game-winner PREDICTIONS (tracked, not picks; straight up)**")
        lines.append(f"- Record: {summarize_w(wled)}" + (f" · last night: {summarize_w(wled[wled.date == wled.date.max()])}"
                                                        if len(wled) else ""))
    if kled is not None:
        lines.append("**🚫 NOT PICKS — Kalshi paper tracking (testing only; real price + fee, $1 each)**")
        if len(kled):
            last = kled.date.max()
            lines += [f"- Paper record: {summarize_k(kled)}", f"- Last night ({last}): {summarize_k(kled[kled.date == last])}",
                      f"- Model vs market: {model_vs_market(kled)}"]
        else:
            lines.append("- nothing graded yet (Oct 21 test: 53-187, -19%)")
    return "\n".join(lines)
