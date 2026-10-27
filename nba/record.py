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


DK_BACKFILL = ROOT / "reports" / "live" / "dk_backfill.csv"


def _bet(actual, dnp, side, line, win_units):
    if pd.isna(actual) or dnp == 1:
        return "void", 0.0
    if actual == line:
        return "push", 0.0
    hit = actual > line if side == "over" else actual < line
    return ("win", win_units) if hit else ("loss", -1.0)


def grade_all() -> pd.DataFrame:
    from nba.ingest import espn_props
    rows = []
    bf = pd.read_csv(DK_BACKFILL, dtype={"pid": str}) if DK_BACKFILL.exists() else pd.DataFrame()
    for day in sorted(p for p in LOCKS.iterdir() if p.is_dir()):
        files = [f for f in day.glob("*.csv") if not f.name.endswith((".kalshi.csv", ".winner.csv"))]
        if not files:
            continue
        has_snap = (ROOT / "reports" / "live" / "dk_lines" / f"{day.name}.csv").exists()
        if not has_snap and (not len(bf) or day.name not in set(bf.date)):  # DK closing lines for days before DK columns existed
            new = espn_props.backfill(day.name, files)
            if len(new):
                bf = pd.concat([bf, new], ignore_index=True)
                bf.to_csv(DK_BACKFILL, index=False)
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
        snap_f = ROOT / "reports" / "live" / "dk_lines" / f"{day.name}.csv"
        if "dk_line" not in m and snap_f.exists() and "locked_at_utc" in m:
            # lock files written before DK lines were attached pre-lock: use the DK line logged by the SAME run
            sn = pd.read_csv(snap_f, dtype={"pid": str})
            sn = sn.rename(columns={"run_utc": "locked_at_utc"})[["locked_at_utc", "gameId", "pid", "lean_stat",
                                                                  "dk_line", "dk_price"]]
            m = m.merge(sn.drop_duplicates(["locked_at_utc", "gameId", "pid", "lean_stat"]),
                        on=["locked_at_utc", "gameId", "pid", "lean_stat"], how="left")
        if "dk_line" not in m and len(bf):
            b = bf[bf.date == day.name][["gameId", "pid", "stat", "dk_line", "dk_price"]].rename(columns={"stat": "lean_stat"})
            m = m.merge(b, on=["gameId", "pid", "lean_stat"], how="left")
        for c in ("dk_line", "dk_price"):
            if c not in m:
                m[c] = np.nan
        m["dk_ok"] = espn_props.ok_mask(m)
        for _, q in m.iterrows():
            actual = q.get(q.lean_stat)
            res, profit = _bet(actual, q.dnp, q.lean_side, q.lean_ref, WIN)
            dres, dunits = _bet(actual, q.dnp, q.lean_side, float(q.dk_line), espn_props.payout(q.dk_price)) \
                if q.dk_ok else ("", 0.0)
            rows.append({"date": day.name, "game": q.game, "player": q.player, "team": q.team,
                         "stat": q.lean_stat, "side": q.lean_side, "ref": q.lean_ref,
                         "prob": round(float(q.lean_prob_cal), 3), "strong": bool(q.lean_prob_cal >= STRONG - 0.005),
                         "actual": actual, "result": res, "units": round(profit, 4),
                         "dk_line": q.dk_line, "dk_price": q.dk_price, "dk_ok": bool(q.dk_ok),
                         "dk_result": dres, "dk_units": round(dunits, 4)})
    led = pd.DataFrame(rows, columns=["date", "game", "player", "team", "stat", "side", "ref", "prob", "strong",
                                      "actual", "result", "units", "dk_line", "dk_price", "dk_ok", "dk_result",
                                      "dk_units"])
    led.to_csv(LEDGER, index=False)
    return led


def dk_view(led: pd.DataFrame) -> pd.DataFrame:
    """The real record: 70%+ leans DraftKings offered at our line or better, graded at DK line + odds."""
    if "dk_ok" not in led:
        return led.iloc[0:0]
    x = led[(led.prob >= STRONG - 0.005) & led.dk_ok.astype(str).str.lower().eq("true")].copy()
    x["result"], x["units"] = x.dk_result, x.dk_units
    return x


KLEDGER = ROOT / "reports" / "live" / "kalshi_record.csv"


KCOLS = ["date", "game", "player", "team", "stat", "side", "k", "price", "cost", "p_model", "market_p", "p_side",
         "exp_roi", "paper_pick", "actual", "yes_hit", "result", "units",
         "fade_side", "fade_cost", "fade_result", "fade_units"]


def add_fade(g: pd.DataFrame) -> pd.DataFrame:
    """Fade = the OPPOSITE side of each Kalshi leg at its real price + fee, frozen at the lock
    (user's choice 2026-10-27: paper only, tests whether betting against the model's long shots pays)."""
    from nba import kalshi
    g = g.copy()
    cy = g["cost_yes"] if "cost_yes" in g else g.yes_ask + kalshi.fee(g.yes_ask)
    cn = g["cost_no"] if "cost_no" in g else (1 - g.yes_bid) + kalshi.fee(1 - g.yes_bid)
    g["fade_side"] = np.where(g.side == "YES", "NO", "YES")
    g["fade_cost"] = np.where(g.side == "YES", cn, cy)
    void = g.result == "void"
    win = g.result == "loss"   # the leg lost -> its opposite won
    g["fade_result"] = np.where(void, "void", np.where(win, "win", "loss"))
    g["fade_units"] = np.where(void, 0.0, np.where(win, (1 - g.fade_cost) / g.fade_cost, -1.0))
    return g


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
        g = add_fade(g)
        out.append(g[KCOLS])
    led = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=KCOLS)
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
        from nba.live_pipeline import TEAM  # ESPN codes (NY, WSH, ...) -> nba.com codes used in the files
        pts = box.assign(team=box.team.map(lambda t: TEAM.get(t, t))).groupby(["gameId", "team"]).pts.sum()
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


def summarize_fade(x: pd.DataFrame) -> str:
    """Record of fading every Kalshi paper leg (opposite side, real price + fee)."""
    if "paper_pick" in x:
        x = x[x.paper_pick.astype(str).str.lower().eq("true")]
    if "fade_result" not in x:
        return "no graded fades yet"
    x = x[x.fade_result != "void"]
    w, l = (x.fade_result == "win").sum(), (x.fade_result == "loss").sum()
    if not w + l:
        return "no graded fades yet"
    u = x.fade_units.sum()
    return f"{w}-{l} ({w / (w + l):.1%}) · {u:+.2f}u · ROI {u / (w + l):+.1%}"


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
        lines.append("**✅ PICKS record (DraftKings line + odds):** no graded picks yet.")
    else:
        last = led.date.max()
        strong = led.prob >= STRONG - 0.005  # recomputed from the stated prob (old ledgers used 65%)
        dv = dk_view(led)
        lines += ["**✅ PICKS record — real DraftKings line + odds (reset Oct 24; earlier days at DK closing line)**",
                  f"- Picks: {fmt(summarize(dv)) if len(dv) else 'none graded yet'}",
                  f"  - last night ({last}): {fmt(summarize(dv, dv.date == last)) if len(dv) else 'none'}",
                  "**Reference only (vs player's own average @ -120, not bettable lines)**",
                  f"- {STRONG:.0%}+ leans: {fmt(summarize(led, strong))}",
                  f"- All leans: {fmt(summarize(led))}"]
        by = []
        for s_, x in dv.groupby("stat"):
            t = summarize(x)
            if t["w"] + t["l"]:
                by.append(f"{s_.upper()} {t['w']}-{t['l']}")
        if by:
            lines.append("- DK picks by stat: " + " · ".join(by))
    if wled is not None:
        lines.append("**📈 Game-winner PREDICTIONS (tracked, not picks; straight up)**")
        lines.append(f"- Record: {summarize_w(wled)}" + (f" · last night: {summarize_w(wled[wled.date == wled.date.max()])}"
                                                        if len(wled) else ""))
    if kled is not None:
        lines.append("**🚫 NOT PICKS — Kalshi paper tracking (testing only; real price + fee, $1 each)**")
        if len(kled):
            last = kled.date.max()
            lines += [f"- Paper record: {summarize_k(kled)}", f"- Last night ({last}): {summarize_k(kled[kled.date == last])}",
                      f"- Model vs market: {model_vs_market(kled)}",
                      f"- 🔁 Fade (opposite side of every paper leg, real price + fee; also NOT picks): {summarize_fade(kled)}",
                      f"  - last night ({last}): {summarize_fade(kled[kled.date == last])}"]
        else:
            lines.append("- nothing graded yet (Oct 21 test: 53-187, -19%)")
    return "\n".join(lines)
