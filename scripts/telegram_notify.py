"""Send the night's projections to Telegram.

    python scripts/telegram_notify.py 2026-10-22

First run of the day: header + one message per game with only the strongest
leans (STRONG_MIN, 70%+) + full CSV of every player attached.
Later runs: only what changed among strongest leans (entered/left the list,
flipped or moved >= 5 pts, injury status) and games that just locked (final
lock = 1 hour before tip), plus the updated CSV. State lives in reports/live/telegram_state_<date>.json
(committed, so hourly Actions runs know what was already sent).

Needs secrets TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID; without them it prints
the messages instead (dry run).
"""
import html
import json
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402
import requests  # noqa: E402

from nba.config import ROOT, VOL_MAX, STRONG_MIN, LOCK_LEAD_MIN, STAT_NAME  # noqa: E402

STRONG = STRONG_MIN  # backtest 2025: stated 70%+ leans hit 75.8%
# strip: secrets pasted with a trailing newline break the API URL (404)
TOKEN = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip() or None
CHAT = (os.environ.get("TELEGRAM_CHAT_ID") or "").strip() or None
API = f"https://api.telegram.org/bot{TOKEN}"


def _check(r, what: str):
    """Print Telegram's reason (never the token) and stop on errors other than formatting."""
    if r.ok:
        return True
    try:
        desc = r.json().get("description", r.text[:200])
    except ValueError:
        desc = r.text[:200]
    print(f"Telegram {what} failed ({r.status_code}): {desc}", flush=True)
    if "chat not found" in desc.lower():
        print("-> check TELEGRAM_CHAT_ID and send your bot a message first (bots can't start chats)", flush=True)
    if "can't parse entities" in desc.lower():
        return False  # caller retries as plain text
    raise SystemExit(f"Telegram {what} failed: {desc}")


def send(text: str):
    if not TOKEN:
        print("---- [dry run] ----\n" + text)
        return
    for chunk in [text[i:i + 3900] for i in range(0, len(text), 3900)]:
        r = requests.post(f"{API}/sendMessage", data={"chat_id": CHAT, "text": chunk, "parse_mode": "HTML",
                                                       "disable_web_page_preview": True}, timeout=30)
        if not _check(r, "sendMessage"):  # formatting problem: send the same text without HTML
            plain = html.unescape(chunk.replace("<b>", "").replace("</b>", ""))
            _check(requests.post(f"{API}/sendMessage", data={"chat_id": CHAT, "text": plain,
                                                              "disable_web_page_preview": True}, timeout=30),
                   "sendMessage (plain)")


def send_file(path: Path, caption: str):
    if not TOKEN:
        print(f"---- [dry run] file {path.name}: {caption}")
        return
    with open(path, "rb") as f:
        _check(requests.post(f"{API}/sendDocument", data={"chat_id": CHAT, "caption": caption},
                             files={"document": (path.name, f)}, timeout=60), "sendDocument")


def lean_line(q) -> str:
    stat = STAT_NAME.get(q.lean_stat, str(q.lean_stat).upper())
    proj = q.get(f"proj_{q.lean_stat}", float("nan"))
    flag = " ⚠️" if str(q.get("star_g1_under", "")).lower() == "true" else ""
    dtd = " (DTD)" if q.get("status") == "Day-To-Day" else ""
    return (f"✅ <b>PICK</b> · <b>{html.escape(q.player)}</b>{dtd}{flag} — {stat} {q.lean_side} {q.lean_ref:.1f} · "
            f"<b>{q.lean_prob_cal:.0%}</b> (proj {proj:.1f})")


def game_message(g, r) -> str:
    tip = pd.Timestamp(g.tip_utc).tz_convert("America/New_York").strftime("%-I:%M %p ET")
    x = r[(r.gameId == int(g.event_id))]
    state = x.lock_state.iloc[0] if len(x) else ""
    if state == "LOCKED":
        at = x.locked_at_utc.iloc[0] if "locked_at_utc" in x else None
        lock = " · 🔒 locked" + (f" {pd.Timestamp(at).tz_convert('America/New_York'):%-I:%M %p}" if pd.notna(at) else "")
    else:
        cut = pd.Timestamp(g.tip_utc) - pd.Timedelta(minutes=LOCK_LEAD_MIN)
        lock = f" · locks {cut.tz_convert('America/New_York'):%-I:%M %p}"
    p = x[x.get("projected", True).astype(str).str.lower() == "true"] if "projected" in x else x
    p = p[p.lean_stat.apply(lambda v: isinstance(v, str)) & (p.proj_min >= 10)].sort_values("lean_prob_cal", ascending=False)
    lines = [f"<b>{g.away} @ {g.home}</b> · {tip}{lock}"]
    wp = ROOT / f"data/live/winners_{pd.Timestamp(g.tip_utc).tz_convert('America/New_York').date()}.parquet"
    if wp.exists():
        from nba.live_pipeline import winner_line
        wb = pd.read_parquet(wp)
        wb = wb[wb.event_id.astype(str) == str(g.event_id)] if len(wb) else wb
        if len(wb):
            lines.append(html.escape(winner_line(wb.iloc[0])))
    strong = p[p.lean_prob_cal.round(2) >= STRONG]  # match the displayed %
    if len(strong):
        lines += [lean_line(q) for _, q in strong.iterrows()]
    else:
        lines.append(f"✅ PICKS: none at {STRONG:.0%}+ this game")
    kp = ROOT / f"data/live/kalshi_{pd.Timestamp(g.tip_utc).tz_convert('America/New_York').date()}.parquet"
    if kp.exists():
        from nba import kalshi
        kb = pd.read_parquet(kp)
        if len(kb):
            kb = kb[(kb.gameId == int(g.event_id)) & kb.paper_pick.astype(str).str.lower().eq("true")]
            kb = kb.sort_values("exp_roi", ascending=False).head(3)
        if len(kb):
            lines.append("🚫 <b>NOT PICKS — Kalshi paper tracking only</b> (do not bet; -19% on its Oct 21 test)")
            lines += [f"🚫 NOT A PICK · {html.escape(q.player)} — {html.escape(kalshi.describe(q))}" for _, q in kb.iterrows()]
    return "\n".join(lines)


def snapshot(r) -> dict:
    s = {}
    for _, q in r.iterrows():
        s[str(q.pid)] = {"player": q.player, "game": int(q.gameId), "status": str(q.get("status", "")),
                         "lean": f"{q.lean_stat}|{q.lean_side}" if isinstance(q.get("lean_stat"), str) else "",
                         "prob": round(float(q.lean_prob_cal), 3) if pd.notna(q.get("lean_prob_cal")) else None,
                         "strong": bool(pd.notna(q.get("lean_prob_cal")) and round(float(q.lean_prob_cal), 2) >= STRONG
                                        and float(q.get("proj_min", 0) or 0) >= 10
                                        and str(q.get("projected", True)).lower() == "true"),
                         "lock": str(q.get("lock_state", ""))}
    return s


def results_messages():
    """Send ✅/❌ for every newly graded game + the updated record (state in
    reports/live/telegram_record_state.json)."""
    from nba import record
    led_path = record.LEDGER
    if not led_path.exists():
        return
    led = pd.read_csv(led_path)
    if led.empty:
        return
    st_path = ROOT / "reports/live/telegram_record_state.json"
    done = set(json.loads(st_path.read_text())) if st_path.exists() else set()
    new = [k for k in (led.date + "|" + led.game).unique() if k not in done]
    if not new:
        return
    for k in sorted(new):
        dte, game = k.split("|")
        x = led[(led.date == dte) & (led.game == game)].sort_values("prob", ascending=False)
        sx = x[x.prob >= STRONG - 0.005]
        lines = [f"🏁 <b>PICK results {html.escape(game)}</b> ({dte}, {STRONG:.0%}+)"]
        for _, q in sx.iterrows():
            icon = {"win": "✅", "loss": "❌", "push": "➖", "void": "⚪"}[q.result]
            act = "DNP" if q.result == "void" else f"{q.actual:g}"
            lines.append(f"{icon} {html.escape(q.player)} {STAT_NAME.get(q.stat, q.stat.upper())} {q.side} {q.ref:g} → {act}")
        if sx.empty:
            lines.append(f"no {STRONG:.0%}+ leans this game")
        lines.append(f"✅ Picks: {record.fmt(record.summarize(sx))}")
        lines.append(f"All leans (reference, not picks): {record.fmt(record.summarize(x))}")
        if record.WLEDGER.exists():
            wl = pd.read_csv(record.WLEDGER)
            wl = wl[(wl.date == dte) & (wl.game == game)]
            for _, q in wl.iterrows():
                icon = "✅" if q.result == "win" else "❌"
                lines.append(f"📈 Prediction (not a pick): {q.pick_team} {q.p_pick:.0%} → {icon} "
                             f"final {game.split('@')[0]} {q.away_pts}-{q.home_pts} {game.split('@')[1]}")
        send("\n".join(lines))
    kled = pd.read_csv(record.KLEDGER) if record.KLEDGER.exists() else None
    wled = pd.read_csv(record.WLEDGER) if record.WLEDGER.exists() else None
    send(html.escape(record.report(led, kled, wled)).replace("**", ""))
    if TOKEN:  # dry runs must not mark anything as sent
        st_path.write_text(json.dumps(sorted(done | set(new))))


def kalshi_results():
    from nba import record
    if not record.KLEDGER.exists():
        return
    k = pd.read_csv(record.KLEDGER)
    if k.empty:
        return
    st_path = ROOT / "reports/live/telegram_kalshi_state.json"
    done = set(json.loads(st_path.read_text())) if st_path.exists() else set()
    new = [x for x in (k.date + "|" + k.game).unique() if x not in done]
    for key_ in sorted(new):
        dte, game = key_.split("|")
        x = k[(k.date == dte) & (k.game == game) & k.paper_pick.astype(str).str.lower().eq("true")]
        if x.empty:
            continue
        lines = [f"🚫 <b>NOT PICKS — Kalshi paper results {html.escape(game)}</b> ({dte})"]
        for _, q in x.iterrows():
            icon = {"win": "✅", "loss": "❌", "void": "⚪"}[q.result]
            rung = f"{q.k}+" if q.side == "YES" else f"under {q.k}"
            lines.append(f"{icon} {html.escape(q.player)} {q.stat.upper()} {rung} @ {q.price * 100:.0f}¢ → "
                         f"{'DNP' if q.result == 'void' else format(q.actual, 'g')} ({q.units:+.2f}u)")
        lines.append("Paper record: " + record.summarize_k(k))
        lines.append("Model vs market: " + record.model_vs_market(k))
        send("\n".join(lines))
    if TOKEN:  # dry runs must not mark anything as sent
        st_path.write_text(json.dumps(sorted(done | set(new))))


def main(d: date):
    kalshi_results()
    results_messages()
    proj = ROOT / f"data/live/projections_{d}.parquet"
    if not proj.exists():
        print(f"no projections for {d}; results only")
        return
    r = pd.read_parquet(proj)
    G = pd.read_parquet(ROOT / f"data/live/games_{d}.parquet")
    csv = ROOT / f"reports/live/game_day_{d}.csv"
    state_path = ROOT / f"reports/live/telegram_state_{d}.json"
    now = snapshot(r)
    if not state_path.exists():
        n_strong = sum(v["strong"] for v in now.values())
        rec = ROOT / "reports/live/record_summary.md"
        rec_txt = html.escape(rec.read_text()).replace("**", "") if rec.exists() else ""
        send(f"🏀 <b>Picks — {d:%a %b %-d}</b>\n{len(G)} games · {n_strong} picks ({STRONG:.0%}+)\n"
             f"✅ PICK = our model's lean at {STRONG:.0%}+. 🚫 NOT A PICK = Kalshi paper tracking (testing only, do not bet). "
             f"📈 PREDICTION = game-winner model (tracked, not a pick).\n"
             f"Lean = stat the model leans on hardest vs the player's own baseline (not a sportsbook line). "
             f"Ranked by model probability. ⚠️ = star first-game under (model overstates these). "
             f"Players whose PRA swings > {VOL_MAX} get no projection. Final lock {LOCK_LEAD_MIN} min before tip; "
             f"full projections in the CSV.\n\n{rec_txt}")
        for _, g in G.sort_values("tip_utc").iterrows():
            send(game_message(g, r))
        if csv.exists():
            send_file(csv, f"Full projections {d}")
    else:
        old = json.loads(state_path.read_text())
        changed_games, notes = set(), []
        for pid, n in now.items():
            o = old.get(pid)
            was = bool(o and o.get("strong", o.get("prob") is not None and o["prob"] >= STRONG))
            if o is not None and n["lock"] == "LOCKED" and o["lock"] != "LOCKED":
                changed_games.add(n["game"]); continue  # final lock -> resend that game
            if n["lock"] == "LOCKED":
                continue
            if not (n["strong"] or was):
                continue  # only the strongest leans are on the board
            if n["strong"] and not was:
                notes.append(f"➕ {n['player']}: {n['lean']} {n['prob']:.0%} now on the board"); changed_games.add(n["game"])
            elif was and not n["strong"]:
                why = f"{n['lean'] or 'no lean'} {n['prob']:.0%}" if n["prob"] is not None else "removed (ruled out?)"
                notes.append(f"➖ {n['player']} off the board ({why})"); changed_games.add(n["game"])
            else:
                if n["status"] != o["status"]:
                    notes.append(f"🩹 {n['player']}: {o['status'] or 'Active'} → {n['status'] or 'Active'}"); changed_games.add(n["game"])
                if n["lean"] != o["lean"] or abs(n["prob"] - o["prob"]) >= 0.05:
                    notes.append(f"↔️ {n['player']}: {o['lean']} {o['prob']:.0%} → {n['lean']} {n['prob']:.0%}")
                    changed_games.add(n["game"])
        for pid, o in old.items():
            was = o.get("strong", o.get("prob") is not None and o["prob"] >= STRONG)
            if pid not in now and was and o["lock"] != "LOCKED":
                notes.append(f"➖ {o['player']} off the board (ruled out?)"); changed_games.add(o["game"])
        if not changed_games:
            print("no changes; nothing sent")
        else:
            if notes:
                send(f"🔄 <b>Update — {pd.Timestamp.now(tz='America/New_York'):%-I:%M %p ET}</b>\n" + "\n".join(html.escape(x) for x in notes[:40]))
            for _, g in G[G.event_id.astype(int).isin(changed_games)].sort_values("tip_utc").iterrows():
                send(game_message(g, r))
            if csv.exists():
                send_file(csv, f"Updated projections {d}")
    if TOKEN:  # dry runs must not mark the board as sent
        state_path.write_text(json.dumps(now))


if __name__ == "__main__":
    main(date.fromisoformat(sys.argv[1]))
