"""Send the night's projections to Telegram.

    python scripts/telegram_notify.py 2026-10-22

First run of the day: header + one message per game with only the strongest
leans (STRONG_MIN, 70%+) + full CSV of every player attached.
Later runs: only what changed among strongest leans (entered/left the list,
flipped or moved >= 5 pts, injury status) and games that just locked (final
lock = 1 hour before tip), plus the updated CSV. State lives in reports/live/telegram_state_<date>.json
(committed, so hourly Actions runs know what was already sent).

Sends to Discord when DISCORD_WEBHOOK_URL is set (preferred, user's choice 2026-10-28), else Telegram
(TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID); without either it prints
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
# Discord webhook (user's choice 2026-10-28): when set it replaces Telegram; one secret, no chat id
DISCORD = (os.environ.get("DISCORD_WEBHOOK_URL") or "").strip() or None
LIVE = bool(DISCORD or TOKEN)  # False = dry run (prints only, nothing marked as sent)


def _to_discord(text: str) -> str:
    """Telegram HTML -> Discord markdown."""
    return html.unescape(text.replace("<b>", "**").replace("</b>", "**").replace("<i>", "*").replace("</i>", "*"))


def _chunks(text: str, size: int = 1900):
    """Split on line breaks so no message passes Discord's 2,000-character limit."""
    out, cur = [], ""
    for line in text.split("\n"):
        while len(line) > size:  # a single huge line
            out.append(cur) if cur else None
            cur = ""
            out.append(line[:size])
            line = line[size:]
        if len(cur) + len(line) + 1 > size:
            out.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        out.append(cur)
    return out


def _discord_post(**kw):
    import time
    for _ in range(5):
        r = requests.post(DISCORD, timeout=60, **kw)
        if r.status_code == 429:  # rate limited: wait as told
            time.sleep(float(r.json().get("retry_after", 1)) + 0.2)
            continue
        if not r.ok:
            raise SystemExit(f"Discord post failed ({r.status_code}): {r.text[:200]}")
        time.sleep(0.6)  # stay under the webhook rate limit
        return
    raise SystemExit("Discord post failed: rate limited 5 times")


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
    if DISCORD:
        for chunk in _chunks(_to_discord(text)):
            _discord_post(json={"content": chunk, "allowed_mentions": {"parse": []}})
        return
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
    if DISCORD:
        with open(path, "rb") as f:
            _discord_post(data={"payload_json": json.dumps({"content": caption, "allowed_mentions": {"parse": []}})},
                          files={"files[0]": (path.name, f)})
        return
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
    from nba.ingest import espn_props
    return (f"✅ <b>PICK</b> · <b>{html.escape(q.player)}</b>{dtd}{flag} — {stat} {q.lean_side} {float(q.dk_line):g} "
            f"<b>{int(q.dk_price):+d}</b> at DraftKings (our line {q.lean_ref:.1f}) · <b>{q.lean_prob_cal:.0%}</b> (proj {proj:.1f})")


def game_message(g, r) -> str:
    """Lock post (user's choice 2026-10-28: clean): game, lock time, ✅ picks or 'no picks'."""
    tip = pd.Timestamp(g.tip_utc).tz_convert("America/New_York").strftime("%-I:%M %p ET")
    x = r[(r.gameId == int(g.event_id))]
    at = x.locked_at_utc.iloc[0] if len(x) and "locked_at_utc" in x else None
    lock = f" · 🔒 {pd.Timestamp(at).tz_convert('America/New_York'):%-I:%M %p}" if at is not None and pd.notna(at) else ""
    p = x[x.get("projected", True).astype(str).str.lower() == "true"] if "projected" in x else x
    p = p[p.lean_stat.apply(lambda v: isinstance(v, str)) & (p.proj_min >= 10)].sort_values("lean_prob_cal", ascending=False)
    strong = p[p.lean_prob_cal.round(2) >= STRONG]  # match the displayed %
    from nba.ingest import espn_props
    pub = strong[espn_props.ok_mask(strong)] if len(strong) else strong
    if pub.empty:  # user 2026-10-28: no picks -> just the teams and start time
        return f"<b>{g.away} @ {g.home}</b> · {tip}"
    return "\n".join([f"<b>{g.away} @ {g.home}</b> · {tip}{lock}"] + [lean_line(q) for _, q in pub.iterrows()])


def snapshot(r) -> dict:
    s = {}
    for _, q in r.iterrows():
        s[str(q.pid)] = {"player": q.player, "game": int(q.gameId), "status": str(q.get("status", "")),
                         "lean": f"{q.lean_stat}|{q.lean_side}" if isinstance(q.get("lean_stat"), str) else "",
                         "prob": round(float(q.lean_prob_cal), 3) if pd.notna(q.get("lean_prob_cal")) else None,
                         "strong": bool(pd.notna(q.get("lean_prob_cal")) and round(float(q.lean_prob_cal), 2) >= STRONG
                                        and float(q.get("proj_min", 0) or 0) >= 10
                                        and str(q.get("projected", True)).lower() == "true"
                                        and bool(q.get("dk_ok", False)) and str(q.get("dk_ok", "")).lower() == "true"),
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
        sx = record.dk_view(x)
        lines = [f"🏁 <b>PICK results {html.escape(game)}</b> ({dte}, DraftKings line + odds)"]
        for _, q in sx.iterrows():
            icon = {"win": "✅", "loss": "❌", "push": "➖", "void": "⚪"}.get(q.result, "·")
            act = "DNP" if q.result == "void" else f"{q.actual:g}"
            lines.append(f"{icon} {html.escape(q.player)} {STAT_NAME.get(q.stat, q.stat.upper())} {q.side} "
                         f"{float(q.dk_line):g} ({int(q.dk_price):+d}) → {act} ({q.units:+.2f}u)")
        if len(sx):  # games with no picks don't get a results post (keep it clean)
            send("\n".join(lines))
    send(record_brief())
    if LIVE:  # dry runs must not mark anything as sent
        st_path.write_text(json.dumps(sorted(done | set(new))))


def record_brief() -> str:
    """One-line pick record (DraftKings line + odds) + last night."""
    from nba import record
    if not record.LEDGER.exists():
        return "📊 <b>Record</b>: no graded picks yet"
    led = pd.read_csv(record.LEDGER)
    dv = record.dk_view(led)
    if dv.empty:
        return "📊 <b>Record</b>: no graded picks yet"
    last = led.date.max()
    tot, ln = record.summarize(dv), record.summarize(dv, dv.date == last)
    def f(t):
        n = t["w"] + t["l"]
        return f"{t['w']}-{t['l']}" + (f" ({t['w'] / n:.0%}) · {t['units']:+.2f}u" if n else "")
    return (f"📊 <b>Record</b> (DraftKings line + odds): {f(tot)}\n"
            f"Last night ({pd.Timestamp(last):%b %-d}): {f(ln) if ln['w'] + ln['l'] else 'no picks'}")


def on_fire(r: pd.DataFrame, n: int = 5) -> str:
    """🔥 Players who scored well above their average last game, ranked by how strongly our data leans today
    (user's choice 2026-10-28). Backtest 2019-25 (5 a day, vs own baseline): stated 76.7% -> hit 79.3%, better than
    stated in 7/7 seasons; unders after a hot game hit 64% vs 70% stated, so they're flagged ⚠️."""
    from nba.config import PROC, LIVE_SEASON
    from nba.ingest import espn_props
    f = PROC / f"player_games_{LIVE_SEASON}.parquet"
    if not f.exists():
        return ""
    pg = pd.read_parquet(f)
    pg = pg[(pg.source != "pending") & pg.pts.notna() & (pg["min"] > 0)].sort_values("game_date")  # finished games only
    last = pg.assign(pid=pg.espn_id.astype(str)).groupby("pid").tail(1).set_index("pid")[["pts"]].rename(columns={"pts": "last_pts"})
    x = r[r.lean_stat.apply(lambda v: isinstance(v, str)) & (r.proj_min >= 20) & r.pts_ref.notna()].copy()
    if "projected" in x:
        x = x[x.projected.astype(str).str.lower() == "true"]
    if "status" in x:
        x = x[~x.status.fillna("").str.lower().isin(["out", "doubtful"])]
    x["pid"] = x.pid.astype(str)
    x = x.join(last, on="pid")
    x = x[x.last_pts.notna()]
    x["hot"] = x.last_pts - x.pts_ref
    x = x[x.hot > 0].nlargest(15, "hot").nlargest(n, "lean_prob_cal")
    if x.empty:
        return ""
    x["is_pick"] = (x.lean_prob_cal.round(2) >= STRONG) & espn_props.ok_mask(x)
    lines = ["🔥 <b>On fire — 5 players our data likes most today</b>"]
    for _, q in x.iterrows():
        stat = STAT_NAME.get(q.lean_stat, str(q.lean_stat).upper())
        warn = " ⚠️" if q.lean_side == "under" else ""
        tag = (f" · ✅ PICK DK {'o' if q.lean_side == 'over' else 'u'}{float(q.dk_line):g} {int(q.dk_price):+d}"
               if q.is_pick else "")
        lines.append(f"• <b>{html.escape(q.player)}</b> ({q.team}) — {q.last_pts:.0f} pts last game (avg {q.pts_ref:.1f}) → "
                     f"{stat} {q.lean_side} {q.lean_ref:g} · <b>{q.lean_prob_cal:.0%}</b>{warn}{tag}")
    lines.append("Lean vs his own average, not a sportsbook line. ✅ PICK = bettable at DraftKings. "
                 "⚠️ under after a hot game (hits less than stated).")
    return "\n".join(lines)


def scoring_watch(r: pd.DataFrame, n: int = 5) -> str:
    """🔥 Most scoring potential tonight (user's choice 2026-10-28: mix of ceiling and upside vs average).
    Ceiling = 90th-percentile points from the 20,000 sims (a 1-in-10 good night); upside = ceiling minus
    the player's own baseline. Ranked by the average of both percentile ranks on the slate. Not picks."""
    x = r[r.get("projected", True).astype(bool) & (r.proj_min >= 20) & r.pts_p90.notna() & r.pts_ref.notna()].copy()
    if "status" in x:
        x = x[~x.status.fillna("").str.lower().isin(["out", "doubtful"])]
    if x.empty:
        return ""
    x["upside"] = x.pts_p90 - x.pts_ref
    x["score"] = (x.pts_p90.rank(pct=True) + x.upside.rank(pct=True)) / 2
    x = x.sort_values(["score", "pts_p90"], ascending=False).head(n)
    lines = ["🔥 <b>Scoring watch — most points potential tonight</b> (NOT picks)",
             "Mix of ceiling (1-in-10 good night from the sims) and upside vs the player's own average."]
    for _, q in x.iterrows():
        lines.append(f"• {html.escape(q.player)} ({q.team}) — proj {q.proj_pts:.1f} · ceiling {q.pts_p90:.0f} · "
                     f"avg {q.pts_ref:.1f} (+{q.upside:.0f})")
    return "\n".join(lines)


def main(d: date):
    results_messages()  # Kalshi paper is graded in the record files only; never posted (user 2026-10-28)
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
        # morning post (user's choice 2026-10-28): record + on-fire list only; games post when they lock
        send(f"🏀 <b>{d:%a %b %-d}</b> · {len(G)} games · picks post as each game locks (1 hr before tip)\n\n"
             + record_brief())
        fire = on_fire(r)
        if fire:
            send(fire)
    else:
        old = json.loads(state_path.read_text())
        locked_now = {n["game"] for pid, n in now.items()
                      if n["lock"] == "LOCKED" and old.get(pid, {}).get("lock") != "LOCKED"}
        for _, g in G[G.event_id.astype(int).isin(locked_now)].sort_values("tip_utc").iterrows():
            send(game_message(g, r))
        if not locked_now:
            print("no new locks; nothing sent")
    if LIVE:  # dry runs must not mark the board as sent
        state_path.write_text(json.dumps(now))


if __name__ == "__main__":
    main(date.fromisoformat(sys.argv[1]))
