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
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402
import requests  # noqa: E402

from nba.config import ROOT, VOL_MAX, STRONG_MIN, LOCK_LEAD_MIN, STAT_NAME  # noqa: E402

MORNING_HOUR = 10  # ET; morning post never goes out earlier (late cron runs after midnight, DK lines not up)
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


WEBHOOK_RE = re.compile(r"^https://(?:(?:ptb|canary)\.)?discord(?:app)?\.com/api/(?:v\d+/)?webhooks/\d+/[\w-]+$")


def _check_webhook():
    """Fail loudly (without printing the secret) if the URL isn't a Discord webhook link."""
    if not WEBHOOK_RE.match(DISCORD):
        host = DISCORD.split("/")[2] if DISCORD.count("/") >= 2 else "?"
        path = "/".join(DISCORD.split("/")[3:5]) if DISCORD.count("/") >= 4 else "?"
        raise SystemExit(f"DISCORD_WEBHOOK_URL is not a webhook link (looks like {host}/{path}/...). It must look like "
                         "https://discord.com/api/webhooks/<number>/<token> - copy it from Channel > Integrations > Webhooks.")


def _discord_post(**kw):
    import time
    _check_webhook()
    for _ in range(5):
        # wait=true: Discord returns the created message, so we know it was really posted
        r = requests.post(DISCORD, params={"wait": "true"}, timeout=60, **kw)
        if r.status_code == 429:  # rate limited: wait as told
            time.sleep(float(r.json().get("retry_after", 1)) + 0.2)
            continue
        if not r.ok:
            raise SystemExit(f"Discord post failed ({r.status_code}): {r.text[:200]}")
        try:
            ok = "id" in r.json()
        except ValueError:
            ok = False
        if not ok:
            raise SystemExit(f"Discord did not create a message ({r.status_code}); check DISCORD_WEBHOOK_URL")
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
    conf = q.get("conf")
    flag += " 🎯" if conf is not None and pd.notna(conf) and float(conf) >= 4 else ""
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
    win = winner_lines(g)
    if pub.empty and not win:  # user 2026-10-28: no picks -> just the teams and start time
        return f"<b>{g.away} @ {g.home}</b> · {tip}"
    return "\n".join([f"<b>{g.away} @ {g.home}</b> · {tip}{lock}"] + [lean_line(q) for _, q in pub.iterrows()] + win)


WIN_MIN = 0.70  # user 2026-11-06: post 70%+ winner predictions at the lock


def winners_morning(d, G) -> str:
    """Morning heads-up (user 2026-11-06): today's 70%+ winner predictions + 🏠 spots, as of the morning run.
    Final versions post at each game's lock; injury news can change or drop them."""
    wp = ROOT / f"data/live/winners_{d}.parquet"
    if not wp.exists():
        return ""
    w = pd.read_parquet(wp)
    if w.empty:
        return ""
    w = w[w.p_pick.astype(float) >= WIN_MIN].copy()
    if w.empty:
        return "📈 <b>Early winners</b>: no team at 70%+ today (as of this morning)"
    tips = G.assign(event_id=G.event_id.astype(str)).set_index("event_id").tip_utc
    w["tip"] = w.event_id.astype(str).map(tips)
    lines = ["📈 <b>Early winners — 70%+</b> (predictions, not picks)",
             "<i>As of this morning, before the final injury report. Confirmed at each game's lock (1 hr before tip).</i>"]
    for _, q in w.sort_values("tip").iterrows():
        t = pd.Timestamp(q.tip).tz_convert("America/New_York").strftime("%-I:%M %p") if pd.notna(q.tip) else ""
        rule = " 🏠 home fav vs back-to-back" if q.pick_team == q.home and float(q.get("b2b_away", 0) or 0) == 1 else ""
        lines.append(f"• <b>{q.pick_team}</b> {float(q.p_pick):.0%} ({q.away} @ {q.home}, {t}){rule}")
    return "\n".join(lines)


def winner_lines(g) -> list:
    """📈 70%+ winner predictions, plus the home-favorite-vs-back-to-back spot (user's rule 2026-11-06).
    Backtest 2019-25 at 70%+: rule 573 games won 79.9% (stated 79.0%); home favs vs rested teams 79.1% (78.4%):
    accurate, but the back-to-back adds no edge beyond the model. Predictions, not picks."""
    wp = ROOT / f"data/live/winners_{pd.Timestamp(g.tip_utc).tz_convert('America/New_York').date()}.parquet"
    if not wp.exists():
        return []
    w = pd.read_parquet(wp)
    w = w[w.event_id.astype(str) == str(g.event_id)] if len(w) else w
    if w.empty:
        return []
    q = w.iloc[0]
    out = []
    if float(q.p_pick) >= WIN_MIN:
        out.append(f"📈 Winner (prediction, not a pick): <b>{q.pick_team}</b> {float(q.p_pick):.0%}")
        if q.pick_team == q.home and float(q.get("b2b_away", 0) or 0) == 1:
            out.append(f"🏠 Home favorite vs a team on a back-to-back: <b>{q.pick_team}</b> (rule: 80% in 2019-25)")
    vt = q.get("value_team")
    if isinstance(vt, str) and vt:
        ours = float(q.p_home) if vt == q.home else 1 - float(q.p_home)
        mkt = float(q.mkt_home) if vt == q.home else 1 - float(q.mkt_home)
        out.append(f"💰 Value (PAPER, not a pick): <b>{vt}</b> {int(q.value_price):+d} at DraftKings · "
                   f"ours {ours:.0%} vs market {mkt:.0%}")
    at = q.get("ats_team")
    if isinstance(at, str) and at:
        our = float(q.proj_margin_home) if at == q.home else -float(q.proj_margin_home)
        out.append(f"📏 ATS (PAPER, not a pick): <b>{at} {float(q.ats_line):+g}</b> ({int(q.ats_price):+d}) at DraftKings · "
                   f"our margin {our:+.1f}, {float(q.ats_edge):.1f} pts better than the spread")
    return out


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
    posted = False
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
            posted = True
    if posted:  # record line only after real results (the morning post carries it otherwise)
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
    out = (f"📊 <b>Record</b> (DraftKings line + odds): {f(tot)}\n"
           f"Last night ({pd.Timestamp(last):%b %-d}): ✅ picks {f(ln) if ln['w'] + ln['l'] else 'none'}")
    return out + recap_last_night(last)


def recap_last_night(last) -> str:
    """Last night's ⭐ top plays (each result) and 📈 70%+ winners (user 2026-11-12: recap in the morning board)."""
    from nba import best_play, record
    lines = []
    if best_play.LEDGER.exists():
        b = pd.read_csv(best_play.LEDGER)
        b = b[b.date == last]
        if len(b):
            icon = {"win": "✅", "loss": "❌", "push": "➖", "void": "⚪"}
            lines.append("⭐ Top plays: " + " · ".join(
                f"{icon.get(q.result, '·')} {html.escape(q.player)} {q.stat.upper()} {q.side} {q.line:g} → {q.actual:g}"
                for _, q in b.sort_values("rank").iterrows()))
    if record.WLEDGER.exists():
        w = pd.read_csv(record.WLEDGER)
        w = w[(w.date == last) & (w.p_pick >= WIN_MIN)]
        if len(w):
            lines.append("📈 70%+ winners: " + " · ".join(
                f"{'✅' if q.result == 'win' else '❌'} {q.pick_team} {q.p_pick:.0%}" for _, q in w.iterrows()))
    if record.VLEDGER.exists():
        v = pd.read_csv(record.VLEDGER)
        lv = v[v.date == last]
        if len(lv) or len(v):
            lines.append("💰 Value (paper): " + (" · ".join(
                f"{'✅' if q.result == 'win' else '❌'} {q.team} {int(q.price):+d}" for _, q in lv.iterrows()) + " · "
                if len(lv) else "") + "season " + record.summarize_v(v))
    if record.ALEDGER.exists():
        a = pd.read_csv(record.ALEDGER)
        la = a[a.date == last]
        if len(la) or len(a):
            ic = {"win": "✅", "loss": "❌", "push": "➖"}
            lines.append("📏 ATS (paper): " + (" · ".join(
                f"{ic.get(q.result, '·')} {q.team} {q.line:+g}" for _, q in la.iterrows()) + " · " if len(la) else "")
                + "season " + record.summarize_v(a))
    return ("\n" + "\n".join(lines)) if lines else ""


def best_play_message(d, r: pd.DataFrame, G: pd.DataFrame) -> str:
    """⭐ Top 5 plays, ranked (nba/best_play.py; replaced 🔥 On fire on 2026-11-11). Frozen at the first post."""
    from nba import best_play
    plays = best_play.load(d)
    if plays is None:
        plays = best_play.picks(r, G)
        if not plays:
            return "⭐ <b>Top plays</b>: none yet (no lean at 60%+ on DraftKings at our number)"
        if LIVE:  # dry runs must not freeze the list
            best_play.save(d, plays)
    lines = ["⭐ <b>Top plays today</b> — ranked, best first (leans, not 70% picks)"]
    for p in plays:
        stat = STAT_NAME.get(p["stat"], p["stat"].upper())
        mark = "⭐" if p["rank"] == 1 else f"{p['rank']}."
        lines.append(f"{mark} <b>{html.escape(p['player'])}</b> ({p['team']} vs {p['opp']}) — {stat} {p['side']} "
                     f"{p['line']:g} <b>{p['price']:+d}</b> · {p['prob']:.0%} (proj {p['proj']:.1f})"
                     + (" 🎯" if p.get("conf") == 4 else ""))
    lines.append("<i>DraftKings lines as of this post; skip a play if he is ruled out or the line moves against us.</i>")
    lines.append("Record — " + best_play.summary())
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
    et_hour = pd.Timestamp.now(tz="America/New_York").hour
    if not state_path.exists() and et_hour < MORNING_HOUR:
        # a late-night run that already counts as "tomorrow": wait for the real morning (DK lines up by then)
        print(f"before {MORNING_HOUR}:00 ET; morning post waits for a later run")
        return
    if not state_path.exists():
        # morning post (user's choice 2026-10-28): record + on-fire list only; games post when they lock
        send(f"🏀 <b>{d:%a %b %-d}</b> · {len(G)} games · picks post as each game locks (1 hr before tip)\n\n"
             + record_brief())
        send(best_play_message(d, r, G))
        early = winners_morning(d, G)
        if early:
            send(early)
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
    if sys.argv[1] == "--test":  # one test message (workflow: discord-test)
        send("✅ NBA board connected — test message")
        print("test message posted" if LIVE else "dry run: no DISCORD_WEBHOOK_URL / Telegram secrets set")
    else:
        main(date.fromisoformat(sys.argv[1]))
