"""Send the night's projections to Telegram.

    python scripts/telegram_notify.py 2026-10-22

First run of the day: header + one message per game + full CSV attached.
Later runs: only what changed since the last send (injury status, leans that
flipped or moved >= 5 pts, players added/removed, games that just locked),
plus the updated CSV. State lives in reports/live/telegram_state_<date>.json
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

from nba.config import ROOT, VOL_MAX  # noqa: E402

STRONG = 0.65  # backtest: stated 65%+ leans hit ~70%+
TOKEN, CHAT = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
API = f"https://api.telegram.org/bot{TOKEN}"


def send(text: str):
    if not TOKEN:
        print("---- [dry run] ----\n" + text)
        return
    for chunk in [text[i:i + 3900] for i in range(0, len(text), 3900)]:
        r = requests.post(f"{API}/sendMessage", data={"chat_id": CHAT, "text": chunk, "parse_mode": "HTML",
                                                       "disable_web_page_preview": True}, timeout=30)
        r.raise_for_status()


def send_file(path: Path, caption: str):
    if not TOKEN:
        print(f"---- [dry run] file {path.name}: {caption}")
        return
    with open(path, "rb") as f:
        requests.post(f"{API}/sendDocument", data={"chat_id": CHAT, "caption": caption},
                      files={"document": (path.name, f)}, timeout=60).raise_for_status()


def lean_line(q) -> str:
    stat = str(q.lean_stat).upper()
    proj = q.get(f"proj_{q.lean_stat}", float("nan"))
    tag = "🔥 " if round(q.lean_prob_cal, 2) >= STRONG else ""
    dtd = " (DTD)" if q.get("status") == "Day-To-Day" else ""
    return (f"{tag}<b>{html.escape(q.player)}</b>{dtd} — {stat} {q.lean_side} {q.lean_ref:.1f} · "
            f"<b>{q.lean_prob_cal:.0%}</b> (proj {proj:.1f})")


def game_message(g, r) -> str:
    tip = pd.Timestamp(g.tip_utc).tz_convert("America/New_York").strftime("%-I:%M %p ET")
    x = r[(r.gameId == int(g.event_id))]
    state = x.lock_state.iloc[0] if len(x) else ""
    lock = " 🔒" if state == "LOCKED" else ""
    p = x[x.get("projected", True) == True]  # noqa: E712
    p = p[p.lean_stat.apply(lambda v: isinstance(v, str)) & (p.proj_min >= 10)].sort_values("lean_prob_cal", ascending=False)
    lines = [f"<b>{g.away} @ {g.home}</b> · {tip}{lock}"]
    strong_mask = p.lean_prob_cal.round(2) >= STRONG  # match the displayed %
    strong, rest = p[strong_mask], p[~strong_mask]
    if len(strong):
        lines.append(f"<b>Strong leans ({STRONG:.0%}+)</b>")
        lines += [lean_line(q) for _, q in strong.iterrows()]
    if len(rest):
        lines.append("Other leans")
        lines += [lean_line(q) for _, q in rest.iterrows()]
    skipped = x[(x.get("projected", True) == False) & (x.proj_min >= 10)]  # noqa: E712
    if len(skipped):
        lines.append("No projection (too volatile / no history): " + ", ".join(html.escape(n) for n in skipped.player))
    return "\n".join(lines)


def snapshot(r) -> dict:
    s = {}
    for _, q in r.iterrows():
        s[str(q.pid)] = {"player": q.player, "game": int(q.gameId), "status": str(q.get("status", "")),
                         "lean": f"{q.lean_stat}|{q.lean_side}" if isinstance(q.get("lean_stat"), str) else "",
                         "prob": round(float(q.lean_prob_cal), 3) if pd.notna(q.get("lean_prob_cal")) else None,
                         "lock": str(q.get("lock_state", ""))}
    return s


def main(d: date):
    r = pd.read_parquet(ROOT / f"data/live/projections_{d}.parquet")
    G = pd.read_parquet(ROOT / f"data/live/games_{d}.parquet")
    csv = ROOT / f"reports/live/game_day_{d}.csv"
    state_path = ROOT / f"reports/live/telegram_state_{d}.json"
    now = snapshot(r)
    if not state_path.exists():
        n_strong = int((r.lean_prob_cal >= STRONG).sum()) if "lean_prob_cal" in r else 0
        send(f"🏀 <b>Projections — {d:%a %b %-d}</b>\n{len(G)} games · {n_strong} strong leans (🔥 = {STRONG:.0%}+)\n"
             f"Lean = stat the model leans on hardest vs the player's own baseline (not a sportsbook line). "
             f"Players whose PRA swings > {VOL_MAX} get no projection. Locks at tip.")
        for _, g in G.sort_values("tip_utc").iterrows():
            send(game_message(g, r))
        if csv.exists():
            send_file(csv, f"Full projections {d}")
    else:
        old = json.loads(state_path.read_text())
        changed_games, notes = set(), []
        for pid, n in now.items():
            o = old.get(pid)
            if o is None:
                notes.append(f"➕ {n['player']} added"); changed_games.add(n["game"])
                continue
            if n["lock"] == "LOCKED" and o["lock"] != "LOCKED":
                changed_games.add(n["game"]); continue
            if n["lock"] == "LOCKED":
                continue
            if n["status"] != o["status"]:
                notes.append(f"🩹 {n['player']}: {o['status'] or 'Active'} → {n['status'] or 'Active'}"); changed_games.add(n["game"])
            if n["lean"] != o["lean"] or (n["prob"] is not None and o["prob"] is not None and abs(n["prob"] - o["prob"]) >= 0.05):
                notes.append(f"↔️ {n['player']}: lean {o['lean'] or '—'} {o['prob'] or ''} → {n['lean'] or '—'} {n['prob'] or ''}")
                changed_games.add(n["game"])
        for pid, o in old.items():
            if pid not in now and o["lock"] != "LOCKED":
                notes.append(f"➖ {o['player']} removed (ruled out?)"); changed_games.add(o["game"])
        if not changed_games:
            print("no changes; nothing sent")
        else:
            send(f"🔄 <b>Update — {pd.Timestamp.now(tz='America/New_York'):%-I:%M %p ET}</b>\n" + "\n".join(html.escape(x) for x in notes[:40]))
            for _, g in G[G.event_id.astype(int).isin(changed_games)].sort_values("tip_utc").iterrows():
                send(game_message(g, r))
            if csv.exists():
                send_file(csv, f"Updated projections {d}")
    state_path.write_text(json.dumps(now))


if __name__ == "__main__":
    main(date.fromisoformat(sys.argv[1]))
