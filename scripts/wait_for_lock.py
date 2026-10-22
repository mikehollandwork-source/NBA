"""Wait inside the Actions job until the next game's final lock time (tip - LOCK_LEAD_MIN).

    python scripts/wait_for_lock.py 2026-10-22 [--within 80]

Reads data/live/games_<date>.parquet (written by the run that just finished).
- If a scheduled game is already inside its lock hour and its lock file isn't final:
  exit 0 now (run again immediately to write the final lock).
- Else if the next lock time is within --within minutes: sleep until it (+15 s), exit 0.
- Else exit 1 (nothing to wait for; the next cron run will handle it).
GitHub's cron can start runs late or skip them, so each run that starts up to
--within minutes before a lock waits for it instead of hoping a run lands on time.
"""
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from nba.config import ROOT, LOCK_LEAD_MIN  # noqa: E402
from nba.live_pipeline import lock_path, _is_final  # noqa: E402

d = date.fromisoformat(sys.argv[1])
within = float(sys.argv[sys.argv.index("--within") + 1]) if "--within" in sys.argv else 80.0
gp = ROOT / f"data/live/games_{d}.parquet"
if not gp.exists():
    sys.exit(1)
G = pd.read_parquet(gp)
now = pd.Timestamp.now(tz="UTC")
todo = []
for _, g in G.iterrows():
    tip = pd.Timestamp(g.tip_utc)
    cutoff = tip - pd.Timedelta(minutes=LOCK_LEAD_MIN)
    if g.status == "Scheduled" and now < tip and not _is_final(lock_path(d, g)):
        todo.append((cutoff, f"{g.away}@{g.home}"))
if not todo:
    print("no games left to lock")
    sys.exit(1)
cutoff, game = min(todo)
if cutoff <= now:
    print(f"{game}: inside lock hour, not final yet -> run now")
    sys.exit(0)
wait = (cutoff - now).total_seconds() + 15
if wait > within * 60:
    print(f"next lock {game} at {cutoff:%H:%M} UTC is {wait / 60:.0f} min away; leaving it to a later run")
    sys.exit(1)
print(f"waiting {wait / 60:.1f} min for {game} final lock at {cutoff:%H:%M} UTC", flush=True)
time.sleep(wait)
sys.exit(0)
