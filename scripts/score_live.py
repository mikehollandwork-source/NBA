"""Score a night's projections against final box scores (ESPN).
    python scripts/score_live.py 2026-10-21
Compares the game-day model sheet and (if present) the preseason sheet."""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from nba.config import LIVE_SEASON, ROOT  # noqa: E402
from nba.ingest import espn  # noqa: E402

d = date.fromisoformat(sys.argv[1])
ev = espn.regular_season_event_ids(d.isoformat(), d.isoformat(), season=LIVE_SEASON)
box = pd.DataFrame([r for e in ev for r in espn.parse_box(espn.fetch_summary(e["event_id"], LIVE_SEASON))])
box = box[box.dnp == 0].rename(columns={"espn_id": "pid"})
box["pid"] = box.pid.astype(str)
box["pra"] = box.pts + box.reb + box.ast
print(f"{len(ev)} final games, {len(box)} player lines")
gd = pd.read_parquet(ROOT / f"data/live/projections_{d}.parquet")
sheets = {"game-day model": gd}
pre = ROOT / f"data/live/projections_{d:%Y%m%d}.parquet"
if pre.exists():
    sheets["preseason sheet"] = pd.read_parquet(pre)
rows = []
for name, p in sheets.items():
    m = box.merge(p[["pid"] + [f"proj_{s}" for s in ("min", "pts", "reb", "ast", "pra")]], on="pid")
    rows.append({"sheet": name, "players": len(m), **{f"mae_{s}": (m[s] - m[f"proj_{s}"]).abs().mean()
                                                       for s in ("min", "pts", "reb", "ast", "pra")}})
print(pd.DataFrame(rows).round(2).to_string(index=False))
