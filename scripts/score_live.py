"""Score a night's LOCKED projections (reports/live/locked/<date>/) against final
ESPN box scores. Only games that are final are scored.
    python scripts/score_live.py 2026-10-21
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from nba.config import LIVE_SEASON, ROOT  # noqa: E402
from nba.ingest import espn  # noqa: E402

d = date.fromisoformat(sys.argv[1])
files = sorted(f for f in (ROOT / "reports" / "live" / "locked" / str(d)).glob("*.csv") if not f.name.endswith((".kalshi.csv", ".winner.csv")))
if not files:
    sys.exit(f"no locked projections for {d}")
L = pd.concat([pd.read_csv(f, dtype={"pid": str}).assign(game=f.stem) for f in files], ignore_index=True)
ev = espn.regular_season_event_ids(d.isoformat(), d.isoformat(), season=LIVE_SEASON)  # completed games only
box = pd.DataFrame([r for e in ev for r in espn.parse_box(espn.fetch_summary(e["event_id"], LIVE_SEASON))])
box = box[box.dnp == 0].rename(columns={"espn_id": "pid"})
box["pid"] = box.pid.astype(str)
box["pra"] = box.pts + box.reb + box.ast
stats = ["min", "pts", "reb", "ast", "pra"]
m = L.merge(box[["pid"] + stats], on="pid", how="inner")
if "proj_min" in m:
    m = m[m.proj_min >= 10]
print(f"{len(ev)} final games; {m.game.nunique()} locked games scored; {len(m)} player lines")
rows = []
for (src, g), x in m.groupby(["lock_source", "game"]):
    rows.append({"game": g, "source": src, "players": len(x),
                 **{f"mae_{s}": (x[s] - x[f"proj_{s}"]).abs().mean() for s in stats}})
by_game = pd.DataFrame(rows)
print(by_game.round(2).to_string(index=False))
print(m.groupby("lock_source").apply(lambda x: pd.Series({f"mae_{s}": (x[s] - x[f"proj_{s}"]).abs().mean() for s in stats}))
      .round(2).to_string())
if "lean_stat" in m and m.lean_stat.notna().any():
    z = m[m.lean_stat.notna()].copy()
    act = [r[r.lean_stat] for _, r in z.iterrows()]
    z["hit"] = [(a > r.lean_ref) if r.lean_side == "over" else (a < r.lean_ref) for a, (_, r) in zip(act, z.iterrows())]
    print(f"leans: {len(z)}  hit rate {z.hit.mean():.1%}  (stated avg {z.lean_prob_cal.mean():.1%})")
out = ROOT / "reports" / "live" / f"score_{d}.csv"
by_game.to_csv(out, index=False)
