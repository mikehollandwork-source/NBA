"""Daily game-day projections.   python scripts/live.py [YYYY-MM-DD] [--retrain]"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nba import live_pipeline  # noqa: E402

d = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else date.today()
r = live_pipeline.run(d, retrain="--retrain" in sys.argv)
print(r.sort_values("proj_pts", ascending=False)[["player", "team", "opp", "proj_min", "proj_pts", "proj_reb", "proj_ast",
                                                  "proj_pra"]].head(20).round(1).to_string(index=False))

# keep locks across sessions (data/ is not in git): commit lock files + sheets
import subprocess  # noqa: E402
root = Path(__file__).resolve().parents[1]
subprocess.run(["git", "add", "reports/live"], cwd=root)
if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=root).returncode:
    subprocess.run(["git", "commit", "-q", "-m", f"Live projections {d} (locks)"], cwd=root)
    print("committed reports/live (push to keep across sessions)")
