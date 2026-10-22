import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nba.config import SEASONS
from nba.ingest import espn
for y in sorted(SEASONS, reverse=True):
    df = espn.build_boxscores(y)
    print(y, df.event_id.nunique(), "games", len(df), "rows", flush=True)
