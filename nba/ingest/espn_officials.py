"""Referees per game from ESPN game summaries (gameInfo.officials).

Historical: one summary call per cached regular-season event -> data/processed/officials.parquet
(event_id, game_date UTC, home, away, ref names). Only the officials list is kept (no odds).
Live: same field fills in once crews are assigned (NBA posts assignments ~9 AM ET on game day).
"""
import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from nba.config import PROC, RAW
from nba.ingest import espn

URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
CACHE = RAW / "espn_officials"


def _teams(event_path: Path):
    d = json.loads(gzip.decompress(event_path.read_bytes()))
    comp = d["header"]["competitions"][0]
    side = {c["homeAway"]: c["team"]["abbreviation"] for c in comp["competitors"]}
    return comp["date"][:10], side.get("home"), side.get("away")


def officials(event_id: str, refresh: bool = False) -> list[str]:
    f = CACHE / f"{event_id}.json"
    if f.exists() and not refresh:
        return json.loads(f.read_text())
    d = espn._get(URL, {"event": event_id})
    refs = [o.get("displayName") for o in (d.get("gameInfo", {}) or {}).get("officials", []) or []]
    if refs or refresh is False:
        CACHE.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(refs))
    return refs


def build(seasons=range(2016, 2027), workers=8) -> pd.DataFrame:
    rows = []
    paths = [p for y in seasons for p in sorted((RAW / "espn" / str(y)).glob("*.json.gz"))]

    def one(p):
        try:
            gd, home, away = _teams(p)
            eid = p.name.split(".")[0]
            return {"season": int(p.parent.name), "event_id": eid, "game_date": gd, "home": home, "away": away,
                    "refs": officials(eid)}
        except Exception as ex:  # keep going; missing refs are just missing
            return {"season": int(p.parent.name), "event_id": p.name.split(".")[0], "error": str(ex)[:80]}

    with ThreadPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(one, paths)):
            rows.append(r)
            if i % 1000 == 0:
                print(i, "of", len(paths), flush=True)
    O = pd.DataFrame(rows)
    O.to_parquet(PROC / "officials.parquet", index=False)
    return O


if __name__ == "__main__":
    O = build()
    print(len(O), "games;", (O.refs.str.len() > 0).mean() if "refs" in O else 0, "with refs")
