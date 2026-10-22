"""Player bio (height in inches, weight, birth date) from ESPN's core API."""
import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from nba.config import PROC

URL = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/athletes/{}"


def _one(pid):
    for _ in range(3):
        try:
            r = requests.get(URL.format(pid), timeout=20)
            if r.status_code == 200:
                d = r.json()
                return {"pid": str(pid), "height_in": d.get("height"), "weight_lb": d.get("weight"),
                        "dob": (d.get("dateOfBirth") or "")[:10]}
            if r.status_code == 404:
                break
        except requests.RequestException:
            pass
    return {"pid": str(pid), "height_in": None, "weight_lb": None, "dob": ""}


def build(pids) -> pd.DataFrame:
    out = PROC / "player_bio.parquet"
    have = pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=["pid"])
    todo = [p for p in pids if str(p).isdigit() and str(p) not in set(have.pid)]
    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(_one, todo))
    bio = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
    bio["dob"] = pd.to_datetime(bio.dob, errors="coerce")
    bio.to_parquet(out, index=False)
    return bio
