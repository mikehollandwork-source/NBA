import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from nba.config import SEASONS, REPORTS
from nba.ingest import reconcile
reps, auds = [], []
for y in SEASONS:
    t, a, rep = reconcile.build(y)
    print(rep, flush=True)
    reps.append(rep)
    if a is not None:
        auds.append(a)
out = REPORTS / "phase2"; out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(reps).to_csv(out / "data_coverage_by_season.csv", index=False)
pd.concat(auds).pivot(index="stat", columns="season", values="exact_match_pct").to_csv(out / "data_audit_espn_vs_pbp_by_season.csv")
print(pd.concat(auds).pivot(index="stat", columns="season", values="exact_match_pct"))
