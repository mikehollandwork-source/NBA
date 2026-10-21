"""Phase 1 end to end: data -> reconcile -> Test A (consistency) -> Test B (drivers).

    python scripts/run_phase1.py            # uses cached downloads in data/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nba.config import REPORTS, SEASON, SEASON_START, SEASON_END, RAW  # noqa: E402
from nba.ingest import bulk, espn, pbp_box, reconcile  # noqa: E402
from nba.analysis import consistency, features, drivers  # noqa: E402

out = REPORTS / "phase1"
out.mkdir(parents=True, exist_ok=True)

print("1/5 bulk nba.com data"); files = bulk.fetch(SEASON)
print("2/5 ESPN box scores"); espn.build_boxscores(SEASON_START, SEASON_END)
print("3/5 play-by-play box + reconcile")
pbp_box.build(files[f"cdnnba_{SEASON}"])
e, p, m = reconcile.build()
audit = reconcile.audit(m); audit.to_csv(out / "data_audit_espn_vs_pbp.csv", index=False)
reconcile.truth_table(e, p, m)
print("4/5 Test A: consistency")
_, a, b, c = consistency.run()
a.to_csv(out / "testA_stat_predictability.csv", index=False)
b.to_csv(out / "testA_raw_vs_per36_cv.csv", index=False)
c.to_csv(out / "testA_player_consistency.csv", index=False)
print("5/5 Test B: drivers")
features.build()
s, r, i = drivers.run()
s.to_csv(out / "testB_model_vs_baselines.csv", index=False)
r.to_csv(out / "testB_feature_group_value.csv", index=False)
i.to_csv(out / "testB_feature_importance.csv", index=False)
print("done ->", out)
