"""Rebuild everything the live projections depend on, from scratch, in the right
order. Skips steps whose outputs already exist unless --force.

    python scripts/rebuild.py            # ~2.5-3 h on a fresh machine
    python scripts/rebuild.py --force    # redo every step

Order matters (same order that produced the committed results):
  1  ESPN box scores, 10 seasons              data/processed/espn_box_<y>.parquet
  2  bulk nba.com data + reconcile + BBRef    data/processed/player_games_<y>.parquet
  3  player bios (ESPN)                       data/processed/player_bio.parquet
  4  pregame features (no preseason prior)    data/processed/features_all.parquet
  5  preseason model (walk-forward) + blend   data/processed/preseason_final.parquet
  6  pregame features again (+ preseason prior)
  7  out-of-sample game projections, all games data/processed/projections_all_games_oos.parquet
     (used for calibrated ranges in the live sheets)
  8  lean calibration (which stat the model leans on per player)
  9  live models are trained on first `scripts/live.py` run (data/models/)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from nba.config import PROC, SEASONS  # noqa: E402

FORCE = "--force" in sys.argv


def step(name, outputs, fn):
    if not FORCE and all(Path(o).exists() for o in outputs):
        print(f"[skip] {name}", flush=True)
        return
    t = time.time()
    print(f"[run ] {name}", flush=True)
    fn()
    print(f"[done] {name} ({(time.time() - t) / 60:.1f} min)", flush=True)


def s1():
    from nba.ingest import espn
    for y in SEASONS:
        espn.build_boxscores(y)


def s2():
    import runpy
    runpy.run_path(str(Path(__file__).parent / "reconcile_all.py"), run_name="__main__")


def s3():
    from nba.analysis.common import load_games
    from nba.ingest import espn_athletes
    espn_athletes.build(load_games(qualified_only=False).pid.unique())


def s4():
    from nba.analysis import features
    pre = PROC / "preseason_final.parquet"
    if pre.exists():  # step 4 is the "no prior" build
        pre.rename(pre.with_suffix(".old"))
    features.build()


def s5():
    from nba.analysis import preseason
    ps, allg, tg = preseason.player_seasons()
    f = preseason.feature_table(ps, allg, tg)
    f.to_parquet(PROC / "preseason_features.parquet", index=False)
    p = preseason.walk_forward(f)
    p.to_parquet(PROC / "preseason_oos.parquet", index=False)
    preseason.final_projection(p, f)


def s6():
    from nba.analysis import features
    features.build()


def s7():
    from nba.analysis import projection
    projection.project(scenarios=("injury_report",), all_games=True)


step("1 ESPN box scores", [PROC / f"espn_box_{y}.parquet" for y in SEASONS], s1)
step("2 reconcile + backfill", [PROC / f"player_games_{y}.parquet" for y in SEASONS], s2)
step("3 player bios", [PROC / "player_bio.parquet"], s3)
step("4 features (no prior)", [PROC / "preseason_oos.parquet"], s4)   # only needed before step 5
step("5 preseason model", [PROC / "preseason_final.parquet", PROC / "preseason_oos.parquet"], s5)
step("6 features (+ prior)", [PROC / "features_all.parquet"], s6)
if FORCE or "pre_pts" not in pd.read_parquet(PROC / "features_all.parquet", columns=None).columns:
    step("6b features (+ prior)", [], s6)
step("7 OOS projections, all games", [PROC / "projections_all_games_oos.parquet"], s7)


def s8():
    import json
    from nba.analysis import lean
    R, _, _ = lean.backtest_calibrated()
    json.dump(lean.fit_pick_calibration(R), open(PROC / "lean_pick_calibration.json", "w"))


step("8 lean calibration", [PROC / "lean_pick_calibration.json"], s8)
print("ready: python scripts/live.py YYYY-MM-DD")
