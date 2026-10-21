"""Phase 2 end to end (assumes scripts/fetch_espn_all.py + scripts/reconcile_all.py ran).

    python scripts/run_phase2.py [features|testA|testB|project|disagree|all]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from nba.config import REPORTS, SEASONS  # noqa: E402
from nba.analysis import consistency, features, drivers, projection, disagreement  # noqa: E402
from nba.analysis import defense_profiles, similar_defense  # noqa: E402
from nba.ingest import espn_athletes  # noqa: E402

out = REPORTS / "phase2"
out.mkdir(parents=True, exist_ok=True)
step = sys.argv[1] if len(sys.argv) > 1 else "all"

if step in ("features", "all"):
    features.build()
if step in ("testA", "all"):
    per, summ = consistency.run_all(SEASONS)
    per.to_csv(out / "testA_by_season.csv", index=False)
    summ.to_csv(out / "testA_stability_summary.csv")
    print(summ.to_string())
if step in ("testB", "all"):
    S, R = drivers.run()
    S.to_csv(out / "testB_model_gain_by_season.csv", index=False)
    R.to_csv(out / "testB_group_drop_loss_by_season.csv", index=False)
    s, r = drivers.summarise(S, R)
    s.to_csv(out / "testB_model_gain_summary.csv"); r.to_csv(out / "testB_group_summary.csv", index=False)
    print(s.to_string()); print(r.to_string())
if step in ("project", "all"):
    p = projection.project()
    acc = projection.accuracy(p)
    acc.to_csv(out / "projection_accuracy_by_season.csv", index=False)
    summ = acc.groupby(["scenario", "stat", "method"]).agg(gain_mean=("gain_pct", "mean"), gain_min=("gain_pct", "min"),
                                                            seasons_sig=("sig", "sum"), rmse=("rmse", "mean"),
                                                            mae=("mae", "mean")).round(3).reset_index()
    summ.to_csv(out / "projection_accuracy_summary.csv", index=False)
    print(summ.to_string())
    cal, rel = projection.calibration(p)
    cal.to_csv(out / "projection_calibration_by_season.csv", index=False)
    rel.to_csv(out / "projection_reliability_p_over_season_avg.csv", index=False)
    print(cal.groupby("stat").mean(numeric_only=True).round(3).to_string()); print(rel.to_string())
if step in ("disagree", "all"):
    d = disagreement.contributions()
    a, b = disagreement.by_size(d), disagreement.by_factor(d)
    a.to_csv(out / "disagreement_by_size.csv", index=False)
    b.to_csv(out / "disagreement_by_factor.csv", index=False)
    disagreement.examples(d, "pts", 25).to_csv(out / "disagreement_examples_pts.csv", index=False)
    print(a.round(3).to_string()); print(b.round(3).to_string())
if step in ("similar", "all"):
    import json
    from nba.analysis.common import load_games
    espn_athletes.build(load_games(qualified_only=False).pid.unique())
    defense_profiles.build()
    p, v, prof = similar_defense.load()
    d = similar_defense.signals(p, v)
    bt, bk, ln = similar_defense.backtest(d)
    bt.to_csv(out / "similar_defense_backtest_by_season.csv", index=False)
    bk.to_csv(out / "similar_defense_signal_buckets.csv", index=False)
    ln.to_csv(out / "similar_defense_vs_season_avg_line.csv", index=False)
    similar_defense.robustness(p, v).to_csv(out / "similar_defense_robustness.csv", index=False)
    json.dump(similar_defense.examples(d, prof, "pts", 2025, 4), open(out / "similar_defense_examples_pts_2025.json", "w"), indent=1)
    print(bt.groupby("stat").mean(numeric_only=True).round(3).to_string())
if step in ("season", "all"):
    from nba.analysis import season_sim
    from nba.analysis import projection as _proj
    _proj.project(scenarios=("injury_report",), all_games=True)
    T, G, R = season_sim.run()
    pg, se, gp = season_sim.summarise(G, R)
    pg.to_csv(out / "season_sim_per_matchup_summary.csv", index=False)
    se.to_csv(out / "season_sim_gameday_sum_summary.csv", index=False)
    gp.to_csv(out / "season_sim_games_played.csv", index=False)
    _, _, S = season_sim.run_preseason()
    S.to_csv(out / "season_sim_preseason_summary.csv", index=False)
    print(pg.round(3).to_string()); print(S.round(3).to_string())
