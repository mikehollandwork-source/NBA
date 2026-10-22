"""Retro-grade a finished date at real Kalshi pre-tip prices from LOCKED projections.
Labelled retro: picks computed after the fact (no result info used), not posted before tip.
    python scripts/kalshi_retro.py 2026-10-21"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from nba import kalshi, record  # noqa: E402
from nba.analysis.common import load_games  # noqa: E402
from nba.config import ROOT, VOL_MAX  # noqa: E402
from nba.live_pipeline import TEAM  # noqa: E402

d = sys.argv[1]
G = pd.read_parquet(ROOT / f"data/live/games_{d.replace('-', '')}.parquet") if (ROOT / f"data/live/games_{d.replace('-', '')}.parquet").exists() \
    else pd.read_parquet(ROOT / f"data/live/games_{d}.parquet")
G["away"], G["home"] = G.away.map(lambda x: TEAM.get(x, x)), G.home.map(lambda x: TEAM.get(x, x))
L = pd.concat([pd.read_csv(f, dtype={"pid": str}) for f in (record.LOCKS / d).glob("*.csv")
               if not f.name.endswith(".kalshi.csv")], ignore_index=True)
L["team"] = L.team.map(lambda x: TEAM.get(x, x))
t = load_games([int(d[:4]) - 1], qualified_only=False)
v = t.groupby("pid").agg(g=("gameId", "size"), mu=("pra", "mean"), sd=("pra", "std"))
v = v[v.g >= 20]
L = L.merge((v.sd / v.mu).rename("cv").reset_index(), on="pid", how="left")
L = L[(L.proj_min >= 10) & (L.cv <= VOL_MAX)]
if "proj_pra" not in L:
    L["proj_pra"] = L.proj_pts + L.proj_reb + L.proj_ast
mk = kalshi.markets(d, G)
mk = mk[mk.player_key.isin(L.player.map(kalshi.key))]
print("rungs for our players:", len(mk), flush=True)
mk = kalshi.pre_tip_quotes(mk)
mk.to_parquet(ROOT / f"data/live/kalshi_quotes_{d}.parquet", index=False)
e = kalshi.edges(mk, L)
pk = kalshi.picks(e)
gr = kalshi.grade(pk, record._box(d))
gr.to_csv(ROOT / f"reports/live/kalshi_retro_{d}.csv", index=False)
nv = gr[gr.result != "void"]
print(f"priced rungs {len(e)} | picks {len(pk)} | {record.summarize_k(gr)}")
print("by stat:", {s: record.summarize_k(x) for s, x in gr.groupby("stat")})
print("by side:", {s: record.summarize_k(x) for s, x in gr.groupby("side")})
print("median bid-ask spread:", round((e.yes_ask - e.yes_bid).median(), 3))
print(gr.sort_values("exp_roi", ascending=False)[["player", "stat", "side", "k", "price", "p_side", "exp_roi", "actual",
                                                   "result", "units"]].head(12).round(2).to_string(index=False))
