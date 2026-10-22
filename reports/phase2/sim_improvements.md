# Simulation upgrades (2026-10-22)

What the numpy simulation drives: the 80% ranges and the Kalshi rung probabilities.
Leans use a direct formula and were tested separately. Code: `nba/analysis/sim_improve.py`; live: `nba/sim.py`.
All tests out-of-sample: settings chosen on 2020-22, results on held-out 2023-25; curves fit on other seasons;
player history = prior games only.

## 1. Player-specific spread — SHIPPED (ranges + Kalshi)
Each stat's variance × the player's own past miss size (last 250 games), shrunk toward the league
curve with the weight of 100 games (picked on 2020-22 log-likelihood).

80% range coverage, 2023-25 (target 80%):

| Players (by own past swing) | PTS before → after | PRA before → after |
|---|---|---|
| steadiest 25% | 84.8% → 81.9% | 84.0% → 81.2% |
| middle 50% | 81.5% → 81.4% | 80.7% → 80.8% |
| swingiest 25% | 78.4% → 81.5% | 78.8% → 81.6% |

Log-likelihood better every held-out season and every stat except AST (flat). Rung Brier: +0.03-0.06% (negligible).
Leans with it (board-eligible, 2019-25): top 5/night 77.2% vs 76.8%, 70%+ 74.9% vs 75.3% → no clear gain, **leans unchanged**.

## 2. Minutes first — SHIPPED (with 1)
Each sim draws minutes from the real spread of actual/projected minutes (5 projection bins), then stats given those minutes.
Latent PTS-REB-AST link drops from 0.64 to 0.22 once minutes are drawn: minutes are what tie the stats together.

| CRPS gain vs current, 2023-25 (36k games) | PTS | REB | AST | PRA | RA |
|---|---|---|---|---|---|
| Placebo: current, new random seed | -0.03% | +0.08% | +0.04% | +0.03% | +0.07% |
| 1 + 2 (seed A / seed B) | +0.01/+0.06% | +0.16/+0.14% | +0.07/+0.04% | **+0.18/+0.18%** | +0.28/+0.24% |

PRA gain positive in 3/3 seasons and ~6× the placebo — real but small.

## 3. Player-specific PTS/REB/AST links — DROPPED
+0.05% PRA CRPS, within placebo noise; PTS slightly worse.

## 4. Number of sims — 4,000 → 20,000
Noise in a rung probability (P(PRA ≥ 35), typical player): ±0.76 pts at 4,000, ±0.32 at 20,000, ±0.21 at 50,000.
20,000 sims ≈ 17 s per 300 players. Ranges and Kalshi now share one simulation (they used separate draws).
